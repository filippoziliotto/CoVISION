#!/usr/bin/env python
"""
VGGT backbone + lightweight classification head that is optimised directly on RGB pairs.
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Ensure repository root (and bundled vggt submodule) is importable.
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "vggt") not in sys.path:
    sys.path.append(str(REPO_ROOT / "vggt"))

from vggt.vggt.models.vggt import VGGT


class TokenSummarizer(nn.Module):
    """Attention-based module that condenses patch tokens into a few learned summary tokens."""

    def __init__(self, token_dim: int, num_tokens: int = 8, num_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, P, C) projected patch tokens for one view.
        Returns:
            summary: (B, num_tokens, C)
        """
        B, _, C = tokens.shape
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.attn(queries, tokens, tokens)
        summary = self.norm(attended + queries)
        summary = summary + self.ffn(summary)
        return summary


# ---------------------- CorrespondenceSummarizer ----------------------
class CorrespondenceSummarizer(nn.Module):
    """
    Build a compact correspondence descriptor from per-view summary tokens.

    This module assumes that each view-level embedding is obtained by flattening
    `num_tokens` summary tokens of dimensionality `token_dim`:
        emb_view: (B, num_tokens * token_dim)

    Given a pair of embeddings (one per view), it:
      1. Reshapes them back to (B, num_tokens, token_dim),
      2. Computes a small similarity matrix between summary tokens of the two views,
      3. Flattens this matrix and projects it to a low-dimensional descriptor.

    This descriptor can be concatenated to standard pair features
    [e_i, e_j, |e_i - e_j|, e_i * e_j] inside the classification head to expose
    explicit correspondence information, while keeping the head lightweight.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        corr_proj_dim: int = 32,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # Two-layer MLP that maps pooled similarity statistics
        # (row/column/global max/mean) to a compact correspondence descriptor.
        # Feature dimension: 4 * num_tokens (row/col max/mean) + 2 (global max/mean).
        pooled_feat_dim = 4 * num_tokens + 2
        self.corr_mlp = nn.Sequential(
            nn.Linear(pooled_feat_dim, corr_proj_dim),
            nn.GELU(),
            nn.Linear(corr_proj_dim, corr_proj_dim),
        )

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_i: (B, num_tokens * token_dim) embedding for view i.
            emb_j: (B, num_tokens * token_dim) embedding for view j.

        Returns:
            corr_feat: (B, corr_proj_dim) correspondence descriptor.
        """
        if emb_i.dim() != 2 or emb_j.dim() != 2:
            raise ValueError(
                "CorrespondenceSummarizer expects 2D tensors shaped (B, num_tokens * token_dim); "
                f"got emb_i={emb_i.shape}, emb_j={emb_j.shape}"
            )
        if emb_i.shape != emb_j.shape:
            raise ValueError(
                f"Shape mismatch between emb_i and emb_j: emb_i={emb_i.shape}, emb_j={emb_j.shape}"
            )

        B, E = emb_i.shape
        expected_E = self.num_tokens * self.token_dim
        if E != expected_E:
            raise ValueError(
                f"Expected embedding dimension {expected_E} (= num_tokens * token_dim), "
                f"got {E}"
            )

        # Reshape back to (B, M, C)
        tokens_i = emb_i.view(B, self.num_tokens, self.token_dim)
        tokens_j = emb_j.view(B, self.num_tokens, self.token_dim)

        # Cosine-style similarity matrix between summary tokens of the two views: (B, M, M)
        tokens_i = F.normalize(tokens_i, p=2, dim=-1)
        tokens_j = F.normalize(tokens_j, p=2, dim=-1)
        sim = torch.einsum("bmc,bnc->bmn", tokens_i, tokens_j)

        # Row/column/global pooling of similarity statistics.
        # Row: how each token in view i matches tokens in view j.
        row_max, _ = sim.max(dim=2)   # (B, M)
        row_mean = sim.mean(dim=2)    # (B, M)

        # Column: how each token in view j matches tokens in view i.
        col_max, _ = sim.max(dim=1)   # (B, M)
        col_mean = sim.mean(dim=1)    # (B, M)

        # Global statistics over all token pairs.
        sim_flat_all = sim.view(B, self.num_tokens * self.num_tokens)
        global_max, _ = sim_flat_all.max(dim=1, keepdim=True)   # (B, 1)
        global_mean = sim_flat_all.mean(dim=1, keepdim=True)    # (B, 1)

        # Concatenate all pooled features into a single descriptor.
        pooled_stats = torch.cat(
            [row_max, row_mean, col_max, col_mean, global_max, global_mean],
            dim=-1,
        )  # shape: (B, 4 * M + 2)

        corr_feat = self.corr_mlp(pooled_stats)
        return corr_feat


def _resolve_layer_indices(layer_mode: str, num_layers: int) -> Optional[Union[int, List[int]]]:
    """Mirror the layer selection rules used in the original trainers."""
    if layer_mode == "all":
        return None

    mode_to_offset = {
        "1st_last": -1,
        "2nd_last": -2,
        "3rd_last": -3,
        "4th_last": -4,
    }
    range_modes = {
        "last_stages": 17,  # 1-based
        "mid_to_last_stages": 12,
    }

    if layer_mode in mode_to_offset:
        offset = mode_to_offset[layer_mode]
        idx = num_layers + offset if offset < 0 else offset
        idx = max(0, min(num_layers - 1, idx))
        return idx

    if layer_mode in range_modes:
        start_1b = range_modes[layer_mode]
        start_idx = max(0, min(num_layers - 1, start_1b - 1))
        return list(range(start_idx, num_layers))

    raise ValueError(
        f"Unknown layer_mode '{layer_mode}'. "
        f"Expected {{'all', '1st_last', '2nd_last', '3rd_last', '4th_last', 'last_stages', 'mid_to_last_stages'}}."
    )

def _resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    """Map a friendly dtype string to a torch dtype; returns None for fp32."""
    name = name.lower()
    if name == "fp32":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported backbone dtype '{name}'. Expected one of ['fp32', 'fp16', 'bf16'].")


class PairwiseHead(nn.Module):
    """Simple MLP head operating on concatenated pair embeddings."""

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout_p: float = 0.2,
        use_corr_features: bool = False,
        use_corr_refine: bool = False,
        num_summary_tokens: Optional[int] = None,
        corr_proj_dim: int = 32,
    ):
        super().__init__()
        inner_dim = max(1, hidden_dim // 2)

        self.use_corr_features = use_corr_features
        self.use_corr_refine = use_corr_refine
        self.corr_summarizer: Optional[CorrespondenceSummarizer] = None

        base_feat_dim = 4 * emb_dim

        # We need a CorrespondenceSummarizer whenever we either concatenate
        # per-layer correspondence features (use_corr_features) or build a
        # cross-layer refinement signal (use_corr_refine). Only the former
        # increases the input dimensionality of the MLP.
        if self.use_corr_features or self.use_corr_refine:
            if num_summary_tokens is None:
                raise ValueError(
                    "num_summary_tokens must be provided when use_corr_features=True or use_corr_refine=True "
                    "(expected to match VGGTHeadModel.summary_tokens)."
                )
            if emb_dim % num_summary_tokens != 0:
                raise ValueError(
                    f"emb_dim={emb_dim} is not divisible by num_summary_tokens={num_summary_tokens}; "
                    "cannot recover per-token dimensionality for correspondence features."
                )
            token_dim = emb_dim // num_summary_tokens
            self.corr_summarizer = CorrespondenceSummarizer(
                num_tokens=num_summary_tokens,
                token_dim=token_dim,
                corr_proj_dim=corr_proj_dim,
            )

        if self.use_corr_features:
            feat_dim = base_feat_dim + corr_proj_dim
        else:
            feat_dim = base_feat_dim

        self.layernorm = nn.LayerNorm(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(inner_dim, 1),
        )

        # Optional cross-layer refinement head over correspondence descriptors:
        # it looks at how the correspondence descriptor changes from the first
        # to the last selected layer and produces a single scalar logit per pair.
        self.corr_refine_mlp: Optional[nn.Module] = None
        if self.use_corr_refine:
            self.corr_refine_mlp = nn.Sequential(
                nn.Linear(3 * corr_proj_dim, corr_proj_dim),
                nn.GELU(),
                nn.Linear(corr_proj_dim, 1),
            )

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> dict:
        """
        emb_i / emb_j shapes:
            - (B, E) for single layer
            - (B, L, E) for multi-layer selections
        """
        multi_layer = False
        B = L = None  # populated only for multi-layer case
        if emb_i.ndim == 3:
            multi_layer = True
            B, L, E = emb_i.shape
            emb_i = emb_i.reshape(B * L, E)
            emb_j = emb_j.reshape(B * L, E)

        pair_feat = torch.cat(
            [emb_i, emb_j, (emb_i - emb_j).abs(), emb_i * emb_j],
            dim=-1,
        )

        corr_feat = None
        if self.use_corr_features or self.use_corr_refine:
            if self.corr_summarizer is None:
                raise RuntimeError(
                    "corr_summarizer is None but use_corr_features/use_corr_refine=True; "
                    "PairwiseHead was not initialised correctly."
                )
            corr_feat = self.corr_summarizer(emb_i, emb_j)
            if self.use_corr_features:
                pair_feat = torch.cat([pair_feat, corr_feat], dim=-1)

        logits = self.net(self.layernorm(pair_feat)).squeeze(-1)

        if multi_layer:
            # Base behaviour: mean over per-layer logits.
            logits = logits.view(B, L).mean(dim=1)

            # Optional cross-layer refinement: look at how correspondence descriptors
            # evolve between the first and last selected layer and add a scalar bias.
            if self.use_corr_refine and corr_feat is not None:
                if self.corr_refine_mlp is None:
                    raise RuntimeError(
                        "corr_refine_mlp is None but use_corr_refine=True; "
                        "PairwiseHead was not initialised correctly."
                    )
                # corr_feat was computed on flattened (B*L, :) embeddings; reshape back.
                corr_feat_seq = corr_feat.view(B, L, -1)  # (B, L, D_corr)
                first_corr = corr_feat_seq[:, 0, :]       # (B, D_corr)
                last_corr = corr_feat_seq[:, -1, :]       # (B, D_corr)
                delta_corr = last_corr - first_corr       # (B, D_corr)
                refine_input = torch.cat(
                    [first_corr, last_corr, delta_corr],
                    dim=-1,
                )  # (B, 3*D_corr)
                refine_logit = self.corr_refine_mlp(refine_input).squeeze(-1)  # (B,)
                logits = logits + refine_logit

        return {"logits": logits}


# ------------------------- Scene-Aware Pairwise Head -------------------------
class SceneAwarePairwiseHead(nn.Module):
    """
    Pairwise head that performs scene- and/or pair-conditioned mixing over VGGT layers.

    Mixing modes (via `mixing` argument):
      - "scene": use only scene-level layer descriptors to compute weights (previous behaviour).
      - "pair":  use only per-layer pair features to compute weights.
      - "both":  combine scene and pair logits before softmax.

    This module assumes that each VGGT layer encodes co-visibility at a different scale.
    For each scene (batch element) it:
      1. Receives per-layer scene descriptors h_{b,l} (one descriptor per layer).
      2. Predicts mixture weights over layers via a small MLP + softmax,
         using scene descriptors, pair features, or both, depending on `mixing`.
      3. Uses those weights to collapse multi-layer pair embeddings into a single embedding
         per view before applying a standard pairwise MLP head.

    Expected shapes:
        emb_i: (B, L, E)   # per-scene, per-layer embedding for view i
        emb_j: (B, L, E)   # per-scene, per-layer embedding for view j
        layer_descriptors: (B, L, E)  # scene-level descriptors per layer (e.g. mean over views)

    If emb_i / emb_j are already single-layer (B, E), the head falls back to standard MLP mode.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 512,
        dropout_p: float = 0.2,
        mixer_hidden_dim: int = 128,
        mixing: str = "scene",
        layer_pos_dim: int = 32,
        max_layers: int = 32,
        scene_uniform_blend: float = 1.0,
    ):
        super().__init__()
        inner_dim = max(1, hidden_dim // 2)

        mixing = mixing.lower()
        if mixing not in {"scene", "pair", "both"}:
            raise ValueError(f"Invalid mixing mode '{mixing}'. Expected one of ['scene', 'pair', 'both'].")
        self.mixing = mixing

        if not (0.0 <= scene_uniform_blend <= 1.0):
            raise ValueError(
                f"scene_uniform_blend must be in [0,1], got {scene_uniform_blend}"
            )
        self.scene_uniform_blend = scene_uniform_blend

        self.layer_pos_dim = layer_pos_dim
        self.max_layers = max_layers
        # Positional embedding over layer indices (0, 1, ..., L-1 up to max_layers).
        self.layer_pos_emb = nn.Embedding(max_layers, layer_pos_dim)

        # Scene-level mixer: maps per-layer scene descriptors to a scalar logit per layer.
        # Applied independently to each (B, L, :) descriptor and then normalised with softmax.
        self.layer_mixer = nn.Sequential(
            nn.Linear(emb_dim + layer_pos_dim, mixer_hidden_dim),
            nn.GELU(),
            nn.Linear(mixer_hidden_dim, 1),
        )

        # Pair-level mixer: maps per-layer pair features to a scalar logit per layer.
        # Used when mixing in {'pair', 'both'}.
        self.pair_mixer = nn.Sequential(
            nn.Linear(4 * emb_dim, mixer_hidden_dim),
            nn.GELU(),
            nn.Linear(mixer_hidden_dim, 1),
        )

        # Standard pairwise MLP on mixed embeddings, same structure as PairwiseHead.
        self.layernorm = nn.LayerNorm(4 * emb_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(inner_dim, 1),
        )

    def _compute_layer_weights(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        layer_descriptors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute mixing weights over layers for a batch of pairs.

        emb_i / emb_j: (B, L, E)
        layer_descriptors: (B, L, E)
        Returns:
            weights: (B, L) softmax-normalised over L.
        """
        if emb_i.ndim != 3 or emb_j.ndim != 3:
            raise ValueError(
                "SceneAwarePairwiseHead._compute_layer_weights expects (B,L,E) tensors for emb_i and emb_j, "
                f"got emb_i={emb_i.shape}, emb_j={emb_j.shape}"
            )

        if emb_i.shape != emb_j.shape:
            raise ValueError(
                "Shape mismatch between emb_i and emb_j in _compute_layer_weights: "
                f"emb_i={emb_i.shape}, emb_j={emb_j.shape}"
            )

        B, L, E = emb_i.shape

        logits = None

        # Scene-level contribution
        if self.mixing in {"scene", "both"}:
            if layer_descriptors is None:
                raise ValueError(
                    "layer_descriptors must be provided when using 'scene' or 'both' mixing modes."
                )
            if layer_descriptors.ndim != 3 or layer_descriptors.shape != (B, L, E):
                raise ValueError(
                    "layer_descriptors must have shape (B, L, E) in _compute_layer_weights, "
                    f"got {layer_descriptors.shape}"
                )
            if L > self.max_layers:
                raise ValueError(
                    f"Number of layers L={L} exceeds max_layers={self.max_layers} "
                    "configured for SceneAwarePairwiseHead positional embeddings."
                )
            # Build per-layer positional embeddings based on layer indices.
            layer_ids = torch.arange(L, device=emb_i.device).unsqueeze(0).expand(B, -1)  # (B, L)
            layer_pos = self.layer_pos_emb(layer_ids)  # (B, L, layer_pos_dim)
            # Concatenate scene descriptors with positional encoding along the feature dimension.
            scene_input = torch.cat([layer_descriptors, layer_pos], dim=-1)  # (B, L, E + layer_pos_dim)
            scene_logits = self.layer_mixer(scene_input).squeeze(-1)  # (B, L)
            logits = scene_logits if logits is None else logits + scene_logits

        # Pair-level contribution
        if self.mixing in {"pair", "both"}:
            # Per-layer pair features: (B, L, 4E)
            pair_feat = torch.cat(
                [
                    emb_i,
                    emb_j,
                    (emb_i - emb_j).abs(),
                    emb_i * emb_j,
                ],
                dim=-1,
            )
            pair_logits = self.pair_mixer(pair_feat).squeeze(-1)  # (B, L)
            logits = pair_logits if logits is None else logits + pair_logits

        if logits is None:
            raise RuntimeError(
                "No logits computed in _compute_layer_weights. This should not happen if mixing is one of "
                "['scene', 'pair', 'both']."
            )

        weights = torch.softmax(logits, dim=-1)  # (B, L)

        # Optional residual blending of scene-dependent weights with a uniform distribution over layers.
        # This keeps a uniform component (like the base head) and adds a scene-dependent bias:
        #   alpha_final = (1 - epsilon) * U + epsilon * alpha_scene
        # where U is the uniform distribution over L layers and epsilon = scene_uniform_blend.
        if self.mixing in {"scene", "both"} and self.scene_uniform_blend < 1.0:
            B, L = weights.shape
            uniform = weights.new_full((B, L), 1.0 / float(L))
            eps = self.scene_uniform_blend
            weights = (1.0 - eps) * uniform + eps * weights

        return weights

    def forward(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        layer_descriptors: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        emb_i / emb_j:
            - (B, E): single-layer embeddings, no mixing (falls back to standard head).
            - (B, L, E): multi-layer embeddings to be mixed via scene/pair/both-conditioned weights.

        layer_descriptors:
            - Required when mixing in {'scene', 'both'}; optional for 'pair'.
            - Should have shape (B, L, E) and encode per-layer scene context
              (e.g., mean over views for that layer) when provided.

        Returns:
            {"logits": (B,) tensor}
        """
        # Single-layer case: behave like PairwiseHead on (B,E) embeddings.
        if emb_i.ndim == 2:
            if emb_j.ndim != 2 or emb_j.shape != emb_i.shape:
                raise ValueError(
                    f"SceneAwarePairwiseHead expects emb_j with shape {emb_i.shape} for single-layer mode, "
                    f"got {emb_j.shape}"
                )
            pair_feat = torch.cat(
                [emb_i, emb_j, (emb_i - emb_j).abs(), emb_i * emb_j],
                dim=-1,
            )
            logits = self.net(self.layernorm(pair_feat)).squeeze(-1)
            return {"logits": logits}

        # Multi-layer case with scene/pair/both-conditioned mixing.
        if emb_i.ndim != 3 or emb_j.ndim != 3:
            raise ValueError(
                f"SceneAwarePairwiseHead expects emb_i/emb_j with ndim 2 or 3, "
                f"got emb_i.ndim={emb_i.ndim}, emb_j.ndim={emb_j.ndim}"
            )

        if emb_i.shape != emb_j.shape:
            raise ValueError(
                f"emb_i and emb_j must have the same shape in multi-layer mode, "
                f"got emb_i={emb_i.shape}, emb_j={emb_j.shape}"
            )

        B, L, E = emb_i.shape

        if self.mixing in {"scene", "both"} and layer_descriptors is None:
            raise ValueError(
                "layer_descriptors must be provided when using 'scene' or 'both' mixing modes "
                "with multi-layer embeddings."
            )
        if layer_descriptors is not None and layer_descriptors.shape != (B, L, E):
            raise ValueError(
                f"layer_descriptors shape {layer_descriptors.shape} does not match "
                f"multi-layer embedding shape {emb_i.shape}"
            )

        # Scene- and/or pair-conditioned mixing: weights over layers per pair.
        weights = self._compute_layer_weights(emb_i, emb_j, layer_descriptors)  # (B, L)
        weights = weights.view(B, L, 1)

        mixed_i = (emb_i * weights).sum(dim=1)  # (B, E)
        mixed_j = (emb_j * weights).sum(dim=1)  # (B, E)

        pair_feat = torch.cat(
            [mixed_i, mixed_j, (mixed_i - mixed_j).abs(), mixed_i * mixed_j],
            dim=-1,
        )
        logits = self.net(self.layernorm(pair_feat)).squeeze(-1)
        return {"logits": logits}

class VGGTHeadModel(nn.Module):
    """Frozen VGGT backbone that exposes pooled per-view embeddings to a trainable head."""

    def __init__(
        self,
        backbone_ckpt: str = "facebook/VGGT-1B",
        backbone_dtype: str = "fp32",
        device: Optional[str] = None,
        layer_mode: str = "all",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        token_proj_dim: int = 256,
        summary_tokens: int = 8,
        summary_heads: int = 4,
        head_type: str = "base",
        mixing_aware: Optional[str] = None,
        use_corr_features: bool = False,
        use_corr_refine: bool = False,
        corr_proj_dim: int = 32,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            print("[MODEL] Requested CUDA but no GPU is available; falling back to CPU for loading.")
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.backbone_dtype = backbone_dtype
        self.layer_mode = layer_mode
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.token_proj_dim = token_proj_dim
        self.summary_tokens = summary_tokens
        self.summary_heads = summary_heads
        self.head_type = head_type
        self.mixing_aware = mixing_aware
        self.use_corr_features = use_corr_features
        self.use_corr_refine = use_corr_refine
        self.corr_proj_dim = corr_proj_dim

        dtype = _resolve_torch_dtype(backbone_dtype)
        dtype_desc = "fp32" if dtype is None else str(dtype)
        print(f"[MODEL] Loading VGGT backbone '{backbone_ckpt}' on {self.device} (dtype={dtype_desc})...")
        # Do not pass map_location to safetensors; load with defaults then move.
        load_kwargs = {}
        # VGGT.__init__ does not accept torch_dtype; cast after loading instead.
        self.backbone = VGGT.from_pretrained(backbone_ckpt, **load_kwargs).to(self.device)
        if dtype is not None:
            self.backbone = self.backbone.to(dtype=dtype)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        print("[MODEL] Backbone ready.")

        if self.head_type not in {"base", "scene_aware"}:
            raise ValueError("head_type must be either 'base' or 'scene_aware'.")
        if self.head_type == "scene_aware":
            if self.mixing_aware is None:
                raise ValueError("mixing_aware must be provided when head_type='scene_aware'.")
            if self.mixing_aware not in {"pair", "scene", "both"}:
                raise ValueError(
                    "mixing_aware must be one of ['pair', 'scene', 'both'] when using a scene-aware head."
                )
        elif self.mixing_aware is not None:
            raise ValueError("mixing_aware is only valid when head_type='scene_aware'.")

        self.head: Optional[nn.Module]
        self.head = None
            
        self.token_projector: Optional[nn.Linear] = None
        self.token_proj_norm: Optional[nn.LayerNorm] = None
        self.token_summarizer: Optional[TokenSummarizer] = None

    def _ensure_batch_dim(self, images: torch.Tensor) -> torch.Tensor:
        """Guarantee a batch dimension for downstream encoding."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        return images

    def train(self, mode: bool = True):
        """Override to keep VGGT frozen regardless of optimizer mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _init_head_if_needed(self, emb_dim: int):
        if self.head is not None:
            return

        if self.head_type == "base":
            print(f"[MODEL] Initialising head with emb_dim={emb_dim}")
            self.head = PairwiseHead(
                emb_dim=emb_dim,
                hidden_dim=self.head_hidden_dim,
                dropout_p=self.head_dropout,
                use_corr_features=self.use_corr_features,
                use_corr_refine=self.use_corr_refine,
                num_summary_tokens=self.summary_tokens
                if (self.use_corr_features or self.use_corr_refine)
                else None,
                corr_proj_dim=self.corr_proj_dim,
            ).to(self.device)
        elif self.head_type == "scene_aware":
            print(
                f"[MODEL] Initialising scene-aware head with emb_dim={emb_dim} "
                f"(mixing={self.mixing_aware})"
            )
            self.head = SceneAwarePairwiseHead(
                emb_dim=emb_dim,
                hidden_dim=self.head_hidden_dim,
                dropout_p=self.head_dropout,
                mixer_hidden_dim=128,
                mixing=self.mixing_aware,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown head_type '{self.head_type}'. Expected one of ['base', 'scene_aware'].")

    def _ensure_projector(self, input_dim: int):
        if self.token_proj_dim <= 0 or self.token_projector is not None:
            return
        self.token_projector = nn.Linear(input_dim, self.token_proj_dim).to(self.device)
        self.token_proj_norm = nn.LayerNorm(self.token_proj_dim).to(self.device)

    def _ensure_summarizer(self, token_dim: int):
        if self.summary_tokens <= 0:
            raise ValueError("summary_tokens must be greater than zero.")
        if self.token_summarizer is None:
            self.token_summarizer = TokenSummarizer(
                token_dim,
                num_tokens=self.summary_tokens,
                num_heads=self.summary_heads,
            ).to(self.device)

    def _project_tokens(self, feats_layer: torch.Tensor) -> torch.Tensor:
        """
        feats_layer: (B, S, P, C_raw)
        Returns: (B, S, P, C_out) where C_out = token_proj_dim (or C_raw if projection disabled).
        """
        B, S, P, C = feats_layer.shape
        feats_layer = feats_layer.to(self.device).float()

        if self.token_proj_dim <= 0:
            return feats_layer

        self._ensure_projector(C)
        tokens_flat = feats_layer.view(B * S * P, C)
        projected = self.token_projector(tokens_flat)
        if self.token_proj_norm is not None:
            projected = self.token_proj_norm(projected)
        projected = projected.view(B, S, P, self.token_proj_dim)
        return projected

    def _summarize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, S, P, C_proj)
        Returns: (B, S, summary_tokens * C_proj)
        """
        B, S, P, C = tokens.shape
        self._ensure_summarizer(C)
        tokens_flat = tokens.view(B * S, P, C)
        summary = self.token_summarizer(tokens_flat)  # (B*S, summary_tokens, C)
        summary = summary.view(B, S, self.summary_tokens * C)
        return summary

    def _compute_layer_embeddings(self, features_all: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Convert VGGT token tensors into flattened view-level embeddings.
        We preserve all patch tokens but reduce their per-token dimensionality through
        a shared projector so the head remains tractable.
        """
        flattened_layers: List[torch.Tensor] = []
        for feats_layer in features_all:
            if feats_layer.dim() < 3:
                raise ValueError(f"Unexpected VGGT feature shape {feats_layer.shape}")

            feats_layer = feats_layer.to(self.device)
            if feats_layer.dim() != 4:
                raise ValueError(
                    "Expected VGGT features shaped (B, S, P, C) for token summarization; "
                    f"got {feats_layer.shape}"
                )
            projected = self._project_tokens(feats_layer)
            summary = self._summarize_tokens(projected)
            flattened_layers.append(summary)

        stacked = torch.stack(flattened_layers, dim=1)  # (B, L, S, D_flat)
        return stacked

    def _extract_view_embeddings(
        self,
        images: torch.Tensor,
        *,
        select_layers: bool = True,
        keep_batch: bool = True,
    ) -> torch.Tensor:
        """
        Run VGGT and return per-view embeddings in view-major order.

        Returns:
            (B, S, L, D) when multiple layers are selected,
            (B, S, D) when a single layer is selected.
            The batch dimension is squeezed out when keep_batch is False.
        """
        images = self._ensure_batch_dim(images.to(self.device))

        with torch.no_grad():
            predictions = self.backbone(images, extract_features=True)
            if "features_all" not in predictions:
                raise RuntimeError("VGGT outputs do not include 'features_all'.")
            raw_layers = [feat.detach() for feat in predictions["features_all"]]

        embeddings = self._compute_layer_embeddings(raw_layers)  # (B, L, S, D)
        if select_layers:
            embeddings = self._select_layers(embeddings)

        view_first = embeddings.permute(0, 2, 1, 3)  # (B, S, L, D)
        if view_first.shape[2] == 1:
            view_first = view_first[:, :, 0, :]  # (B, S, D)

        if not keep_batch:
            if view_first.shape[0] != 1:
                raise ValueError(
                    f"Expected a single batch element when keep_batch=False, got {view_first.shape[0]}"
                )
            view_first = view_first.squeeze(0)
        return view_first

    def _normalize_pair_indices(
        self, pair_indices: Optional[torch.Tensor], batch_size: int, num_views: int
    ) -> torch.Tensor:
        """
        Standardise pair_indices to shape (B, P, 2), broadcasting when needed.
        If pair_indices is None, default to the single pair (0, 1) for two-view inputs.
        """
        if pair_indices is None:
            if num_views != 2:
                raise ValueError(
                    "pair_indices must be provided when passing more than two views."
                )
            pair_indices = torch.tensor([[0, 1]], device=self.device)

        if not torch.is_tensor(pair_indices):
            pair_indices = torch.as_tensor(pair_indices, device=self.device)
        else:
            pair_indices = pair_indices.to(self.device)

        if pair_indices.dim() == 2:
            pair_indices = pair_indices.unsqueeze(0).expand(batch_size, -1, -1)
        elif pair_indices.dim() == 3:
            if pair_indices.shape[0] == 1 and batch_size > 1:
                pair_indices = pair_indices.expand(batch_size, -1, -1)
            elif pair_indices.shape[0] != batch_size:
                raise ValueError(
                    f"pair_indices batch dimension {pair_indices.shape[0]} does not "
                    f"match images batch size {batch_size}."
                )
        else:
            raise ValueError(
                f"pair_indices must have shape (P,2) or (B,P,2); got {pair_indices.shape}"
            )

        if pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices last dimension must be 2, got {pair_indices.shape}")
        return pair_indices

    def _gather_pair_embeddings(
        self, view_embeddings: torch.Tensor, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collect per-pair embeddings from view-major tensors.

        view_embeddings: (B, S, D) or (B, S, L, D)
        pair_indices: (B, P, 2)
        Returns:
            emb_i, emb_j flattened across the batch (shapes match head expectations).
        """
        if view_embeddings.dim() not in (3, 4):
            raise ValueError(f"Unexpected view_embeddings shape {view_embeddings.shape}")

        batch_emb_i: List[torch.Tensor] = []
        batch_emb_j: List[torch.Tensor] = []

        for b in range(view_embeddings.shape[0]):
            idx = pair_indices[b]
            if idx.numel() == 0:
                continue
            views_b = view_embeddings[b]
            emb_i = views_b.index_select(0, idx[:, 0])
            emb_j = views_b.index_select(0, idx[:, 1])
            batch_emb_i.append(emb_i)
            batch_emb_j.append(emb_j)

        if not batch_emb_i:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        emb_i_cat = torch.cat(batch_emb_i, dim=0)
        emb_j_cat = torch.cat(batch_emb_j, dim=0)
        return emb_i_cat, emb_j_cat

    def _select_layers(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Select subset of layers according to layer_mode."""
        num_layers = embeddings.shape[1]
        idx_spec = _resolve_layer_indices(self.layer_mode, num_layers)
        if idx_spec is None:
            return embeddings
        if isinstance(idx_spec, int):
            return embeddings[:, idx_spec : idx_spec + 1, :, :]
        idx_tensor = torch.tensor(idx_spec, dtype=torch.long, device=embeddings.device)
        return embeddings.index_select(dim=1, index=idx_tensor)

    def forward(self, images: torch.Tensor) -> dict:
        """
        images: tensor shaped (B, S, 3, H, W) or (S, 3, H, W).
        For pairwise training, S should be 2 and the single pair (0, 1) is scored.
        Returns:
            {"logits": (num_pairs_total,) tensor}
        """
        logits = self.score_pairs(images, pair_indices=None)
        return {"logits": logits}

    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run VGGT on an arbitrary number of views and return per-view embeddings.
        Returns (S, E) or (S, L, E) depending on layer selection.
        """
        return self._extract_view_embeddings(images, select_layers=True, keep_batch=False)

    def score_pair_indices(self, view_embeddings: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        """
        Given per-view embeddings and pair indices (P,2), return logits (P,).

        This helper is mainly intended for inference/debugging, where `view_embeddings`
        typically comes from `encode_views`:
            - (S, E)  for single-layer selection
            - (S, L, E) for multi-layer selection

        In the multi-layer case and with a scene-aware head, we:
          1. Treat the input as a single-scene batch of shape (1, S, L, E),
          2. Compute scene-level layer descriptors by averaging over views,
          3. Repeat descriptors per pair and pass them to SceneAwarePairwiseHead.
        """
        if pair_idx.dim() == 2:
            pair_idx = pair_idx.unsqueeze(0)
        elif pair_idx.dim() != 3:
            raise ValueError(f"pair_idx must be (P,2) or (B,P,2), got {pair_idx.shape}")

        # Normalise view embeddings to have an explicit batch dimension.
        if view_embeddings.dim() == 2:
            # (S, E) -> (1, S, E)
            batch_view = view_embeddings.unsqueeze(0)
        elif view_embeddings.dim() == 3:
            # Could be (S, L, E) or (B, S, E); disambiguate by aligning with pair_idx batch.
            if pair_idx.shape[0] == view_embeddings.shape[0] and pair_idx.shape[0] > 1:
                # Assume (B, S, E)
                batch_view = view_embeddings
            else:
                # Assume single-scene multi-layer embeddings (S, L, E) -> (1, S, L, E)
                batch_view = view_embeddings.unsqueeze(0)
        elif view_embeddings.dim() == 4:
            batch_view = view_embeddings  # (B, S, L, E)
        else:
            raise ValueError(f"Unexpected view_embeddings shape {view_embeddings.shape}")

        pair_idx = pair_idx.to(self.device)
        batch_view = batch_view.to(self.device)

        emb_i, emb_j = self._gather_pair_embeddings(batch_view, pair_idx)
        if emb_i.numel() == 0:
            return torch.empty(0, device=self.device)

        self._init_head_if_needed(emb_dim=emb_i.shape[-1])

        # Scene-aware path for multi-layer embeddings.
        if self.head_type == "scene_aware" and emb_i.ndim == 3:
            if batch_view.dim() != 4:
                raise ValueError(
                    f"Scene-aware head expects multi-layer view embeddings shaped (B,S,L,D); "
                    f"got {batch_view.shape}"
                )
            B_scenes, num_views, L, D = batch_view.shape
            _, P, _ = pair_idx.shape

            layer_descriptors = None
            if self.mixing_aware in {"scene", "both"}:
                # Scene-level descriptor per layer: mean over views -> (B, L, D)
                scene_layer_desc = batch_view.mean(dim=1)

                # Repeat descriptors per pair and flatten to (B*P, L, D)
                layer_descriptors = (
                    scene_layer_desc
                    .unsqueeze(1)                 # (B, 1, L, D)
                    .expand(B_scenes, P, -1, -1)  # (B, P, L, D)
                    .reshape(-1, L, D)            # (B*P, L, D)
                )

            logits = self.head(
                emb_i,
                emb_j,
                layer_descriptors=layer_descriptors,
            )["logits"]
        else:
            logits = self.head(emb_i, emb_j)["logits"]

        return logits

    def score_pairs(
        self,
        images: torch.Tensor,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score arbitrary image pairs from a set of views.
        pair_indices:
            - None: expects exactly two views and scores the single pair (0, 1).
            - (P, 2): same pairs are applied to every batch element.
            - (B, P, 2): per-example pair specification.
        Returns logits flattened across all pairs in the batch.
        """
        images = self._ensure_batch_dim(images)
        batch_size, num_views = images.shape[:2]
        pair_indices = self._normalize_pair_indices(pair_indices, batch_size, num_views)

        view_embeddings = self._extract_view_embeddings(
            images, select_layers=True, keep_batch=True
        )
            
        # Prepare layer descriptors if we have multi-layer embeddings
        # and we are using the scene-aware head.
        layer_descriptors = None
        if (
            self.head_type == "scene_aware"
            and view_embeddings.dim() == 4  # (B, S, L, D)
            and self.mixing_aware in {"scene", "both"}
        ):
            # Scene-level descriptor per layer: mean over S views.
            # shape: (B, L, D)
            scene_layer_desc = view_embeddings.mean(dim=1)

            B_scenes, num_views, _, _ = view_embeddings.shape
            _, P, _ = pair_indices.shape  # pair_indices is (B, P, 2) after normalisation

            # We will flatten pairs as (B_scenes * P, L, D),
            # so we need to repeat descriptors per pair.
            layer_descriptors = (
                scene_layer_desc
                .unsqueeze(1)                 # (B, 1, L, D)
                .expand(B_scenes, P, -1, -1)  # (B, P, L, D)
                .reshape(-1, scene_layer_desc.shape[1], scene_layer_desc.shape[2])  # (B*P, L, D)
            )

        emb_i, emb_j = self._gather_pair_embeddings(view_embeddings, pair_indices)
        if emb_i.numel() == 0:
            return torch.empty(0, device=self.device)

        self._init_head_if_needed(emb_dim=emb_i.shape[-1])

        if self.head_type == "scene_aware" and emb_i.ndim == 3:
            logits = self.head(
                emb_i,
                emb_j,
                layer_descriptors=layer_descriptors,
            )["logits"]
        else:
            logits = self.head(emb_i, emb_j)["logits"]

        return logits

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor, select_layers: bool = True) -> Dict[str, torch.Tensor]:
        """
        Run the frozen VGGT backbone and return per-layer, per-view embeddings
        before the classification head.

        Args:
            images: (B, S, 3, H, W) tensor or (S, 3, H, W) with at least two views.
            select_layers: whether to apply layer selection (layer_mode) before
                returning the embeddings.

        Returns:
            {
                "embeddings": (B, L, S, D),
                "emb_i": (B, L, D) or (B, D) if a single layer is selected (first view),
                "emb_j": same shape as emb_i (second view),
            }
        """

        images = self._ensure_batch_dim(images)
        if images.size(1) < 2:
            raise ValueError(f"VGGTHeadModel expects at least 2 views, got {images.size(1)}.")

        predictions = self.backbone(images, extract_features=True)
        if "features_all" not in predictions:
            raise RuntimeError("VGGT outputs do not include 'features_all'.")

        raw_layers = [feat.detach() for feat in predictions["features_all"]]
        embeddings = self._compute_layer_embeddings(raw_layers)
        if select_layers:
            embeddings = self._select_layers(embeddings)

        emb_i = embeddings[:, :, 0, :]
        emb_j = embeddings[:, :, 1, :]

        if emb_i.shape[1] == 1:
            emb_i = emb_i[:, 0, :]
            emb_j = emb_j[:, 0, :]

        return {
            "embeddings": embeddings,
            "emb_i": emb_i,
            "emb_j": emb_j,
        }

    def head_parameters(self) -> Iterable[torch.nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.token_proj_dim > 0 and self.token_projector is not None:
            params += list(self.token_projector.parameters())
            if self.token_proj_norm is not None:
                params += list(self.token_proj_norm.parameters())
        if self.token_summarizer is not None:
            params += list(self.token_summarizer.parameters())
        if self.head is not None:
            params += list(self.head.parameters())
        return params

    def get_head_state(self) -> dict:
        state = {}
        if self.head is not None:
            state["head"] = self.head.state_dict()
        if self.token_proj_dim > 0 and self.token_projector is not None:
            state["token_projector"] = self.token_projector.state_dict()
        if self.token_proj_dim > 0 and self.token_proj_norm is not None:
            state["token_proj_norm"] = self.token_proj_norm.state_dict()
        if self.token_summarizer is not None:
            state["token_summarizer"] = self.token_summarizer.state_dict()
        return state

    def load_head_state(self, state_dict: dict):
        if not state_dict:
            return
        if "head" in state_dict:
            if self.head is None:
                raise RuntimeError("Head must be initialised before loading weights.")
            self.head.load_state_dict(state_dict["head"])
        if "token_projector" in state_dict and self.token_projector is not None:
            self.token_projector.load_state_dict(state_dict["token_projector"])
        if "token_proj_norm" in state_dict and self.token_proj_norm is not None:
            self.token_proj_norm.load_state_dict(state_dict["token_proj_norm"])
        if "token_summarizer" in state_dict and self.token_summarizer is not None:
            self.token_summarizer.load_state_dict(state_dict["token_summarizer"])

def _parse_pair_indices_cli(raw_pairs: Optional[Sequence[str]], num_views: int) -> Optional[torch.Tensor]:
    """Parse CLI-friendly pair specs like ['0-1', '0,2'] into a tensor."""
    if not raw_pairs:
        return None
    parsed = []
    for spec in raw_pairs:
        cleaned = spec.replace("-", ",")
        parts = [p for p in cleaned.split(",") if p != ""]
        if len(parts) != 2:
            raise ValueError(f"Invalid pair spec '{spec}'. Use forms like '0-1' or '0,1'.")
        i, j = int(parts[0]), int(parts[1])
        if i < 0 or j < 0 or i >= num_views or j >= num_views:
            raise ValueError(f"Pair indices {i},{j} out of range for num_views={num_views}.")
        parsed.append((i, j))
    return torch.tensor(parsed, dtype=torch.long)


def main():
    """
    Minimal CLI to sanity check VGGTHeadModel forward and pair scoring with random tensors.
    This will still load the specified VGGT backbone; ensure weights are available locally.
    """
    parser = argparse.ArgumentParser(
        description="Smoke-test VGGTHeadModel with random inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone_ckpt", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--backbone_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, mps, or cpu.")
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="all",
        choices=[
            "all",
            "1st_last",
            "2nd_last",
            "3rd_last",
            "4th_last",
            "last_stages",
            "mid_to_last_stages",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--square_size", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16", "bf16"],
        default="float32",
        help="Random input tensor dtype for the smoke test.",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--token_proj_dim", type=int, default=256)
    parser.add_argument("--summary_tokens", type=int, default=8)
    parser.add_argument("--summary_heads", type=int, default=4)
    parser.add_argument(
        "--mixing_aware",
        type=str,
        default=None,
        choices=["pair", "scene", "both"],
        help="Mixing strategy for the scene-aware head.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="base",
        choices=["base", "scene_aware"],
        help="Head architecture to instantiate for the smoke test.",
    )
    parser.add_argument("--pair_indices", nargs="*", help="Optional pairs like '0-1 0-2'. Required when num_views>2.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_corr_features",
        action="store_true",
        help="Enable correspondence-based features in the base head.",
    )
    parser.add_argument(
        "--use_corr_refine",
        action="store_true",
        help="Enable cross-layer refinement based on correspondence descriptors in the base head.",
    )
    args = parser.parse_args()

    if args.mixing_aware is not None and args.head_type != "scene_aware":
        raise ValueError("--mixing_aware is only valid when --head_type is 'scene_aware'.")
    if args.head_type == "scene_aware" and args.mixing_aware is None:
        raise ValueError("Specify --mixing_aware when using --head_type scene_aware.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    pair_idx = _parse_pair_indices_cli(args.pair_indices, args.num_views)
    if pair_idx is None and args.num_views != 2:
        raise ValueError("Provide --pair_indices when using more than two views.")

    print(
        f"[INFO] Initialising VGGTHeadModel on {device} "
        f"(backbone={args.backbone_ckpt}, dtype={args.backbone_dtype})"
    )
    try:
        model = VGGTHeadModel(
            backbone_ckpt=args.backbone_ckpt,
            backbone_dtype=args.backbone_dtype,
            device=str(device),
            layer_mode=args.layer_mode,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
            token_proj_dim=args.token_proj_dim,
            summary_tokens=args.summary_tokens,
            summary_heads=args.summary_heads,
            head_type=args.head_type,
            mixing_aware=args.mixing_aware,
            use_corr_features=args.use_corr_features,
            use_corr_refine=args.use_corr_refine,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load VGGTHeadModel: {exc}")
        raise

    images = torch.randn(
        args.batch_size,
        args.num_views,
        3,
        518,
        518,
        device=device,
        dtype=dtype,
    )

    logits = model.score_pairs(images, pair_indices=pair_idx)
    print(
        f"[OK] Forward success | logits shape={tuple(logits.shape)}, "
        f"dtype={logits.dtype}, device={logits.device}"
    )
    if model.head is not None:
        print(
            f"[INFO] Head params={sum(p.numel() for p in model.head.parameters()):,}, "
            f"trainable={sum(p.numel() for p in model.head.parameters() if p.requires_grad):,}"
        )


__all__ = ["PairwiseHead", "VGGTHeadModel"]


if __name__ == "__main__":
    main()
