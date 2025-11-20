#!/usr/bin/env python
"""
VGGT backbone + lightweight classification head that is optimised directly on RGB pairs.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

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


class PairwiseHead(nn.Module):
    """Simple MLP head operating on concatenated pair embeddings."""

    def __init__(self, emb_dim: int, hidden_dim: int = 512, dropout_p: float = 0.2):
        super().__init__()
        inner_dim = max(1, hidden_dim // 2)
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

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> dict:
        """
        emb_i / emb_j shapes:
            - (B, E) for single layer
            - (B, L, E) for multi-layer selections
        """
        multi_layer = False
        if emb_i.ndim == 3:
            multi_layer = True
            B, L, E = emb_i.shape
            emb_i = emb_i.reshape(B * L, E)
            emb_j = emb_j.reshape(B * L, E)

        pair_feat = torch.cat(
            [emb_i, emb_j, (emb_i - emb_j).abs(), emb_i * emb_j],
            dim=-1,
        )
        logits = self.net(self.layernorm(pair_feat)).squeeze(-1)
        if multi_layer:
            logits = logits.view(B, L).mean(dim=1)
        return {"logits": logits}


class VGGTHeadModel(nn.Module):
    """Frozen VGGT backbone that exposes pooled per-view embeddings to a trainable head."""

    def __init__(
        self,
        backbone_ckpt: str = "facebook/VGGT-1B",
        device: Optional[str] = None,
        layer_mode: str = "all",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        token_proj_dim: int = 256,
        summary_tokens: int = 8,
        summary_heads: int = 4,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.layer_mode = layer_mode
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.token_proj_dim = token_proj_dim
        self.summary_tokens = summary_tokens
        self.summary_heads = summary_heads

        print(f"[MODEL] Loading VGGT backbone '{backbone_ckpt}' on {self.device}...")
        self.backbone = VGGT.from_pretrained(backbone_ckpt).to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        print("[MODEL] Backbone ready.")

        self.head: Optional[PairwiseHead] = None
        self.token_projector: Optional[nn.Linear] = None
        self.token_proj_norm: Optional[nn.LayerNorm] = None
        self.token_summarizer: Optional[TokenSummarizer] = None

    def train(self, mode: bool = True):
        """Override to keep VGGT frozen regardless of optimizer mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _init_head_if_needed(self, emb_dim: int):
        if self.head is None:
            print(f"[MODEL] Initialising head with emb_dim={emb_dim}")
            self.head = PairwiseHead(
                emb_dim=emb_dim,
                hidden_dim=self.head_hidden_dim,
                dropout_p=self.head_dropout,
            ).to(self.device)

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
        images: tensor shaped (B, S, 3, H, W). S must be 2 for pairwise training.
        Returns:
            {"logits": (B,) tensor}
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.size(1) != 2:
            raise ValueError(f"VGGTHeadModel expects exactly 2 views, got {images.size(1)}.")
        images = images.to(self.device)

        with torch.no_grad():
            predictions = self.backbone(images, extract_features=True)
            if "features_all" not in predictions:
                raise RuntimeError("VGGT outputs do not include 'features_all'.")
            raw_layers = [feat.detach() for feat in predictions["features_all"]]

        embeddings = self._compute_layer_embeddings(raw_layers)
        selected = self._select_layers(embeddings)

        emb_i = selected[:, :, 0, :]
        emb_j = selected[:, :, 1, :]

        if emb_i.shape[1] == 1:
            emb_i = emb_i[:, 0, :]
            emb_j = emb_j[:, 0, :]

        self._init_head_if_needed(emb_dim=emb_i.shape[-1])
        logits = self.head(emb_i, emb_j)["logits"]
        return {"logits": logits}

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor, select_layers: bool = True) -> Dict[str, torch.Tensor]:
        """
        Run the frozen VGGT backbone and return per-layer, per-view embeddings
        before the classification head.

        Args:
            images: (B, S, 3, H, W) tensor or (S, 3, H, W) with S expected to be 2.
            select_layers: whether to apply layer selection (layer_mode) before
                returning the embeddings.

        Returns:
            {
                "embeddings": (B, L, S, D),
                "emb_i": (B, L, D) or (B, D) if a single layer is selected,
                "emb_j": same shape as emb_i,
            }
        """

        if images.dim() == 4:
            images = images.unsqueeze(0)
        if images.size(1) != 2:
            raise ValueError(f"VGGTHeadModel expects exactly 2 views, got {images.size(1)}.")
        images = images.to(self.device)

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


__all__ = ["PairwiseHead", "VGGTHeadModel"]
