#!/usr/bin/env python
"""
VGGT backbone + lightweight classification head that is optimised directly on RGB pairs.
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

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
        backbone_dtype: str = "fp32",
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
        self.backbone_dtype = backbone_dtype
        self.layer_mode = layer_mode
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.token_proj_dim = token_proj_dim
        self.summary_tokens = summary_tokens
        self.summary_heads = summary_heads

        dtype = _resolve_torch_dtype(backbone_dtype)
        dtype_desc = "fp32" if dtype is None else str(dtype)
        print(f"[MODEL] Loading VGGT backbone '{backbone_ckpt}' on {self.device} (dtype={dtype_desc})...")
        load_kwargs = {"map_location": self.device}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.backbone = VGGT.from_pretrained(backbone_ckpt, **load_kwargs).to(self.device)
        if dtype is not None:
            self.backbone = self.backbone.to(dtype=dtype)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        print("[MODEL] Backbone ready.")

        self.head: Optional[PairwiseHead] = None
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
        """
        if pair_idx.dim() == 2:
            pair_idx = pair_idx.unsqueeze(0)
        elif pair_idx.dim() != 3:
            raise ValueError(f"pair_idx must be (P,2) or (B,P,2), got {pair_idx.shape}")

        if view_embeddings.dim() == 2:
            batch_view = view_embeddings.unsqueeze(0)  # (1, S, D)
        elif view_embeddings.dim() == 3:
            if pair_idx.shape[0] == view_embeddings.shape[0] and pair_idx.shape[0] > 1:
                batch_view = view_embeddings  # (B, S, D)
            else:
                batch_view = view_embeddings.unsqueeze(0)  # (1, S, L, D) or (1, S, D)
        elif view_embeddings.dim() == 4:
            batch_view = view_embeddings  # (B, S, L, D)
        else:
            raise ValueError(f"Unexpected view_embeddings shape {view_embeddings.shape}")

        pair_idx = pair_idx.to(self.device)
        emb_i, emb_j = self._gather_pair_embeddings(batch_view, pair_idx)
        if emb_i.numel() == 0:
            return torch.empty(0, device=self.device)

        self._init_head_if_needed(emb_dim=emb_i.shape[-1])
        return self.head(emb_i, emb_j)["logits"]

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
        emb_i, emb_j = self._gather_pair_embeddings(view_embeddings, pair_indices)
        if emb_i.numel() == 0:
            return torch.empty(0, device=self.device)

        self._init_head_if_needed(emb_dim=emb_i.shape[-1])
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
    parser.add_argument("--pair_indices", nargs="*", help="Optional pairs like '0-1 0-2'. Required when num_views>2.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
