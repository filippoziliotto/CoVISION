#!/usr/bin/env python
"""
Zero-shot evaluation for VGGT embeddings using cosine similarity between paired views.

This mirrors the zero_shot_eval path in train/trainer.py while borrowing the data/model
setup from vggt_trainer/train_head.py. It runs the frozen VGGT backbone, extracts
per-view embeddings, and predicts covisibility based on cosine similarity.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

# Make repository root importable (same as train_head.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.utils import format_metric_line, plot_iou_curves  # noqa: E402
from vggt_trainer.args import build_vggt_trainer_parser  # noqa: E402
from vggt_trainer.data import build_pair_dataloaders_from_args  # noqa: E402
from vggt_trainer.model import VGGTHeadModel, _resolve_layer_indices  # noqa: E402
from vggt_trainer.utils import (
    configure_torch_multiprocessing,
    maybe_save_predictions_csv,
    resolve_device,
    set_seed,
)  # noqa: E402


def _empty_metrics() -> Dict[str, float]:
    return dict(
        loss=float("nan"),
        acc=float("nan"),
        f1=float("nan"),
        roc_auc=float("nan"),
        pr_auc=float("nan"),
        graph_iou_best=float("nan"),
        graph_iou_auc=float("nan"),
        graph_iou_best_thres=float("nan"),
        iou_thresholds=None,
        iou_curve=None,
    )


def _cosine_preds(emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
    """
    Compute similarity scores in [0,1] for a batch of embeddings.
    Handles both (B, D) and (B, L, D) cases by averaging across layers.
    """
    if emb_i.ndim == 2:
        sims = F.cosine_similarity(emb_i, emb_j, dim=-1)
    elif emb_i.ndim == 3:
        emb_i_norm = F.normalize(emb_i, dim=-1)
        emb_j_norm = F.normalize(emb_j, dim=-1)
        sims_layer = (emb_i_norm * emb_j_norm).sum(dim=-1)
        sims = sims_layer.mean(dim=1)
    else:
        raise ValueError(f"Unexpected embedding shapes: {emb_i.shape}, {emb_j.shape}")
    return (sims + 1.0) / 2.0


def _pool_layer_embeddings(
    feats_layer: torch.Tensor,
    emb_mode: str = "avg_max",
    token_chunks: int = 4,
    target_hw: int = 8,
) -> torch.Tensor:
    """
    Convert one VGGT feature tensor (one layer) into per-view embeddings without any
    learned projection/summarization layers.
    Supports feature shapes (B, S, P, C) / (B, S, C, P) or (B, S, C, H, W).
    Returns: (B, S, D) L2-normalised embeddings.
    """
    feats = feats_layer.float()

    if feats.dim() == 4:
        # Token features
        if feats.shape[2] > feats.shape[3]:
            # (B, S, P, C) -> keep
            tokens = feats
        else:
            # (B, S, C, P) -> (B, S, P, C)
            tokens = feats.permute(0, 1, 3, 2)

        mean_tok = tokens.mean(dim=2)
        if emb_mode == "avg":
            emb = mean_tok
        elif emb_mode == "avg_max":
            max_tok = tokens.amax(dim=2)
            emb = torch.cat([mean_tok, max_tok], dim=-1)
        elif emb_mode == "chunked":
            B, S, P, C = tokens.shape
            n_chunks = max(1, int(token_chunks))
            if P < n_chunks:
                emb = mean_tok
            else:
                P_trim = (P // n_chunks) * n_chunks
                tok_trim = tokens[:, :, :P_trim, :]  # (B, S, P_trim, C)
                tok_chunk = tok_trim.view(B, S, n_chunks, -1, C)  # (B, S, n_chunks, P_seg, C)
                emb = tok_chunk.mean(dim=3).reshape(B, S, n_chunks * C)
        else:
            raise ValueError(f"Unknown emb_mode '{emb_mode}'")

    elif feats.dim() == 5:
        # Spatial maps: (B, S, C, H, W)
        B, S, C, H, W = feats.shape
        maps = feats.view(B * S, C, H, W)
        pooled = F.adaptive_avg_pool2d(maps, (target_hw, target_hw))
        pooled_flat = pooled.view(B * S, C, -1)  # (B*S, C, P')

        if emb_mode == "avg":
            emb_flat = pooled.view(B * S, -1)
        elif emb_mode == "avg_max":
            mean_sp = pooled_flat.mean(dim=-1)
            max_sp = pooled_flat.amax(dim=-1)
            grid_flat = pooled.view(B * S, -1)
            emb_flat = torch.cat([grid_flat, mean_sp, max_sp], dim=1)
        elif emb_mode == "chunked":
            P = pooled_flat.shape[-1]
            n_chunks = max(1, int(token_chunks))
            if P < n_chunks:
                emb_flat = pooled.view(B * S, -1)
            else:
                P_trim = (P // n_chunks) * n_chunks
                feats_trim = pooled_flat[:, :, :P_trim]
                feats_chunk = feats_trim.view(B * S, C, n_chunks, -1)
                mean_chunks = feats_chunk.mean(dim=-1)
                emb_flat = mean_chunks.view(B * S, C * n_chunks)
        else:
            raise ValueError(f"Unknown emb_mode '{emb_mode}'")

        emb = emb_flat.view(B, S, -1)
    else:
        raise ValueError(f"Unexpected feature shape {feats.shape}")

    emb = F.normalize(emb, dim=-1)
    return emb


def _build_pair_embeddings(
    preds: dict,
    layer_mode: str,
    emb_mode: str = "avg_max",
    token_chunks: int = 4,
    target_hw: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Build per-view embeddings directly from VGGT raw features (no trainable layers).
    Returns emb_i / emb_j shaped either (B, D) or (B, L, D) depending on layer selection.
    """
    if "features_all" not in preds:
        raise RuntimeError("VGGT outputs do not include 'features_all'.")
    feat_layers = preds["features_all"]
    if not isinstance(feat_layers, (list, tuple)) or len(feat_layers) == 0:
        raise RuntimeError("features_all must be a non-empty list of tensors.")

    idx_spec = _resolve_layer_indices(layer_mode, len(feat_layers))
    if idx_spec is None:
        selected = feat_layers
    elif isinstance(idx_spec, int):
        selected = [feat_layers[idx_spec]]
    else:
        selected = [feat_layers[i] for i in idx_spec]

    emb_layers = []
    for feats in selected:
        emb_layers.append(_pool_layer_embeddings(feats, emb_mode=emb_mode, token_chunks=token_chunks, target_hw=target_hw))

    stacked = torch.stack(emb_layers, dim=1)  # (B, L, S, D)
    if stacked.shape[2] < 2:
        raise RuntimeError(f"Expected at least 2 views per sample, got shape {stacked.shape}")

    emb_i = stacked[:, :, 0, :]
    emb_j = stacked[:, :, 1, :]
    if emb_i.shape[1] == 1:
        emb_i = emb_i[:, 0, :]
        emb_j = emb_j[:, 0, :]
    return {"emb_i": emb_i, "emb_j": emb_j}


@torch.no_grad()
def zero_shot_eval(
    model: VGGTHeadModel,
    loader: Optional[torch.utils.data.DataLoader],
    *,
    layer_mode: str,
    emb_mode: str,
    token_chunks: int = 4,
    return_preds: bool = False,
) -> Dict[str, float]:
    """Run cosine-similarity covisibility predictions over a dataloader."""
    if loader is None:
        return _empty_metrics()

    model.eval()
    criterion = nn.BCELoss()

    all_scores, all_labels = [], []
    accs, f1s = [], []
    pred_records = [] if return_preds else None

    for batch in tqdm(loader, desc="Zero-shot Eval", leave=False):
        images = batch["images"].to(model.device, non_blocking=True)
        labels = batch["label"].to(model.device, non_blocking=True).view(-1)

        preds_raw = model.backbone(images, extract_features=True)
        feats = _build_pair_embeddings(
            preds_raw,
            layer_mode=layer_mode,
            emb_mode=emb_mode,
            token_chunks=token_chunks,
        )
        preds = _cosine_preds(feats["emb_i"], feats["emb_j"])

        all_scores.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        y_pred = (preds >= 0.5).float().cpu().numpy()
        y_true = (labels >= 0.5).float().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

        if return_preds and pred_records is not None:
            paths_i = batch.get("img_path_i")
            paths_j = batch.get("img_path_j")
            if paths_i is not None and paths_j is not None:
                paths_i_list = list(paths_i) if isinstance(paths_i, (list, tuple)) else [paths_i]
                paths_j_list = list(paths_j) if isinstance(paths_j, (list, tuple)) else [paths_j]
                labels_list = labels.detach().cpu().tolist()
                preds_list = preds.detach().cpu().tolist()
                if (
                    len(paths_i_list) == len(paths_j_list) == len(preds_list)
                    and len(labels_list) == len(preds_list)
                ):
                    for p_i, p_j, lbl, pred_val in zip(
                        paths_i_list, paths_j_list, labels_list, preds_list
                    ):
                        pred_records.append((p_i, p_j, float(lbl), float(pred_val)))

    if not all_scores:
        return _empty_metrics()

    scores_np = torch.cat(all_scores, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    loss = criterion(
        torch.from_numpy(scores_np.astype(np.float32)),
        torch.from_numpy(labels_np.astype(np.float32)),
    ).item()

    y_bin = (scores_np >= 0.5).astype(np.float32)
    y_true_bin = (labels_np >= 0.5).astype(np.float32)
    acc = float(np.mean(accs)) if accs else float("nan")
    f1 = float(np.mean(f1s)) if f1s else float("nan")

    if len(np.unique(y_true_bin)) < 2 or len(np.unique(scores_np)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        try:
            roc_auc = float(roc_auc_score(y_true_bin, scores_np))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true_bin, scores_np))
        except Exception:
            pr_auc = float("nan")

    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6

    for t in thresholds:
        pred_t = (scores_np >= t).astype(np.float32)
        inter = np.logical_and(pred_t, y_true_bin).sum()
        union = np.logical_or(pred_t, y_true_bin).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    if hasattr(np, "trapezoid"):
        graph_iou_auc = float(np.trapezoid(ious, thresholds))
    else:
        graph_iou_auc = float(np.trapz(ious, thresholds))
    graph_iou_best = float(np.max(ious))
    graph_iou_best_thres = float(thresholds[int(np.argmax(ious))])

    metrics = dict(
        loss=loss,
        acc=acc,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        graph_iou_best=graph_iou_best,
        graph_iou_auc=graph_iou_auc,
        graph_iou_best_thres=graph_iou_best_thres,
        iou_thresholds=np.asarray(thresholds, dtype=np.float32),
        iou_curve=np.asarray(ious, dtype=np.float32),
    )
    if return_preds and pred_records is not None:
        metrics["pred_records"] = pred_records
    return metrics


def main():
    parser = build_vggt_trainer_parser()
    args = parser.parse_args()

    if args.mixing_aware is not None and args.head_type != "scene_aware":
        raise ValueError("--mixing_aware is only valid when --head_type is 'scene_aware'.")
    if args.head_type == "scene_aware" and args.mixing_aware is None:
        raise ValueError("Specify --mixing_aware when using --head_type scene_aware.")

    set_seed(args.seed)
    configure_torch_multiprocessing(args.num_workers)
    device = resolve_device(args.device)
    print(f"[SETUP] Using device {device}")

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        meta,
    ) = build_pair_dataloaders_from_args(args, device=device)

    print(
        f"[DATA] train_pairs={meta['train_pairs']} "
        f"(scenes={meta['train_scenes']}), "
        f"val_pairs={meta['val_pairs']} "
        f"(scenes={meta['val_scenes']})"
    )
    if train_dataset is None:
        raise RuntimeError("Zero-shot evaluation requires a non-empty training set for stats.")

    model = VGGTHeadModel(
        backbone_ckpt=args.backbone_ckpt,
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

    print("[EVAL] Running zero-shot on training split...")
    train_metrics = zero_shot_eval(
        model,
        train_loader,
        layer_mode=args.layer_mode,
        emb_mode=args.emb_mode,
        token_chunks=getattr(args, "token_chunks", 4),
    )
    print(f"[ZERO-SHOT][train] {format_metric_line(train_metrics)}")

    val_metrics: Optional[Dict[str, float]] = None
    if val_loader is not None and val_dataset is not None:
        print("[EVAL] Running zero-shot on validation split...")
        val_metrics = zero_shot_eval(
            model,
            val_loader,
            layer_mode=args.layer_mode,
            emb_mode=args.emb_mode,
            token_chunks=getattr(args, "token_chunks", 4),
            return_preds=True,
        )
        print(f"[ZERO-SHOT][val]   {format_metric_line(val_metrics)}")

    if val_metrics is not None:
        pred_records = val_metrics.pop("pred_records", None)
        if pred_records:
            preds_dir = Path(args.output_dir) / args.dataset_type / args.mode
            preds_path = preds_dir / "preds.csv"
            best_auc_path = preds_dir / "best_auc.txt"
            imgs1, imgs2, labels_list, preds_list = zip(*pred_records)
            best_auc, saved = maybe_save_predictions_csv(
                imgs1,
                imgs2,
                labels_list,
                preds_list,
                val_metrics.get("roc_auc", float("nan")),
                preds_path,
                best_auc_path,
            )
            if saved:
                print(f"[EVAL] Saved predictions to {preds_path} (roc_auc={best_auc:.4f})")
            else:
                auc_val = val_metrics.get("roc_auc", float("nan"))
                if isinstance(auc_val, float) and math.isnan(auc_val):
                    print("[EVAL] Skipped saving predictions; roc_auc is NaN.")
                else:
                    print(f"[EVAL] Skipped saving predictions; best roc_auc={best_auc:.4f}")
        else:
            print("[EVAL] No prediction records available to save.")

        plots_dir = os.path.join(args.output_dir, "zero_shot_plots")
        plot_iou_curves(
            train_metrics.get("iou_thresholds"),
            train_metrics.get("iou_curve"),
            val_metrics.get("iou_thresholds"),
            val_metrics.get("iou_curve"),
            out_path=os.path.join(plots_dir, "zero_shot_iou_curve.png"),
            title="Zero-shot Graph IoU vs Threshold",
            latest_path=os.path.join(plots_dir, "zero_shot_iou_curve_latest.png"),
        )


if __name__ == "__main__":
    main()
