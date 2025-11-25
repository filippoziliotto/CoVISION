#!/usr/bin/env python
"""
Shared helpers used across the VGGT trainer entrypoints.

Keeping these utilities here avoids re-implementing the same
seed/device/metric helpers in each script.
"""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

DeviceLike = Union[str, torch.device, None]


def set_seed(seed: int):
    """Seed Python, NumPy and Torch (CPU/GPU) and enable deterministic kernels where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            # Best-effort: TF32 flags are not available on all backends.
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # warn_only avoids hard failures on ops without deterministic kernels.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some environments (e.g., older torch builds) may not support deterministic algos.
        pass


def resolve_device(device: DeviceLike = None) -> torch.device:
    """
    Return a torch.device, defaulting to GPU when available and no override is provided.
    Accepts str/torch.device/None for convenience.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not exist and return the Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_best_auc(best_auc_path: Optional[Union[str, Path]]) -> float:
    """Read the stored best AUC value (returns -inf if missing/invalid)."""
    if best_auc_path is None:
        return float("-inf")
    path = Path(best_auc_path)
    if not path.is_file():
        return float("-inf")
    try:
        return float(path.read_text().strip())
    except Exception:
        return float("-inf")


def maybe_save_predictions_csv(
    image_paths_1: Sequence[str],
    image_paths_2: Sequence[str],
    labels: Sequence[float],
    preds: Sequence[float],
    current_auc: float,
    output_path: Union[str, Path],
    best_auc_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, bool]:
    """
    Save a predictions CSV when the provided AUC improves over the stored best.

    Returns (best_auc_after_update, saved_flag).
    """
    best_auc = _read_best_auc(best_auc_path)
    try:
        auc_val = float(current_auc)
    except Exception:
        return best_auc, False
    if math.isnan(auc_val):
        return best_auc, False
    if auc_val <= best_auc:
        return best_auc, False

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_1", "image_2", "label", "pred"])
        for img1, img2, lbl, pred in zip(image_paths_1, image_paths_2, labels, preds):
            writer.writerow([img1, img2, lbl, pred])

    if best_auc_path is not None:
        best_path = Path(best_auc_path)
        ensure_dir(best_path.parent)
        best_path.write_text(f"{auc_val:.6f}")

    return auc_val, True


def configure_torch_multiprocessing(num_workers: int, strategy: str = "file_system") -> None:
    """
    Use a file-backed sharing strategy when dataloader workers are enabled to avoid
    shared-memory exhaustion on constrained systems (e.g., small /dev/shm).
    """
    if num_workers <= 0:
        return
    try:
        torch.multiprocessing.set_sharing_strategy(strategy)
    except (RuntimeError, ValueError):
        # Best effort; fall back to the default strategy.
        pass


def compute_graph_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute the graph IoU metrics used during training loops.
    Returns both the best IoU over thresholds and the AUC across thresholds.
    """
    if probs.size == 0:
        return {"graph_IOU": float("nan"), "graph_AUC": float("nan")}

    y_true = (labels >= 0.5).astype(np.float32)
    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6

    for t in thresholds:
        pred = (probs >= t).astype(np.float32)
        inter = np.logical_and(pred, y_true).sum()
        union = np.logical_or(pred, y_true).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    if hasattr(np, "trapezoid"):
        graph_auc = float(np.trapezoid(ious, thresholds))
    else:
        graph_auc = float(np.trapz(ious, thresholds))
    best_iou = float(np.max(ious))
    return {"graph_IOU": best_iou, "graph_AUC": graph_auc}


def count_parameters(model: torch.nn.Module, head: Optional[torch.nn.Module]) -> Dict[str, int]:
    """Return a small dictionary with parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    if hasattr(model, "head_parameters"):
        head_params = sum(p.numel() for p in model.head_parameters())
    elif head is not None:
        head_params = sum(p.numel() for p in head.parameters())
    else:
        head_params = 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "head": head_params, "trainable": trainable}


__all__ = [
    "configure_torch_multiprocessing",
    "compute_graph_metrics",
    "count_parameters",
    "ensure_dir",
    "maybe_save_predictions_csv",
    "resolve_device",
    "set_seed",
]
