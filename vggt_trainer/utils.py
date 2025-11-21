#!/usr/bin/env python
"""
Shared helpers used across the VGGT trainer entrypoints.

Keeping these utilities here avoids re-implementing the same
seed/device/metric helpers in each script.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

DeviceLike = Union[str, torch.device, None]


def set_seed(seed: int):
    """Seed Python, NumPy and Torch (CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    head_params = sum(p.numel() for p in head.parameters()) if head is not None else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "head": head_params, "trainable": trainable}


__all__ = [
    "compute_graph_metrics",
    "count_parameters",
    "ensure_dir",
    "resolve_device",
    "set_seed",
]
