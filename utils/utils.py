import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def adj_to_edges(adj: np.ndarray, threshold: float = 0.5):
    """
    adj: (N, N) adjacency / score matrix
    threshold: binarization threshold; for pure 0/1 adj just use 0.5
    returns list of (i, j) edges with i < j
    """
    adj = np.asarray(adj)
    N = adj.shape[0]
    # upper-triangular indices only (avoid double-counting i,j and j,i)
    i_idx, j_idx = np.triu_indices(N, k=1)
    mask = adj[i_idx, j_idx] >= threshold
    edges = list(zip(i_idx[mask], j_idx[mask]))
    return edges


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


def pairwise_ranking_loss(
    scores: torch.Tensor,
    strengths: torch.Tensor,
    margin: float = 0.1,
    num_samples: int = 1024,
) -> torch.Tensor:
    """
    Pairwise ranking loss: if strength_i > strength_j + margin,
    encourage score_i > score_j by at least `margin`.

    scores:    (B,) raw logits (or scores)
    strengths: (B,) continuous edge strengths (e.g., from rel_mat)
    """
    # Ensure 1D
    scores = scores.view(-1)
    strengths = strengths.view(-1)

    B = scores.size(0)
    if B < 2:
        return torch.tensor(0.0, device=scores.device)

    # Randomly sample index pairs
    idx1 = torch.randint(0, B, (num_samples,), device=scores.device)
    idx2 = torch.randint(0, B, (num_samples,), device=scores.device)

    s1 = strengths[idx1]
    s2 = strengths[idx2]

    # Keep only pairs where strength_i is significantly higher than strength_j
    mask = s1 > (s2 + margin)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Scores should respect the same ordering
    score_diff = scores[idx1] - scores[idx2]  # want this > margin
    loss = F.relu(margin - score_diff).mean()
    return loss


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss on logits."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)  # p_t
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def plot_iou_curves(
    thr_train: np.ndarray,
    iou_train: np.ndarray,
    thr_val: np.ndarray,
    iou_val: np.ndarray,
    out_path: str,
    title: str,
    latest_path: Optional[str] = None,
) -> None:
    """Plot train/val IoU curves vs threshold and save to file(s)."""
    if thr_train is None or iou_train is None or thr_val is None or iou_val is None:
        return
    x_thr = thr_val if thr_val.shape == thr_train.shape else thr_val
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x_thr, iou_train, label="train")
    plt.plot(x_thr, iou_val, label="val")
    plt.xlabel("Threshold")
    plt.ylabel("Graph IoU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if latest_path:
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        plt.savefig(latest_path, dpi=200)
    plt.close()


def plot_training_history(history: dict, plots_dir: str) -> None:
    """Plot standard training curves to the given directory."""
    num_epochs = len(history.get("train_loss", []))
    if num_epochs == 0:
        return
    os.makedirs(plots_dir, exist_ok=True)
    epochs = np.arange(1, num_epochs + 1)

    def _save(fig_path: str):
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss")
    plt.legend()
    _save(os.path.join(plots_dir, "loss.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.legend()
    _save(os.path.join(plots_dir, "accuracy.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_f1"], label="train")
    plt.plot(epochs, history["val_f1"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Train/Val F1")
    plt.legend()
    _save(os.path.join(plots_dir, "f1.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_roc_auc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("Validation ROC AUC")
    plt.legend()
    _save(os.path.join(plots_dir, "roc_auc.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_graph_iou_best"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Graph IoU (best)")
    plt.title("Validation Graph IoU (best)")
    plt.legend()
    _save(os.path.join(plots_dir, "graph_iou_best.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_graph_iou_auc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Graph IoU AUC")
    plt.title("Validation Graph IoU AUC")
    plt.legend()
    _save(os.path.join(plots_dir, "graph_iou_auc.png"))

    if history.get("train_soft_iou") and history.get("val_soft_iou"):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_soft_iou"], label="train")
        plt.plot(epochs, history["val_soft_iou"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Soft IoU")
        plt.title("Train/Val Soft IoU")
        plt.legend()
        _save(os.path.join(plots_dir, "soft_iou.png"))
