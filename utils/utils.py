import os
import random
from datetime import datetime
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Common scalar metrics we want to log across stdout/file/wandb
METRIC_LOG_KEYS = (
    "loss",
    "acc",
    "f1",
    "roc_auc",
    "pr_auc",
    "graph_iou_best",
    "graph_iou_auc",
    "graph_iou_best_thres",
    "soft_iou",
)


class RunLogger:
    """Lightweight logger that mirrors messages to stdout and an optional file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def create_run_logger(out_dir: str, log_file: Optional[str]) -> RunLogger:
    """Create a RunLogger writing to `out_dir/log_file` (or stdout only if empty)."""
    if log_file:
        log_path = log_file if os.path.isabs(log_file) else os.path.join(out_dir, log_file)
    else:
        log_path = None
    return RunLogger(log_path)


def format_metric_line(metrics: Dict[str, float], keys: Iterable[str] = METRIC_LOG_KEYS) -> str:
    """Format a metrics dict into a stable, ordered string for logging."""
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        val = metrics[key]
        if val is None:
            val_str = "nan"
        else:
            try:
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    val_str = "nan"
                elif key == "loss":
                    val_str = f"{float(val):.4f}"
                else:
                    val_str = f"{float(val):.3f}"
            except Exception:
                val_str = str(val)
        parts.append(f"{key}={val_str}")
    return " | ".join(parts)


def prefix_metrics(metrics: Dict[str, float], prefix: str, keys: Iterable[str] = METRIC_LOG_KEYS) -> Dict[str, float]:
    """Create a new dict with prefixed metric keys, keeping only known scalar metrics."""
    payload: Dict[str, float] = {}
    for key in keys:
        if key not in metrics:
            continue
        try:
            payload[f"{prefix}_{key}"] = float(metrics[key])
        except Exception:
            # Ignore non-scalar entries such as arrays
            continue
    return payload

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


# ---------------------------------------------------------------------
# Wandb helpers (lazy import, optional)
# ---------------------------------------------------------------------
def setup_wandb(project: Optional[str], run_name: Optional[str], config: dict, disabled: bool):
    """Initialize wandb if available/enabled, else return None."""
    if disabled or project is None:
        return None
    try:
        import wandb  # type: ignore
        if not hasattr(wandb, "init"):
            print("[WARN] wandb imported but has no 'init' attribute. Disabling wandb logging.")
            return None
        wandb.init(project=project, name=run_name, config=config)
        return wandb
    except Exception as e:
        print(f"[WARN] Failed to initialize wandb ({e}). Disabling wandb logging.")
        return None


def wandb_log(wandb_ref, data: dict, step: Optional[int] = None) -> None:
    if wandb_ref is None:
        return
    try:
        wandb_ref.log(data, step=step)
    except Exception as e:
        print(f"[WARN] wandb.log failed: {e}")


def wandb_save(wandb_ref, path: str) -> None:
    if wandb_ref is None:
        return
    try:
        wandb_ref.save(path)
    except Exception as e:
        print(f"[WARN] wandb.save failed for {path}: {e}")


def wandb_finish(wandb_ref) -> None:
    if wandb_ref is None:
        return
    try:
        wandb_ref.finish()
    except Exception as e:
        print(f"[WARN] wandb.finish failed: {e}")
