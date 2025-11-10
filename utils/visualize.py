

import os
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap
except ImportError:
    umap = None
    
from sklearn.metrics import precision_recall_curve
from utils.utils import adj_to_edges


def build_adj_matrices(
    num_views: int,
    pair_indices: Sequence[Tuple[int, int]],
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct GT / predicted / score adjacency matrices for a scene.

    Args:
        num_views: number of images/views in the split.
        pair_indices: list of (i, j) indices for each evaluated pair.
        y_true: list/array of ground-truth labels (0/1) for each pair.
        y_scores: list/array of continuous scores (e.g. similarity/overlap) for each pair.
        threshold: scalar threshold applied to y_scores to binarize predictions.

    Returns:
        gt_adj:   (num_views, num_views) uint8 matrix with GT edges.
        pred_adj: (num_views, num_views) uint8 matrix with predicted edges.
        score_mat:(num_views, num_views) float32 matrix with continuous scores.
    """
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    gt_adj = np.zeros((num_views, num_views), dtype=np.uint8)
    pred_adj = np.zeros((num_views, num_views), dtype=np.uint8)
    score_mat = np.zeros((num_views, num_views), dtype=np.float32)

    for (i, j), gt_ij, s_ij in zip(pair_indices, y_true, y_scores):
        if i < 0 or j < 0 or i >= num_views or j >= num_views:
            continue
        gt_adj[i, j] = gt_adj[j, i] = gt_ij
        score_mat[i, j] = score_mat[j, i] = s_ij
        pred_ij = 1 if s_ij >= threshold else 0
        pred_adj[i, j] = pred_adj[j, i] = pred_ij

    # zero diagonal (no self-edges)
    np.fill_diagonal(gt_adj, 0)
    np.fill_diagonal(pred_adj, 0)
    np.fill_diagonal(score_mat, 0.0)

    return gt_adj, pred_adj, score_mat


def save_scene_feat_results(
    out_root: str,
    scene_id: str,
    split_name: str,
    num_views: int,
    pair_indices: Sequence[Tuple[int, int]],
    y_true: Sequence[int],
    y_scores: Sequence[float],
    best_threshold: float,
    metrics: Dict[str, Any],
    global_csv_name: str = "global_results_feat.csv",
) -> None:
    """Save per-scene feature-space results and adjacency matrices.

    This helper is intended to be called from vggt_feat_eval.py once all
    metrics for a given (scene, split) have been computed.

    Layout on disk:
        data/predictions_feat/
            {scene_id}/
                split_{split_name}/
                    gt_adj.npy
                    pred_adj.npy
                    score_mat.npy
                    metrics.csv  (one row for this split)
        data/predictions_feat/{global_csv_name}
            aggregated results over all scenes/splits (append mode).
    """
    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    scene_dir = os.path.join(out_root, scene_id)
    split_dir = os.path.join(scene_dir, f"split_{split_name}")
    os.makedirs(split_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build and save adjacency matrices
    # ------------------------------------------------------------------
    gt_adj, pred_adj, score_mat = build_adj_matrices(
        num_views=num_views,
        pair_indices=pair_indices,
        y_true=y_true,
        y_scores=y_scores,
        threshold=best_threshold,
    )

    np.save(os.path.join(split_dir, "gt_adj.npy"), gt_adj)
    np.save(os.path.join(split_dir, "pred_adj.npy"), pred_adj)
    np.save(os.path.join(split_dir, "score_mat.npy"), score_mat)

    # ------------------------------------------------------------------
    # Save per-split metrics
    # ------------------------------------------------------------------
    metrics_row = dict(metrics)  # shallow copy
    metrics_row["scene_id"] = scene_id
    metrics_row["split"] = split_name
    metrics_row["num_views"] = int(num_views)
    metrics_row["num_pairs"] = int(len(y_true))
    metrics_row["best_threshold"] = float(best_threshold)

    split_metrics_path = os.path.join(split_dir, "metrics.csv")
    df_split = pd.DataFrame([metrics_row])
    df_split.to_csv(split_metrics_path, index=False)

    # ------------------------------------------------------------------
    # Append to global CSV under out_root
    # ------------------------------------------------------------------
    global_csv_path = os.path.join(out_root, global_csv_name)
    if os.path.exists(global_csv_path):
        df_global = pd.read_csv(global_csv_path)
        df_global = pd.concat([df_global, df_split], ignore_index=True)
    else:
        df_global = df_split
    df_global.to_csv(global_csv_path, index=False)
    return gt_adj, pred_adj, score_mat


# -----------------------------------------------------------
# 2D feature embedding visualization
# -----------------------------------------------------------
def plot_feature_embedding_2d(
    embeddings: np.ndarray,
    labels: Sequence[int] | None,
    out_path: str,
    method: str = "tsne",
    title: str | None = None,
    random_state: int = 0,
    adj: np.ndarray | None = None,
    edge_threshold: float = 0.5,
    **kwargs: Any,
) -> None:
    """
    Project high-dimensional feature embeddings to 2D and visualize them.

    Args:
        embeddings: (N, D) array of feature vectors.
        labels: optional length-N node labels (for coloring points).
        out_path: where to save the resulting PNG.
        method: "tsne", "umap", or "pca".
        adj: optional (N, N) adjacency / score matrix. If provided,
             edges with adj[i,j] >= edge_threshold will be drawn.
        edge_threshold: threshold on adj to decide which edges to draw.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D), got shape={embeddings.shape}")

    N = embeddings.shape[0]
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != N:
            raise ValueError(f"labels length {labels.shape[0]} does not match embeddings N={N}")

    # For very small N, t-SNE is overkill → switch to PCA
    if N < 11:
        return
    method = "pca"

    method = method.lower()
    if method == "tsne":
        perplexity = min(30.0, max(5.0, (N - 1) / 3.0))
        default_kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            init="pca",
        )
        default_kwargs.update(kwargs)
        projector = TSNE(random_state=random_state, **default_kwargs)
    elif method == "umap":
        if umap is None:
            raise ImportError("UMAP is not installed. Please `pip install umap-learn` to use method='umap'.")
        default_kwargs = dict(
            n_components=2,
            n_neighbors=min(15, max(5, N // 10)),
            min_dist=0.1,
        )
        default_kwargs.update(kwargs)
        projector = umap.UMAP(random_state=random_state, **default_kwargs)
    elif method == "pca":
        projector = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown method '{method}', expected 'tsne', 'umap' or 'pca'.")

    # Optional: pre-PCA if D is large and using t-SNE/UMAP
    if method in ("tsne", "umap") and embeddings.shape[1] > 50:
        pca = PCA(n_components=min(N, ), random_state=random_state)
        embeddings = pca.fit_transform(embeddings)

    emb_2d = projector.fit_transform(embeddings)  # (N, 2)

    # Build edge list from adjacency if provided
    edges = []
    if adj is not None:
        adj = np.asarray(adj)
        if adj.shape != (N, N):
            raise ValueError(f"adjacency shape {adj.shape} does not match N={N}")
        edges = adj_to_edges(adj, threshold=edge_threshold)

    plt.figure(figsize=(6, 6))

    # Draw edges (if any)
    for i, j in edges:
        x = [emb_2d[i, 0], emb_2d[j, 0]]
        y = [emb_2d[i, 1], emb_2d[j, 1]]
        plt.plot(x, y, linewidth=0.5, alpha=0.3, color="black")

    # Draw all nodes
    if labels is None:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=20, alpha=0.7)
    else:
        labels = labels.astype(int)
        uniq = np.unique(labels)
        for lab in uniq:
            mask = labels == lab
            plt.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=20,
                alpha=0.7,
                label=str(lab),
            )
        plt.legend(title="label", fontsize=8)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title(title or f"{method.upper()} embedding (N={N})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
 
# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_iou_curve(thresholds, ious, out_path):
    plt.figure()
    plt.plot(thresholds, ious)
    plt.xlabel("Threshold")
    plt.ylabel("Graph IoU")
    plt.title("Graph IoU vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, out_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(y_true, y_scores, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
