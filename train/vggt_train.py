#!/usr/bin/env python
import sys
import os
from tqdm import tqdm

# Add the main folder to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Dataset import: load_dataset.py
# ---------------------------------------------------------------------
from dataset.load_dataset import build_dataloaders

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Edge classifier head (trainable)
# ---------------------------------------------------------------------
class EdgeClassifier(nn.Module):
    """
    Takes two node embeddings e_i, e_j (E-dim each) and predicts edge strength in [0,1].
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout_p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(p=dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Dropout(p=dropout_p),
            #nn.Sigmoid(),  # output in [0,1]
        )
        self.layernorm = nn.LayerNorm(4 * emb_dim)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        emb_i, emb_j:
            - shape (B, E)  for single-layer embeddings
            - or    (B, L, E) for multi-layer embeddings (L layers)

        Returns:
            - shape (B,) edge scores (logits)
        """
        if emb_i.ndim == 2:
            # Single-layer case: (B, E)
            x = torch.cat(
                [emb_i, emb_j, torch.abs(emb_i - emb_j), emb_i * emb_j],
                dim=-1,
            )  # (B, 4E)
            x = self.layernorm(x)
            out = self.mlp(x).squeeze(-1)  # (B,)
            return out

        elif emb_i.ndim == 3:
            # Multi-layer case: (B, L, E)
            B, L, E = emb_i.shape
            emb_i_flat = emb_i.reshape(B * L, E)
            emb_j_flat = emb_j.reshape(B * L, E)

            x = torch.cat(
                [
                    emb_i_flat,
                    emb_j_flat,
                    torch.abs(emb_i_flat - emb_j_flat),
                    emb_i_flat * emb_j_flat,
                ],
                dim=-1,
            )  # (B*L, 4E)
            x = self.layernorm(x)
            out_flat = self.mlp(x).squeeze(-1)  # (B*L,)

            # Reshape back to (B, L) and average scores over layers
            out = out_flat.view(B, L).mean(dim=1)  # (B,)
            return out

        else:
            raise ValueError(
                f"EdgeClassifier expected emb_i with ndim 2 or 3, got shape={emb_i.shape}"
            )


# ---------------------------------------------------------------------
# MultiLayerEdgeClassifier: uses first, middle, last VGGT layers
# ---------------------------------------------------------------------
class MultiLayerEdgeClassifier(nn.Module):
    """
    Edge classifier that explicitly uses first, middle, and last VGGT layers,
    with separate MLPs for low/mid/high features and a learned fusion before final prediction.

    Inputs:
        emb_i, emb_j:
            - (B, E)       for single-layer embeddings
            - (B, L, E)    for multi-layer embeddings (L layers)
    Output:
        - (B,) logits for edge presence/strength
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout_p: float = 0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        # Separate layer norms for low/mid/high branches
        self.pair_ln_low = nn.LayerNorm(4 * emb_dim)
        self.pair_ln_mid = nn.LayerNorm(4 * emb_dim)
        self.pair_ln_high = nn.LayerNorm(4 * emb_dim)

        # Shared MLP definition helper
        def make_pair_mlp():
            return nn.Sequential(
                nn.Linear(4 * emb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
            )

        # Three separate MLPs for low/mid/high-level pair features
        self.pair_mlp_low = make_pair_mlp()
        self.pair_mlp_mid = make_pair_mlp()
        self.pair_mlp_high = make_pair_mlp()

        # Fusion MLP to combine low/mid/high features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * (hidden_dim // 2), hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

        # Final classifier head
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_pair(self, e_i: torch.Tensor, e_j: torch.Tensor, ln: nn.Module, mlp: nn.Module) -> torch.Tensor:
        """Encode a single pair of embeddings (B, E) → (B, hidden_dim // 2)."""
        x = torch.cat(
            [e_i, e_j, torch.abs(e_i - e_j), e_i * e_j],
            dim=-1,
        )  # (B, 4E)
        x = ln(x)
        h = mlp(x)
        return h

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        # Single-layer case
        if emb_i.ndim == 2:
            h_single = self._encode_pair(emb_i, emb_j, self.pair_ln_mid, self.pair_mlp_mid)
            logits = self.final_mlp(torch.cat([h_single, h_single, h_single], dim=-1))  # mimic 3-branch fusion
            return logits.squeeze(-1)

        # Multi-layer case
        if emb_i.ndim != 3 or emb_j.ndim != 3:
            raise ValueError(
                f"MultiLayerEdgeClassifier expects emb_i, emb_j with ndim 2 or 3, got {emb_i.shape}, {emb_j.shape}"
            )

        B, L, E = emb_i.shape
        if E != self.emb_dim:
            raise ValueError(f"Expected embedding dim={self.emb_dim}, got {E} (shape={emb_i.shape})")

        # Pick representative layers
        idx_first = 0
        idx_mid = L // 2
        idx_last = L - 1

        # Extract layer embeddings
        e_i_first, e_j_first = emb_i[:, idx_first, :], emb_j[:, idx_first, :]
        e_i_mid, e_j_mid = emb_i[:, idx_mid, :], emb_j[:, idx_mid, :]
        e_i_last, e_j_last = emb_i[:, idx_last, :], emb_j[:, idx_last, :]

        # Encode pairs at each scale
        h_first = self._encode_pair(e_i_first, e_j_first, self.pair_ln_low, self.pair_mlp_low)
        h_mid = self._encode_pair(e_i_mid, e_j_mid, self.pair_ln_mid, self.pair_mlp_mid)
        h_last = self._encode_pair(e_i_last, e_j_last, self.pair_ln_high, self.pair_mlp_high)

        # Fuse features (learned combination instead of mean)
        h_cat = torch.cat([h_first, h_mid, h_last], dim=-1)  # (B, 3 * hidden_dim // 2)
        h_fused = self.fusion_mlp(h_cat)

        logits = self.final_mlp(h_fused).squeeze(-1)  # (B,)
        return logits

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

# ---------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------
def train_epoch(
    classifier: EdgeClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    max_grad_norm: float = 0.0,
    aug_swap_prob: float = 0.5,
    use_reg_loss: bool = False,
    reg_lambda: float = 1.0,
    use_iou_loss: bool = False,
    iou_lambda: float = 0.3,
    use_rank_loss: bool = False,
    rank_lambda: float = 0.1,
    rank_margin: float = 0.1,
    rank_num_samples: int = 1024,
):
    classifier.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_edges = 0
    accs, f1s = [], []
    total_soft_iou = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Progress")):
        # Support (feat_i, feat_j, labels) or (feat_i, feat_j, labels, strengths)
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
            strengths = labels  # fallback: same as binary
        else:
            feat_i, feat_j, labels, strengths = batch

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)
        strengths = strengths.to(device)
        
        # 1) Randomly swap (feat_i, feat_j) for some pairs
        if aug_swap_prob > 0.0:
            swap_mask = (torch.rand(labels.size(0), device=device) < aug_swap_prob)
            if swap_mask.any():
                tmp = feat_i[swap_mask].clone()
                feat_i[swap_mask] = feat_j[swap_mask]
                feat_j[swap_mask] = tmp

        preds = classifier(feat_i, feat_j)  # (B,)

        # Main BCE loss on binary labels
        loss_bce = criterion(preds, labels)

        # Optional regression loss on continuous strengths from rel_mat
        if use_reg_loss:
            probs = torch.sigmoid(preds)  # (B,) in [0,1]
            loss_reg = F.mse_loss(probs, strengths)
            loss = loss_bce + reg_lambda * loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=preds.device)
            loss = loss_bce

        # Optional soft IoU loss
        if use_iou_loss:
            # Only compute probs if not already done
            if not use_reg_loss:
                probs = torch.sigmoid(preds)
            eps = 1e-6
            soft_inter = (probs * labels).sum()
            soft_union = probs.sum() + labels.sum() - soft_inter + eps
            soft_iou = soft_inter / soft_union
            loss_iou = 1.0 - soft_iou
            loss = loss + iou_lambda * loss_iou
        else:
            loss_iou = torch.tensor(0.0, device=preds.device)
            # still define soft_iou for logging
            if not use_reg_loss:
                probs = torch.sigmoid(preds)
            eps = 1e-6
            soft_inter = (probs * labels).sum()
            soft_union = probs.sum() + labels.sum() - soft_inter + eps
            soft_iou = soft_inter / soft_union

        # Accumulate total_soft_iou
        total_soft_iou += soft_iou.item() * labels.numel()

        # Optional pairwise ranking loss on continuous strengths
        if use_rank_loss:
            loss_rank = pairwise_ranking_loss(
                scores=preds,
                strengths=strengths,
                margin=rank_margin,
                num_samples=rank_num_samples,
            )
            loss = loss + rank_lambda * loss_rank
        else:
            loss_rank = torch.tensor(0.0, device=preds.device)

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_edges += labels.numel()

        # Metrics
        y_pred = (preds >= 0.5).float().cpu().numpy()
        y_true = (labels >= 0.5).float().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    avg_loss = total_loss / max(1, total_edges)
    avg_acc = np.mean(accs) if accs else np.nan
    avg_f1 = np.mean(f1s) if f1s else np.nan
    if use_iou_loss:
        avg_soft_iou = total_soft_iou / max(1, total_edges)
    else:
        avg_soft_iou = float("nan")

    return dict(loss=avg_loss, acc=avg_acc, f1=avg_f1, soft_iou=avg_soft_iou)


@torch.no_grad()
def eval_epoch(
    classifier: EdgeClassifier,
    val_loader: DataLoader,
    device: str = "cpu",
    use_reg_loss: bool = False,
    reg_lambda: float = 1.0,
    use_iou_loss: bool = False,
):
    """Evaluate classifier on validation loader.

    Uses BCEWithLogitsLoss on raw logits and computes metrics on
    sigmoid probabilities in [0,1].
    """
    classifier.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits, all_labels, all_strengths = [], [], []
    accs, f1s = [], []

    for batch in tqdm(val_loader, desc="Validation Progress"):
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
            strengths = labels  # fallback
        else:
            feat_i, feat_j, labels, strengths = batch

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)
        strengths = strengths.to(device)

        logits = classifier(feat_i, feat_j)  # (B,) raw logits

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_strengths.append(strengths.detach().cpu())

        # For per-batch metrics, threshold at 0.5 on sigmoid outputs
        probs = torch.sigmoid(logits)
        y_pred = (probs >= 0.5).float().cpu().numpy()
        y_true = (labels >= 0.5).float().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    if not all_logits:
        return dict(
            loss=np.nan,
            acc=np.nan,
            f1=np.nan,
            roc_auc=np.nan,
            pr_auc=np.nan,
            graph_iou_best=np.nan,
            graph_iou_auc=np.nan,
            graph_iou_best_thres=np.nan,
        )

    # Stack all logits and labels
    logits_all = torch.cat(all_logits, dim=0)      # (N,)
    labels_all = torch.cat(all_labels, dim=0)      # (N,)
    strengths_all = torch.cat(all_strengths, dim=0)  # (N,)

    # Main BCE loss
    loss_bce = criterion(logits_all, labels_all)

    # Optional regression loss
    if use_reg_loss:
        probs_all_t = torch.sigmoid(logits_all)
        loss_reg = F.mse_loss(probs_all_t, strengths_all)
        loss = loss_bce + reg_lambda * loss_reg
    else:
        loss = loss_bce

    loss = loss.item()

    # Probabilities in [0,1] for metrics
    probs_all = torch.sigmoid(logits_all).cpu().numpy()
    y_true = labels_all.cpu().numpy()

    # Binary metrics at threshold 0.5
    y_bin = (probs_all >= 0.5).astype(np.float32)
    y_true_bin = (y_true >= 0.5).astype(np.float32)
    acc = np.mean(accs)
    f1 = np.mean(f1s)

    # ROC/PR AUC
    if len(np.unique(y_true_bin)) < 2 or len(np.unique(probs_all)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        try:
            roc_auc = roc_auc_score(y_true_bin, probs_all)
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_true_bin, probs_all)
        except Exception:
            pr_auc = np.nan

    # Graph IoU sweep and AUC over thresholds in [0,1]
    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6

    for t in thresholds:
        pred_t = (probs_all >= t).astype(np.float32)
        inter = np.logical_and(pred_t, y_true_bin).sum()
        union = np.logical_or(pred_t, y_true_bin).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    if hasattr(np, "trapezoid"):
        graph_iou_auc = np.trapezoid(ious, thresholds)
    else:
        graph_iou_auc = np.trapz(ious, thresholds)
    best_iou = max(ious)
    best_thres = thresholds[np.argmax(ious)]

    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    ious_arr = np.asarray(ious, dtype=np.float32)

    # Soft IoU (batch-level, not thresholded)
    if use_iou_loss:
        # Use sigmoid probs and true labels
        probs_torch = torch.sigmoid(logits_all)
        labels_torch = labels_all
        eps = 1e-6
        soft_inter = (probs_torch * labels_torch).sum()
        soft_union = probs_torch.sum() + labels_torch.sum() - soft_inter + eps
        soft_iou_value = (soft_inter / soft_union).item()
    else:
        soft_iou_value = float("nan")

    return dict(
        loss=loss,
        acc=acc,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        graph_iou_best=best_iou,
        graph_iou_auc=graph_iou_auc,
        graph_iou_best_thres=best_thres,
        iou_thresholds=thresholds_arr,
        iou_curve=ious_arr,
        soft_iou=soft_iou_value,
    )


# ---------------------------------------------------------------------
# Zero-shot evaluation: cosine similarity only
# ---------------------------------------------------------------------
@torch.no_grad()
def zero_shot_eval(
    val_loader: DataLoader,
    device: str = "cpu",
):
    """
    Zero-shot evaluation: no classifier, just use cosine similarity between
    precomputed embeddings as edge scores.
    """
    criterion = nn.BCELoss()

    all_scores, all_labels = [], []
    accs, f1s = [], []

    for batch in tqdm(val_loader, desc="Zero-shot Eval Progress"):
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
        else:
            feat_i, feat_j, labels, _ = batch  # ignore strengths

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)

        # Cosine similarity:
        # - if features are (B, E): one similarity per pair
        # - if features are (B, L, E): one similarity per layer, then averaged.
        if feat_i.ndim == 2:
            sims = F.cosine_similarity(feat_i, feat_j, dim=-1)  # (B,)
        elif feat_i.ndim == 3:
            # Normalize along feature dim, compute per-layer sims, then average over L
            feat_i_norm = F.normalize(feat_i, dim=-1)
            feat_j_norm = F.normalize(feat_j, dim=-1)
            sims_layer = (feat_i_norm * feat_j_norm).sum(dim=-1)  # (B, L)
            sims = sims_layer.mean(dim=1)  # (B,)
        else:
            raise ValueError(
                f"Unexpected feature shapes in zero_shot_eval: {feat_i.shape}, {feat_j.shape}"
            )

        # Map to [0, 1] to be comparable to probabilities
        preds = (sims + 1.0) / 2.0

        all_scores.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        y_pred = (preds >= 0.5).float().cpu().numpy()
        y_true = (labels >= 0.5).float().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    if not all_scores:
        return dict(
            loss=np.nan,
            acc=np.nan,
            f1=np.nan,
            roc_auc=np.nan,
            pr_auc=np.nan,
            graph_iou_best=np.nan,
            graph_iou_auc=np.nan,
            graph_iou_best_thres=np.nan,
        )

    y_scores = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # "Loss" here is just BCE between mapped similarities and labels
    loss = criterion(
        torch.from_numpy(y_scores.astype(np.float32)),
        torch.from_numpy(y_true.astype(np.float32)),
    ).item()

    # Binary metrics at 0.5
    y_bin = (y_scores >= 0.5).astype(np.float32)
    y_true_bin = (y_true >= 0.5).astype(np.float32)
    acc = np.mean(accs)
    f1 = np.mean(f1s)

    # ROC/PR AUC
    if len(np.unique(y_true_bin)) < 2 or len(np.unique(y_scores)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        try:
            roc_auc = roc_auc_score(y_true_bin, y_scores)
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_true_bin, y_scores)
        except Exception:
            pr_auc = np.nan

    # Graph IoU sweep and AUC over thresholds
    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6

    for t in thresholds:
        pred_t = (y_scores >= t).astype(np.float32)
        inter = np.logical_and(pred_t, y_true_bin).sum()
        union = np.logical_or(pred_t, y_true_bin).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    if hasattr(np, "trapezoid"):
        graph_iou_auc = np.trapezoid(ious, thresholds)
    else:
        graph_iou_auc = np.trapz(ious, thresholds)
    best_iou = max(ious)
    best_thres = thresholds[np.argmax(ious)]

    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    ious_arr = np.asarray(ious, dtype=np.float32)

    return dict(
        loss=loss,
        acc=acc,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        graph_iou_best=best_iou,
        graph_iou_auc=graph_iou_auc,
        graph_iou_best_thres=best_thres,
        iou_thresholds=thresholds_arr,
        iou_curve=ious_arr,
    )

# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train EdgeClassifier on precomputed embeddings/adjacency (CoVisGraphDataset)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-5,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of graphs per batch).",
    )
    parser.add_argument(
        "--max_neg_ratio",
        type=float,
        default=1.0,
        help="Max negative:positive ratio for edge pairs.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="train/classifier")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0.0 disables clipping).",
    )
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="Co-Vision")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="Disable wandb logging",
    )
    # Early stopping + LR scheduler
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without Graph IoU AUC improvement).",
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in val Graph IoU AUC to reset early stopping.",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Factor to reduce LR on plateau (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
        help="LR scheduler patience in epochs (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set, run zero-shot evaluation using cosine similarity only (no training).",
    )
    parser.add_argument(
        "--aug_swap_prob",
        type=float,
        default=0.5,
        help="Probability of swapping node pairs as data augmentation during training.",
    )
    parser.add_argument(
        "--use_reg_loss",
        action="store_true",
        help="If set, add auxiliary regression loss to continuous edge strengths (rel_mat).",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.3,
        help="Weight for the regression loss term.",
    )
    parser.add_argument(
        "--use_rank_loss",
        action="store_true",
        help="If set, add a pairwise ranking loss based on continuous edge strengths.",
    )
    parser.add_argument(
        "--rank_lambda",
        type=float,
        default=0.1,
        help="Weight for the ranking loss term.",
    )
    parser.add_argument(
        "--rank_margin",
        type=float,
        default=0.1,
        help="Margin for the ranking loss on score differences.",
    )
    parser.add_argument(
        "--rank_num_samples",
        type=int,
        default=1024,
        help="Number of pairwise constraints sampled per batch for ranking loss.",
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="1st_last",
        choices=["all", "1st_last", "2nd_last", "3rd_last", "4th_last"],
        help="Which VGGT layer(s) to use: 'all' or one of the last-k layers.",
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
        help="Split mode: 'scene_disjoint' (default), 'version_disjoint', or 'graph'."
    )
    parser.add_argument(
        "--use_iou_loss",
        action="store_true",
        help="If set, add a batch-level soft IoU loss term."
    )
    parser.add_argument(
        "--iou_lambda",
        type=float,
        default=0.3,
        help="Weight for the soft IoU loss term."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/val split ratio."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["hm3d", "gibson"],
        help="Dataset type / source of embeddings: 'gibson' or 'hm3d'."
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Using device: {device}")

    # Wandb (optional)
    try:
        import wandb
        _wandb_available = hasattr(wandb, "init")
        if not _wandb_available:
            print("[WARN] wandb imported but has no 'init' attribute. Disabling wandb logging.")
    except ImportError:
        wandb = None  # type: ignore
        _wandb_available = False

    use_wandb = (
        (not args.wandb_off)
        and _wandb_available
        and (args.wandb_project is not None)
    )
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize wandb ({e}). Disabling wandb logging.")
            use_wandb = False

    # Build dataloaders
    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=4,
        train_ratio=0.8 if args.dataset_type == "gibson" else 0.9,
        max_neg_ratio=args.max_neg_ratio,
        hard_neg_ratio=0.5,
        hard_neg_rel_thr=0.2,
        layer_mode=args.layer_mode,
        split_mode=args.data_split_mode,
    )

    print(
        f"[INFO] Dataset: train_pairs={len(train_ds)}, "
        f"val_pairs={len(val_ds)}"
    )
    print(
        f"[INFO] Graphs: total={meta['num_graphs']}, "
        f"train={meta['num_train_graphs']}, val={meta['num_val_graphs']}"
    )
    print(
        f"[INFO] Scenes: total={meta['num_scenes']}, "
        f"train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )

    # Zero-shot mode: just evaluate cosine similarity between embeddings
    if args.zero_shot:
        print("[INFO] Running zero-shot evaluation (no training).")

        train_metrics = zero_shot_eval(train_loader, device=device)
        val_metrics = zero_shot_eval(val_loader, device=device)

        print(
            "[Zero-shot Train] "
            f"acc={train_metrics['acc']:.3f} | "
            f"f1={train_metrics['f1']:.3f} | "
            f"roc_auc={train_metrics['roc_auc']:.3f} | "
            f"pr_auc={train_metrics['pr_auc']:.3f} | "
            f"graph_iou_best={train_metrics['graph_iou_best']:.3f} | "
            f"graph_iou_auc={train_metrics['graph_iou_auc']:.3f}"
        )
        print(
            "[Zero-shot Val]   "
            f"acc={val_metrics['acc']:.3f} | "
            f"f1={val_metrics['f1']:.3f} | "
            f"roc_auc={val_metrics['roc_auc']:.3f} | "
            f"pr_auc={val_metrics['pr_auc']:.3f} | "
            f"graph_iou_best={val_metrics['graph_iou_best']:.3f} | "
            f"graph_iou_auc={val_metrics['graph_iou_auc']:.3f}"
        )

        # Plot IoU vs threshold for zero-shot train/val
        zs_plots_dir = os.path.join(args.out_dir, "zero_shot_plots")
        os.makedirs(zs_plots_dir, exist_ok=True)

        thr_train = train_metrics.get("iou_thresholds", None)
        iou_train = train_metrics.get("iou_curve", None)
        thr_val = val_metrics.get("iou_thresholds", None)
        iou_val = val_metrics.get("iou_curve", None)

        if thr_train is not None and iou_train is not None and thr_val is not None and iou_val is not None:
            # Use validation thresholds for x-axis if both are the same length
            x_thr = thr_val if thr_val.shape == thr_train.shape else thr_val
            plt.figure(figsize=(6, 4))
            plt.plot(x_thr, iou_train, label="train")
            plt.plot(x_thr, iou_val, label="val")
            plt.xlabel("Threshold")
            plt.ylabel("Graph IoU")
            plt.title("Zero-shot Graph IoU vs Threshold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(zs_plots_dir, "zero_shot_iou_curve.png"), dpi=200)
            plt.close()

        if use_wandb:
            import wandb as _wandb_mod
            _wandb_mod.log(
                {
                    "zero_shot_train_acc": train_metrics["acc"],
                    "zero_shot_train_f1": train_metrics["f1"],
                    "zero_shot_train_roc_auc": train_metrics["roc_auc"],
                    "zero_shot_train_pr_auc": train_metrics["pr_auc"],
                    "zero_shot_train_graph_iou_best": train_metrics["graph_iou_best"],
                    "zero_shot_train_graph_iou_auc": train_metrics["graph_iou_auc"],
                    "zero_shot_val_acc": val_metrics["acc"],
                    "zero_shot_val_f1": val_metrics["f1"],
                    "zero_shot_val_roc_auc": val_metrics["roc_auc"],
                    "zero_shot_val_pr_auc": val_metrics["pr_auc"],
                    "zero_shot_val_graph_iou_best": val_metrics["graph_iou_best"],
                    "zero_shot_val_graph_iou_auc": val_metrics["graph_iou_auc"],
                }
            )
            _wandb_mod.finish()

        return

    # Infer emb_dim from first sample
    first_batch = next(iter(train_loader))
    feat_i_batch = first_batch[0]  # (B, E) or (B, L, E)
    if feat_i_batch.ndim == 2:
        emb_dim = feat_i_batch.shape[1]
    elif feat_i_batch.ndim == 3:
        emb_dim = feat_i_batch.shape[-1]
    else:
        raise ValueError(f"Unexpected feature batch shape: {feat_i_batch.shape}")
    print(f"[INFO] Embedding dim: {emb_dim}")

    classifier = EdgeClassifier(emb_dim=emb_dim, hidden_dim=256).to(device)
    # swap to multi-layer
    #classifier = MultiLayerEdgeClassifier(emb_dim=emb_dim, hidden_dim=256).to(device)
    
    # Trainable parameters
    n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params}")

    # Optimizer
    if args.optimizer.lower() == "adamw":
        OptimCls = torch.optim.AdamW
    else:
        OptimCls = torch.optim.Adam

    optimizer = OptimCls(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"[INFO] Optimizer: {args.optimizer.upper()} "
        f"(lr={args.lr}, weight_decay={args.weight_decay})"
    )

    # LR scheduler (ReduceLROnPlateau on validation Graph IoU AUC)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # maximize val_graph_iou_auc
        factor=args.lr_factor,
        patience=args.lr_patience,
        #verbose=True,
    )

    # Early stopping state (based on val_graph_iou_auc, higher is better)
    best_val_graph_iou_auc = -float("inf")
    epochs_no_improve = 0

    # History tracking for metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "val_graph_iou_best": [],
        "val_graph_iou_auc": [],
        "train_graph_iou_best": [],
        "train_graph_iou_auc": [],
        "train_soft_iou": [],
        "val_soft_iou": [],
    }

    print("[INFO] Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            classifier,
            train_loader,
            optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
            aug_swap_prob=args.aug_swap_prob,
            use_reg_loss=args.use_reg_loss,
            reg_lambda=args.reg_lambda,
            use_iou_loss=args.use_iou_loss,
            iou_lambda=args.iou_lambda,
            use_rank_loss=args.use_rank_loss,
            rank_lambda=args.rank_lambda,
            rank_margin=args.rank_margin,
            rank_num_samples=args.rank_num_samples,
        )
        # Evaluate on training set for IoU curve (no gradient)
        train_eval_metrics = eval_epoch(
            classifier,
            train_loader,
            device=device,
            use_reg_loss=args.use_reg_loss,
            reg_lambda=args.reg_lambda,
            use_iou_loss=args.use_iou_loss,
        )
        val_metrics = eval_epoch(
            classifier,
            val_loader,
            device=device,
            use_reg_loss=args.use_reg_loss,
            reg_lambda=args.reg_lambda,
            use_iou_loss=args.use_iou_loss,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.3f} | "
            f"train_f1={train_metrics['f1']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.3f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.3f} | "
            f"val_graph_iou_best={val_metrics['graph_iou_best']:.3f} | "
            f"val_graph_iou_auc={val_metrics['graph_iou_auc']:.3f}"
        )

        # Step LR scheduler on validation Graph IoU AUC
        scheduler.step(val_metrics["graph_iou_auc"])

        # Early stopping logic based on validation Graph IoU AUC
        current_val_iou_auc = val_metrics["graph_iou_auc"]
        if (
            not np.isnan(current_val_iou_auc)
            and current_val_iou_auc > best_val_graph_iou_auc + args.es_min_delta
        ):
            best_val_graph_iou_auc = current_val_iou_auc
            epochs_no_improve = 0
            # Optionally, you could save a "best" checkpoint here
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}] "
            f"LR={current_lr:.6e} | "
            f"no_improve={epochs_no_improve} | "
            f"best_val_graph_iou_auc={best_val_graph_iou_auc:.3f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "train_f1": train_metrics["f1"],
                    "train_soft_iou": train_metrics["soft_iou"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_graph_iou_best": val_metrics["graph_iou_best"],
                    "val_graph_iou_auc": val_metrics["graph_iou_auc"],
                    "val_graph_iou_best_thres": val_metrics["graph_iou_best_thres"],
                    "val_soft_iou": val_metrics["soft_iou"],
                    "lr": current_lr,
                },
                step=epoch,
            )

        # Append history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_graph_iou_best"].append(val_metrics["graph_iou_best"])
        history["val_graph_iou_auc"].append(val_metrics["graph_iou_auc"])
        history["train_graph_iou_best"].append(train_eval_metrics["graph_iou_best"])
        history["train_graph_iou_auc"].append(train_eval_metrics["graph_iou_auc"])
        history["train_soft_iou"].append(train_metrics["soft_iou"])
        history["val_soft_iou"].append(val_metrics["soft_iou"])

        # Plot per-epoch Graph IoU vs threshold for train and val
        thr_train = train_eval_metrics.get("iou_thresholds", None)
        iou_train = train_eval_metrics.get("iou_curve", None)
        thr_val = val_metrics.get("iou_thresholds", None)
        iou_val = val_metrics.get("iou_curve", None)

        if thr_train is not None and iou_train is not None and thr_val is not None and iou_val is not None:
            plots_dir = os.path.join(args.out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            # Use validation thresholds for x-axis if shapes match, otherwise use validation's
            x_thr = thr_val if thr_val.shape == thr_train.shape else thr_val
            plt.figure(figsize=(6, 4))
            plt.plot(x_thr, iou_train, label="train")
            plt.plot(x_thr, iou_val, label="val")
            plt.xlabel("Threshold")
            plt.ylabel("Graph IoU")
            plt.title(f"Graph IoU vs Threshold (Epoch {epoch:03d})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"iou_curve_epoch_{epoch:03d}.png"), dpi=200)
            # Also overwrite a 'latest' curve for convenience
            plt.savefig(os.path.join(plots_dir, "iou_curve_latest.png"), dpi=200)
            plt.close()

        # Early stopping check
        if epochs_no_improve >= args.es_patience:
            print(
                f"[EarlyStopping] No improvement in val Graph IoU AUC "
                f"for {epochs_no_improve} epochs. Stopping training."
            )
            break

    # Save plots of metrics
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    num_epochs = len(history["train_loss"])
    if num_epochs == 0:
        print("[WARN] No epochs logged, skipping plots.")
    else:
        epochs = np.arange(1, num_epochs + 1)

        # Loss
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_loss"], label="train")
        plt.plot(epochs, history["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss.png"), dpi=200)
        plt.close()

        # Accuracy
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_acc"], label="train")
        plt.plot(epochs, history["val_acc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train/Val Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy.png"), dpi=200)
        plt.close()

        # F1
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_f1"], label="train")
        plt.plot(epochs, history["val_f1"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.title("Train/Val F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "f1.png"), dpi=200)
        plt.close()

        # ROC AUC (val)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["val_roc_auc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("ROC AUC")
        plt.title("Validation ROC AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_auc.png"), dpi=200)
        plt.close()

        # Graph IoU (best)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["val_graph_iou_best"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Graph IoU (best)")
        plt.title("Validation Graph IoU (best)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "graph_iou_best.png"), dpi=200)
        plt.close()

        # Graph IoU AUC
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["val_graph_iou_auc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Graph IoU AUC")
        plt.title("Validation Graph IoU AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "graph_iou_auc.png"), dpi=200)
        plt.close()

        # Soft IoU
        if len(history["train_soft_iou"]) > 0 and len(history["val_soft_iou"]) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, history["train_soft_iou"], label="train")
            plt.plot(epochs, history["val_soft_iou"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Soft IoU")
            plt.title("Train/Val Soft IoU")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "soft_iou.png"), dpi=200)
            plt.close()

    # Save classifier
    clf_path = os.path.join(args.out_dir, "edge_classifier.pth")
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "emb_dim": emb_dim,
            "config": vars(args),
        },
        clf_path,
    )
    print(f"[INFO] Saved classifier to {clf_path}")

    if use_wandb:
        wandb.save(clf_path)
        wandb.finish()


if __name__ == "__main__":
    main()