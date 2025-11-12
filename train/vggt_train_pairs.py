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
# Dataset import: load_dataset_pairs.py
# ---------------------------------------------------------------------
from dataset.load_dataset_pairs import build_dataloaders_pairs, build_dataloaders_pairs_cross
from utils.utils import set_seed, pairwise_ranking_loss
from models.PairView import MultiLayerEdgeClassifier, EdgeClassifier

warnings.filterwarnings("ignore")




# ---------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------
def train_epoch(
    classifier: nn.Module,
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
        # batch: (feat_i, feat_j, labels) or (feat_i, feat_j, labels, strengths)
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
            strengths = labels
        else:
            feat_i, feat_j, labels, strengths = batch

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)
        strengths = strengths.to(device)

        # Random swap augmentation
        if aug_swap_prob > 0.0:
            swap_mask = (torch.rand(labels.size(0), device=device) < aug_swap_prob)
            if swap_mask.any():
                tmp = feat_i[swap_mask].clone()
                feat_i[swap_mask] = feat_j[swap_mask]
                feat_j[swap_mask] = tmp

        preds = classifier(feat_i, feat_j)  # (B,)

        # Main BCE loss
        loss_bce = criterion(preds, labels)

        # Optional regression loss on strengths
        if use_reg_loss:
            probs = torch.sigmoid(preds)
            loss_reg = F.mse_loss(probs, strengths)
            loss = loss_bce + reg_lambda * loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=preds.device)
            loss = loss_bce

        # Optional soft IoU loss
        if use_iou_loss:
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
            if not use_reg_loss:
                probs = torch.sigmoid(preds)
            eps = 1e-6
            soft_inter = (probs * labels).sum()
            soft_union = probs.sum() + labels.sum() - soft_inter + eps
            soft_iou = soft_inter / soft_union

        total_soft_iou += soft_iou.item() * labels.numel()

        # Optional ranking loss
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
    classifier: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    use_reg_loss: bool = False,
    reg_lambda: float = 1.0,
    use_iou_loss: bool = False,
):
    classifier.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits, all_labels, all_strengths = [], [], []
    accs, f1s = [], []

    for batch in tqdm(val_loader, desc="Validation Progress"):
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
            strengths = labels
        else:
            feat_i, feat_j, labels, strengths = batch

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)
        strengths = strengths.to(device)

        logits = classifier(feat_i, feat_j)  # (B,)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_strengths.append(strengths.detach().cpu())

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
            iou_thresholds=None,
            iou_curve=None,
            soft_iou=np.nan,
        )

    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    strengths_all = torch.cat(all_strengths, dim=0)

    loss_bce = criterion(logits_all, labels_all)
    if use_reg_loss:
        probs_all_t = torch.sigmoid(logits_all)
        loss_reg = F.mse_loss(probs_all_t, strengths_all)
        loss = loss_bce + reg_lambda * loss_reg
    else:
        loss = loss_bce

    loss = loss.item()

    probs_all = torch.sigmoid(logits_all).cpu().numpy()
    y_true = labels_all.cpu().numpy()

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

    # Graph IoU sweep
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

    # Soft IoU (batch-level)
    if use_iou_loss:
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
    Zero-shot evaluation: cosine similarity between precomputed embeddings.
    """
    criterion = nn.BCELoss()

    all_scores, all_labels = [], []
    accs, f1s = [], []

    for batch in tqdm(val_loader, desc="Zero-shot Eval Progress"):
        if len(batch) == 3:
            feat_i, feat_j, labels = batch
        else:
            feat_i, feat_j, labels, _ = batch

        feat_i = feat_i.to(device)
        feat_j = feat_j.to(device)
        labels = labels.to(device)

        if feat_i.ndim == 2:
            sims = F.cosine_similarity(feat_i, feat_j, dim=-1)
        elif feat_i.ndim == 3:
            feat_i_norm = F.normalize(feat_i, dim=-1)
            feat_j_norm = F.normalize(feat_j, dim=-1)
            sims_layer = (feat_i_norm * feat_j_norm).sum(dim=-1)  # (B, L)
            sims = sims_layer.mean(dim=1)
        else:
            raise ValueError(
                f"Unexpected feature shapes in zero_shot_eval: {feat_i.shape}, {feat_j.shape}"
            )

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
            iou_thresholds=None,
            iou_curve=None,
        )

    y_scores = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    loss = criterion(
        torch.from_numpy(y_scores.astype(np.float32)),
        torch.from_numpy(y_true.astype(np.float32)),
    ).item()

    y_bin = (y_scores >= 0.5).astype(np.float32)
    y_true_bin = (y_true >= 0.5).astype(np.float32)
    acc = np.mean(accs)
    f1 = np.mean(f1s)

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
        description="Train EdgeClassifier on pair-level VGGT embeddings (pairs.npz)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="train/classifier_pairs")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
    )
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="Co-Vision")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_off",
        action="store_true",
    )
    # Early stopping + LR scheduler
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set, run zero-shot eval (cosine similarity) only.",
    )
    parser.add_argument(
        "--aug_swap_prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--use_reg_loss",
        action="store_true",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--use_rank_loss",
        action="store_true",
    )
    parser.add_argument(
        "--rank_lambda",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rank_margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rank_num_samples",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="1st_last",
        choices=["all", "1st_last", "2nd_last", "3rd_last", "4th_last"],
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
    )
    parser.add_argument(
        "--use_iou_loss",
        action="store_true",
    )
    parser.add_argument(
        "--iou_lambda",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["hm3d", "gibson"],
        help="Dataset type / source of pair embeddings: 'gibson' or 'hm3d'.",
    )
    parser.add_argument(
        "--emb_mode",
            type=str,
            default="avg",
            choices=["avg", "avg_max", "chunked"],
            help="Embedding aggregation mode.",
    )
    parser.add_argument(
        "--keep_all_data",
        action="store_true",
        help="If set, do not subsample negatives (keep all data). Overrides max_neg_ratio.",
    )
    parser.add_argument(
        "--hard_neg_ratio",
        type=float,
        default=0.5,
        help="Fraction of sampled negatives that are hard.",
    )
    parser.add_argument(
        "--max_neg_ratio",
        type=float,
        default=1.0,
        help="Maximum number of negatives to keep per positive (ratio). Use <=0 to keep all negatives.",
    )
    parser.add_argument(
        "--eval_dataset_type",
        type=str,
        default=None,
        choices=["hm3d", "gibson"],
        help=(
            "If set (and different from --dataset_type), train on --dataset_type "
            "and evaluate on this dataset type using a held-out split."
        ),
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

    # Build dataloaders (pair-level)
    if args.eval_dataset_type is not None and args.eval_dataset_type != args.dataset_type:
        # Cross-dataset setup: train on args.dataset_type, evaluate on args.eval_dataset_type
        train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs_cross(
            train_dataset_type=args.dataset_type,
            eval_dataset_type=args.eval_dataset_type,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=4,
            max_neg_ratio=args.max_neg_ratio if not args.keep_all_data else -1.0,
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_rel_thr=0.3,
            layer_mode=args.layer_mode,
            split_mode=args.data_split_mode,
            emb_mode=args.emb_mode,
        )
    else:
        # Standard single-dataset setup: train and eval on the same dataset
        default_train_ratio = 0.8 if args.dataset_type == "gibson" else 0.9
        train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs(
            dataset_type=args.dataset_type,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=4,
            train_ratio=default_train_ratio,
            max_neg_ratio=args.max_neg_ratio if not args.keep_all_data else -1.0,
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_rel_thr=0.3,
            layer_mode=args.layer_mode,
            split_mode=args.data_split_mode,
            emb_mode=args.emb_mode,
        )

    print(
        f"[INFO] Dataset: train_pairs={len(train_ds)}, "
        f"val_pairs={len(val_ds)}"
    )
    print(
        f"[INFO] Pair graphs: total={meta['num_graphs']}, "
        f"train={meta['num_train_graphs']}, val={meta['num_val_graphs']}"
    )
    print(
        f"[INFO] Scenes: total={meta['num_scenes']}, "
        f"train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )

    # Zero-shot mode
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

        zs_plots_dir = os.path.join(args.out_dir, "zero_shot_plots")
        os.makedirs(zs_plots_dir, exist_ok=True)

        thr_train = train_metrics.get("iou_thresholds", None)
        iou_train = train_metrics.get("iou_curve", None)
        thr_val = val_metrics.get("iou_thresholds", None)
        iou_val = val_metrics.get("iou_curve", None)

        if thr_train is not None and iou_train is not None and thr_val is not None and iou_val is not None:
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
            wandb.log(
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
            wandb.finish()

        return

    # Infer emb_dim
    first_batch = next(iter(train_loader))
    feat_i_batch = first_batch[0]
    if feat_i_batch.ndim == 2:
        emb_dim = feat_i_batch.shape[1]
    elif feat_i_batch.ndim == 3:
        emb_dim = feat_i_batch.shape[-1]
    else:
        raise ValueError(f"Unexpected feature batch shape: {feat_i_batch.shape}")
    print(f"[INFO] Embedding dim: {emb_dim}")

    classifier = EdgeClassifier(emb_dim=emb_dim, hidden_dim=256).to(device)
    # classifier = MultiLayerEdgeClassifier(emb_dim=emb_dim, hidden_dim=256).to(device)

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

    # LR scheduler on val Graph IoU AUC
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    best_val_graph_iou_auc = -float("inf")
    epochs_no_improve = 0

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

        scheduler.step(val_metrics["graph_iou_auc"])

        current_val_iou_auc = val_metrics["graph_iou_auc"]
        if (
            not np.isnan(current_val_iou_auc)
            and current_val_iou_auc > best_val_graph_iou_auc + args.es_min_delta
        ):
            best_val_graph_iou_auc = current_val_iou_auc
            epochs_no_improve = 0
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

        # Per-epoch IoU curves
        thr_train = train_eval_metrics.get("iou_thresholds", None)
        iou_train = train_eval_metrics.get("iou_curve", None)
        thr_val = val_metrics.get("iou_thresholds", None)
        iou_val = val_metrics.get("iou_curve", None)

        if thr_train is not None and iou_train is not None and thr_val is not None and iou_val is not None:
            plots_dir = os.path.join(args.out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
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
            plt.savefig(os.path.join(plots_dir, "iou_curve_latest.png"), dpi=200)
            plt.close()

        if epochs_no_improve >= args.es_patience:
            print(
                f"[EarlyStopping] No improvement in val Graph IoU AUC "
                f"for {epochs_no_improve} epochs. Stopping training."
            )
            break

    # Final plots
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    num_epochs = len(history["train_loss"])
    if num_epochs > 0:
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

        # ROC AUC
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["val_roc_auc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("ROC AUC")
        plt.title("Validation ROC AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_auc.png"), dpi=200)
        plt.close()

        # Graph IoU best
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
    clf_path = os.path.join(args.out_dir, "edge_classifier_pairs.pth")
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