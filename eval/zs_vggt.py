#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Make repo root importable (so we can import vggt module)
sys.path.append(os.path.abspath("vggt"))

# Remove FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from sklearn.neighbors import KDTree
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
)
from scipy.stats import pearsonr

# VGGT imports (adjust if your paths differ)
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------------------------------------------------
# 1. Point cloud overlap utilities
# ---------------------------------------------------------
def build_pointclouds(world_points, num_samples=5000):
    """
    world_points: tensor (1, N, H, W, 3)
    Returns: list of N arrays (Mi, 3)
    """
    world_np = world_points.squeeze(0).cpu().numpy()  # (N, H, W, 3)
    pointclouds = []
    for i in range(world_np.shape[0]):
        pts = world_np[i].reshape(-1, 3)
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) > num_samples:
            idx = np.random.choice(len(pts), num_samples, replace=False)
            pts = pts[idx]
        pointclouds.append(pts)
    return pointclouds


def compute_eps_from_poses(poses_44, frac=0.05):
    """
    poses_44: list of (4,4) camera extrinsics [R|t]
    eps: fraction of average camera baseline
    """
    centers = np.array([p[:3, 3] for p in poses_44])
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    mask = dists > 0
    if not np.any(mask):
        return 0.05  # fallback
    avg_baseline = dists[mask].mean()
    eps = frac * avg_baseline
    return eps, avg_baseline


def symmetric_point_overlap(pts_a, pts_b, eps):
    """
    Symmetric overlap: average of A->B and B->A fractions of points
    within distance < eps.
    """
    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)

    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)

    f_ab = np.mean(d_ab < eps)
    f_ba = np.mean(d_ba < eps)
    return 0.5 * (f_ab + f_ba)


# ---------------------------------------------------------
# 2. Metrics (Co-VisiON style)
# ---------------------------------------------------------
def compute_metrics(y_true, y_scores, do_norm=True):
    """
    y_true:   (M,) array of {0,1}
    y_scores: (M,) array of continuous overlap scores in [0, +∞)
    Returns dict with Graph IoU best, IoU AUC, ROC AUC, PR AUC, etc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if len(y_true) == 0:
        return dict(
            pairs=0,
            acc=np.nan,
            f1=np.nan,
            graph_iou_best=np.nan,
            graph_iou_auc=np.nan,
            graph_iou_best_thres=np.nan,
            roc_auc=np.nan,
            pr_auc=np.nan,
            pearson_r=np.nan,
        )

    # Optional per-scene normalization for IoU curves
    if do_norm:
        low, high = np.percentile(y_scores, 1), np.percentile(y_scores, 99)
        norm_scores = np.clip((y_scores - low) / (high - low + 1e-8), 0.0, 1.0)
    else:
        # >>> paper-style: just use y_scores as is (assumes they’re already in [0,1])
        norm_scores = y_scores.copy()
        # If scores not in [0,1] you may want to rescale; here we assume they are.

    # Sweep thresholds in [0,1] on normalized scores
    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6
    y_true_bin = y_true.astype(np.float32)

    for t in thresholds:
        pred_t = (norm_scores >= t).astype(np.float32)
        inter = np.logical_and(pred_t, y_true_bin).sum()
        union = np.logical_or(pred_t, y_true_bin).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    graph_iou_auc = auc(thresholds, ious)
    best_iou = max(ious)
    best_thres = thresholds[np.argmax(ious)]

    # Classification at best IoU threshold
    y_pred = (norm_scores >= best_thres).astype(np.float32)
    acc = accuracy_score(y_true_bin, y_pred)
    f1 = f1_score(y_true_bin, y_pred)

    # Ranking metrics (on raw scores)
    if len(np.unique(y_true_bin)) < 2 or len(np.unique(y_scores)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
            roc_auc = auc(fpr, tpr)
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = average_precision_score(y_true_bin, y_scores)
        except Exception:
            pr_auc = np.nan

    # Pearson correlation between scores and labels
    try:
        corr, _ = pearsonr(y_true_bin, y_scores)
    except Exception:
        corr = np.nan

    return dict(
        pairs=len(y_true),
        acc=acc,
        f1=f1,
        graph_iou_best=best_iou,
        graph_iou_auc=graph_iou_auc,
        graph_iou_best_thres=best_thres,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        pearson_r=corr,
    )


# ---------------------------------------------------------
# 3. Evaluate one scene/split
# ---------------------------------------------------------
def evaluate_split(model, device, saved_obs_dir, do_norm=True, num_samples=5000):
    """
    saved_obs_dir: .../Scene-X/split_id/saved_obs
    Uses:
      - PNGs inside saved_obs_dir
      - GroundTruth.csv inside saved_obs_dir
    Returns metric dict or None if something fails.
    """
    gt_csv = os.path.join(saved_obs_dir, "GroundTruth.csv")
    if not os.path.isfile(gt_csv):
        print(f"[WARN] Missing GroundTruth.csv in {saved_obs_dir}, skipping")
        return None

    # 1) Load image paths (sorted to define consistent index)
    img_files = sorted(
        f for f in os.listdir(saved_obs_dir)
        if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
    )
    img_paths = [os.path.join(saved_obs_dir, f) for f in img_files]
    if len(img_paths) == 0:
        print(f"[WARN] No images in {saved_obs_dir}, skipping")
        return None

    # Map basename -> index
    name_to_idx = {os.path.basename(p): idx for idx, p in enumerate(img_paths)}

    # 2) Load + preprocess images for VGGT
    imgs_tensor = load_and_preprocess_images(img_paths, mode="crop").to(device)

    with torch.no_grad():
        predictions = model(imgs_tensor)

    world_points = predictions["world_points"]         # (1, N, H, W, 3)
    pose_encoding = predictions["pose_enc"]            # (1, N, 9)

    # 3) Decode poses for eps
    _, N, H, W, _ = world_points.shape   # (1, N, H, W, 3)
    image_hw = (H, W)

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )
    poses_44 = []
    for i in range(extrinsics.shape[1]):
        Rt = extrinsics[0, i].cpu().numpy()  # (3,4)
        pose44 = np.eye(4, dtype=np.float32)
        pose44[:3, :] = Rt
        poses_44.append(pose44)

    eps, avg_baseline = compute_eps_from_poses(poses_44, frac=0.05)
    print(f"[INFO] {saved_obs_dir}: N={N}, baseline={avg_baseline:.3f}, eps={eps:.4f}")

    # 4) Build point clouds
    pointclouds = build_pointclouds(world_points, num_samples=num_samples)

    # 5) Build y_true / y_scores from GroundTruth.csv
    df = pd.read_csv(gt_csv)
    y_true, y_scores = [], []

    for _, row in df.iterrows():
        p1 = os.path.basename(row["image_1"])
        p2 = os.path.basename(row["image_2"])
        label = int(row["label"])

        if p1 not in name_to_idx or p2 not in name_to_idx:
            print(f"[WARN] Missing image in index: {p1} or {p2}, skipping pair")
            continue

        i = name_to_idx[p1]
        j = name_to_idx[p2]

        frac = symmetric_point_overlap(pointclouds[i], pointclouds[j], eps=eps)
        y_true.append(label)
        y_scores.append(frac)

    if len(y_true) == 0:
        print(f"[WARN] No valid pairs in {saved_obs_dir}")
        return None

    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    metrics = compute_metrics(y_true, y_scores)
    return metrics


# ---------------------------------------------------------
# 4. Main loop over scenes
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot VGGT co-visibility evaluation using point clouds."
    )
    parser.add_argument(
        "--more_vis_root",
        type=str,
        default="data/vast/cc7287/gvgg-1/temp/More_vis",
        help="Root folder with scenes, e.g. data/vast/cc7287/gvgg-1/temp/More_vis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Max points per point cloud.",
    )
    parser.add_argument(
        "--no_norm",
        action="store_true",
        help="Disable per-scene percentile normalization of scores for IoU curves.",
    )
    args = parser.parse_args()

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Using device: {device}")

    # Load VGGT once
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    more_vis_root = args.more_vis_root
    if not os.path.isdir(more_vis_root):
        raise RuntimeError(f"{more_vis_root} is not a directory")

    scene_dirs = sorted(
        d for d in os.listdir(more_vis_root)
        if os.path.isdir(os.path.join(more_vis_root, d))
    )

    all_scene_metrics = []
    print(f"[INFO] Found {len(scene_dirs)} scenes under {more_vis_root}")

    for scene_name in scene_dirs:
        scene_dir = os.path.join(more_vis_root, scene_name)
        split_names = sorted(
            d for d in os.listdir(scene_dir)
            if os.path.isdir(os.path.join(scene_dir, d))
        )

        split_metrics = []
        for split_name in split_names:
            saved_obs = os.path.join(scene_dir, split_name, "saved_obs")
            if not os.path.isdir(saved_obs):
                continue

            m = evaluate_split(
                model,
                device,
                saved_obs,
                do_norm=not args.no_norm,
                num_samples=args.num_samples,
            )
            if m is not None:
                split_metrics.append((split_name, m))

        if not split_metrics:
            print(f"[WARN] No valid splits for scene {scene_name}")
            continue

        # Aggregate per-scene: average over splits
        pairs_total = sum(m["pairs"] for _, m in split_metrics)
        scene_graph_iou_best = np.mean([m["graph_iou_best"] for _, m in split_metrics])
        scene_graph_iou_auc = np.mean([m["graph_iou_auc"] for _, m in split_metrics])

        print(f"\n=== Scene {scene_name} ===")
        for split_name, m in split_metrics:
            print(
                f"  Split {split_name}: "
                f"pairs={m['pairs']} | "
                f"IoU_best={m['graph_iou_best']:.3f} | "
                f"IoU_AUC={m['graph_iou_auc']:.3f}"
            )
        print(
            f"  Scene avg: IoU_best={scene_graph_iou_best:.3f} | "
            f"IoU_AUC={scene_graph_iou_auc:.3f}"
        )

        all_scene_metrics.append(dict(
            scene=scene_name,
            pairs=pairs_total,
            graph_iou_best=scene_graph_iou_best,
            graph_iou_auc=scene_graph_iou_auc,
        ))

    if not all_scene_metrics:
        print("[WARN] No scenes evaluated.")
        return

    # Global averages (unweighted over scenes)
    avg_iou_best = np.mean([m["graph_iou_best"] for m in all_scene_metrics])
    avg_iou_auc = np.mean([m["graph_iou_auc"] for m in all_scene_metrics])
    total_pairs = sum(m["pairs"] for m in all_scene_metrics)

    print("\n=== Dataset Summary (scene-averaged) ===")
    print(f"Scenes: {len(all_scene_metrics)} | Total pairs: {total_pairs}")
    print(f"Graph IoU (best, scene-avg): {avg_iou_best:.3f}")
    print(f"Graph IoU AUC (scene-avg):   {avg_iou_auc:.3f}")


if __name__ == "__main__":
    main()