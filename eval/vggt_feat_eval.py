
#!/usr/bin/env python
import os
import argparse
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    average_precision_score,
)
from scipy.stats import pearsonr
from utils.visualize import (
    save_scene_feat_results, plot_feature_embedding_2d,
    plot_iou_curve, plot_pr_curve, plot_roc_curve)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Remove FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
# Automatically infer dataset root and config
HM3D_ROOT_CANDIDATE = "data/scratch/cc7287/mvdust3r_projects/HM3D"
GIBSON_ROOT_CANDIDATE = "data/vast/cc7287/gvgg-1"

if os.path.isdir(HM3D_ROOT_CANDIDATE):
    # Use HM3D
    USE_HM3D = True
    HM3D_ROOT = os.path.join(HM3D_ROOT_CANDIDATE, "dust3r_vpr_mask/data/hvgg/parta/")
    HVGG_ROOT = HM3D_ROOT
    MORE_VIS_ROOT = os.path.join(HVGG_ROOT, "temp/More_vis")
    BASE_CSV_NAME = "GroundTruth"
elif os.path.isdir(GIBSON_ROOT_CANDIDATE):
    # Use Gibson
    USE_HM3D = False
    GIBSON_ROOT = GIBSON_ROOT_CANDIDATE
    HVGG_ROOT = GIBSON_ROOT
    MORE_VIS_ROOT = os.path.join(GIBSON_ROOT)
    BASE_CSV_NAME = "GroundTruth"
else:
    raise RuntimeError(
        "Could not find either HM3D or Gibson dataset root. "
        f"Checked: '{HM3D_ROOT_CANDIDATE}' and '{GIBSON_ROOT_CANDIDATE}'"
    )

PRED_ROOT = "data/predictions_feat"
NUM_SAMPLES = 5000
VISUALIZE = True

# ---------------------------------------------------------------------
# VGGT imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath("vggt"))
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images

# ---------------------------------------------------------------------
# Per-split evaluation: {SCENE_ID}.basis/{split_id}/saved_obs
# ---------------------------------------------------------------------
def evaluate_split(scene_id, split_id, split_dir, model, device, scene_out_base, debug=False):
    """
    scene_id: e.g. 'E1NrAhMoqvB'
    split_id: string, e.g. '0', '1', ...
    split_dir: .../E1NrAhMoqvB.basis/0
    scene_out_base: data/predictions_feat/{scene_id}
    """

    png_dir = os.path.join(split_dir, "saved_obs")
    if not os.path.isdir(png_dir):
        print(f"[WARN] No saved_obs in split {split_id} of scene {scene_id}")
        return None
    
    # --- NEW: load per-split GroundTruth.csv ---
    gt_csv = os.path.join(png_dir, "GroundTruth.csv")
    if not os.path.isfile(gt_csv):
        print(f"[WARN] No GroundTruth.csv in {split_dir}")
        return None
    split_df = pd.read_csv(gt_csv)
    # Drop header row if it exists (check column names or first row)
    if not str(split_df.iloc[0, -1]).isdigit():
        split_df = split_df.iloc[1:].reset_index(drop=True)
        # columns: [path_or_name_a, path_or_name_b, label]

    img_files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    if len(img_files) < 2:
        print(f"[WARN] Not enough images in split {split_id} of scene {scene_id}")
        return None

    images_abs = [os.path.join(png_dir, f) for f in img_files]
    images_rel = [os.path.relpath(p, HVGG_ROOT).replace("\\", "/") for p in images_abs]
    images_rel = [f"./{p}" for p in images_rel]
    path_to_idx = {rel: idx for idx, rel in enumerate(images_rel)}

    if split_df.empty:
        print(f"[WARN] No GT pairs for scene {scene_id}, split {split_id}")
        return None

    # Output dir for this split
    split_out_dir = os.path.join(scene_out_base, f"split_{split_id}")
    os.makedirs(split_out_dir, exist_ok=True)

    # Run VGGT to extract features
    images_pre = load_and_preprocess_images(images_abs, mode="crop").to(device)
    with torch.no_grad():
        predictions = model(images_pre)
        
    # ---------------------------------------------------------------------
    # Feature layer selection
    # ---------------------------------------------------------------------
    # Options: "last", "2nd_last", "3rd_last", "all"
    LAYER_MODE = "last"
    #print(f"[INFO] Using layer mode: {LAYER_MODE}")
    
    def make_image_embeddings(feats_layer, target_hw=8):
        """
        feats_layer: one feature tensor for a given layer.
        Expected shapes (after model forward):
            - (1, N, C, H, W)  spatial maps
            - (1, N, C, P) or (1, N, P, C)  token features

        Returns:
            emb: (N, D) L2-normalized per-image embeddings
        """
        feats = feats_layer.squeeze(0)  # drop batch -> (N, ...)

        if feats.ndim == 3:
            # (N, C, P) OR (N, P, C)
            # heuristic: treat the middle dim as channels if it is smaller than the last
            if feats.shape[1] <= feats.shape[2]:
                # (N, C, P) -> avg over tokens
                emb = feats.mean(dim=-1)  # (N, C)
            else:
                # (N, P, C) -> (N, C, P) -> avg over tokens
                feats = feats.permute(0, 2, 1)  # (N, C, P)
                emb = feats.mean(dim=-1)        # (N, C)
        elif feats.ndim == 4:
            # (N, C, H, W) spatial maps
            N, C, H, W = feats.shape
            pooled = F.adaptive_avg_pool2d(feats, (target_hw, target_hw))  # (N, C, target_hw, target_hw)
            emb = pooled.view(N, -1)  # (N, C * target_hw * target_hw)
        else:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

        emb = F.normalize(emb.cpu(), dim=1)  # (N, D)
        return emb

    # Use intermediate encoder features for similarity
    # Prefer multi-layer outputs if available
    if "features_all" in predictions:
        feat_layers = predictions["features_all"]  # list of tensors
        L = len(feat_layers)
        
        if LAYER_MODE == "last" or L < 2:
            feats = feat_layers[-1]
        elif LAYER_MODE == "2nd_last" and L >= 2:
            feats = feat_layers[-2]
        elif LAYER_MODE == "3rd_last" and L >= 3:
            feats = feat_layers[-3]
        elif LAYER_MODE == "all":
            # average last up-to-3 layers (or fewer if not available)
            k = min(3, L)
            stacked = torch.stack(feat_layers[-k:], dim=0)  # (k, 1, N, ...)
            feats = stacked.mean(dim=0)                     # (1, N, ...)
        else:
            # fallback
            feats = feat_layers[-1]

    elif "encoder_feats" in predictions:
        feats = predictions["encoder_feats"]  # (1, N, C, H, W) or similar
    elif "features" in predictions:
        feats = predictions["features"]       # (1, N, ...)
    else:
        raise RuntimeError("VGGT output does not contain dense features (encoder_feats, features, or features_all)")

    # Convert selected layer to per-image embeddings
    pooled_feats_flat = make_image_embeddings(feats)

    # Utility: resolve GT csv paths to local indices
    def to_rel(path_or_name):
        s = str(path_or_name)
        # case 1: already a full hvgg-style path
        if s.startswith("./temp/More_vis") or s.startswith("temp/More_vis") or HVGG_ROOT in s:
            if s.startswith("./"):
                return s
            abs_p = s if os.path.isabs(s) else os.path.join(HVGG_ROOT, s.lstrip("./"))
            return abs_p.replace(HVGG_ROOT, "./")
        # case 2: just a filename like "best_color_0.png"
        abs_p = os.path.join(png_dir, s)
        return abs_p.replace(HVGG_ROOT, "./")

    # Scores
    y_true = []
    y_scores = []
    pair_indices = []

    for (img_a, img_b, label) in split_df.itertuples(index=False):
        label = int(label)
        img_a_rel = to_rel(img_a)
        img_b_rel = to_rel(img_b)
        if img_a_rel not in path_to_idx or img_b_rel not in path_to_idx:
            if debug:
                print(f"[WARN] skipping missing {img_a_rel} or {img_b_rel}")
            continue
        i = path_to_idx[img_a_rel]
        j = path_to_idx[img_b_rel]
        pair_indices.append((i, j))
        # Cosine similarity
        sim = torch.dot(pooled_feats_flat[i], pooled_feats_flat[j]).item()
        y_true.append(label)
        y_scores.append(sim)

    if len(y_true) == 0:
        print(f"[WARN] No valid pairs for scene {scene_id}, split {split_id}")
        return None

    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)
    n_pairs = len(y_true)

    # Per-split normalization (for IoU sweep only)
    low, high = np.percentile(y_scores, 1), np.percentile(y_scores, 99)
    norm_scores = np.clip((y_scores - low) / (high - low + 1e-8), 0.0, 1.0)

    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    for t in thresholds:
        preds_t = (norm_scores >= t).astype(int)
        inter = np.logical_and(preds_t, y_true).sum()
        union = np.logical_or(preds_t, y_true).sum()
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

    graph_iou_auc = auc(thresholds, ious)
    best_iou = max(ious)
    best_t = thresholds[np.argmax(ious)]
    y_pred = (norm_scores >= best_t).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_scores)) < 2:
        print(f"[WARN] Degenerate case for {scene_id} split {split_id}: constant labels or scores.")
        roc_auc = np.nan
        pr_auc = np.nan
        corr = np.nan
    else:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, y_scores)
        corr, _ = pearsonr(y_true, y_scores)

    print(f"=== Scene {scene_id}, split {split_id} ===")
    print(f"Pairs: {n_pairs}")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")
    print(f"Graph IoU (best): {best_iou:.3f} @ th={best_t:.3f}")
    print(f"Graph IoU AUC: {graph_iou_auc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f} | r: {corr:.3f}\n")

    # Optionally save per-split artifacts and adjacency matrices
    if VISUALIZE:
        # Save plots
        plot_iou_curve(thresholds, ious, os.path.join(split_out_dir, "iou_curve.png"))
        if not np.isnan(roc_auc):
            plot_roc_curve(fpr, tpr, roc_auc, os.path.join(split_out_dir, "roc_curve.png"))
        if not np.isnan(pr_auc):
            plot_pr_curve(y_true, y_scores, pr_auc, os.path.join(split_out_dir, "pr_curve.png"))

        # Save adjacency matrices and metrics via helper
        gt_adj, pred_adj, score_mat = save_scene_feat_results(
            out_root=PRED_ROOT,
            scene_id=scene_id,
            split_name=split_id,
            num_views=len(images_abs),
            pair_indices=pair_indices,
            y_true=y_true,
            y_scores=y_scores,
            best_threshold=best_t,
            metrics={
                "acc": acc,
                "f1": f1,
                "graph_iou_best": best_iou,
                "graph_iou_auc": graph_iou_auc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "pearson_r": corr,
            },
        )
        # Inside vggt_feat_eval.py, after you build adjacency:
        gt_adj = np.load(os.path.join(split_out_dir, "gt_adj.npy"))  # or use what you already have in memory
        
        # Plot feature embedding 2D visualization
        plot_feature_embedding_2d(
            embeddings=pooled_feats_flat.numpy(),
            labels=None,  # or some per-view labels if you have them
            adj=gt_adj,
            edge_threshold=0.5,
            out_path=os.path.join(split_out_dir, "tsne_gt_edges.png"),
            method="tsne",
            title=f"{scene_id} split {split_id} – GT graph in feature space",
        )

    return {
        "scene_id": scene_id,
        "split_id": split_id,
        "pairs": n_pairs,
        "acc": acc,
        "f1": f1,
        "graph_iou_best": best_iou,
        "graph_iou_auc": graph_iou_auc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pearson_r": corr,
    }


# ---------------------------------------------------------------------
# Per-scene evaluation: average over splits
# ---------------------------------------------------------------------
def evaluate_scene(scene_dir, model, device="cpu"):
    scene_name = os.path.basename(scene_dir)          # e.g. E1NrAhMoqvB.basis
    if USE_HM3D:
        scene_id = scene_name.split(".basis")[0]
    else:
        scene_id = scene_name.split("-")[0]  # e.g. data/vast/cc7287/gvgg-1/temp/More_vis/Adrian-1
    scene_out_base = os.path.join(PRED_ROOT, f"{scene_id}")
    os.makedirs(scene_out_base, exist_ok=True)

    # numeric splits under this scene
    split_results = []
    for split_name in sorted(os.listdir(scene_dir)):
        split_path = os.path.join(scene_dir, split_name)
        if not os.path.isdir(split_path):
            continue
        if not split_name.isdigit():
            continue
        res = evaluate_split(scene_id, split_name, split_path, model, device, scene_out_base)
        if res is not None:
            split_results.append(res)

    if not split_results:
        print(f"[WARN] No valid splits for scene {scene_id}")
        return None

    # pair-weighted scene averages
    total_pairs = sum(r["pairs"] for r in split_results)
    def wmean(key):
        return sum(r[key] * r["pairs"] for r in split_results) / total_pairs

    scene_metrics = {
        "scene_id": scene_id,
        "num_splits": len(split_results),
        "pairs": total_pairs,
        "acc": wmean("acc"),
        "f1": wmean("f1"),
        "graph_iou_best": wmean("graph_iou_best"),
        "graph_iou_auc": wmean("graph_iou_auc"),
        "roc_auc": wmean("roc_auc"),
        "pr_auc": wmean("pr_auc"),
        "pearson_r": wmean("pearson_r"),
    }

    print(f"=== Scene {scene_id} (averaged over {len(split_results)} splits) ===")
    print(f"Pairs: {total_pairs}")
    print(f"Accuracy: {scene_metrics['acc']:.3f} | F1: {scene_metrics['f1']:.3f}")
    print(f"Graph IoU*: {scene_metrics['graph_iou_best']:.3f}")
    print(f"Graph IoU AUC: {scene_metrics['graph_iou_auc']:.3f}")
    print(f"ROC AUC: {scene_metrics['roc_auc']:.3f}")
    print(f"PR AUC: {scene_metrics['pr_auc']:.3f}")
    print(f"Pearson r: {scene_metrics['pearson_r']:.3f}\n")

    # save per-scene metrics
    pd.DataFrame([scene_metrics]).to_csv(
        os.path.join(scene_out_base, "scene_metrics.csv"), index=False
    )

    return scene_metrics


# ---------------------------------------------------------------------
# Main: loop over scenes
# ---------------------------------------------------------------------
def main():
    global USE_HM3D, HM3D_ROOT, HVGG_ROOT, MORE_VIS_ROOT, BASE_CSV_NAME, GIBSON_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_scene",
        choices=["hm3d", "gibson"],
        default="gibson",
        help="Override automatic dataset detection and use the specified scene dataset.",
    )
    args = parser.parse_args()

    # Handle override of dataset detection logic
    if args.use_scene == "hm3d":
        USE_HM3D = True
        HM3D_ROOT = os.path.join("data/scratch/cc7287/mvdust3r_projects/HM3D", "dust3r_vpr_mask/data/hvgg/parta/")
        HVGG_ROOT = HM3D_ROOT
        MORE_VIS_ROOT = os.path.join(HVGG_ROOT, "temp/More_vis")
        BASE_CSV_NAME = "GroundTruth"
        print("[INFO] Overriding: using HM3D dataset.")
    elif args.use_scene == "gibson":
        USE_HM3D = False
        GIBSON_ROOT = "data/vast/cc7287/gvgg-1"
        HVGG_ROOT = GIBSON_ROOT
        MORE_VIS_ROOT = os.path.join(GIBSON_ROOT)
        BASE_CSV_NAME = "GroundTruth"
        print("[INFO] Overriding: using Gibson dataset.")
    else:
        print("[INFO] Using automatic dataset detection.")
        # USE_HM3D etc. are already set by the auto-detection logic above.
        if USE_HM3D:
            print("[INFO] Auto-detected: using HM3D dataset.")
        else:
            print("[INFO] Auto-detected: using Gibson dataset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    os.makedirs(PRED_ROOT, exist_ok=True)

    # --- Initialize model once ---
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # --- Collect all scene folders ---
    scene_dirs = []
    if USE_HM3D:
        # HM3D: use .basis dirs under MORE_VIS_ROOT
        for name in sorted(os.listdir(MORE_VIS_ROOT)):
            if not name.endswith(".basis"):
                continue
            full_path = os.path.join(MORE_VIS_ROOT, name)
            if os.path.isdir(full_path):
                scene_dirs.append(full_path)
    else:
        # Gibson: look for scene dirs matching "*-1" under More_vis, or temp/More_vis
        gibson_roots = []
        more_vis1 = os.path.join(MORE_VIS_ROOT, "More_vis")
        more_vis2 = os.path.join(MORE_VIS_ROOT, "temp", "More_vis")
        if os.path.isdir(more_vis1):
            gibson_roots.append(more_vis1)
        if os.path.isdir(more_vis2):
            gibson_roots.append(more_vis2)
        if not gibson_roots:
            print(f"[ERROR] No More_vis directory found under {MORE_VIS_ROOT}")
            return
        import fnmatch
        for root in gibson_roots:
            for name in sorted(os.listdir(root)):
                if not fnmatch.fnmatch(name, "*-1"):
                    continue
                scene_path = os.path.join(root, name)
                if not os.path.isdir(scene_path):
                    continue
                # Check if scene contains a split dir with saved_obs inside
                found = False
                for split_name in os.listdir(scene_path):
                    split_path = os.path.join(scene_path, split_name)
                    if not os.path.isdir(split_path):
                        continue
                    saved_obs_path = os.path.join(split_path, "saved_obs")
                    if os.path.isdir(saved_obs_path):
                        found = True
                        break
                if found:
                    scene_dirs.append(scene_path)

    if not scene_dirs:
        print(f"[ERROR] No scenes found under {MORE_VIS_ROOT}")
        return

    print(f"[INFO] Found {len(scene_dirs)} scenes.")

    # --- Evaluate each scene ---
    all_scene_results = []
    for scene_dir in scene_dirs:
        res = evaluate_scene(scene_dir, model, device=device)
        if res is not None:
            all_scene_results.append(res)

    if not all_scene_results:
        print("[WARN] No scenes evaluated.")
        return

    # --- Global (pair-weighted) averages across all scenes ---
    total_pairs = sum(r["pairs"] for r in all_scene_results)

    def wmean(key):
        return sum(r[key] * r["pairs"] for r in all_scene_results) / total_pairs

    print("\n=== Global Summary over Scenes ===")
    print(f"Scenes evaluated: {len(all_scene_results)}")
    print(f"Total pairs: {total_pairs}")
    print(f"Accuracy: {wmean('acc'):.3f}")
    print(f"F1:       {wmean('f1'):.3f}")
    print(f"Graph IoU*: {wmean('graph_iou_best'):.3f}")
    print(f"Graph IoU AUC: {wmean('graph_iou_auc'):.3f}")
    print(f"ROC AUC:  {wmean('roc_auc'):.3f}")
    print(f"PR AUC:   {wmean('pr_auc'):.3f}")
    print(f"Pearson r:{wmean('pearson_r'):.3f}")

    # Save all-scene summary
    summary_df = pd.DataFrame(all_scene_results)
    summary_path = os.path.join(PRED_ROOT, f"summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Global summary saved to {summary_path}")

if __name__ == "__main__":
    main()