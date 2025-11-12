#!/usr/bin/env python
import os
import sys
import argparse

from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

# Make sure we can import vggt from the repo root
sys.path.append(os.path.abspath("vggt"))
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from save_embeds import make_image_embeddings

# Remove FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


PRED_ROOT = None


def discover_scene_splits(base_root: str) -> List[dict]:
    """
    Discover all (scene_version, split_id, saved_obs) under a single gvgg-{i} root or HM3D hvgg root.

    For HM3D (hvgg), search under all parts and drop the .basis suffix from scene_version.
    For Gibson (gvgg-{i}), search under base_root/temp/More_vis and base_root/More_vis.
    """
    results: List[dict] = []

    if "hvgg" in base_root:
        # HM3D layout: base_root/part{a..l}/temp/More_vis/{scene_id}.basis/{split}/saved_obs
        hm3d_parts = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        for part in hm3d_parts:
            part_root = os.path.join(base_root, f"part{part}", "temp", "More_vis")
            if not os.path.isdir(part_root):
                continue
            for scene_name in sorted(os.listdir(part_root)):
                scene_dir = os.path.join(part_root, scene_name)
                if not os.path.isdir(scene_dir):
                    continue
                # Remove .basis suffix for saving
                scene_id = scene_name.split(".basis")[0]
                for split_name in sorted(os.listdir(scene_dir)):
                    if not split_name.isdigit():
                        continue
                    split_dir = os.path.join(scene_dir, split_name)
                    if not os.path.isdir(split_dir):
                        continue
                    saved_obs = os.path.join(split_dir, "saved_obs")
                    if not os.path.isdir(saved_obs):
                        continue
                    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                    if not os.path.isfile(gt_csv):
                        continue
                    results.append(
                        dict(
                            scene_version=scene_id,
                            split_id=split_name,
                            saved_obs=saved_obs,
                            gt_csv=gt_csv,
                        )
                    )
        return results
    else:
        # Gibson gvgg-{i} layout: search under base_root/temp/More_vis and base_root/More_vis
        # This is the Gibson layout (gvgg-{i})
        candidates = [
            os.path.join(base_root, "temp", "More_vis"),
            os.path.join(base_root, "More_vis"),
        ]
        candidates = [c for c in candidates if os.path.isdir(c)]
        if not candidates:
            print(f"[WARN] No More_vis directories found under {base_root}")
            return results

        for more_vis_root in candidates:
            for scene_name in sorted(os.listdir(more_vis_root)):
                scene_dir = os.path.join(more_vis_root, scene_name)
                if not os.path.isdir(scene_dir):
                    continue

                for split_name in sorted(os.listdir(scene_dir)):
                    split_dir = os.path.join(scene_dir, split_name)
                    if not os.path.isdir(split_dir):
                        continue

                    saved_obs = os.path.join(split_dir, "saved_obs")
                    if not os.path.isdir(saved_obs):
                        continue

                    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                    if not os.path.isfile(gt_csv):
                        print(f"[WARN] Missing GroundTruth.csv in {saved_obs}, skipping.")
                        continue

                    results.append(
                        dict(
                            scene_version=scene_name,
                            split_id=split_name,
                            saved_obs=saved_obs,
                            gt_csv=gt_csv,
                        )
                    )

        return results

def extract_all_layer_embeds(predictions: dict, mode: str = "avg_max") -> np.ndarray:
    """
    Extract per-layer, per-view embeddings from VGGT predictions using the same
    per-image embedding logic as in save_embeds.make_image_embeddings.

    We use `features_all`, which is a list of length L.
    Each element is a tensor of features for all views in the batch.

    Returns:
        emb_layers: (L, S, E) float32
            L = num layers, S = views (e.g. 2 for a pair), E = embedding dim.
    """
    if "features_all" not in predictions:
        raise RuntimeError("predictions does not contain 'features_all'.")

    feat_layers = predictions["features_all"]
    if not isinstance(feat_layers, (list, tuple)) or len(feat_layers) == 0:
        raise RuntimeError("predictions['features_all'] must be a non-empty list of tensors.")

    layer_embs = []
    for feats_layer in feat_layers:
        # Reuse the same aggregation logic as in save_embeds.py
        emb = make_image_embeddings(feats_layer, mode=mode)  # (S, E) where S = #views
        layer_embs.append(emb.cpu().numpy().astype(np.float32))

    # Stack over layers → (L, S, E)
    emb_layers = np.stack(layer_embs, axis=0)
    return emb_layers

def process_scene_split(
    model: VGGT,
    device: torch.device,
    scene_version: str,
    split_id: str,
    saved_obs: str,
    gt_csv: str,
    mode: str = "avg_max",
) -> None:
    """
    For a single (scene_version, split_id), read GroundTruth.csv,
    run VGGT on each image pair, and save pair embeddings to:

        data/predictions_feat/{scene_version}/split_{split_id}/pair_embs/pairs.npz

    Args:
        mode: Controls how per-image embeddings are aggregated, consistent with save_embeds.make_image_embeddings.
    """
    # List all PNGs to define a consistent index space
    img_files = sorted(f for f in os.listdir(saved_obs) if f.endswith(".png"))
    if not img_files:
        print(f"[WARN] No PNGs found in {saved_obs}, skipping.")
        return

    name_to_idx = {f: idx for idx, f in enumerate(img_files)}

    df = pd.read_csv(gt_csv)
    if df.empty:
        print(f"[WARN] Empty GroundTruth.csv in {saved_obs}, skipping.")
        return

    # Optional continuous strengths from rel_mat.npy
    rel_mat_path = os.path.join(saved_obs, "rel_mat.npy")
    rel_mat = None
    if os.path.isfile(rel_mat_path):
        try:
            rel_mat = np.load(rel_mat_path)
        except Exception as e:
            print(f"[WARN] Failed to load rel_mat.npy in {saved_obs}: {e}")
            rel_mat = None

    pair_emb_i_list: List[np.ndarray] = []
    pair_emb_j_list: List[np.ndarray] = []
    labels: List[float] = []
    strengths: List[float] = []
    pair_i_idx: List[int] = []
    pair_j_idx: List[int] = []

    rows = list(df.itertuples(index=False))

    print(
        f"[INFO] Processing scene_version={scene_version}, split={split_id}, "
        f"pairs={len(rows)}"
    )

    for row in tqdm(rows, desc=f"{scene_version} split {split_id}", leave=False):
        # CSV columns: image_1, image_2, label
        img1_path = row[0]
        img2_path = row[1]
        label = float(row[2])

        basename1 = os.path.basename(img1_path)
        basename2 = os.path.basename(img2_path)

        if basename1 not in name_to_idx or basename2 not in name_to_idx:
            # Inconsistent path / filenames, skip
            continue

        i = name_to_idx[basename1]
        j = name_to_idx[basename2]

        full1 = os.path.join(saved_obs, basename1)
        full2 = os.path.join(saved_obs, basename2)

        if (not os.path.isfile(full1)) or (not os.path.isfile(full2)):
            continue

        # Load & preprocess the two images
        try:
            images_pre = load_and_preprocess_images(
                [full1, full2],
                mode="crop",
            ).to(device)
        except Exception as e:
            print(f"[WARN] Failed to load/preprocess pair ({full1}, {full2}): {e}")
            continue

        with torch.no_grad():
            preds = model(images_pre, extract_features=True)

        emb_layers = extract_all_layer_embeds(preds, mode=mode)  # (L, S, E)
        L, S, E = emb_layers.shape
        if S < 2:
            print(f"[WARN] Expected at least 2 views in embeddings for pair, got {emb_layers.shape}")
            continue

        emb_i = emb_layers[:, 0, :]  # (L, E) for image 1
        emb_j = emb_layers[:, 1, :]  # (L, E) for image 2

        # Continuous strength from rel_mat if available
        if rel_mat is not None and rel_mat.shape[0] > max(i, j):
            strength_ij = float(rel_mat[i, j])
        else:
            strength_ij = label

        pair_emb_i_list.append(emb_i)
        pair_emb_j_list.append(emb_j)
        labels.append(label)
        strengths.append(strength_ij)
        pair_i_idx.append(i)
        pair_j_idx.append(j)

    if not pair_emb_i_list:
        print(
            f"[WARN] No valid pairs processed for scene_version={scene_version}, "
            f"split={split_id}."
        )
        return

    # Stack into arrays
    emb_i = np.stack(pair_emb_i_list, axis=0)  # (P, L, E)
    emb_j = np.stack(pair_emb_j_list, axis=0)  # (P, L, E)
    labels_np = np.array(labels, dtype=np.float32)
    strengths_np = np.array(strengths, dtype=np.float32)
    pair_i_np = np.array(pair_i_idx, dtype=np.int32)
    pair_j_np = np.array(pair_j_idx, dtype=np.int32)
    img_files_np = np.array(img_files)

    out_dir = os.path.join(
        PRED_ROOT,
        scene_version,
        f"split_{split_id}",
        "pair_embs",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pairs_{mode}.npz")

    np.savez_compressed(
        out_path,
        emb_i=emb_i,
        emb_j=emb_j,
        labels=labels_np,
        strengths=strengths_np,
        pair_i=pair_i_np,
        pair_j=pair_j_np,
        img_files=img_files_np,
    )

    print(
        f"[INFO] Saved {emb_i.shape[0]} pair embeddings for "
        f"{scene_version} split {split_id} to {out_path}"
    )


def main():
    global PRED_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Save pair-wise VGGT all-layer embeddings for each scene_version/split "
            "in a given gvgg-{i} root."
        )
    )
    parser.add_argument(
        "--base_root",
        type=str,
        default="data/vast/cc7287/gvgg-1",
        help="Base root for Gibson (e.g. data/vast/cc7287/gvgg-1) or for HM3D: the shared hvgg root (e.g. data/scratch/cc7287/mvdust3r_projects/HM3D/dust3r_vpr_mask/data/hvgg)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for VGGT (e.g. cuda, mps, cpu). Default: auto",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=-1,
        help="Optional cap on number of scene_versions to process (for debugging).",
    )
    parser.add_argument(
        "--mode",
        choices=["avg", "avg_max", "chunked"],
        default="avg",
        help="How to aggregate VGGT features into per-image embeddings (avg, avg_max, chunked).",
    )
    args = parser.parse_args()

    # Device
    if args.device is not None:
        device_str = args.device
    else:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # Set PRED_ROOT based on base_root
    if "hvgg" in args.base_root:
        PRED_ROOT = "data/predictions_feat/hvgg"
    else:
        PRED_ROOT = "data/predictions_feat/gvgg"
    os.makedirs(PRED_ROOT, exist_ok=True)
    print(f"[INFO] Saving pair embeddings under: {PRED_ROOT}")

    # Init VGGT
    print("[INFO] Initializing VGGT...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("[INFO] VGGT initialized.")

    # Discover scene splits under this base_root
    scene_splits = discover_scene_splits(args.base_root)
    if not scene_splits:
        print(f"[WARN] No valid scene splits under {args.base_root}")
        return

    print(f"[INFO] Found {len(scene_splits)} (scene_version, split) combos.")

    # Optional cap
    if args.max_scenes > 0:
        scene_splits = scene_splits[: args.max_scenes]

    # Process each (scene_version, split_id)
    for meta in scene_splits:
        process_scene_split(
            model=model,
            device=device,
            scene_version=meta["scene_version"],
            split_id=meta["split_id"],
            saved_obs=meta["saved_obs"],
            gt_csv=meta["gt_csv"],
            mode=args.mode,
        )


if __name__ == "__main__":
    main()