#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

# Remove FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ---------------------------------------------------------------------
# Paths / constants (mirroring vggt_feat_eval.py)
# ---------------------------------------------------------------------
HM3D_ROOT_CANDIDATE = "data/scratch/cc7287/mvdust3r_projects/HM3D/nooooooo"
GIBSON_ROOT_CANDIDATE = "data/vast/cc7287/gvgg-1"

if os.path.isdir(HM3D_ROOT_CANDIDATE):
    # Use HM3D
    USE_HM3D = False
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

# ---------------------------------------------------------------------
# VGGT imports
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath("vggt"))
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images


# ---------------------------------------------------------------------
# Helper: build per-image embeddings from VGGT feature tensor
# ---------------------------------------------------------------------
def make_image_embeddings(
    feats_layer: torch.Tensor,
    target_hw: int = 8,
    mode: str = "avg_max",
    token_chunks: int = 4,
) -> torch.Tensor:
    """
    Convert a single VGGT feature tensor to per-image embeddings (N, D).

    feats_layer: one feature tensor for a given layer.
    Expected shapes (after model forward):
        - (1, N, C, H, W)  spatial maps
        - (1, N, C, P) or (1, N, P, C)  token features

    Args:
        target_hw: for spatial maps, target H=W when using 2D pooling.
        mode:
            - "avg": (default) previous behavior.
              * tokens  (N, C, P): global mean over tokens -> (N, C)
              * spatial (N, C, H, W): adaptive avg pool to (target_hw, target_hw),
                then flatten -> (N, C * target_hw * target_hw)
            - "avg_max": keep both mean and max statistics over tokens / pooled grid.
              * tokens: concat [mean, max] over tokens -> (N, 2*C)
              * spatial: pool to (target_hw, target_hw), flatten, then
                concat [mean, max] over spatial positions -> (N, 2*C*target_hw*target_hw)
            - "chunked": split tokens / spatial cells into `token_chunks` segments
              and keep their means; this preserves some coarse structure.
              * tokens: (N, C, P) -> (N, C * token_chunks)
              * spatial: pooled to (target_hw, target_hw), flattened to (N, C, P'),
                then chunked similarly.

        token_chunks: number of chunks for "chunked" mode.

    Returns:
        emb: (N, D) L2-normalized per-image embeddings (torch.Tensor, on CPU)
    """
    feats = feats_layer.squeeze(0)  # drop batch -> (N, ...)

    if feats.ndim == 3:
        # Token features: (N, C, P) or (N, P, C)
        if feats.shape[1] <= feats.shape[2]:
            # (N, C, P)
            feats_tok = feats
        else:
            # (N, P, C) -> (N, C, P)
            feats_tok = feats.permute(0, 2, 1)

        N, C, P = feats_tok.shape

        if mode == "avg":
            # Original behavior: global mean over tokens
            emb = feats_tok.mean(dim=-1)  # (N, C)

        elif mode == "avg_max":
            mean_tok = feats_tok.mean(dim=-1)       # (N, C)
            max_tok = feats_tok.amax(dim=-1)        # (N, C)
            emb = torch.cat([mean_tok, max_tok], dim=1)  # (N, 2*C)

        elif mode == "chunked":
            # Split tokens into token_chunks segments and average within each
            n_chunks = max(1, int(token_chunks))
            if P < n_chunks:
                # Fallback to simple mean if too few tokens
                emb = feats_tok.mean(dim=-1)
            else:
                # Trim so it is divisible by n_chunks
                P_trim = (P // n_chunks) * n_chunks
                feats_trim = feats_tok[:, :, :P_trim]                # (N, C, P_trim)
                feats_chunk = feats_trim.view(N, C, n_chunks, -1)    # (N, C, n_chunks, P_seg)
                mean_chunks = feats_chunk.mean(dim=-1)               # (N, C, n_chunks)
                emb = mean_chunks.view(N, C * n_chunks)              # (N, C * n_chunks)
        else:
            raise ValueError(f"Unknown mode '{mode}' for token features.")

    elif feats.ndim == 4:
        # Spatial maps: (N, C, H, W)
        N, C, H, W = feats.shape

        # First pool to a fixed grid for controllable dimension
        pooled = F.adaptive_avg_pool2d(feats, (target_hw, target_hw))  # (N, C, target_hw, target_hw)
        pooled_flat = pooled.view(N, C, -1)  # (N, C, P'), P' = target_hw * target_hw

        if mode == "avg":
            # Original behavior: keep all pooled cells flattened
            emb = pooled.view(N, -1)  # (N, C * target_hw * target_hw)

        elif mode == "avg_max":
            # Global mean and max over pooled cells per channel, but maintain grid structure
            mean_sp = pooled_flat.mean(dim=-1)        # (N, C)
            max_sp = pooled_flat.amax(dim=-1)         # (N, C)
            # Also keep the flattened grid so the representation is richer
            grid_flat = pooled.view(N, -1)            # (N, C * target_hw * target_hw)
            emb = torch.cat([grid_flat, mean_sp, max_sp], dim=1)  # (N, C*hw + 2*C)

        elif mode == "chunked":
            # Treat pooled cells as tokens and apply chunking over them
            P = pooled_flat.shape[-1]
            n_chunks = max(1, int(token_chunks))
            if P < n_chunks:
                emb = pooled.view(N, -1)
            else:
                P_trim = (P // n_chunks) * n_chunks
                feats_trim = pooled_flat[:, :, :P_trim]               # (N, C, P_trim)
                feats_chunk = feats_trim.view(N, C, n_chunks, -1)     # (N, C, n_chunks, P_seg)
                mean_chunks = feats_chunk.mean(dim=-1)                # (N, C, n_chunks)
                emb = mean_chunks.view(N, C * n_chunks)               # (N, C * n_chunks)
        else:
            raise ValueError(f"Unknown mode '{mode}' for spatial features.")

    else:
        raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

    emb = F.normalize(emb.cpu(), dim=1)  # (N, D)
    return emb


# ---------------------------------------------------------------------
# Per-split: extract and save embeddings
# ---------------------------------------------------------------------
def extract_split_embeddings(scene_id: str,
                             split_id: str,
                             split_dir: str,
                             model: VGGT,
                             device: str) -> None:
    """
    For a given scene split, run VGGT and save embeddings for all layers.

    Output:
      data/predictions_feat/{scene_id}/split_{split_id}/embs/all_embeds.npy
        with shape (L, N, E)
      data/predictions_feat/{scene_id}/split_{split_id}/embs/last_embeds.npy
        with shape (N, E)
    """
    png_dir = os.path.join(split_dir, "saved_obs")
    if not os.path.isdir(png_dir):
        print(f"[WARN] No saved_obs in split {split_id} of scene {scene_id}")
        return

    # Discover images
    img_files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    if len(img_files) == 0:
        print(f"[WARN] No PNGs in split {split_id} of scene {scene_id}")
        return

    images_abs = [os.path.join(png_dir, f) for f in img_files]

    # Prepare output dir
    # We now save under scene_id *with* its suffix (e.g. Adrian-1),
    # and use fixed filenames "all_embeds.npy" and "last_embeds.npy" per split.
    split_out_dir = os.path.join(PRED_ROOT, scene_id, f"split_{split_id}", "embs")
    os.makedirs(split_out_dir, exist_ok=True)
    out_all_path = os.path.join(split_out_dir, "all_embeds_avgmax.npz")
    out_last_path = os.path.join(split_out_dir, "last_embeds_avgmax.npz")

    # If already saved, skip
    if os.path.isfile(out_all_path):
        print(f"[INFO] Embeddings already exist for {scene_id} split {split_id}, skipping.")
        return

    # Run VGGT
    images_pre = load_and_preprocess_images(images_abs, mode="crop").to(device)
    with torch.no_grad():
        predictions = model(images_pre, extract_features=True)

    # Determine list of feature layers
    if "features_all" in predictions.keys():
        feat_layers = predictions["features_all"]
    else:
        raise RuntimeError(
            "VGGT predictions do not contain any of features_all / features / encoder_feats."
        )

    # Compute embeddings for all layers
    layer_embs = [make_image_embeddings(feats_layer) for feats_layer in feat_layers]

    # Check all embeddings have the same shape
    emb_shapes = [e.shape for e in layer_embs]
    if not all(s == emb_shapes[0] for s in emb_shapes):
        raise RuntimeError(
            f"Mismatch in embedding shapes across layers: {emb_shapes}"
        )

    all_emb = np.stack([e.numpy() for e in layer_embs], axis=0)  # (L, N, E)
    last_emb = all_emb[-1]  # (N, E)

    np.savez_compressed(out_all_path, all=all_emb, last=last_emb)
    # np.savez_compressed(out_last_path, last_emb)
    print(f"[INFO] Saved embeddings for scene {scene_id}, split {split_id} -> {out_all_path} (shape={all_emb.shape})")

    # load npz for verification
    # loaded = np.load(out_all_path).get("all")

# ---------------------------------------------------------------------
# Per-scene: loop splits
# ---------------------------------------------------------------------
def process_scene(scene_dir: str, model: VGGT, device: str = "cpu") -> None:
    """
    scene_dir:
      - HM3D:  .../More_vis/{scene_id}.basis
      - Gibson: .../More_vis/{SceneName}-1
    """
    scene_name = os.path.basename(scene_dir)

    if USE_HM3D:
        scene_id = scene_name.split(".basis")[0]
    else:
        # Gibson naming like "Adrian-1", keep the full name including suffix
        # so that different versions (Adrian-1, Adrian-2, ...) are separated.
        scene_id = scene_name

    print(f"[SCENE] {scene_id}")

    # Splits under this scene: numeric dirs containing saved_obs
    for split_name in sorted(os.listdir(scene_dir)):
        split_path = os.path.join(scene_dir, split_name)
        if not os.path.isdir(split_path):
            continue
        if not split_name.isdigit():
            continue

        saved_obs = os.path.join(split_path, "saved_obs")
        if not os.path.isdir(saved_obs):
            continue

        extract_split_embeddings(scene_id, split_name, split_path, model, device)


# ---------------------------------------------------------------------
# Main
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

    # Override dataset selection if requested
    if args.use_scene == "hm3dnooo":
        USE_HM3D = True
        HM3D_ROOT = os.path.join(
            "data/scratch/cc7287/mvdust3r_projects/HM3D",
            "dust3r_vpr_mask/data/hvgg/parta/",
        )
        HVGG_ROOT = HM3D_ROOT
        MORE_VIS_ROOT = os.path.join(HVGG_ROOT, "temp", "More_vis")
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
        if USE_HM3D:
            print("[INFO] Auto-detected: using HM3D dataset.")
        else:
            print("[INFO] Auto-detected: using Gibson dataset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    os.makedirs(PRED_ROOT, exist_ok=True)

    # Initialize VGGT once
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Collect scene dirs (same logic as vggt_feat_eval.py)
    scene_dirs = []
    if USE_HM3D:
        # HM3D: .basis dirs
        for name in sorted(os.listdir(MORE_VIS_ROOT)):
            if not name.endswith(".basis"):
                continue
            full_path = os.path.join(MORE_VIS_ROOT, name)
            if os.path.isdir(full_path):
                scene_dirs.append(full_path)
    else:
        # Gibson: look under More_vis or temp/More_vis
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
                if not fnmatch.fnmatch(name, "*1"):
                    continue
                scene_path = os.path.join(root, name)
                if not os.path.isdir(scene_path):
                    continue
                # Only keep scenes that actually have saved_obs in some split
                found = False
                for split_name in os.listdir(scene_path):
                    split_path = os.path.join(scene_path, split_name)
                    if not os.path.isdir(split_path):
                        continue
                    saved_obs = os.path.join(split_path, "saved_obs")
                    if os.path.isdir(saved_obs):
                        found = True
                        break
                if found:
                    scene_dirs.append(scene_path)

    if not scene_dirs:
        print(f"[ERROR] No scenes found under {MORE_VIS_ROOT}")
        return

    print(f"[INFO] Found {len(scene_dirs)} scenes.")

    # Process each scene
    for scene_dir in scene_dirs:
        process_scene(scene_dir, model, device=device)

    print("[INFO] Done extracting and saving embeddings.")


if __name__ == "__main__":
    main()