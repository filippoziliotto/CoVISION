#!/usr/bin/env python
"""
Extract per-layer VGGT embeddings for all images in each split and store them
as compressed Zarr arrays. This mirrors feature_extractor/save_embeds.py but
uses the same forward-feature pipeline as save_pair_embeds_raw.py (no pooling
modes), yielding the token-summarised embeddings directly from VGGT.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import zarr

# Silence FutureWarnings from upstream dependencies
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Ensure imports work regardless of cwd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "vggt"))

from feature_extractor.save_pair_embeds import discover_scene_splits, _abs_path
from vggt_trainer.model import VGGTHeadModel
from vggt.vggt.utils.load_fn import load_and_preprocess_images


PRED_ROOT = None


def _chunk_1d(length: int, cap: int = 1024) -> tuple:
    return (max(1, min(cap, int(length))),)


def save_embeds_to_zarr(out_path: str, all_emb: np.ndarray, last_emb: np.ndarray, img_files: np.ndarray) -> None:
    """Write per-layer and last-layer embeddings plus filenames to a Zarr store."""

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.SHUFFLE)
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=True)

    # all_emb: (L, N, E)
    all_chunk = (1, max(1, min(64, all_emb.shape[1])), all_emb.shape[2])
    root.create_dataset(
        "all",
        data=all_emb.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=all_chunk,
    )

    # last_emb: (N, E)
    last_chunk = (max(1, min(256, last_emb.shape[0])), last_emb.shape[1])
    root.create_dataset(
        "last",
        data=last_emb.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=last_chunk,
    )

    if img_files.size > 0:
        max_len = max(len(s) for s in img_files)
    else:
        max_len = 1
    root.create_dataset(
        "img_files",
        data=img_files.astype(f"<U{max_len}"),
        dtype=f"<U{max_len}",
        compressor=compressor,
        chunks=_chunk_1d(img_files.shape[0]),
    )


def process_scene_split(
    model: VGGTHeadModel,
    device: torch.device,
    scene_version: str,
    split_id: str,
    saved_obs: str,
) -> None:
    """
    Extract embeddings for all images in a (scene_version, split_id) and save to Zarr.
    Embeddings are produced via VGGTHeadModel's token summarization pipeline (no pooling modes).
    """

    img_files = sorted(f for f in os.listdir(saved_obs) if f.endswith(".png"))
    if not img_files:
        print(f"[WARN] No PNGs found in {saved_obs}, skipping.")
        return

    images_abs = [os.path.join(saved_obs, f) for f in img_files]

    out_dir = os.path.join(PRED_ROOT, scene_version, f"split_{split_id}", "embs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embs_raw.zarr")

    if os.path.exists(out_path):
        print(f"[INFO] Embeddings already exist for {scene_version} split {split_id}, skipping.")
        return

    try:
        images_pre = load_and_preprocess_images(images_abs, mode="crop").to(device)
    except Exception as e:
        print(f"[WARN] Failed to load/preprocess images for {scene_version} split {split_id}: {e}")
        return

    # VGGT expects a batch dimension; wrap all images as one sequence of views.
    if images_pre.dim() == 4:
        images_pre = images_pre.unsqueeze(0)

    with torch.no_grad():
        preds = model.backbone(images_pre, extract_features=True)

    if "features_all" not in preds:
        print(f"[WARN] VGGT outputs missing 'features_all' for {scene_version} split {split_id}, skipping.")
        return

    raw_layers = [feat.detach() for feat in preds["features_all"]]
    embeddings = model._compute_layer_embeddings(raw_layers)  # (B, L, S, D)
    embeddings = model._select_layers(embeddings)

    all_emb = embeddings.squeeze(0).cpu().numpy().astype(np.float32)  # (L, N, D)
    last_emb = all_emb[-1]
    img_files_np = np.array(img_files)

    save_embeds_to_zarr(out_path, all_emb=all_emb, last_emb=last_emb, img_files=img_files_np)

    print(
        f"[INFO] Saved embeddings for {scene_version} split {split_id} to {out_path} "
        f"(shape={all_emb.shape})"
    )


def main():
    global PRED_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Save per-layer VGGT embeddings (per image) into compressed Zarr stores "
            "for each scene_version/split under a base root, using the forward-feature "
            "token summarization pipeline."
        )
    )
    parser.add_argument(
        "--base_root",
        type=str,
        default=_abs_path("data/vast/cc7287/gvgg-1"),
        help="Base root for Gibson (e.g. data/vast/cc7287/gvgg-1) or HM3D hvgg root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for VGGT (e.g. cuda, mps, cpu). Default: auto",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=-1,
        help="Optional cap on number of scene_versions to process (for debugging).",
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="all",
        help="Layer selection mode passed to VGGTHeadModel (e.g. all, last_stages, 2nd_last).",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default="facebook/VGGT-1B",
        help="Backbone checkpoint identifier passed to VGGTHeadModel.",
    )
    parser.add_argument(
        "--token_proj_dim",
        type=int,
        default=256,
        help="Token projection dimension used before summarization.",
    )
    parser.add_argument(
        "--summary_tokens",
        type=int,
        default=8,
        help="Number of summary tokens per layer per view.",
    )
    parser.add_argument(
        "--summary_heads",
        type=int,
        default=4,
        help="Attention heads for token summarization.",
    )
    args = parser.parse_args()

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

    base_root = _abs_path(args.base_root)
    if "hvgg" in base_root:
        PRED_ROOT = _abs_path("data/predictions_feat/hvgg")
    else:
        PRED_ROOT = _abs_path("data/predictions_feat/gvgg")
    os.makedirs(PRED_ROOT, exist_ok=True)
    print(f"[INFO] Saving embeddings under: {PRED_ROOT}")

    print("[INFO] Initializing VGGTHeadModel...")
    model = VGGTHeadModel(
        backbone_ckpt=args.backbone_ckpt,
        device=device_str,
        layer_mode=args.layer_mode,
        token_proj_dim=args.token_proj_dim,
        summary_tokens=args.summary_tokens,
        summary_heads=args.summary_heads,
    )
    model.eval()
    print("[INFO] VGGTHeadModel initialized.")

    scene_splits = discover_scene_splits(base_root)
    if not scene_splits:
        print(f"[WARN] No valid scene splits under {args.base_root}")
        return

    print(f"[INFO] Found {len(scene_splits)} (scene_version, split) combos.")

    if args.max_scenes > 0:
        scene_splits = scene_splits[: args.max_scenes]

    for meta in scene_splits:
        process_scene_split(
            model=model,
            device=device,
            scene_version=meta["scene_version"],
            split_id=meta["split_id"],
            saved_obs=meta["saved_obs"],
        )


if __name__ == "__main__":
    main()
