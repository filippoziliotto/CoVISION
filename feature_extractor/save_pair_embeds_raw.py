#!/usr/bin/env python
"""
Extract raw per-layer VGGT pair embeddings and store them in a Zarr archive.

This mirrors feature_extractor/save_pair_embeds.py but keeps the full
(pairs, layers, dim) tensors for each view using chunked, compressed Zarr
arrays. We stick to the emb_i / emb_j layout to match the .npz structure.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
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
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt_trainer.model import VGGTHeadModel


PRED_ROOT = None


def _chunk_1d(length: int, cap: int = 1024) -> tuple:
    return (max(1, min(cap, int(length))),)


def save_pairs_to_zarr(
    out_path: str,
    emb_i: np.ndarray,
    emb_j: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    img_files: np.ndarray,
):
    """Write embeddings and metadata to a directory-backed Zarr store."""

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.SHUFFLE)
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=True)

    chunk_feats = (1, emb_i.shape[1], emb_i.shape[2])
    root.create_dataset(
        "emb_i",
        data=emb_i.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=chunk_feats,
    )
    root.create_dataset(
        "emb_j",
        data=emb_j.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=chunk_feats,
    )

    root.create_dataset(
        "labels",
        data=labels.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=_chunk_1d(labels.shape[0]),
    )
    root.create_dataset(
        "strengths",
        data=strengths.astype(np.float32),
        dtype=np.float32,
        compressor=compressor,
        chunks=_chunk_1d(strengths.shape[0]),
    )
    root.create_dataset(
        "pair_i",
        data=pair_i.astype(np.int32),
        dtype=np.int32,
        compressor=compressor,
        chunks=_chunk_1d(pair_i.shape[0]),
    )
    root.create_dataset(
        "pair_j",
        data=pair_j.astype(np.int32),
        dtype=np.int32,
        compressor=compressor,
        chunks=_chunk_1d(pair_j.shape[0]),
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
    gt_csv: str,
) -> None:
    """
    For a single (scene_version, split_id), read GroundTruth.csv, run VGGTHeadModel
    to collect per-layer embeddings for each pair, and write them to pairs_raw.zarr.
    """

    img_files = sorted(f for f in os.listdir(saved_obs) if f.endswith(".png"))
    if not img_files:
        print(f"[WARN] No PNGs found in {saved_obs}, skipping.")
        return

    name_to_idx = {f: idx for idx, f in enumerate(img_files)}

    df = pd.read_csv(gt_csv)
    if df.empty:
        print(f"[WARN] Empty GroundTruth.csv in {saved_obs}, skipping.")
        return

    rel_mat_path = os.path.join(saved_obs, "rel_mat.npy")
    rel_mat = None
    if os.path.isfile(rel_mat_path):
        try:
            rel_mat = np.load(rel_mat_path)
        except Exception as e:
            print(f"[WARN] Failed to load rel_mat.npy in {saved_obs}: {e}")

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
        img1_path = row[0]
        img2_path = row[1]
        label = float(row[2])

        basename1 = os.path.basename(img1_path)
        basename2 = os.path.basename(img2_path)

        if basename1 not in name_to_idx or basename2 not in name_to_idx:
            continue

        i = name_to_idx[basename1]
        j = name_to_idx[basename2]

        full1 = os.path.join(saved_obs, basename1)
        full2 = os.path.join(saved_obs, basename2)

        if (not os.path.isfile(full1)) or (not os.path.isfile(full2)):
            continue

        try:
            images_pre = load_and_preprocess_images([full1, full2], mode="crop").to(device)
        except Exception as e:
            print(f"[WARN] Failed to load/preprocess pair ({full1}, {full2}): {e}")
            continue

        feats = model.forward_features(images_pre)
        emb_all = feats["embeddings"].detach().cpu().numpy().astype(np.float32)
        if emb_all.shape[2] < 2:
            print(f"[WARN] Expected 2 views in embeddings, got {emb_all.shape}")
            continue

        emb_i = emb_all[0, :, 0, :]
        emb_j = emb_all[0, :, 1, :]

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

    emb_i_np = np.stack(pair_emb_i_list, axis=0)
    emb_j_np = np.stack(pair_emb_j_list, axis=0)
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
    out_path = os.path.join(out_dir, "pairs_raw.zarr")

    save_pairs_to_zarr(
        out_path,
        emb_i=emb_i_np,
        emb_j=emb_j_np,
        labels=labels_np,
        strengths=strengths_np,
        pair_i=pair_i_np,
        pair_j=pair_j_np,
        img_files=img_files_np,
    )

    print(
        f"[INFO] Saved {emb_i_np.shape[0]} raw pair embeddings for "
        f"{scene_version} split {split_id} to {out_path}"
    )


def main():
    global PRED_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Save per-layer VGGT pair embeddings (pre-head) into compressed Zarr stores "
            "for each scene_version/split in a given base root."
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
    print(f"[INFO] Saving pair embeddings under: {PRED_ROOT}")

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
            gt_csv=meta["gt_csv"],
        )


if __name__ == "__main__":
    main()
