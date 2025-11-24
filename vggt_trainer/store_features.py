#!/usr/bin/env python
"""
Precompute and store preprocessed images/pairs into Zarr shards for fast training.

Supports both pairwise and multiview layouts:
  - Pairwise: stacked pair tensors (N, 2, 3, H, W) + labels/strengths.
  - Multiview: one group per scene containing images (N, 3, H, W), pair indices (P, 2),
    labels/strengths, and scene metadata.

This script reuses the discovery/splitting logic from vggt_trainer.data to stay aligned
with the main training data path.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from collections import OrderedDict

import numpy as np
import torch
import zarr

# Ensure repository root (and bundled vggt module) is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "vggt") not in sys.path:
    sys.path.append(str(REPO_ROOT / "vggt"))

from vggt_trainer.data import (  # noqa: E402
    PairRecord,
    _discover_scene_splits,
    _load_pairs_for_split,
    _load_scene_pairs,
    _split_meta,
    load_and_preprocess_images,
    load_and_preprocess_images_square,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'. Expected fp16/bf16/fp32.")


def _str_array(strings: Sequence[str]) -> np.ndarray:
    max_len = max(1, max(len(s) for s in strings))
    return np.asarray(strings, dtype=f"<U{max_len}")


def _write_pair_shard(
    shard_pairs: List[Tuple[np.ndarray, float, float, str, str]],
    out_dir: Path,
    shard_idx: int,
    compressor,
) -> None:
    if not shard_pairs:
        return
    store = zarr.DirectoryStore(out_dir / f"pair_shard_{shard_idx:05d}.zarr")
    root = zarr.group(store=store, overwrite=True)

    images = np.stack([p[0] for p in shard_pairs], axis=0)  # (N,2,3,H,W)
    labels = np.array([p[1] for p in shard_pairs], dtype=np.float32)
    strengths = np.array([p[2] for p in shard_pairs], dtype=np.float32)
    scene_versions = _str_array([p[3] for p in shard_pairs])
    split_ids = _str_array([p[4] for p in shard_pairs])

    root.create_dataset(
        "images",
        data=images,
        compressor=compressor,
        dtype=images.dtype,
        chunks=(1,) + images.shape[1:],
    )
    root.create_dataset("labels", data=labels, compressor=compressor, chunks=(min(1024, len(labels)),))
    root.create_dataset("strengths", data=strengths, compressor=compressor, chunks=(min(1024, len(strengths)),))
    root.create_dataset("scene_version", data=scene_versions, compressor=compressor, chunks=(min(1024, len(scene_versions)),))
    root.create_dataset("split_id", data=split_ids, compressor=compressor, chunks=(min(1024, len(split_ids)),))
    root.attrs["count"] = len(shard_pairs)
    print(f"[WRITE] Pair shard {shard_idx} -> {store.path} (samples={len(shard_pairs)})")


def _write_scene_shard(
    scenes: List[dict],
    out_dir: Path,
    shard_idx: int,
    compressor,
) -> None:
    if not scenes:
        return
    store = zarr.DirectoryStore(out_dir / f"scene_shard_{shard_idx:05d}.zarr")
    root = zarr.group(store=store, overwrite=True)

    for idx, scene in enumerate(scenes):
        grp = root.create_group(f"scene_{idx:05d}")
        grp.create_dataset(
            "images",
            data=scene["images"],
            compressor=compressor,
            dtype=scene["images"].dtype,
            chunks=(1,) + scene["images"].shape[1:],
        )
        pair_chunk = max(1, min(1024, scene["pairs"].shape[0]))
        grp.create_dataset(
            "pairs",
            data=scene["pairs"],
            compressor=compressor,
            dtype=scene["pairs"].dtype,
            chunks=(pair_chunk, 2),
        )
        label_chunk = max(1, min(1024, len(scene["labels"])))
        grp.create_dataset(
            "labels",
            data=scene["labels"],
            compressor=compressor,
            dtype=scene["labels"].dtype,
            chunks=(label_chunk,),
        )
        strength_chunk = max(1, min(1024, len(scene["strengths"])))
        grp.create_dataset(
            "strengths",
            data=scene["strengths"],
            compressor=compressor,
            dtype=scene["strengths"].dtype,
            chunks=(strength_chunk,),
        )
        grp.attrs["scene_version"] = scene["scene_version"]
        grp.attrs["split_id"] = scene["split_id"]
    root.attrs["count"] = len(scenes)
    print(f"[WRITE] Scene shard {shard_idx} -> {store.path} (scenes={len(scenes)})")


def _load_pair_tensor(
    record: PairRecord,
    preprocess_mode: str,
    square_size: int,
    dtype: torch.dtype,
    image_cache: Optional["OrderedDict[str, torch.Tensor]"] = None,
    max_cache_items: int = 0,
) -> np.ndarray:
    def _load_single(path: str) -> torch.Tensor:
        if image_cache is not None and path in image_cache:
            tensor = image_cache.pop(path)
            image_cache[path] = tensor
            return tensor

        if preprocess_mode == "square":
            img, _ = load_and_preprocess_images_square([path], target_size=square_size)
            tensor = img[0]
        else:
            img = load_and_preprocess_images([path], mode=preprocess_mode)
            tensor = img[0]

        if image_cache is not None:
            image_cache[path] = tensor
            if max_cache_items > 0 and len(image_cache) > max_cache_items:
                image_cache.popitem(last=False)
        return tensor

    img_i = _load_single(record.img_i)
    img_j = _load_single(record.img_j)
    images = torch.stack([img_i, img_j], dim=0)
    return images.to(dtype=dtype).cpu().numpy()


def _process_pairwise(
    split_meta: List[dict],
    args,
    subset: str,
    rng: random.Random,
    compressor,
):
    if not split_meta:
        print(f"[SKIP] No {subset} splits found.")
        return
    out_dir = _ensure_dir(Path(args.output_dir) / args.dataset_type / "pairwise" / subset)

    all_pairs: List[PairRecord] = []
    for idx, meta in enumerate(split_meta):
        split_pairs = _load_pairs_for_split(meta, args.max_pairs_per_split, rng)
        all_pairs.extend(split_pairs)
        print(
            f"[LOAD] {subset} split {idx+1}/{len(split_meta)} | scene={meta['scene_version']} "
            f"split_id={meta['split_id']} pairs={len(split_pairs)} total={len(all_pairs)}"
        )

    shard: List[Tuple[np.ndarray, float, float, str, str]] = []
    shard_idx = 0
    dtype = _select_dtype(args.dtype)
    image_cache = OrderedDict() if args.image_cache_size > 0 else None
    for rec in all_pairs:
        tensor = _load_pair_tensor(
            rec,
            args.preprocess_mode,
            args.square_size,
            dtype,
            image_cache=image_cache,
            max_cache_items=args.image_cache_size,
        )
        shard.append((tensor, rec.label, rec.strength, rec.scene_version, rec.split_id))
        if len(shard) >= args.shard_size_pairs:
            _write_pair_shard(shard, out_dir, shard_idx, compressor)
            shard_idx += 1
            shard = []

    if shard:
        _write_pair_shard(shard, out_dir, shard_idx, compressor)


def _process_multiview(
    split_meta: List[dict],
    args,
    subset: str,
    rng: random.Random,
    compressor,
):
    if not split_meta:
        print(f"[SKIP] No {subset} splits found.")
        return
    out_dir = _ensure_dir(Path(args.output_dir) / args.dataset_type / "multiview" / subset)

    shard_scenes: List[dict] = []
    shard_idx = 0
    dtype = _select_dtype(args.dtype)

    for idx, meta in enumerate(split_meta):
        pairs, img_files, _ = _load_scene_pairs(meta["saved_obs"])
        if not img_files:
            print(f"[WARN] No images under {meta['saved_obs']}, skipping.")
            continue
        if not pairs:
            print(f"[WARN] No labeled pairs under {meta['saved_obs']}, skipping.")
            continue
        if args.max_pairs_per_scene > 0 and len(pairs) > args.max_pairs_per_scene:
            pairs = rng.sample(pairs, k=args.max_pairs_per_scene)

        img_paths = [os.path.join(meta["saved_obs"], f) for f in img_files]
        if args.preprocess_mode == "square":
            images, _ = load_and_preprocess_images_square(img_paths, target_size=args.square_size)
        else:
            images = load_and_preprocess_images(img_paths, mode=args.preprocess_mode)
        images = images.to(dtype=dtype).cpu().numpy()

        pair_idx = np.array([[p[0], p[1]] for p in pairs], dtype=np.int32)
        labels = np.array([p[2] for p in pairs], dtype=np.float32)
        strengths = np.array([p[3] for p in pairs], dtype=np.float32)

        shard_scenes.append(
            dict(
                images=images,
                pairs=pair_idx,
                labels=labels,
                strengths=strengths,
                scene_version=meta["scene_version"],
                split_id=meta["split_id"],
            )
        )
        print(
            f"[LOAD] {subset} scene {idx+1}/{len(split_meta)} | scene={meta['scene_version']} "
            f"split_id={meta['split_id']} images={len(img_files)} pairs={len(pairs)}"
        )

        if len(shard_scenes) >= args.shard_size_scenes:
            _write_scene_shard(shard_scenes, out_dir, shard_idx, compressor)
            shard_idx += 1
            shard_scenes = []

    if shard_scenes:
        _write_scene_shard(shard_scenes, out_dir, shard_idx, compressor)


def _build_splits(args) -> Tuple[List[dict], List[dict]]:
    meta = _discover_scene_splits(args.dataset_type)
    default_ratio = 0.8 if args.dataset_type == "gibson" else 0.9
    train_meta, val_meta, _ = _split_meta(
        meta,
        seed=args.seed,
        train_ratio=args.train_ratio if args.train_ratio is not None else default_ratio,
        split_mode=args.split_mode,
        split_index_path=args.split_index_path or None,
    )
    return train_meta, val_meta


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute preprocessed tensors into Zarr shards.")
    parser.add_argument("--mode", choices=["pairwise", "multiview"], required=True)
    parser.add_argument("--dataset_type", choices=["gibson", "hm3d"], default="gibson")
    parser.add_argument("--split_mode", choices=["scene_disjoint", "version_disjoint", "graph"], default="scene_disjoint")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_index_path", type=str, default="")
    parser.add_argument("--square_size", type=int, default=518)
    parser.add_argument("--preprocess_mode", choices=["square", "crop", "pad"], default="square")
    parser.add_argument("--max_pairs_per_split", type=int, default=-1, help="Pairwise: cap pairs per split (<=0 disables).")
    parser.add_argument("--max_pairs_per_scene", type=int, default=-1, help="Multiview: cap pairs per scene (<=0 disables).")
    parser.add_argument("--shard_size_pairs", type=int, default=500, help="Samples per pairwise shard.")
    parser.add_argument("--shard_size_scenes", type=int, default=20, help="Scenes per multiview shard.")
    parser.add_argument("--output_dir", type=str, default="runs/precomputed")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compress_level", type=int, default=5, help="Zstd level for Blosc.")
    parser.add_argument("--image_cache_size",type=int,default=1024, help="LRU cache size for preprocessed images during pairwise mode (set 0 to disable).")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    compressor = zarr.Blosc(cname="zstd", clevel=args.compress_level, shuffle=zarr.Blosc.SHUFFLE)
    train_meta, val_meta = _build_splits(args)

    print(
        f"[INFO] Mode={args.mode} dataset={args.dataset_type} dtype={args.dtype} "
        f"output={Path(args.output_dir) / args.dataset_type}"
    )

    if args.mode == "pairwise":
        _process_pairwise(train_meta, args, subset="train", rng=rng, compressor=compressor)
        _process_pairwise(val_meta, args, subset="val", rng=rng, compressor=compressor)
    else:
        _process_multiview(train_meta, args, subset="train", rng=rng, compressor=compressor)
        _process_multiview(val_meta, args, subset="val", rng=rng, compressor=compressor)

    print("[DONE] Precomputation complete.")


if __name__ == "__main__":
    main()
