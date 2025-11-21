#!/usr/bin/env python
"""
Utilities for building dataloaders that read RGB image pairs directly from saved_obs folders.

This mirrors the discovery logic used when saving embeddings, but returns tensors that are
ready to be consumed by VGGT without going through the intermediate NPZ stage.
"""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vggt.vggt.utils.load_fn import (
    load_and_preprocess_images,
    load_and_preprocess_images_square,
)

# Dataset layouts taken from dataset/load_dataset.py to keep behavior consistent.
GIBSON_BASE_PATTERN = "data/vast/cc7287/gvgg-{i}"
HM3D_BASE_PATTERN = (
    "data/scratch/cc7287/mvdust3r_projects/HM3D/dust3r_vpr_mask/data/hvgg/part{i}"
)
HM3D_PARTS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "l"]

# Collate helper for multiview scenes (one sample per batch).
def collate_scene(batch):
    return batch[0]


@dataclass
class PairRecord:
    scene_version: str
    split_id: str
    saved_obs: str
    img_i: str
    img_j: str
    label: float
    strength: float
    depth_path: Optional[str] = None
    depth_idx_i: Optional[int] = None
    depth_idx_j: Optional[int] = None


def _default_train_ratio(dataset_type: str, override: Optional[float]) -> float:
    if override is not None:
        return override
    return 0.8 if dataset_type == "gibson" else 0.9


def _discover_scene_splits(dataset_type: str) -> List[Dict[str, str]]:
    """Mimic dataset.load_dataset._discover_scene_splits without importing private helpers."""
    splits: List[Dict[str, str]] = []

    if dataset_type == "hm3d":
        for part in HM3D_PARTS:
            base_root = HM3D_BASE_PATTERN.format(i=part)
            more_vis_root = os.path.join(base_root, "temp", "More_vis")
            if not os.path.isdir(more_vis_root):
                continue
            for scene_name in sorted(os.listdir(more_vis_root)):
                scene_dir = os.path.join(more_vis_root, scene_name)
                if not os.path.isdir(scene_dir) or not scene_name.endswith(".basis"):
                    continue
                scene_id = scene_name.split(".basis")[0]
                for split_name in sorted(os.listdir(scene_dir)):
                    split_dir = os.path.join(scene_dir, split_name)
                    if not (os.path.isdir(split_dir) and split_name.isdigit()):
                        continue
                    saved_obs = os.path.join(split_dir, "saved_obs")
                    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                    if os.path.isdir(saved_obs) and os.path.isfile(gt_csv):
                        splits.append(
                            dict(
                                scene_version=scene_id,
                                split_id=split_name,
                                saved_obs=saved_obs,
                            )
                        )
    else:
        for i in range(1, 6):
            base_root = GIBSON_BASE_PATTERN.format(i=i)
            candidates = [
                os.path.join(base_root, "temp", "More_vis"),
                os.path.join(base_root, "More_vis"),
            ]
            for more_vis_root in candidates:
                if not os.path.isdir(more_vis_root):
                    continue
                for scene_name in sorted(os.listdir(more_vis_root)):
                    scene_dir = os.path.join(more_vis_root, scene_name)
                    if not os.path.isdir(scene_dir):
                        continue
                    for split_name in sorted(os.listdir(scene_dir)):
                        split_dir = os.path.join(scene_dir, split_name)
                        if not (os.path.isdir(split_dir) and split_name.isdigit()):
                            continue
                        saved_obs = os.path.join(split_dir, "saved_obs")
                        gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                        if os.path.isdir(saved_obs) and os.path.isfile(gt_csv):
                            splits.append(
                                dict(
                                    scene_version=scene_name,
                                    split_id=split_name,
                                    saved_obs=saved_obs,
                                )
                            )
    return splits


def _split_meta(
    meta: List[Dict[str, str]],
    seed: int,
    train_ratio: float,
    split_mode: str,
    split_index_path: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Sequence[str]]]:
    """Split meta entries deterministically according to split_mode."""
    if not meta:
        raise RuntimeError("No saved_obs splits discovered.")

    rng = random.Random(seed)
    for item in meta:
        if "base_scene" not in item:
            item["base_scene"] = item["scene_version"].split("-")[0]

    if split_index_path and os.path.isfile(split_index_path):
        with open(split_index_path, "r") as f:
            js = json.load(f)
        if "train" not in js or "val" not in js:
            raise ValueError(f"Split index at {split_index_path} missing 'train'/'val' keys.")
        train_keys = {(d["scene_version"], str(d["split_id"])) for d in js["train"]}
        val_keys = {(d["scene_version"], str(d["split_id"])) for d in js["val"]}
        train_meta = [m for m in meta if (m["scene_version"], m["split_id"]) in train_keys]
        val_meta = [m for m in meta if (m["scene_version"], m["split_id"]) in val_keys]
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}
        print(f"[DATA] Loaded split index from {split_index_path}")
    elif split_mode == "scene_disjoint":
        scenes = sorted({m["base_scene"] for m in meta})
        rng.shuffle(scenes)
        pivot = max(1, int(len(scenes) * train_ratio))
        train_scenes = set(scenes[:pivot])
        train_meta = [m for m in meta if m["base_scene"] in train_scenes]
        val_meta = [m for m in meta if m["base_scene"] not in train_scenes]
    elif split_mode == "version_disjoint":
        versions = sorted({m["scene_version"] for m in meta})
        rng.shuffle(versions)
        pivot = max(1, int(len(versions) * train_ratio))
        train_versions = set(versions[:pivot])
        train_meta = [m for m in meta if m["scene_version"] in train_versions]
        val_meta = [m for m in meta if m["scene_version"] not in train_versions]
        train_scenes = {m["base_scene"] for m in train_meta}
    elif split_mode == "graph":
        idxs = list(range(len(meta)))
        rng.shuffle(idxs)
        pivot = max(1, int(len(idxs) * train_ratio))
        train_idx = set(idxs[:pivot])
        train_meta = [meta[i] for i in idxs if i in train_idx]
        val_meta = [meta[i] for i in idxs if i not in train_idx]
        train_scenes = {m["base_scene"] for m in train_meta}
    else:
        raise ValueError(
            f"Invalid split_mode '{split_mode}'. "
            "Expected one of {'scene_disjoint', 'version_disjoint', 'graph'}."
        )

    val_scenes = {m["base_scene"] for m in val_meta}
    stats = dict(
        train_scenes=sorted(train_scenes),
        val_scenes=sorted(val_scenes),
    )
    return train_meta, val_meta, stats


def _load_rel_matrix(saved_obs: str) -> Optional[np.ndarray]:
    rel_path = os.path.join(saved_obs, "rel_mat.npy")
    if os.path.isfile(rel_path):
        try:
            rel = np.load(rel_path)
            return rel
        except Exception:
            pass
    return None


def _load_pairs_for_split(
    split_meta: Dict[str, str],
    max_pairs_per_split: int,
    rng: random.Random,
) -> List[PairRecord]:
    """Read GroundTruth.csv for a single saved_obs directory."""
    saved_obs = split_meta["saved_obs"]
    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
    if not os.path.isfile(gt_csv):
        return []

    rel_mat = _load_rel_matrix(saved_obs)
    depth_path = os.path.join(saved_obs, "saved_dep.npy")
    has_depth = os.path.isfile(depth_path)
    img_files = sorted(f for f in os.listdir(saved_obs) if f.endswith(".png"))
    name_to_idx = {f: idx for idx, f in enumerate(img_files)}

    pairs: List[PairRecord] = []
    with open(gt_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basename_i = os.path.basename(row["image_1"])
            basename_j = os.path.basename(row["image_2"])
            if basename_i not in name_to_idx or basename_j not in name_to_idx:
                continue
            path_i = os.path.join(saved_obs, basename_i)
            path_j = os.path.join(saved_obs, basename_j)
            if (not os.path.isfile(path_i)) or (not os.path.isfile(path_j)):
                continue

            label = float(row.get("label", 0))
            idx_i = name_to_idx[basename_i]
            idx_j = name_to_idx[basename_j]

            if rel_mat is not None and rel_mat.shape[0] > max(idx_i, idx_j):
                strength = float(rel_mat[idx_i, idx_j])
            else:
                strength = label

            pairs.append(
                PairRecord(
                    scene_version=split_meta["scene_version"],
                    split_id=split_meta["split_id"],
                    saved_obs=saved_obs,
                    img_i=path_i,
                    img_j=path_j,
                    label=label,
                    strength=strength,
                    depth_path=depth_path if has_depth else None,
                    depth_idx_i=idx_i if has_depth else None,
                    depth_idx_j=idx_j if has_depth else None,
                )
            )

    if max_pairs_per_split > 0 and len(pairs) > max_pairs_per_split:
        pairs = rng.sample(pairs, k=max_pairs_per_split)

    return pairs


def _load_pairs_for_meta(
    meta_list: List[Dict[str, str]],
    max_pairs_per_split: int,
    seed: int,
) -> List[PairRecord]:
    rng = random.Random(seed)
    all_pairs: List[PairRecord] = []
    for idx, split_meta in enumerate(meta_list):
        split_pairs = _load_pairs_for_split(split_meta, max_pairs_per_split, rng)
        all_pairs.extend(split_pairs)
        if idx % 10 == 0:
            print(
                f"[DATA] Loaded {len(split_pairs)} pairs from "
                f"{split_meta['scene_version']} split {split_meta['split_id']} "
                f"(running total={len(all_pairs)})"
            )
    return all_pairs


def _resize_pair_to_square(images: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad/crop a pair tensor (N=2,3,H,W) to (2,3,target_size,target_size)."""
    processed = []
    for img in images:
        c, h, w = img.shape
        if h > target_size:
            start = (h - target_size) // 2
            img = img[:, start : start + target_size, :]
            h = img.shape[1]
        if w > target_size:
            start = (w - target_size) // 2
            img = img[:, :, start : start + target_size]
            w = img.shape[2]

        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            img = F.pad(
                img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        processed.append(img)
    return torch.stack(processed, dim=0)


def load_pair_tensor(
    image_paths: Sequence[str],
    preprocess_mode: str,
    square_size: int,
) -> torch.Tensor:
    """Load and preprocess two RGB paths into a tensor shaped (2, 3, square_size, square_size)."""
    if preprocess_mode == "square":
        images, _ = load_and_preprocess_images_square(
            list(image_paths), target_size=square_size
        )
    else:
        images = load_and_preprocess_images(list(image_paths), mode=preprocess_mode)
        images = _resize_pair_to_square(images, square_size)
    return images


class PairImageDataset(Dataset):
    """Simple dataset wrapper around PairRecord entries."""

    def __init__(
        self,
        pairs: List[PairRecord],
        preprocess_mode: str = "square",
        square_size: int = 518,
        cache_images: bool = True,
        max_cache_items: int = 0,
    ):
        if not pairs:
            raise RuntimeError("PairImageDataset cannot be instantiated with zero samples.")
        self.pairs = pairs
        self.preprocess_mode = preprocess_mode
        self.square_size = square_size
        self.cache_images = cache_images
        self.max_cache_items = max_cache_items
        self._image_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict() if cache_images else None
        self._depth_cache = {}

    def _load_single_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image, caching the result."""
        if self.cache_images and path in self._image_cache:
            # Move to the end to keep LRU ordering.
            tensor = self._image_cache.pop(path)
            self._image_cache[path] = tensor
            return tensor

        if self.preprocess_mode == "square":
            imgs, _ = load_and_preprocess_images_square([path], target_size=self.square_size)
            tensor = imgs[0]
        else:
            imgs = load_and_preprocess_images([path], mode=self.preprocess_mode)
            imgs = _resize_pair_to_square(imgs, self.square_size)
            tensor = imgs[0]

        if self.cache_images:
            self._image_cache[path] = tensor
            # If max_cache_items > 0, evict the oldest entry when exceeding the limit.
            if self.max_cache_items > 0 and len(self._image_cache) > self.max_cache_items:
                self._image_cache.popitem(last=False)
        return tensor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.pairs[index]
        img_i = self._load_single_image(record.img_i)
        img_j = self._load_single_image(record.img_j)
        images = torch.stack([img_i, img_j], dim=0)
        depths = None
        if record.depth_path and os.path.isfile(record.depth_path):
            depth_arr = self._depth_cache.get(record.depth_path)
            if depth_arr is None:
                depth_arr = np.load(record.depth_path, allow_pickle=False)
                self._depth_cache[record.depth_path] = depth_arr
            if (
                record.depth_idx_i is not None
                and record.depth_idx_j is not None
                and depth_arr.ndim >= 3
                and max(record.depth_idx_i, record.depth_idx_j) < depth_arr.shape[0]
            ):
                depth_i = torch.from_numpy(depth_arr[record.depth_idx_i]).float()
                depth_j = torch.from_numpy(depth_arr[record.depth_idx_j]).float()
                depths = torch.stack([depth_i, depth_j], dim=0)
        return {
            "images": images,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "strength": torch.tensor(record.strength, dtype=torch.float32),
            "scene_version": record.scene_version,
            "split_id": record.split_id,
            "depths": depths,
        }


def build_image_pair_dataloaders(
    dataset_type: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: Optional[float],
    split_mode: str,
    split_index_path: Optional[str],
    preprocess_mode: str,
    square_size: int,
    max_pairs_per_split: int,
    device: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader], PairImageDataset, Optional[PairImageDataset], Dict]:
    """Entry point used by the training script."""
    train_ratio = _default_train_ratio(dataset_type, train_ratio)
    index_ratio = 0.8 if dataset_type == "gibson" else 0.9
    default_split_index = f"dataset/splits/pairview/pairview_{dataset_type}_{seed}_{index_ratio:.1f}.json"
    split_index_path = split_index_path or default_split_index
    if not os.path.isfile(split_index_path):
        split_index_path = None
    split_meta = _discover_scene_splits(dataset_type)
    train_meta, val_meta, stats = _split_meta(
        split_meta, seed, train_ratio, split_mode, split_index_path=split_index_path
    )

    train_pairs = _load_pairs_for_meta(train_meta, max_pairs_per_split, seed)
    # Keep validation untouched: do not subsample pairs for held-out metrics.
    val_pairs = _load_pairs_for_meta(val_meta, -1, seed + 1) if val_meta else []

    # Caching images in each worker can explode RAM; keep caching only when single-process loading.
    cache_images = num_workers == 0
    train_dataset = PairImageDataset(
        train_pairs,
        preprocess_mode=preprocess_mode,
        square_size=square_size,
        cache_images=cache_images,
    )
    val_dataset = (
        PairImageDataset(
            val_pairs,
            preprocess_mode=preprocess_mode,
            square_size=square_size,
            cache_images=cache_images,
        )
        if val_pairs
        else None
    )

    use_cuda_pinning = (
        device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    )
    pin_memory = use_cuda_pinning
    prefetch_factor = 2 if num_workers > 0 else None
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

    meta = dict(
        train_pairs=len(train_dataset),
        val_pairs=len(val_dataset) if val_dataset is not None else 0,
        train_scenes=len(stats["train_scenes"]),
        val_scenes=len(stats["val_scenes"]),
        train_ratio=train_ratio,
        split_mode=split_mode,
        split_index_path=split_index_path or "",
    )

    print(
        f"[DATA] train_pairs={meta['train_pairs']} "
        f"(scenes={meta['train_scenes']}), "
        f"val_pairs={meta['val_pairs']} "
        f"(scenes={meta['val_scenes']})"
    )

    return train_loader, val_loader, train_dataset, val_dataset, meta


__all__ = [
    "PairImageDataset",
    "PairRecord",
    "build_image_pair_dataloaders",
    "load_pair_tensor",
]

# -------------------------------------------------------------------------
# Multi-view scene dataset (images + pair labels)
# -------------------------------------------------------------------------
def _load_scene_pairs(saved_obs: str) -> Tuple[List[Tuple[int, int, float, float]], List[str], Optional[str]]:
    """Return pair list and ordered image files under a saved_obs directory."""
    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
    if not os.path.isfile(gt_csv):
        return [], [], None

    rel_mat = _load_rel_matrix(saved_obs)
    depth_path = os.path.join(saved_obs, "saved_dep.npy")
    has_depth = os.path.isfile(depth_path)

    img_files = sorted(f for f in os.listdir(saved_obs) if f.endswith(".png"))
    name_to_idx = {f: idx for idx, f in enumerate(img_files)}

    pairs: List[Tuple[int, int, float, float]] = []
    with open(gt_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basename_i = os.path.basename(row["image_1"])
            basename_j = os.path.basename(row["image_2"])
            if basename_i not in name_to_idx or basename_j not in name_to_idx:
                continue
            idx_i = name_to_idx[basename_i]
            idx_j = name_to_idx[basename_j]
            label = float(row.get("label", 0))
            if rel_mat is not None and rel_mat.shape[0] > max(idx_i, idx_j):
                strength = float(rel_mat[idx_i, idx_j])
            else:
                strength = label
            pairs.append((idx_i, idx_j, label, strength))

    return pairs, img_files, depth_path if has_depth else None


class MultiViewSceneDataset(Dataset):
    """Each sample is one saved_obs scene with all images and its labeled pairs."""

    def __init__(
        self,
        split_meta: List[Dict[str, str]],
        max_pairs_per_scene: int,
        preprocess_mode: str,
        square_size: int,
        seed: int,
    ):
        if not split_meta:
            raise RuntimeError("MultiViewSceneDataset cannot be created with empty split metadata.")
        self.split_meta = split_meta
        self.max_pairs_per_scene = max_pairs_per_scene
        self.preprocess_mode = preprocess_mode
        self.square_size = square_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.split_meta)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.split_meta[index]
        pairs, img_files, depth_path = _load_scene_pairs(meta["saved_obs"])
        if not img_files:
            raise RuntimeError(f"No PNG images found under {meta['saved_obs']}")

        # Cache preprocessed scene images to avoid repeated I/O across epochs.
        cache_key = meta["saved_obs"]
        if not hasattr(self, "_scene_image_cache"):
            self._scene_image_cache = {}
        if cache_key in self._scene_image_cache:
            images = self._scene_image_cache[cache_key]
        else:
            img_paths = [os.path.join(meta["saved_obs"], f) for f in img_files]
            images, _ = load_and_preprocess_images_square(img_paths, target_size=self.square_size)
            self._scene_image_cache[cache_key] = images

        if self.max_pairs_per_scene > 0 and len(pairs) > self.max_pairs_per_scene:
            pairs = self.rng.sample(pairs, k=self.max_pairs_per_scene)

        pair_idx = torch.tensor([p[:2] for p in pairs], dtype=torch.long)
        labels = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
        strengths = torch.tensor([p[3] for p in pairs], dtype=torch.float32)

        depth_tensor = None
        if depth_path and os.path.isfile(depth_path):
            depth_arr = np.load(depth_path, allow_pickle=False)
            if depth_arr.ndim >= 3 and depth_arr.shape[0] == len(img_files):
                depth_tensor = torch.from_numpy(depth_arr).float()

        return {
            "images": images,  # (N,3,H,W)
            "pairs": pair_idx,  # (P,2)
            "labels": labels,  # (P,)
            "strengths": strengths,  # (P,)
            "scene_version": meta["scene_version"],
            "split_id": meta["split_id"],
            "depths": depth_tensor,  # (N,H,W) or None
        }


def build_multiview_dataloaders(
    dataset_type: str,
    train_ratio: Optional[float],
    split_mode: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    preprocess_mode: str,
    square_size: int,
    max_pairs_per_scene: int,
    split_index_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Dict]:
    """Build train/val dataloaders for multiview training (one scene per batch)."""
    train_ratio = _default_train_ratio(dataset_type, train_ratio)
    index_ratio = 0.8 if dataset_type == "gibson" else 0.9
    default_split_index = f"dataset/splits/multiview/multiview_{seed}_{index_ratio:.1f}.json"
    split_index_path = split_index_path or default_split_index
    if not os.path.isfile(split_index_path):
        split_index_path = None
    split_meta = _discover_scene_splits(dataset_type)
    train_meta, val_meta, stats = _split_meta(
        split_meta, seed, train_ratio, split_mode, split_index_path=split_index_path
    )

    train_ds = MultiViewSceneDataset(
        train_meta,
        max_pairs_per_scene=max_pairs_per_scene,
        preprocess_mode=preprocess_mode,
        square_size=square_size,
        seed=seed,
    )
    val_ds = (
        MultiViewSceneDataset(
            val_meta,
            max_pairs_per_scene=max_pairs_per_scene,
            preprocess_mode=preprocess_mode,
            square_size=square_size,
            seed=seed + 1,
        )
        if val_meta
        else None
    )

    use_cuda_pinning = (
        device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    )
    pin_memory = use_cuda_pinning
    prefetch_factor = 2 if num_workers > 0 else None
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_scene,
    )
    if prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
        )

    meta = dict(
        train_scenes=len(train_ds),
        val_scenes=len(val_ds) if val_ds is not None else 0,
        train_ratio=train_ratio,
        split_mode=split_mode,
        max_pairs_per_scene=max_pairs_per_scene,
        split_index_path=split_index_path or "",
    )
    return train_loader, val_loader, meta
