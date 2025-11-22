#!/usr/bin/env python
"""
Datasets and loaders for precomputed Zarr shards produced by store_features.py.

Pairwise shards:
    images: (N,2,3,H,W), labels, strengths, scene_version, split_id

Multiview shards:
    One group per scene with images (N,3,H,W), pairs (P,2), labels, strengths, attrs.
"""
from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import zarr

# Ensure repository root (and bundled vggt module) is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "vggt") not in sys.path:
    sys.path.append(str(REPO_ROOT / "vggt"))

from vggt_trainer.data import collate_scene  # noqa: E402


def _list_shards(base_dir: Path, pattern: str) -> List[Path]:
    root = base_dir / pattern
    if not root.is_dir():
        return []
    shards = sorted(root.glob("*.zarr"))
    return shards


class PrecomputedPairDataset(Dataset):
    """Indexable view over pairwise shards."""

    def __init__(self, shard_paths: Sequence[Path]):
        if not shard_paths:
            raise RuntimeError("No pairwise shards found.")
        self.shard_paths = list(shard_paths)
        self.shard_lengths: List[int] = []
        self.prefix: List[int] = [0]
        for p in self.shard_paths:
            root = zarr.open(str(p), mode="r")
            length = int(len(root["labels"]))
            self.shard_lengths.append(length)
            self.prefix.append(self.prefix[-1] + length)
        self.total = self.prefix[-1]
        self._store_cache = {}

    def __len__(self) -> int:
        return self.total

    def _get_store(self, path: Path):
        store = self._store_cache.get(path)
        if store is None:
            store = zarr.open(str(path), mode="r")
            self._store_cache[path] = store
        return store

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.prefix, index) - 1
        local_idx = index - self.prefix[shard_idx]
        store = self._get_store(self.shard_paths[shard_idx])
        images = torch.from_numpy(store["images"][local_idx])
        label = torch.tensor(store["labels"][local_idx], dtype=torch.float32)
        strength = torch.tensor(store["strengths"][local_idx], dtype=torch.float32)
        scene_version = str(store["scene_version"][local_idx])
        split_id = str(store["split_id"][local_idx])
        return {
            "images": images,
            "label": label,
            "strength": strength,
            "scene_version": scene_version,
            "split_id": split_id,
        }


class PrecomputedSceneDataset(Dataset):
    """Indexable view over multiview shards (one group per scene)."""

    def __init__(self, shard_paths: Sequence[Path]):
        if not shard_paths:
            raise RuntimeError("No multiview shards found.")
        self.shard_paths = list(shard_paths)
        self.scene_index: List[Tuple[Path, str]] = []
        for p in self.shard_paths:
            root = zarr.open(str(p), mode="r")
            for key in root.group_keys():
                self.scene_index.append((p, key))
        if not self.scene_index:
            raise RuntimeError("No scenes discovered inside multiview shards.")
        self._store_cache = {}

    def __len__(self) -> int:
        return len(self.scene_index)

    def _get_group(self, path: Path, key: str):
        store = self._store_cache.get(path)
        if store is None:
            store = zarr.open(str(path), mode="r")
            self._store_cache[path] = store
        return store[key]

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self.scene_index):
            raise IndexError(index)
        path, key = self.scene_index[index]
        grp = self._get_group(path, key)
        images = torch.from_numpy(grp["images"][...])
        pairs = torch.from_numpy(grp["pairs"][...]).long()
        labels = torch.from_numpy(grp["labels"][...]).float()
        strengths = torch.from_numpy(grp["strengths"][...]).float()
        return {
            "images": images,
            "pairs": pairs,
            "labels": labels,
            "strengths": strengths,
            "scene_version": grp.attrs.get("scene_version", ""),
            "split_id": grp.attrs.get("split_id", ""),
        }


def build_precomputed_pair_loader(
    base_dir: str,
    subset: str,
    batch_size: int,
    num_workers: int = 0,
    device: Optional[str] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, dict]:
    shard_paths = _list_shards(Path(base_dir), f"pairwise/{subset}")
    dataset = PrecomputedPairDataset(shard_paths)
    use_cuda_pinning = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda_pinning,
        persistent_workers=persistent_workers if persistent_workers is not None else num_workers > 0,
    )
    if prefetch_factor and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    meta = {
        "samples": len(dataset),
        "shards": len(shard_paths),
        "subset": subset,
    }
    return loader, meta


def build_precomputed_multiview_loader(
    base_dir: str,
    subset: str,
    batch_size: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, dict]:
    shard_paths = _list_shards(Path(base_dir), f"multiview/{subset}")
    dataset = PrecomputedSceneDataset(shard_paths)
    use_cuda_pinning = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda_pinning,
        persistent_workers=persistent_workers if persistent_workers is not None else num_workers > 0,
        collate_fn=collate_scene,
    )
    if prefetch_factor and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    meta = {
        "scenes": len(dataset),
        "shards": len(shard_paths),
        "subset": subset,
    }
    return loader, meta


__all__ = [
    "PrecomputedPairDataset",
    "PrecomputedSceneDataset",
    "build_precomputed_pair_loader",
    "build_precomputed_multiview_loader",
]
