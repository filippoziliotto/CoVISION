#!/usr/bin/env python
import os
import glob
from typing import List, Dict, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle
import argparse


PRED_ROOT = "data/predictions_feat"
GIBSON_BASE_PATTERN = "data/vast/cc7287/gvgg-{i}"  # i in [1..5]


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def _discover_gibson_scene_splits() -> List[Dict]:
    """
    Walk gvgg-1..5 and find all (scene_version, split_id, root_path).

    Returns list of dicts:
      {
        "scene_version": "Adrian-3",
        "split_id": "0",
        "root": "/.../gvgg-3/temp/More_vis/Adrian-3/0/saved_obs"
      }
    """
    splits = []

    for i in range(1, 6):
        base_root = GIBSON_BASE_PATTERN.format(i=i)
        # Most common layout: base_root/temp/More_vis
        candidates = [
            os.path.join(base_root, "temp", "More_vis"),
            os.path.join(base_root, "More_vis"),
        ]
        candidates = [c for c in candidates if os.path.isdir(c)]
        if not candidates:
            continue

        for more_vis_root in candidates:
            for scene_name in sorted(os.listdir(more_vis_root)):
                scene_dir = os.path.join(more_vis_root, scene_name)
                if not os.path.isdir(scene_dir):
                    continue

                # scene_name should already include version suffix (e.g. Adrian-3)
                for split_name in sorted(os.listdir(scene_dir)):
                    split_dir = os.path.join(scene_dir, split_name)
                    if not os.path.isdir(split_dir):
                        continue

                    saved_obs = os.path.join(split_dir, "saved_obs")
                    if not os.path.isdir(saved_obs):
                        continue

                    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                    if not os.path.isfile(gt_csv):
                        continue

                    splits.append(
                        dict(
                            scene_version=scene_name,
                            split_id=split_name,
                            saved_obs=saved_obs,
                        )
                    )

    return splits


def _load_embeddings(scene_version: str, split_id: str) -> np.ndarray:
    """
    Load embeddings from data/predictions_feat/{scene_version}/split_{split}/embs/all_embeds.npy
    If only last_embeds.npy exists, treat it as a single-layer embedding (1, N, E).
    """
    emb_dir = os.path.join(
        PRED_ROOT,
        scene_version,
        f"split_{split_id}",
        "embs",
    )
    all_path = os.path.join(emb_dir, "all_embeds.npy")
    last_path = os.path.join(emb_dir, "last_embeds.npy")

    if os.path.isfile(all_path):
        emb = np.load(all_path)
        # Expect (L, N, E)
        if emb.ndim != 3:
            raise ValueError(f"all_embeds.npy must be (L, N, E), got {emb.shape}")
    elif os.path.isfile(last_path):
        emb_last = np.load(last_path)
        if emb_last.ndim != 2:
            raise ValueError(f"last_embeds.npy must be (N, E), got {emb_last.shape}")
        # Expand to a single layer: (1, N, E)
        emb = emb_last[None, ...]
    else:
        raise FileNotFoundError(f"Missing embeddings: {all_path} or {last_path}")

    return emb.astype(np.float32)

def _build_adjacency_from_gt(gt_csv_path: str, saved_obs_dir: str, N: int) -> np.ndarray:
    """
    Build NxN adjacency matrix from GroundTruth.csv.

    GroundTruth.csv:
        image_1,image_2,label
        ./temp/More_vis/Andover-1/0/saved_obs/best_color_0.png,...
    We map basenames best_color_k.png -> index k based on sorted filenames.
    """
    # We assume embeddings were computed using sorted PNGs:
    img_files = sorted(f for f in os.listdir(saved_obs_dir) if f.endswith(".png"))
    if len(img_files) != N:
        # Not fatal but worth warning
        print(
            f"[WARN] N={N} embeddings but {len(img_files)} PNGs in {saved_obs_dir}. "
            "Indexing may be inconsistent."
        )
    name_to_idx = {f: idx for idx, f in enumerate(img_files)}

    df = pd.read_csv(gt_csv_path)
    A = np.zeros((N, N), dtype=np.float32)

    for _, row in df.iterrows():
        p1 = os.path.basename(row["image_1"])
        p2 = os.path.basename(row["image_2"])
        lbl = int(row["label"])

        if p1 not in name_to_idx or p2 not in name_to_idx:
            continue
        i = name_to_idx[p1]
        j = name_to_idx[p2]
        A[i, j] = lbl
        A[j, i] = lbl

    return A


# -------------------------------------------------------
# Dataset: edge-level pairs from (embeddings, adjacency)
# -------------------------------------------------------
class EdgePairDataset(Dataset):
    """
    Each item is (feat_i, feat_j, label) from one of the graphs.
    Now supports negative subsampling via max_neg_ratio and hard negative mining using rel_mat.
    """

    def __init__(
        self,
        graphs: List[Dict],
        max_neg_ratio: float = 1.0,
        hard_neg_ratio: float = 0.5,
        hard_neg_rel_thr: float = 0.3,
        layer_mode: str = "1st_last",
    ):
        """
        graphs: list of dicts with keys:
          - "emb": (N, E) float32
          - "adj": (N, N) float32 in {0,1}
          - "rel": (N, N) float32 continuous or None
          - "scene_version": str
          - "split_id": str
        max_neg_ratio: float, maximum number of negatives to keep per positive (ratio)
        hard_neg_ratio: float in [0,1], fraction of selected negatives that should be "hard"
        hard_neg_rel_thr: minimum rel_mat[i,j] value for a negative to be considered hard
        """
        self.layer_mode = layer_mode
        self.pairs = []
        total_pos = 0
        total_neg = 0

        for g in graphs:
            emb = g["emb"]
            A = g["adj"]
            rel = g.get("rel", None)

            # emb can be (N, E) or (L, N, E). Normalize to (L, N, E).
            if emb.ndim == 2:
                emb = emb[None, ...]  # (1, N, E)
            elif emb.ndim != 3:
                raise ValueError(f"Expected emb to have shape (N, E) or (L, N, E), got {emb.shape}")
            L, N, E = emb.shape

            # Decide which layers to use
            mode = self.layer_mode
            if mode == "all":
                layer_indices = None  # use all layers, keep as (L, E)
            else:
                mode_to_offset = {
                    "1st_last": -1,
                    "2nd_last": -2,
                    "3rd_last": -3,
                    "4th_last": -4,
                }
                if mode not in mode_to_offset:
                    raise ValueError(
                        f"Unknown layer_mode '{mode}'. "
                        f"Expected one of {list(mode_to_offset.keys()) + ['all']}."
                    )
                offset = mode_to_offset[mode]
                # Convert negative offset to concrete index in [0, L-1]
                idx = (L + offset) if offset < 0 else offset
                idx = max(0, min(L - 1, idx))
                layer_indices = idx  # single int

            pos_pairs = []
            neg_pairs_hard = []
            neg_pairs_easy = []

            for i in range(N):
                for j in range(i + 1, N):
                    lbl = A[i, j]

                    # Continuous edge strength from rel_mat if available, else fall back to binary label
                    if rel is not None and rel.shape == A.shape:
                        strength_ij = float(rel[i, j])
                    else:
                        strength_ij = float(lbl)

                    # Select node features depending on layer_mode
                    if layer_indices is None:
                        # Use all layers: feat_i, feat_j are (L, E)
                        feat_i = emb[:, i, :].astype(np.float32)
                        feat_j = emb[:, j, :].astype(np.float32)
                    else:
                        # Single layer: feat_i, feat_j are (E,)
                        feat_i = emb[layer_indices, i, :].astype(np.float32)
                        feat_j = emb[layer_indices, j, :].astype(np.float32)

                    triple = (
                        feat_i,           # node i features
                        feat_j,           # node j features
                        float(lbl),       # binary GT
                        strength_ij,      # continuous strength
                    )

                    if lbl == 1:
                        pos_pairs.append(triple)
                    else:
                        # Negative edge: decide if it is hard or easy
                        if (
                            rel is not None
                            and rel.shape == A.shape
                            and rel[i, j] >= hard_neg_rel_thr
                        ):
                            neg_pairs_hard.append(triple)
                        else:
                            neg_pairs_easy.append(triple)

            num_pos = len(pos_pairs)
            if num_pos == 0:
                print(
                    f"[WARN] Graph {g.get('scene_version', '?')} split {g.get('split_id', '?')} has no positive edges, skipping."
                )
                continue

            # Always keep all positives
            self.pairs.extend(pos_pairs)
            total_pos += num_pos

            # If no negatives or disabled, skip negative sampling
            num_neg_total = len(neg_pairs_hard) + len(neg_pairs_easy)
            if max_neg_ratio <= 0 or num_neg_total == 0:
                continue

            # Total negatives we want to keep from this graph
            k_total_neg = int(num_pos * max_neg_ratio)
            k_total_neg = min(k_total_neg, num_neg_total)
            if k_total_neg <= 0:
                continue

            # How many of those should be hard
            k_hard = min(len(neg_pairs_hard), int(k_total_neg * hard_neg_ratio))
            k_easy = k_total_neg - k_hard

            sampled_neg = []
            if k_hard > 0:
                sampled_neg.extend(random.sample(neg_pairs_hard, k_hard))
            if k_easy > 0 and len(neg_pairs_easy) > 0:
                k_easy = min(k_easy, len(neg_pairs_easy))
                sampled_neg.extend(random.sample(neg_pairs_easy, k_easy))

            self.pairs.extend(sampled_neg)
            total_neg += len(sampled_neg)

        print(
            f"[INFO] EdgePairDataset: total pairs={len(self.pairs)} "
            f"(pos={total_pos}, neg={total_neg}, neg/pos={total_neg / max(1, total_pos):.2f})"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        feat_i, feat_j, lbl, strength = self.pairs[idx]
        return (
            torch.from_numpy(feat_i),
            torch.from_numpy(feat_j),
            torch.tensor(lbl, dtype=torch.float32),
            torch.tensor(strength, dtype=torch.float32),
        )


# -------------------------------------------------------
# Main builder: from gvgg-1..5 + predictions_feat
# -------------------------------------------------------
def build_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 0,
    max_neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    hard_neg_rel_thr: float = 0.3,
    layer_mode: str = "1st_last",
    split_mode: str = "scene_disjoint",
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict]:
    """
    Discover all (scene_version, split) graphs across gvgg-1..5,
    load embeddings + GT adjacency, split into train/val, and build edge-level dataloaders.

    Args:
        batch_size: int, DataLoader batch size.
        num_workers: int, DataLoader num_workers.
        train_ratio: float, fraction of data for training.
        seed: int, random seed.
        max_neg_ratio: float, max negative:positive ratio.
        hard_neg_ratio: float, fraction of negatives that should be "hard".
        hard_neg_rel_thr: float, minimum rel_mat value for a negative to be "hard".
        layer_mode: str, which VGGT layer(s) to use.
        split_mode: str, split strategy. One of:
            - "scene_disjoint": disjoint base scenes between train and val (all versions of a scene go to the same split).
            - "version_disjoint": disjoint scene_version between train and val (e.g., Adrian-1 in train, Adrian-2 in val); all splits of a scene_version go to the same side. Base scenes may appear in both splits but with different versions.
            - "graph": random split over graphs (scene_version + split_id).
    Returns:
        train_loader, val_loader, train_ds, val_ds, meta
    """
    # 1) Discover all splits
    split_meta = _discover_gibson_scene_splits()
    if not split_meta:
        raise RuntimeError("No Gibson splits found under gvgg-1..5")

    # 2) Build graph objects with emb + adj + rel (if available)
    graphs = []
    for s in split_meta:
        scene_version = s["scene_version"]
        split_id = s["split_id"]
        saved_obs = s["saved_obs"]

        # Embeddings
        try:
            emb = _load_embeddings(scene_version, split_id)  # (N, E)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        # GT adjacency
        gt_csv_path = os.path.join(saved_obs, "GroundTruth.csv")
        if not os.path.isfile(gt_csv_path):
            print(f"[WARN] Missing GroundTruth.csv in {saved_obs}, skipping")
            continue

        if emb.ndim == 3:
            # emb: (L, N, E)
            N = emb.shape[1]
        else:
            # emb: (N, E)
            N = emb.shape[0]

        A = _build_adjacency_from_gt(gt_csv_path, saved_obs, N)

        # Continuous edge strengths from rel_mat.npy (if available)
        rel_mat_path = os.path.join(saved_obs, "rel_mat.npy")
        if os.path.isfile(rel_mat_path):
            try:
                rel = np.load(rel_mat_path)
            except Exception as e:
                print(f"[WARN] Failed to load rel_mat.npy in {saved_obs}: {e}")
                rel = None
        else:
            rel = None

        graphs.append(
            dict(
                scene_version=scene_version,
                split_id=split_id,
                emb=emb,
                adj=A,
                rel=rel,
            )
        )

    if not graphs:
        raise RuntimeError("No valid graphs (embeddings + GT) found.")

    # Compute base_scene for each graph
    for g in graphs:
        g["base_scene"] = g["scene_version"].split("-")[0]

    # Compute base_scenes once for all split modes
    base_scenes = sorted({g["base_scene"] for g in graphs})

    # 3) Split strategy
    if split_mode == "scene_disjoint":
        # Scene-wise train/val split (by base_scene)
        scenes_shuffled = sk_shuffle(base_scenes, random_state=seed)
        n_train = int(len(base_scenes) * train_ratio)
        train_scenes = set(scenes_shuffled[:n_train])
        val_scenes = set(scenes_shuffled[n_train:])
        train_graphs = [g for g in graphs if g["base_scene"] in train_scenes]
        val_graphs = [g for g in graphs if g["base_scene"] in val_scenes]
    elif split_mode == "version_disjoint":
        # Disjoint scene_version between train and val (all splits of a scene_version go to the same side)
        scene_versions = sorted({g["scene_version"] for g in graphs})
        scene_versions_shuffled = sk_shuffle(scene_versions, random_state=seed)
        n_train = int(len(scene_versions) * train_ratio)
        train_versions = set(scene_versions_shuffled[:n_train])
        val_versions = set(scene_versions_shuffled[n_train:])
        train_graphs = [g for g in graphs if g["scene_version"] in train_versions]
        val_graphs = [g for g in graphs if g["scene_version"] in val_versions]
        train_scenes = {g["base_scene"] for g in train_graphs}
        val_scenes = {g["base_scene"] for g in val_graphs}
    elif split_mode == "graph":
        # Graph-level random split (scene_version/split_id)
        all_idx = list(range(len(graphs)))
        all_idx = sk_shuffle(all_idx, random_state=seed)
        n_train_graphs = int(len(all_idx) * train_ratio)
        train_idx = all_idx[:n_train_graphs]
        val_idx = all_idx[n_train_graphs:]
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        # For meta
        train_scenes = {g["base_scene"] for g in train_graphs}
        val_scenes = {g["base_scene"] for g in val_graphs}
    else:
        raise ValueError(
            f"Invalid split_mode '{split_mode}'. Valid values are 'scene_disjoint', 'version_disjoint', and 'graph'."
        )

    # 4) Build datasets
    train_ds = EdgePairDataset(
        train_graphs,
        max_neg_ratio=max_neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_rel_thr=hard_neg_rel_thr,
        layer_mode=layer_mode,
    )
    val_ds = EdgePairDataset(
        val_graphs,
        max_neg_ratio=max_neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_rel_thr=hard_neg_rel_thr,
        layer_mode=layer_mode,
    )

    # 5) Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Meta info for logging
    meta = dict(
        num_graphs=len(graphs),
        num_train_graphs=len(train_graphs),
        num_val_graphs=len(val_graphs),
        num_scenes=len(base_scenes),
        num_train_scenes=len(train_scenes),
        num_val_scenes=len(val_scenes),
        emb_dim=graphs[0]["emb"].shape[-1],
        split_mode=split_mode,
    )

    print(
        f"[INFO] Found {meta['num_graphs']} valid (scene_version, split) graphs. "
        f"Scenes: total={meta['num_scenes']}, train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )
    print(
        f"[INFO] Splits: train_graphs={meta['num_train_graphs']}, "
        f"val_graphs={meta['num_val_graphs']}"
    )
    print(f"[INFO] Embedding dim: {meta['emb_dim']}")

    return train_loader, val_loader, train_ds, val_ds, meta

# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_neg_ratio", type=float, default=1.0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--hard_neg_rel_thr", type=float, default=0.3)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="version_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
        help="Split mode: 'scene_disjoint', 'version_disjoint', or 'graph'."
    )
    args = parser.parse_args()

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_neg_ratio=args.max_neg_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode="1st_last",
        split_mode=args.split_mode,
    )

    print(f"[CHECK] Train dataset size: {len(train_ds)}")
    print(f"[CHECK] Val dataset size:   {len(val_ds)}")
    
    first_sample = next(iter(train_loader))[0]
    print(f"[CHECK] First batch feature shape: {first_sample.shape}")