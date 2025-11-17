#!/usr/bin/env python
import os
import glob
from typing import List, Dict, Tuple
import random
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle
import argparse

PRED_ROOT = "data/predictions_feat/gvgg"
GIBSON_BASE_PATTERN = "data/vast/cc7287/gvgg-{i}"  # i in [1..5]
HM3D_BASE_PATTERN = "data/scratch/cc7287/mvdust3r_projects/HM3D/dust3r_vpr_mask/data/hvgg/part{i}"  # i in [a,b,c,d,e,f,g,h,i,l]
HM3D_PARTS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "l"]
DEBUG = False

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def _discover_scene_splits(dataset_type: str) -> List[Dict]:
    """
    Discover all (scene_version, split_id, saved_obs) across Gibson or HM3D,
    depending on dataset_type.

    Returns list of dicts:
      {
        "scene_version": "Adrian-3" or "E1NrAhMoqvB",
        "split_id": "0",
        "saved_obs": "/.../saved_obs"
      }
    """
    splits = []

    if dataset_type == "hm3d":
        # HM3D: data/.../HM3D/dust3r_vpr_mask/data/hvgg/part{letter}/temp/More_vis/{scene_id}.basis/{split}/saved_obs
        for part in HM3D_PARTS:
            base_root = HM3D_BASE_PATTERN.format(i=part)
            more_vis_root = os.path.join(base_root, "temp", "More_vis")
            if not os.path.isdir(more_vis_root):
                continue

            for scene_name in sorted(os.listdir(more_vis_root)):
                scene_dir = os.path.join(more_vis_root, scene_name)
                if not os.path.isdir(scene_dir):
                    continue
                if not scene_name.endswith(".basis"):
                    continue

                scene_id = scene_name.split(".basis")[0]

                for split_name in sorted(os.listdir(scene_dir)):
                    split_dir = os.path.join(scene_dir, split_name)
                    if not os.path.isdir(split_dir):
                        continue
                    if not split_name.isdigit():
                        continue

                    saved_obs = os.path.join(split_dir, "saved_obs")
                    if not os.path.isdir(saved_obs):
                        continue

                    gt_csv = os.path.join(saved_obs, "GroundTruth.csv")
                    if not os.path.isfile(gt_csv):
                        continue

                    splits.append(
                        dict(
                            scene_version=scene_id,
                            split_id=split_name,
                            saved_obs=saved_obs,
                        )
                    )
    else:
        # Gibson: data/vast/cc7287/gvgg-{i}/(temp/)?More_vis/{SceneName}-{version}/{split}/saved_obs
        for i in range(1, 6):
            base_root = GIBSON_BASE_PATTERN.format(i=i)
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

                    for split_name in sorted(os.listdir(scene_dir)):
                        split_dir = os.path.join(scene_dir, split_name)
                        if not os.path.isdir(split_dir):
                            continue
                        if not split_name.isdigit():
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


def _load_embeddings(scene_version: str, split_id: str, emb_mode: str = "avg_max") -> np.ndarray:
    """
    Load embeddings for a given scene_version and split_id.
    Tries NPZ files for the specified emb_mode first, then falls back to old NPY files.
    Returns emb as (L, N, E) np.ndarray (float32).
    """
    emb_dir = os.path.join(
        PRED_ROOT,
        scene_version,
        f"split_{split_id}",
        "embs",
    )
    all_npz = os.path.join(emb_dir, f"all_embeds_{emb_mode}.npz")
    last_npz = os.path.join(emb_dir, f"last_embeds_{emb_mode}.npz")
    all_path = os.path.join(emb_dir, "all_embeds.npy")
    last_path = os.path.join(emb_dir, "last_embeds.npy")

    # 1. Try all_embeds_{emb_mode}.npz
    if os.path.isfile(all_npz):
        data = np.load(all_npz)
        if "all" in data.files:
            emb = data["all"]
        elif "arr_0" in data.files:
            emb = data["arr_0"]
        else:
            raise ValueError(f"{all_npz} does not contain an 'all' array (found keys: {data.files})")
        if emb.ndim != 3:
            raise ValueError(f"{all_npz}: expected shape (L, N, E), got {emb.shape}")
    # 2. Try last_embeds_{emb_mode}.npz
    elif os.path.isfile(last_npz):
        data = np.load(last_npz)
        if "last" in data.files:
            emb_last = data["last"]
        elif "arr_0" in data.files:
            emb_last = data["arr_0"]
        else:
            raise ValueError(f"{last_npz} does not contain a 'last' array (found keys: {data.files})")
        if emb_last.ndim != 2:
            raise ValueError(f"{last_npz}: expected shape (N, E), got {emb_last.shape}")
        emb = emb_last[None, ...]  # expand to (1, N, E)
    # 3. Try all_embeds.npy (old)
    elif os.path.isfile(all_path):
        emb = np.load(all_path)
        if emb.ndim != 3:
            raise ValueError(f"{all_path} must be (L, N, E), got {emb.shape}")
    # 4. Try last_embeds.npy (old)
    elif os.path.isfile(last_path):
        emb_last = np.load(last_path)
        if emb_last.ndim != 2:
            raise ValueError(f"{last_path} must be (N, E), got {emb_last.shape}")
        emb = emb_last[None, ...]
    else:
        raise FileNotFoundError(
            f"Missing embeddings: checked {all_npz}, {last_npz}, {all_path}, {last_path}"
        )
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
# Deterministic meta-level split helpers for scene splits
# -------------------------------------------------------
def _split_split_meta(
    split_meta: List[Dict],
    seed: int,
    train_ratio: float,
    split_mode: str,
) -> Tuple[List[Dict], List[Dict], List[str], set, set]:
    """
    Split the list returned by _discover_scene_splits into train/val meta lists.
    """
    if not split_meta:
        return [], [], [], set(), set()

    for m in split_meta:
        if "base_scene" not in m:
            # Gibson uses 'SceneName-version'; HM3D has no '-version'
            m["base_scene"] = m["scene_version"].split("-")[0]

    base_scenes = sorted({m["base_scene"] for m in split_meta})

    if split_mode == "scene_disjoint":
        scenes_shuffled = sk_shuffle(base_scenes, random_state=seed)
        n_train = int(len(base_scenes) * train_ratio)
        train_scenes = set(scenes_shuffled[:n_train])
        val_scenes = set(scenes_shuffled[n_train:])
        train_meta = [m for m in split_meta if m["base_scene"] in train_scenes]
        val_meta = [m for m in split_meta if m["base_scene"] in val_scenes]

    elif split_mode == "version_disjoint":
        scene_versions = sorted({m["scene_version"] for m in split_meta})
        scene_versions_shuffled = sk_shuffle(scene_versions, random_state=seed)
        n_train = int(len(scene_versions) * train_ratio)
        train_versions = set(scene_versions_shuffled[:n_train])
        val_versions = set(scene_versions_shuffled[n_train:])
        train_meta = [m for m in split_meta if m["scene_version"] in train_versions]
        val_meta = [m for m in split_meta if m["scene_version"] in val_versions]
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}

    elif split_mode == "graph":
        all_idx = list(range(len(split_meta)))
        all_idx = sk_shuffle(all_idx, random_state=seed)
        n_train_graphs = int(len(all_idx) * train_ratio)
        train_idx = set(all_idx[:n_train_graphs])
        val_idx = set(all_idx[n_train_graphs:])
        train_meta = [split_meta[i] for i in all_idx if i in train_idx]
        val_meta = [split_meta[i] for i in all_idx if i in val_idx]
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}
    else:
        raise ValueError(
            f"Invalid split_mode '{split_mode}'. "
            f"Valid values are 'scene_disjoint', 'version_disjoint', and 'graph'."
        )

    return train_meta, val_meta, base_scenes, train_scenes, val_scenes


def _load_split_index_raw(path: str) -> Dict:
    with open(path, "r") as f:
        js = json.load(f)
    if "train" not in js or "val" not in js:
        raise ValueError(f"Split index at {path} missing 'train'/'val' keys.")
    return js


def _write_split_index_raw(
    path: str,
    train_meta: List[Dict],
    val_meta: List[Dict],
    *,
    seed: int,
    train_ratio: float,
    split_mode: str,
    dataset_type: str,
    emb_mode: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "info": {
            "seed": seed,
            "train_ratio": train_ratio,
            "split_mode": split_mode,
            "dataset_type": dataset_type,
            "emb_mode": emb_mode,
            "source": "load_dataset.py",
        },
        "train": [{"scene_version": m["scene_version"], "split_id": m["split_id"]} for m in train_meta],
        "val": [{"scene_version": m["scene_version"], "split_id": m["split_id"]} for m in val_meta],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
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

            # Compute and log per-graph statistics
            num_pos = len(pos_pairs)
            num_neg_hard = len(neg_pairs_hard)
            num_neg_easy = len(neg_pairs_easy)
            num_neg_total = num_neg_hard + num_neg_easy
            scene_version = g.get("scene_version", "?")
            split_id = g.get("split_id", "?")
            if DEBUG:
                print(
                    f"[DEBUG] Graph {scene_version} split {split_id}: N={N}, pos={num_pos}, neg_hard={num_neg_hard}, neg_easy={num_neg_easy}, neg_total={num_neg_total}"
                )

            if num_pos == 0:
                print(
                    f"[WARN] Graph {scene_version} split {split_id} has no positive edges, skipping."
                )
                continue

            # Always keep all positives
            self.pairs.extend(pos_pairs)
            total_pos += num_pos

            # Negative subsampling
            if num_neg_total == 0:
                continue

            # If max_neg_ratio <= 0, keep all negatives (no subsampling)
            if max_neg_ratio <= 0:
                sampled_neg = neg_pairs_hard + neg_pairs_easy
            else:
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

            kept_neg = len(sampled_neg)
            if DEBUG:
                print(
                    f"[DEBUG] Kept negatives for {scene_version} split {split_id}: kept={kept_neg} / total={num_neg_total}"
                )

            self.pairs.extend(sampled_neg)
            total_neg += kept_neg

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
# -------------------------------------------------------
# Deterministic meta-level split and subset loading
# -------------------------------------------------------
def build_dataloaders(
    dataset_type: str = "gibson",
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 0,
    max_neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    hard_neg_rel_thr: float = 0.3,
    layer_mode: str = "1st_last",
    split_mode: str = "scene_disjoint",
    emb_mode: str = "avg_max",
    subset: str = "both",                 # new
    split_index_path: str = None,         # new
    persist_split_index: bool = False,    # new
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict]:
    """
    Discover all (scene_version, split) graphs, deterministically split at the meta level,
    and then load only the requested subset(s). This avoids loading heavy arrays for the
    unused split and guarantees reproducible splits with the same seed/ratio, optionally
    persisted to disk.
    """
    # 1) Set PRED_ROOT based on dataset_type
    global PRED_ROOT
    if dataset_type == "hm3d":
        PRED_ROOT = "data/predictions_feat/hvgg"
    else:
        PRED_ROOT = "data/predictions_feat/gvgg"

    random.seed(seed)
    np.random.seed(seed)

    # 2) Discover meta (scene_version, split_id, saved_obs) without loading arrays
    split_meta = _discover_scene_splits(dataset_type)
    if not split_meta:
        raise RuntimeError("No scene splits found under dataset roots")

    # 3) Determine deterministic meta split
    if split_index_path and os.path.isfile(split_index_path):
        js = _load_split_index_raw(split_index_path)
        train_keys = {(d["scene_version"], str(d["split_id"])) for d in js["train"]}
        val_keys = {(d["scene_version"], str(d["split_id"])) for d in js["val"]}
        train_meta = [m for m in split_meta if (m["scene_version"], m["split_id"]) in train_keys]
        val_meta = [m for m in split_meta if (m["scene_version"], m["split_id"]) in val_keys]
        for m in split_meta:
            if "base_scene" not in m:
                m["base_scene"] = m["scene_version"].split("-")[0]
        base_scenes = sorted({m["base_scene"] for m in split_meta})
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}
    else:
        train_meta, val_meta, base_scenes, train_scenes, val_scenes = _split_split_meta(
            split_meta, seed=seed, train_ratio=train_ratio, split_mode=split_mode
        )
        if split_index_path and persist_split_index:
            _write_split_index_raw(
                split_index_path,
                train_meta,
                val_meta,
                seed=seed,
                train_ratio=train_ratio,
                split_mode=split_mode,
                dataset_type=dataset_type,
                emb_mode=emb_mode,
            )

    # 4) Load heavy arrays only for the requested subset(s)
    if subset not in {"both", "train", "val"}:
        raise ValueError("subset must be one of {'both','train','val'}")

    def _load_graphs_from_meta_raw(meta_list: List[Dict]) -> List[Dict]:
        graphs = []
        for s in meta_list:
            scene_version = s["scene_version"]
            split_id = s["split_id"]
            saved_obs = s["saved_obs"]

            # Embeddings
            try:
                emb = _load_embeddings(scene_version, split_id, emb_mode=emb_mode)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue

            # GT adjacency
            gt_csv_path = os.path.join(saved_obs, "GroundTruth.csv")
            if not os.path.isfile(gt_csv_path):
                print(f"[WARN] Missing GroundTruth.csv in {saved_obs}, skipping")
                continue

            if emb.ndim == 3:
                N = emb.shape[1]  # (L, N, E)
            else:
                N = emb.shape[0]  # (N, E)

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
                    base_scene=scene_version.split("-")[0],
                )
            )
        return graphs

    train_graphs = _load_graphs_from_meta_raw(train_meta) if subset in {"both", "train"} else []
    val_graphs = _load_graphs_from_meta_raw(val_meta)     if subset in {"both", "val"} else []

    if not train_graphs and not val_graphs:
        raise RuntimeError("No valid graphs (embeddings + GT) found for the requested subset(s).")

    # 5) Build datasets
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

    # 6) Dataloaders
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

    # Meta info
    emb_dim = None
    if train_graphs:
        emb_dim = train_graphs[0]["emb"].shape[-1]
    elif val_graphs:
        emb_dim = val_graphs[0]["emb"].shape[-1]
    else:
        emb_dim = -1

    meta = dict(
        num_graphs=len(train_graphs) + len(val_graphs),
        num_train_graphs=len(train_graphs),
        num_val_graphs=len(val_graphs),
        num_scenes=len({g["base_scene"] for g in (train_graphs + val_graphs)}),
        num_train_scenes=len({g["base_scene"] for g in train_graphs}),
        num_val_scenes=len({g["base_scene"] for g in val_graphs}),
        emb_dim=emb_dim,
        split_mode=split_mode,
        emb_mode=emb_mode,
        subset=subset,
        split_index_path=split_index_path or "",
    )

    print(
        f"[INFO] Splits (meta): train={len(train_meta)} graphs, val={len(val_meta)} graphs"
    )
    print(
        f"[INFO] Loaded graphs: train={meta['num_train_graphs']}, val={meta['num_val_graphs']}"
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_neg_ratio", type=float, default=1.0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--hard_neg_rel_thr", type=float, default=0.3)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
        help="Split mode: 'scene_disjoint', 'version_disjoint', or 'graph'."
    )
    parser.add_argument(
        "--layer_mode", type=str, default="all", help="Layer mode for embeddings.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["gibson", "hm3d"],
        help="Dataset type: 'gibson' or 'hm3d'."
    )
    parser.add_argument(
        "--emb_mode",
        type=str,
        default="avg_max",
        choices=["avg", "avg_max", "chunked"],
        help="Which embedding mode to load (matches save_embeds.py)."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=["both", "train", "val"],
        help="Which subset to actually load: 'train', 'val', or 'both'.",
    )
    parser.add_argument(
        "--split_index_path",
        type=str,
        default="",
        help="Optional path to a JSON split index for deterministic splits. If it exists, it's used; if missing and --persist_split_index is set, it will be written.",
    )
    parser.add_argument(
        "--persist_split_index",
        action="store_true",
        help="When set with --split_index_path (and file missing), write the computed split to disk for future reuse.",
    )
    parser.add_argument(
        "--keep_all_data",
        action="store_true",
        help="If set, do not subsample negatives (keep all data). Overrides max_neg_ratio.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio at the meta level.",
    )
    args = parser.parse_args()

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,  # default single-dataset split; adjust if exposing via CLI
        max_neg_ratio=args.max_neg_ratio if not args.keep_all_data else -1.0,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode=args.layer_mode,
        split_mode=args.split_mode,
        emb_mode=args.emb_mode,
        subset=args.subset,
        split_index_path= f"dataset/splits/multiview/multiview_{args.seed}_{args.train_ratio}.json",#args.split_index_path or None,
        persist_split_index= True #args.persist_split_index,
    )

    print(f"[CHECK] Train dataset size: {len(train_ds)}")
    print(f"[CHECK] Val dataset size:   {len(val_ds)}")
    
    first_sample = next(iter(train_loader))[0]
    print(f"[CHECK] First batch feature shape: {first_sample.shape}")