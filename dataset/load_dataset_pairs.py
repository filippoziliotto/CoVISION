#!/usr/bin/env python
import os
from typing import List, Dict, Tuple
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle
import argparse


PRED_ROOT = "data/predictions_feat"


def _resolve_layer_indices(layer_mode: str, num_layers: int):
    """
    Resolve which VGGT layers to use based on layer_mode.
    Returns None for all layers, a single int for one layer, or a list[int] for a range.
    """
    if layer_mode == "all":
        return None

    mode_to_offset = {
        "1st_last": -1,
        "2nd_last": -2,
        "3rd_last": -3,
        "4th_last": -4,
    }
    range_modes = {
        "last_stages": 17,         # layers 17..L
        "mid_to_last_stages": 12,  # layers 12..L
    }

    if layer_mode in mode_to_offset:
        offset = mode_to_offset[layer_mode]
        idx = (num_layers + offset) if offset < 0 else offset
        return max(0, min(num_layers - 1, idx))

    if layer_mode in range_modes:
        start_1b = range_modes[layer_mode]
        start_idx = max(0, min(num_layers - 1, start_1b - 1))
        return list(range(start_idx, num_layers))

    raise ValueError(
        f"Unknown layer_mode '{layer_mode}'. "
        f"Expected one of {list(mode_to_offset.keys()) + list(range_modes.keys()) + ['all']}."
    )


# -------------------------------------------------------
# Discover pair npz files under data/predictions_feat
# -------------------------------------------------------
def _discover_pair_graphs(emb_mode: str = "avg", pred_root: str = PRED_ROOT) -> List[Dict]:
    """
    Scan pred_root for all (scene_version, split_id) that
    have pair_embs/pairs_{emb_mode}.npz or fallback to pair_embs/pairs.npz.

    Expected layout:
        {pred_root}/{scene_version}/split_{split_id}/pair_embs/pairs_{emb_mode}.npz
        (fallback: pairs.npz)
    """
    graphs = []
    if not os.path.isdir(pred_root):
        raise RuntimeError(f"PRED_ROOT '{pred_root}' does not exist.")

    for scene_version in sorted(os.listdir(pred_root)):
        scene_dir = os.path.join(pred_root, scene_version)
        if not os.path.isdir(scene_dir):
            continue

        for split_name in sorted(os.listdir(scene_dir)):
            split_dir = os.path.join(scene_dir, split_name)
            if not os.path.isdir(split_dir):
                continue
            if not split_name.startswith("split_"):
                continue

            split_id = split_name.split("_", 1)[1]
            pair_dir = os.path.join(split_dir, "pair_embs")
            pair_path_mode = os.path.join(pair_dir, f"pairs_{emb_mode}.npz")
            pair_path_fallback = os.path.join(pair_dir, "pairs.npz")

            # Prefer pairs_{emb_mode}.npz, else fallback to pairs.npz, else skip
            if os.path.isfile(pair_path_mode):
                actual_pair_path = pair_path_mode
            elif os.path.isfile(pair_path_fallback):
                actual_pair_path = pair_path_fallback
            else:
                continue

            graphs.append(
                dict(
                    scene_version=scene_version,
                    split_id=split_id,
                    pair_path=actual_pair_path,
                )
            )

    if not graphs:
        raise RuntimeError(
            f"No pair_embs/pairs_{emb_mode}.npz or pairs.npz found under {pred_root}"
        )

    return graphs


# -------------------------------------------------------
# Utility helpers for pair-level graphs
# -------------------------------------------------------
def _load_graphs_from_meta(meta_graphs: List[Dict]) -> List[Dict]:
    """
    Given meta_graphs from _discover_pair_graphs, load the actual npz contents.
    """
    graphs = []
    for m in meta_graphs:
        scene_version = m["scene_version"]
        split_id = m["split_id"]
        pair_path = m["pair_path"]

        data = np.load(pair_path, allow_pickle=False)
        emb_i = data["emb_i"]          # (P, L, E) or (P, E)
        emb_j = data["emb_j"]          # (P, L, E) or (P, E)
        labels = data["labels"]        # (P,)
        strengths = data["strengths"]  # (P,)

        graphs.append(
            dict(
                scene_version=scene_version,
                split_id=split_id,
                emb_i=emb_i,
                emb_j=emb_j,
                labels=labels,
                strengths=strengths,
            )
        )
    return graphs


# -------------------------------------------------------
# New: Meta-level split helpers for deterministic splits
# -------------------------------------------------------
def _split_meta_graphs(
    meta_graphs: List[Dict],
    seed: int,
    train_ratio: float,
    split_mode: str,
) -> Tuple[List[Dict], List[Dict], List[str], set, set]:
    """
    Split *meta* graphs (without loading arrays) into train/val according to split_mode and train_ratio.

    Returns:
        train_meta, val_meta, base_scenes, train_scenes, val_scenes
    """
    if not meta_graphs:
        return [], [], [], set(), set()

    # compute base_scene from scene_version (e.g., "Adrian-3" -> "Adrian")
    for m in meta_graphs:
        if "base_scene" not in m:
            m["base_scene"] = m["scene_version"].split("-")[0]

    base_scenes = sorted({m["base_scene"] for m in meta_graphs})

    if split_mode == "scene_disjoint":
        scenes_shuffled = sk_shuffle(base_scenes, random_state=seed)
        n_train = int(len(base_scenes) * train_ratio)
        train_scenes = set(scenes_shuffled[:n_train])
        val_scenes = set(scenes_shuffled[n_train:])
        train_meta = [m for m in meta_graphs if m["base_scene"] in train_scenes]
        val_meta = [m for m in meta_graphs if m["base_scene"] in val_scenes]

    elif split_mode == "version_disjoint":
        scene_versions = sorted({m["scene_version"] for m in meta_graphs})
        scene_versions_shuffled = sk_shuffle(scene_versions, random_state=seed)
        n_train = int(len(scene_versions) * train_ratio)
        train_versions = set(scene_versions_shuffled[:n_train])
        val_versions = set(scene_versions_shuffled[n_train:])
        train_meta = [m for m in meta_graphs if m["scene_version"] in train_versions]
        val_meta = [m for m in meta_graphs if m["scene_version"] in val_versions]
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}

    elif split_mode == "graph":
        all_idx = list(range(len(meta_graphs)))
        all_idx = sk_shuffle(all_idx, random_state=seed)
        n_train_graphs = int(len(all_idx) * train_ratio)
        train_idx = set(all_idx[:n_train_graphs])
        val_idx = set(all_idx[n_train_graphs:])
        train_meta = [meta_graphs[i] for i in all_idx if i in train_idx]
        val_meta = [meta_graphs[i] for i in all_idx if i in val_idx]
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}
    else:
        raise ValueError(
            f"Invalid split_mode '{split_mode}'. "
            f"Valid: 'scene_disjoint', 'version_disjoint', 'graph'."
        )

    return train_meta, val_meta, base_scenes, train_scenes, val_scenes


def _load_split_index_pairs(path: str) -> Dict:
    with open(path, "r") as f:
        js = json.load(f)
    if "train" not in js or "val" not in js:
        raise ValueError(f"Split index at {path} missing 'train'/'val' keys.")
    return js


def _write_split_index_pairs(
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
            "source": "load_dataset_pairs.py",
        },
        "train": [{"scene_version": m["scene_version"], "split_id": m["split_id"]} for m in train_meta],
        "val": [{"scene_version": m["scene_version"], "split_id": m["split_id"]} for m in val_meta],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _split_graphs(
    graphs: List[Dict],
    seed: int,
    train_ratio: float,
    split_mode: str,
) -> Tuple[List[Dict], List[Dict], List[str], set, set]:
    """
    Split graphs into train/val according to split_mode and train_ratio.

    Returns:
        train_graphs, val_graphs, base_scenes, train_scenes, val_scenes
    """
    if not graphs:
        return [], [], [], set(), set()

    # Base scene name (without version suffix), e.g. "Adrian" from "Adrian-3"
    for g in graphs:
        if "base_scene" not in g:
            g["base_scene"] = g["scene_version"].split("-")[0]

    base_scenes = sorted({g["base_scene"] for g in graphs})

    if split_mode == "scene_disjoint":
        scenes_shuffled = sk_shuffle(base_scenes, random_state=seed)
        n_train = int(len(base_scenes) * train_ratio)
        train_scenes = set(scenes_shuffled[:n_train])
        val_scenes = set(scenes_shuffled[n_train:])
        train_graphs = [g for g in graphs if g["base_scene"] in train_scenes]
        val_graphs = [g for g in graphs if g["base_scene"] in val_scenes]

    elif split_mode == "version_disjoint":
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
        all_idx = list(range(len(graphs)))
        all_idx = sk_shuffle(all_idx, random_state=seed)
        n_train_graphs = int(len(all_idx) * train_ratio)
        train_idx = all_idx[:n_train_graphs]
        val_idx = all_idx[n_train_graphs:]
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        train_scenes = {g["base_scene"] for g in train_graphs}
        val_scenes = {g["base_scene"] for g in val_graphs}
    else:
        raise ValueError(
            f"Invalid split_mode '{split_mode}'. "
            f"Valid: 'scene_disjoint', 'version_disjoint', 'graph'."
        )

    return train_graphs, val_graphs, base_scenes, train_scenes, val_scenes


# -------------------------------------------------------
# Pair-level dataset from pairs.npz
# -------------------------------------------------------
class EdgePairDatasetPairs(Dataset):
    """
    Dataset built directly from precomputed pair embeddings.

    Each graph corresponds to one (scene_version, split_id) with:
        emb_i:    (P, L, E) or (P, E)
        emb_j:    (P, L, E) or (P, E)
        labels:   (P,)
        strengths:(P,)
    We then build (feat_i, feat_j, label, strength) pairs with optional
    negative subsampling and hard negative mining on strengths.
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
          - "scene_version": str
          - "split_id": str
          - "emb_i": (P, L, E) or (P, E)
          - "emb_j": (P, L, E) or (P, E)
          - "labels": (P,)
          - "strengths": (P,)
        max_neg_ratio: float, maximum number of negatives to keep per positive (ratio)
        hard_neg_ratio: float in [0,1], fraction of selected negatives that should be "hard"
        hard_neg_rel_thr: minimum strength value for a negative to be considered hard
        layer_mode: "all" or one of ["1st_last", "2nd_last", "3rd_last", "4th_last",
                                     "last_stages", "mid_to_last_stages"]
        """
        self.layer_mode = layer_mode
        self.pairs = []
        total_pos = 0
        total_neg = 0

        for g in graphs:
            scene_version = g["scene_version"]
            split_id = g["split_id"]

            emb_i = g["emb_i"].astype(np.float32)
            emb_j = g["emb_j"].astype(np.float32)
            labels = g["labels"].astype(np.float32)
            strengths = g["strengths"].astype(np.float32)

            # Ensure shape (P, L, E)
            if emb_i.ndim == 2:
                # (P, E) -> (P, 1, E)
                emb_i = emb_i[:, None, :]
                emb_j = emb_j[:, None, :]
            elif emb_i.ndim != 3:
                raise ValueError(
                    f"Expected emb_i with ndim 2 or 3, got {emb_i.shape} "
                    f"for {scene_version} split {split_id}"
                )

            P, L, E = emb_i.shape

            # Decide which layers to use
            layer_indices = _resolve_layer_indices(self.layer_mode, L)

            pos_pairs = []
            neg_pairs_hard = []
            neg_pairs_easy = []

            for p in range(P):
                lbl = labels[p]
                str_p = strengths[p]

                if layer_indices is None:
                    # Use all layers: (L, E)
                    feat_i = emb_i[p, :, :]  # (L, E)
                    feat_j = emb_j[p, :, :]  # (L, E)
                elif isinstance(layer_indices, int):
                    # Single layer: (E,)
                    feat_i = emb_i[p, layer_indices, :]  # (E,)
                    feat_j = emb_j[p, layer_indices, :]  # (E,)
                else:
                    # Multiple layers: (L_sub, E)
                    feat_i = emb_i[p, layer_indices, :]  # (L_sub, E)
                    feat_j = emb_j[p, layer_indices, :]  # (L_sub, E)

                triple = (
                    feat_i.astype(np.float32),
                    feat_j.astype(np.float32),
                    float(lbl),
                    float(str_p),
                )

                if lbl == 1.0:
                    pos_pairs.append(triple)
                else:
                    # Negative: check hardness via strengths
                    if str_p >= hard_neg_rel_thr:
                        neg_pairs_hard.append(triple)
                    else:
                        neg_pairs_easy.append(triple)

            # Compute and print per-graph statistics
            num_pos = len(pos_pairs)
            num_neg_hard = len(neg_pairs_hard)
            num_neg_easy = len(neg_pairs_easy)
            num_neg_total = num_neg_hard + num_neg_easy
            print(
                f"[DEBUG] Graph {scene_version} split {split_id}: "
                f"P={P}, pos={num_pos}, neg_hard={num_neg_hard}, "
                f"neg_easy={num_neg_easy}, neg_total={num_neg_total}"
            )

            if num_pos == 0:
                print(
                    f"[WARN] Graph {scene_version} split {split_id} "
                    f"has no positive pairs, skipping."
                )
                continue

            # Keep all positives
            self.pairs.extend(pos_pairs)
            total_pos += num_pos

            # Negative subsampling
            if num_neg_total == 0:
                continue

            # If max_neg_ratio <= 0, keep all negatives (no subsampling)
            if max_neg_ratio <= 0:
                sampled_neg = neg_pairs_hard + neg_pairs_easy
            else:
                k_total_neg = int(num_pos * max_neg_ratio)
                k_total_neg = min(k_total_neg, num_neg_total)
                if k_total_neg <= 0:
                    continue

                k_hard = min(len(neg_pairs_hard), int(k_total_neg * hard_neg_ratio))
                k_easy = k_total_neg - k_hard

                sampled_neg = []
                if k_hard > 0:
                    sampled_neg.extend(random.sample(neg_pairs_hard, k_hard))
                if k_easy > 0 and len(neg_pairs_easy) > 0:
                    k_easy = min(k_easy, len(neg_pairs_easy))
                    sampled_neg.extend(random.sample(neg_pairs_easy, k_easy))

            kept_neg = len(sampled_neg)
            print(
                f"[DEBUG] Kept negatives for {scene_version} split {split_id}: "
                f"kept={kept_neg} / total={num_neg_total}"
            )

            self.pairs.extend(sampled_neg)
            total_neg += kept_neg

        print(
            f"[INFO] EdgePairDatasetPairs: total pairs={len(self.pairs)} "
            f"(pos={total_pos}, neg={total_neg}, "
            f"neg/pos={total_neg / max(1, total_pos):.2f})"
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
# Main builder for pair embeddings
# -------------------------------------------------------
def build_dataloaders_pairs(
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
    emb_mode: str = "avg",
    subset: str = "both",                   # new: 'train' | 'val' | 'both'
    split_index_path: str = None,           # new: optional path to load/save deterministic split
    persist_split_index: bool = False,      # new: if True and split_index_path is set, write it when missing
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict]:
    """
    Build dataloaders directly from pair-level embeddings for a single dataset
    (either Gibson or HM3D), splitting that dataset into train/val.

    Important change: we now split at the *meta* level and then only load the
    requested subset(s), so heavy arrays for the unused split are never loaded.
    Deterministic splits can be persisted/loaded via split_index_path.
    """
    if dataset_type == "hm3d":
        pred_root = "data/predictions_feat/hvgg"
    else:
        pred_root = "data/predictions_feat/gvgg"

    random.seed(seed)
    np.random.seed(seed)

    # 1) Discover all pair graphs (meta only)
    meta_graphs = _discover_pair_graphs(emb_mode=emb_mode, pred_root=pred_root)
    if not meta_graphs:
        raise RuntimeError(f"No valid pair graphs found in predictions_feat for dataset_type='{dataset_type}'.")

    # 2) Determine train/val meta-splits deterministically
    if split_index_path and os.path.isfile(split_index_path):
        js = _load_split_index_pairs(split_index_path)
        train_keys = {(d["scene_version"], str(d["split_id"])) for d in js["train"]}
        val_keys = {(d["scene_version"], str(d["split_id"])) for d in js["val"]}
        train_meta = [m for m in meta_graphs if (m["scene_version"], m["split_id"]) in train_keys]
        val_meta = [m for m in meta_graphs if (m["scene_version"], m["split_id"]) in val_keys]
        # Compute sets for reporting
        for m in meta_graphs:
            if "base_scene" not in m:
                m["base_scene"] = m["scene_version"].split("-")[0]
        base_scenes = sorted({m["base_scene"] for m in meta_graphs})
        train_scenes = {m["base_scene"] for m in train_meta}
        val_scenes = {m["base_scene"] for m in val_meta}
    else:
        train_meta, val_meta, base_scenes, train_scenes, val_scenes = _split_meta_graphs(
            meta_graphs,
            seed=seed,
            train_ratio=train_ratio,
            split_mode=split_mode,
        )
        if split_index_path and persist_split_index:
            _write_split_index_pairs(
                split_index_path,
                train_meta,
                val_meta,
                seed=seed,
                train_ratio=train_ratio,
                split_mode=split_mode,
                dataset_type=dataset_type,
                emb_mode=emb_mode,
            )

    # 3) Load only the requested subsets
    if subset not in {"both", "train", "val"}:
        raise ValueError("subset must be one of {'both','train','val'}")

    train_graphs = _load_graphs_from_meta(train_meta) if subset in {"both", "train"} else []
    val_graphs = _load_graphs_from_meta(val_meta)     if subset in {"both", "val"} else []

    # 4) Build datasets
    train_ds = EdgePairDatasetPairs(
        train_graphs,
        max_neg_ratio=max_neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_rel_thr=hard_neg_rel_thr,
        layer_mode=layer_mode,
    )
    val_ds = EdgePairDatasetPairs(
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

    # Embedding dim from any available side
    emb_dim = None
    if train_graphs:
        emb_dim = train_graphs[0]["emb_i"].shape[-1]
    elif val_graphs:
        emb_dim = val_graphs[0]["emb_i"].shape[-1]
    else:
        emb_dim = -1

    meta = dict(
        num_graphs=len(meta_graphs),
        num_train_graphs=len(train_meta),
        num_val_graphs=len(val_meta),
        num_scenes=len(base_scenes),
        num_train_scenes=len(train_scenes),
        num_val_scenes=len(val_scenes),
        emb_dim=emb_dim,
        split_mode=split_mode,
        dataset_type=dataset_type,
        emb_mode=emb_mode,
        subset=subset,
        split_index_path=split_index_path or "",
    )

    print(
        f"[INFO] Pair graphs: total={meta['num_graphs']}, "
        f"train_meta={meta['num_train_graphs']}, val_meta={meta['num_val_graphs']}"
    )
    print(
        f"[INFO] Scenes: total={meta['num_scenes']}, "
        f"train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )
    print(f"[INFO] Pair embedding dim: {meta['emb_dim']}")

    return train_loader, val_loader, train_ds, val_ds, meta


# -------------------------------------------------------
# Cross-dataset builder for pair embeddings
# -------------------------------------------------------
def build_dataloaders_pairs_cross(
    train_dataset_type: str,
    eval_dataset_type: str,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 0,
    max_neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    hard_neg_rel_thr: float = 0.3,
    layer_mode: str = "1st_last",
    split_mode: str = "scene_disjoint",
    emb_mode: str = "avg",
    train_train_ratio: float = None,
    eval_train_ratio: float = None,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict]:
    """
    Build dataloaders for cross-dataset training:

      - Load train_dataset_type (e.g. 'hm3d'), split it with train_train_ratio
        and keep ONLY the train split for training.
      - Load eval_dataset_type (e.g. 'gibson'), split it with eval_train_ratio
        and keep ONLY the validation split for evaluation.

    Example: train_dataset_type='hm3d', eval_dataset_type='gibson'
      - HM3D uses 0.9 train ratio (keep only that part for training).
      - Gibson uses 0.8 train ratio (keep only the held-out 20% for validation).
    """
    random.seed(seed)
    np.random.seed(seed)

    def _root_for(dtype: str) -> str:
        return "data/predictions_feat/hvgg" if dtype == "hm3d" else "data/predictions_feat/gvgg"

    # Default train ratios if not provided
    if train_train_ratio is None:
        train_train_ratio = 0.8 if train_dataset_type == "gibson" else 0.9
    if eval_train_ratio is None:
        eval_train_ratio = 0.8 if eval_dataset_type == "gibson" else 0.9

    # --------- TRAIN DATASET ---------
    train_root = _root_for(train_dataset_type)
    train_meta_graphs = _discover_pair_graphs(emb_mode=emb_mode, pred_root=train_root)
    train_graphs_all = _load_graphs_from_meta(train_meta_graphs)
    if not train_graphs_all:
        raise RuntimeError(f"No valid pair graphs found for train_dataset_type='{train_dataset_type}'.")

    (
        train_graphs_train,
        train_graphs_val_unused,
        base_scenes_train,
        train_scenes_train,
        val_scenes_train_unused,
    ) = _split_graphs(
        train_graphs_all,
        seed=seed,
        train_ratio=train_train_ratio,
        split_mode=split_mode,
    )

    # --------- EVAL DATASET ---------
    eval_root = _root_for(eval_dataset_type)
    eval_meta_graphs = _discover_pair_graphs(emb_mode=emb_mode, pred_root=eval_root)
    eval_graphs_all = _load_graphs_from_meta(eval_meta_graphs)
    if not eval_graphs_all:
        raise RuntimeError(f"No valid pair graphs found for eval_dataset_type='{eval_dataset_type}'.")

    (
        eval_graphs_train_unused,
        eval_graphs_val,
        base_scenes_eval,
        train_scenes_eval_unused,
        val_scenes_eval,
    ) = _split_graphs(
        eval_graphs_all,
        seed=seed,
        train_ratio=eval_train_ratio,
        split_mode=split_mode,
    )

    # --------- Build datasets/loaders ---------
    train_ds = EdgePairDatasetPairs(
        train_graphs_train,
        max_neg_ratio=max_neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_rel_thr=hard_neg_rel_thr,
        layer_mode=layer_mode,
    )
    val_ds = EdgePairDatasetPairs(
        eval_graphs_val,
        max_neg_ratio=max_neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        hard_neg_rel_thr=hard_neg_rel_thr,
        layer_mode=layer_mode,
    )

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

    emb_dim_train = train_graphs_all[0]["emb_i"].shape[-1]
    emb_dim_eval = eval_graphs_all[0]["emb_i"].shape[-1]
    if emb_dim_train != emb_dim_eval:
        raise ValueError(
            f"Embedding dims differ between datasets: {emb_dim_train} (train) vs {emb_dim_eval} (eval)."
        )
    emb_dim = emb_dim_train

    all_base_scenes = sorted(set(base_scenes_train) | set(base_scenes_eval))

    meta = dict(
        num_graphs=len(train_graphs_all) + len(eval_graphs_all),
        num_train_graphs=len(train_graphs_train),
        num_val_graphs=len(eval_graphs_val),
        num_scenes=len(all_base_scenes),
        num_train_scenes=len(train_scenes_train),
        num_val_scenes=len(val_scenes_eval),
        emb_dim=emb_dim,
        split_mode=split_mode,
        train_dataset_type=train_dataset_type,
        eval_dataset_type=eval_dataset_type,
        emb_mode=emb_mode,
    )

    print(
        f"[INFO] Cross Pair graphs (train={train_dataset_type}, eval={eval_dataset_type}): "
        f"train_graphs={meta['num_train_graphs']}, val_graphs={meta['num_val_graphs']}"
    )
    print(
        f"[INFO] Scenes (train={train_dataset_type}, eval={eval_dataset_type}): "
        f"total={meta['num_scenes']}, train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )
    print(f"[INFO] Pair embedding dim: {meta['emb_dim']}")

    return train_loader, val_loader, train_ds, val_ds, meta


# -------------------------------------------------------
# CLI smoke test
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_neg_ratio", type=float, default=1.0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--hard_neg_rel_thr", type=float, default=0.3)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="1st_last",
        choices=[
            "all",
            "1st_last",
            "2nd_last",
            "3rd_last",
            "4th_last",
            "last_stages",
            "mid_to_last_stages",
        ],
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["hm3d", "gibson"],
        help="Dataset type / source of pair embeddings: 'gibson' or 'hm3d'.",
    )
    parser.add_argument(
        "--emb_mode",
        type=str,
        default="avg",
        choices=["avg", "avg_max", "chunked"],
        help="Dataset type / source of pair embeddings: 'gibson' or 'hm3d'.",
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
    args = parser.parse_args()

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=0.8,  # default for single-dataset builder; can be exposed if needed
        max_neg_ratio=args.max_neg_ratio if not args.keep_all_data else -1.0,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode=args.layer_mode,
        split_mode=args.split_mode,
        emb_mode=args.emb_mode,
        subset=args.subset,
        split_index_path=args.split_index_path or None,
        persist_split_index=args.persist_split_index,
    )

    print(f"[CHECK] Train dataset size: {len(train_ds)}")
    print(f"[CHECK] Val dataset size:   {len(val_ds)}")

    first_batch = next(iter(train_loader))
    feat_i = first_batch[0]
    print(f"[CHECK] First batch feat_i shape: {feat_i.shape}")
