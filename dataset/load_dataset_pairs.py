#!/usr/bin/env python
import os
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle
import argparse


PRED_ROOT = "data/predictions_feat"


# -------------------------------------------------------
# Discover pair npz files under data/predictions_feat
# -------------------------------------------------------
def _discover_pair_graphs(emb_mode:str = "avg") -> List[Dict]:
    """
    Scan data/predictions_feat for all (scene_version, split_id) that
    have pair_embs/pairs.npz.

    Expected layout:
        data/predictions_feat/{scene_version}/split_{split_id}/pair_embs/pairs.npz
    """
    graphs = []
    if not os.path.isdir(PRED_ROOT):
        raise RuntimeError(f"PRED_ROOT '{PRED_ROOT}' does not exist.")

    for scene_version in sorted(os.listdir(PRED_ROOT)):
        scene_dir = os.path.join(PRED_ROOT, scene_version)
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
            pair_path = os.path.join(pair_dir, f"pairs_{emb_mode}.npz")

            if not os.path.isfile(pair_path):
                continue

            graphs.append(
                dict(
                    scene_version=scene_version,
                    split_id=split_id,
                    pair_path=pair_path,
                )
            )

    if not graphs:
        raise RuntimeError(f"No pair_embs/pairs_{emb_mode}.npz found under data/predictions_feat")

    return graphs


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
        layer_mode: "all" or one of ["1st_last", "2nd_last", "3rd_last", "4th_last"]
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
            mode = self.layer_mode
            if mode == "all":
                layer_indices = None  # keep (L, E)
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
                idx = (L + offset) if offset < 0 else offset
                idx = max(0, min(L - 1, idx))
                layer_indices = idx  # int index

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
                else:
                    # Single layer: (E,)
                    feat_i = emb_i[p, layer_indices, :]  # (E,)
                    feat_j = emb_j[p, layer_indices, :]  # (E,)

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
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict]:
    """
    Build dataloaders directly from pair-level embeddings:

      data/predictions_feat/{scene_version}/split_{split_id}/pair_embs/pairs.npz

    Args largely mirror load_dataset.build_dataloaders.

    Args:
        dataset_type: "gibson" (default) or "hm3d", used to pick gvgg/hvgg roots.
        split_mode: "scene_disjoint", "version_disjoint", or "graph" (same semantics as load_dataset).
        ...
    """
    global PRED_ROOT
    if dataset_type == "hm3d":
        PRED_ROOT = "data/predictions_feat/hvgg"
    else:
        PRED_ROOT = "data/predictions_feat/gvgg"

    random.seed(seed)
    np.random.seed(seed)

    # 1) Discover all pair graphs
    meta_graphs = _discover_pair_graphs(emb_mode=emb_mode)

    # 2) Load each pairs.npz into memory
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

    if not graphs:
        raise RuntimeError("No valid pair graphs found in predictions_feat.")

    # Base scene name (without version suffix), e.g. "Adrian" from "Adrian-3"
    for g in graphs:
        g["base_scene"] = g["scene_version"].split("-")[0]

    base_scenes = sorted({g["base_scene"] for g in graphs})

    # 3) Split strategy (scene_disjoint / version_disjoint / graph)
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

    # Meta info
    emb_dim = graphs[0]["emb_i"].shape[-1]
    meta = dict(
        num_graphs=len(graphs),
        num_train_graphs=len(train_graphs),
        num_val_graphs=len(val_graphs),
        num_scenes=len(base_scenes),
        num_train_scenes=len(train_scenes),
        num_val_scenes=len(val_scenes),
        emb_dim=emb_dim,
        split_mode=split_mode,
        dataset_type=dataset_type,
        emb_mode=emb_mode,
    )

    print(
        f"[INFO] Pair graphs: total={meta['num_graphs']}, "
        f"train={meta['num_train_graphs']}, val={meta['num_val_graphs']}"
    )
    print(
        f"[INFO] Scenes: total={meta['num_scenes']}, "
        f"train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
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
        choices=["all", "1st_last", "2nd_last", "3rd_last", "4th_last"],
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
    args = parser.parse_args()

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_neg_ratio=args.max_neg_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode=args.layer_mode,
        split_mode=args.split_mode,
        emb_mode=args.emb_mode
    )

    print(f"[CHECK] Train dataset size: {len(train_ds)}")
    print(f"[CHECK] Val dataset size:   {len(val_ds)}")

    first_batch = next(iter(train_loader))
    feat_i = first_batch[0]
    print(f"[CHECK] First batch feat_i shape: {feat_i.shape}")