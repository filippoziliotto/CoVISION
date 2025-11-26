#!/usr/bin/env python
"""
Quick utility to inspect the binary label distribution of the VGGT datasets.

It mirrors the dataloader construction used by train_head.py (pairwise/multiview
and raw/precomputed data) and reports how many positives/negatives are present.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Ensure repository root is importable for shared utilities.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_trainer.args import build_vggt_trainer_parser
from vggt_trainer.data import (
    build_multiview_dataloaders_from_args,
    build_pair_dataloaders_from_args,
    build_precomputed_dataloaders_from_args,
)
from vggt_trainer.utils import resolve_device, set_seed


def _parse_args():
    """
    Reuse the trainer parser so dataset-related flags behave exactly like the
    training script. Additional flags are only for this inspection script.
    """
    parser = build_vggt_trainer_parser()
    parser.description = "Inspect binary label balance for VGGT datasets."
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Limit batches scanned per split (-1 to scan everything).",
    )
    parser.add_argument(
        "--no_precomputed",
        action="store_true",
        help="Force reading raw PNGs even if precomputed_root is set.",
    )
    return parser.parse_args()


def _build_loaders(args, device: torch.device) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], Dict]:
    """
    Construct dataloaders following the same logic as train_head.py.
    """
    use_precomputed = bool(args.precomputed_root) and not args.no_precomputed
    if use_precomputed:
        root = Path(args.precomputed_root)
        if not root.exists():
            print(f"[WARN] Precomputed root '{root}' not found; falling back to raw PNGs.")
            use_precomputed = False
            args.precomputed_root = ""

    if use_precomputed:
        train_loader, val_loader, meta = build_precomputed_dataloaders_from_args(
            args,
            device=device,
            train_transform=None,
            val_transform=None,
        )
    elif args.mode == "pairwise":
        train_loader, val_loader, _, _, meta = build_pair_dataloaders_from_args(
            args,
            device=device,
            train_transform=None,
            val_transform=None,
        )
    else:
        train_loader, val_loader, meta = build_multiview_dataloaders_from_args(
            args,
            device=device,
            train_transform=None,
            val_transform=None,
        )
    return train_loader, val_loader, meta


def _accumulate_labels(tensor: torch.Tensor, stats: Dict[str, float]) -> None:
    """Update running stats for a label tensor."""
    labels = tensor.detach().float().view(-1)
    if labels.numel() == 0:
        return
    stats["total"] += labels.numel()
    stats["pos"] += float((labels >= 0.5).sum().item())
    stats["sum"] += float(labels.sum().item())
    stats["min"] = min(stats["min"], float(labels.min().item()))
    stats["max"] = max(stats["max"], float(labels.max().item()))


def _summarize_loader(
    loader: Optional[torch.utils.data.DataLoader],
    mode: str,
    split_name: str,
    max_batches: int,
) -> Optional[Dict[str, float]]:
    """Iterate over a loader and tally binary label statistics."""
    if loader is None:
        return None

    stats = dict(total=0.0, pos=0.0, sum=0.0, min=float("inf"), max=float("-inf"), batches=0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if 0 < max_batches <= batch_idx:
                break
            label_tensor = batch["labels"] if mode == "multiview" else batch["label"]
            _accumulate_labels(label_tensor, stats)
            stats["batches"] += 1

    if stats["total"] == 0:
        return stats

    stats["neg"] = stats["total"] - stats["pos"]
    stats["pos_ratio"] = stats["pos"] / stats["total"]
    stats["neg_ratio"] = stats["neg"] / stats["total"]
    stats["mean"] = stats["sum"] / stats["total"]
    stats["imbalance_gap"] = abs(stats["pos"] - stats["neg"]) / stats["total"]
    return stats


def _print_stats(split: str, stats: Optional[Dict[str, float]]) -> None:
    if stats is None:
        print(f"[INFO] No {split} split available.")
        return
    if stats["total"] == 0:
        print(f"[WARN] {split}: no labels found in scanned batches.")
        return

    pos = int(stats["pos"])
    neg = int(stats["neg"])
    total = int(stats["total"])
    print(
        f"[STATS] {split}: labels={total} | "
        f"pos={pos} ({stats['pos_ratio']*100:.2f}%) | "
        f"neg={neg} ({stats['neg_ratio']*100:.2f}%) | "
        f"pos/neg={stats['pos'] / max(1.0, stats['neg']):.3f} | "
        f"mean={stats['mean']:.4f} | "
        f"range=[{stats['min']:.3f}, {stats['max']:.3f}] | "
        f"imbalance_gap={stats['imbalance_gap']*100:.2f}pp | "
        f"batches_scanned={int(stats['batches'])}"
    )


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Single-process loading avoids macOS spawn pickling issues with worker_init_fn.
    if args.num_workers != 0:
        print(f"[INFO] Overriding num_workers={args.num_workers} -> 0 for this scan to avoid spawn pickling issues.")
        args.num_workers = 0
        args.disable_persistent_workers = True

    print(f"[INFO] Checking label balance | mode={args.mode} | dataset={args.dataset_type}")
    train_loader, val_loader, meta = _build_loaders(args, device)
    print(f"[INFO] Loaded splits | train_meta={meta}")

    train_stats = _summarize_loader(train_loader, args.mode, "train", args.max_batches)
    val_stats = _summarize_loader(val_loader, args.mode, "val", args.max_batches)

    _print_stats("train", train_stats)
    _print_stats("val", val_stats)


if __name__ == "__main__":
    main()
