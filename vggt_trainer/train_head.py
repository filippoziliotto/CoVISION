#!/usr/bin/env python
"""
Entry-point script for training a lightweight head on top of frozen VGGT features.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add the main folder to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt_trainer.args import build_vggt_trainer_parser
from vggt_trainer.data import (
    build_image_pair_dataloaders,
    build_multiview_dataloaders,
)
from vggt_trainer.model import VGGTHeadModel
from utils.utils import setup_wandb, wandb_finish, wandb_log, wandb_save


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_graph_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Mirror the graph metrics used in the existing trainer:
        - graph_IOU: best IoU across probability thresholds
        - graph_AUC: area under the IoU-threshold curve
    """
    if probs.size == 0:
        return {"graph_IOU": float("nan"), "graph_AUC": float("nan")}

    y_true = (labels >= 0.5).astype(np.float32)
    thresholds = np.linspace(0.0, 1.0, 100)
    ious = []
    eps = 1e-6

    for t in thresholds:
        pred = (probs >= t).astype(np.float32)
        inter = np.logical_and(pred, y_true).sum()
        union = np.logical_or(pred, y_true).sum()
        iou = inter / (union + eps)
        ious.append(iou)

    if hasattr(np, "trapezoid"):
        graph_auc = float(np.trapezoid(ious, thresholds))
    else:
        graph_auc = float(np.trapz(ious, thresholds))
    best_iou = float(np.max(ious))
    return {"graph_IOU": best_iou, "graph_AUC": graph_auc}


def run_epoch(
    model: VGGTHeadModel,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    log_every: int,
    max_steps: int,
    multiview: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []
    skipped_empty_pairs = 0

    if (not is_train) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    loop_desc = "Train" if is_train else "Eval"
    progress = tqdm(loader, desc=loop_desc, leave=False)

    with grad_ctx:
        for step, batch in enumerate(progress, start=1):
            if multiview:
                images = batch["images"].to(model.device, non_blocking=True)
                pair_idx = batch["pairs"].to(model.device, non_blocking=True)
                labels = batch["labels"].to(model.device, non_blocking=True).view(-1)
                # Some scenes can have zero labeled pairs; skip them to avoid indexing errors.
                if pair_idx.numel() == 0 or labels.numel() == 0:
                    skipped_empty_pairs += 1
                    continue
                if pair_idx.dim() != 2 or pair_idx.shape[1] != 2:
                    raise ValueError(f"Expected pair_idx shape (P,2), got {pair_idx.shape}")
                view_embs = model.encode_views(images)
                logits = model.score_pair_indices(view_embs, pair_idx).view(-1)
            else:
                images = batch["images"].to(model.device, non_blocking=True)
                labels = batch["label"].to(model.device, non_blocking=True).view(-1)

                outputs = model(images)
                logits = outputs["logits"].view(-1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.head_parameters(), grad_clip)
                optimizer.step()

            batch_size = labels.shape[0]
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

            if is_train and log_every > 0 and step % log_every == 0:
                avg_loss = total_loss / max(1, total_samples)
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                print(f"[TRAIN] step={step} loss={avg_loss:.4f}")

            if max_steps > 0 and step >= max_steps:
                break

    progress.close()
    avg_loss = total_loss / max(1, total_samples)

    if all_probs:
        probs_np = torch.cat(all_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        metrics = compute_graph_metrics(probs_np, labels_np)
    else:
        metrics = {"graph_IOU": float("nan"), "graph_AUC": float("nan")}

    metrics["loss"] = avg_loss
    if skipped_empty_pairs > 0:
        metrics["skipped_empty_pairs"] = skipped_empty_pairs
        print(f"[INFO] Skipped {skipped_empty_pairs} batch(es) with no labeled pairs.")
    return metrics


def save_head_checkpoint(
    model: VGGTHeadModel,
    optimizer: Optional[torch.optim.Optimizer],
    args,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
):
    payload = {
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
        "head_state": model.get_head_state(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)
    print(f"[CKPT] Saved checkpoint to {path}")


def main():
    parser = build_vggt_trainer_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run = setup_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        disabled=args.wandb_off,
    )

    try:
        device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SETUP] Using device {device}")

        if args.mode == "pairwise":
            (
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
                meta,
            ) = build_image_pair_dataloaders(
                dataset_type=args.dataset_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                train_ratio=args.train_ratio,
                split_mode=args.split_mode,
                split_index_path=args.split_index_path or None,
                preprocess_mode=args.preprocess_mode,
                square_size=args.square_size,
                max_pairs_per_split=args.max_pairs_per_split,
                device=device,
            )
            print(
                f"[DATA] Loaded {meta['train_pairs']} train pairs "
                f"({meta['train_scenes']} scenes). "
                f"Val pairs={meta['val_pairs']} ({meta['val_scenes']} scenes)"
            )
            wandb_log(
                wandb_run,
                {
                    "data/train_pairs": meta["train_pairs"],
                    "data/val_pairs": meta["val_pairs"],
                    "data/train_scenes": meta["train_scenes"],
                    "data/val_scenes": meta["val_scenes"],
                },
            )
        else:
            train_loader, val_loader, meta = build_multiview_dataloaders(
                dataset_type=args.dataset_type,
                train_ratio=args.train_ratio,
                split_mode=args.split_mode,
                seed=args.seed,
                batch_size=1,
                num_workers=args.num_workers,
                preprocess_mode=args.preprocess_mode,
                square_size=args.square_size,
                max_pairs_per_scene=args.max_pairs_per_scene,
                split_index_path=args.split_index_path or None,
                device=device,
            )
            train_dataset = None
            val_dataset = None
            print(
                f"[DATA] Multiview scenes: train={meta['train_scenes']}, "
                f"val={meta['val_scenes']}, max_pairs_per_scene={meta['max_pairs_per_scene']}"
            )
            wandb_log(
                wandb_run,
                {
                    "data/train_scenes": meta["train_scenes"],
                    "data/val_scenes": meta["val_scenes"],
                    "data/max_pairs_per_scene": meta["max_pairs_per_scene"],
                },
            )

        model = VGGTHeadModel(
            backbone_ckpt=args.backbone_ckpt,
            device=device,
            layer_mode=args.layer_mode,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
            token_proj_dim=args.token_proj_dim,
            summary_tokens=args.summary_tokens,
            summary_heads=args.summary_heads,
        )

        # Make sure the head exists before constructing the optimizer by running a dry pass.
        if args.mode == "pairwise" and train_loader is not None:
            sample = train_dataset[0]
            with torch.no_grad():
                _ = model(sample["images"].unsqueeze(0).to(model.device))
        elif args.mode == "multiview" and train_loader is not None:
            sample_scene = next(iter(train_loader))
            with torch.no_grad():
                emb_sample = model.encode_views(sample_scene["images"].to(model.device))
                if emb_sample.dim() == 2:
                    model._init_head_if_needed(emb_dim=emb_sample.shape[-1])
                else:
                    model._init_head_if_needed(emb_dim=emb_sample.shape[-1])

        total_params = sum(p.numel() for p in model.parameters())
        head_params = sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[PARAM] Total parameters (VGGT + head): {total_params:,}")
        print(f"[PARAM] Head parameters only: {head_params:,}")
        print(f"[PARAM] Trainable parameters: {trainable_params:,}")
        wandb_log(
            wandb_run,
            {
                "params/total": total_params,
                "params/head": head_params,
                "params/trainable": trainable_params,
            },
        )

        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_graph_auc = -1.0
        for epoch in range(1, args.epochs + 1):
            print(f"\n[EPOCH {epoch}] -----------------------------")
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                max_steps=args.max_train_steps,
                multiview=(args.mode == "multiview"),
            )
            print(
                f"[TRAIN] epoch={epoch} loss={train_metrics['loss']:.4f} "
                f"graph_IOU={train_metrics['graph_IOU']:.4f} "
                f"graph_AUC={train_metrics['graph_AUC']:.4f}"
            )
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/graph_IOU": train_metrics["graph_IOU"],
                    "train/graph_AUC": train_metrics["graph_AUC"],
                },
                step=epoch,
            )

            if not args.skip_eval and val_loader is not None:
                # Allow validation whenever a loader exists (multiview sets val_dataset=None).
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    grad_clip=0.0,
                    log_every=0,
                    max_steps=-1,
                    multiview=(args.mode == "multiview"),
                )
                print(
                    f"[VAL]   epoch={epoch} loss={val_metrics['loss']:.4f} "
                    f"graph_IOU={val_metrics['graph_IOU']:.4f} "
                    f"graph_AUC={val_metrics['graph_AUC']:.4f}"
                )
                wandb_log(
                    wandb_run,
                    {
                        "epoch": epoch,
                        "val/loss": val_metrics["loss"],
                        "val/graph_IOU": val_metrics["graph_IOU"],
                        "val/graph_AUC": val_metrics["graph_AUC"],
                    },
                    step=epoch,
                )
                if val_metrics["graph_AUC"] > best_graph_auc:
                    best_graph_auc = val_metrics["graph_AUC"]
                    ckpt_path = Path(args.output_dir) / "best_head.pt"
                    save_head_checkpoint(
                        model,
                        optimizer,
                        args,
                        epoch,
                        metrics={
                            "train": train_metrics,
                            "val": val_metrics,
                        },
                        path=ckpt_path,
                    )
                    wandb_save(wandb_run, str(ckpt_path))
            else:
                # Still keep track of the best training metric for logging.
                if train_metrics["graph_AUC"] > best_graph_auc:
                    best_graph_auc = train_metrics["graph_AUC"]

            if args.save_every > 0 and epoch % args.save_every == 0:
                ckpt_path = Path(args.output_dir) / f"head_epoch{epoch}.pt"
                save_head_checkpoint(
                    model,
                    optimizer,
                    args,
                    epoch,
                    metrics={"train": train_metrics},
                    path=ckpt_path,
                )
                wandb_save(wandb_run, str(ckpt_path))

        wandb_log(wandb_run, {"best/graph_AUC": best_graph_auc})
        print(f"[DONE] Training finished. Best metric={best_graph_auc:.4f}")
    finally:
        wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
