#!/usr/bin/env python
"""
Entry-point script for training a lightweight head on top of frozen VGGT features.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

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
from vggt_trainer.model import VGGTHeadModel
from vggt_trainer.utils import (
    configure_torch_multiprocessing,
    compute_graph_metrics,
    count_parameters,
    ensure_dir,
    resolve_device,
    set_seed,
)
from utils.utils import setup_wandb, wandb_finish, wandb_log, wandb_save


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
    use_autocast = (not multiview) and str(model.device).startswith("cuda") and torch.cuda.is_available()

    with grad_ctx:
        for step, batch in enumerate(progress, start=1):
            images = batch["images"].to(model.device, non_blocking=True)
            if multiview:
                pair_idx = batch["pairs"].to(model.device, non_blocking=True)
                labels = batch["labels"].to(model.device, non_blocking=True).view(-1)
                # Some scenes can have zero labeled pairs; skip them to avoid indexing errors.
                if pair_idx.numel() == 0 or labels.numel() == 0:
                    skipped_empty_pairs += 1
                    continue
                if pair_idx.dim() != 2 or pair_idx.shape[1] != 2:
                    raise ValueError(f"Expected pair_idx shape (P,2), got {pair_idx.shape}")
            else:
                pair_idx = None
                labels = batch["label"].to(model.device, non_blocking=True).view(-1)

            with torch.cuda.amp.autocast(enabled=use_autocast):
                logits = model.score_pairs(images, pair_indices=pair_idx).view(-1)
                if logits.numel() == 0:
                    skipped_empty_pairs += 1
                    continue
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


def prepare_dataloaders(
    args,
    device: torch.device,
    wandb_run,
):
    """Build train/val dataloaders based on the selected mode."""
    use_precomputed = bool(args.precomputed_root)
    if use_precomputed:
        train_loader, val_loader, meta = build_precomputed_dataloaders_from_args(
            args,
            device=device,
        )
        train_dataset = None
        val_dataset = None
        if args.mode == "pairwise":
            print(
                f"[DATA] Precomputed pairwise | train_pairs={meta['train_pairs']} "
                f"val_pairs={meta['val_pairs']} shards={meta['train_shards']}/{meta['val_shards']}"
            )
            wandb_log(
                wandb_run,
                {
                    "data/train_pairs": meta["train_pairs"],
                    "data/val_pairs": meta["val_pairs"],
                    "data/train_shards": meta["train_shards"],
                    "data/val_shards": meta["val_shards"],
                },
            )
        else:
            print(
                f"[DATA] Precomputed multiview | train_scenes={meta['train_scenes']} "
                f"val_scenes={meta['val_scenes']} shards={meta['train_shards']}/{meta['val_shards']}"
            )
            wandb_log(
                wandb_run,
                {
                    "data/train_scenes": meta["train_scenes"],
                    "data/val_scenes": meta["val_scenes"],
                    "data/train_shards": meta["train_shards"],
                    "data/val_shards": meta["val_shards"],
                },
            )
    else:
        if args.mode == "pairwise":
            (
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
                meta,
            ) = build_pair_dataloaders_from_args(args, device=device)
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
            train_loader, val_loader, meta = build_multiview_dataloaders_from_args(
                args,
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

    return train_loader, val_loader, train_dataset, val_dataset, meta


def warmup_head(model: VGGTHeadModel, args, train_loader, train_dataset):
    """
    Run a tiny forward pass to instantiate the head before constructing the optimizer.
    """
    if args.mode == "pairwise":
        if train_dataset is not None and len(train_dataset) > 0:
            sample = train_dataset[0]
            with torch.no_grad():
                _ = model(sample["images"].unsqueeze(0).to(model.device))
        elif train_loader is not None:
            try:
                batch = next(iter(train_loader))
                with torch.no_grad():
                    _ = model(batch["images"].to(model.device))
            except StopIteration:
                pass
    elif args.mode == "multiview" and train_loader is not None:
        sample_scene = next(iter(train_loader))
        with torch.no_grad():
            emb_sample = model.encode_views(sample_scene["images"].to(model.device))
            model._init_head_if_needed(emb_dim=emb_sample.shape[-1])


def log_parameter_counts(model: VGGTHeadModel, wandb_run):
    counts = count_parameters(model, model.head)
    print(f"[PARAM] Total parameters (VGGT + head): {counts['total']:,}")
    print(f"[PARAM] Head parameters only: {counts['head']:,}")
    print(f"[PARAM] Trainable parameters: {counts['trainable']:,}")
    wandb_log(
        wandb_run,
        {
            "params/total": counts["total"],
            "params/head": counts["head"],
            "params/trainable": counts["trainable"],
        },
    )


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
    output_dir = ensure_dir(args.output_dir)

    wandb_run = setup_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        disabled=args.wandb_off,
    )

    try:
        configure_torch_multiprocessing(args.num_workers)
        device = resolve_device(args.device)
        print(f"[SETUP] Using device {device}")

        train_loader, val_loader, train_dataset, val_dataset, meta = prepare_dataloaders(
            args=args,
            device=device,
            wandb_run=wandb_run,
        )

        model = VGGTHeadModel(
            backbone_ckpt=args.backbone_ckpt,
            backbone_dtype=args.backbone_dtype,
            device=str(device),
            layer_mode=args.layer_mode,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
            token_proj_dim=args.token_proj_dim,
            summary_tokens=args.summary_tokens,
            summary_heads=args.summary_heads,
        )

        warmup_head(model, args, train_loader, train_dataset)
        log_parameter_counts(model, wandb_run)

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
                    ckpt_path = output_dir / "best_head.pt"
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
                ckpt_path = output_dir / f"head_epoch{epoch}.pt"
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
