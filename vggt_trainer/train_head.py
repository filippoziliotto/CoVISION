#!/usr/bin/env python
"""
Entry-point script for training a lightweight head on top of frozen VGGT features.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional
import math

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
from vggt_trainer.transform import CoVggtAug
from vggt_trainer.utils import (
    configure_torch_multiprocessing,
    compute_graph_metrics,
    count_parameters,
    ensure_dir,
    maybe_save_predictions_csv,
    resolve_device,
    set_seed,
)
from utils.utils import setup_wandb, wandb_finish, wandb_log, wandb_save


def _train_augmentation_from_args(args) -> Optional[CoVggtAug]:
    """Build the training-time augmentation pipeline from CLI args."""
    aug = CoVggtAug(
        pair_permutation_p=args.aug_pair_permutation_p,
        pair_keep_ratio=args.aug_pair_keep_ratio,
        hflip_p=args.aug_hflip_p,
        color_jitter=args.aug_color_jitter,
        gaussian_noise_std=args.aug_noise_std,
    )
    return None if aug.is_noop else aug


def run_epoch(
    model: VGGTHeadModel,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    log_every: int,
    max_steps: int,
    multiview: bool = False,
    return_preds: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []
    skipped_empty_pairs = 0
    pred_records = [] if (return_preds and not multiview) else None

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

            # Use the recommended torch.amp.autocast API for CUDA.
            autocast_device = "cuda" if use_autocast else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=use_autocast):
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

            probs = torch.sigmoid(logits).detach().cpu()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu())

            if pred_records is not None:
                paths_i = batch.get("img_path_i")
                paths_j = batch.get("img_path_j")
                if paths_i is not None and paths_j is not None:
                    if isinstance(paths_i, str):
                        paths_i_list = [paths_i]
                    elif isinstance(paths_i, (list, tuple)):
                        paths_i_list = list(paths_i)
                    else:
                        paths_i_list = list(paths_i)

                    if isinstance(paths_j, str):
                        paths_j_list = [paths_j]
                    elif isinstance(paths_j, (list, tuple)):
                        paths_j_list = list(paths_j)
                    else:
                        paths_j_list = list(paths_j)

                    labels_list = labels.detach().cpu().tolist()
                    preds_list = probs.tolist()
                    if (
                        len(paths_i_list) == len(paths_j_list) == len(preds_list)
                        and len(labels_list) == len(preds_list)
                    ):
                        for p_i, p_j, lbl, pred_val in zip(
                            paths_i_list, paths_j_list, labels_list, preds_list
                        ):
                            pred_records.append((p_i, p_j, float(lbl), float(pred_val)))

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
    if pred_records is not None:
        metrics["pred_records"] = pred_records
    if skipped_empty_pairs > 0:
        metrics["skipped_empty_pairs"] = skipped_empty_pairs
        print(f"[INFO] Skipped {skipped_empty_pairs} batch(es) with no labeled pairs.")
    return metrics


def prepare_dataloaders(
    args,
    device: torch.device,
    wandb_run,
    log_step: Optional[int] = None,
):
    """Build train/val dataloaders based on the selected mode."""
    train_transform = _train_augmentation_from_args(args)
    use_precomputed = bool(args.precomputed_root)
    if use_precomputed:
        train_loader, val_loader, meta = build_precomputed_dataloaders_from_args(
            args,
            device=device,
            train_transform=train_transform,
            val_transform=None,
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
                step=log_step,
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
                step=log_step,
            )
    else:
        if args.mode == "pairwise":
            (
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
                meta,
            ) = build_pair_dataloaders_from_args(
                args,
                device=device,
                train_transform=train_transform,
                val_transform=None,
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
                step=log_step,
            )
        else:
            train_loader, val_loader, meta = build_multiview_dataloaders_from_args(
                args,
                device=device,
                train_transform=train_transform,
                val_transform=None,
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
                step=log_step,
            )

    return train_loader, val_loader, train_dataset, val_dataset, meta


def warmup_head(model: VGGTHeadModel, args, train_loader, train_dataset):
    """
    Run a tiny forward pass to instantiate the head before constructing the optimizer.
    """
    def _ensure_head(emb_dim: int):
        model._init_head_if_needed(emb_dim=emb_dim)

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
            _ensure_head(emb_dim=emb_sample.shape[-1])


def log_parameter_counts(model: VGGTHeadModel, wandb_run, step: Optional[int] = None):
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
        step=step,
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


def load_head_checkpoint(model: VGGTHeadModel, ckpt_path: Path):
    """Load head weights from a checkpoint payload or raw state_dict."""
    payload = torch.load(ckpt_path, map_location=model.device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a dict; cannot load head weights.")

    if "head_state" in payload:
        state = payload["head_state"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        # Assume the payload itself is a compatible state dict.
        state = payload

    model.load_head_state(state)
    print(f"[CKPT] Loaded head weights from {ckpt_path}")


def save_best_head_weights(model: VGGTHeadModel, path: Path, wandb_run):
    """Persist only the head-related weights for lightweight reuse."""
    ensure_dir(path.parent)
    torch.save(model.get_head_state(), path)
    print(f"[CKPT] Saved best head weights to {path}")
    wandb_save(wandb_run, str(path))


def main():
    parser = build_vggt_trainer_parser()
    args = parser.parse_args()

    if not args.save_ckpt_path:
        args.save_ckpt_path = f"runs/precomputed/{args.mode}/head_ckpt/best_head.pth"
    best_head_ckpt_path = Path(args.save_ckpt_path)

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
            log_step=0,
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
            head_type=args.head_type,
        )

        warmup_head(model, args, train_loader, train_dataset)
        log_parameter_counts(model, wandb_run, step=0)

        criterion = nn.BCEWithLogitsLoss()

        if args.eval_ckpt_model:
            if not args.ckpt_path:
                raise ValueError("Provide --ckpt_path when using --eval_ckpt_model.")
            ckpt_path = Path(args.ckpt_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

            load_head_checkpoint(model, ckpt_path)
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_split = "val" if val_loader is not None else "train"
            if eval_loader is None:
                raise RuntimeError("No dataloader available for evaluation.")

            eval_metrics = run_epoch(
                model=model,
                loader=eval_loader,
                criterion=criterion,
                optimizer=None,
                grad_clip=0.0,
                log_every=0,
                max_steps=-1,
                multiview=(args.mode == "multiview"),
            )
            print(
                f"[EVAL-{eval_split.upper()}] loss={eval_metrics['loss']:.4f} "
                f"graph_IOU={eval_metrics['graph_IOU']:.4f} "
                f"graph_AUC={eval_metrics['graph_AUC']:.4f}"
            )
            wandb_log(
                wandb_run,
                {
                    f"{eval_split}/loss": eval_metrics["loss"],
                    f"{eval_split}/graph_IOU": eval_metrics["graph_IOU"],
                    f"{eval_split}/graph_AUC": eval_metrics["graph_AUC"],
                },
                step=0,
            )
            wandb_log(wandb_run, {"best/graph_AUC": eval_metrics["graph_AUC"]}, step=0)
            return

        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_graph_auc = -1.0
        best_pred_auc = float("-inf")
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
                    return_preds=(args.mode == "pairwise"),
                )
                pred_records = val_metrics.pop("pred_records", None)
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
                if pred_records:
                    preds_dir = Path(args.output_dir) / args.dataset_type / args.mode
                    preds_path = preds_dir / "preds.csv"
                    best_auc_path = preds_dir / "best_auc.txt"
                    imgs1, imgs2, labels_list, preds_list = zip(*pred_records)
                    best_pred_auc, saved = maybe_save_predictions_csv(
                        imgs1,
                        imgs2,
                        labels_list,
                        preds_list,
                        val_metrics.get("graph_AUC", float("nan")),
                        preds_path,
                        best_auc_path,
                    )
                    if saved:
                        print(f"[VAL]   Saved predictions to {preds_path} (graph_AUC={best_pred_auc:.4f})")
                    else:
                        auc_val = val_metrics.get("graph_AUC", float("nan"))
                        if isinstance(auc_val, float) and math.isnan(auc_val):
                            print("[VAL]   Skipped saving predictions; graph_AUC is NaN.")
                        else:
                            print(f"[VAL]   Skipped saving predictions; best graph_AUC={best_pred_auc:.4f}")
                else:
                    print("[VAL]   No prediction records available to save.")
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
                    save_best_head_weights(model, best_head_ckpt_path, wandb_run)
            else:
                # Still keep track of the best training metric for logging.
                if train_metrics["graph_AUC"] > best_graph_auc:
                    best_graph_auc = train_metrics["graph_AUC"]
                    save_best_head_weights(model, best_head_ckpt_path, wandb_run)

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

        wandb_log(wandb_run, {"best/graph_AUC": best_graph_auc}, step=args.epochs)
        print(f"[DONE] Training finished. Best metric={best_graph_auc:.4f}")
    finally:
        wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
