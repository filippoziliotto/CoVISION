#!/usr/bin/env python
"""
Argument helpers for the in-pipeline VGGT head trainer.
Only used within the vggt_trainer module to keep the main training script tidy.
"""
import argparse


def build_vggt_trainer_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the VGGT head finetuning script."""
    parser = argparse.ArgumentParser(
        description="Finetune a lightweight head on top of frozen VGGT features directly from RGB pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="multiview",
        choices=["pairwise", "multiview"],
        help="Training mode: pairwise edges or multiview scenes.",
    )

    # Data-related arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["gibson", "hm3d"],
        help="Which dataset layout to scan for saved_obs splits.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Train split ratio. When omitted, uses 0.8 for Gibson and 0.9 for HM3D.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
        help="How to partition splits between train and val.",
    )
    parser.add_argument(
        "--split_index_path",
        type=str,
        default="",
        help="Optional JSON split index to reuse deterministic train/val splits (defaults to dataset/splits/*).",
    )
    parser.add_argument(
        "--precomputed_root",
        type=str,
        default="runs/precomputed",
        help="Root directory containing precomputed Zarr shards (set empty to read raw PNGs).",
    )
    parser.add_argument(
        "--max_pairs_per_split",
        type=int,
        default=-1,
        help="Randomly subsample at most this many pairs per scene split for TRAINING (validation keeps all; set <=0 to disable).",
    )
    parser.add_argument(
        "--max_pairs_per_scene",
        type=int,
        default=-1,
        help="[multiview] Max pairs sampled per scene (set <=0 to take all).",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="square",
        choices=["square", "crop", "pad"],
        help="Image resize/padding strategy before feeding VGGT.",
    )
    parser.add_argument(
        "--square_size",
        type=int,
        default=518,
        help="Final (H, W) fed to VGGT when preprocess_mode='square'.",
    )
    parser.add_argument(
        "--aug_pair_permutation_p",
        type=float,
        default=0.0,
        help="Probability of swapping (i, j) order inside a pair (0 disables).",
    )
    parser.add_argument(
        "--aug_pair_keep_ratio",
        type=float,
        default=1.0,
        help="[multiview] Randomly keep this fraction of labeled pairs per scene (1 disables).",
    )
    parser.add_argument(
        "--aug_hflip_p",
        type=float,
        default=0.0,
        help="Per-image horizontal flip probability (0 disables).",
    )
    parser.add_argument(
        "--aug_color_jitter",
        type=float,
        default=0.0,
        help="Color jitter strength for brightness/contrast/saturation; set 0 to disable.",
    )
    parser.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.0,
        help="Stddev for per-pixel Gaussian noise added after preprocessing (0 disables).",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Optional eval batch size; defaults to train batch size when omitted.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Number of image pairs per batch.")
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="DataLoader prefetch_factor when num_workers > 0 (set <=0 to disable).",
    )
    parser.add_argument(
        "--disable_persistent_workers",
        action="store_true",
        help="Disable persistent_workers in DataLoaders (can reduce RAM/CPU pressure).",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for all RNGs.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda, cuda:0, mps, cpu). Auto-detects when omitted.",
    )

    # Feature extraction arguments
    parser.add_argument(
        "--emb_mode",
        type=str,
        default="chunked",
        choices=["avg", "avg_max", "chunked"],
        help="How to pool VGGT tokens into per-image embeddings.",
    )
    parser.add_argument(
        "--token_chunks",
        type=int,
        default=5,
        help="Chunk count for emb_mode='chunked'. Ignored otherwise.",
    )
    parser.add_argument(
        "--layer_mode",
        type=str,
        default="all",
        choices=[
            "all",
            "1st_last",
            "2nd_last",
            "3rd_last",
            "4th_last",
            "last_stages",
            "mid_to_last_stages",
        ],
        help="Which VGGT layers to pass to the head.",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default="facebook/VGGT-1B",
        help="Identifier passed to VGGT.from_pretrained.",
    )
    parser.add_argument(
        "--backbone_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for loading the VGGT backbone weights.",
    )
    parser.add_argument(
        "--token_proj_dim",
        type=int,
        default=256,
        help="Dimensionality of the shared token projector (set <=0 to disable projection).",
    )
    parser.add_argument(
        "--summary_tokens",
        type=int,
        default=8,
        help="Number of learned summary tokens per view used to condense VGGT patches.",
    )
    parser.add_argument(
        "--summary_heads",
        type=int,
        default=4,
        help="Number of attention heads for the token summarizer.",
    )
    parser.add_argument(
        "--mixing_aware",
        type=str,
        default=None,
        choices=["pair", "scene", "both"],
        help="Layer mixing strategy for the scene-aware head (requires --head_type scene_aware).",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="base",
        choices=["base", "scene_aware"],
        help="Head architecture to use on top of VGGT features.",
    )

    # Optimisation arguments
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the head.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping (0 disables).")
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Enable learning rate scheduling (monitors graph_AUC when possible).",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="plateau",
        choices=["step", "plateau"],
        help="Learning rate scheduler type. Only used when --use_scheduler is set.",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=512, help="Hidden size of the MLP head.")
    parser.add_argument("--head_dropout", type=float, default=0.2, help="Dropout probability inside the head.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="Optional limit on the number of training batches per epoch.",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss instead of BCE for the primary classification term.",
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Stop training when graph_AUC does not improve for a fixed patience window (alias --use_ealy_stopping).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Number of epochs without graph_AUC improvement before early stopping (only when enabled).",
    )

    # Logging / checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/vggt_trainer",
        help="Where to store checkpoints and logs.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save an extra checkpoint every N epochs (0 disables periodic saves).",
    )
    parser.add_argument(
        "--save_ckpt_path",
        type=str,
        default=None,
        help="Path to store the best head weights (defaults to runs/precomputed/{mode}/head_ckpt/best_head.pth).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Print training progress every N batches.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Co-Vision-FT",
        help="Weights & Biases project name (set None to disable).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name to group runs.",
    )
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="Disable wandb logging regardless of project/run name.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only train on the training set without running validation.",
    )
    parser.add_argument(
        "--eval_ckpt_model",
        action="store_true",
        help="Run a single evaluation using a provided head checkpoint and exit.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Checkpoint path to load the head weights from when --eval_ckpt_model is set.",
    )
    # Useless but for wandb compatibility
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="No effect; only for wandb compatibility.",
    ) # default=False
    parser.add_argument(
        "--use_triangle_loss",
        action="store_true",
        help="Enable triangle/transitivity regulariser in multiview training.",
    )
    # optionally
    parser.add_argument(
        "--triangle_loss_weight",
        type=float,
        default=0.1,
        help="Weight for the triangle/transitivity loss term.",
    )

    return parser


__all__ = ["build_vggt_trainer_parser"]
