import argparse


def build_multiview_parser() -> argparse.ArgumentParser:
    """Argument parser for train/vggt_train.py."""
    parser = argparse.ArgumentParser(
        description="Train Co-VGGT on precomputed embeddings/adjacency (CoVisGraphDataset)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for EdgeClassifier MLP.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-5,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of graphs per batch).",
    )
    parser.add_argument(
        "--max_neg_ratio",
        type=float,
        default=1.0,
        help="Max negative:positive ratio for edge pairs.",
    )
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="train/classifier")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0.0 disables clipping).",
    )
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="Co-Vision")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="train.log",
        help="Filename for run logs (placed inside out_dir). Use empty string to disable file logging.",
    )
    # Early stopping + LR scheduler
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without Graph IoU AUC improvement).",
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in val Graph IoU AUC to reset early stopping.",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Factor to reduce LR on plateau (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
        help="LR scheduler patience in epochs (ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "step", "none"],
        help="LR scheduler type. Always monitors validation loss.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=10,
        help="Step size (epochs) for StepLR when --lr_scheduler=step.",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set, run zero-shot evaluation using cosine similarity only (no training).",
    )
    parser.add_argument(
        "--aug_swap_prob",
        type=float,
        default=0.5,
        help="Probability of swapping node pairs as data augmentation during training.",
    )
    parser.add_argument(
        "--use_reg_loss",
        action="store_true",
        help="If set, add auxiliary regression loss to continuous edge strengths (rel_mat).",
    )
    parser.add_argument(
        "--use_low_entropy_loss",
        action="store_true",
        help="If set, add a low-entropy regularization term on layer gates (for attention heads).",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.3,
        help="Weight for the regression loss term.",
    )
    parser.add_argument(
        "--use_rank_loss",
        action="store_true",
        help="If set, add a pairwise ranking loss based on continuous edge strengths.",
    )
    parser.add_argument(
        "--rank_lambda",
        type=float,
        default=0.1,
        help="Weight for the ranking loss term.",
    )
    parser.add_argument(
        "--rank_margin",
        type=float,
        default=0.1,
        help="Margin for the ranking loss on score differences.",
    )
    parser.add_argument(
        "--rank_num_samples",
        type=int,
        default=1024,
        help="Number of pairwise constraints sampled per batch for ranking loss.",
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
        help="Which VGGT layer(s) to use: 'all', one of the last-k layers, or a stage range "
             "('last_stages' = layers 17..end, 'mid_to_last_stages' = layers 12..end).",
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
        help="Split mode: 'scene_disjoint' (default), 'version_disjoint', or 'graph'."
    )
    parser.add_argument(
        "--use_iou_loss",
        action="store_true",
        help="If set, add a batch-level soft IoU loss term."
    )
    parser.add_argument(
        "--iou_lambda",
        type=float,
        default=0.3,
        help="Weight for the soft IoU loss term."
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="If set, use focal BCE loss instead of standard BCE.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha weighting term for focal BCE loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma focusing term for focal BCE loss.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/val split ratio."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gibson",
        choices=["hm3d", "gibson"],
        help="Dataset type / source of embeddings: 'gibson' or 'hm3d'."
    )
    parser.add_argument(
        "--emb_mode",
        type=str,
        default="avg_max",
        choices=["avg", "avg_max", "chunked"],
        help="Which embedding mode to load (used to pick all_embeds_{emb_mode}.npz).",
    )
    parser.add_argument(
        "--keep_all_data",
        action="store_true",
        help="If set, do not subsample negatives (keep all data). Overrides max_neg_ratio.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="edge",
        choices=["edge", "gated", "attention", "attention_entropy"],
        help="Which classifier head to use: 'edge' (EdgeClassifier) or 'gated' (GatedLayerFusion).",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="If set, save training/validation plots to out_dir.",
    )
    parser.add_argument("--split_index_path", type=str, default=None,
                        help="Optional path to a persisted meta-level split index (.json).")
    parser.add_argument("--persist_split_index", action="store_true",
                        help="If set (and a split path is given/derived), write the split file if missing.")
    return parser


def build_pairview_parser() -> argparse.ArgumentParser:
    """Argument parser for train/vggt_train_pairs.py."""
    parser = argparse.ArgumentParser(
        description="Train EdgeClassifier on pair-level VGGT embeddings (pairs.npz)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="train/classifier_pairs")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
    )
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="Co-Vision")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_off",
        action="store_true",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="train.log",
        help="Filename for run logs (placed inside out_dir). Use empty string to disable file logging.",
    )
    # Early stopping + LR scheduler
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "step", "none"],
        help="LR scheduler type. Always monitors validation loss.",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=10,
        help="Step size (epochs) for StepLR when --lr_scheduler=step.",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set, run zero-shot eval (cosine similarity) only.",
    )
    parser.add_argument(
        "--aug_swap_prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--use_reg_loss",
        action="store_true",
    )
    parser.add_argument(
        "--use_low_entropy_loss",
        action="store_true",
        help="If set, add a low-entropy regularization term on layer gates (for attention heads).",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--use_rank_loss",
        action="store_true",
    )
    parser.add_argument(
        "--rank_lambda",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rank_margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rank_num_samples",
        type=int,
        default=1024,
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
        help="Which VGGT layer(s) to use: 'all', one of the last-k layers, or a stage range "
             "('last_stages' = layers 17..end, 'mid_to_last_stages' = layers 12..end).",
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="scene_disjoint",
        choices=["scene_disjoint", "version_disjoint", "graph"],
    )
    parser.add_argument(
        "--use_iou_loss",
        action="store_true",
    )
    parser.add_argument(
        "--iou_lambda",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="If set, use focal BCE loss instead of standard BCE.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha weighting term for focal BCE loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma focusing term for focal BCE loss.",
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
            help="Embedding aggregation mode.",
    )
    parser.add_argument(
        "--keep_all_data",
        action="store_true",
        help="If set, do not subsample negatives (keep all data). Overrides max_neg_ratio.",
    )
    parser.add_argument(
        "--hard_neg_ratio",
        type=float,
        default=0.5,
        help="Fraction of sampled negatives that are hard.",
    )
    parser.add_argument(
        "--max_neg_ratio",
        type=float,
        default=1.0,
        help="Maximum number of negatives to keep per positive (ratio). Use <=0 to keep all negatives.",
    )
    parser.add_argument(
        "--eval_dataset_type",
        type=str,
        default=None,
        choices=["hm3d", "gibson"],
        help=(
            "If set (and different from --dataset_type), train on --dataset_type "
            "and evaluate on this dataset type using a held-out split."
        ),
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="edge",
        choices=["edge", "gated", "attention", "weighted_edge"],
        help="Which classifier head to use: 'edge' (EdgeClassifier) or 'gated' (GatedLayerFusion).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for EdgeClassifier MLP.",
    )
    parser.add_argument("--split_index_path", type=str, default=None,
                        help="Optional path to a persisted meta-level split index (.json).")
    parser.add_argument("--persist_split_index", action="store_true",
                        help="If set (and a split path is given/derived), write the split file if missing.")
    return parser
