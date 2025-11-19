#!/usr/bin/env python
import os
import sys
import warnings

import torch

# Add the main folder to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.load_dataset_pairs import build_dataloaders_pairs
from models.PairView import EdgeClassifier, GatedLayerFusion, AttentiveLayerFusion
from train.args import build_pairview_parser
from train.trainer import Trainer, infer_embedding_dim
from utils.utils import create_run_logger, set_seed, setup_wandb

warnings.filterwarnings("ignore")


def _default_pairview_split_path(seed: int, dataset_type: str) -> str:
    ratio = 0.8 if dataset_type == "gibson" else 0.9
    split_dir = os.path.join("dataset", "splits", "pairview")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"pairview_{seed}_{ratio}.json")


def _build_pairview_dataloaders(args):
    """Build dataloaders for pair-view training, optionally mixing train/eval datasets."""
    if args.eval_dataset_type is not None and args.eval_dataset_type != args.dataset_type:
        train_dataset_type = args.dataset_type
        eval_dataset_type = args.eval_dataset_type

        train_ratio_train = 0.8 if train_dataset_type == "gibson" else 0.9
        eval_ratio_eval = 0.8 if eval_dataset_type == "gibson" else 0.9

        split_path_train = args.split_index_path or _default_pairview_split_path(
            args.seed, train_dataset_type
        )
        split_path_eval = args.split_index_path or _default_pairview_split_path(
            args.seed, eval_dataset_type
        )

        train_loader, _, train_ds, _, meta_train = build_dataloaders_pairs(
            dataset_type=train_dataset_type,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=4,
            train_ratio=train_ratio_train,
            max_neg_ratio=(args.max_neg_ratio if not args.keep_all_data else -1.0),
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_rel_thr=0.3,
            layer_mode=args.layer_mode,
            split_mode=args.data_split_mode,
            emb_mode=args.emb_mode,
            subset="train",
            split_index_path=split_path_train,
            persist_split_index=True,
        )

        _, val_loader, _, val_ds, meta_eval = build_dataloaders_pairs(
            dataset_type=eval_dataset_type,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=4,
            train_ratio=eval_ratio_eval,
            max_neg_ratio=(args.max_neg_ratio if not args.keep_all_data else -1.0),
            hard_neg_ratio=args.hard_neg_ratio,
            hard_neg_rel_thr=0.3,
            layer_mode=args.layer_mode,
            split_mode=args.data_split_mode,
            emb_mode=args.emb_mode,
            subset="val",
            split_index_path=split_path_eval,
            persist_split_index=True,
        )
        return train_loader, val_loader, train_ds, val_ds, meta_train, meta_eval

    default_train_ratio = 0.8 if args.dataset_type == "gibson" else 0.9
    split_path = args.split_index_path or _default_pairview_split_path(
        args.seed, args.dataset_type
    )

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=4,
        train_ratio=default_train_ratio,
        max_neg_ratio=(args.max_neg_ratio if not args.keep_all_data else -1.0),
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=0.3,
        layer_mode=args.layer_mode,
        split_mode=args.data_split_mode,
        emb_mode=args.emb_mode,
        subset="both",
        split_index_path=split_path,
        persist_split_index=args.persist_split_index,
    )
    return train_loader, val_loader, train_ds, val_ds, meta, None


def main():
    parser = build_pairview_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    logger = create_run_logger(args.out_dir, args.log_file)
    logger.log(f"[INFO] Using device: {device}")

    wandb_run = setup_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        disabled=args.wandb_off,
    )

    (
        train_loader,
        val_loader,
        train_ds,
        val_ds,
        meta_train,
        meta_val,
    ) = _build_pairview_dataloaders(args)

    logger.log(
        f"[INFO] Dataset: train_pairs={len(train_ds)}, "
        f"val_pairs={len(val_ds)}"
    )
    if meta_val is None:
        logger.log(
            f"[INFO] Pair graphs: total={meta_train['num_graphs']}, "
            f"train={meta_train['num_train_graphs']}, val={meta_train['num_val_graphs']}"
        )
        logger.log(
            f"[INFO] Scenes: total={meta_train['num_scenes']}, "
            f"train={meta_train['num_train_scenes']}, val={meta_train['num_val_scenes']}"
        )
    else:
        logger.log(
            f"[INFO] Train graphs: total={meta_train['num_graphs']} | "
            f"train_scenes={meta_train['num_train_scenes']} | "
            f"val_scenes={meta_train['num_val_scenes']}"
        )
        logger.log(
            f"[INFO] Eval graphs: total={meta_val['num_graphs']} | "
            f"train_scenes={meta_val['num_train_scenes']} | "
            f"val_scenes={meta_val['num_val_scenes']}"
        )

    trainer = Trainer(
        args,
        device,
        wandb_run=wandb_run,
        checkpoint_name="edge_classifier_pairs.pth",
        logger=logger,
    )
    try:
        if trainer.run_zero_shot(train_loader, val_loader):
            return

        emb_dim = infer_embedding_dim(train_loader)
        logger.log(f"[INFO] Embedding dim: {emb_dim}")

        if args.head_type == "edge":
            classifier = EdgeClassifier(
                emb_dim=emb_dim,
                hidden_dim=args.hidden_dim,
            ).to(device)
        elif args.head_type == "gated":
            classifier = GatedLayerFusion(
                emb_dim=emb_dim,
                hidden_dim=args.hidden_dim,
                vec_gate=False,
            ).to(device)
        elif args.head_type == "attention":
            classifier = AttentiveLayerFusion(
                emb_dim=emb_dim,
                hidden_dim=args.hidden_dim,
            ).to(device)
        else:
            raise ValueError(
                f"Unknown head_type '{args.head_type}'. Expected 'edge', 'gated', or 'attention'."
            )

        n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        logger.log(f"[INFO] Trainable parameters: {n_params}")

        trainer.fit(classifier, train_loader, val_loader, emb_dim)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
