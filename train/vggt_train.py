#!/usr/bin/env python
import os
import sys
import warnings

import torch

# Add the main folder to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.load_dataset import build_dataloaders
from models.MultiView import EdgeClassifier, GatedLayerFusion
from train.args import build_multiview_parser
from train.trainer import Trainer, infer_embedding_dim
from utils.utils import create_run_logger, set_seed, setup_wandb

warnings.filterwarnings("ignore")


def _default_multiview_split_path(seed: int, dataset_type: str) -> str:
    ratio = 0.8 if dataset_type == "gibson" else 0.9
    split_dir = os.path.join("dataset", "splits", "multiview")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"multiview_{seed}_{ratio}.json")


def main():
    parser = build_multiview_parser()
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

    effective_train_ratio = 0.8 if args.dataset_type == "gibson" else 0.9
    split_path = args.split_index_path or _default_multiview_split_path(
        args.seed, args.dataset_type
    )

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=4,
        train_ratio=effective_train_ratio,
        seed=args.seed,
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

    logger.log(
        f"[INFO] Dataset: train_pairs={len(train_ds)}, "
        f"val_pairs={len(val_ds)}"
    )
    logger.log(
        f"[INFO] Graphs: total={meta['num_graphs']}, "
        f"train={meta['num_train_graphs']}, val={meta['num_val_graphs']}"
    )
    logger.log(
        f"[INFO] Scenes: total={meta['num_scenes']}, "
        f"train={meta['num_train_scenes']}, val={meta['num_val_scenes']}"
    )
    if "emb_mode" in meta:
        logger.log(f"[INFO] Embedding mode: {meta['emb_mode']}")

    trainer = Trainer(
        args,
        device,
        wandb_run=wandb_run,
        checkpoint_name="edge_classifier.pth",
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
        else:
            raise ValueError(f"Unknown head_type '{args.head_type}'. Expected 'edge' or 'gated'.")

        n_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        logger.log(f"[INFO] Trainable parameters: {n_params}")

        trainer.fit(classifier, train_loader, val_loader, emb_dim)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
