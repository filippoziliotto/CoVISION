import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
from typing import List


def _default_multiview_split_path(seed: int, dataset_type: str, train_ratio: float) -> str:
    """Return a dataset-specific split file to avoid mixing gibson/hm3d splits."""
    split_dir = os.path.join("dataset", "splits", "multiview")
    os.makedirs(split_dir, exist_ok=True)
    ratio_str = f"{train_ratio:.2f}"
    return os.path.join(split_dir, f"multiview_{dataset_type}_{seed}_{ratio_str}.json")


# ---------------------------------------------------------
# Small MLP probe for co-visibility classification
# ---------------------------------------------------------
class CoVisProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
        )

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pair_feat))


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def build_pair_features(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Concatenate two feature vectors along the last dim."""
    return torch.cat([feat_a, feat_b], dim=-1)


# ---------------------------------------------------------
# Layer-wise Probe Trainer
# ---------------------------------------------------------
class LayerwiseProbeTrainer:
    def __init__(self, embed_dim: int, num_layers: int, lr: float = 1e-3, device: str = "cuda"):
        self.device = device
        self.num_layers = num_layers

        # One probe per layer
        self.probes = nn.ModuleList([CoVisProbe(embed_dim).to(device) for _ in range(num_layers)])
        self.opt = torch.optim.Adam(self.probes.parameters(), lr=lr)
        self.bce = torch.nn.BCELoss()

    def train_step(self, features_all: List[torch.Tensor], labels: torch.Tensor) -> float:
        """Train all probes on a batch and return summed loss."""
        total_loss = 0.0
        self.opt.zero_grad()

        for layer_idx, feats in enumerate(features_all):
            # feats: [B, 2, 1, E] -> squeeze the singleton patch dim
            feat_a = feats[:, 0, 0, :]  # [B, E]
            feat_b = feats[:, 1, 0, :]  # [B, E]
            pair_feat = build_pair_features(feat_a, feat_b)
            pred = self.probes[layer_idx](pair_feat).squeeze(-1)  # [B]
            loss = self.bce(pred, labels.float())
            total_loss += loss

        total_loss.backward()
        self.opt.step()
        return float(total_loss.item())

    @torch.no_grad()
    def evaluate(self, features_all: List[torch.Tensor], labels: torch.Tensor) -> List[float]:
        accuracies = []
        for layer_idx, feats in enumerate(features_all):
            feat_a = feats[:, 0, 0, :]
            feat_b = feats[:, 1, 0, :]
            pair_feat = build_pair_features(feat_a, feat_b)
            pred = self.probes[layer_idx](pair_feat).squeeze(-1)
            pred_binary = (pred > 0.5).long()
            acc = (pred_binary == labels).float().mean().item()
            accuracies.append(acc)
        return accuracies


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _pair_batch_to_layer_list(feat_i: torch.Tensor, feat_j: torch.Tensor) -> List[torch.Tensor]:
    """Convert [B, L, E] pair embeddings into per-layer [B, 2, 1, E] list."""
    if feat_i.shape != feat_j.shape:
        raise ValueError(f"feat_i and feat_j shapes differ: {feat_i.shape} vs {feat_j.shape}")
    if feat_i.dim() != 3:
        raise ValueError(f"Expected feat_i dim=3 ([B, L, E]), got {feat_i.shape}")
    _, layers, _ = feat_i.shape
    return [
        torch.stack([feat_i[:, l, :], feat_j[:, l, :]], dim=1).unsqueeze(2)
        for l in range(layers)
    ]


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_layerwise_probes(
    trainer: LayerwiseProbeTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 10,
    device: str = "cuda",
):
    num_layers = trainer.num_layers

    for epoch in range(1, num_epochs + 1):
        trainer.probes.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # batch: (feat_i, feat_j, lbl, strength)
            feat_i, feat_j, lbl, _ = batch
            feat_i = feat_i.to(device)
            feat_j = feat_j.to(device)
            labels = lbl.to(device)

            features_all = _pair_batch_to_layer_list(feat_i, feat_j)
            batch_size = labels.size(0)
            loss = trainer.train_step(features_all, labels)
            epoch_loss += loss * batch_size
            total_samples += batch_size

        avg_loss = epoch_loss / max(1, total_samples)
        print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")

        if val_loader is not None and len(val_loader) > 0:
            trainer.probes.eval()
            all_preds_per_layer = [[] for _ in range(num_layers)]
            all_labels_per_layer = [[] for _ in range(num_layers)]
            with torch.no_grad():
                for batch in val_loader:
                    feat_i, feat_j, lbl, _ = batch
                    feat_i = feat_i.to(device)
                    feat_j = feat_j.to(device)
                    labels = lbl.to(device)
                    features_all = _pair_batch_to_layer_list(feat_i, feat_j)
                    for layer_idx in range(num_layers):
                        feats = features_all[layer_idx]  # [B, 2, 1, E]
                        feat_a = feats[:, 0, 0, :]
                        feat_b = feats[:, 1, 0, :]
                        pair_feat = build_pair_features(feat_a, feat_b)
                        preds = trainer.probes[layer_idx](pair_feat).squeeze(-1)
                        all_preds_per_layer[layer_idx].append(preds.detach().cpu())
                        all_labels_per_layer[layer_idx].append(labels.detach().cpu())
            for layer_idx in range(num_layers):
                probs = torch.cat(all_preds_per_layer[layer_idx], dim=0).numpy()
                labels_all = torch.cat(all_labels_per_layer[layer_idx], dim=0).numpy()
                y_true_bin = (labels_all >= 0.5).astype(np.float32)
                y_pred_bin = (probs >= 0.5).astype(np.float32)
                acc = (y_pred_bin == y_true_bin).mean()
                thresholds = np.linspace(0.0, 1.0, 100)
                ious = []
                for t in thresholds:
                    pred_t = (probs >= t).astype(np.float32)
                    inter = np.logical_and(pred_t, y_true_bin).sum()
                    union = np.logical_or(pred_t, y_true_bin).sum()
                    iou = inter / (union + 1e-6)
                    ious.append(iou)
                best_iou = float(np.max(ious))
                best_thr = float(thresholds[int(np.argmax(ious))])
                graph_iou_auc = float(np.trapz(ious, thresholds))
                print(f"[Epoch {epoch}] Layer {layer_idx}: val_acc={acc:.3f}, val_graph_iou_best={best_iou:.3f}, val_graph_iou_auc={graph_iou_auc:.3f}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train layerwise probes on multiview embeddings.")
    parser.add_argument("--dataset_type", type=str, default="gibson", choices=["gibson", "hm3d"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_neg_ratio", type=float, default=1.0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--hard_neg_rel_thr", type=float, default=0.3)
    parser.add_argument("--split_mode", type=str, default="scene_disjoint",
                        choices=["scene_disjoint", "version_disjoint", "graph"])
    parser.add_argument("--emb_mode", type=str, default="avg_max",
                        choices=["avg", "avg_max", "chunked"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split_index_path", type=str, default="")
    parser.add_argument("--persist_split_index", action="store_true")
    args = parser.parse_args()

    # Add repo root to sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    from dataset.load_dataset import build_dataloaders

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )
    train_ratio = args.train_ratio if args.train_ratio is not None else (
        0.8 if args.dataset_type == "gibson" else 0.9
    )
    split_path = args.split_index_path or _default_multiview_split_path(args.seed, args.dataset_type, train_ratio)

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=train_ratio,
        seed=args.seed,
        max_neg_ratio=args.max_neg_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode="all",
        split_mode=args.split_mode,
        emb_mode=args.emb_mode,
        subset="both",
        split_index_path=split_path,
        persist_split_index=args.persist_split_index,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check split index path or embedding files.")

    # Infer num_layers and embed_dim from a sample
    sample_feat_i, _, _, _ = train_ds[0]
    num_layers = sample_feat_i.shape[0]
    embed_dim = sample_feat_i.shape[1]

    trainer = LayerwiseProbeTrainer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        lr=1e-3,
        device=device,
    )

    train_layerwise_probes(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
