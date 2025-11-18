import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
from typing import List, Tuple


def _default_pair_split_path(seed: int, dataset_type: str, train_ratio: float) -> str:
    """Return a dataset-specific split file to avoid mixing gibson/hm3d splits."""
    split_dir = os.path.join("dataset", "splits", "pairview")
    os.makedirs(split_dir, exist_ok=True)
    ratio_str = f"{train_ratio:.2f}"
    return os.path.join(split_dir, f"pairview_{dataset_type}_{seed}_{ratio_str}.json")


# ---------------------------------------------------------
# Small MLP probe for co-visibility classification
# ---------------------------------------------------------
class CoVisProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, pair_feat):
        return torch.sigmoid(self.mlp(pair_feat))


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def pool_tokens(x, patch_start_idx):
    """
    x : [B, S, P, C]
    Extract pooled token features for each image:
        - average over patch tokens only (ignore camera + register tokens)
    Returns: [B, S, C]
    """
    # remove camera + register tokens (first N tokens)
    x = x[:, :, patch_start_idx:, :]                 # [B, S, P-patch_start, C]
    feat = x.mean(dim=2)                             # avg over tokens => [B, S, C]
    return feat


def build_pair_features(feat_A, feat_B):
    """
    feat_A, feat_B: [B, C]
    Return concatenated pair features [B, 2C]
    """
    return torch.cat([feat_A, feat_B], dim=-1)


# ---------------------------------------------------------
# Layer-wise Probe Trainer
# ---------------------------------------------------------
class LayerwiseProbeTrainer:

    def __init__(self, embed_dim: int, num_layers: int, patch_start_idx: int, lr=1e-3, device="cuda"):
        """
        embed_dim: size of VGGT token dimension
        num_layers: number of layers/output blocks in aggregated_tokens_list
        """
        self.device = device
        self.num_layers = num_layers
        self.patch_start_idx = patch_start_idx

        # One probe per layer
        self.probes = nn.ModuleList([CoVisProbe(embed_dim).to(device) for _ in range(num_layers)])
        self.opt = torch.optim.Adam(self.probes.parameters(), lr=lr)
        self.bce = torch.nn.BCELoss()

    # -----------------------------------------------------

    def train_step(self, features_all: List[torch.Tensor], labels: torch.Tensor):
        """
        features_all: list of length L
            each element is [B, S, P, 2C] because VGGT outputs *concat_inter* = [frame | global]
        labels: [B]  ∈ {0,1} co-vis labels
        """

        B = labels.shape[0]
        total_loss = 0

        self.opt.zero_grad()

        # Loop over layers
        for layer_idx, feats in enumerate(features_all):
            # feats is [B, S, P, 2C]
            # But we only need patch tokens of each image pair

            # --- For Co-VisiON: S=2 always ---
            # Use slice with singleton dimension, then squeeze after pooling
            feat_A = pool_tokens(feats[:, 0:1, ...], self.patch_start_idx).squeeze(1)   # [B, C]
            feat_B = pool_tokens(feats[:, 1:2, ...], self.patch_start_idx).squeeze(1)   # [B, C]

            pair_feat = build_pair_features(feat_A, feat_B)           # [B, 2C]

            pred = self.probes[layer_idx](pair_feat).squeeze(-1)      # [B]
            loss = self.bce(pred, labels.float())
            total_loss += loss

        total_loss.backward()
        self.opt.step()

        return total_loss.item()

    # -----------------------------------------------------

    @torch.no_grad()
    def evaluate(self, features_all: List[torch.Tensor], labels: torch.Tensor):
        """
        Return per-layer AUC-like (simple accuracy)
        """
        B = labels.shape[0]
        accuracies = []

        for layer_idx, feats in enumerate(features_all):
            feat_A = pool_tokens(feats[:, 0:1, ...], self.patch_start_idx).squeeze(1)
            feat_B = pool_tokens(feats[:, 1:2, ...], self.patch_start_idx).squeeze(1)

            pair_feat = build_pair_features(feat_A, feat_B)
            pred = self.probes[layer_idx](pair_feat).squeeze(-1)

            pred_binary = (pred > 0.5).long()
            acc = (pred_binary == labels).float().mean().item()
            accuracies.append(acc)

        return accuracies

# ---------------------------------------------------------
# Training loop for layer-wise probes
# ---------------------------------------------------------
def train_layerwise_probes(
    trainer: LayerwiseProbeTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 10,
    device: str = "cuda",
):
    """Simple training loop for layer-wise co-visibility probes.

    Expects each batch from the dataloaders to be of the form:
        (features_all, labels)

    where:
        - features_all is either:
            * a list/tuple of length L with tensors of shape [B, S, P, C_concat], or
            * a tensor of shape [L, B, S, P, C_concat] or [B, L, S, P, C_concat]
        - labels is a tensor of shape [B]

    The function trains all layer-wise probes jointly using trainer.train_step
    and, if val_loader is provided, prints per-layer accuracies at each epoch.
    """
    num_layers = trainer.num_layers

    def _to_layer_list(features_batch):
        # Case 1: already a list/tuple of tensors [B, S, P, C]
        if isinstance(features_batch, (list, tuple)):
            return [f.to(device) for f in features_batch]

        # Case 2: single tensor, try to interpret its dimensions
        if isinstance(features_batch, torch.Tensor):
            if features_batch.dim() == 5:
                # Possible shapes:
                #   [L, B, S, P, C] or [B, L, S, P, C]
                if features_batch.shape[0] == num_layers:
                    # [L, B, S, P, C] -> list over L
                    return [features_batch[l].to(device) for l in range(num_layers)]
                elif features_batch.shape[1] == num_layers:
                    # [B, L, S, P, C] -> list over L
                    return [features_batch[:, l].to(device) for l in range(num_layers)]
                else:
                    raise ValueError(
                        f"Unexpected features shape {features_batch.shape} for num_layers={num_layers}."
                    )
            else:
                raise ValueError(
                    f"Expected 5D tensor for features_batch, got shape {features_batch.shape}."
                )

        raise TypeError(
            f"Unsupported type for features_batch: {type(features_batch)}."
        )

    for epoch in range(1, num_epochs + 1):
        trainer.probes.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            if len(batch) != 2:
                raise ValueError(
                    "Each batch must be (features_all, labels) for probe training."
                )
            features_all, labels = batch
            labels = labels.to(device)

            feats_list = _to_layer_list(features_all)

            batch_size = labels.size(0)
            loss = trainer.train_step(feats_list, labels)
            epoch_loss += loss * batch_size
            total_samples += batch_size

        avg_loss = epoch_loss / max(1, total_samples)
        print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")

        if val_loader is not None:
            trainer.probes.eval()
            layer_acc_sums = [0.0 for _ in range(num_layers)]
            val_samples = 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) != 2:
                        raise ValueError(
                            "Each batch must be (features_all, labels) for probe evaluation."
                        )
                    features_all, labels = batch
                    labels = labels.to(device)
                    feats_list = _to_layer_list(features_all)

                    batch_size = labels.size(0)
                    accs = trainer.evaluate(feats_list, labels)  # list of len L

                    for i, acc in enumerate(accs):
                        layer_acc_sums[i] += acc * batch_size
                    val_samples += batch_size

            layer_accs = [s / max(1, val_samples) for s in layer_acc_sums]
            acc_str = " ".join(f"L{i}: {a:.3f}" for i, a in enumerate(layer_accs))
            print(f"[Epoch {epoch}] Val per-layer acc: {acc_str}")


# ---------------------------------------------------------
# Main function for command-line training of layerwise probes
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train layerwise probes on pair embeddings.")
    parser.add_argument("--dataset_type", type=str, default="gibson", choices=["gibson", "hm3d"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_neg_ratio", type=float, default=1.0)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5)
    parser.add_argument("--hard_neg_rel_thr", type=float, default=0.3)
    parser.add_argument("--split_mode", type=str, default="scene_disjoint",
                        choices=["scene_disjoint", "version_disjoint", "graph"])
    parser.add_argument("--emb_mode", type=str, default="avg",
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

    # Import build_dataloaders_pairs here after sys.path update
    from dataset.load_dataset_pairs import build_dataloaders_pairs

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else 
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
        "cpu"
    )
    train_ratio = args.train_ratio if args.train_ratio is not None else (
        0.8 if args.dataset_type == "gibson" else 0.9
    )
    split_path = args.split_index_path or _default_pair_split_path(args.seed, args.dataset_type, train_ratio)

    train_loader, val_loader, train_ds, val_ds, meta = build_dataloaders_pairs(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        train_ratio=train_ratio,
        max_neg_ratio=args.max_neg_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_rel_thr=args.hard_neg_rel_thr,
        layer_mode="all",
        split_mode=args.split_mode,
        emb_mode=args.emb_mode,
        split_index_path=split_path,
        persist_split_index=args.persist_split_index,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check split index path or pair embedding files.")

    # Infer num_layers and embed_dim from a sample
    sample_feat_i, _, _, _ = train_ds[0]
    num_layers = sample_feat_i.shape[0]
    embed_dim = sample_feat_i.shape[1]

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

    trainer = LayerwiseProbeTrainer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        patch_start_idx=0,
        lr=1e-3,
        device=device,
    )

    for epoch in range(1, args.num_epochs + 1):
        trainer.probes.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # Each batch: (feat_i, feat_j, lbl, strength)
            feat_i, feat_j, lbl, strength = batch
            feat_i = feat_i.to(device)
            feat_j = feat_j.to(device)
            labels = lbl.to(device)
            batch_size = feat_i.size(0)
            features_all = _pair_batch_to_layer_list(feat_i, feat_j)
            loss = trainer.train_step(features_all, labels)
            epoch_loss += loss * batch_size
            total_samples += batch_size

        avg_loss = epoch_loss / max(1, total_samples)
        print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None and len(val_loader) > 0:
            trainer.probes.eval()
            layer_acc_sums = [0.0 for _ in range(num_layers)]
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    feat_i, feat_j, lbl, strength = batch
                    feat_i = feat_i.to(device)
                    feat_j = feat_j.to(device)
                    labels = lbl.to(device)
                    batch_size = feat_i.size(0)
                    features_all = _pair_batch_to_layer_list(feat_i, feat_j)
                    accs = trainer.evaluate(features_all, labels)
                    for i, acc in enumerate(accs):
                        layer_acc_sums[i] += acc * batch_size
                    val_samples += batch_size
            layer_accs = [s / max(1, val_samples) for s in layer_acc_sums]
            acc_str = " ".join(f"L{i}: {a:.3f}" for i, a in enumerate(layer_accs))
            print(f"[Epoch {epoch}] Val per-layer acc: {acc_str}")


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
