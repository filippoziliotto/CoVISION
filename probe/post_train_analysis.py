import os
import sys
from typing import List
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assume other imports and definitions here...

def train_layerwise_probes(
    trainer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    num_epochs: int,
) -> dict | None:
    best_val_loss = float("inf")
    best_epoch = -1
    best_layer_best_iou = None
    best_layer_auc = None
    best_state_dict = None

    for epoch in range(num_epochs):
        trainer.probes.train()
        for batch in train_loader:
            # Training step here...
            pass

        if val_loader is not None:
            trainer.probes.eval()
            val_loss_sum = 0.0
            val_total = 0
            layer_best_iou = []
            layer_graph_iou_auc = []

            num_layers = trainer.probes.num_layers

            with torch.no_grad():
                for batch in val_loader:
                    pair_feat = batch["pair_feat"].to(device)
                    labels = batch["labels"].to(device)

                    for layer_idx in range(num_layers):
                        preds = trainer.probes[layer_idx](pair_feat).squeeze(-1)
                        loss = trainer.bce(preds, labels.float())
                        val_loss_sum += loss.item() * labels.size(0)

                    val_total += labels.size(0)

            val_loss = val_loss_sum / max(1, val_total * num_layers)

            # Collect per-layer metrics (assumed to be computed by trainer)
            for layer_idx in range(num_layers):
                best_iou = trainer.best_iou_per_layer[layer_idx]
                graph_iou_auc = trainer.graph_iou_auc_per_layer[layer_idx]
                layer_best_iou.append(best_iou)
                layer_graph_iou_auc.append(graph_iou_auc)

            layer_best_iou = np.array(layer_best_iou)
            layer_graph_iou_auc = np.array(layer_graph_iou_auc)

            print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            for layer_idx in range(num_layers):
                print(
                    f"Layer {layer_idx}: best_iou={layer_best_iou[layer_idx]:.4f}, "
                    f"graph_iou_auc={layer_graph_iou_auc[layer_idx]:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_layer_best_iou = layer_best_iou.copy()
                best_layer_auc = layer_graph_iou_auc.copy()
                best_state_dict = deepcopy(trainer.probes.state_dict())
                print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if best_state_dict is not None:
        trainer.probes.load_state_dict(best_state_dict)

    if val_loader is not None and best_epoch >= 0:
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_layer_best_iou": best_layer_best_iou,
            "best_layer_auc": best_layer_auc,
        }
    else:
        return None


def main():
    # Setup code here...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = None  # Assume initialized
    val_loader = None    # Assume initialized
    trainer = None       # Assume initialized
    num_epochs = 10      # Assume set
    output_dir = "./output"  # Assume set

    train_stats = train_layerwise_probes(
        trainer, train_loader, val_loader, device, num_epochs
    )

    # Save final probes weights
    probes_path = os.path.join(output_dir, "probes.pth")
    torch.save(trainer.probes.state_dict(), probes_path)
    print(f"Saved probes weights to {probes_path}")

    # Save predictions
    val_predictions_path = os.path.join(output_dir, "val_predictions.npz")
    # Assume val_predictions saved here...
    print(f"Saved validation predictions to {val_predictions_path}")

    if train_stats is not None:
        best_weights_path = os.path.join(output_dir, "best_layer_weights.npz")
        np.savez_compressed(
            best_weights_path,
            best_epoch=train_stats["best_epoch"],
            best_val_loss=train_stats["best_val_loss"],
            best_layer_best_iou=train_stats["best_layer_best_iou"],
            best_layer_auc=train_stats["best_layer_auc"],
        )
        print(f"Saved best layer weights and metrics to {best_weights_path}")
