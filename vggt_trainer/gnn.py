#!/usr/bin/env python
"""
Simple Graph Transformer module for refining per-view embeddings.

This module is intentionally lightweight and self-contained so it can be
plugged into VGGTHeadModel without requiring changes to the backbone or
the head.

The idea:
    - Each view embedding is treated as a "node token".
    - A TransformerEncoder is applied over the S node tokens.
    - This implicitly performs dense message passing, i.e., a fully-connected graph.
    - The refined embeddings can then be fed into the existing pairwise head.

Usage:
    from vggt_trainer.gnn import GraphTransformer

    gnn = GraphTransformer(
        emb_dim=512,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
    )

    refined = gnn(views)          # (B, S, D)
"""

import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    """
    Transformer-based graph reasoning.

    Input:
        x: Tensor of shape (B, S, D)
            B = batch size
            S = number of views (graph nodes)
            D = embedding dimension

    Output:
        out: Tensor of shape (B, S, D) with refined per-view embeddings.

    Why it works:
        - Self-attention computes attention weights between all node pairs.
        - This is equivalent to message passing on a fully-connected graph.
        - Stronger than a simple GNN because multi-head attention supports
          richer interactions and dynamic adjacency.

    This module is deliberately small so that training the head (and this
    transformer) remains fast on top of frozen VGGT features.
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        hidden_dim = int(emb_dim * mlp_ratio)

        # Single TransformerEncoderLayer: multi-head attention + MLP
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,       # critical: input shape is (B, S, D)
            norm_first=True,
        )

        # Stack num_layers identical layers
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) view embeddings

        Returns:
            (B, S, D) refined embeddings
        """
        if x.dim() != 3:
            raise ValueError(
                f"GraphTransformer expects input (B, S, D), received {x.shape}"
            )

        return self.encoder(x)


__all__ = ["GraphTransformer"]
