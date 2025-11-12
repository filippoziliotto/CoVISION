import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Edge classifier head (trainable)
# ---------------------------------------------------------------------
class EdgeClassifier(nn.Module):
    """
    Takes two node embeddings e_i, e_j (E-dim each) and predicts edge strength (logit).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout_p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(p=dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Dropout(p=dropout_p),
        )
        self.layernorm = nn.LayerNorm(4 * emb_dim)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        emb_i, emb_j:
            - shape (B, E)   for single-layer embeddings
            - or    (B, L, E) for multi-layer embeddings (L layers)

        Returns:
            - shape (B,) edge scores (logits)
        """
        if emb_i.ndim == 2:
            # Single-layer case: (B, E)
            x = torch.cat(
                [emb_i, emb_j, torch.abs(emb_i - emb_j), emb_i * emb_j],
                dim=-1,
            )  # (B, 4E)
            x = self.layernorm(x)
            out = self.mlp(x).squeeze(-1)  # (B,)
            return out

        elif emb_i.ndim == 3:
            # Multi-layer case: (B, L, E)
            B, L, E = emb_i.shape
            emb_i_flat = emb_i.reshape(B * L, E)
            emb_j_flat = emb_j.reshape(B * L, E)

            x = torch.cat(
                [
                    emb_i_flat,
                    emb_j_flat,
                    torch.abs(emb_i_flat - emb_j_flat),
                    emb_i_flat * emb_j_flat,
                ],
                dim=-1,
            )  # (B*L, 4E)
            x = self.layernorm(x)
            out_flat = self.mlp(x).squeeze(-1)  # (B*L,)

            # Reshape back to (B, L) and average scores over layers
            out = out_flat.view(B, L).mean(dim=1)  # (B,)
            return out

        else:
            raise ValueError(
                f"EdgeClassifier expected emb_i with ndim 2 or 3, got shape={emb_i.shape}"
            )


# ---------------------------------------------------------------------
# MultiLayerEdgeClassifier: uses first, middle, last VGGT layers
# ---------------------------------------------------------------------
class MultiLayerEdgeClassifier(nn.Module):
    """
    Edge classifier that explicitly uses first, middle, and last VGGT layers,
    with separate MLPs for low/mid/high features and a learned fusion before final prediction.

    Inputs:
        emb_i, emb_j:
            - (B, E)       for single-layer embeddings
            - (B, L, E)    for multi-layer embeddings (L layers)
    Output:
        - (B,) logits for edge presence/strength
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout_p: float = 0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        # Separate layer norms for low/mid/high branches
        self.pair_ln_low = nn.LayerNorm(4 * emb_dim)
        self.pair_ln_mid = nn.LayerNorm(4 * emb_dim)
        self.pair_ln_high = nn.LayerNorm(4 * emb_dim)

        # Shared MLP definition helper
        def make_pair_mlp():
            return nn.Sequential(
                nn.Linear(4 * emb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
            )

        # Three separate MLPs for low/mid/high-level pair features
        self.pair_mlp_low = make_pair_mlp()
        self.pair_mlp_mid = make_pair_mlp()
        self.pair_mlp_high = make_pair_mlp()

        # Fusion MLP to combine low/mid/high features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * (hidden_dim // 2), hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

        # Final classifier head
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_pair(self, e_i: torch.Tensor, e_j: torch.Tensor, ln: nn.Module, mlp: nn.Module) -> torch.Tensor:
        """Encode a single pair of embeddings (B, E) → (B, hidden_dim // 2)."""
        x = torch.cat(
            [e_i, e_j, torch.abs(e_i - e_j), e_i * e_j],
            dim=-1,
        )  # (B, 4E)
        x = ln(x)
        h = mlp(x)
        return h

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        # Single-layer case
        if emb_i.ndim == 2:
            h_single = self._encode_pair(emb_i, emb_j, self.pair_ln_mid, self.pair_mlp_mid)
            logits = self.final_mlp(torch.cat([h_single, h_single, h_single], dim=-1))  # mimic 3-branch fusion
            return logits.squeeze(-1)

        # Multi-layer case
        if emb_i.ndim != 3 or emb_j.ndim != 3:
            raise ValueError(
                f"MultiLayerEdgeClassifier expects emb_i, emb_j with ndim 2 or 3, got {emb_i.shape}, {emb_j.shape}"
            )

        B, L, E = emb_i.shape
        if E != self.emb_dim:
            raise ValueError(f"Expected embedding dim={self.emb_dim}, got {E} (shape={emb_i.shape})")

        # Pick representative layers
        idx_first = 0
        idx_mid = L // 2
        idx_last = L - 1

        # Extract layer embeddings
        e_i_first, e_j_first = emb_i[:, idx_first, :], emb_j[:, idx_first, :]
        e_i_mid, e_j_mid = emb_i[:, idx_mid, :], emb_j[:, idx_mid, :]
        e_i_last, e_j_last = emb_i[:, idx_last, :], emb_j[:, idx_last, :]

        # Encode pairs at each scale
        h_first = self._encode_pair(e_i_first, e_j_first, self.pair_ln_low, self.pair_mlp_low)
        h_mid = self._encode_pair(e_i_mid, e_j_mid, self.pair_ln_mid, self.pair_mlp_mid)
        h_last = self._encode_pair(e_i_last, e_j_last, self.pair_ln_high, self.pair_mlp_high)

        # Fuse features (learned combination instead of mean)
        h_cat = torch.cat([h_first, h_mid, h_last], dim=-1)  # (B, 3 * hidden_dim // 2)
        h_fused = self.fusion_mlp(h_cat)

        logits = self.final_mlp(h_fused).squeeze(-1)  # (B,)
        return logits
