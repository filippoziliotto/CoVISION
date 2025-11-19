import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Edge classifier head (trainable)
# ---------------------------------------------------------------------
class EdgeClassifier(nn.Module):
    """
    Takes two node embeddings e_i, e_j (E-dim each) and predicts edge strength in [0,1].
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
            #nn.Sigmoid(),  # output in [0,1]
        )
        self.layernorm = nn.LayerNorm(4 * emb_dim)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor):
        """
        emb_i, emb_j:
            - shape (B, E)  for single-layer embeddings
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
            return {"logits": out}

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
            return {"logits": out}

        else:
            raise ValueError(
                f"EdgeClassifier expected emb_i with ndim 2 or 3, got shape={emb_i.shape}"
            )


# ---------------------------------------------------------------------
# GatedLayerFusionClassifier: uses first, middle, last VGGT layers
# ---------------------------------------------------------------------
class GatedLayerFusion(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, vec_gate=False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(4*emb_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.gate = nn.Linear(4*emb_dim, hidden_dim if vec_gate else 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def encode_pair(self, ei, ej):
        x = torch.cat([ei, ej, (ei-ej).abs(), ei*ej], dim=-1)
        h = self.enc(x)
        g = torch.sigmoid(self.gate(x))     # (B,1) or (B,H)
        return h, g

    def forward(self, emb_i, emb_j):
        if emb_i.ndim == 2:  # single-layer
            h, _ = self.encode_pair(emb_i, emb_j)
            logits = self.head(self.norm(h)).squeeze(-1)
            return {"logits": logits}

        B, L, E = emb_i.shape
        hs, gs = [], []
        for l in range(L):
            h, g = self.encode_pair(emb_i[:,l], emb_j[:,l])
            hs.append(h); gs.append(g)
        H = torch.stack(hs, dim=1)          # (B,L,H)
        G = torch.stack(gs, dim=1)          # (B,L,1 or H)
        if G.dim()==3 and G.size(-1)==1:    # scalar gate
            G = G.expand_as(H)
        hbar = (G * H).sum(dim=1) / (G.sum(dim=1) + 1e-6)
        logits = self.head(self.norm(hbar)).squeeze(-1)
        return {"logits": logits}


import torch
import torch.nn as nn

class AttentiveLayerFusion(nn.Module):
    """
    Improved fusion head for pairwise embeddings.
    Input:
        emb_i, emb_j either (B, E) or (B, L, E)
    Output:
        logits (B,)
    """
    def __init__(self, emb_dim, hidden_dim=256, vec_gate=False, num_heads=4, dropout_p=0.1):
        super().__init__()
        D = 4 * emb_dim  # concat [ei, ej, |ei-ej|, ei*ej]

        # Per-layer pair encoder
        self.pair_ln = nn.LayerNorm(D)
        self.pair_ff = nn.Sequential(
            nn.Linear(D, hidden_dim), nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )

        # Lightweight gating over layers (scalar score per layer)
        self.gate_mlp = nn.Sequential(
            nn.Linear(D, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Cross-layer attention with a learned [CLS] token
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_p, batch_first=True)

        # Final head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Temperature for gating softmax
        self.tau = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _pair_features(ei, ej):
        return torch.cat([ei, ej, (ei - ej).abs(), ei * ej], dim=-1)

    def forward(self, emb_i, emb_j):
        # Single-layer case: (B, E)
        if emb_i.ndim == 2:
            z = self._pair_features(emb_i, emb_j)            # (B, 4E)
            h = self.pair_ff(self.pair_ln(z))                 # (B, H)
            logits = self.head(self.norm(h)).squeeze(-1)        # (B,)
            return {"logits": logits}

        # Multi-layer case: (B, L, E)
        B, L, E = emb_i.shape

        # Per-layer pair encoding
        z = self._pair_features(emb_i, emb_j)                # (B, L, 4E)
        H = self.pair_ff(self.pair_ln(z))                    # (B, L, H)

        # Cross-layer attention via a learned [CLS]
        cls = self.cls.expand(B, 1, -1)                      # (B, 1, H)
        X = torch.cat([cls, H], dim=1)                       # (B, 1+L, H)
        X_out, _ = self.mha(X, X, X, need_weights=False)
        cls_out = X_out[:, 0]                                # (B, H)

        # Gated pooling across layers (softmax on scalar gates)
        gate_scores = self.gate_mlp(z).squeeze(-1)           # (B, L)
        alpha = torch.softmax(gate_scores / (self.tau.abs() + 1e-6), dim=1)  # (B, L)
        pooled = (alpha.unsqueeze(-1) * H).sum(dim=1)        # (B, H)

        # Fuse CLS-attended summary and gated pooling
        fused = 0.5 * (cls_out + pooled)                     # (B, H)
        logits = self.head(self.norm(fused)).squeeze(-1)       # (B,)
        return {"logits": logits}


# ---------------------------------------------------------------------
# EntropyAttentiveLayerFusion: returns gate_alpha for entropy regularization
# ---------------------------------------------------------------------
class EntropyAttentiveLayerFusion(nn.Module):
    """
    AttentiveLayerFusion head, but also returns per-layer gate distribution (alpha)
    when in the multi-layer case, for entropy regularization.
    """
    def __init__(self, emb_dim, hidden_dim=256, vec_gate=False, num_heads=4, dropout_p=0.1):
        super().__init__()
        D = 4 * emb_dim  # concat [ei, ej, |ei-ej|, ei*ej]

        # Per-layer pair encoder
        self.pair_ln = nn.LayerNorm(D)
        self.pair_ff = nn.Sequential(
            nn.Linear(D, hidden_dim), nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )

        # Lightweight gating over layers (scalar score per layer)
        self.gate_mlp = nn.Sequential(
            nn.Linear(D, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Cross-layer attention with a learned [CLS] token
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_p, batch_first=True)

        # Final head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Temperature for gating softmax
        self.tau = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _pair_features(ei, ej):
        return torch.cat([ei, ej, (ei - ej).abs(), ei * ej], dim=-1)

    def forward(self, emb_i, emb_j):
        # Single-layer case: (B, E)
        if emb_i.ndim == 2:
            z = self._pair_features(emb_i, emb_j)            # (B, 4E)
            h = self.pair_ff(self.pair_ln(z))                 # (B, H)
            logits = self.head(self.norm(h)).squeeze(-1)        # (B,)
            return {"logits": logits}

        # Multi-layer case: (B, L, E)
        B, L, E = emb_i.shape

        # Per-layer pair encoding
        z = self._pair_features(emb_i, emb_j)                # (B, L, 4E)
        H = self.pair_ff(self.pair_ln(z))                    # (B, L, H)

        # Cross-layer attention via a learned [CLS]
        cls = self.cls.expand(B, 1, -1)                      # (B, 1, H)
        X = torch.cat([cls, H], dim=1)                       # (B, 1+L, H)
        X_out, _ = self.mha(X, X, X, need_weights=False)
        cls_out = X_out[:, 0]                                # (B, H)

        # Gated pooling across layers (softmax on scalar gates)
        gate_scores = self.gate_mlp(z).squeeze(-1)           # (B, L)
        alpha = torch.softmax(gate_scores / (self.tau.abs() + 1e-6), dim=1)  # (B, L)
        pooled = (alpha.unsqueeze(-1) * H).sum(dim=1)        # (B, H)

        # Fuse CLS-attended summary and gated pooling
        fused = 0.5 * (cls_out + pooled)                     # (B, H)
        logits = self.head(self.norm(fused)).squeeze(-1)       # (B,)
        return {"logits": logits, "gate_alpha": alpha}