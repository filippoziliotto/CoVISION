import numpy as np
import random
import torch
import torch.nn.functional as F

def adj_to_edges(adj: np.ndarray, threshold: float = 0.5):
    """
    adj: (N, N) adjacency / score matrix
    threshold: binarization threshold; for pure 0/1 adj just use 0.5
    returns list of (i, j) edges with i < j
    """
    adj = np.asarray(adj)
    N = adj.shape[0]
    # upper-triangular indices only (avoid double-counting i,j and j,i)
    i_idx, j_idx = np.triu_indices(N, k=1)
    mask = adj[i_idx, j_idx] >= threshold
    edges = list(zip(i_idx[mask], j_idx[mask]))
    return edges


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        


def pairwise_ranking_loss(
    scores: torch.Tensor,
    strengths: torch.Tensor,
    margin: float = 0.1,
    num_samples: int = 1024,
) -> torch.Tensor:
    """
    Pairwise ranking loss: if strength_i > strength_j + margin,
    encourage score_i > score_j by at least `margin`.

    scores:    (B,) raw logits (or scores)
    strengths: (B,) continuous edge strengths (e.g., from rel_mat)
    """
    # Ensure 1D
    scores = scores.view(-1)
    strengths = strengths.view(-1)

    B = scores.size(0)
    if B < 2:
        return torch.tensor(0.0, device=scores.device)

    # Randomly sample index pairs
    idx1 = torch.randint(0, B, (num_samples,), device=scores.device)
    idx2 = torch.randint(0, B, (num_samples,), device=scores.device)

    s1 = strengths[idx1]
    s2 = strengths[idx2]

    # Keep only pairs where strength_i is significantly higher than strength_j
    mask = s1 > (s2 + margin)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Scores should respect the same ordering
    score_diff = scores[idx1] - scores[idx2]  # want this > margin
    loss = F.relu(margin - score_diff).mean()
    return loss
