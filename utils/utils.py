import numpy as np

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