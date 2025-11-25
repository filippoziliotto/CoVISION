import torch
from typing import Optional


def triangle_transitivity_loss(
    pair_idx: torch.Tensor,          # (P, 2)
    logits: torch.Tensor,            # (P,)
    strengths: Optional[torch.Tensor] = None,  # (P,), same order as pair_idx
    num_views: Optional[int] = None,
    max_triangles: int = 1024,
    margin: float = 0.0,
    strength_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Soft triangle/transitivity regulariser on the predicted co-visibility graph for a single scene.

    Given:
        - pair_idx: indices of labeled pairs (P, 2), with node ids in [0, num_views).
        - logits: model logits for each pair (P,).
        - strengths: optional per-pair strength values (P,) aligned with pair_idx.
        - num_views: total number of views in the scene. If None, inferred from pair_idx.
        - max_triangles: upper bound on number of triangles to use per scene.
        - margin: additional margin in the transitivity constraint.
        - strength_threshold: if not None, ignore triangles whose path edges are weaker than this.

    We enforce (softly) that if edges (i,j) and (j,k) are strong, then (i,k) should not be
    extremely weak, via:

        p_ik >= p_ij + p_jk - 1 + margin

    with a hinge penalty on violations.
    """
    device = logits.device

    if pair_idx.ndim != 2 or pair_idx.shape[1] != 2:
        raise ValueError(f"triangle_transitivity_loss expects pair_idx shape (P,2), got {pair_idx.shape}")
    if logits.ndim != 1 or logits.shape[0] != pair_idx.shape[0]:
        raise ValueError(
            f"triangle_transitivity_loss expects logits shape (P,), got {logits.shape} "
            f"for P={pair_idx.shape[0]}"
        )
    if strengths is not None:
        if strengths.shape != logits.shape:
            raise ValueError(
                f"triangle_transitivity_loss expects strengths shape {logits.shape}, "
                f"got {strengths.shape}"
            )
        strengths = strengths.to(device)

    pair_idx = pair_idx.to(device)
    logits = logits.to(device)

    P = pair_idx.shape[0]
    if P < 3:
        return logits.new_tensor(0.0)

    # Infer number of views if not provided.
    if num_views is None:
        num_views = int(pair_idx.max().item()) + 1
    if num_views < 3:
        return logits.new_tensor(0.0)

    # Build an adjacency matrix mapping (i,j) -> edge index e, or -1 if missing.
    edge_mat = logits.new_full((num_views, num_views), fill_value=-1, dtype=torch.long)
    for e in range(P):
        i = int(pair_idx[e, 0].item())
        j = int(pair_idx[e, 1].item())
        edge_mat[i, j] = e
        edge_mat[j, i] = e

    # Enumerate all triangles (i < j < k) where all three edges exist.
    triangles = []
    for i in range(num_views - 2):
        for j in range(i + 1, num_views - 1):
            e_ij = edge_mat[i, j].item()
            if e_ij < 0:
                continue
            for k in range(j + 1, num_views):
                e_jk = edge_mat[j, k].item()
                e_ik = edge_mat[i, k].item()
                if e_jk < 0 or e_ik < 0:
                    continue
                triangles.append((e_ij, e_jk, e_ik))

    if not triangles:
        return logits.new_tensor(0.0)

    tri_tensor = torch.tensor(triangles, dtype=torch.long, device=device)  # (T, 3)
    T = tri_tensor.shape[0]

    # Optionally subsample triangles to keep compute bounded.
    if T > max_triangles > 0:
        perm = torch.randperm(T, device=device)[:max_triangles]
        tri_tensor = tri_tensor[perm]
        T = tri_tensor.shape[0]

    idx_ab = tri_tensor[:, 0]  # (T,)
    idx_bc = tri_tensor[:, 1]
    idx_ac = tri_tensor[:, 2]

    probs = torch.sigmoid(logits)  # (P,)

    p_ab = probs[idx_ab]
    p_bc = probs[idx_bc]
    p_ac = probs[idx_ac]

    # Optional strength-based weighting / filtering.
    if strengths is not None:
        s_ab = strengths[idx_ab]
        s_bc = strengths[idx_bc]
        # Use the weaker of the two "path" edges as a weight.
        path_strength = torch.min(s_ab, s_bc)  # (T,)

        if strength_threshold is not None:
            # Keep only triangles where both path edges are sufficiently strong.
            mask = path_strength >= strength_threshold
            if not torch.any(mask):
                return logits.new_tensor(0.0)
            p_ab = p_ab[mask]
            p_bc = p_bc[mask]
            p_ac = p_ac[mask]
            path_strength = path_strength[mask]

        # Normalised weights for the loss.
        weights = path_strength / (path_strength.mean() + 1e-8)
    else:
        weights = None

    # Soft transitivity target: p_ac >= p_ab + p_bc - 1 + margin
    target = p_ab + p_bc - 1.0 + margin
    violations = torch.relu(target - p_ac)  # (T,)

    if weights is not None:
        loss = (violations * weights).mean()
    else:
        loss = violations.mean()

    return loss