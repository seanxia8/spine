"""Cluster-conditioned attention masks for sparse attention layers."""

import torch

from spine.model import sparse

__all__ = ["mask_cluster_id", "create_incluster_mask", "create_boundary_mask"]


def mask_cluster_id(
    x_in: sparse.SparseTensor,
    cluster_id: torch.Tensor,
    confidence: torch.Tensor = None,
    mode: str = "in_cluster",
):
    """Create attention mask from cluster assignments.

    Args:
        x_in: sparse.SparseTensor (N, C) - input features
        cluster_id: (N,) - cluster assignment for each voxel (-1 for noise)
        confidence: (N,) optional per-voxel confidence in [0, 1].
            When provided, cross-cluster suppression is scaled by
            ``min(conf_i, conf_j)`` instead of being hard -inf.
            Low confidence → near-uniform attention.
            High confidence → sharp within-cluster masking.
        mode: str - type of mask to create:
            - 'in_cluster': attend only within same cluster
            - 'boundary': attend to cluster boundary regions

    Returns:
        attention_mask: mask suitable for torch.nn.MultiheadAttention
            - For 'attn_mask': (B, N, N)
            - Padded for events in one batch
    """
    assert (
        x_in.F.shape[0] == cluster_id.shape[0]
    ), f"Feature count {x_in.F.shape[0]} != cluster_id count {cluster_id.shape[0]}"
    batch_indices = x_in.C[:, 0]
    unique_batches = torch.unique(batch_indices, sorted=True)
    device = x_in.F.device

    # Return a list of per-event (N_b, N_b) masks rather than a single padded
    # (B, max_N, max_N) tensor.  The padded form requires O(B * max_N^2) memory
    # upfront (e.g. 16 events × 14500^2 × 4 B ≈ 13 GiB) even though blocks.py
    # only ever uses one event's slice at a time.
    masks_list = []
    for b in unique_batches:
        batch_mask_bool = batch_indices == b
        batch_cluster_ids = cluster_id[batch_mask_bool]
        batch_confidence = (
            confidence[batch_mask_bool] if confidence is not None else None
        )
        x_batch_C = x_in.C[batch_mask_bool, 1:].float()
        if mode == "in_cluster":
            mask_b = create_incluster_mask(
                batch_cluster_ids, confidence=batch_confidence
            )
        elif mode == "boundary":
            mask_b = create_boundary_mask(
                x_batch_C, batch_cluster_ids, confidence=batch_confidence
            )
            del x_batch_C
        else:
            raise ValueError(f"Unknown mask mode: {mode}")

        masks_list.append(mask_b.detach())
        del mask_b

    return masks_list


def create_incluster_mask(cluster_id, confidence=None, mask_scale: float = 15.0):
    """Build the (N, N) additive attention-bias mask for within-cluster attention.

    Parameters
    ----------
    cluster_id : torch.Tensor
        (N,) integer cluster assignment; -1 = noise.
    confidence : torch.Tensor, optional
        (N,) per-voxel confidence in [0, 1].  When ``None``, the mask is hard:
        0 for allowed pairs, -inf for blocked pairs.  When provided, blocked
        pairs receive ``-confidence_ij * mask_scale`` where
        ``confidence_ij = min(confidence_i, confidence_j)``.

        This means:
        - confidence → 0 (garbage clusters): mask → 0 everywhere → uniform attention.
        - confidence → 1 (sharp clusters):   mask → -mask_scale ≈ -inf → hard block.
    mask_scale : float, default 15.0
        Magnitude of the soft suppression.  ``exp(-15) ≈ 3e-7``, effectively
        zero after softmax when confidence = 1.
    """
    device = cluster_id.device
    N = cluster_id.shape[0]

    same_cluster = cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)  # (N, N)
    not_noise = cluster_id >= 0
    both_not_noise = not_noise.unsqueeze(0) & not_noise.unsqueeze(1)  # (N, N)
    allow_attention = same_cluster & both_not_noise
    allow_attention.diagonal().fill_(True)

    if confidence is None:
        # Hard binary mask: original behaviour
        mask = torch.full((N, N), float("-inf"), device=device, dtype=torch.float32)
        mask[allow_attention] = 0.0
    else:
        # Soft mask: cross-cluster suppression scales with joint confidence.
        # Allowed (same-cluster) pairs are always 0; blocked pairs range
        # from 0 (uncertain) to -mask_scale (fully confident).
        conf_ij = torch.minimum(
            confidence.unsqueeze(1),  # (N, 1)
            confidence.unsqueeze(0),  # (1, N)
        )  # (N, N) — conservative: use the less confident voxel
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        blocked = (~allow_attention) & both_not_noise
        mask[blocked] = -(conf_ij[blocked] * mask_scale)
        del conf_ij

    del same_cluster, not_noise, both_not_noise, allow_attention
    return mask


def create_boundary_mask(
    coords, cluster_id, neighbor_r=5, confidence=None, mask_scale: float = 15.0
):
    """Build the (N, N) additive attention-bias mask for boundary-aware attention.

    Interior voxels attend only within their cluster; boundary voxels also
    attend to any voxel within ``neighbor_r`` (to probe cross-cluster regions).

    Parameters
    ----------
    coords : torch.Tensor
        (N, 3) voxel coordinates.
    cluster_id : torch.Tensor
        (N,) integer cluster assignment; -1 = noise.
    neighbor_r : float, default 5
        Radius within which boundary voxels may attend cross-cluster.
    confidence : torch.Tensor, optional
        (N,) per-voxel confidence in [0, 1].  When provided, suppressed pairs
        (those that are neither same-cluster nor boundary-proximal) receive
        ``-min(conf_i, conf_j) * mask_scale`` instead of hard -inf.
        Low confidence → near-uniform attention; high confidence → hard block.
    mask_scale : float, default 15.0
        Magnitude of soft suppression.  See ``create_incluster_mask``.
    """
    device = cluster_id.device
    N = cluster_id.shape[0]
    dist_matrix = torch.cdist(coords, coords)
    within_radius = dist_matrix <= neighbor_r

    is_boundary = detect_boundary_vox(dist_matrix, cluster_id, neighbor_r)

    same_cluster = cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1)
    not_noise = cluster_id >= 0
    both_not_noise = not_noise.unsqueeze(0) & not_noise.unsqueeze(1)

    is_boundary_i = is_boundary.unsqueeze(1)
    is_boundary_j = is_boundary.unsqueeze(0)

    allow_attention = ((is_boundary_i | is_boundary_j) & within_radius) | same_cluster
    allow_attention = allow_attention & both_not_noise
    allow_attention.diagonal().fill_(True)

    if confidence is None:
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        mask[~allow_attention] = float("-inf")
    else:
        conf_ij = torch.minimum(confidence.unsqueeze(1), confidence.unsqueeze(0))
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        blocked = (~allow_attention) & both_not_noise
        mask[blocked] = -(conf_ij[blocked] * mask_scale)
        del conf_ij

    del same_cluster, not_noise, both_not_noise, allow_attention
    del is_boundary_i, is_boundary_j, is_boundary, dist_matrix, within_radius
    return mask


def detect_boundary_vox(dist, cluster_id, radius=5):
    """
    Detect which voxels are at cluster boundaries.

    Returns:
        is_boundary: (N_b,) boolean tensor
    """

    # Get neighbors within radius
    neighbors_mask = (dist <= radius) & (dist > 0)

    # Check cluster IDs
    # For each pair (i, j), check if they're neighbors AND have different clusters
    same_cluster_matrix = cluster_id.unsqueeze(0) == cluster_id.unsqueeze(
        1
    )  # (N_b, N_b)
    different_cluster_neighbors = neighbors_mask & (~same_cluster_matrix)  # (N_b, N_b)

    # A voxel is a boundary if it has ANY neighbor with different cluster
    is_boundary = different_cluster_neighbors.any(dim=1)  # (N_b,)

    return is_boundary
