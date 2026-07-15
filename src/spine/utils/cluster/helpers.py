import numpy as np
import torch
import MinkowskiEngine as ME

from sklearn.neighbors import kneighbors_graph

def knn_sklearn(coords, k):
    """Create a kNN graph using `scikit-learn`.

    Parameters
    ----------
    coords : Union[np.ndarray, torch.Tensor]
        (N, 3) Set of point coordinates
    k : int
        Number of neighbors in the kNN graph

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (2, E) Edge index
    """
    # If there is less than two points, no edge to be found
    if len(coords) < 2:
        return np.empty((2, 0), dtype=np.int64)

    # Get the appropriate number of neighbors
    k = min(k, len(coords)-1)

    # Dispatch
    if isinstance(coords, torch.Tensor):
        device = coords.device
        G = kneighbors_graph(
                coords.cpu().numpy(), n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return torch.Tensor(out).long().to(device=device)

    elif isinstance(coords, np.ndarray):
        G = kneighbors_graph(coords, n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return out

    else:
        raise ValueError(
                f"Coordinate format not recognized: {type(coords)}. Should be "
                 "either `np.ndarray` or `torch.Tensor`.")


def build_parent_of_from_kernel_map(parent_st: ME.SparseTensor, child_st: ME.SparseTensor) -> torch.Tensor:
    """
    Build parent_of[child_idx] = parent_idx using ME coordinate_manager kernel map.

    This maps *directly* between a bottleneck tensor and a full-res tensor by using
    stride = parent_stride / child_stride.

    Returns
    -------
    parent_of : (N_child,) long, -1 if no parent found
    """

    cm = parent_st.coordinate_manager
    ik = parent_st.coordinate_map_key # parent_key
    ok = child_st.coordinate_map_key  # child_key
    D = child_st.D
    dev = child_st.F.device

    ps = torch.tensor(parent_st.tensor_stride, device=dev, dtype=torch.int64)
    cs = torch.tensor(child_st.tensor_stride, device=dev, dtype=torch.int64)
    factor = (ps // cs).tolist()

    offsets = torch.empty((0, D), dtype=torch.int32)
    kmap = cm.get_kernel_map(
        ik, ok,
        kernel_size=factor,
        stride=factor,
        dilation=[1]*D,
        region_type=ME.RegionType.HYPER_CUBE,
        region_offset=offsets,
        is_transpose=True,
        is_pool=False,
    )

    parent_of = torch.full((child_st.F.shape[0],), -1, device=dev, dtype=torch.long)
    for _, ij in kmap.items():
        if ij.numel() == 0:
            continue
        ij = ij.view(2,-1) if ij.dim() != 2 else ij
        if ij.size(0) != 2:
            ij = ij.t()
        ii = ij[0].to(dev, torch.long)
        oo = ij[1].to(dev, torch.long)
        parent_of[oo] = ii

    return parent_of

from typing import Tuple
def build_cluster_token_topk(
        *,
        parent_of: torch.Tensor,
        child_cluster_id: torch.Tensor,
        child_feats: torch.Tensor,
        n_parent: int,
        K: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each parent voxel, find its top-K cluster IDs among children and build tokens
    by mean-pooling child_feats per (parent, cluster). Weight tokens by normalized counts.

    Returns
    -------
    tokens   : (N_parent, K, D_in) float  (already weighted by topk_w)
    topk_cid : (N_parent, K) long
    topk_w   : (N_parent, K) float (sum=1 over valid entries)
    """
    dev = child_feats.device
    parent_of = parent_of.to(dev)
    child_cluster_id = child_cluster_id.to(dev)

    valid = (parent_of >= 0) & (parent_of < n_parent)
    if valid.sum() == 0:
        D = chilid_feats.shape[1]
        return(
            torch.zeros((n_parent, K, D), device = dev, dtype=child_feats.dtype),
            torch.full((n_parent, K), -1, device=dev, dtype=torch.long),
            torch.zeros((n_parent, K), device=dev, dtype=child_feats.dtype),
        )
    p = parent_of[valid]
    c = child_cluster_id[valid]
    f = child_feats[valid]

    # ---- collision-free grouping by (parent, cluster) ----
    pairs = torch.stack([p, c], dim=1)  # (Nv, 2)
    uniq_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)
    pair_parent = uniq_pairs[:, 0]  # (Ppairs,)
    pair_cid = uniq_pairs[:, 1]  # (Ppairs,)

    Ppairs = uniq_pairs.shape[0]
    # mean pooled feature per pair + count
    D = f.shape[1]
    pair_sum = torch.zeros((Ppairs, D), device=dev, dtype=f.dtype)
    pair_sum.scatter_add_(0, inv[:, None].expand(-1, D), f)
    pair_cnt = torch.bincount(inv, minlength=Ppairs).to(device=dev, dtype=f.dtype)
    pair_feat = pair_sum / pair_cnt.clamp_min(1).unsqueeze(1)  # (Ppairs, D)

    # topK by count per parent: sort by count desc then stable sort by parent
    order = torch.argsort(pair_cnt, descending=True)
    pair_parent = pair_parent[order]
    pair_cid = pair_cid[order]
    pair_cnt = pair_cnt[order]
    pair_feat = pair_feat[order]

    order2 = torch.argsort(pair_parent, stable=True)
    pair_parent = pair_parent[order2]
    pair_cid = pair_cid[order2]
    pair_cnt = pair_cnt[order2]
    pair_feat = pair_feat[order2]

    counts_per_parent = torch.bincount(pair_parent, minlength=n_parent)
    starts = torch.cumsum(counts_per_parent, dim=0) - counts_per_parent
    idx_all = torch.arange(pair_parent.numel(), device=dev)
    rank = idx_all - starts[pair_parent]
    keep = rank < K

    sel_p = pair_parent[keep]
    sel_r = rank[keep].to(torch.long)
    sel_c = pair_cid[keep]
    sel_w = pair_cnt[keep]
    sel_f = pair_feat[keep]

    topk_cid = torch.full((n_parent, K), -1, device=dev, dtype=torch.long)
    topk_w = torch.zeros((n_parent, K), device=dev, dtype=f.dtype)
    tokens = torch.zeros((n_parent, K, D), device=dev, dtype=f.dtype)

    topk_cid[sel_p, sel_r] = sel_c
    topk_w[sel_p, sel_r] = sel_w
    tokens[sel_p, sel_r] = sel_f

    # normalize weights per parent and weight tokens
    wsum = topk_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    topk_w = topk_w / wsum
    tokens = tokens * topk_w.unsqueeze(-1)

    return tokens, topk_cid, topk_w
def tokens_to_sparse_on_parents(
    parent_st: ME.SparseTensor,
    tokens: torch.Tensor,     # (N_parent, K, D)
) -> ME.SparseTensor:
    """
    Represent K tokens per parent voxel as *channels* on the parent SparseTensor.
    Output features: (N_parent, K, D), coords identical to parent_st.C
    """
    assert tokens.shape[0] == parent_st.F.shape[0]
    Np, K, D = tokens.shape
    result = []
    for k in range(K):
        feat = tokens[:, k, :].reshape(Np, D).contiguous()
        x = ME.SparseTensor(
            features=feat,
            coordinates=parent_st.C,
            coordinate_manager=parent_st.coordinate_manager,
            tensor_stride=parent_st.tensor_stride,
            device=feat.device,
        )
        result.append(x)

    # Reuse the same coordinates/CM so row order matches parent indices exactly.
    return result

@torch.no_grad()
def compute_node_confidence(graph, num_shapes: int) -> torch.Tensor:
    """Compute per-voxel clustering confidence from edge probabilities.

    For each voxel, confidence is the mean over all its graph edges of
    how far each edge score is from the decision boundary (0.5):
    ``confidence_i = mean_e( |edge_prob_e - 0.5| * 2 )``

    This maps to [0, 1]:  0 means every edge is maximally uncertain
    (score = 0.5, i.e. random / early training), 1 means every edge is
    fully decided (score = 0 or 1, i.e. confident clustering).

    Parameters
    ----------
    graph : dict
        Output of ClusterGraphConstructor.__call__, containing at minimum
        ``edge_prob`` (TensorBatch), ``edge_index`` (TensorBatch),
        ``node_clusts`` (IndexBatch), ``edge_clusts`` (IndexBatch), and
        ``node_pred`` (TensorBatch).
    num_shapes : int
        Number of semantic shapes (= len(constructor.shapes)).

    Returns
    -------
    torch.Tensor
        (N_filtered,) float tensor of per-voxel confidence in [0, 1].
        Voxels with no edges (isolated or from empty shapes) get 0.
    """
    edge_prob_flat  = graph['edge_prob'].tensor   # (E_total,)
    edge_index_flat = graph['edge_index'].tensor  # (E_total, 2) local per-shape indices
    node_clusts     = graph['node_clusts']        # IndexBatch: [b][s] → local node indices
    edge_clusts     = graph['edge_clusts']        # IndexBatch: [b][s] → local edge indices

    num_nodes = graph['node_pred'].tensor.shape[0]
    device    = edge_prob_flat.device

    conf_sum = torch.zeros(num_nodes, device=device)
    conf_cnt = torch.zeros(num_nodes, device=device)

    for b in range(node_clusts.batch_size):
        # IndexBatch.__getitem__ strips the stored offset, so add it back to
        # get indices into the global flat tensors.
        node_off = int(node_clusts.offsets[b])
        edge_off = int(edge_clusts.offsets[b])

        for s in range(num_shapes):
            local_node_idx = node_clusts[b][s]  # (N_bs,) local to batch b
            local_edge_idx = edge_clusts[b][s]  # (E_bs,) local to batch b
            if local_edge_idx.numel() == 0:
                continue

            global_edge_idx = local_edge_idx + edge_off   # index into edge_index_flat
            ep        = edge_prob_flat[global_edge_idx]    # (E_bs,)
            edge_conf = (ep - 0.5).abs().mul_(2)           # [0, 1]

            local_ei = edge_index_flat[global_edge_idx]    # (E_bs, 2) per-shape node idx
            # Map per-shape local node index → global node index in conf_sum
            src_g = local_node_idx[local_ei[:, 0]] + node_off
            dst_g = local_node_idx[local_ei[:, 1]] + node_off

            conf_sum.scatter_add_(0, src_g, edge_conf)
            conf_sum.scatter_add_(0, dst_g, edge_conf)
            conf_cnt.scatter_add_(0, src_g, torch.ones_like(edge_conf))
            conf_cnt.scatter_add_(0, dst_g, torch.ones_like(edge_conf))

    return conf_sum.div_(conf_cnt.clamp_(min=1e-8))


def mask_cluster_id(
    x_in: ME.SparseTensor,
    cluster_id: torch.Tensor,
    confidence: torch.Tensor = None,
    mode: str = "in_cluster"
):
    """Create attention mask from cluster assignments.

    Args:
        x_in: ME.SparseTensor (N, C) - input features
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
    assert x_in.F.shape[0] == cluster_id.shape[0], (
        f"Feature count {x_in.F.shape[0]} != cluster_id count {cluster_id.shape[0]}")
    batch_indices = x_in.C[:, 0]
    unique_batches = torch.unique(batch_indices, sorted=True)
    device = x_in.F.device

    # Return a list of per-event (N_b, N_b) masks rather than a single padded
    # (B, max_N, max_N) tensor.  The padded form requires O(B * max_N^2) memory
    # upfront (e.g. 16 events × 14500^2 × 4 B ≈ 13 GiB) even though blocks.py
    # only ever uses one event's slice at a time.
    masks_list = []
    for b in unique_batches:
        batch_mask_bool = (batch_indices == b)
        batch_cluster_ids = cluster_id[batch_mask_bool]
        batch_confidence = confidence[batch_mask_bool] if confidence is not None else None
        x_batch_C = x_in.C[batch_mask_bool, 1:].float()
        if mode == "in_cluster":
            mask_b = create_incluster_mask(batch_cluster_ids, confidence=batch_confidence)
        elif mode == "boundary":
            mask_b = create_boundary_mask(x_batch_C, batch_cluster_ids,
                                          confidence=batch_confidence)
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

    same_cluster    = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1))  # (N, N)
    not_noise       = (cluster_id >= 0)
    both_not_noise  = not_noise.unsqueeze(0) & not_noise.unsqueeze(1)       # (N, N)
    allow_attention = same_cluster & both_not_noise
    allow_attention.diagonal().fill_(True)

    if confidence is None:
        # Hard binary mask: original behaviour
        mask = torch.full((N, N), float('-inf'), device=device, dtype=torch.float32)
        mask[allow_attention] = 0.0
    else:
        # Soft mask: cross-cluster suppression scales with joint confidence.
        # Allowed (same-cluster) pairs are always 0; blocked pairs range
        # from 0 (uncertain) to -mask_scale (fully confident).
        conf_ij = torch.minimum(
            confidence.unsqueeze(1),   # (N, 1)
            confidence.unsqueeze(0),   # (1, N)
        )                              # (N, N) — conservative: use the less confident voxel
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        blocked = (~allow_attention) & both_not_noise
        mask[blocked] = -(conf_ij[blocked] * mask_scale)
        del conf_ij

    del same_cluster, not_noise, both_not_noise, allow_attention
    return mask

def create_boundary_mask(coords, cluster_id, neighbor_r=5,
                         confidence=None, mask_scale: float = 15.0):
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
    dist_matrix = torch.cdist(coords, coords)  # (N, N)
    within_radius = (dist_matrix <= neighbor_r)

    is_boundary = detect_boundary_vox(dist_matrix, cluster_id, neighbor_r)

    same_cluster   = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1))
    not_noise      = (cluster_id >= 0)
    both_not_noise = not_noise.unsqueeze(0) & not_noise.unsqueeze(1)

    is_boundary_i = is_boundary.unsqueeze(1)  # (N, 1)
    is_boundary_j = is_boundary.unsqueeze(0)  # (1, N)

    allow_attention = (
        ((is_boundary_i | is_boundary_j) & within_radius) | same_cluster
    )
    allow_attention = allow_attention & both_not_noise
    allow_attention.diagonal().fill_(True)

    if confidence is None:
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        mask[~allow_attention] = float('-inf')
    else:
        conf_ij = torch.minimum(
            confidence.unsqueeze(1),
            confidence.unsqueeze(0),
        )  # (N, N)
        mask = torch.zeros((N, N), device=device, dtype=torch.float32)
        blocked = (~allow_attention) & both_not_noise
        mask[blocked] = -(conf_ij[blocked] * mask_scale)
        del conf_ij

    del same_cluster, not_noise, both_not_noise, allow_attention
    del is_boundary_i, is_boundary_j, is_boundary

    return mask

def detect_boundary_vox(dist, cluster_id, radius = 5):
    """
    Detect which voxels are at cluster boundaries.

    Returns:
        is_boundary: (N_b,) boolean tensor
    """

    # Get neighbors within radius
    neighbors_mask = (dist <= radius) & (dist > 0)

    # Check cluster IDs
    # For each pair (i, j), check if they're neighbors AND have different clusters
    same_cluster_matrix = (cluster_id.unsqueeze(0) == cluster_id.unsqueeze(1))  # (N_b, N_b)
    different_cluster_neighbors = neighbors_mask & (~same_cluster_matrix)  # (N_b, N_b)

    # A voxel is a boundary if it has ANY neighbor with different cluster
    is_boundary = different_cluster_neighbors.any(dim=1)  # (N_b,)

    return is_boundary


def detect_boundary_vox_efficient(coords, cluster_id, radius=5):
    """
    Detect boundaries efficiently using KNN instead of full distance matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(coords.cpu().numpy())
    distances, indices = nbrs.radius_neighbors(coords.cpu().numpy())

    device = coords.device
    is_boundary = torch.zeros(len(coords), dtype=torch.bool, device=device)

    for i, (dist_i, idx_i) in enumerate(zip(distances, indices)):
        if len(idx_i) > 1:  # Has neighbors (excluding self)
            neighbor_clusters = cluster_id[torch.tensor(idx_i, device=device)]
            my_cluster = cluster_id[i]
            # Boundary if any neighbor has different cluster
            if (neighbor_clusters != my_cluster).any():
                is_boundary[i] = True

    return is_boundary