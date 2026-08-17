"""Supervised iterative dense clustering model and its loss."""

import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean, scatter_add

from spine.data import TensorBatch, IndexBatch
from spine.utils.cluster.graph import ClusterGraphConstructor
from spine.constants.factory import enum_factory
from spine.utils.globals import (
    SHAPE_COL, SHOWR_SHP, TRACK_SHP, DELTA_SHP, MICHL_SHP)
from .layer.cluster import kernel_factory, loss_factory
from .layer.factories import loss_fn_factory
from .layer.cluster.attn_graph_spice_embedder import AttnGraphSPICEEmbedder
from spine.utils.cluster.helpers import compute_node_confidence
from spine.utils.globals import BATCH_COL, COORD_COLS, SHAPE_COL, VALUE_COL, GHOST_SHP

__all__ = ['GraphSPICEIter', 'GraphSPICEIterLoss']


class GraphSPICEIter(torch.nn.Module):
    """Graph Scalable Proposal-free Instance Clustering Engine (Graph-SPICE).

    Graph-SPICE has two components:
    1. Voxel embedder: UNet-type CNN architecture used for feature
       extraction and feature embeddings.
    2. Edge probability kernel function: A kernel function (any callable
       that takes two node attribute vectors to give a edge proability score).

    Prediction is done in two steps:
    1. A neighbor graph (ex. KNN, Radius) is constructed to compute
       edge probabilities between neighboring edges;
    2. Edges with low probability scores are dropped;
    3. The voxels are clustered through connected component clustering.

    A typical configuration is broken down into multiple components:

    .. code-block:: yaml

        model:
          name: graph_spice
          modules:
            graph_spice:
              <Basic parameters>
              embedder:
                <Feature embedding configuration block>
              kernel:
                <Edge kernel function configuration block>
              constructor:
                <Graph construction base parameters>
                graph:
                  <Graph configuration block>
                orphan:
                  <Orphan assignment configuration block>

    See configuration file(s) prefixed with `graph_spice` under the `config`
    directory for detailed examples of working configurations.
    """
    MODULES = ['constructor', 'embedder', 'kernel']

    def __init__(self, graph_spice, graph_spice_loss=None):
        """Initialize the Graph-SPICE model.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        graph_spice_loss : dict, optional
            Graph-SPICE loss configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the model configuration
        self.process_model_config(**graph_spice)

    @staticmethod
    def _build_edge_proj(cfg, num_features):
        """Build N stop-gradient linear layers for edge feature extraction.

        Parameters
        ----------
        cfg : dict or None
            ``n_layers``   : int   — number of layers (0 or None → disabled)
            ``hidden_dim`` : int   — hidden width (default: num_features)
        num_features : int
            Input/output dimension (must match the bilinear kernel).

        Returns
        -------
        nn.Sequential or None
        """
        if cfg is None:
            return None
        n = cfg.get('n_layers', 0)
        if n <= 0:
            return None
        hidden = cfg.get('hidden_dim', num_features)
        layers = []
        for i in range(n):
            in_d  = num_features if i == 0     else hidden
            out_d = num_features if i == n - 1 else hidden
            layers.append(torch.nn.Linear(in_d, out_d))
            if i < n - 1:
                layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def process_model_config(self, embedder, kernel, constructor,
                             shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                             use_raw_features=False, invert=True,
                             make_clusters=False, n_iterations=2,
                             use_attention=True, use_confidence_mask=True,
                             do_iterate=None, edge_projection=None,
                             track_iter_metrics=False, log_step=1,
                             feature_graph_warmup_steps=0,
                             cluster_semantic_pool=False,
                             use_pred_sem_filter=False,
                             pred_sem_filter_warmup_steps=20000):
        """Initialize the underlying modules.

        Parameters
        ----------
        embedder : dict
            Pixel embedding configuration
        kernel : dict
            Edge kernel configuration
        constructor : dict
            Edge index construction configuration
        shapes : List[str]
            List of shape names to construct clusters for
        use_raw_features : bool, default False
            Use the list of embedder features as is, without the output layers
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        make_clusters : bool, default False
            If `True`, builds a list of cluster indexes
        n_iterations : int, default 2
            Total number of forward passes through the embedder.
            1 = single-pass baseline (no graph feedback, no attention).
            2 = one refinement step (original behaviour).
            k > 2 = convergence study.
        use_attention : bool, default True
            If True, each refinement pass receives the previous cluster IDs as
            attention conditioning. Set to False to ablate the attention while
            keeping the iterative loop (iteration without attention).
        use_confidence_mask : bool, default True
            If True, cross-cluster suppression in the attention mask is scaled
            by per-voxel edge-score confidence (soft gate). If False, the mask
            falls back to the original hard {0, -inf} binary behaviour.
        do_iterate : bool, optional
            Deprecated alias. If provided, overrides n_iterations:
            False -> n_iterations=1, True -> n_iterations=2.
            Kept for backward compatibility.
        track_iter_metrics : bool, default False
            If True, computes ARI/purity/efficiency/SBD for every intermediate
            iteration (0 through n_iterations-2) and stores them in the result
            dict as ``{metric}_iter{k}`` (e.g. ``ari_iter0``, ``sbd_shower_iter0``).
            These scalars are logged to wandb automatically. The final iteration
            metrics (no suffix) remain the responsibility of the loss function's
            ``evaluate_clustering_metrics`` flag.
            Only active when ``clust_label`` is provided (training/validation).
        feature_graph_warmup_steps : int, default 0
            Number of forward passes before the feature-space kNN graph is
            enabled for refinement iterations (k >= 1).  During warmup only
            spatial kNN is used, giving the contrastive loss time to build a
            discriminative feature space before feature-kNN edges are proposed.
            0 = feature kNN active from the start (if ``feature_graph`` is
            configured in the constructor block).
        cluster_semantic_pool : bool, default False
            If `True`, aggregate the semantic predictions within each cluster and use the aggregated prediction as the final semantic prediction.
        use_pred_sem_filter : bool, default False
            If `True`, once `pred_sem_filter_warmup_steps` forward passes have
            elapsed, gate per-shape graph construction using the model's own
            predicted semantics (argmax over all num_classes, e.g. including
            LowE) instead of the true seg_label. Clustering itself still only
            runs on the shapes in `shapes` (e.g. shower/track/michel/delta) --
            a voxel predicted as a shape outside that list (e.g. LowE) simply
            matches no shape's subgraph and ends up excluded from clustering,
            exactly as LowE voxels already are under true-label gating.
        pred_sem_filter_warmup_steps : int, default 20000
            Number of forward passes before switching graph-gating from true
            seg_label to predicted semantics. 0 = switch immediately (if
            use_pred_sem_filter is True).
        """
        # Backward compatibility with do_iterate flag
        if do_iterate is not None:
            n_iterations = 2 if do_iterate else 1

        self.n_iter = n_iterations
        self.use_attention = use_attention
        self.use_confidence_mask = use_confidence_mask
        self.feature_graph_warmup_steps = feature_graph_warmup_steps
        # Build attention layers whenever n_iterations > 1 so the ablation
        # (use_attention=False) shares the same architecture as the full system.
        #enable_attention = (n_iterations > 1)

        self.embedder = AttnGraphSPICEEmbedder(
                **embedder, use_raw_features=use_raw_features,
                enable_attention=self.use_attention)

        # Initialize the kernel function (must be owned here to be loaded)
        self.kernel_fn = kernel_factory(kernel)

        # Optional stop-gradient edge projection for the raw-features path.
        self.edge_proj = self._build_edge_proj(edge_projection,
                                               kernel['num_features'])

        self.constructor = ClusterGraphConstructor(
                **constructor, kernel_fn=self.kernel_fn, edge_proj=self.edge_proj,
                shapes=shapes, invert=invert, training=self.training)

        # Parse the set of shapes to cluster
        self.shapes = enum_factory('shape', shapes)
        self.cluster_semantic_pool = cluster_semantic_pool
        self.use_pred_sem_filter = use_pred_sem_filter
        self.pred_sem_filter_warmup_steps = pred_sem_filter_warmup_steps
        # Store model parameters
        self.use_raw_features = use_raw_features
        self.invert = invert
        self.make_clusters = make_clusters
        self.track_iter_metrics = track_iter_metrics
        self.log_step = log_step
        self._forward_count = 0

        # Precompute all metric key names so the CSV schema is consistent on
        # every row (NaN placeholders on non-log steps, real values on log steps).
        _eval_base = ['ari', 'purity', 'efficiency', 'sbd']
        _eval_keys = _eval_base + [
            f'{m}_{s}' for m in _eval_base for s in self.shapes]
        _coh_mi = ['cohesion_within', 'cohesion_between', 'cohesion_ratio',
                   'mi_mean', 'mi_max', 'mi_sum', 'mig']
        # Final-iteration model keys: only cohesion + MI (evaluate metrics come
        # from the loss, not from here).
        self._metric_keys = list(_coh_mi)
        # Intermediate-iteration keys (one set per refinement step).
        for k in range(1, n_iterations):
            self._metric_keys += [f'{key}_iter{k}'
                                  for key in _eval_keys + _coh_mi]

    def filter_class(self, data, seg_label, clust_label=None):
        """Filter the list of pixels to those in the list of requested shapes.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Parameters
        ----------
        data : TensorBatch
            (M, 1+ + D + Nf) restricted tensor of voxel/value pairs
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : TensorBatch
            (M, 1 + D + N_c) Restricted tnesor of cluster labels
        index : torch.Tensor
            (M) Index to narrow down the original tensor
        counts : torch.Tensor
            (B) Number of restricted points in each batch entry
        """
        # Convert shapes to a torch tensor for easy comparison
        shapes = torch.tensor(self.shapes, device=data.device)

        # Create an index of the valid input rows
        mask = (seg_label.tensor[:, SHAPE_COL] == shapes.view(-1, 1)).any(dim=0)
        index = torch.where(mask)[0]

        # Restrict the input
        offsets = data.edges[:-1]
        data = TensorBatch(
                data.tensor[index], batch_size=data.batch_size,
                has_batch_col=True)

        # Restrict the label tensors
        assert seg_label.shape[0] == mask.shape[0], (
                 "The segmentation label tensor is of the wrong shape: "
                f"{seg_label.shape[0]} != {mask.shape[0]}")
        seg_label = TensorBatch(seg_label.tensor[index], data.counts)

        if clust_label is not None:
            assert clust_label.shape[0] == mask.shape[0], (
                     "The cluster label tensor is of the wrong shape: "
                    f"{clust_label.shape[0]} != {mask.shape[0]}")
            clust_label = TensorBatch(clust_label.tensor[index], data.counts)

        # Store the index as an IndexBatch
        index = IndexBatch(index, offsets, data.counts)

        return data, seg_label, clust_label, index

    def _get_features(self, result):
        """Extract the features used by the kernel from an embedder result."""
        if self.use_raw_features:
            return result['features']
        return result['hypergraph_features']

    def _get_graph_seg_label(self, seg_label, segmentation):
        """Get the seg_label-like tensor used to gate per-shape graph construction.

        Returns the true seg_label during warmup. Once use_pred_sem_filter is
        enabled and warmup has elapsed, substitutes the model's own predicted
        semantics (argmax over the raw, un-pooled per-voxel segmentation
        logits -- never the cluster_semantic_pool output) for the shape
        column, so per-shape graph construction no longer depends on ground
        truth. Clustering itself is unaffected: it still only runs on shapes
        in self.shapes, so a voxel predicted as a shape outside that list
        (e.g. LowE) simply matches no shape's subgraph, same as today.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) True per-voxel segmentation label
        segmentation : TensorBatch, optional
            (N, num_classes) Raw per-voxel segmentation logits for this pass
            (segmentation_iter0 or segmentation_iter_last)

        Returns
        -------
        TensorBatch
            seg_label, or a substitute with the shape column replaced by the
            model's predicted shape
        """
        use_pred = (self.use_pred_sem_filter and segmentation is not None
                    and self._forward_count >= self.pred_sem_filter_warmup_steps)
        if not use_pred:
            return seg_label

        with torch.no_grad():
            pred_shape = torch.argmax(
                    segmentation.tensor, dim=1).to(seg_label.tensor.dtype)
        new_tensor = seg_label.tensor.clone()
        new_tensor[:, SHAPE_COL] = pred_shape
        return TensorBatch(new_tensor, seg_label.counts)

    @torch.no_grad()
    def _compute_cohesion(self, features, node_labels, sample_size=2048):
        """Compute the cluster cohesion ratio in embedding space.

        For a set of voxels, this measures how much more similar same-cluster
        voxels are to each other (in cosine distance) than different-cluster
        voxels. A higher ratio indicates that the embedding space is more
        cluster-discriminative.

        Parameters
        ----------
        features : TensorBatch
            (N, D) voxel embedding features from the embedder.
        node_labels : list of torch.Tensor
            Per-batch ground-truth cluster IDs, as returned in graph['node_label'].
        sample_size : int, default 2048
            Maximum number of voxels to sample for the O(N²) cosine similarity
            computation. Sampling is stratified across the full batch.

        Returns
        -------
        within : float
            Mean cosine similarity between voxels in the same cluster.
        between : float
            Mean cosine similarity between voxels in different clusters.
        ratio : float
            within / |between| — the cohesion ratio. Higher is better.
        """
        feats = features.tensor.float()  # (N, D)

        # Build globally unique cluster IDs across all entries in the batch
        # so that cluster 0 in batch 0 and cluster 0 in batch 1 are distinct.
        all_ids = []
        offset = 0
        for labels_b in node_labels:
            if not isinstance(labels_b, torch.Tensor):
                labels_b = torch.tensor(labels_b, dtype=torch.long,
                                        device=feats.device)
            labels_b = labels_b.long()
            n_b = labels_b.shape[0]
            if n_b > 0:
                all_ids.append(labels_b + offset)
                offset += int(labels_b.max().item()) + 1
            else:
                all_ids.append(labels_b)

        cluster_ids = torch.cat(all_ids).to(feats.device)   # (N,)
        N = feats.shape[0]

        # Subsample to keep the O(N²) similarity matrix tractable
        if N > sample_size:
            idx = torch.randperm(N, device=feats.device)[:sample_size]
            feats = feats[idx]
            cluster_ids = cluster_ids[idx]

        # Guard against zero-norm features (e.g. early in training)
        norms = feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        feats_n = feats / norms                              # unit-normed
        K = feats_n @ feats_n.T                              # (N, N) cosine sim

        same = (cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1))
        diag = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
        same = same & ~diag                                  # exclude self

        within_vals  = K[same]
        between_vals = K[~same & ~diag]

        if within_vals.numel() == 0 and between_vals.numel() == 0:
            return float('nan'), float('nan'), float('nan')

        if between_vals.numel() == 0:
            # All sampled voxels in one cluster: perfect within-cluster cohesion,
            # no cross-cluster pairs to compare against. Treat between as 0 so
            # the ratio is large (good).
            within  = within_vals.mean().item()
            between = 0.0
        elif within_vals.numel() == 0:
            # All sampled voxels in distinct clusters: no within-cluster pairs.
            # Treat within as 0 so the ratio is 0 (poor cohesion).
            within  = 0.0
            between = between_vals.mean().item()
        else:
            within  = within_vals.mean().item()
            between = between_vals.mean().item()

        ratio = within / (abs(between) + 1e-8)

        return within, between, ratio

    @torch.no_grad()
    def _compute_mi_metrics(self, features, seg_label, sample_size=4096):
        """Compute Gaussian MI between each embedding dimension and semantic type.

        Uses the Gaussian approximation:

        .. math::

            I(X_d; Y) \\approx \\frac{1}{2}
            \\log \\left(\\frac{\\mathrm{Var}(X_d)}{
            \\mathbb{E}_y[\\mathrm{Var}(X_d | Y=y)]}\\right)

        Aggregated scalars returned
        - **mi_mean**: mean MI across all embedding dimensions
        - **mi_max**: MI of the single most-informative dimension
        - **mi_sum**: total MI summed over dimensions
        - **mig**: Mutual Information Gap (Chen et al. 2018) — MI of the top
          dimension minus the second, normalised by label entropy H(Y).
          A value close to 1 means one dimension dominates; close to 0 means
          information is spread uniformly or the embedding is entangled.

        Parameters
        ----------
        features : TensorBatch
            (N, D) voxel embedding tensor from the embedder.
        seg_label : TensorBatch
            (N, 1+D+1) segmentation label tensor; column ``VALUE_COL`` holds
            the integer semantic-type label for each voxel.
        sample_size : int, default 4096
            Cap on the number of voxels used. Drawn without replacement.

        Returns
        -------
        dict
            Keys ``mi_mean``, ``mi_max``, ``mi_sum``, ``mig`` — all floats.

        References
        ----------
        - Chen et al. (2018) "Isolating Sources of Disentanglement in VAEs"
          (defines MIG). https://arxiv.org/abs/1802.04942
        - Cover & Thomas, *Elements of Information Theory*, ch. 2 & 8.
        """
        feats  = features.tensor.float()               # (N, D)
        labels = seg_label.tensor[:, VALUE_COL].long() # (N,)

        # Subsample so this stays O(N·D) even for large events
        N = feats.shape[0]
        if N > sample_size:
            idx    = torch.randperm(N, device=feats.device)[:sample_size]
            feats  = feats[idx]
            labels = labels[idx]

        eps = 1e-8
        D   = feats.shape[1]
        N   = feats.shape[0]

        # Need at least 2 samples for unbiased variance; fewer → undefined.
        if N < 2:
            return {'mi_mean': float('nan'), 'mi_max': float('nan'),
                    'mi_sum': float('nan'), 'mig': float('nan')}

        # --- Marginal variance per dimension ---
        # nan_to_num before clamp: clamp passes NaN through unchanged in PyTorch.
        var_total = torch.nan_to_num(feats.var(dim=0), nan=eps).clamp(min=eps)

        # --- Conditional variance: E_y[ Var(X_d | Y=y) ] ---
        var_cond = torch.zeros(D, device=feats.device)
        unique_labels = labels.unique()
        for c in unique_labels:
            mask = (labels == c)
            n_c  = mask.sum().item()
            if n_c < 2:
                continue
            p_c = n_c / N
            var_cond += p_c * torch.nan_to_num(
                feats[mask].var(dim=0), nan=0.0).clamp(min=0)
        var_cond = var_cond.clamp(min=eps)

        # --- Gaussian MI per dimension (nats) ---
        MI = 0.5 * (var_total.log() - var_cond.log())
        MI = MI.clamp(min=0)                           # (D,)

        mi_mean = MI.mean().item()
        mi_max  = MI.max().item()
        mi_sum  = MI.sum().item()

        # --- MIG: top-dim gap normalised by H(Y) ---
        counts = torch.stack([(labels == c).sum() for c in unique_labels]).float()
        p_y    = counts / counts.sum()
        H_Y    = -(p_y * (p_y + eps).log()).sum().item()

        sorted_MI, _ = MI.sort(descending=True)
        # Need at least two dimensions for the gap to be meaningful
        if sorted_MI.shape[0] >= 2 and H_Y > eps:
            mig = ((sorted_MI[0] - sorted_MI[1]) / H_Y).item()
        else:
            mig = 0.0

        return {'mi_mean': mi_mean, 'mi_max': mi_max,
                'mi_sum': mi_sum, 'mig': mig}

    def forward(self, data, seg_label, clust_label=None):
        """Run a batch of data through the forward function.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Returns
        -------
        dict
            Dictionary of outputs. Always contains:
            - 'filter_index': index mapping back to the original voxel set
            - 'iter_used': number of iterations actually completed
            - 'segmentation_iter0': segmentation logits from iteration 0
            When n_iterations > 1 and the event is non-degenerate:
            - 'segmentation': segmentation logits from the final iteration
            - all graph constructor outputs (edge scores, node predictions, ...)
        """
        # Only compute expensive clustering metrics on log steps
        do_metrics = (
            self.track_iter_metrics and
            ((self._forward_count + 1) % self.log_step == 0)
        )
        self._forward_count += 1

        # Filter the input down to the requested shapes
        data, seg_label, clust_label, index = self.filter_class(
                data, seg_label, clust_label)

        # Guard: if mask/crop augmentation emptied every batch entry of tracked
        # voxels, skip the entire forward pass — the embedder and graph
        # constructor cannot handle a completely empty SparseTensor.
        if data.tensor.shape[0] == 0:
            return {'filter_index': index, 'iter_used': 0, 'skip_backward': True}

        # --- Iteration 0: unconditional pass, no cluster feedback ---
        result = self.embedder(data, cluster_id_full=None)
        segmentation_iter0 = result.get('segmentation_iter0')
        if segmentation_iter0 is not None and not self.use_attention and self.n_iter > 1:
            # Detach iter-0 seg so only the final pass contributes gradient,
            # making n_iter>1/use_attention=False equivalent to n_iter=1.
            # Guard n_iter>1: for single-pass models the detach would kill seg_loss.
            segmentation_iter0 = TensorBatch(
                segmentation_iter0.tensor.detach(), segmentation_iter0.counts)

        # Pre-fill all metric keys with NaN so the CSV schema is identical on
        # every row. Real values overwrite on log steps further below.
        if self.track_iter_metrics:
            result.update({k: float('nan') for k in self._metric_keys})

        # Coordinates are fixed across all iterations (only features change).
        coords = result['coordinates']
        coords = TensorBatch(coords.data[:, coords.coord_cols], coords.counts)

        seg_label_graph = self._get_graph_seg_label(seg_label, segmentation_iter0)
        graph = self.constructor(coords, self._get_features(result),
                                 seg_label_graph, clust_label)

        # Single-pass baseline: return immediately without any graph feedback.
        if self.n_iter == 1:
            
            if self.make_clusters or self.cluster_semantic_pool:
                with torch.no_grad():
                    clusts, clust_shapes = self.constructor.fit_predict(graph)
                if self.make_clusters:
                    result['clusts'] = clusts
                    result['clust_shapes'] = clust_shapes
            if do_metrics and 'node_label' in graph:
                w, b, r = self._compute_cohesion(
                        result['features'], graph['node_label'])
                result.update({'cohesion_within': w,
                               'cohesion_between': b,
                               'cohesion_ratio': r})
                result.update(self._compute_mi_metrics(
                        result['features'], seg_label))
            # node_pred (if computed above) is already in graph at this point,
            # so result/output (passed on to the loss) picks it up here.
            result.update(graph)
            result.update({'iter_used': 1, 'filter_index': index})

            if self.cluster_semantic_pool:
                node_pred = graph.get('node_pred')
                valid = node_pred.tensor >= 0

                segmentation_iter0_raw = segmentation_iter0.tensor
                with torch.no_grad():
                    node_argmax = torch.argmax(segmentation_iter0_raw, dim=1)
                    one_hot = F.one_hot(node_argmax, num_classes=segmentation_iter0_raw.shape[1]).float()
                    vote_counts = scatter_add(one_hot[valid], node_pred.tensor[valid], dim=0)
                    vote_fracs  = vote_counts / vote_counts.sum(dim=1, keepdim=True).clamp(min=1)

                    hard_vox = torch.zeros(node_pred.tensor.shape[0], vote_fracs.shape[1],
                                           dtype=segmentation_iter0_raw.dtype,
                                           device=segmentation_iter0_raw.device)
                    hard_vox[valid] = vote_fracs[node_pred.tensor[valid]]

                # STE: forward = vote-fraction soft probs, gradient flows via raw logits
                soft_probs = F.softmax(segmentation_iter0_raw, dim=1)
                pooled = soft_probs - soft_probs.detach() + hard_vox    
                # pooled = segmentation_iter0_raw - segmentation_iter0_raw.detach() + hard_vox
                segmentation_iter0 = TensorBatch(pooled, segmentation_iter0.counts)

            if segmentation_iter0 is not None:
                result['segmentation_iter0'] = segmentation_iter0
            return result

        # --- Iterations 1 .. n_iter-1: cluster-conditioned refinement ---
        intermediate_metrics = {}

        for k in range(1, self.n_iter):
            # fit_predict is needed for attention conditioning and/or metrics.
            # Call it at most once per iteration.
            if (self.use_attention or do_metrics) \
                    and 'node_pred' not in graph:
                with torch.no_grad():
                    self.constructor.fit_predict(graph)

            if self.use_attention:
                node_pred = graph.get('node_pred')

                # Degenerate event: no clusters found, stop early.
                if node_pred is None:
                    result.update(graph)
                    result.update({'iter_used': k, 'filter_index': index})
                    if segmentation_iter0 is not None:
                        result['segmentation_iter0'] = segmentation_iter0
                    if intermediate_metrics:
                        result.update(intermediate_metrics)
                    return result

                cluster_ids = node_pred.tensor.detach()

                # Per-voxel confidence: how far each edge score is from 0.5.
                # Near 0 = garbage clustering (uniform attention); near 1 = sharp
                # clusters (hard within-cluster masking).
                node_conf = (compute_node_confidence(graph, len(self.shapes))
                             if self.use_confidence_mask else None)
            else:
                # Ablation: iterate without attention conditioning.
                cluster_ids = None
                node_conf = None

            # Compute clustering metrics for graph from iteration k.
            # Stored as {metric}_iter{k} (e.g. ari_iter0, sbd_shower_iter0).
            # Only when ground-truth cluster labels are available.
            if do_metrics and 'node_label' in graph:
                with torch.no_grad():
                    m = self.constructor.evaluate(graph, mean=True)
                for metric_key, val in m.items():
                    intermediate_metrics[f'{metric_key}_iter{k}'] = val

                # Cohesion ratio: uses the embedder features from the CURRENT
                # result (iteration k) before they are cleared and replaced.
                w, b, r = self._compute_cohesion(
                        result['features'], graph['node_label'])
                intermediate_metrics.update({
                    f'cohesion_within_iter{k}':  w,
                    f'cohesion_between_iter{k}': b,
                    f'cohesion_ratio_iter{k}':   r,
                })

                # MI-based disentanglement for features of iteration k
                for key, val in self._compute_mi_metrics(
                        result['features'], seg_label).items():
                    intermediate_metrics[f'{key}_iter{k}'] = val

            result.clear()
            del result
            result = self.embedder(data, cluster_id_full=cluster_ids,
                                   node_confidence=node_conf)
            if self.track_iter_metrics:
                result.update({k: float('nan') for k in self._metric_keys})
            use_union = (
                self.constructor.union_spatial_fn is not None and
                self._forward_count >= self.feature_graph_warmup_steps
            )
            seg_label_graph = self._get_graph_seg_label(
                    seg_label, result.get('segmentation_iter_last'))
            graph = self.constructor(coords, self._get_features(result),
                                     seg_label_graph, clust_label,
                                     use_union_graph=use_union)

        # --- Finalize ---
        if self.make_clusters:
            with torch.no_grad():
                clusts, clust_shapes = self.constructor.fit_predict(graph)
            result['clusts'] = clusts
            result['clust_shapes'] = clust_shapes

        # Cohesion + MI for the final iteration (no suffix, mirrors final cluster
        # metrics from the loss which also have no suffix).
        if do_metrics and 'node_label' in graph:
            w, b, r = self._compute_cohesion(
                    result['features'], graph['node_label'])
            result.update({'cohesion_within': w,
                           'cohesion_between': b,
                           'cohesion_ratio': r})
            result.update(self._compute_mi_metrics(
                    result['features'], seg_label))

        result.update(graph)
        result.update({'iter_used': self.n_iter, 'filter_index': index})
        if segmentation_iter0 is not None:
            result['segmentation_iter0'] = segmentation_iter0
        if intermediate_metrics:
            result.update(intermediate_metrics)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


# Maps integer semantic-type label → human-readable name used in wandb keys.
# Only the four shapes retained by filter_class are included; LOWES_SHP and
# GHOST_SHP are never present after filtering and must not be logged.
_SHAPE_NAMES = {
    SHOWR_SHP: 'shower',
    TRACK_SHP: 'track',
    MICHL_SHP: 'michel',
    DELTA_SHP: 'delta',
}


class GraphSPICEIterLoss(torch.nn.Module):
    """Loss function for Graph-SPICE.

    For use in config:

    ..  code-block:: yaml

        model:
          name: graph_spice
          modules:
            graph_spice_loss:
              <Basic parameters>
              edge_loss:
                <Edge loss configuration block>

    See configuration files prefixed with `graph_spice` under the `config`
    directory for detailed examples of working configurations.

    See Also
    --------
    :class:`GraphSPICE`
    """

    def __init__(self, graph_spice, graph_spice_loss=None):
        """Intialize the Graph-SPICE loss.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        graph_spice_loss : dict
            Graph-SPICE loss configuration dictionary
        seg_loss: dict
            Segmentation loss configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Process the loss configuration
        self.process_graph_loss_config(**graph_spice_loss)
        self.process_seg_loss_config()

        # Process the main mode configuration for its crucial elements
        self.process_model_config(**graph_spice)

    def process_graph_loss_config(self, evaluate_clustering_metrics=False,
                                  w_edge=1.0, w_seg_iter0=0.3, w_seg_iter_last=0.5,
                                  w_contrastive=0.0, contrastive_loss=None,
                                  **graph_loss):
        """Process the loss configuration

        Parameters
        ----------
        evaluate_clustering_metrics : bool, default False
            If `True`, evaluates the clustering accuracy directly, rather than
            simply reporting an edge assignment acurracy
        w_contrastive : float, default 0.0
            Weight for the contrastive loss term (0 = disabled).
        contrastive_loss : dict, optional
            Config for ContrastiveLoss: keys ``contrastive_margin``,
            ``contrastive_n``, and ``contrastive_k_pairs``.
            contrastive_n is the number of features to sample for the contrastive loss.
            contrastive_k_pairs is the number of pairs to sample for the contrastive loss.
        **loss : dict
            Loss configuration dictionary
        """
        # Store basic parameters
        self.evaluate_clustering_metrics = evaluate_clustering_metrics
        self.w_edge          = w_edge
        self.w_seg_iter0     = w_seg_iter0
        self.w_seg_iter_last = w_seg_iter_last
        self.w_contrastive   = w_contrastive

        # Contrastive loss (only instantiated when enabled)
        if w_contrastive > 0:
            cfg = contrastive_loss or {}
            self.contrastive_loss_fn = loss_factory({
                'name': 'contrastive',
                'margin': cfg.get('contrastive_margin', 1.0),
                'k_pairs': cfg.get('contrastive_k_pairs', 1),
            })
            self.contrastive_n = cfg.get('contrastive_n', 256)
        else:
            self.contrastive_loss_fn = None
            self.contrastive_n = 256

        # Flatten the optional edge_loss sub-block into the top-level kwargs
        # so EdgeLoss receives e.g. hard_negative_mining directly.
        edge_loss_cfg = graph_loss.pop('edge_loss', {})
        graph_loss.update(edge_loss_cfg)

        # Initialize the edge loss function
        self.loss_fn = loss_factory(graph_loss)

    def process_seg_loss_config(self, loss='ce'):
        try:
            self.seg_loss_fn = loss_fn_factory(loss, reduction='none')
        except KeyError:
            raise ValueError(
                f"Unknown loss function `{loss}` provided. ")

    def process_model_config(self, constructor,
                             shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                             invert=True, cluster_semantic_pool=False, **kwargs):
        """Process the model configuration

        Parameters
        ----------
        constructor : dict, optional
            Edge index construction configuration
        shapes : List[int], default [0, 1, 2, 3]
            List of semantic shapes to run DBSCAN on
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        cluster_semantic_pool : bool, default False
            Mirrors the model's flag of the same name (received here via the
            shared graph_spice config block).
        """
        self.cluster_semantic_pool = cluster_semantic_pool
        # Initialize the graph constructor (used to produce node assignments)
        if self.evaluate_clustering_metrics or self.cluster_semantic_pool:
            self.constructor = ClusterGraphConstructor(
                    **constructor, shapes=shapes, invert=invert)

    def filter_class(self, seg_label, clust_label, filter_index):
        """Filter the list of pixels to those in the list of requested shapes.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor

        Parameters
        ----------
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : TensorBatch
            (M, 1 + D + N_c) Restricted tnesor of cluster labels
        """
        seg_label = TensorBatch(
                seg_label.tensor[filter_index.index], filter_index.counts)
        clust_label = TensorBatch(
                clust_label.tensor[filter_index.index], filter_index.counts)

        return seg_label, clust_label

    def forward(self, seg_label, clust_label, filter_index, segmentation_iter_last=None, segmentation_iter0=None,
                **output):
        """Run a batch of data through the loss function.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labelresul
            - N_c is is the number of cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor
        **output : dict
            Output of the Graph-SPICE model

        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Narrow down the labels to those corresponding to the relevant shapes
        seg_label, clust_label = self.filter_class(seg_label, clust_label, filter_index)
        seg_label_t = seg_label.tensor[:, VALUE_COL].long()
        # Pass the output through the loss function
        result = self.loss_fn(
                seg_label=seg_label, clust_label=clust_label, **output)
        result['w_edge'] = self.w_edge
        result['edge_loss'] = self.w_edge * result['loss']
        total_loss = self.w_edge * result['loss']

        # 2. SEGMENTATION ITER1 - Auxiliary (feature guidance)
        # Ensures base features are semantically discriminative
        if segmentation_iter0 is not None:
            segmentation_iter0_t = segmentation_iter0.tensor
            if self.cluster_semantic_pool:
                cluster_id = output['node_pred'].tensor.clone()
                num_real_clusters = int(cluster_id.max().item()) + 1
                # Voxels with no cluster (e.g. predicted-LowE, excluded from
                # every shape's subgraph) still get trained on: fold them into
                # pseudo-clusters instead of dropping them, so they contribute
                # the same 1/n_present aggregate share as any other present
                # class. Bucket them by their OWN true class first -- a single
                # shared pseudo-cluster would average together voxels of
                # different true classes (e.g. a mispredicted-track voxel and
                # a genuinely-LowE voxel), producing a meaningless mixed label.
                no_cluster = cluster_id < 0
                if no_cluster.any():
                    _, remap = seg_label_t[no_cluster].unique(return_inverse=True)
                    cluster_id[no_cluster] = num_real_clusters + remap
                    num_clusters = num_real_clusters + int(remap.max().item()) + 1
                else:
                    num_clusters = num_real_clusters

                cluster_size = scatter_add(torch.ones_like(cluster_id, dtype=torch.float32),
                                            cluster_id, dim=0, dim_size=num_clusters)
                cluster_label = scatter_mean(seg_label_t.float(), cluster_id, dim=0,
                                            dim_size=num_clusters).round().long()
                weights = torch.zeros_like(seg_label_t, dtype=torch.float32)
                present_classes = cluster_label.unique()
                n_present = present_classes.shape[0]
                for k in present_classes:
                    is_cluster_k = cluster_label == k
                    n_clusters_k = is_cluster_k.sum().item()
                    is_voxel_k = is_cluster_k[cluster_id]
                    weights[is_voxel_k] = 1 / (cluster_size[cluster_id][is_voxel_k] * n_clusters_k * n_present)
                seg_loss_1, seg_acc_1, acc_class_1, _ = self.get_loss_accuracy(
                        segmentation_iter0_t, seg_label_t, weights)

            else:
                seg_loss_1, seg_acc_1, acc_class_1, _ = self.get_loss_accuracy(
                        segmentation_iter0_t, seg_label_t)
            result['seg_loss_iter0'] = seg_loss_1 * self.w_seg_iter0
            result['seg_acc_iter0'] = seg_acc_1
            result['w_seg_iter0'] = self.w_seg_iter0
            for shp_id, shp_name in _SHAPE_NAMES.items():
                if shp_id < len(acc_class_1):
                    result[f'seg_acc_{shp_name}_iter1'] = float(acc_class_1[shp_id])

            # Auxiliary weight (smaller)
            total_loss += self.w_seg_iter0 * seg_loss_1

        # 3. SEGMENTATION ITER2 - Auxiliary (attention reward)
        # Rewards attention for improving semantic predictions
        if segmentation_iter_last is not None:
            segmentation_t = segmentation_iter_last.tensor
            seg_loss_2, seg_acc_2, acc_class_2, _ = self.get_loss_accuracy(
                    segmentation_t, seg_label_t)
            result['seg_loss_iter_last'] = seg_loss_2 * self.w_seg_iter_last
            result['seg_acc_iter_last'] = seg_acc_2
            result['w_seg_iter_last'] = self.w_seg_iter_last
            for shp_id, shp_name in _SHAPE_NAMES.items():
                if shp_id < len(acc_class_2):
                    result[f'seg_acc_{shp_name}_iter2'] = float(acc_class_2[shp_id])

            # Auxiliary weight (moderate)
            total_loss += self.w_seg_iter_last * seg_loss_2

        # 4. CONTRASTIVE LOSS in FEATURE SPACE
        if self.w_contrastive > 0 and self.contrastive_loss_fn is not None:
            c_loss = self.contrastive_loss_fn(
                output['features'].tensor, output['node_label'].tensor,
                self.contrastive_n)
            result['contrastive_loss'] = float(c_loss)
            total_loss += self.w_contrastive * c_loss

        result.update({'loss': total_loss})
        
        # If requested, compute clustering metrics
        if self.evaluate_clustering_metrics:
            with torch.no_grad():
            # Assign cluster IDs to each of the input points, if not yet done
                if 'node_pred' not in output:
                    self.constructor.fit_predict(output)
                    # Evaluate clustering metrics
                metrics = self.constructor.evaluate(output, mean=True)
            # Append metrics to the result dictionary
            result.update(metrics)

        return result

    def get_loss_accuracy(self, logits, labels, weights=None):
        """Computes the loss, global and classwise accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            (N, N_c) Output logits from the network for each voxel
        labels : torch.Tensor
            (N) Target values for each voxel
        weights : torch.Tensor, optional
            (N) Tensor of weights for each pixel in the batch

        Returns
        -------
        torch.Tensor
            Cross-entropy loss value
        float
            Global accuracy
        np.ndarray
            (N_c) Vector of class-wise accuracy
        torch.Tensor
            (N) Updated set of weights for each pixel in the batch
        """
        # If there is no input, nothing to do
        num_classes = logits.shape[1]
        if not len(logits):
            return 0., 1., np.ones(num_classes, dtype=np.float32), weights

        # Count the number of voxels in each class
        counts = torch.empty(num_classes,
                dtype=torch.long, device=labels.device)
        for c in range(num_classes):
            counts[c] = torch.sum(labels == c).item()

        # Compute the loss
        if weights is None:
            if hasattr(self.seg_loss_fn, 'lambda_dice'):  # Check if it's a CE_DICE_Loss
                ce_losses, dice_loss = self.seg_loss_fn(logits, labels)
                loss = ce_losses.mean() + self.loss_fn.lambda_dice * dice_loss
            else:
                loss = self.seg_loss_fn(logits, labels).mean()
        else:
            if hasattr(self.seg_loss_fn, 'lambda_dice'):  # Check if it's a CE_DICE_Loss
                ce_losses, dice_loss = self.seg_loss_fn(logits, labels)
                # Only weight the CE part with sample weights
                loss = (weights * ce_losses).sum() / weights.sum() + self.loss_fn.lambda_dice * dice_loss
            else:
                loss = (weights * self.seg_loss_fn(logits, labels)).sum() / weights.sum()

        # Compute the accuracies
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)

            # Per-class prediction accuracy
            acc_class = np.ones(num_classes, dtype=np.float32)
            for c in range(num_classes):
                if counts[c] > 0:
                    mask = torch.nonzero(labels == c).flatten()
                    acc_class[c] = (preds[mask] == c).sum().item() / counts[c]

            # Global prediction accuracy
            acc = (preds == labels).sum().item() / torch.sum(counts).item()

        return loss, acc, acc_class, weights
