"""Iterative Graph-SPICE: cluster-conditioned refinement of dense clustering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch_scatter import scatter_add, scatter_mean

from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP
from spine.constants.factory import enum_factory
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory

from ..registry import ModelSpec
from .attn_embedder import AttnGraphSPICEEmbedder
from .constructor import ClusterGraphConstructor
from .factories import kernel_factory, loss_factory
from .model import GraphSPICE, GraphSPICELoss

__all__ = ["GraphSPICEIter", "GraphSPICEIterLoss", "compute_node_confidence"]

DEFAULT_SHAPES = ("shower", "track", "michel", "delta")

# Names of the clustered shapes used in per-shape accuracy keys
SHAPE_NAMES = {
    SHOWR_SHP: "shower",
    TRACK_SHP: "track",
    MICHL_SHP: "michel",
    DELTA_SHP: "delta",
}


def compute_node_confidence(graph: dict[str, Any], num_shapes: int) -> torch.Tensor:
    """Compute per-voxel clustering confidence from edge probabilities.

    For each voxel, ``confidence = mean_e(|edge_prob_e - 0.5| * 2)`` over its
    graph edges: 0 when every edge is maximally uncertain, 1 when every edge
    is fully decided. Voxels without edges get 0.

    Parameters
    ----------
    graph : dict
        Output of :class:`ClusterGraphConstructor`, with ``edge_prob``,
        ``edge_index``, ``node_clusts``, ``edge_clusts`` and ``node_pred``.
    num_shapes : int
        Number of semantic shapes in the graph.

    Returns
    -------
    torch.Tensor
        (N,) per-voxel confidence in [0, 1].
    """
    edge_prob = graph["edge_prob"].tensor
    edge_index = graph["edge_index"].tensor
    node_clusts = graph["node_clusts"]
    edge_clusts = graph["edge_clusts"]

    num_nodes = graph["node_pred"].tensor.shape[0]
    conf_sum = torch.zeros(num_nodes, device=edge_prob.device)
    conf_cnt = torch.zeros(num_nodes, device=edge_prob.device)
    for b in range(node_clusts.batch_size):
        # Per-entry access strips the batch offsets, add them back
        node_off = int(node_clusts.offsets[b])
        edge_off = int(edge_clusts.offsets[b])
        for s in range(num_shapes):
            local_nodes = node_clusts[b][s]
            local_edges = edge_clusts[b][s]
            if local_edges.numel() == 0:
                continue

            edges = local_edges + edge_off
            edge_conf = (edge_prob[edges] - 0.5).abs().mul_(2)
            local_ei = edge_index[edges]
            src = local_nodes[local_ei[:, 0]] + node_off
            dst = local_nodes[local_ei[:, 1]] + node_off

            conf_sum.scatter_add_(0, src, edge_conf)
            conf_sum.scatter_add_(0, dst, edge_conf)
            conf_cnt.scatter_add_(0, src, torch.ones_like(edge_conf))
            conf_cnt.scatter_add_(0, dst, torch.ones_like(edge_conf))

    return conf_sum.div_(conf_cnt.clamp_(min=1e-8))


class GraphSPICEIter(torch.nn.Module):
    """Graph-SPICE with iterative, cluster-conditioned refinement.

    Iteration 0 is a plain Graph-SPICE pass. Each following iteration feeds
    the clusters predicted by the previous one back into the embedder, whose
    decoder attends within those clusters, and rebuilds the graph.

    .. code-block:: yaml

        model:
          provider: graph_spice_iter
          modules:
            graph_spice:
              n_iterations: 2
              use_attention: true
              embedder:
                uresnet_attn:
                  <Backbone + attention configuration block>
              kernel:
                <Edge kernel function configuration block>
              constructor:
                <Graph construction configuration block>
            graph_spice_loss:
              <Edge loss and auxiliary segmentation loss weights>
    """

    filter_class = GraphSPICE.filter_class

    def __init__(self, graph_spice: dict[str, Any]) -> None:
        """Initialize the model.

        Parameters
        ----------
        graph_spice : dict
            Iterative Graph-SPICE configuration dictionary
        """
        super().__init__()
        self.process_model_config(**graph_spice)

    @staticmethod
    def _build_edge_proj(cfg: dict[str, Any] | None, num_features: int):
        """Build the optional MLP applied to node features before the kernel.

        Parameters
        ----------
        cfg : dict, optional
            ``n_layers`` (0 or missing disables it) and ``hidden_dim``
            (default ``num_features``)
        num_features : int
            Input/output dimension, matching the kernel
        """
        if cfg is None or cfg.get("n_layers", 0) <= 0:
            return None

        n = cfg["n_layers"]
        hidden = cfg.get("hidden_dim", num_features)
        layers = []
        for i in range(n):
            in_d = num_features if i == 0 else hidden
            out_d = num_features if i == n - 1 else hidden
            layers.append(torch.nn.Linear(in_d, out_d))
            if i < n - 1:
                layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers)

    def process_model_config(
        self,
        embedder: dict[str, Any],
        kernel: dict[str, Any],
        constructor: dict[str, Any],
        shapes: list[str] | tuple[str, ...] = DEFAULT_SHAPES,
        use_raw_features: bool = False,
        invert: bool = True,
        make_clusters: bool = False,
        n_iterations: int = 2,
        use_attention: bool = True,
        use_confidence_mask: bool = True,
        do_iterate: bool | None = None,
        edge_projection: dict[str, Any] | None = None,
        track_iter_metrics: bool = False,
        log_step: int = 1,
        feature_graph_warmup_steps: int = 0,
        cluster_semantic_pool: bool = False,
        use_pred_sem_filter: bool = False,
        pred_sem_filter_warmup_steps: int = 20000,
    ) -> None:
        """Initialize the underlying modules.

        Parameters
        ----------
        embedder : dict
            Attention embedder configuration
        kernel : dict
            Edge kernel configuration
        constructor : dict
            Graph constructor configuration (may include ``union_graph``)
        shapes : sequence of str
            Shapes to construct clusters for
        use_raw_features : bool, default False
            Use the embedder features as is, without the output layers
        invert : bool, default True
            Invert the edge scores so that 0 is on and 1 is off
        make_clusters : bool, default False
            If `True`, builds a list of cluster indexes
        n_iterations : int, default 2
            Total number of embedder passes. 1 is the single-pass baseline,
            2 is one refinement step, more is a convergence study.
        use_attention : bool, default True
            Condition each refinement pass on the previous clusters. `False`
            keeps the iterations without attention (ablation).
        use_confidence_mask : bool, default True
            Scale the attention conditioning by per-voxel edge confidence.
        do_iterate : bool, optional
            Deprecated alias: `False` sets 1 iteration, `True` sets 2.
        edge_projection : dict, optional
            Node feature projection before the kernel, see
            :meth:`_build_edge_proj`.
        track_iter_metrics : bool, default False
            Log clustering, cohesion and mutual-information metrics for every
            iteration as ``{metric}_iter{k}`` (requires cluster labels).
        log_step : int, default 1
            Period, in forward passes, of the intermediate metrics.
        feature_graph_warmup_steps : int, default 0
            Forward passes before the constructor ``union_graph`` replaces the
            base graph in refinement iterations.
        cluster_semantic_pool : bool, default False
            Single pass only: replace the iteration-0 semantics by the
            per-cluster vote fractions (straight-through gradient).
        use_pred_sem_filter : bool, default False
            After ``pred_sem_filter_warmup_steps`` passes, gate the per-shape
            graphs with the predicted semantics instead of the true labels.
        pred_sem_filter_warmup_steps : int, default 20000
            Forward passes before switching to predicted-semantics gating.
        """
        if do_iterate is not None:
            n_iterations = 2 if do_iterate else 1

        self.n_iter = n_iterations
        self.use_attention = use_attention
        self.use_confidence_mask = use_confidence_mask
        self.feature_graph_warmup_steps = feature_graph_warmup_steps

        self.embedder = AttnGraphSPICEEmbedder(
            **embedder,
            use_raw_features=use_raw_features,
            enable_attention=use_attention,
        )
        self.kernel_fn = kernel_factory(kernel)
        self.edge_proj = self._build_edge_proj(edge_projection, kernel["num_features"])
        self.constructor = ClusterGraphConstructor(
            **deepcopy(constructor),
            kernel_fn=self.kernel_fn,
            edge_proj=self.edge_proj,
            shapes=shapes,
            invert=invert,
            training=self.training,
        )

        self.shapes = enum_factory("shape", shapes)
        self.cluster_semantic_pool = cluster_semantic_pool
        self.use_pred_sem_filter = use_pred_sem_filter
        self.pred_sem_filter_warmup_steps = pred_sem_filter_warmup_steps
        self.use_raw_features = use_raw_features
        self.invert = invert
        self.make_clusters = make_clusters
        self.track_iter_metrics = track_iter_metrics
        self.log_step = log_step
        self._forward_count = 0

        # All metric keys, so the CSV schema is identical on every row (NaN
        # placeholders off the logging cadence)
        eval_base = ["ari", "purity", "efficiency", "sbd"]
        eval_keys = eval_base + [
            f"{m}_{s}" for m in eval_base for s in self.constructor.shapes
        ]
        coh_mi = [
            "cohesion_within",
            "cohesion_between",
            "cohesion_ratio",
            "mi_mean",
            "mi_max",
            "mi_sum",
            "mig",
        ]
        self._metric_keys = list(coh_mi)
        for k in range(1, n_iterations):
            self._metric_keys += [f"{key}_iter{k}" for key in eval_keys + coh_mi]

    def _get_features(self, result: dict[str, Any]) -> TensorBatch:
        """Return the features used by the kernel."""
        if self.use_raw_features:
            return result["features"]
        return result["hypergraph_features"]

    def _get_graph_seg_label(
        self, seg_label: TensorBatch, segmentation: TensorBatch | None
    ) -> TensorBatch:
        """Return the semantics used to gate the per-shape graphs.

        The true labels during warmup; afterwards (if ``use_pred_sem_filter``)
        the argmax of the raw per-voxel segmentation logits of this pass.
        """
        use_pred = (
            self.use_pred_sem_filter
            and segmentation is not None
            and self._forward_count >= self.pred_sem_filter_warmup_steps
        )
        if not use_pred:
            return seg_label

        labels = seg_label.values.torch_tensor()
        with torch.no_grad():
            pred_shape = torch.argmax(segmentation.tensor, dim=1).to(labels.dtype)

        return TensorBatch(pred_shape, seg_label.counts)

    @torch.no_grad()
    def _compute_cohesion(
        self, features: TensorBatch, node_labels: TensorBatch, sample_size: int = 2048
    ) -> tuple[float, float, float]:
        """Compute the cluster cohesion of the embedding space.

        Returns the mean cosine similarity within clusters, between clusters,
        and ``within / |between|`` on up to ``sample_size`` random voxels.
        """
        feats = features.tensor.float()

        # Globally unique cluster IDs across batch entries
        all_ids, offset = [], 0
        for b in range(node_labels.batch_size):
            labels_b = node_labels[b]
            if not isinstance(labels_b, torch.Tensor):
                labels_b = torch.tensor(labels_b, dtype=torch.long, device=feats.device)
            labels_b = labels_b.long()
            if labels_b.shape[0] > 0:
                all_ids.append(labels_b + offset)
                offset += int(labels_b.max().item()) + 1
            else:
                all_ids.append(labels_b)

        cluster_ids = torch.cat(all_ids).to(feats.device)
        if feats.shape[0] > sample_size:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:sample_size]
            feats = feats[idx]
            cluster_ids = cluster_ids[idx]

        feats_n = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = feats_n @ feats_n.T
        same = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
        diag = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
        same = same & ~diag
        within_vals = sim[same]
        between_vals = sim[~same & ~diag]

        if within_vals.numel() == 0 and between_vals.numel() == 0:
            return float("nan"), float("nan"), float("nan")

        # Missing pair types count as zero similarity
        within = within_vals.mean().item() if within_vals.numel() else 0.0
        between = between_vals.mean().item() if between_vals.numel() else 0.0

        return within, between, within / (abs(between) + 1e-8)

    @torch.no_grad()
    def _compute_mi_metrics(
        self, features: TensorBatch, seg_label: TensorBatch, sample_size: int = 4096
    ) -> dict[str, float]:
        """Gaussian mutual information between embedding dimensions and semantics.

        Uses ``I(X_d; Y) ~ 0.5 * log(Var(X_d) / E_y[Var(X_d | Y = y)])`` and
        returns its mean, max and sum over dimensions, and the Mutual
        Information Gap (top two dimensions, normalized by ``H(Y)``).
        """
        feats = features.tensor.float()
        labels = seg_label.values.torch_tensor().long()
        if feats.shape[0] > sample_size:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:sample_size]
            feats = feats[idx]
            labels = labels[idx]

        eps = 1e-8
        N, D = feats.shape
        if N < 2:
            return {k: float("nan") for k in ("mi_mean", "mi_max", "mi_sum", "mig")}

        # nan_to_num before clamp: clamp passes NaN through
        var_total = torch.nan_to_num(feats.var(dim=0), nan=eps).clamp(min=eps)
        var_cond = torch.zeros(D, device=feats.device)
        unique_labels = labels.unique()
        for c in unique_labels:
            mask = labels == c
            n_c = mask.sum().item()
            if n_c < 2:
                continue
            var_cond += (n_c / N) * torch.nan_to_num(
                feats[mask].var(dim=0), nan=0.0
            ).clamp(min=0)
        var_cond = var_cond.clamp(min=eps)

        mi = (0.5 * (var_total.log() - var_cond.log())).clamp(min=0)

        counts = torch.stack([(labels == c).sum() for c in unique_labels]).float()
        p_y = counts / counts.sum()
        h_y = -(p_y * (p_y + eps).log()).sum().item()
        sorted_mi, _ = mi.sort(descending=True)
        mig = 0.0
        if sorted_mi.shape[0] >= 2 and h_y > eps:
            mig = ((sorted_mi[0] - sorted_mi[1]) / h_y).item()

        return {
            "mi_mean": mi.mean().item(),
            "mi_max": mi.max().item(),
            "mi_sum": mi.sum().item(),
            "mig": mig,
        }

    def _final_metrics(self, result, graph, seg_label, suffix=""):
        """Cohesion and mutual-information metrics of the current features."""
        within, between, ratio = self._compute_cohesion(
            result["features"], graph["node_label"]
        )
        metrics = {
            "cohesion_within": within,
            "cohesion_between": between,
            "cohesion_ratio": ratio,
            **self._compute_mi_metrics(result["features"], seg_label),
        }
        return {f"{k}{suffix}": v for k, v in metrics.items()}

    def _pool_semantics(self, segmentation: TensorBatch, node_pred: TensorBatch):
        """Replace voxel semantics with per-cluster vote fractions.

        The forward value is the vote fraction of the voxel's cluster; the
        gradient flows through the voxel's own softmax (straight-through).
        """
        logits = segmentation.tensor
        clusters = node_pred.tensor
        valid = clusters >= 0
        with torch.no_grad():
            one_hot = torch.nn.functional.one_hot(
                torch.argmax(logits, dim=1), num_classes=logits.shape[1]
            ).float()
            votes = scatter_add(one_hot[valid], clusters[valid], dim=0)
            fracs = votes / votes.sum(dim=1, keepdim=True).clamp(min=1)
            hard = torch.zeros(
                clusters.shape[0],
                fracs.shape[1],
                dtype=logits.dtype,
                device=logits.device,
            )
            hard[valid] = fracs[clusters[valid]]

        soft = torch.nn.functional.softmax(logits, dim=1)
        return TensorBatch(soft - soft.detach() + hard, segmentation.counts)

    def forward(
        self,
        data: TensorBatch,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
    ) -> dict[str, Any]:
        """Run a batch of data through the iterative model.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
        clust_label : ClusterLabelBatch, optional
            Cluster labels, needed for edge labels and clustering metrics

        Returns
        -------
        dict
            Graph outputs of the last iteration, ``filter_index``,
            ``iter_used``, ``segmentation_iter0`` and, with attention passes,
            ``segmentation_iter_last``. ``skip_backward`` is set when no voxel
            of the requested shapes is present.
        """
        # Only compute expensive clustering metrics on log steps
        do_metrics = (
            self.track_iter_metrics and (self._forward_count + 1) % self.log_step == 0
        )
        self._forward_count += 1

        data, seg_label, clust_label, index = self.filter_class(
            data, seg_label, clust_label
        )

        # Augmentation can empty every entry: nothing to embed or cluster
        if data.tensor.shape[0] == 0:
            return {"filter_index": index, "iter_used": 0, "skip_backward": True}

        cluster_labels = None
        if clust_label is not None:
            cluster_labels = clust_label.voxel_field(self.constructor.target_col)

        # Iteration 0: unconditional pass
        result = self.embedder(data, cluster_id_full=None)
        segmentation_iter0 = result.get("segmentation_iter0")
        if (
            segmentation_iter0 is not None
            and not self.use_attention
            and self.n_iter > 1
        ):
            # Only the final pass trains the semantics when iterating
            # without attention (equivalent to the single-pass baseline)
            segmentation_iter0 = TensorBatch(
                segmentation_iter0.tensor.detach(), segmentation_iter0.counts
            )
        if self.track_iter_metrics:
            result.update({k: float("nan") for k in self._metric_keys})

        # Coordinates are fixed across iterations, only features change
        coordinate_batch = result["coordinates"]
        coords = TensorBatch(
            coordinate_batch.tensor[:, coordinate_batch.coord_cols],
            coordinate_batch.counts,
        )
        graph = self.constructor(
            coords,
            self._get_features(result),
            self._get_graph_seg_label(seg_label, segmentation_iter0),
            cluster_labels,
        )

        # Iterations 1 .. n_iter - 1: cluster-conditioned refinement
        intermediate_metrics: dict[str, float] = {}
        for k in range(1, self.n_iter):
            if (self.use_attention or do_metrics) and "node_pred" not in graph:
                with torch.no_grad():
                    self.constructor.fit_predict(graph)

            cluster_ids, node_conf = None, None
            if self.use_attention:
                node_pred = graph.get("node_pred")
                if node_pred is None:
                    # Degenerate event: no clusters found, stop early
                    result.update(graph)
                    result.update({"iter_used": k, "filter_index": index})
                    if segmentation_iter0 is not None:
                        result["segmentation_iter0"] = segmentation_iter0
                    result.update(intermediate_metrics)
                    return result

                cluster_ids = node_pred.tensor.detach()
                if self.use_confidence_mask:
                    node_conf = compute_node_confidence(graph, len(self.shapes))

            # Metrics of the graph produced by the previous pass
            if do_metrics and "node_label" in graph:
                with torch.no_grad():
                    metrics = self.constructor.evaluate(graph, mean=True)
                for key, val in metrics.items():
                    intermediate_metrics[f"{key}_iter{k}"] = val
                intermediate_metrics.update(
                    self._final_metrics(result, graph, seg_label, suffix=f"_iter{k}")
                )

            result.clear()
            del result
            result = self.embedder(
                data, cluster_id_full=cluster_ids, node_confidence=node_conf
            )
            if self.track_iter_metrics:
                result.update({key: float("nan") for key in self._metric_keys})
            use_union = (
                self.constructor.union_spatial_fn is not None
                and self._forward_count >= self.feature_graph_warmup_steps
            )
            graph = self.constructor(
                coords,
                self._get_features(result),
                self._get_graph_seg_label(
                    seg_label, result.get("segmentation_iter_last")
                ),
                cluster_labels,
                use_union_graph=use_union,
            )

        # Semantic pooling only applies to the single-pass baseline
        pool = self.cluster_semantic_pool and self.n_iter == 1
        if self.make_clusters or pool:
            with torch.no_grad():
                clusts, clust_shapes = self.constructor.fit_predict(graph)
            if self.make_clusters:
                result["clusts"] = clusts
                result["clust_shapes"] = clust_shapes

        if do_metrics and "node_label" in graph:
            result.update(self._final_metrics(result, graph, seg_label))

        result.update(graph)
        result.update({"iter_used": self.n_iter, "filter_index": index})
        if pool:
            segmentation_iter0 = self._pool_semantics(
                segmentation_iter0, graph["node_pred"]
            )
        if segmentation_iter0 is not None:
            result["segmentation_iter0"] = segmentation_iter0
        result.update(intermediate_metrics)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


class GraphSPICEIterLoss(torch.nn.Module):
    """Loss of iterative Graph-SPICE.

    Weighted sum of the edge loss on the last graph, the semantic
    segmentation losses of the first (``w_seg_iter0``) and last
    (``w_seg_iter_last``) passes and an optional contrastive loss.

    .. code-block:: yaml

        graph_spice_loss:
          provider: edge
          loss: binary_log_dice_ce
          edge_loss:
            hard_negative_mining: false
          evaluate_clustering_metrics: true
          w_edge: 1.0
          w_seg_iter0: 0.3
          w_seg_iter_last: 0.5
          w_contrastive: 0.0
    """

    get_edge_labels = GraphSPICELoss.get_edge_labels
    filter_class = staticmethod(GraphSPICELoss.filter_class)

    def __init__(
        self, graph_spice: dict[str, Any], graph_spice_loss: dict[str, Any]
    ) -> None:
        """Initialize the loss.

        Parameters
        ----------
        graph_spice : dict
            Iterative Graph-SPICE configuration dictionary
        graph_spice_loss : dict
            Loss configuration dictionary
        """
        super().__init__()
        self.process_graph_loss_config(**graph_spice_loss)
        self.seg_loss_fn = loss_fn_factory("ce", reduction="none")
        self.process_model_config(**graph_spice)

    def process_graph_loss_config(
        self,
        evaluate_clustering_metrics: bool = False,
        w_edge: float = 1.0,
        w_seg_iter0: float = 0.3,
        w_seg_iter_last: float = 0.5,
        w_contrastive: float = 0.0,
        contrastive_loss: dict[str, Any] | None = None,
        edge_loss: dict[str, Any] | None = None,
        **graph_loss: Any,
    ) -> None:
        """Process the loss configuration.

        Parameters
        ----------
        evaluate_clustering_metrics : bool, default False
            If `True`, evaluate the clustering metrics of the final graph
        w_edge, w_seg_iter0, w_seg_iter_last, w_contrastive : float
            Weights of the edge, first-pass segmentation, last-pass
            segmentation and contrastive losses (0 disables contrastive)
        contrastive_loss : dict, optional
            ``contrastive_margin``, ``contrastive_k_pairs`` and
            ``contrastive_n`` (number of anchors)
        edge_loss : dict, optional
            Extra edge-loss parameters (e.g. ``hard_negative_mining``)
        **graph_loss : dict
            Edge loss configuration
        """
        self.evaluate_clustering_metrics = evaluate_clustering_metrics
        self.w_edge = w_edge
        self.w_seg_iter0 = w_seg_iter0
        self.w_seg_iter_last = w_seg_iter_last
        self.w_contrastive = w_contrastive

        self.contrastive_loss_fn = None
        self.contrastive_n = 256
        if w_contrastive > 0:
            cfg = contrastive_loss or {}
            self.contrastive_loss_fn = loss_factory(
                {
                    "provider": "contrastive",
                    "margin": cfg.get("contrastive_margin", 1.0),
                    "k_pairs": cfg.get("contrastive_k_pairs", 1),
                }
            )
            self.contrastive_n = cfg.get("contrastive_n", 256)

        self.loss_fn = loss_factory({**graph_loss, **(edge_loss or {})})

    def process_model_config(
        self,
        constructor: dict[str, Any],
        shapes: list[str] | tuple[str, ...] = DEFAULT_SHAPES,
        invert: bool = True,
        cluster_semantic_pool: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Build the constructor used for clustering metrics, if needed.

        ``cluster_semantic_pool`` mirrors the model flag: the first-pass
        segmentation loss is then balanced across predicted clusters.
        """
        self.target_col = constructor.get("target_col", "cluster")
        self.cluster_semantic_pool = cluster_semantic_pool
        self.constructor = None
        if self.evaluate_clustering_metrics:
            self.constructor = ClusterGraphConstructor(
                **deepcopy(constructor), shapes=shapes, invert=invert
            )

    def forward(
        self,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch,
        filter_index: IndexBatch,
        segmentation_iter_last: TensorBatch | None = None,
        segmentation_iter0: TensorBatch | None = None,
        **output: Any,
    ) -> dict[str, Any]:
        """Compute the combined loss.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
        clust_label : ClusterLabelBatch
            Cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor
        segmentation_iter_last, segmentation_iter0 : TensorBatch, optional
            Segmentation logits of the last and first passes
        **output : dict
            Other outputs of the model

        Returns
        -------
        dict
            Losses, accuracies and optional clustering metrics
        """
        seg_label, clust_label = self.filter_class(seg_label, clust_label, filter_index)
        seg_label_t = seg_label.values.torch_tensor().long()
        if "edge_label" not in output:
            output["edge_label"] = self.get_edge_labels(
                clust_label,
                output["edge_index"],
                output["node_clusts"],
                output["edge_clusts"],
            )

        result = self.loss_fn(seg_label=seg_label, clust_label=clust_label, **output)
        result["w_edge"] = self.w_edge
        result["edge_loss"] = self.w_edge * result["loss"]
        total_loss = self.w_edge * result["loss"]

        # Auxiliary semantics of the first pass (feature guidance)
        if segmentation_iter0 is not None:
            weights = None
            if self.cluster_semantic_pool:
                weights = self.cluster_balanced_weights(
                    output["node_pred"].tensor, seg_label_t
                )
            loss, acc, acc_class = self.get_loss_accuracy(
                segmentation_iter0.tensor, seg_label_t, weights
            )
            result["seg_loss_iter0"] = loss * self.w_seg_iter0
            result["seg_acc_iter0"] = acc
            result["w_seg_iter0"] = self.w_seg_iter0
            result.update(self.shape_accuracies(acc_class, "iter1"))
            total_loss += self.w_seg_iter0 * loss

        # Auxiliary semantics of the last pass (attention reward)
        if segmentation_iter_last is not None:
            loss, acc, acc_class = self.get_loss_accuracy(
                segmentation_iter_last.tensor, seg_label_t
            )
            result["seg_loss_iter_last"] = loss * self.w_seg_iter_last
            result["seg_acc_iter_last"] = acc
            result["w_seg_iter_last"] = self.w_seg_iter_last
            result.update(self.shape_accuracies(acc_class, "iter2"))
            total_loss += self.w_seg_iter_last * loss

        if self.contrastive_loss_fn is not None:
            c_loss = self.contrastive_loss_fn(
                output["features"].tensor,
                output["node_label"].tensor,
                self.contrastive_n,
            )
            result["contrastive_loss"] = float(c_loss)
            total_loss += self.w_contrastive * c_loss

        result["loss"] = total_loss

        if self.evaluate_clustering_metrics:
            with torch.no_grad():
                if "node_pred" not in output:
                    self.constructor.fit_predict(output)
                result.update(self.constructor.evaluate(output, mean=True))

        return result

    @staticmethod
    def shape_accuracies(acc_class: np.ndarray, suffix: str) -> dict[str, float]:
        """Per-shape accuracy entries ``seg_acc_{shape}_{suffix}``."""
        return {
            f"seg_acc_{name}_{suffix}": float(acc_class[shape])
            for shape, name in SHAPE_NAMES.items()
            if shape < len(acc_class)
        }

    @staticmethod
    def cluster_balanced_weights(
        cluster_id: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weights giving each present class, and each cluster within a class,
        the same total weight.

        Voxels outside any cluster are grouped into one pseudo-cluster per
        true class so they still contribute. A cluster's class is the rounded
        mean of its true labels.

        Parameters
        ----------
        cluster_id : torch.Tensor
            (N) Predicted cluster of each voxel (-1 if none)
        labels : torch.Tensor
            (N) True semantic class of each voxel
        """
        cluster_id = cluster_id.clone()
        num_clusters = int(cluster_id.max().item()) + 1
        no_cluster = cluster_id < 0
        if no_cluster.any():
            _, remap = labels[no_cluster].unique(return_inverse=True)
            cluster_id[no_cluster] = num_clusters + remap
            num_clusters += int(remap.max().item()) + 1

        cluster_size = scatter_add(
            torch.ones_like(cluster_id, dtype=torch.float32),
            cluster_id,
            dim=0,
            dim_size=num_clusters,
        )
        cluster_label = (
            scatter_mean(labels.float(), cluster_id, dim=0, dim_size=num_clusters)
            .round()
            .long()
        )
        weights = torch.zeros_like(labels, dtype=torch.float32)
        present = cluster_label.unique()
        for k in present:
            is_cluster_k = cluster_label == k
            is_voxel_k = is_cluster_k[cluster_id]
            weights[is_voxel_k] = 1 / (
                cluster_size[cluster_id][is_voxel_k]
                * is_cluster_k.sum().item()
                * present.shape[0]
            )

        return weights

    def get_loss_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | float, float, np.ndarray]:
        """Compute the (optionally weighted) CE loss, global and class accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            (N, N_c) Output logits from the network for each voxel
        labels : torch.Tensor
            (N) Target values for each voxel
        weights : torch.Tensor, optional
            (N) Tensor of weights for each pixel in the batch
        """
        num_classes = logits.shape[1]
        if len(logits) == 0:
            return 0.0, 1.0, np.ones(num_classes, dtype=np.float32)

        counts = torch.empty(num_classes, dtype=torch.long, device=labels.device)
        for c in range(num_classes):
            counts[c] = torch.sum(labels == c).item()

        point_loss = self.seg_loss_fn(logits, labels)
        if weights is None:
            loss = point_loss.mean()
        else:
            loss = (weights * point_loss).sum() / weights.sum()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc_class = np.ones(num_classes, dtype=np.float32)
            for c in range(num_classes):
                if counts[c] > 0:
                    mask = torch.nonzero(labels == c).flatten()
                    acc_class[c] = (preds[mask] == c).sum().item() / counts[c]
            acc = (preds == labels).sum().item() / torch.sum(counts).item()

        return loss, acc, acc_class


MODEL_SPEC = ModelSpec("graph_spice_iter", GraphSPICEIter, GraphSPICEIterLoss)
