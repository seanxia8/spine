"""Class and methods to convert the output of Graph-SPICE into a set of
pixel cluster assignments using a graph.
"""

import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph

from spine.constants import CLUST_COL, SHAPE_COL
from spine.constants.factory import enum_factory
from spine.data import IndexBatch, ObjectList, TensorBatch
from spine.utils.gnn.cluster import form_clusters
from spine.utils.metrics import ari, eff, pur, sbd

from .ccc import ConnectedComponentClusterer

__all__ = ["ClusterGraphConstructor"]


def radius_topk_graph(x, k, r, batch=None, loop=False, max_search_neighbors=None):
    """The k closest neighbors within radius r -- nothing beyond r, ever.

    Unlike `radius_graph`'s own `max_num_neighbors` cap, which (per the
    torch_cluster docstring) picks an arbitrary/random subset once a node
    has more candidates than the cap, this explicitly sorts every candidate
    within r by distance and keeps the k closest. A node with fewer than k
    true neighbors within r simply ends up with fewer than k edges -- no
    padding from outside the radius is ever considered.

    Parameters
    ----------
    x : torch.Tensor
        (N, D) Point coordinates
    k : int
        Maximum number of neighbors to keep per node
    r : float
        Radius within which candidates are considered at all
    batch : torch.Tensor, optional
        (N) Batch index per point
    loop : bool, default False
        If `True`, include self-loops
    max_search_neighbors : int, optional
        Candidate cap passed to the underlying radius search (not the final
        per-node edge count). Must be large enough to capture the true local
        density, or the closest-k selection below inherits the same
        random-truncation issue one level removed. Defaults to
        ``max(10 * k, 64)``.

    Returns
    -------
    torch.Tensor
        (2, E) Edge index of the k-nearest-within-r graph
    """
    search_cap = max_search_neighbors or max(10 * k, 64)
    edge_index = radius_graph(
        x, r, batch=batch, loop=loop, max_num_neighbors=search_cap)
    if edge_index.shape[1] == 0:
        return edge_index

    src, dst = edge_index
    dist = (x[src] - x[dst]).norm(dim=1)

    # Sort by distance, then a *stable* sort by source node: this groups
    # edges by source while preserving the ascending-distance order within
    # each group.
    order = torch.argsort(dist)
    order = order[torch.argsort(src[order], stable=True)]
    src_grouped = src[order]

    # Rank of each edge within its source node's group (0 = closest),
    # computed without any Python-level loop over groups.
    _, counts = torch.unique_consecutive(src_grouped, return_counts=True)
    group_starts = torch.cumsum(counts, 0) - counts
    rank = (torch.arange(src_grouped.shape[0], device=x.device)
            - torch.repeat_interleave(group_starts, counts))

    return edge_index[:, order[rank < k]]


class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type graph
    construction and node predictions in Graph-SPICE clustering.
    """

    def __init__(
        self,
        graph,
        shapes,
        edge_threshold,
        kernel_fn=None,
        edge_proj=None,
        min_size=0,
        invert=True,
        label_edges=False,
        target_col=CLUST_COL,
        training=False,
        orphan=None,
        union_graph=None,
    ):
        """Initialize the cluster graph constructor.

        Parameters
        ----------
        graph : dict
            Graph construction configuration dictionary
        shapes : List[str]
            List of shape names to construct clusters for
        edge_threshold : float
            Edge score below which it is disconnected (or above which it is,
            if the `inverted` parameter is turned on
        kernel_fn : callable, optional
            Kernel function computing edge scores from edge features
        min_size : int, default 0
            Minimum number of points below which pixels are considered orphans
            to be merged into touching larger clusters
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        label_edges : bool, default False
            If `True`, use cluster labels to label the edges as on or off
        target_col : int, default CLUST_COL
            Index of the column which specifies the label cluster ID for each point
        training : bool, default False
            If `True`, this constructor is being used at train time
        orphan : dict, optional
            Orphan clustering configuration dictionary
        union_graph : dict, optional
            When provided, completely replaces ``graph`` for refinement
            iterations (k >= 1) once the feature-graph warmup is over.
            Must contain two sub-configs:
              - ``spatial``: kNN/radius config for the coordinate-space edges
              - ``feature``: kNN/radius config for the feature-space edges
            The two edge sets are unioned and deduplicated before kernel
            scoring.  ``graph`` is only used during warmup (or if
            ``union_graph`` is absent), so there is no redundant graph build.
            Example::

                union_graph:
                  spatial:
                    name: knn
                    k: 5
                  feature:
                    name: knn
                    k: 5
                    cosine: true

        Raises
        ------
        ValueError
            If the graph type is not supported.
        """
        # Parse the set of shapes to cluster
        self.shapes = enum_factory("shape", shapes)

        # Store other basic properties
        self.threshold = edge_threshold
        self.min_size = min_size
        self.invert = invert
        self.label_edges = label_edges
        self.kernel_fn          = kernel_fn
        self.edge_proj          = edge_proj
        self.target_col = target_col

        def _build_graph_fn(cfg, label):
            name = cfg.get("name")
            assert name, f"Must provide 'name' in {label} config."
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            if name == "knn":
                return partial(knn_graph, **kwargs)
            elif name == "radius":
                return partial(radius_graph, **kwargs)
            elif name == "radius_topk":
                return partial(radius_topk_graph, **kwargs)
            raise ValueError(
                f"{label} graph name '{name}' not recognised. "
                "Must be 'knn', 'radius', or 'radius_topk'."
            )

        # Base graph — used during warmup or when union_graph is absent
        self.graph_fn = _build_graph_fn(graph, "graph")

        # Union graph — two sub-fns built from spatial + feature sub-configs.
        # When active it completely replaces graph_fn; no redundant build.
        self.union_spatial_fn = None
        self.union_feature_fn = None
        if union_graph is not None:
            assert "spatial" in union_graph and "feature" in union_graph, (
                "union_graph must contain both 'spatial' and 'feature' sub-configs."
            )
            self.union_spatial_fn = _build_graph_fn(union_graph["spatial"],
                                                     "union_graph.spatial")
            self.union_feature_fn = _build_graph_fn(union_graph["feature"],
                                                     "union_graph.feature")

        # Initialize the cluster assignment class
        self.ccc = ConnectedComponentClusterer(min_size, orphan)

    def __call__(self, coords, features, seg_label, clust_label=None,
                 use_union_graph=False):
        """Constructs graphs for all the entries in a batch, one per shape.

        Parameters
        ----------
        coords : TensorBatch
            (N, 3) Point coordinates
        features : TensorBatch
            (N, N_f) Set of graph embeddings
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        use_union_graph : bool, default False
            If `True`, uses the union_graph to construct the graph
        """
        # If edge labeling is required, make sure clust_label is provided
        assert (
            not self.label_edges or clust_label is not None
        ), "If edge labels are to be produced, must provide `clust_label`."

        # Loop over the unique batch indices, build a list of graphs for each
        graph = defaultdict(list)
        edge_offset = 0
        edge_counts, edge_offsets = [], []
        for b in range(coords.batch_size):
            # Build graphs (one per semantic type)
            clust_label_b = clust_label[b] if clust_label is not None else None
            graphs_b, edge_count = self.build_graph(
                coords[b], features[b], seg_label[b], clust_label_b,
                use_union_graph=use_union_graph,
            )

            # Append the output
            edge_counts.append(edge_count)
            edge_offsets.append(edge_offset)
            for key, value in graphs_b.items():
                if key.endswith("clusts"):
                    is_edge = key.startswith("edge")
                    offset = edge_offset if is_edge else coords.edges[b]
                    for c in value:
                        graph[key].append(offset + c)

                else:
                    graph[key].append(value)

            # Increment the offset
            edge_offset += edge_count

        # Concatenate the graph attributes together
        is_tensor = isinstance(coords.tensor, torch.Tensor)
        cat = torch.cat if is_tensor else np.concatenate
        for key, value in graph.items():
            if key.endswith("clusts"):
                # Turn indexes into index batches
                counts = [len(self.shapes)] * coords.batch_size
                single_counts = [len(c) for c in value]
                is_edge = key.startswith("edge")
                spans = edge_counts if is_edge else coords.counts
                graph[key] = IndexBatch(value, spans, counts, single_counts)

            else:
                # Turn edge index/attributes into tensor batches
                value = cat(value)
                graph[key] = TensorBatch(value, edge_counts)

        # Add the input node information to the graph
        graph["node_coords"] = coords
        graph["node_features"] = features
        graph["node_shapes"] = TensorBatch(
            seg_label.tensor[:, SHAPE_COL], seg_label.counts
        )
        if clust_label is not None:
            graph["node_label"] = TensorBatch(
                clust_label.tensor[:, self.target_col], clust_label.counts
            )

        return graph

    def build_graph(self, coords, features, seg_label, clust_label=None,
                    use_union_graph=False):
        """Construct a graph for a single batch id and semantic class that
        will be used for connected components clustering.

        Parameters
        ----------
        coords : torch.Tensor
            (N, 3) Tensor of point coordinates
        features : torch.Tensor
            (N, N_f) Graph embedding features to be used for edge prediction
        seg_clusts : List[List[int]]
            (S) One pixel index per semantic type
        seg_label : torch.Tensor
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : torch.Tensor, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        use_union_graph : bool, default False
            If `True`, uses the feature-space kNN graph to construct the graph
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of graph properties
        """
        # Loop over the semantic types, build a graph for each
        graph = defaultdict(list)
        edge_count = 0
        for s in self.shapes:
            # Get the index of points which belong to this class
            seg_index = torch.where(seg_label[:, SHAPE_COL] == s)[0]
            graph["node_clusts"].append(seg_index)

            # If there are no points, append empty, proceed
            if not len(seg_index):
                graph["edge_clusts"].append(
                    torch.empty(0, dtype=torch.long, device=coords.device)
                )
                graph["edge_index"].append(
                    torch.empty((0, 2), dtype=torch.long, device=coords.device)
                )
                graph["edge_shape"].append(
                    torch.empty(0, dtype=torch.long, device=coords.device)
                )
                graph["edge_attr"].append(
                    torch.empty(0, dtype=features.dtype, device=features.device)
                )
                if self.label_edges:
                    graph["edge_label"].append(
                        torch.empty(0, dtype=torch.long, device=coords.device)
                    )
                continue

            coords_s = coords[seg_index]
            if use_union_graph and self.union_spatial_fn is not None:
                # Post-warmup union mode: build spatial + feature entirely from
                # union_graph sub-configs.  graph_fn is NOT called — no waste.
                edge_index = self.union_spatial_fn(coords_s)
                with torch.no_grad():
                    feat_ei = self.union_feature_fn(features[seg_index])
                edge_index = torch.unique(
                    torch.cat([edge_index, feat_ei], dim=1), dim=1)
            else:
                # Warmup or no union_graph configured: base spatial kNN only
                edge_index = self.graph_fn(coords_s)

            graph["edge_clusts"].append(
                edge_count + torch.arange(edge_index.shape[1], device=coords.device)
            )
            edge_count += edge_index.shape[1]

            # Spatial distance per edge — new information for the kernel since it
            # only sees feature vectors.  For pure spatial-kNN edges this is
            # bounded by the kNN radius; for long-range feature-kNN-only edges it
            # can be much larger, letting the kernel calibrate its confidence.
            spatial_dist = (
                coords_s[edge_index[0]] - coords_s[edge_index[1]]
            ).norm(dim=1).detach()

            # Produce edge predictions.
            features_s = features[seg_index]
            if self.edge_proj is not None:
                features_s = self.edge_proj(features_s)
            edge_attr = self.kernel_fn(
                features_s[edge_index[0]], features_s[edge_index[1]],
                spatial_dist=spatial_dist,
            )

            # Append
            graph["edge_index"].append(edge_index.T)
            graph["edge_shape"].append(
                torch.full(
                    (edge_index.shape[1],), s, dtype=torch.long, device=coords.device
                )
            )
            graph["edge_attr"].append(edge_attr.flatten())

            if self.label_edges:
                node_label = clust_label[seg_index, self.target_col]
                edge_label = node_label[edge_index[0]] == node_label[edge_index[1]]
                graph["edge_label"].append(edge_label.long())

        # Concatenate the graph attributes
        for key, value in graph.items():
            if key != "node_clusts" and key != "edge_clusts":
                graph[key] = torch.cat(value)

        # Convert edge logits to sigmoid scores
        graph["edge_prob"] = torch.sigmoid(graph["edge_attr"])

        return graph, edge_count

    def fit_predict(self, graph, edge_mode="edge_pred", threshold=None, min_size=None):
        """Perform connected components clustering on a batch.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        edge_mode : str, default 'edge_pred'
            Attribute of the graph used to get the edge status
        threshold : float, optional
            Override the edge score threshold set in the configuration
        min_size : int, optional
            Override the minimum cluster size set in the configuration

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Node assignments
        """
        # No gradients through this prediction
        with torch.no_grad():
            # Assign edge predictions based on the edge scores
            threshold = threshold if threshold is not None else self.threshold
            if self.invert:
                edge_pred = (graph["edge_prob"].tensor <= threshold).long()
            else:
                edge_pred = (graph["edge_prob"].tensor >= threshold).long()

            graph["edge_pred"] = TensorBatch(edge_pred, graph["edge_prob"].counts)

            # Assign each node to a cluster
            node_pred = self.ccc(
                graph["node_coords"],
                graph["edge_index"],
                graph[edge_mode],
                graph["node_clusts"],
                graph["edge_clusts"],
            )

            graph["node_pred"] = node_pred

            # Loop over entries in the batch, build fragments
            node_clusts = graph["node_clusts"]
            clusts, counts, single_counts, shapes = [], [], [], []
            for b in range(node_pred.batch_size):
                # Loop over shapes in the entry
                counts_b = 0
                for s, shape in enumerate(self.shapes):
                    # Get the list of clusters for this (entry, shape) pair
                    index_b_s = node_clusts[b][s]
                    clusts_b_s, counts_b_s = form_clusters(
                        node_pred[b][index_b_s, None], column=0
                    )

                    # Offset the cluster indexes appropriately, append
                    for i, c in enumerate(clusts_b_s):
                        clusts_b_s[i] = node_pred.edges[b] + index_b_s[c]

                    # Append
                    clusts.extend(clusts_b_s)
                    single_counts.extend(counts_b_s)
                    shapes.append(shape * np.ones(len(clusts_b_s), dtype=int))
                    counts_b += len(clusts_b_s)

                counts.append(counts_b)

            # Make an IndexBatch out of the list
            clusts = IndexBatch(clusts, node_pred.counts, counts, single_counts)
            clust_shapes = TensorBatch(np.concatenate(shapes), counts)

            # Return
            return clusts, clust_shapes

    def evaluate(self, graph, mean=False):
        """Evaluate the clustering accuracy of a graph.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        mean : bool, default False
            If `True`, returns the batch-averaged metric values

        Returns
        -------
        dict
            Dictionary of accuracy metrics
        """
        # No gradients through this evaluation
        result = defaultdict(list)
        metrics = {"ari": ari, "purity": pur, "efficiency": eff, "sbd": sbd}
        batch_size = graph["node_coords"].batch_size
        with torch.no_grad():
            # Loop over the batches
            for b in range(batch_size):
                # Get the node predictions and labels (convert to numpy for numba metrics)
                node_label_b = graph["node_label"][b].cpu().numpy().astype(np.int64)
                node_pred_b = graph["node_pred"][b].cpu().numpy().astype(np.int64)

                # Compute shape-agnostic metrics
                for m, metric in metrics.items():
                    result[m].append(metric(node_pred_b, node_label_b))

                # Loop over the semantic types
                for s, shape in enumerate(self.shapes):
                    # Narrow down the predictions and labels to this shape
                    node_index = graph["node_clusts"][b][s].cpu().numpy()
                    node_label_b_s = node_label_b[node_index]
                    node_pred_b_s = node_pred_b[node_index]

                    # If there are no points of this type, skip metric
                    # computation (absent class — not a failure).
                    if not len(node_index):
                        for m in metrics:
                            result[f"{m}_{shape}"].append(float('nan'))
                        continue

                    # Otherwise, compute the metrics
                    for m, metric in metrics.items():
                        result[f"{m}_{shape}"].append(
                            metric(node_pred_b_s, node_label_b_s)
                        )

        # Compute batch averaged metrics, return.
        # Use nanmean so that absent-class entries (nan) are excluded rather
        # than poisoning the average; all-nan → nan (logged but filtered by wandb).
        if mean:
            for key, value in result.items():
                result[key] = np.nanmean(value)

        return result

    @staticmethod
    def get_entry(graph, batch_id, semantic_id):
        """Narrow down the graph to one specific (batch_id, shape) pair.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        batch_id : int
            Batch index
        semantic_id : int
            Semantic type

        Returns
        -------
        dict
            Dictionary of graph attributes for one (batch_id, shape) pair
        """
        # Loop over graph keys, narrow down wherever relevant
        single_graph = {}
        node_index = graph["node_clusts"][batch_id][semantic_id]
        edge_index = graph["edge_clusts"][batch_id][semantic_id]
        for key, value in graph.items():
            if key.endswith("clusts"):
                continue
            elif key.startswith("node"):
                single_graph[key] = graph[key][b][node_index]
            elif key.startswith("edge"):
                single_graph[key] = graph[key][b][edge_index]
            else:
                raise KeyError(f"Graph key not recognized: {key}")

        return single_graph
