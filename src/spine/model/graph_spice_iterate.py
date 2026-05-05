"""Supervised iterative dense clustering model and its loss."""

import torch

from spine.data import TensorBatch, IndexBatch
from spine.utils.cluster.graph import ClusterGraphConstructor
from spine.constants.factory import enum_factory
from spine.utils.globals import (
    SHAPE_COL, SHOWR_SHP, TRACK_SHP, DELTA_SHP, MICHL_SHP)
from .layer.cluster import kernel_factory, loss_factory
from .layer.factories import loss_fn_factory
from .layer.cluster.attn_graph_spice_embedder import AttnGraphSPICEEmbedder
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

    def process_model_config(self, embedder, kernel, constructor,
                             shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                             use_raw_features=False, invert=True,
                             make_clusters=False, n_iterations=2,
                             use_attention=True, do_iterate=None):
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
        do_iterate : bool, optional
            Deprecated alias. If provided, overrides n_iterations:
            False -> n_iterations=1, True -> n_iterations=2.
        """
        # Backward compatibility with do_iterate flag
        if do_iterate is not None:
            n_iterations = 2 if do_iterate else 1

        self.n_iter = n_iterations
        self.use_attention = use_attention

        # Build attention layers whenever n_iterations > 1 so the ablation
        # (use_attention=False) shares the same architecture as the full system.
        enable_attention = (n_iterations > 1)

        # Initialize the embedder
        self.embedder = AttnGraphSPICEEmbedder(
                **embedder, use_raw_features=use_raw_features,
                enable_attention=enable_attention)

        # Initialize the kernel function (must be owned here to be loaded)
        self.kernel_fn = kernel_factory(kernel)

        # Initialize the graph constructor
        self.constructor = ClusterGraphConstructor(
                **constructor, kernel_fn=self.kernel_fn, shapes=shapes,
                invert=invert, training=self.training)

        # Parse the set of shapes to cluster
        self.shapes = enum_factory('shape', shapes)

        # Store model parameters
        self.use_raw_features = use_raw_features
        self.invert = invert
        self.make_clusters = make_clusters

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
            - 'segmentation_iter1': segmentation logits from iteration 0
            When n_iterations > 1 and the event is non-degenerate:
            - 'segmentation': segmentation logits from the final iteration
            - all graph constructor outputs (edge scores, node predictions, ...)
        """
        # Filter the input down to the requested shapes
        data, seg_label, clust_label, index = self.filter_class(
                data, seg_label, clust_label)

        # --- Iteration 0: unconditional pass, no cluster feedback ---
        result = self.embedder(data, cluster_id_full=None)
        segmentation_iter1 = result.get('segmentation_iter1')

        # Coordinates are fixed across all iterations (only features change).
        coords = result['coordinates']
        coords = TensorBatch(coords.data[:, coords.coord_cols], coords.counts)

        graph = self.constructor(coords, self._get_features(result),
                                 seg_label, clust_label)

        # Single-pass baseline: return immediately without any graph feedback.
        if self.n_iter == 1:
            if self.make_clusters:
                with torch.no_grad():
                    clusts, clust_shapes = self.constructor.fit_predict(graph)
                result['clusts'] = clusts
                result['clust_shapes'] = clust_shapes
            result.update(graph)
            result.update({'iter_used': 1, 'filter_index': index})
            if segmentation_iter1 is not None:
                result['segmentation_iter1'] = segmentation_iter1
            return result

        # --- Iterations 1 .. n_iter-1: cluster-conditioned refinement ---
        for k in range(1, self.n_iter):
            if self.use_attention:
                # Obtain cluster IDs from the previous graph to condition
                # the decoder's attention.
                if 'node_pred' not in graph:
                    with torch.no_grad():
                        self.constructor.fit_predict(graph)
                node_pred = graph.get('node_pred')

                # Degenerate event: no clusters found, stop early.
                if node_pred is None:
                    result.update(graph)
                    result.update({'iter_used': k, 'filter_index': index})
                    if segmentation_iter1 is not None:
                        result['segmentation_iter1'] = segmentation_iter1
                    return result

                cluster_ids = node_pred.tensor.detach()
            else:
                # Ablation: iterate without attention conditioning.
                cluster_ids = None

            result.clear()
            del result
            result = self.embedder(data, cluster_id_full=cluster_ids)
            graph = self.constructor(coords, self._get_features(result),
                                     seg_label, clust_label)

        # --- Finalize ---
        if self.make_clusters:
            with torch.no_grad():
                clusts, clust_shapes = self.constructor.fit_predict(graph)
            result['clusts'] = [c.detach() for c in clusts]
            result['clust_shapes'] = clust_shapes

        result.update(graph)
        result.update({'iter_used': self.n_iter, 'filter_index': index})
        if segmentation_iter1 is not None:
            result['segmentation_iter1'] = segmentation_iter1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


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

        self.w_edge = 1.0
        self.w_seg_iter1 = 0.3
        self.w_seg_iter2 = 0.5
    def process_graph_loss_config(self, evaluate_clustering_metrics=False, **graph_loss):
        """Process the loss configuration

        Parameters
        ----------
        evaluate_clustering_metrics : bool, default False
            If `True`, evaluates the clustering accuracy directly, rather than
            simply reporting an edge assignment acurracy
        **loss : dict
            Loss configurationd dictionary
        """
        # Store basic parameters
        self.evaluate_clustering_metrics = evaluate_clustering_metrics

        # Initialize the loss function
        self.loss_fn = loss_factory(graph_loss)

    def process_seg_loss_config(self, loss='ce'):
        try:
            self.seg_loss_fn = loss_fn_factory(loss, reduction='none')
        except KeyError:
            raise ValueError(
                f"Unknown loss function `{loss}` provided. ")

    def process_model_config(self, constructor,
                             shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                             invert=True, **kwargs):
        """Process the model configuration

        Parameters
        ----------
        constructor : dict, optional
            Edge index construction configuration
        shapes : List[int], default [0, 1, 2, 3]
            List of semantic shapes to run DBSCAN on
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        """
        # Initialize the graph constructor (used to produce node assignments)
        if self.evaluate_clustering_metrics:
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

    def forward(self, seg_label, clust_label, filter_index, segmentation=None, segmentation_iter1=None,
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
        if segmentation_iter1 is not None:
            segmentation_iter1_t = segmentation_iter1.tensor
            seg_loss_1, seg_acc_1, _ = self.get_loss_accuracy(segmentation_iter1_t, seg_label_t)
            result['seg_loss_iter1'] = seg_loss_1 * self.w_seg_iter1
            result['seg_acc_iter1'] = seg_acc_1
            result['w_seg_iter1'] = self.w_seg_iter1

            # Auxiliary weight (smaller)
            total_loss += self.w_seg_iter1 * seg_loss_1

        # 3. SEGMENTATION ITER2 - Auxiliary (attention reward)
        # Rewards attention for improving semantic predictions
        if segmentation is not None:
            segmentation_t = segmentation.tensor
            seg_loss_2, seg_acc_2, _ = self.get_loss_accuracy(segmentation_t, seg_label_t)
            result['seg_loss_iter2'] = seg_loss_2 * self.w_seg_iter2
            result['seg_acc_iter2'] = seg_acc_2
            result['w_seg_iter2'] = self.w_seg_iter2

            # Auxiliary weight (moderate)
            total_loss += self.w_seg_iter2 * seg_loss_2

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
            # Per-class prediction accuracy
            preds = torch.argmax(logits, dim=-1)
            #acc_class = torch.ones(num_classes, dtype=np.float32)
            #for c in range(num_classes):
                #if counts[c] > 0:
                    #mask = torch.nonzero(labels == c).flatten()
                    #acc_class[c] = (preds[mask] == c).sum().item() / counts[c]

            # Global prediction accuracy
            acc = (preds == labels).sum().item() / torch.sum(counts).item()

        return loss, acc, weights
