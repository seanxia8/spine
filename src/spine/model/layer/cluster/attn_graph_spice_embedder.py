"""Feature embedding for pixel supervised connected-component clustering."""

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from spine.data import TensorBatch

from spine.utils.globals import COORD_COLS, VALUE_COL
#from spine.utils.cluster.helpers import build_cluster_token_topk, build_parent_of_from_kernel_map, tokens_to_sparse_on_parents
#from spine.utils.cluster.helpers import mask_cluster_id

from spine.model.layer.cnn.uresnet_attn_layers import UResNetAttnEncoder, UResNetAttnDecoder
__all__ = ['AttnGraphSPICEEmbedder']


class AttnGraphSPICEEmbedder(nn.Module):
    """Model which produces embeddings of an input sparse point cloud, with gated attention mask."""
    MODULES = ['uresnet_attn']

    def __init__(self, uresnet_attn, enable_attention=True, **base):
        """Initialize the embedding model.

        Parameters
        ----------
        uresnet_attn : dict
            Backbone UResNetAttn configuration
        **base : dict, optional
            Basic parameters
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the uresnet backbone
        self.encoder = UResNetAttnEncoder(uresnet_attn)
        self.decoder = UResNetAttnDecoder(uresnet_attn, enable_attention=enable_attention)
        self.num_filters = self.encoder.num_filters
        self.spatial_size = self.encoder.spatial_size
        self.enable_attention = enable_attention
        assert self.spatial_size is not None, (
                "Must provide a spatial size to compute normalized coordinates.")

        # Process the rest of the configuration
        self.process_model_config(**base)

        # Define output layers, if there is a need for them
        if not self.use_raw_features:
            self.out_spatial = nn.Sequential(
                    nn.Linear(self.num_filters, self.spatial_embedding_dim),
                    nn.Tanh())
            self.out_feature = nn.Linear(
                    self.num_filters, self.feature_embedding_dim)
            self.out_cov = nn.Linear(self.num_filters, 2)
            self.out_occupancy = nn.Linear(self.num_filters, 1)

        if self.predict_semantics:
            assert self.num_classes is not None, (
                    "Must specify the number of classes predicting semantics.")
            self.out_seg = nn.Linear(self.num_filters, self.num_classes)


    def process_model_config(self, predict_semantics=False, num_classes=None,
                             coord_conv=True, covariance_mode='softplus',
                             occupancy_mode='softplus', feature_embedding_dim=16,
                             spatial_embedding_dim=3, use_raw_features=False,):
        """Process the embedding parameters.

        Parameters
        ----------
        predict_semantics : bool, default False
            If `True`, the embedder will output semantic predictions
        num_classes : int, optional
            Number of classes to classify the voxels as
        coord_conv : bool, default True
            If `True`, include the normalized pixel coordinates as a set of
            input features to the backbone UResNet
        covariance_mode : str, default 'softplus'
            Activation used to predict cluster covariance (spatial extent)
        occupancy_mode : str, default 'softplus'
            Activation used to predict cluster occupancy (pixel count)
        feature_embedding_dim : int, default 16
            Number of features per pixel in embedding space
        spatial_embedding_space : int, default 3
            Number of spatial features per pixel in embedding space
        use_raw_features : bool, default False
            Use the list of embedder features as is, without the output layers
        """
        # Store basic properties
        self.num_classes = num_classes
        self.coord_conv = coord_conv
        self.predict_semantics  = predict_semantics
        self.use_raw_features   = use_raw_features
        self.covariance_mode = covariance_mode
        self.occupancy_mode = occupancy_mode

        self.feature_embedding_dim = feature_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.hyper_dimension = (
                self.spatial_embedding_dim + self.feature_embedding_dim + 3)
        
        # Initialize covariance activation function
        if self.covariance_mode == 'exp':
            self.cov_func = torch.exp
        elif self.covariance_mode == 'softplus':
            self.cov_func = nn.Softplus()
        else:
            raise ValueError(
                    f"Covariance mode not recognized: {self.covariance_mode}")
        
        # Initialize occupancy activation function
        if self.occupancy_mode == 'exp':
            self.occ_func = torch.exp
        elif self.occupancy_mode == 'softplus':
            self.occ_func = nn.Softplus()
        else:
            raise ValueError(
                    f"Occupancy mode not recognized: {self.covariance_mode}")

    from typing import Optional
    def forward(self, data, *, cluster_id_full: Optional[torch.Tensor] = None,
                node_confidence: Optional[torch.Tensor] = None):
        """Compute the embeddings for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            Input voxel/value batch.
        cluster_id_full : torch.Tensor, optional
            (N,) integer cluster assignment from the previous iteration.
            Drives the attention mask in the decoder.
        node_confidence : torch.Tensor, optional
            (N,) per-voxel clustering confidence in [0, 1].
            When provided, cross-cluster suppression in the attention mask is
            scaled by confidence rather than being a hard -inf block.
        """
        # Build an input feature tensor
        coords = data.tensor[:, :VALUE_COL]
        features = data.tensor[:, VALUE_COL].view(-1, 1)

        # If requested, append the normalized coordinates to the feature tensor
        half_size = self.spatial_size/2
        points = coords[:, 1:]
        normalized_coords = (points - half_size)/half_size
        if self.coord_conv:
            features = torch.cat([normalized_coords, features], dim=1)

        # Pass it through the backbone UResNet, extract output features
        x_in = ME.SparseTensor(features, coordinates=coords)
        enc_out = self.encoder(x_in)
        x_bn = enc_out["final_tensor"]
        skips = enc_out["encoder_tensors"]
        coords = TensorBatch(coords, data.counts, coord_cols=COORD_COLS)

        if cluster_id_full is None or not self.enable_attention:
            # Unconditional pass: no cluster conditioning available or attention
            # disabled. Run a single decoder pass and return.
            dec1, _ = self.decoder(x_bn, encoder_tensors=skips, cluster_ids=None)
            output_features = dec1[-1].F
            features = TensorBatch(output_features, data.counts)
            result = {'coordinates': coords, 'features': features}
            if not self.use_raw_features:
                # Produce hypergraph_features so _get_features() works regardless
                # of which pass (first or attention) calls it.
                proj_feats = output_features
                spatial_embeddings = self.out_spatial(proj_feats)
                feature_embeddings = self.out_feature(proj_feats)
                out = self.out_cov(proj_feats); covariance = self.cov_func(out)
                out = self.out_occupancy(proj_feats); occupancy = self.occ_func(out)
                hypergraph_features = torch.cat(
                    [spatial_embeddings, feature_embeddings, covariance, occupancy], dim=1)
                result['hypergraph_features'] = TensorBatch(hypergraph_features, data.counts)
            if self.predict_semantics:
                segmentation = self.out_seg(output_features)
                result['segmentation_iter0'] = TensorBatch(segmentation, data.counts)
            return result

        # Attention-conditioned pass: skip dec1 entirely — its output is not
        # used when cluster_id_full is provided, and omitting it frees the
        # decoder activations before the memory-intensive attention softmax.
        dec2, _attn = self.decoder(x_bn, encoder_tensors=skips, cluster_ids=cluster_id_full,
                                   node_confidence=node_confidence)
        output_features = dec2[-1].F
        features = TensorBatch(output_features, data.counts)
        result = {'coordinates': coords, 'features': features}

        # Mean entropy across all available attention heads and layers.
        # _attn: list-of-lists [layer][head] of detached tensors; last dim is
        # the attended sequence treated as a probability distribution.
        if self.decoder.analysis_mode:
            entropies = [-(aw.float().clamp(min=1e-8).log()
                        * aw.float().clamp(min=1e-8)).sum(-1).mean().item()
                        for layer_attn in _attn
                        for aw in layer_attn if aw is not None]
            result['attn_entropy'] = (sum(entropies) / len(entropies)
                                        if entropies else float('nan'))

        # If requested, pass the raw output features through final layers
        if not self.use_raw_features:
            proj_feats = output_features
            # Spatial Embeddings (offset by the normalized coordinates)
            spatial_embeddings = self.out_spatial(proj_feats)

            # Feature Embeddings
            feature_embeddings = self.out_feature(proj_feats)

            # Covariance
            out = self.out_cov(proj_feats)
            covariance = self.cov_func(out)

            # Occupancy
            out = self.out_occupancy(proj_feats)
            occupancy = self.occ_func(out)

            # Bundle the features together
            hypergraph_features = torch.cat(
                    [spatial_embeddings, feature_embeddings, covariance, occupancy],
                    dim=1)

            # Convert the output to tensor batches
            spatial_embeddings = TensorBatch(
                    spatial_embeddings + normalized_coords, data.counts)
            feature_embeddings = TensorBatch(feature_embeddings, data.counts)
            covariance = TensorBatch(covariance, data.counts)
            occupancy = TensorBatch(occupancy, data.counts)
            hypergraphneeded_features = TensorBatch(hypergraph_features, data.counts)

            # Append results
            result.update({
                    'spatial_embeddings': spatial_embeddings,
                    'feature_embeddings': feature_embeddings,
                    'covariance': covariance,
                    'occupancy': occupancy,
                    'hypergraph_features': TensorBatch(hypergraph_features, data.counts)
            })

        # If requested, add a semantic prediction to the output
        if self.predict_semantics:
            # Segmentation layer
            segmentation = self.out_seg(output_features)

            # Append results
            result['segmentation_iter_last'] = TensorBatch(segmentation, data.counts)
            #result['attention_heatmap'] = TensorBatch(_attn, data.counts)

        return result
