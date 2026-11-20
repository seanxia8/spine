"""Cluster-conditioned feature embedder for iterative Graph-SPICE."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import TensorBatch
from spine.model import sparse
from spine.model.cnn.uresnet_attn_layers import UResNetAttnDecoder, UResNetAttnEncoder

__all__ = ["AttnGraphSPICEEmbedder"]


class AttnGraphSPICEEmbedder(torch.nn.Module):
    """Graph-SPICE embedder whose decoder can attend within predicted clusters.

    Without cluster assignments the decoder runs a plain UResNet pass. With
    cluster assignments from a previous iteration, the last decoder level
    applies cluster-conditioned attention (see :class:`UResNetAttnDecoder`).
    """

    def __init__(
        self,
        uresnet_attn: dict[str, Any],
        enable_attention: bool = True,
        **base: Any,
    ) -> None:
        """Initialize the embedder.

        Parameters
        ----------
        uresnet_attn : dict
            Backbone UResNet + attention configuration
        enable_attention : bool, default True
            Whether to build the decoder attention block
        **base : dict, optional
            Basic parameters, see :meth:`process_model_config`
        """
        super().__init__()

        self.encoder = UResNetAttnEncoder(uresnet_attn)
        self.decoder = UResNetAttnDecoder(
            uresnet_attn, enable_attention=enable_attention
        )
        self.num_filters = self.encoder.num_filters
        self.spatial_size = self.encoder.spatial_size
        self.dimension = self.encoder.dimension
        self.enable_attention = enable_attention
        if self.spatial_size is None:
            raise ValueError(
                "Must provide a spatial size to compute normalized coordinates."
            )

        self.process_model_config(**base)

        if not self.use_raw_features:
            self.out_spatial = torch.nn.Sequential(
                torch.nn.Linear(self.num_filters, self.spatial_embedding_dim),
                torch.nn.Tanh(),
            )
            self.out_feature = torch.nn.Linear(
                self.num_filters, self.feature_embedding_dim
            )
            self.out_cov = torch.nn.Linear(self.num_filters, 2)
            self.out_occupancy = torch.nn.Linear(self.num_filters, 1)

        if self.predict_semantics:
            if self.num_classes is None:
                raise ValueError(
                    "Must specify the number of classes predicting semantics."
                )
            self.out_seg = torch.nn.Linear(self.num_filters, self.num_classes)

    def process_model_config(
        self,
        predict_semantics: bool = False,
        num_classes: int | None = None,
        coord_conv: bool = True,
        covariance_mode: str = "softplus",
        occupancy_mode: str = "softplus",
        feature_embedding_dim: int = 16,
        spatial_embedding_dim: int = 3,
        use_raw_features: bool = False,
    ) -> None:
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
        spatial_embedding_dim : int, default 3
            Number of spatial features per pixel in embedding space
        use_raw_features : bool, default False
            Use the list of embedder features as is, without the output layers
        """
        self.num_classes = num_classes
        self.coord_conv = coord_conv
        self.predict_semantics = predict_semantics
        self.use_raw_features = use_raw_features
        self.covariance_mode = covariance_mode
        self.occupancy_mode = occupancy_mode
        self.feature_embedding_dim = feature_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.hyper_dimension = spatial_embedding_dim + feature_embedding_dim + 3

        if covariance_mode == "exp":
            self.cov_func = torch.exp
        elif covariance_mode == "softplus":
            self.cov_func = torch.nn.Softplus()
        else:
            raise ValueError(f"Covariance mode not recognized: {covariance_mode}")

        if occupancy_mode == "exp":
            self.occ_func = torch.exp
        elif occupancy_mode == "softplus":
            self.occ_func = torch.nn.Softplus()
        else:
            raise ValueError(f"Occupancy mode not recognized: {occupancy_mode}")

    def _hypergraph(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Return the spatial, feature, covariance and occupancy embeddings."""
        return [
            self.out_spatial(features),
            self.out_feature(features),
            self.cov_func(self.out_cov(features)),
            self.occ_func(self.out_occupancy(features)),
        ]

    def forward(
        self,
        data: TensorBatch,
        *,
        cluster_id_full: torch.Tensor | None = None,
        node_confidence: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute the embeddings for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            Input voxel/value batch.
        cluster_id_full : torch.Tensor, optional
            (N,) cluster assignment from the previous iteration, which drives
            the decoder attention. Without it, a plain decoder pass is run.
        node_confidence : torch.Tensor, optional
            (N,) per-voxel clustering confidence in [0, 1].

        Returns
        -------
        dict
            ``coordinates`` and ``features``, plus the hypergraph embeddings
            unless ``use_raw_features``, and ``segmentation_iter0`` (plain
            pass) or ``segmentation_iter_last`` (attention pass) if
            ``predict_semantics``.
        """
        # Build the input features, optionally with normalized coordinates
        coordinates = data.batch_coordinates
        features = data.values.torch_tensor().view(-1, 1)
        half_size = self.spatial_size / 2
        normalized_coords = (coordinates[:, 1:] - half_size) / half_size
        if self.coord_conv:
            features = torch.cat([normalized_coords, features], dim=1)

        x_in = sparse.SparseTensor(
            features, coordinates=coordinates.int(), batch_size=data.batch_size
        )
        enc_out = self.encoder(x_in)
        coordinate_batch = TensorBatch(
            coordinates,
            data.counts,
            has_batch_col=True,
            coord_cols=tuple(range(1, 1 + self.dimension)),
        )

        attend = cluster_id_full is not None and self.enable_attention
        decoder_tensors, attn = self.decoder(
            enc_out["final_tensor"],
            enc_out["encoder_tensors"],
            cluster_ids=cluster_id_full if attend else None,
            node_confidence=node_confidence if attend else None,
        )
        output_features = decoder_tensors[-1].aligned_features()
        result: dict[str, Any] = {
            "coordinates": coordinate_batch,
            "features": TensorBatch(output_features, data.counts),
        }

        if not attend:
            if not self.use_raw_features:
                result["hypergraph_features"] = TensorBatch(
                    torch.cat(self._hypergraph(output_features), dim=1), data.counts
                )
            if self.predict_semantics:
                result["segmentation_iter0"] = TensorBatch(
                    self.out_seg(output_features), data.counts
                )
            return result

        # Mean entropy over attention heads and layers (analysis only)
        if self.decoder.analysis_mode:
            entropies = [
                -(aw.float().clamp(min=1e-8).log() * aw.float().clamp(min=1e-8))
                .sum(-1)
                .mean()
                .item()
                for layer_attn in attn
                for aw in layer_attn
                if aw is not None
            ]
            result["attn_entropy"] = (
                sum(entropies) / len(entropies) if entropies else float("nan")
            )

        if not self.use_raw_features:
            spatial, feature, covariance, occupancy = self._hypergraph(output_features)
            result.update(
                {
                    "spatial_embeddings": TensorBatch(
                        spatial + normalized_coords, data.counts
                    ),
                    "feature_embeddings": TensorBatch(feature, data.counts),
                    "covariance": TensorBatch(covariance, data.counts),
                    "occupancy": TensorBatch(occupancy, data.counts),
                    "hypergraph_features": TensorBatch(
                        torch.cat([spatial, feature, covariance, occupancy], dim=1),
                        data.counts,
                    ),
                }
            )

        if self.predict_semantics:
            result["segmentation_iter_last"] = TensorBatch(
                self.out_seg(output_features), data.counts
            )

        return result
