"""Edge kernel functions that produce edge weights from node features."""

import torch
import torch.nn as nn

from spine.model.layer.common.mlp import MLP

__all__ = ["DefaultKernel", "MixedKernel", "BilinearKernel", "MLPKernel"]


class DefaultKernel(nn.Module):
    """Kernel producing edge score based on feature L2 similarity.

    This Kernel assumes that the upstream embedder produces a set of spatial
    and embedding coordinates and computes the L2 similarity between the two
    node feature vectors. It scales the L2 distance by the covariance and
    penalizes for cluster size dissimilarity.
    """

    name = "default"

    def __init__(self, num_features, eps=1e-3):
        """Initializes the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        eps : float
            Features regularization factor
        """
        # Initialize the parent class
        super().__init__()

        # Store the parameters
        self.num_features = num_features
        self.eps = eps

    def compute_edge_weight(
        self, sp_emb1, sp_emb2, ft_emb1, ft_emb2, cov1, cov2, occ1, occ2
    ):
        """Converts the output of the embedder into an edge score.

        Parameters
        ----------
        sp_emb1 : torch.Tensor
            (E, 3) Spatial embeddings of the source nodes
        sp_emb2 : torch.Tensor
            (E, 3) Spatial embeddings of the target nodes
        ft_emb1 : torch.Tensor
            (E, N_f) Feature embeddings of the source nodes
        ft_emb2 : torch.Tensor
            (E, N_f) Feature embeddings of the target nodes
        cov1 : torch.Tensor
            (E, 2) Spatial extent of the cluster the source nodes belongs to
        cov2 : torch.Tensor
            (E, 2) Spatial extent of the cluster the target nodes belongs to
        occ1 : torch.Tensor
            (E, 1) Multiplicity of the cluster the source nodes belongs to
        occ2 : torch.Tensor
            (E, 1) Multiplicity of the cluster the target nodes belongs to

        Returns
        -------
        torch.Tensor
            Scalar value of the edge weight
        """
        # Measure spatial distance between nodes, weighted by cluster covariance
        sp_cov_i = cov1[:, 0]
        sp_cov_j = cov2[:, 0]
        sp_i = ((sp_emb1 - sp_emb2) ** 2).sum(dim=1) / (sp_cov_i**2 + self.eps)
        sp_j = ((sp_emb1 - sp_emb2) ** 2).sum(dim=1) / (sp_cov_j**2 + self.eps)

        # Measure feature distance between nodes, weighted by cluster covariance
        ft_cov_i = cov1[:, 1]
        ft_cov_j = cov2[:, 1]
        ft_i = ((ft_emb1 - ft_emb2) ** 2).sum(dim=1) / (ft_cov_i**2 + self.eps)
        ft_j = ((ft_emb1 - ft_emb2) ** 2).sum(dim=1) / (ft_cov_j**2 + self.eps)

        # Convert the L2 distances to a probability measure (Gaussian kernel)
        p_ij = torch.exp(-sp_i - ft_i)
        p_ji = torch.exp(-sp_j - ft_j)

        pvec = torch.clamp(p_ij + p_ji - p_ij * p_ji, min=0, max=1)

        # Scale down the probability if the cluster sizes are highly different
        r1, r2 = occ1.flatten(), occ2.flatten()
        r = torch.max(
            (r2 + self.eps) / (r1 + self.eps), (r1 + self.eps) / (r2 + self.eps)
        )
        pvec /= r

        return pvec

    def forward(self, x1, x2, **kwargs):
        """Computes the kernel edge score of ll node pairs in the graph.

        This kernel expectes a set of (3 + N_f + 2 + 1) features per node:
        - 3 spatial embedding features
        - N_f feature embedding features
        - 2 covariance features
        - 1 occupancy feature

        Parameters
        ----------
        x1 : torch.Tensor
            (E, 3 + N_f + 2 + 1) Features of the source nodes
        x2 : torch.Tensor
            (E, 3 + N_f + 2 + 1) Features of the target nodes
        """
        # Decompose the two feature sets into their consituents
        assert (
            x1.shape[1] == x2.shape[1] == 6 + self.num_features
        ), "The combined feature vector is not of the expected shape."

        nf = self.num_features
        splits = [3, nf + 3, nf + 5]
        sp_emb1, ft_emb1, cov1, occ1 = torch.tensor_split(x1, splits, dim=1)
        sp_emb2, ft_emb2, cov2, occ2 = torch.tensor_split(x1, splits, dim=1)

        # Compute the edge weight, make sure it's between 0 and 1
        weight = self.compute_edge_weight(
            sp_emb1, sb_emb2, ft_emb1, ft_emb2, cov1, cov2, occ1, occ2
        )
        weight = torch.clamp(w, min=0 + 1e-6, max=1 - 1e-6)

        # Convert probability to a logit, return
        result = torch.logit(weight)

        return result


class MixedKernel(DefaultKernel):
    """Kernel producing edge score based on feature L2 similarity and cosine
    similarity between node features.

    This Kernel assumes that the upstream embedder produces a set of spatial
    and embedding coordinates and computes the L2 similarity between the two
    node feature vectors. It scales the L2 distance by the covariance and
    penalizes for cluster size dissimilarity.

    In addition, it computes cosine similarity between the feature vectors.
    """

    name = "mixed"

    def __init__(self, num_features, eps=1e-3):
        """Initializes the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        eps : float
            Features regularization factor
        """
        # Initialize the parent class
        super().__init__(num_features, eps)

        # Initialize the cosine similarity layer
        self.cos = nn.CosineSimilarity(dim=1)

    def compute_edge_weight_coord(
        self, coord1, coord2, tan1, tan2, coord_cov1, coord_cov2, tan_cov1, tan_cov2
    ):
        """Converts the the spatial coordinates information into an edge score.

        Parameters
        ----------
        coord1 : torch.Tensor
            (E, 3) Coordinates of the source nodes
        coord2 : torch.Tensor
            (E, 3) Coordinates of the target nodes
        tan1 : torch.Tensor
            (E, N_f) Tangents of the source nodes
        tan2 : torch.Tensor
            (E, N_f) Tangents of the target nodes
        coord_cov1 : torch.Tensor
            (E, 3) Spatial extent of the cluster the source nodes belongs to
        coord_cov2 : torch.Tensor
            (E, 3) Spatial extent of the cluster the target nodes belongs to
        tan_cov1 : torch.Tensor
            (E, 1) Multiplicity of the cluster the source nodes belongs to
        tan_cov2 : torch.Tensor
            (E, 1) Multiplicity of the cluster the target nodes belongs to

        Returns
        -------
        torch.Tensor
            Scalar value of the edge weight
        """
        # Compute the average position of the two nodes
        coord_cov = (coord_cov1 + coord_cov2) / 2.0

        # Compute the vector which joins the two nodes and its length
        chord = coord1 - coord2
        chord_dist = torch.pow(chord, 2)
        dist = torch.sum(chord_dist * coord_cov, dim=1)

        # Weight is proportial to the distance
        coords_weight = torch.exp(dist)

        # Affinity
        a1 = torch.abs(self.cos(chord, tan1))
        a2 = torch.abs(self.cos(chord, tan2))
        norm_factor = torch.sum(chord_dist, dim=1)
        tan_cov = (tan_cov1 + tan_cov2) / 2.0
        a = a1 * a2 * tan_cov / (norm_factor + 1e-5)
        affinity_weight = torch.exp(-a)

        pvec = coords_weight * affinity_weight

        return pvec

    def forward(self, x1, x2, **kwargs):
        """Computes the kernel edge score of all node pairs in the graph.

        This kernel expectes a set of (3 + 3 + 3 + N_f + 2 + 3 + 1 + 1)
        features per node:
        - 3 coordinates
        - 3 tangent components
        - 3 spatial embedding features
        - 16 feature embedding features
        - 2 covariance features
        - 3 coordinate covariance features
        - 1 tangent coviarance feature
        - 1 occupancy feature

        Parameters
        ----------
        x1 : torch.Tensor
            (E, 3 + 3 + 3 + N_f + 2 + 3 + 1 + 1) Features of the source nodes
        x2 : torch.Tensor
            (E, 3 + 3 + 3 + N_f + 2 + 3 + 1 + 1) Features of the targer nodes
        """
        # Decompose the two feature sets into their consituents
        assert (
            x1.shape[1] == x2.shape[1] == 16 + self.num_features
        ), "The combined feature vector is not of the expected shape."

        nf = self.num_features
        splits = [3, 6, 9, nf + 9, nf + 11, nf + 14, nf + 15]
        coord1, tan1, sp_emb1, ft_emb1, cov1, coord_cov1, tan_cov1, occ1 = (
            torch.tensor_split(x1, splits, dim=1)
        )
        coord2, tan2, sp_emb2, ft_emb2, cov2, coord_cov2, tan_cov2, occ2 = (
            torch.tensor_split(x1, splits, dim=1)
        )

        # Compute the L2 edge weight
        weight1 = self.compute_edge_weight(
            sp_emb1, sb_emb2, ft_emb1, ft_emb2, cov1, cov2, occ1, occ2
        )

        # Compute the cosine edge weight
        weight2 = self.compute_edge_weight_coord(
            coord1, coord2, tan1, tan2, coord_cov1, coord_cov2, tan_cov1, tan_cov2
        )

        # Combine the scroes and make sure they are between 0 and 1
        weight = torch.clamp(w1 * w2, min=0 + 1e-6, max=1 - 1e-6)

        # Convert probability to a logit, return
        result = torch.logit(weight)

        return result


class BilinearKernel(nn.Module):
    """Kernel producing edges scores based on a learnable bilinear layer."""

    name = "bilinear"

    def __init__(self, num_features, bias=False, use_spatial_dist=False,
                 bilinear_out=1, mlp_hidden=None, spatial_dist_scale=1.0):
        """Initializes the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        bias : bool, default False
            If `True`, allows for an overall bias in the bilinear layer
        use_spatial_dist : bool, default False
            If `True`, the 3D spatial distance between the two nodes is
            concatenated to the bilinear outputs before the MLP head.  This
            lets the kernel distinguish local spatial-kNN edges (small dist)
            from long-range feature-kNN-only edges (large dist).
            Requires ``spatial_dist`` to be passed at call time.
        bilinear_out : int, default 1
            Number of parallel bilinear outputs.  With 1 (default) the kernel
            is a single bilinear form and behaves identically to before.
            With K > 1 the kernel computes K bilinear forms simultaneously,
            giving K different pairwise comparison modes whose gradients shape
            the feature space in K directions per step.  A small nonlinear MLP
            head then combines them (+ optional spatial_dist) into the final
            edge logit.
        mlp_hidden : int, optional
            Hidden width of the MLP head used when ``bilinear_out > 1`` or
            ``use_spatial_dist=True``.  Defaults to
            ``max(2 * bilinear_out, 32)``.
        spatial_dist_scale : float, default 1.0
            Denominator used to normalise ``spatial_dist`` before applying
            ``log1p``: the kernel receives ``log1p(spatial_dist / scale)``.
            Set to the typical edge length (e.g. the kNN neighbourhood radius)
            so that near-range edges map to ~log1p(1)≈0.69 and far edges are
            compressed into a bounded range.  Default 1.0 preserves old
            behaviour (no scaling before log1p).
        """
        # Initialize the parent class
        super().__init__()

        self.num_features = num_features
        self.use_spatial_dist = use_spatial_dist
        self.spatial_dist_scale = spatial_dist_scale

        # K parallel bilinear forms: (E, N_f) x (E, N_f) → (E, K)
        self.bilin = nn.Bilinear(num_features, num_features, bilinear_out, bias=bias)

        # Scalar additive weight for spatial_dist when bilinear_out=1.
        # Keeps the simple additive form: logit = bilinear + w * dist.
        self.dist_weight = None
        self.head = None
        if bilinear_out == 1:
            if use_spatial_dist:
                # Additive: logit = bilinear(f_i,f_j) + w * spatial_dist
                self.dist_weight = nn.Linear(1, 1, bias=False)
            # else: raw bilinear scalar, nothing extra needed
        else:
            # K>1: MLP head combines K bilinear outputs + optional spatial_dist
            mlp_in = bilinear_out + (1 if use_spatial_dist else 0)
            hidden = mlp_hidden if mlp_hidden is not None else max(2 * bilinear_out, 32)
            self.head = nn.Sequential(
                nn.Linear(mlp_in, hidden),
                nn.ELU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x1, x2, spatial_dist=None):
        """Computes the kernel edge score of all node pairs in the graph.

        Parameters
        ----------
        x1 : torch.Tensor
            (E, N_f) Features of the source nodes
        x2 : torch.Tensor
            (E, N_f) Features of the target nodes
        spatial_dist : torch.Tensor, optional
            (E,) 3D Euclidean distance between the two nodes in voxel units.
            Only used when ``use_spatial_dist=True``.
        """
        assert (
            x1.shape[1] == x2.shape[1] == self.num_features
        ), "The feature vector is not of the expected shape."

        out = self.bilin(x1, x2)  # (E, bilinear_out)

        if self.use_spatial_dist and spatial_dist is not None:
            dist_norm = torch.log1p(spatial_dist / self.spatial_dist_scale).unsqueeze(1)
            if self.dist_weight is not None:
                # bilinear_out=1 path: simple additive scalar weight
                out = out + self.dist_weight(dist_norm)
            elif self.head is not None:
                # bilinear_out>1 path: MLP combines K outputs + normalised dist
                out = self.head(torch.cat([out, dist_norm], dim=1))
        elif self.head is not None:
            out = self.head(out)

        return out


class MLPKernel(nn.Module):
    """Kernel producing edges scores based on an MLP and a linear layer."""

    name = "mlp"

    def __init__(self, num_features, bias=False, mlp=None):
        """Initializes the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        bias : bool, default False
            If `True`, allows for an overall bias in the bilinear layer
        mlp : dict, optional
            MLP architecture configuration, see :class:`MLP`
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the underlying MLP
        if mlp is None:
            mlp = {
                "depth": 2,
                "width": 32,
                "activation": "elu",
                "normalization": "batch_norm",
            }

        self.mlp = MLP(in_channels=num_features, **mlp)

        # Initialize the final linear layer
        self.lin = nn.Linear(2 * self.mlp.feature_size, 1, bias=bias)

    def forward(self, x1, x2, **kwargs):
        """Computes the kernel edge score of all node pairs in the graph.

        Parameters
        ----------
        x1 : torch.Tensor
            (E, N_f) Features of the source nodes
        x2 : torch.Tensor
            (E, N_f) Features of the target nodes
        """
        # Pass the node features through the MLP
        f1 = self.mlp(x1)
        f2 = self.mlp(x2)

        # Concatenate them and pass them through the linear layer, return
        result = self.lin(torch.cat([f1, f2], dim=1))

        return result
