"""Binary edge-classification loss for GraphSPICE."""

from __future__ import annotations

from typing import cast

import torch

from spine.config.factory import Config
from spine.data import ClusterLabelBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory, metric_fn_factory
from spine.model.common.weighting import get_class_weights

__all__ = ["EdgeLoss", "ContrastiveLoss"]


class BinaryMetric(torch.nn.Module):
    """Typing contract for binary metrics built by the shared factory."""

    name: str


class EdgeLoss(torch.nn.Module):
    """Supervise the binary edge logits produced by GraphSPICE."""

    name = "edge"

    def __init__(
        self,
        loss: Config = "binary_log_dice_ce",
        invert: bool = True,
        balance_loss: bool = False,
        equal_sampling: bool = False,
        min_sample_edges: int = 1000,
        metric: Config | None = "iou",
        hard_negative_mining: bool = False,
        hard_negative_keep_easy: float = 0.2,
    ) -> None:
        """Initialize the edge-classification loss.

        Parameters
        ----------
        loss : str or mapping, default "binary_log_dice_ce"
            Binary loss configuration.
        invert : bool, default True
            Treat disconnected edges as the positive class.
        balance_loss : bool, default False
            Weight individual edges to balance the two target classes.
        equal_sampling : bool, default False
            Draw the same number of examples from each target class.
        min_sample_edges : int, default 1000
            Minimum number of examples drawn from each class when equal
            sampling is enabled. Sampling uses replacement when necessary.
        metric : str or mapping, optional
            Additional binary metric configuration.
        hard_negative_mining : bool, default False
            Restrict the loss to currently mis-predicted edges plus a random
            fraction of correctly predicted ones. Accuracy and the metric are
            still reported on all edges.
        hard_negative_keep_easy : float, default 0.2
            Fraction of correctly predicted edges kept by hard negative mining.
        """
        # Initialize the parent class
        super().__init__()

        # Validate and store edge-selection parameters
        if min_sample_edges < 1:
            raise ValueError("`min_sample_edges` must be positive.")

        self.invert = invert
        self.balance_loss = balance_loss
        self.equal_sampling = equal_sampling
        self.min_sample_edges = min_sample_edges
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_keep_easy = hard_negative_keep_easy

        # Initialize the edge objective and optional assignment metric
        self.loss_fn = loss_fn_factory(loss, reduction="none")
        self.metric_fn = (
            None if metric is None else cast(BinaryMetric, metric_fn_factory(metric))
        )

    def sample_edges(
        self,
        edge_logits: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw an equal number of edges from both target classes.

        Parameters
        ----------
        edge_logits : torch.Tensor
            ``(E,)`` edge-logit predictions.
        edge_labels : torch.Tensor
            ``(E,)`` binary edge labels.

        Returns
        -------
        tuple of torch.Tensor
            Sampled logits and corresponding labels. If either class is
            absent, the original tensors are returned unchanged.
        """
        # Identify the available edges in each binary target class
        class_indices = [
            torch.where(edge_labels == class_id)[0] for class_id in range(2)
        ]
        if any(len(indices) == 0 for indices in class_indices):
            return edge_logits, edge_labels

        # Choose one shared sample count, respecting the configured minimum
        sample_count = max(
            min(len(indices) for indices in class_indices),
            self.min_sample_edges,
        )

        # Sample each class with replacement only when necessary
        sampled_indices = []
        for indices in class_indices:
            if sample_count <= len(indices):
                permutation = torch.randperm(
                    len(indices),
                    device=indices.device,
                )[:sample_count]
                sampled_indices.append(indices[permutation])
            else:
                selection = torch.randint(
                    len(indices),
                    (sample_count,),
                    device=indices.device,
                )
                sampled_indices.append(indices[selection])

        index = torch.cat(sampled_indices)
        return edge_logits[index], edge_labels[index]

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        edge_attr: TensorBatch,
        edge_label: TensorBatch,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Compute the loss and metrics for one batch of graph edges.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Voxel cluster labels. Present for the shared loss interface and
            unused by this loss.
        edge_attr : TensorBatch
            ``(E,)`` edge-logit predictions.
        edge_label : TensorBatch
            ``(E,)`` binary edge labels.
        **kwargs : object, optional
            Additional upstream outputs unused by this loss.

        Returns
        -------
        dict
            Loss, binary accuracy, number of supervised edges, and the
            configured optional metric.
        """
        # Validate and normalize edge logits and binary targets
        edge_logits = edge_attr.torch_tensor().flatten()
        edge_labels = edge_label.torch_tensor().flatten()
        if len(edge_logits) != len(edge_labels):
            raise ValueError(
                "Edge logits and labels must have the same length, got "
                f"{len(edge_logits)} and {len(edge_labels)}."
            )
        if torch.any((edge_labels != 0) & (edge_labels != 1)):
            raise ValueError("Edge labels must be binary (0 or 1).")
        edge_labels = edge_labels.long()

        # Invert first so predictions, mining and metrics share a label space
        if self.invert:
            edge_labels = torch.logical_not(edge_labels).long()

        # Metrics are reported on all edges, not the supervised subset
        all_predictions = (edge_logits > 0.0).long()
        all_labels = edge_labels

        # Keep mis-predicted edges and a random fraction of correct ones
        if self.hard_negative_mining:
            wrong = all_predictions != edge_labels
            easy = (
                torch.rand(len(edge_logits), device=edge_logits.device)
                < self.hard_negative_keep_easy
            )
            keep = wrong | easy
            edge_logits, edge_labels = edge_logits[keep], edge_labels[keep]

        # Apply optional equal-class sampling
        if self.equal_sampling:
            edge_logits, edge_labels = self.sample_edges(
                edge_logits,
                edge_labels,
            )

        num_edges = len(edge_logits)

        # Reduce the objective, preserving a differentiable zero for empty graphs
        if num_edges == 0:
            loss = edge_logits.sum() * 0.0
        else:
            loss = self.loss_fn(edge_logits, edge_labels.float())
            if self.balance_loss:
                weights = get_class_weights(
                    edge_labels,
                    num_classes=2,
                    per_class=False,
                )
                loss *= weights
            loss = loss.mean()

        # Compute binary assignment accuracy and the optional metric
        accuracy = 1.0
        if len(all_labels) > 0:
            accuracy = float((all_predictions == all_labels).sum() / len(all_labels))

        metric = {}
        if self.metric_fn is not None:
            metric[self.metric_fn.name] = self.metric_fn(
                all_labels,
                all_predictions,
            )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "count": num_edges,
            **metric,
        }


class ContrastiveLoss(torch.nn.Module):
    """Triplet-style contrastive loss on node embeddings.

    For up to ``n`` random anchors, the ``k_pairs`` nearest same-cluster and
    nearest other-cluster embeddings are averaged (soft centroids), then
    ``mean(max(0, margin + |a - p| - |a - n|))`` is computed on L2-normalized
    vectors.
    """

    name = "contrastive"

    def __init__(self, margin: float = 1.0, k_pairs: int = 1) -> None:
        super().__init__()
        self.margin = margin
        self.k_pairs = k_pairs

    @torch.no_grad()
    def sample_features(
        self, features: torch.Tensor, labels: torch.Tensor, n: int = 256
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """Select anchors with their nearest positive and negative neighbors."""
        n = min(n, len(labels))
        anchors, positives, negatives = [], [], []
        for i in torch.randperm(len(labels), device=labels.device)[:n].tolist():
            pos_pool = (labels == labels[i]).nonzero(as_tuple=False).squeeze(1)
            pos_pool = pos_pool[pos_pool != i]
            neg_pool = (labels != labels[i]).nonzero(as_tuple=False).squeeze(1)
            if pos_pool.numel() < 1 or neg_pool.numel() < 1:
                continue
            pos_dists = (features[pos_pool] - features[i]).norm(dim=1)
            neg_dists = (features[neg_pool] - features[i]).norm(dim=1)
            k_pos = min(self.k_pairs, pos_pool.numel())
            k_neg = min(self.k_pairs, neg_pool.numel())
            anchors.append(i)
            positives.append(pos_pool[pos_dists.topk(k_pos, largest=False).indices])
            negatives.append(neg_pool[neg_dists.topk(k_neg, largest=False).indices])

        return anchors, positives, negatives

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, n: int = 256
    ) -> torch.Tensor:
        anchors, positives, negatives = self.sample_features(features, labels, n)
        if not anchors:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        normalize = torch.nn.functional.normalize
        a = normalize(features[anchors], dim=1)
        p = normalize(torch.stack([features[k].mean(0) for k in positives]), dim=1)
        n_f = normalize(torch.stack([features[k].mean(0) for k in negatives]), dim=1)
        return torch.clamp(
            self.margin + (a - p).norm(dim=1) - (a - n_f).norm(dim=1), min=0
        ).mean()
