"""Joint semantic segmentation, point proposal and fragmentation provider.

Runs iterative Graph-SPICE on every voxel so that its own semantic head and
optional point proposal head replace the standalone UResNet(-PPN): one stage
publishes ``seg_pred``, ``fragment_clusts``/``fragment_shapes`` and, with a
``ppn`` block, ``ppn_points``.

.. code-block:: yaml

    chain:
      stages:
        - name: semantic_fragmentation
          provider: graph_spice_iter
          uses: graph_spice
          loss: graph_spice_loss
        - name: particle_aggregation
          ...
"""

from __future__ import annotations

from typing import Any

import torch

from spine.data import ClusterLabelBatch, TensorBatch
from spine.model.full_chain.label import ClusterLabelAdapter
from spine.model.graph_spice.iterate import GraphSPICEIter, GraphSPICEIterLoss

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult
from .fragmentation import FragmentationStage

PREFIX = "graph_spice_iter_"


class GraphSPICEIterStage(ChainStage):
    """Produce semantics, optional points and fragments with one backbone."""

    requires = frozenset({"point_data"})
    optional = frozenset({"seg_label", "clust_label"})
    provides = frozenset({"seg_pred", "fragment_clusts", "fragment_shapes"})
    replaces = frozenset({"clust_label"})

    def __init__(
        self,
        name: str,
        model: GraphSPICEIter,
        label_adapter: ClusterLabelAdapter | None,
    ) -> None:
        """Initialize the stage.

        Parameters
        ----------
        name : str
            Stage name.
        model : GraphSPICEIter
            Iterative Graph-SPICE model embedding every voxel.
        label_adapter : ClusterLabelAdapter, optional
            Adapter relabeling cluster truth with the predicted semantics.
        """
        super().__init__(name)
        self.model = model
        self.label_adapter = label_adapter
        if model.ppn is not None:
            self.provides = self.provides | {"ppn_points"}

    def forward(self, state: ChainState) -> StageResult:
        """Run the model and publish canonical products.

        Truth labels only drive graph gating (pre-warmup) and edge labels at
        train time; at evaluation the graphs use predicted semantics.
        """
        point_data = state.require("point_data", self.name)
        if getattr(point_data, "orig_index", None) is not None:
            raise ValueError(
                "The graph_spice_iter provider does not support an upstream "
                "stage which changes the voxel set (e.g. deghosting) yet."
            )
        data: TensorBatch = point_data.data
        seg_label: TensorBatch | None = state.get("seg_label")
        clust_label: ClusterLabelBatch | None = state.get("clust_label")

        training = self.model.training
        result = self.model(
            data,
            seg_label if training else None,
            clust_label if training else None,
        )
        outputs: dict[str, Any] = {f"{PREFIX}{k}": v for k, v in result.items()}

        # Final semantics: last (attention) pass if any, else the first pass
        segmentation = result.get("segmentation_iter_last")
        if segmentation is None:
            segmentation = result["segmentation_iter0"]
        seg_pred = TensorBatch(
            torch.argmax(segmentation.torch_tensor(), dim=1), segmentation.counts
        )
        fragments = FragmentationStage._restore_filtered_indexes(
            result["clusts"], result["filter_index"], data.counts
        ).to_numpy()
        products: dict[str, Any] = {
            "seg_pred": seg_pred,
            "fragment_clusts": fragments,
            "fragment_shapes": result["clust_shapes"],
        }
        outputs.update(segmentation=segmentation, **products)

        # Point proposals keep their canonical (unprefixed) names downstream
        if self.model.ppn is not None:
            outputs.update({k: v for k, v in result.items() if k.startswith("ppn_")})
            products["ppn_points"] = result["ppn_points"]

        if (
            self.label_adapter is not None
            and seg_label is not None
            and clust_label is not None
        ):
            adapted = self.label_adapter(clust_label, seg_label, seg_pred)
            products["clust_label"] = adapted
            outputs["clust_label_adapt"] = adapted

        return StageResult(products, outputs)


class GraphSPICEIterLossStage(ChainLossStage):
    """Route the namespaced model outputs to the native objective."""

    def __init__(self, name: str, loss: GraphSPICEIterLoss) -> None:
        """Initialize the loss adapter.

        Parameters
        ----------
        name : str
            Stage name.
        loss : GraphSPICEIterLoss
            Native objective (edges, semantics, optional points).
        """
        super().__init__(name)
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the objective on raw (unadapted) truth labels.

        The model built its edge labels from raw cluster truth, so the same
        labels supervise it here.
        """
        for key in ("seg_label", "clust_label"):
            if data.get(key) is None:
                raise ValueError(f"The graph_spice_iter loss requires `{key}`.")
        output = {
            key.removeprefix(PREFIX): value
            for key, value in data.items()
            if key.startswith(PREFIX)
        }
        return self.loss(
            seg_label=data["seg_label"],
            clust_label=data["clust_label"],
            ppn_label=data.get("ppn_label"),
            **output,
        )


def build_graph_spice_iter_stage(
    name: str, config: dict[str, Any], owner: Any
) -> ChainStage:
    """Build the stage from a ``graph_spice`` block.

    The block is a standalone ``graph_spice_iter`` model block; the stage
    forces ``filter_input: false`` and ``make_clusters: true``.
    ``adapt_labels`` behaves as for the segmentation provider.
    """
    model_config = config.get("graph_spice")
    if not isinstance(model_config, dict):
        raise ValueError(
            "The graph_spice_iter provider requires a `graph_spice` block."
        )
    model = GraphSPICEIter(
        {**model_config, "filter_input": False, "make_clusters": True}
    )
    owner.add_module("graph_spice_iter", model)

    adapt_labels = config.get("adapt_labels")
    if adapt_labels is False:
        label_adapter = None
    elif adapt_labels is None or adapt_labels is True:
        label_adapter = ClusterLabelAdapter()
    elif isinstance(adapt_labels, dict):
        label_adapter = ClusterLabelAdapter(**adapt_labels)
    else:
        raise TypeError("`adapt_labels` must be a boolean or mapping.")

    return GraphSPICEIterStage(name, model, label_adapter)


def build_graph_spice_iter_loss(
    name: str, config: dict[str, Any], owner: Any
) -> ChainLossStage | None:
    """Build the native objective when a loss block is referenced."""
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    model_config = {**config["graph_spice"], "filter_input": False}
    loss = GraphSPICEIterLoss(model_config, loss_config)
    owner.add_module("graph_spice_iter_loss", loss)
    return GraphSPICEIterLossStage(name, loss)


PROVIDER_SPEC = register_provider(
    ProviderSpec(
        "graph_spice_iter", build_graph_spice_iter_stage, build_graph_spice_iter_loss
    )
)
