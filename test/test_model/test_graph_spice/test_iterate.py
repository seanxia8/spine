"""Tests for the opt-in modes of iterative Graph-SPICE."""

import pytest
import torch

from spine.data import TensorBatch
from spine.model.graph_spice.iterate import GraphSPICEIter

EMBEDDER = {
    "predict_semantics": True,
    "num_classes": 5,
    "uresnet_attn": {
        "num_input": 4,
        "filters": 4,
        "depth": 3,
        "reps": 1,
        "spatial_size": 16,
        "embed_dim": 8,
        "num_heads": 2,
        "checkpoint_attn": False,
    },
}


def make_model(**options):
    """Build a tiny two-pass model embedding every voxel."""
    config = {
        "use_raw_features": True,
        "n_iterations": 2,
        "embedder": EMBEDDER,
        "kernel": {"provider": "bilinear", "num_features": 4},
        "constructor": {
            "edge_threshold": 0.5,
            "graph": {"provider": "knn", "k": 3},
        },
        "filter_input": False,
        **options,
    }
    return GraphSPICEIter(config).eval()


def make_data():
    """Two events of unique random voxels."""
    generator = torch.Generator().manual_seed(0)
    rows = []
    for batch_id, count in enumerate((40, 30)):
        coords = torch.randperm(16**3, generator=generator)[:count]
        xyz = torch.stack([coords // 256, coords // 16 % 16, coords % 16], dim=1)
        rows.append(
            torch.cat(
                [torch.full((count, 1), batch_id), xyz, torch.rand(count, 1)], dim=1
            ).float()
        )
    return TensorBatch(
        torch.cat(rows), counts=[40, 30], has_batch_col=True, coord_cols=(1, 2, 3)
    )


@pytest.mark.parametrize("ppn_source", ["iter0", "last"])
def test_unfiltered_inference_with_point_proposals(ppn_source):
    """Without labels, every voxel is embedded and the PPN head runs."""
    model = make_model(ppn={"classify_endpoints": False}, ppn_source=ppn_source)
    data = make_data()
    with torch.no_grad():
        result = model(data)

    num_voxels = len(data.tensor)
    assert result["iter_used"] == 2
    assert torch.equal(result["filter_index"].index, torch.arange(num_voxels))
    assert result["segmentation_iter_last"].tensor.shape == (num_voxels, 5)
    assert len(result["ppn_points"].tensor) == num_voxels
    assert "final_tensor" not in result and "decoder_tensors" not in result


def test_unfiltered_input_requires_semantics():
    """Gating by predictions needs a semantic head."""
    embedder = dict(EMBEDDER, predict_semantics=False)
    with pytest.raises(ValueError, match="predict_semantics"):
        make_model(embedder=embedder)


def test_invalid_ppn_source():
    """Only the first or last pass can feed the point proposal head."""
    with pytest.raises(ValueError, match="ppn_source"):
        make_model(ppn={}, ppn_source="middle")
