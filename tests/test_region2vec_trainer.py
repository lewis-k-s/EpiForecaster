"""Bounded Region2Vec training through its real Zarr input and artifact path."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from training.region2vec_trainer import (
    EncoderConfig,
    Region2VecTrainer,
    RegionTrainerConfig,
)


def _write_region_graph(path: Path) -> None:
    features = np.array(
        [
            [i + 1, i + 2, 100 + 50 * i, 10 + 2 * i, 2.1 + i / 10, 41.1 + i / 10]
            for i in range(10)
        ],
        dtype=np.float32,
    )
    edges = [(i, i + 1) for i in range(9)] + [(i + 1, i) for i in range(9)]
    edge_index = np.array(edges, dtype=np.int64).T
    flows = np.zeros((10, 10), dtype=np.float32)
    flows[np.arange(9), np.arange(1, 10)] = 2.0
    dataset = xr.Dataset(
        {
            "features": (("region", "feature"), features),
            "edge_index": (("axis", "edge"), edge_index),
            "flows": (("source", "target"), flows),
            "region_ids": (("region",), np.array([f"r{i}" for i in range(10)])),
        },
        attrs={"metadata": json.dumps({"flow_source": "mobility"})},
    )
    dataset.to_zarr(path, mode="w", zarr_format=2)


@pytest.mark.region
@pytest.mark.integration
@pytest.mark.parametrize("use_mobility_flows", [False, True])
def test_region2vec_trains_and_writes_embeddings(tmp_path, use_mobility_flows) -> None:
    graph_path = tmp_path / "graph.zarr"
    _write_region_graph(graph_path)
    output_dir = tmp_path / "artifacts"
    config = RegionTrainerConfig.from_dict(
        {
            "data": {
                "zarr_path": str(graph_path),
                "use_mobility_flows": use_mobility_flows,
            },
            "encoder": {"hidden_dim": 8, "embedding_dim": 4, "hidden_norm": "layer"},
            "training": {
                "epochs": 1,
                "device": "cpu",
                "seed": 7,
                "enable_mixed_precision": False,
                "lr_scheduler": "none",
            },
            "sampling": {
                "positive_pairs": 4,
                "negative_pairs": 4,
                "hop_pairs": 4,
                "min_flow_threshold": 0.1,
            },
            "loss": {"loss_type": "community" if use_mobility_flows else "spatial"},
            "output": {
                "output_dir": str(output_dir),
                "log_dir": str(tmp_path / "logs"),
                "wandb_mode": "disabled",
                "save_numpy": False,
            },
            "clustering": {"enabled": False},
        },
        base_dir=tmp_path,
    )
    trainer = Region2VecTrainer(config)
    assert (trainer.flow_matrix is not None) is use_mobility_flows

    result = trainer.run()

    assert result["epochs"] == 1
    assert math.isfinite(result["best_loss"])
    assert result["region_ids"] == [f"r{i}" for i in range(10)]
    embedding_path = Path(result["artifacts"]["embedding_path"])
    metrics_path = Path(result["artifacts"]["metrics_path"])
    assert embedding_path.is_file()
    assert metrics_path.is_file()
    assert (output_dir / "region_embeddings.partial.pt").is_file()
    payload = torch.load(embedding_path, weights_only=False)
    assert payload["embeddings"].shape == (10, 4)
    assert torch.isfinite(payload["embeddings"]).all()
    assert payload["region_ids"] == result["region_ids"]
    assert len(json.loads(metrics_path.read_text())["history"]) == 1


@pytest.mark.region
def test_region_encoder_rejects_invalid_activation_and_norm() -> None:
    for options, message in (
        ({"activation": "swish"}, "encoder.activation"),
        ({"hidden_norm": "instance"}, "encoder.hidden_norm"),
    ):
        with pytest.raises(ValueError, match=message):
            EncoderConfig(**options)
