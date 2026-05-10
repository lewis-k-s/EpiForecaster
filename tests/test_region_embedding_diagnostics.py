import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from dataviz.region_embedding_diagnostics import (
    DiagnosticsConfig,
    run_diagnostics,
)


def _write_artifact(path: Path, region_ids: list[str]) -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
            [0.9, 0.2, 0.1],
            [1.0, 0.3, 0.0],
            [0.4, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "embeddings": embeddings,
            "region_ids": region_ids,
            "config": {"sampling": {"min_flow_threshold": 0.5}},
        },
        path,
    )


def _write_region_graph(
    path: Path, region_ids: list[str], *, include_flows: bool
) -> None:
    features = np.array(
        [
            [1.0, 4.0, 100.0, 10.0, 2.1, 41.1],
            [2.0, 5.0, 150.0, 12.0, 2.2, 41.2],
            [3.0, 6.0, 300.0, 20.0, 2.3, 41.3],
            [4.0, 7.0, 320.0, 21.0, 2.4, 41.4],
            [5.0, 8.0, 200.0, 15.0, 2.5, 41.5],
        ],
        dtype=np.float32,
    )
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=np.int64,
    )
    data_vars = {
        "region_ids": (("region",), np.array(region_ids, dtype=object)),
        "features": (("region", "feature"), features),
        "edge_index": (("axis", "edge"), edge_index),
    }
    if include_flows:
        flows = np.array(
            [
                [0.0, 5.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        data_vars["flows"] = (("source", "target"), flows)
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "metadata": json.dumps(
                {"flow_source": "mobility" if include_flows else "adjacency"}
            )
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def test_region_embedding_diagnostics_outputs_expected_files(tmp_path: Path) -> None:
    region_ids = ["r0", "r1", "r2", "r3", "r4"]
    artifact_path = tmp_path / "region_embeddings.pt"
    graph_path = tmp_path / "region_graph.zarr"
    output_dir = tmp_path / "diagnostics"
    _write_artifact(artifact_path, region_ids)
    _write_region_graph(graph_path, region_ids, include_flows=True)

    artifacts = run_diagnostics(
        DiagnosticsConfig(
            embeddings=artifact_path,
            region_graph=graph_path,
            geojson=tmp_path / "missing.geojson",
            output_dir=output_dir,
            cluster_count=3,
            max_scatter_points=100,
        )
    )

    assert Path(artifacts["metrics"]).exists()
    assert Path(artifacts["region_metrics"]).exists()
    assert Path(artifacts["spatial_autocorr"]).exists()
    assert Path(artifacts["nearest_neighbor_overlap"]).exists()
    assert Path(artifacts["embedding_pca_scatter"]).exists()
    assert Path(artifacts["distance_by_hop"]).exists()
    assert "embedding_map_clusters" not in artifacts

    metrics = json.loads(Path(artifacts["metrics"]).read_text(encoding="utf-8"))
    assert metrics["integrity"]["num_regions"] == 5
    assert metrics["integrity"]["missing_in_artifact"] == 0
    assert metrics["flow_alignment"]["available"] is True

    region_df = pd.read_csv(artifacts["region_metrics"])
    expected_columns = {
        "region_id",
        "embedding_norm",
        "nearest_neighbor_distance",
        "graph_degree",
        "out_flow",
        "in_flow",
        "pca1",
        "pca2",
        "cluster",
    }
    assert expected_columns.issubset(region_df.columns)
    assert len(region_df) == 5

    moran_df = pd.read_csv(artifacts["spatial_autocorr"])
    assert {"dimension", "moran_i", "p_norm", "status"}.issubset(moran_df.columns)


def test_region_embedding_diagnostics_handles_missing_flow(tmp_path: Path) -> None:
    region_ids = ["r0", "r1", "r2", "r3", "r4"]
    artifact_path = tmp_path / "region_embeddings.pt"
    graph_path = tmp_path / "region_graph_no_flow.zarr"
    output_dir = tmp_path / "diagnostics_no_flow"
    _write_artifact(artifact_path, region_ids)
    _write_region_graph(graph_path, region_ids, include_flows=False)

    artifacts = run_diagnostics(
        DiagnosticsConfig(
            embeddings=artifact_path,
            region_graph=graph_path,
            output_dir=output_dir,
            cluster_count=3,
        )
    )

    metrics = json.loads(Path(artifacts["metrics"]).read_text(encoding="utf-8"))
    assert metrics["flow_alignment"]["available"] is False
    region_df = pd.read_csv(artifacts["region_metrics"])
    assert region_df["out_flow"].isna().all()
