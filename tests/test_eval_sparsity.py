from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from dataviz.eval_head_plots import _compute_canonical_sparsity


def test_sparsity_reads_selected_run_and_combines_biomarker_masks(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "aligned.zarr"
    dimensions = ("run_id", TEMPORAL_COORD, REGION_COORD)
    xr.Dataset(
        data_vars={
            "cases_mask": (dimensions, np.array([[[1], [1]], [[1], [0]]])),
            "edar_biomarker_N1_mask": (
                dimensions,
                np.array([[[1], [1]], [[1], [0]]]),
            ),
            "edar_biomarker_N2_mask": (
                dimensions,
                np.array([[[1], [1]], [[0], [1]]]),
            ),
        },
        coords={
            "run_id": ["0_Baseline   ", "1_Baseline   "],
            TEMPORAL_COORD: pd.date_range("2024-01-01", periods=2),
            REGION_COORD: ["08001"],
        },
    ).to_zarr(dataset_path)

    rows = _compute_canonical_sparsity(
        dataset_path=dataset_path,
        run_id="1_Baseline",
    ).set_index("target")

    assert rows.loc["cases", "sparsity_pct"] == 50.0
    assert rows.loc["wastewater", "sparsity_pct"] == 0.0
