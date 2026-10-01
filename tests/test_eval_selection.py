from pathlib import Path

import pandas as pd

from evaluation.selection import (
    load_window_selection_specs_from_granular,
    select_windows_by_loss,
    select_worst_windows_by_loss,
)


def test_granular_artifact_selects_exact_windows_by_crps(tmp_path: Path) -> None:
    granular_path = tmp_path / "test_granular.parquet"
    pd.DataFrame(
        [
            ("test", "cases", 1, 10, 1.0),
            ("test", "cases", 1, 10, 3.0),
            ("test", "hospitalizations", 1, 10, 10.0),
            ("test", "cases", 2, 11, 1.0),
            ("test", "cases", 3, 12, 3.0),
            ("test", "cases", 4, 13, 8.0),
            ("val", "cases", 5, 14, 100.0),
        ],
        columns=["split", "target", "node_id", "window_start", "crps"],
    ).assign(abs_error=100.0).to_parquet(granular_path)

    specs = load_window_selection_specs_from_granular(
        granular_csv=granular_path,
        split="test",
    )
    assert [(spec.node_id, spec.window_start, spec.score) for spec in specs] == [
        (2, 11, 1.0),
        (3, 12, 3.0),
        (1, 10, 6.0),
        (4, 13, 8.0),
    ]
    assert specs[2].observed_targets == ("cases", "hospitalizations")
    assert specs[2].observed_points == 3

    quartiles = select_windows_by_loss(window_specs=specs, samples_per_group=1)
    assert [quartiles[name][0].node_id for name in quartiles] == [2, 3, 1, 4]
    worst = select_worst_windows_by_loss(window_specs=specs, k=2)
    assert [(spec.node_id, spec.window_start) for spec in worst["Worst MAE"]] == [
        (4, 13),
        (1, 10),
    ]
