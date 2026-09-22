"""Native evaluation, WIS, and schema-v4 export tests (slice EVL-1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.query_contract import ForecastQuery
from evaluation.eval_loop import (
    assert_native_confirmation_period_safe,
    evaluate_native_loader,
)
from evaluation.granular_export import GRANULAR_FIELDNAMES
from evaluation.metrics import TorchWeightedIntervalScoreAccumulator
from tests.test_native_trainer_wiring import _trainer_config, _write_native_root
from training.native_trainer import NativeTrainer

pytestmark = pytest.mark.epiforecaster


def test_wis_matches_hand_computation_and_component_identity() -> None:
    accumulator = TorchWeightedIntervalScoreAccumulator(
        device=torch.device("cpu"), interval_levels=(0.5, 0.8), horizon=2
    )
    targets = torch.tensor([[0.0, 4.0]])
    median = torch.tensor([[1.0, 1.0]])
    intervals = {
        "50": (torch.tensor([[0.0, 0.0]]), torch.tensor([[2.0, 2.0]])),
        "80": (torch.tensor([[-1.0, -1.0]]), torch.tensor([[3.0, 3.0]])),
    }

    batch = accumulator.update(
        median=median,
        targets=targets,
        intervals=intervals,
        observed_mask=torch.ones_like(targets),
    )
    result = accumulator.finalize()

    # K=2 -> normalizer=2.5. Point components are 0.5*|error|/2.5.
    # Dispersion is ((0.5/2)*2 + (0.2/2)*4)/2.5 = 0.36.
    # The second target lies 1 above the 80% upper bound and 2 above the
    # 50% upper bound, so underprediction=(1+2)/2.5=1.2.
    assert batch.point.tolist()[0] == pytest.approx([0.2, 0.6])
    assert batch.dispersion.tolist()[0] == pytest.approx([0.36, 0.36])
    assert batch.underprediction.tolist()[0] == pytest.approx([0.0, 1.2])
    assert batch.overprediction.tolist()[0] == pytest.approx([0.0, 0.0])
    assert batch.wis.tolist()[0] == pytest.approx([0.56, 2.16])
    assert result.wis == pytest.approx(1.36)
    assert result.wis == pytest.approx(
        result.point
        + result.dispersion
        + result.underprediction
        + result.overprediction
    )
    assert result.wis_per_h == pytest.approx([0.56, 2.16])


def test_wis_zero_width_intervals_and_masked_rows() -> None:
    accumulator = TorchWeightedIntervalScoreAccumulator(
        device=torch.device("cpu"), interval_levels=(0.5,), horizon=1
    )
    target = torch.tensor([[2.0], [100.0]])
    point = torch.tensor([[2.0], [0.0]])
    bounds = torch.tensor([[2.0], [0.0]])
    batch = accumulator.update(
        median=point,
        targets=target,
        intervals={"50": (bounds, bounds)},
        observed_mask=torch.tensor([[1.0], [0.0]]),
    )
    result = accumulator.finalize()

    assert batch.wis[0, 0].item() == 0.0
    assert result.wis == 0.0
    assert result.dispersion == 0.0
    assert result.observed_count == 1
    assert result.effective_count == 1.0


def test_confirmation_period_guard_raises_unless_disabled() -> None:
    dates = np.datetime64("2020-12-28", "D") + np.arange(14, dtype="timedelta64[D]")
    query = ForecastQuery(
        source_type="municipality",
        source_id="01001",
        forecast_origin=1,
        horizon_days=7,
        requested_heads=frozenset({"cases"}),
    )
    with pytest.raises(ValueError, match="confirmation period"):
        assert_native_confirmation_period_safe(
            queries=[query], date_grid=dates, confirmation_guard=True
        )
    assert_native_confirmation_period_safe(
        queries=[query], date_grid=dates, confirmation_guard=False
    )


def test_native_eval_smoke_writes_v4_granular_and_aggregates(
    tmp_path: Path,
) -> None:
    _write_native_root(tmp_path)
    trainer = NativeTrainer(_trainer_config(tmp_path))
    assert trainer.val_loader is not None
    granular = tmp_path / "native_eval.parquet"
    aggregate_dir = tmp_path / "eval_outputs"

    loss, metrics, aggregate = evaluate_native_loader(
        model=trainer.model,
        loader=trainer.val_loader,
        criterion=trainer.criterion,
        device=torch.device("cpu"),
        split_name="Val",
        granular_output_path=granular,
        aggregate_output_dir=aggregate_dir,
    )

    assert np.isfinite(loss)
    for head in ("cases", "hosp", "deaths"):
        assert metrics[f"observed_count_{head}"] > 0
        assert np.isfinite(metrics[f"wis_{head}"])
        assert metrics[f"wis_{head}"] == pytest.approx(
            metrics[f"wis_point_{head}"]
            + metrics[f"wis_dispersion_{head}"]
            + metrics[f"wis_underprediction_{head}"]
            + metrics[f"wis_overprediction_{head}"]
        )

    rows = pd.read_parquet(granular)
    assert list(rows.columns) == GRANULAR_FIELDNAMES
    assert set(rows["source_type"]) == {"municipality", "abs", "comarca"}
    assert (
        rows.loc[rows["source_type"] != "municipality", "region_id"]
        .fillna("")
        .eq("")
        .all()
    )
    municipal = rows[rows["source_type"] == "municipality"]
    assert municipal["region_id"].astype(str).equals(municipal["source_id"].astype(str))
    assert (rows["population"] > 0).all()
    assert (rows["horizon"] >= 1).all()

    metadata = json.loads(
        granular.with_suffix(".parquet.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == "4"
    assert metadata["native_support"] is True
    assert metadata["confirmation_guard"] is True

    assert set(aggregate["aggregation"]) == {
        "source_horizon",
        "head_horizon",
        "head_all",
    }
    head_all = aggregate[aggregate["aggregation"] == "head_all"]
    assert set(head_all["target"]) == {"cases", "hospitalizations", "deaths"}
    for row in head_all.itertuples(index=False):
        assert row.wis == pytest.approx(
            row.wis_point
            + row.wis_dispersion
            + row.wis_underprediction
            + row.wis_overprediction
        )
    assert (aggregate_dir / "val_native_aggregate_metrics.csv").exists()
    summary_path = aggregate_dir / "val_native_wis_summary.json"
    assert summary_path.exists()
    assert set(json.loads(summary_path.read_text(encoding="utf-8"))) == {
        "cases",
        "hospitalizations",
        "deaths",
    }


def test_native_trainer_writes_stage4_metadata_after_best_state(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = _trainer_config(tmp_path)
    config.training.max_batches = 1
    config.output.write_granular_eval = True
    config.output.granular_eval_format = "csv"
    config.output.granular_eval_filename = "{split}_granular.csv"
    config.native.fold_index = None

    result = NativeTrainer(config).run()

    granular = Path(result["evaluation_artifacts"]["granular"])
    metadata = json.loads(
        granular.with_suffix(".csv.meta.json").read_text(encoding="utf-8")
    )
    assert granular.exists()
    assert metadata["schema_version"] == "4"
    assert metadata["experiment_name"] == "native_trainer_test"
    assert metadata["objective"] == "nll"
    assert metadata["fold"] is None
    assert metadata["generalization_claim"].startswith("held-out source")
