from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import logging
import math
import pandas as pd
import pytest
import torch

import evaluation.eval_loop as eval_loop_module
from dataviz.granular_comparison import compare_granular_csvs
from evaluation.eval_loop import eval_checkpoint, evaluate_loader
from evaluation.losses import JointInferenceLoss


class _DummyDataset:
    def __init__(self) -> None:
        self._temporal_coords = list(pd.date_range("2024-01-01", periods=8, freq="D"))
        self.aligned_data_path = Path("/tmp/dummy-dataset.zarr")
        self._region_ids = ["08001", "08002"]
        self._region_labels = ["Region A", "08002"]
        self._region_name_source = Path("/tmp/regions.geojson")
        self.run_id = "real"

    def __len__(self) -> int:
        return 2


class _DummyLoader:
    def __init__(self, batches: list[object]) -> None:
        self._batches = batches
        self.dataset = _DummyDataset()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward_batch(self, batch_data, region_embeddings=None, **kwargs):  # noqa: ANN001
        return batch_data.model_outputs, batch_data.targets_dict


def _make_batch() -> SimpleNamespace:
    batch = SimpleNamespace(
        target_node=torch.tensor([0, 1], dtype=torch.long),
        window_start=torch.tensor([0, 1], dtype=torch.long),
        node_labels=["region-a", "region-b"],
        hosp_hist=torch.zeros((2, 3, 3), dtype=torch.float32),
        model_outputs={
            "pred_hosp": torch.tensor(
                [[0.0, 1.0, 2.5], [0.0, 1.5, 3.0]],
                dtype=torch.float32,
            ),
            "pred_ww": torch.tensor(
                [[0.0, 0.4, 0.4], [0.0, 0.5, 0.5]],
                dtype=torch.float32,
            ),
            "pred_cases": torch.tensor(
                [[0.0, 3.0, 4.0], [0.0, 5.0, 6.0]],
                dtype=torch.float32,
            ),
            "pred_deaths": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]],
                dtype=torch.float32,
            ),
            "physics_residual": torch.zeros((2, 2), dtype=torch.float32),
        },
        targets_dict={
            "hosp": torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32),
            "ww": torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            "cases": torch.tensor([[2.0, 4.0], [8.0, 6.0]], dtype=torch.float32),
            "deaths": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            "hosp_mask": torch.ones((2, 2), dtype=torch.float32),
            "ww_mask": torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
            "cases_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            "deaths_mask": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        },
        mob_batch=None,
    )
    batch.to = lambda device, **_: batch
    return batch


def _assert_metric_dicts_match(
    left: dict[str, float | list[float]],
    right: dict[str, float | list[float]],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, list):
            assert left_value == pytest.approx(right_value)
            continue
        if isinstance(left_value, float) and isinstance(right_value, float):
            if math.isnan(left_value) and math.isnan(right_value):
                continue
        assert left_value == pytest.approx(right_value)


def test_evaluate_loader_writes_granular_csv_without_changing_metrics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _DummyModel()
    batch = _make_batch()
    loader = _DummyLoader([batch])
    criterion = JointInferenceLoss(obs_weight_sum=4.0, w_sir=0.0)

    baseline_loss, baseline_metrics, baseline_node_mae = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        horizon=2,
        device=torch.device("cpu"),
    )

    granular_csv = tmp_path / "eval_granular.csv"
    with caplog.at_level(logging.INFO):
        granular_loss, granular_metrics, granular_node_mae = evaluate_loader(
            model=model,
            loader=loader,
            criterion=criterion,
            horizon=2,
            device=torch.device("cpu"),
            granular_csv_path=granular_csv,
            granular_metadata={"max_batches": 50},
            log_every=1,
        )

    assert granular_loss == pytest.approx(baseline_loss)
    _assert_metric_dicts_match(granular_metrics, baseline_metrics)
    assert granular_node_mae == pytest.approx(baseline_node_mae)
    assert any("stage=batch_fetched" in record.message for record in caplog.records)
    assert any("stage=granular_write_complete" in record.message for record in caplog.records)

    df = pd.read_csv(granular_csv, dtype={"region_id": str})
    assert list(df.columns) == [
        "split",
        "target",
        "node_id",
        "region_id",
        "region_label",
        "window_start",
        "window_start_date",
        "horizon",
        "target_index",
        "target_date",
        "observed",
        "abs_error",
        "sq_error",
        "smape_num",
        "smape_den",
    ]
    assert len(df) == (
        granular_metrics["observed_count_hosp"]
        + granular_metrics["observed_count_ww"]
        + granular_metrics["observed_count_cases"]
        + granular_metrics["observed_count_deaths"]
    )
    assert not (
        (df["target"] == "cases") & (df["node_id"] == 0) & (df["horizon"] == 2)
    ).any()
    assert not (
        (df["target"] == "wastewater") & (df["node_id"] == 1) & (df["horizon"] == 1)
    ).any()

    hosp_row = df[
        (df["target"] == "hospitalizations")
        & (df["node_id"] == 0)
        & (df["horizon"] == 1)
    ].iloc[0]
    assert hosp_row["window_start"] == 0
    assert hosp_row["region_id"] == "08001"
    assert hosp_row["region_label"] == "Region A"
    assert hosp_row["window_start_date"] == "2024-01-01T00:00:00"
    assert hosp_row["target_index"] == 3
    assert hosp_row["target_date"] == "2024-01-04T00:00:00"
    assert hosp_row["observed"] == pytest.approx(1.0)
    assert hosp_row["abs_error"] == pytest.approx(0.0)

    fallback_row = df[
        (df["target"] == "hospitalizations")
        & (df["node_id"] == 1)
        & (df["horizon"] == 1)
    ].iloc[0]
    assert fallback_row["region_id"] == "08002"
    assert fallback_row["region_label"] == "08002"

    meta = pd.read_json(granular_csv.with_suffix(".csv.meta.json"), typ="series")
    assert meta["schema_version"] == "1"
    assert bool(meta["observed_only"]) is True
    assert meta["max_batches"] == 50
    assert meta["region_name_source"] == "/tmp/regions.geojson"
    assert meta["run_id"] == "real"
    assert "training_seed" in meta.index
    assert "node_split_strategy" in meta.index
    assert "node_split_population_bins" in meta.index
    assert "val_split" in meta.index
    assert "test_split" in meta.index


class _ExplodingModel(_DummyModel):
    def forward_batch(self, batch_data, region_embeddings=None, **kwargs):  # noqa: ANN001
        raise RuntimeError("boom")


def test_evaluate_loader_logs_batch_context_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _ExplodingModel()
    batch = _make_batch()
    loader = _DummyLoader([batch])
    criterion = JointInferenceLoss(obs_weight_sum=4.0, w_sir=0.0)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="boom"):
            evaluate_loader(
                model=model,
                loader=loader,
                criterion=criterion,
                horizon=2,
                device=torch.device("cpu"),
                split_name="Eval",
                log_every=1,
            )

    assert any(
        "[eval] Evaluation failed: split=Eval batch=0 stage=forward_batch"
        in record.message
        for record in caplog.records
    )


def test_eval_checkpoint_threads_eval_overrides_to_loader_and_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel()
    config = SimpleNamespace(
        data=SimpleNamespace(dataset_path=Path("/tmp/dummy-dataset.zarr"), window_stride=7),
        model=SimpleNamespace(forecast_horizon=2, input_window_length=3),
        output=SimpleNamespace(
            write_granular_eval=True,
            resolve_granular_eval_filename=lambda *, split: f"{split}_granular.csv",
            experiment_name="crossval-test",
        ),
        training=SimpleNamespace(
            loss=SimpleNamespace(),
            node_split_population_bins=5,
            node_split_strategy="random",
            test_split=0.2,
            val_split=0.1,
            seed=42,
            batch_size=32,
            prefetch_factor=4,
            pin_memory=True,
        ),
    )
    dataset = _DummyDataset()
    loader = SimpleNamespace(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
    )
    captured: dict[str, object] = {}

    def _fake_load_model_from_checkpoint(checkpoint_path, *, device, overrides):  # noqa: ANN001
        captured["load_device"] = device
        captured["load_overrides"] = overrides
        return model, config, {"checkpoint_path": checkpoint_path}

    def _fake_build_loader_from_config(  # noqa: ANN001
        cfg,
        *,
        split,
        batch_size,
        device,
        num_workers,
        pin_memory,
        prefetch_factor,
    ):
        captured["loader_args"] = {
            "cfg": cfg,
            "split": split,
            "batch_size": batch_size,
            "device": device,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor,
        }
        return loader, None

    def _fake_get_loss_from_config(*args, **kwargs):  # noqa: ANN001
        return SimpleNamespace()

    def _fake_evaluate_loader(**kwargs):  # noqa: ANN001
        captured["evaluate_kwargs"] = kwargs
        return 1.25, {"mae": 2.5}, {1: 0.5}

    monkeypatch.setattr(
        eval_loop_module,
        "load_model_from_checkpoint",
        _fake_load_model_from_checkpoint,
    )
    monkeypatch.setattr(
        eval_loop_module,
        "build_loader_from_config",
        _fake_build_loader_from_config,
    )
    monkeypatch.setattr(
        eval_loop_module,
        "get_loss_from_config",
        _fake_get_loss_from_config,
    )
    monkeypatch.setattr(eval_loop_module, "evaluate_loader", _fake_evaluate_loader)

    checkpoint_path = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("stub", encoding="utf-8")

    result = eval_checkpoint(
        checkpoint_path=checkpoint_path,
        split="test",
        device="cpu",
        overrides=["training.device=cpu", "training.prefetch_factor=2"],
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=0,
        log_every=3,
    )

    assert result["eval_loss"] == pytest.approx(1.25)
    assert result["eval_metrics"]["mae"] == pytest.approx(2.5)
    assert captured["loader_args"] == {
        "cfg": config,
        "split": "test",
        "batch_size": 8,
        "device": "cpu",
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 0,
    }
    assert captured["evaluate_kwargs"]["log_every"] == 3
    assert captured["evaluate_kwargs"]["granular_metadata"]["eval_num_workers"] == 0
    assert captured["evaluate_kwargs"]["granular_metadata"]["eval_prefetch_factor"] is None
    assert captured["evaluate_kwargs"]["granular_metadata"]["eval_pin_memory"] is False


def _write_granular_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compare_granular_csvs_strict_join_and_aggregates(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    base_rows = [
        {
            "split": "test",
            "target": "cases",
            "node_id": 1,
            "region_id": "08001",
            "region_label": "region-a",
            "window_start": 0,
            "window_start_date": "2024-01-01",
            "horizon": 1,
            "target_index": 3,
            "target_date": "2024-01-04",
            "observed": 10.0,
            "abs_error": 2.0,
            "sq_error": 4.0,
            "smape_num": 4.0,
            "smape_den": 20.0,
        },
        {
            "split": "test",
            "target": "cases",
            "node_id": 1,
            "region_id": "08001",
            "region_label": "region-a",
            "window_start": 0,
            "window_start_date": "2024-01-01",
            "horizon": 2,
            "target_index": 4,
            "target_date": "2024-01-05",
            "observed": 12.0,
            "abs_error": 1.0,
            "sq_error": 1.0,
            "smape_num": 2.0,
            "smape_den": 24.0,
        },
    ]
    cand_rows = [
        {
            **base_rows[0],
            "abs_error": 1.0,
            "sq_error": 1.0,
            "smape_num": 2.0,
            "smape_den": 20.0,
        },
        {
            **base_rows[1],
            "abs_error": 3.0,
            "sq_error": 9.0,
            "smape_num": 6.0,
            "smape_den": 24.0,
        },
    ]

    _write_granular_fixture(baseline_csv, base_rows)
    _write_granular_fixture(candidate_csv, cand_rows)

    output_dir = tmp_path / "compare"
    artifacts = compare_granular_csvs(
        baseline_csv=baseline_csv,
        candidate_csv=candidate_csv,
        output_dir=output_dir,
    )

    assert artifacts["matched_rows"] == 2

    paired_df = pd.read_csv(output_dir / "paired_row_deltas.csv")
    assert paired_df["abs_error_uplift"].tolist() == pytest.approx([1.0, -2.0])

    horizon_df = pd.read_csv(output_dir / "horizon_aggregates.csv")
    horizon_one = horizon_df[horizon_df["horizon"] == 1].iloc[0]
    assert horizon_one["baseline_mae"] == pytest.approx(2.0)
    assert horizon_one["candidate_mae"] == pytest.approx(1.0)
    assert horizon_one["abs_error_uplift_mean"] == pytest.approx(1.0)

    region_df = pd.read_csv(output_dir / "region_aggregates.csv")
    region_row = region_df.iloc[0]
    assert region_row["baseline_mae"] == pytest.approx(1.5)
    assert region_row["candidate_mae"] == pytest.approx(2.0)
    assert region_row["abs_error_uplift_mean"] == pytest.approx(-0.5)
    assert region_row["baseline_rmse"] == pytest.approx((2.5) ** 0.5)
    assert region_row["candidate_rmse"] == pytest.approx((5.0) ** 0.5)

    expected_plots = [
        "region_time_heatmap.png",
        "rolling_time_uplift.png",
        "horizon_uplift_curve.png",
        "region_gain_loss_bars.png",
        "target_summary.png",
        "target_choropleths.png",
    ]
    for plot_name in expected_plots:
        assert (output_dir / plot_name).exists()


def test_compare_granular_csvs_fails_on_poor_join_coverage(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    _write_granular_fixture(
        baseline_csv,
        [
            {
                "split": "test",
                "target": "cases",
                "node_id": 1,
                "region_id": "08001",
                "region_label": "region-a",
                "window_start": 0,
                "window_start_date": "2024-01-01",
                "horizon": 1,
                "target_index": 3,
                "target_date": "2024-01-04",
                "observed": 10.0,
                "abs_error": 2.0,
                "sq_error": 4.0,
                "smape_num": 4.0,
                "smape_den": 20.0,
            },
            {
                "split": "test",
                "target": "cases",
                "node_id": 1,
                "region_id": "08001",
                "region_label": "region-a",
                "window_start": 0,
                "window_start_date": "2024-01-01",
                "horizon": 2,
                "target_index": 4,
                "target_date": "2024-01-05",
                "observed": 12.0,
                "abs_error": 1.0,
                "sq_error": 1.0,
                "smape_num": 2.0,
                "smape_den": 24.0,
            },
        ],
    )
    _write_granular_fixture(
        candidate_csv,
        [
            {
                "split": "test",
                "target": "cases",
                "node_id": 1,
                "region_id": "08001",
                "region_label": "region-a",
                "window_start": 0,
                "window_start_date": "2024-01-01",
                "horizon": 1,
                "target_index": 3,
                "target_date": "2024-01-04",
                "observed": 10.0,
                "abs_error": 1.0,
                "sq_error": 1.0,
                "smape_num": 2.0,
                "smape_den": 20.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="Granular join coverage below threshold"):
        compare_granular_csvs(
            baseline_csv=baseline_csv,
            candidate_csv=candidate_csv,
            output_dir=tmp_path / "compare",
        )
