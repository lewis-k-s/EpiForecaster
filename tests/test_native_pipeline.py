"""Bounded native pipeline contracts from input identities to evaluation files."""

import json
from pathlib import Path
from dataclasses import fields

from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from cli import cli
from data.build_native_source_sets import build_for_config
from data.native_inputs import native_dataset
from data.native_source_sets import load_source_set_manifest
from tests.native_fixtures import pipeline_config, _trainer_config, _write_native_root
from training.native_trainer import NativeRuntime, source_set_manifest_split
from evaluation.granular_export import GRANULAR_FIELDNAMES

pytestmark = pytest.mark.epiforecaster


@pytest.mark.parametrize(
    "strict,mobility,wastewater",
    [(True, True, True), (False, False, True), (True, False, False)],
)
def test_bounded_native_pipeline(
    tmp_path: Path, strict: bool, mobility: bool, wastewater: bool, monkeypatch
) -> None:
    config = pipeline_config(
        tmp_path,
        strict=strict,
        mobility=mobility,
        vaccination=True,
        wastewater=wastewater,
    )
    asset = build_for_config(config, Path(config.native.fold_manifest))
    manifest = load_source_set_manifest(Path(config.native.fold_manifest))
    dataset = native_dataset(config)
    positions = source_set_manifest_split(
        dataset.queries,
        [dataset.head_at(i) for i in range(len(dataset))],
        manifest,
        run_index=0,
    )
    sources = [
        {
            (dataset.queries[i].source_type, dataset.queries[i].source_id)
            for i in selected
        }
        for selected in positions
    ]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert sources[left].isdisjoint(sources[right])
    for left, right in ((0, 1), (1, 2)):
        assert max(
            dataset.queries[i].forecast_origin + config.model.forecast_horizon
            for i in positions[left]
        ) < min(dataset.queries[i].forecast_origin for i in positions[right])
    for selected in positions:
        assert {dataset.head_at(i) for i in selected} == (
            {"cases", "hosp", "deaths", "ww"}
            if wastewater
            else {"cases", "hosp", "deaths"}
        )
    assert asset["provenance"]["query_inventory_count"] == len(dataset)
    assert pd.read_parquet(
        tmp_path / "splits/assignments.parquet"
    ).source_id.dtype.name.startswith("string")
    _run_pipeline(config, monkeypatch)
    changed = config.model.input_window_length
    config.model.input_window_length += 1
    with pytest.raises(ValueError, match="stale"):
        NativeRuntime(config)
    config.model.input_window_length = changed
    cases = xr.open_zarr(tmp_path / "real_with_id.zarr").load()
    cases["cases"].values[0, 0, 0] += 0.1
    cases.to_zarr(tmp_path / "real_with_id.zarr", mode="w")
    with pytest.raises(ValueError, match="stale"):
        NativeRuntime(config)


def _run_pipeline(config, monkeypatch):
    """Train two steps, reload the checkpoint, and check both evaluation exports."""
    tmp_path = Path(config.native.root)
    runtime = NativeRuntime(config)
    initial = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    monkeypatch.setenv("EPIFORECASTER_MODEL_ID", "acceptance")
    result = CliRunner().invoke(
        cli, ["train", "epiforecaster", "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output
    model_root = tmp_path / "outputs/native_trainer_test/acceptance"
    assert not list(model_root.glob("*granular*"))
    assert not list(model_root.glob("test_*"))
    checkpoint = model_root / "checkpoints/best_model.pt"
    record = torch.load(checkpoint, weights_only=False)
    assert record["global_step"] == 2
    assert len(record["train_loss_trace"][0]) == 2
    assert np.isfinite(record["train_loss_trace"]).all()
    for prefix in (
        "input_adapter.",
        "base.backbone.",
        "base.cases_head.",
        "base.hosp_head.",
        "base.deaths_head.",
    ):
        assert any(
            not torch.equal(initial[k], v)
            for k, v in record["model_state_dict"].items()
            if k.startswith(prefix)
        ), prefix
    if config.native.wastewater_enabled:
        assert any(
            not torch.equal(initial[k], v)
            for k, v in record["model_state_dict"].items()
            if k.startswith("wastewater_head.")
        )
    assert record["best_val_selection_metric"] == "mean_target_wis"
    terminal = json.loads((model_root / "training_dynamics_manifest.json").read_text())
    assert terminal["status"] == "complete"
    # A saved resume setting must have no effect on standalone evaluation.
    record["config"]["training"]["resume_checkpoint_path"] = str(
        tmp_path / "missing_parent.pt"
    )
    torch.save(record, checkpoint)
    output = tmp_path / "evaluation"
    summaries = {}
    for split in ("val", "test"):
        previous = None
        for repeat in range(2):
            result = CliRunner().invoke(
                cli,
                [
                    "eval",
                    "epiforecaster",
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                    "--output",
                    str(output),
                    "--summary-json",
                    str(output / f"{split}_summary.json"),
                    "--granular",
                    "--override",
                    "training.max_batches=8",
                ],
            )
            assert result.exit_code == 0, result.output
            current = json.loads((output / f"{split}_summary.json").read_text())
            rows = pd.read_parquet(output / f"{split}_granular.parquet")
            if previous is not None:
                assert current["metrics"] == previous[0]["metrics"]
                pd.testing.assert_frame_equal(rows, previous[1])
            previous = current, rows
        summary = json.loads((output / f"{split}_summary.json").read_text())
        metrics = summary["metrics"]
        assert np.isfinite(summary["joint_wis"])
        summaries[split] = summary
        if config.native.wastewater_enabled:
            assert metrics["censored_count_ww"] > 0
            assert np.isfinite(metrics["censored_nll_ww"])
        heads = (
            ("cases", "hosp", "deaths", "ww")
            if config.native.wastewater_enabled
            else ("cases", "hosp", "deaths")
        )
        for head in heads:
            assert metrics[f"observed_count_{head}"] > 0
            assert np.isfinite(metrics[f"wis_{head}"])
            assert metrics[f"wis_{head}"] == pytest.approx(
                sum(
                    metrics[f"wis_{part}_{head}"]
                    for part in (
                        "point",
                        "dispersion",
                        "underprediction",
                        "overprediction",
                    )
                )
            )
        granular = pd.read_parquet(output / f"{split}_granular.parquet")
        metadata = json.loads(
            (output / f"{split}_granular.parquet.meta.json").read_text()
        )
        assert metadata["schema_version"] == "4"
        assert metadata["native_support"] is True
        assert metadata["confirmation_guard"] is True
        assert metadata["objective"] == "crps"
        assert metadata["generalization_claim"].startswith("held-out native sources")
        assert list(granular.columns) == GRANULAR_FIELDNAMES
        assert (granular.population > 0).all()
        assert (granular.horizon >= 1).all()
        municipal = granular[granular.source_type == "municipality"]
        assert municipal.region_id.astype(str).equals(municipal.source_id.astype(str))
        assert (
            granular.loc[granular.source_type != "municipality", "region_id"]
            .fillna("")
            .eq("")
            .all()
        )
        assert set(granular.target) == (
            {"cases", "hospitalizations", "deaths", "wastewater"}
            if config.native.wastewater_enabled
            else {"cases", "hospitalizations", "deaths"}
        )
        if config.native.wastewater_enabled:
            ww = granular[granular.target == "wastewater"]
            selected = getattr(runtime, f"{split}_loader").dataset.indices
            expected_assays = set()
            for i in selected[: 8 * config.training.batch_size]:
                item = runtime.dataset[i]
                if item["head"] != "ww":
                    continue
                observed = item["target_mask"] & (
                    item["target_extra"]["censor_code"] == 0
                )
                for assay, present in zip(
                    runtime.dataset._ww.assays, observed.any(dim=0), strict=True
                ):
                    if present:
                        expected_assays.add(f"{item['query'].source_id}::{assay}")
            assert set(ww.source_id) == expected_assays
            # Granular WIS has only uncensored assays; censor likelihood is reported separately.
            assert len(ww) == metrics["observed_count_ww"]
        excess = pd.read_csv(output / f"{split}_native_excess_deaths.csv")
        np.testing.assert_allclose(
            excess.observed_excess,
            excess.observed_all_cause - excess.expected_all_cause,
        )
        np.testing.assert_allclose(
            excess.predicted_excess,
            excess.predicted_all_cause - excess.expected_all_cause,
        )
        assert not granular.empty
        for suffix in (
            "native_aggregate_metrics.csv",
            "native_excess_deaths.csv",
            "window_support.csv",
            "example_selection.csv",
            "example_availability.csv",
        ):
            assert not pd.read_csv(output / f"{split}_{suffix}").empty
        for suffix in ("example_history.parquet", "example_rows.parquet"):
            assert not pd.read_parquet(output / f"{split}_{suffix}").empty
        json.loads((output / f"{split}_native_wis_summary.json").read_text())
    assert not (tmp_path / ".native_eval_runtime").exists()
    return record, summaries


def _assert_batch_equal(left, right) -> None:
    def compare(expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                compare(expected[key], actual[key])
        else:
            assert expected == actual

    for field in fields(left):
        if not field.name.startswith("_"):
            compare(getattr(left, field.name), getattr(right, field.name))


def _poison_after(root: Path, boundary: int) -> None:
    """Change future labels and their stored transforms, leaving masks fixed."""
    poison = 1e20
    cases = xr.open_zarr(root / "real_with_id.zarr").load()
    cases["cases"].values[:, boundary + 1 :, :] = np.log1p(poison)
    cases.to_zarr(root / "real_with_id.zarr", mode="w")
    path = root / "obs_admissions.zarr"
    admission = xr.open_zarr(path).load()
    future = admission.week_end.values > cases.date.values[boundary]
    admission["value"].values[future] = poison
    admission.to_zarr(path, mode="w")
    # Reprocess raw sources so fitted baselines and filtered histories are tested.
    frame = pd.read_csv(
        root / "deaths_source.csv", dtype={"codi_abs": "string"}, parse_dates=["data"]
    )
    frame["casos"] = frame["casos"].astype(float)
    frame.loc[frame.data > pd.Timestamp(cases.date.values[boundary]), "casos"] = poison
    frame.to_csv(root / "deaths_source.csv", index=False)
    frame = pd.read_csv(root / "wastewater_source.csv")
    future = pd.to_datetime(
        frame["id mostra"].str.rsplit("-", n=3).str[-3:].str.join("-")
    ) > pd.Timestamp(cases.date.values[boundary])
    for assay in ("N1", "N2", "IP4"):
        column = f"{assay}(CG/L)"
        frame.loc[future & (frame[column] > frame["LD(CG/L)"]), column] = poison
    frame.to_csv(root / "wastewater_source.csv", index=False)
    from tests.native_fixtures import process_pipeline_observations

    process_pipeline_observations(root)


def _write_zero_censored_history(root: Path) -> None:
    """First positive IP4 samples occur after both held-out boundaries."""
    from tests.native_fixtures import process_pipeline_observations

    path = root / "wastewater_source.csv"
    frame = pd.read_csv(path)
    # Source order is day then plant; the first positive value is on day 113.
    frame.loc[np.arange(len(frame)) // 3 <= 112, "IP4(CG/L)"] = 0.0
    frame.to_csv(path, index=False)
    process_pipeline_observations(root)


def test_held_out_poison_cannot_change_training_or_pre_origin_history(
    tmp_path: Path, monkeypatch
) -> None:
    """Rebuild splits and run the CLI for clean, poisoned val, and poisoned test inputs."""
    from data.native_batch import collate_native_batch

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_config = pipeline_config(clean_root, strict=True, mobility=True)
    _write_zero_censored_history(clean_root)
    clean_asset = build_for_config(
        clean_config, Path(clean_config.native.fold_manifest)
    )
    clean = NativeRuntime(clean_config)
    clean_record, clean_scores = _run_pipeline(clean_config, monkeypatch)
    clean.model.load_state_dict(clean_record["model_state_dict"])
    clean.model.eval()
    clean_batches = list(clean.train_loader)
    for split in ("val", "test"):
        selected = getattr(clean, f"{split}_loader").dataset.indices
        boundary = min(clean.dataset.queries[i].forecast_origin for i in selected)
        root = tmp_path / f"poison_{split}"
        root.mkdir()
        config = pipeline_config(root, strict=True, mobility=True)
        _write_zero_censored_history(root)
        _poison_after(root, boundary)
        asset = build_for_config(config, Path(config.native.fold_manifest))
        poisoned = NativeRuntime(config)
        assert asset["fold_metrics"] == clean_asset["fold_metrics"]
        assert poisoned.source_sets.assignments == clean.source_sets.assignments
        assert poisoned.split_provenance != clean.split_provenance
        # Check every training batch, including labels, context, OD, and routing.
        assert len(poisoned.train_loader) == len(clean.train_loader)
        for expected, actual in zip(clean_batches, poisoned.train_loader, strict=True):
            _assert_batch_equal(expected, actual)
        # At the boundary no poisoned observation is available yet. Later
        # origins may use earlier held-out observations once they are available.
        positions = [
            i
            for i, q in enumerate(clean.dataset.queries)
            if q.forecast_origin == boundary
        ]
        expected = collate_native_batch([clean.dataset[i] for i in positions])
        actual = collate_native_batch([poisoned.dataset[i] for i in positions])
        assert any(
            not torch.equal(expected.head_target[h], actual.head_target[h])
            for h in expected.head_target
            if expected.head_target[h] is not None
        )
        for field in (
            "hist",
            "hist_mask",
            "ctx_feat",
            "temporal_covariates",
            "od_node_features",
            "od_adjacency",
            "od_target_weights",
        ):
            torch.testing.assert_close(
                getattr(actual, field), getattr(expected, field), rtol=0, atol=0
            )
        for key in expected.relation_ctx:
            torch.testing.assert_close(
                actual.relation_ctx[key], expected.relation_ctx[key], rtol=0, atol=0
            )
        record, scores = _run_pipeline(config, monkeypatch)
        assert record["train_loss_trace"] == clean_record["train_loss_trace"]
        for key, value in clean_record["model_state_dict"].items():
            torch.testing.assert_close(
                record["model_state_dict"][key], value, rtol=0, atol=0
            )
        poisoned.model.load_state_dict(record["model_state_dict"])
        poisoned.model.eval()
        with torch.no_grad():
            left, right = clean.model(expected), poisoned.model(actual)
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(
                    right[key], left[key], rtol=0, atol=0, equal_nan=True
                )
        # Positive control: the poison reaches evaluation labels and scores.
        assert scores[split]["joint_wis"] > clean_scores[split]["joint_wis"] * 5
        rows = pd.read_parquet(root / f"evaluation/{split}_granular.parquet")
        assert rows.loc[rows.target == "deaths", "observed_raw"].max() > 1e15


def test_native_pipeline_compile_on_and_off(tmp_path: Path, monkeypatch) -> None:
    """One paired compiler acceptance run, with fixed mobility and vaccination."""
    records, scores = [], []
    for compile_enabled in (False, True):
        root = tmp_path / str(compile_enabled)
        root.mkdir()
        config = pipeline_config(root, strict=True, mobility=True)
        config.native.compile_sir_rollout = compile_enabled
        config.native.compile_loss_scores = compile_enabled
        build_for_config(config, Path(config.native.fold_manifest))
        record, score = _run_pipeline(config, monkeypatch)
        records.append(record)
        scores.append(score)
    assert (
        records[0]["model_state_dict"].keys() == records[1]["model_state_dict"].keys()
    )
    np.testing.assert_allclose(
        records[0]["train_loss_trace"],
        records[1]["train_loss_trace"],
        rtol=2e-4,
        atol=2e-5,
    )
    for key, value in records[0]["model_state_dict"].items():
        torch.testing.assert_close(
            records[1]["model_state_dict"][key], value, rtol=2e-4, atol=2e-5
        )
    for split in ("val", "test"):
        assert scores[0][split]["joint_wis"] == pytest.approx(
            scores[1][split]["joint_wis"], rel=2e-4
        )


def test_incomplete_admission_week_is_not_visible_to_daily_queries(
    tmp_path: Path,
) -> None:
    _write_native_root(tmp_path)
    config = _trainer_config(tmp_path)
    before = native_dataset(config)
    position = next(
        i
        for i, q in enumerate(before.queries)
        if q.source_id == "101"
        and q.forecast_origin == 15
        and before.head_at(i) == "deaths"
    )
    hist = before[position]["hist"].clone()
    admission = xr.open_zarr(tmp_path / "obs_admissions.zarr").load()
    admission["value"].values[2, 0] *= 1000
    admission.to_zarr(tmp_path / "obs_admissions.zarr", mode="w")
    after = native_dataset(config)
    assert torch.equal(hist, after[position]["hist"])
    sunday = next(
        i
        for i, q in enumerate(before.queries)
        if q.source_id == "101"
        and q.forecast_origin == 21
        and before.head_at(i) == "deaths"
    )
    assert not torch.equal(before[sunday]["hist"], after[sunday]["hist"])


@pytest.mark.parametrize("family", ["normal", "student_t"])
def test_censor_score_matches_training_and_evaluation(
    tmp_path: Path, family: str
) -> None:
    from tests.native_fixtures import _write_native_wastewater
    from data.native_batch import collate_native_batch
    from evaluation.losses import NativeJointInferenceLoss
    from evaluation.eval_loop import _native_wastewater_eval_view

    _write_native_root(tmp_path)
    _write_native_wastewater(tmp_path)
    config = _trainer_config(tmp_path, wastewater_enabled=True)
    dataset = native_dataset(config)
    row = next(i for i in range(len(dataset)) if dataset.head_at(i) == "ww")
    batch = collate_native_batch([dataset[row]])
    batch.head_extra["ww"]["censor_code"].fill_(1)
    batch.head_extra["ww"]["detection_limit"].fill_(10)
    loc = torch.full_like(batch.head_target["ww"], 7, requires_grad=True)
    scale = torch.full_like(loc, 0.7, requires_grad=True)
    outputs = {"dist_ww_loc": loc, "dist_ww_scale": scale}
    loss = NativeJointInferenceLoss(
        horizon_days=config.model.forecast_horizon,
        distribution_family=family,
        distribution_df=5.0,
        observation_loss="nll",
    )
    training, count, *_ = loss._wastewater_loss(
        outputs, batch, device=torch.device("cpu")
    )
    view = _native_wastewater_eval_view(
        model_outputs=outputs,
        batch=batch,
        dataset=dataset,
        criterion=loss,
        device=torch.device("cpu"),
    )
    assert view is not None
    assert training.item() == pytest.approx(
        (view["censored_nll_sum"] / view["censored_count"]).item(), rel=1e-6
    )
    assert count.item() == view["censored_count"].item()
    training.backward()
    assert torch.isfinite(loc.grad).all() and loc.grad.abs().sum() > 0
    assert torch.isfinite(scale.grad).all() and scale.grad.abs().sum() > 0
