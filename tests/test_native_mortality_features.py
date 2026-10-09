"""Causal mortality preprocessing and native input contracts."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from data.native_batch import collate_native_batch
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from data.preprocess.processors.native_mortality_features_processor import (
    MortalityFeaturesConfig,
    MORTALITY_FEATURE_VARIABLES,
    add_native_mortality_features,
    qa_native_mortality_features,
    write_native_mortality_features,
)
from models.configs import EpiForecasterConfig, NativeSupportConfig
from models.native_forecaster import NativeEpiForecaster, NativeForecasterConfig
from scripts.workflow import run_config, run_support
from scripts.workflow.config_templates import load_config_template
from tests.native_dataset_fixtures import ABS_POP, H, L, _write_native_root
from tests.native_fixtures import _write_native_wastewater


def _parent() -> xr.Dataset:
    values = np.arange(1, 21, dtype=np.float32)[:, None]
    mask = np.ones_like(values, dtype=bool)
    mask[2] = False
    return xr.Dataset(
        {
            "value": (("date", "abs_id"), values),
            "mask": (("date", "abs_id"), mask),
            "age": (("date", "abs_id"), np.ones_like(values)),
            "expected": (("date", "abs_id"), np.full_like(values, 2)),
            "excess": (("date", "abs_id"), values - 2),
        },
        coords={"date": np.datetime64("2020-01-01") + np.arange(20), "abs_id": ["101"]},
        attrs={"schema": "obs_deaths_abs_excess", "baseline": "2017-19"},
    )


def test_lag_means_exclude_current_and_future_and_use_observed_support() -> None:
    parent = _parent()
    config = MortalityFeaturesConfig()
    result = add_native_mortality_features(parent, config)
    # At day index 7 use values 1..7, excluding the masked count 3.
    assert result.mortality_mean_short[7, 0].item() == pytest.approx(25 / 6)
    assert result.mortality_coverage_short[7, 0].item() == pytest.approx(6 / 7)
    assert result.mortality_coverage_long[7, 0].item() == pytest.approx(6 / 14)
    assert result.mortality_mean_short[0, 0].item() == 0
    changed = parent.copy(deep=True)
    changed["value"].values[7:] = 10000
    changed["excess"] = changed.value - changed.expected
    altered = add_native_mortality_features(changed, config)
    xr.testing.assert_identical(
        result[list(MORTALITY_FEATURE_VARIABLES)].isel(date=slice(0, 8)),
        altered[list(MORTALITY_FEATURE_VARIABLES)].isel(date=slice(0, 8)),
    )
    assert qa_native_mortality_features(parent, result, config)["qa_passed"]
    delayed = add_native_mortality_features(parent, replace(config, lag_days=2))
    assert delayed.mortality_mean_short[7, 0].item() == pytest.approx(18 / 5)
    corrupted = result.copy(deep=True)
    corrupted.mortality_mean_short.values[7, 0] += 1
    with pytest.raises(AssertionError):
        qa_native_mortality_features(parent, corrupted, config)
    empty = parent.copy(deep=True)
    empty["mask"][:] = False
    for name in ("short", "long"):
        features = add_native_mortality_features(empty, config)
        assert not features[f"mortality_mean_{name}"].values.any()
        assert not features[f"mortality_coverage_{name}"].values.any()


def test_feature_asset_roundtrip_preserves_targets_and_rejects_overwrite(
    tmp_path,
) -> None:
    parent = _parent()
    config = MortalityFeaturesConfig()
    output = tmp_path / "mortality.zarr"
    write_native_mortality_features(
        add_native_mortality_features(parent, config), output
    )
    with xr.open_zarr(output) as restored:
        assert qa_native_mortality_features(parent, restored.load(), config)[
            "qa_passed"
        ]
    with pytest.raises(FileExistsError):
        write_native_mortality_features(parent, output)
    with pytest.raises(ValueError, match="positive integers"):
        MortalityFeaturesConfig(lag_days=0)


@pytest.mark.parametrize("include_vaccination", [False, True])
def test_native_inputs_enabled_disabled_and_both_abs_heads(
    tmp_path, include_vaccination
) -> None:
    _write_native_root(tmp_path)
    _write_native_wastewater(tmp_path)
    with xr.open_zarr(tmp_path / "obs_deaths_abs_excess.zarr") as opened:
        features = add_native_mortality_features(
            opened.load(), MortalityFeaturesConfig()
        )
    filename = "mortality.zarr"
    write_native_mortality_features(features, tmp_path / filename)
    base = NativeDatasetConfig(
        input_window_length=L,
        forecast_horizon=H,
        wastewater_enabled=True,
        include_vaccination=include_vaccination,
    )
    old = NativeSupportDataset(tmp_path, tmp_path / "real_with_id.zarr", base)
    new = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        replace(
            base,
            deaths_filename=filename,
            mortality_features_enabled=True,
        ),
    )
    assert old.queries == new.queries
    disabled = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        replace(base, deaths_filename=filename),
    )
    torch.testing.assert_close(disabled[0]["hist"], old[0]["hist"])
    representatives = {}
    for i, q in enumerate(new.queries):
        head = new.head_at(i)
        if q.source_id not in {"101", "01001", "E1"} or head in representatives:
            continue
        item, original = new[i], old[i]
        representatives[head] = item
        assert item["hist"].shape == (L, original["hist"].shape[1] + 5)
        torch.testing.assert_close(item["target"], original["target"])
        torch.testing.assert_close(item["hist"][:, :6], original["hist"][:, :6])
        torch.testing.assert_close(item["hist"][:, 11:], original["hist"][:, 6:])
        if head in {"cases", "ww"}:
            assert not item["hist"][:, 6:11].any()
        else:
            assert item["hist"][0, 6].item() == pytest.approx(
                np.log1p(1 / ABS_POP["101"] * 1e5)
            )
            assert item["hist"][-1, 9].item() == 1
            assert item["hist"][-1, 10].item() == pytest.approx(
                min(q.forecast_origin, 14) / 14
            )
    assert {"cases", "hosp", "deaths", "ww"} == representatives.keys()
    batch = collate_native_batch(list(representatives.values()))
    assert batch.hist.shape[-1] == new[0]["hist"].shape[-1] + 5
    model = NativeEpiForecaster(
        NativeForecasterConfig(
            f_native=batch.hist.shape[-1],
            d_ctx=batch.ctx_feat.shape[-1],
            sequence_length=L,
            forecast_horizon=H,
            mobility_enabled=False,
            wastewater_enabled=True,
            include_vaccination=include_vaccination,
        )
    )
    output = model(batch)
    loss = sum(
        output[key].mean()
        for key in ("pred_cases", "pred_deaths", "pred_hosp_weekly", "pred_ww")
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.input_adapter.input_proj.weight.grad[:, 11:16].abs().sum() > 0
    with pytest.raises(KeyError):
        NativeSupportDataset(
            tmp_path,
            tmp_path / "real_with_id.zarr",
            replace(base, mortality_features_enabled=True),
        )
    with pytest.raises(ValueError, match="one Zarr filename"):
        NativeSupportConfig(deaths_filename="../bad.zarr")
    with pytest.raises(ValueError, match="boolean"):
        NativeSupportConfig(mortality_features_enabled=1)


def test_immutable_mortality_preprocessing_and_train_template(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    workflow = load_config_template(
        repo / "workflow/config/mn5_native_mortality_features.yaml"
    )
    assert run_config.validate_workflow_contract(workflow) == "preprocess"
    pointer = tmp_path / workflow["preprocessing"]["input_pointer"]
    pointer.parent.mkdir(parents=True)
    parent_path = pointer.parent / "obs_deaths_abs_excess.zarr"
    _parent().to_zarr(parent_path, zarr_format=2)
    pointer.write_text(
        yaml.safe_dump(
            {
                "outs": [
                    {
                        "path": parent_path.name,
                        "md5": workflow["preprocessing"]["parent_md5"],
                    }
                ]
            }
        )
    )
    # Producer hashes come from the real source files; fixtures supply pinned data.
    for name in (
        "data/preprocess/processors/native_mortality_features_processor.py",
        "scripts/workflow/run_support.py",
    ):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((repo / name).read_bytes())
    run_root = tmp_path / "run"
    marker = run_support._run_native_mortality_features(workflow, tmp_path, run_root)
    assert json.loads((marker.parent / "qa.json").read_text())["qa_passed"]
    with pytest.raises(FileExistsError):
        run_support._run_native_mortality_features(workflow, tmp_path, run_root)
    workflow["preprocessing"]["parent_md5"] = "0" * 32 + ".dir"
    with pytest.raises(ValueError, match="pinned input"):
        run_support._run_native_mortality_features(
            workflow, tmp_path, tmp_path / "other"
        )
    train = load_config_template(
        repo / "workflow/config/drafts/mn5_native_mortality_10ep.yaml"
    )
    assert run_config.validate_workflow_contract(train) == "train"
    config = EpiForecasterConfig.load(
        repo / train["training"]["base_config"],
        overrides=train["training"]["overrides"],
        strict=True,
    )
    assert config.training.epochs == 10
    assert config.native.mortality_features_enabled
    assert config.native.validation_fraction == 0.3
    assert config.evaluation.minimum_validation_origins == 56
    assert config.native.wastewater_enabled
    assert config.data.dataset_path.endswith("real_with_id_v3.zarr")
