"""Controlled native CPU performance settings preserve scientific inputs."""

import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from data.native_batch import collate_native_batch
from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from models.configs import EpiForecasterConfig, NativeSupportConfig, TrainingParams
from models.sir_rollforward import SIRRollForward
from evaluation.losses import NativeJointInferenceLoss, _count_once_slots
from tests.native_dataset_fixtures import H, L, _write_native_root
from tests.test_native_batch import _item
from tests.test_native_losses import (
    H as LOSS_H,
    _daily_outputs,
    _default_three_head_batch,
    _make_batch,
    _query,
    _weekly_outputs,
)
from tests.test_native_trainer_artifacts import _artifact_config
from training.native_trainer import (
    NativeTrainer,
    compile_native_loss_scores,
    configure_native_worker_threads,
)
from tests.config_fixtures import config_fixture

pytestmark = pytest.mark.epiforecaster


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_od_cache_rejects_invalid_capacity(size: object) -> None:
    with pytest.raises(ValueError, match="od_graph_cache_size"):
        NativeDatasetConfig(od_graph_cache_size=size)
    with pytest.raises(ValueError, match="od_graph_cache_size"):
        NativeSupportConfig(od_graph_cache_size=size)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_worker_blas_rejects_invalid_budget(threads: object) -> None:
    with pytest.raises(ValueError, match="worker_blas_threads"):
        TrainingParams(worker_blas_threads=threads)


def test_perf_settings_survive_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config = config_fixture()
    updated = EpiForecasterConfig.apply_overrides(
        config,
        [
            "training.worker_blas_threads=1",
            "native.od_graph_cache_size=512",
            "native.cache_od_target_weights=true",
            "native.precompute_loss_metadata=true",
        ],
    )
    saved = tmp_path / "config.yaml"
    saved.write_text(yaml.safe_dump(updated.to_dict()))
    loaded = EpiForecasterConfig.from_file(str(saved))
    assert loaded.training.worker_blas_threads == 1
    assert loaded.native.od_graph_cache_size == 512
    assert loaded.native.cache_od_target_weights is True
    assert loaded.native.precompute_loss_metadata is True


def test_od_caches_preserve_inputs_and_bound_capacity(tmp_path: Path) -> None:
    _write_native_root(tmp_path, with_od=True)
    common = dict(input_window_length=L, forecast_horizon=H, spatial_encoder="hetero")
    control = NativeSupportDataset(
        tmp_path, tmp_path / "real_with_id.zarr", NativeDatasetConfig(**common)
    )
    cached = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(
            **common, od_graph_cache_size=2, cache_od_target_weights=True
        ),
    )
    positions = [
        next(i for i, q in enumerate(cached.queries) if q.forecast_origin == origin)
        for origin in (14, 15, 16)
    ]
    for position in positions * 2:
        expected, actual = control[position], cached[position]
        for key in ("hist", "od_node_features", "od_adjacency", "od_target_weights"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert list(cached._od_graph_cache) == [15, 16]
    assert not control._od_target_weight_cache
    for source_type, source_id in (("municipality", "01001"), ("abs", "101")):
        weights = cached._od_target_weights(source_type, source_id)
        assert cached._od_target_weights(source_type, source_id) is weights
        np.testing.assert_array_equal(
            weights, control._od_target_weights(source_type, source_id)
        )
    batch = collate_native_batch([cached[p] for p in positions])
    expected_batch = collate_native_batch([control[p] for p in positions])
    torch.testing.assert_close(batch.od_adjacency, expected_batch.od_adjacency)


def test_thread_budget_applies_to_new_process_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in names:
        monkeypatch.setenv(name, "4")
    parent_threads = torch.get_num_threads()
    configure_native_worker_threads(None, device_type="cpu")
    assert all(os.environ[name] == "4" for name in names)
    with pytest.raises(ValueError, match="CUDA native spawn"):
        configure_native_worker_threads(1, device_type="cpu")
    configure_native_worker_threads(1, device_type="cuda")
    assert torch.get_num_threads() == parent_threads
    runtime = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json, os, torch; "
                "print(json.dumps({'threads':torch.get_num_threads(), "
                "'blas':os.environ['OPENBLAS_NUM_THREADS']}))",
            ],
            text=True,
        )
    )
    assert runtime == {"threads": 1, "blas": "1"}


@pytest.mark.parametrize("precompute", [False, True])
def test_collation_loss_metadata_switch(tmp_path: Path, precompute: bool) -> None:
    _write_native_root(tmp_path, with_od=True)
    dataset = NativeSupportDataset(
        tmp_path,
        tmp_path / "real_with_id.zarr",
        NativeDatasetConfig(input_window_length=L, forecast_horizon=H),
    )
    batch = collate_native_batch([dataset[0]], precompute_loss_metadata=precompute)
    head = next(h for h, rows in batch.head_index.items() if rows is not None)
    assert batch.loss_metadata_valid(head) is precompute
    batch.to("cpu")
    assert batch.loss_metadata_valid(head) is precompute


@pytest.mark.parametrize("observation_loss", ["nll", "crps"])
def test_precomputed_routing_preserves_losses_and_gradients(
    observation_loss: str,
) -> None:
    batches = [_default_three_head_batch(), _default_three_head_batch()]
    batches[1].prepare_loss_metadata()
    criterion = NativeJointInferenceLoss(
        horizon_days=LOSS_H,
        observation_loss=observation_loss,
        distribution_family="student_t",
    )
    source = {**_daily_outputs(3), **_weekly_outputs(1, torch.tensor([[True, False]]))}
    scores, gradients = [], []
    for batch in batches:
        outputs = {
            k: v.clone().requires_grad_(v.is_floating_point())
            for k, v in source.items()
        }
        result = criterion(outputs, batch)
        result["total"].backward()
        scores.append(result)
        gradients.append({k: v.grad for k, v in outputs.items() if v.grad is not None})
    for key in scores[0]:
        torch.testing.assert_close(scores[0][key], scores[1][key], rtol=0, atol=0)
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)


def test_routing_metadata_detects_mutation_and_still_deduplicates() -> None:
    batch = _default_three_head_batch()
    batch.head_index["cases"] = torch.tensor([0, 0])
    batch.head_population["cases"] = torch.tensor([250.0, 250.0])
    batch.prepare_loss_metadata()
    assert _count_once_slots(batch, batch.head_index["cases"]).tolist() == [0]
    batch.head_index["cases"][1] = 2
    assert not batch.loss_metadata_valid("cases")
    assert _count_once_slots(batch, batch.head_index["cases"]).tolist() == [0, 1]
    batch.prepare_loss_metadata()
    batch.queries[2] = replace(batch.queries[0], requested_heads=frozenset({"deaths"}))
    assert not batch.loss_metadata_valid("cases")
    assert _count_once_slots(batch, batch.head_index["cases"]).tolist() == [0]


def test_population_metadata_does_not_hide_later_invalid_values() -> None:
    batch = _default_three_head_batch()
    batch.prepare_loss_metadata()
    batch.head_population["cases"][0] = 0.0
    criterion = NativeJointInferenceLoss(horizon_days=LOSS_H)
    with pytest.raises(ValueError, match="positive and finite"):
        criterion(
            {**_daily_outputs(3), **_weekly_outputs(1, torch.tensor([[True, False]]))},
            batch,
        )


def test_routing_metadata_survives_pickle_and_rejects_stale_values() -> None:
    batch = _default_three_head_batch()
    batch.prepare_loss_metadata()
    copied = pickle.loads(pickle.dumps(batch))
    assert copied.loss_metadata_valid("cases")
    copied.head_population["cases"][0] = 0.0
    copied = pickle.loads(pickle.dumps(copied))
    assert not copied.loss_metadata_valid("cases")
    assert copied.loss_metadata_valid("hosp")


def test_routing_metadata_survives_real_spawned_loader() -> None:
    dataset = [_item("municipality", {"cases"}, source_id=str(i)) for i in (1, 2, 3, 4)]
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        multiprocessing_context="spawn",
        timeout=60,
        collate_fn=partial(collate_native_batch, precompute_loss_metadata=True),
    )
    batch = next(iter(loader))
    for head, rows in batch.head_index.items():
        if rows is not None:
            assert batch.loss_metadata_valid(head)
            assert batch.cached_count_once_slots(rows) is not None


def test_trainer_records_active_loss_metadata(tmp_path: Path) -> None:
    _write_native_root(tmp_path)
    config = _artifact_config(tmp_path)
    config.native.precompute_loss_metadata = True
    trainer = NativeTrainer(config, model_id="metadata_active")
    trainer.run()
    samples = [
        json.loads(line)
        for line in (trainer.experiment_dir / "training_dynamics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert samples
    assert all(sample["loss_metadata_active_heads"] > 0 for sample in samples)


def test_wastewater_metadata_preserves_mask_and_censor_checks() -> None:
    batch = _make_batch(
        queries=[_query("edar", "E1", 10, {"ww"})],
        ww_rows=(0,),
        daily_width=2,
        ww_mask=torch.tensor([[[True, True, False], [True, True, True]]]),
        ww_censor_code=torch.tensor([[[0, 1, 255], [0, 0, 0]]], dtype=torch.uint8),
    )
    cached = deepcopy(batch)
    cached.prepare_loss_metadata()
    assert cached._ww_has_censored
    criterion = NativeJointInferenceLoss(
        horizon_days=2, observation_loss="crps", distribution_family="student_t"
    )
    loc = torch.zeros(1, 2, 3)
    outputs = {
        "dist_ww_loc": loc,
        "pred_ww": loc,
        "dist_ww_scale": torch.ones_like(loc),
    }
    for key, value in criterion(outputs, batch).items():
        torch.testing.assert_close(
            value, criterion(outputs, cached)[key], rtol=0, atol=0
        )
    cached.head_extra["ww"]["censor_code"][0, 0, 0] = 2
    assert not cached.loss_metadata_valid("ww")
    with pytest.raises(ValueError, match="censor_code 0 or 1"):
        criterion(outputs, cached)
    with pytest.raises(ValueError, match="censor_code 0 or 1"):
        cached.prepare_loss_metadata()


def test_compiled_sir_core_preserves_values_gradients_and_strict_checks() -> None:
    eager = SIRRollForward(strict=True)
    compiled = SIRRollForward(strict=True)
    compiled._compiled_rollout = torch.compile(
        compiled._rollout, backend="aot_eager", fullgraph=True, dynamic=True
    )
    rates = {
        "beta_t": torch.full((3, 5), 0.2),
        "gamma_t": torch.full((3, 5), 0.1),
        "mortality_t": torch.full((3, 5), 0.005),
        "hospitalization_rate_t": torch.full((3, 5), 0.02),
        "hospital_recovery_t": torch.full((3, 5), 0.03),
        "hospital_mortality_t": torch.full((3, 5), 0.01),
    }
    states = {
        "S0": torch.full((3,), 0.7),
        "I0": torch.full((3,), 0.2),
        "H0": torch.full((3,), 0.05),
        "R0": torch.full((3,), 0.05),
        "population": torch.ones(3),
    }
    outputs, gradients = [], []
    for model in (eager, compiled):
        inputs = {k: v.clone().requires_grad_(True) for k, v in rates.items()}
        result = model(**inputs, **states)
        sum(result[k].sum() for k in result if k != "beta_t").backward()
        outputs.append(result)
        gradients.append({k: v.grad for k, v in inputs.items()})
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key])
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key])
    invalid = dict(states, population=torch.zeros(3))
    with pytest.raises(ValueError, match="Population must be positive"):
        compiled(**rates, **invalid)
    invalid = dict(states, S0=torch.full((3,), -0.1))
    with pytest.raises(ValueError, match="Initial states must be non-negative"):
        compiled(**rates, **invalid)


def test_compiled_scores_preserve_masked_loss_and_gradients() -> None:
    eager = NativeJointInferenceLoss(
        horizon_days=LOSS_H, observation_loss="crps", distribution_family="student_t"
    )
    compiled = deepcopy(eager)
    compile_native_loss_scores(compiled, backend="aot_eager")
    values, gradients = [], []
    for criterion in (eager, compiled):
        loc_source = torch.tensor([[0.0, -10.0, 0.2, 50.0]] * 2, requires_grad=True)
        scale_source = torch.tensor([[0.0, 0.3, 0.1, 0.5]] * 2, requires_grad=True)
        loc, scale = loc_source[:, 1:], scale_source[:, 1:]
        assert not loc.is_contiguous()
        score = criterion._weighted_masked_distribution_loss_from_weights(
            loc=loc,
            scale=scale,
            target=torch.tensor([[1.0, 0.1, float("nan")]] * 2),
            weights=torch.tensor([[1.0, 1.0, 0.0]] * 2),
        )
        score.backward()
        values.append(score)
        gradients.append((loc_source.grad, scale_source.grad))
    torch.testing.assert_close(values[0], values[1])
    for before, after in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(before, after)


def test_compiled_scores_preserve_variable_assay_shapes_and_broadcasts() -> None:
    eager = NativeJointInferenceLoss(
        horizon_days=28, observation_loss="crps", distribution_family="student_t"
    )
    compiled = deepcopy(eager)
    compile_native_loss_scores(compiled, backend="aot_eager")
    for shape in ((3, 28, 3), (6, 28, 3), (6, 4), (1, 28, 3)):
        scores, gradients = [], []
        for criterion in (eager, compiled):
            loc = torch.linspace(-1, 1, int(np.prod(shape))).reshape(shape)
            loc.requires_grad_(True)
            scale = torch.full(shape, 0.3, requires_grad=True)
            score = criterion._weighted_masked_distribution_loss_from_weights(
                loc=loc,
                scale=scale,
                target=torch.zeros(shape),
                weights=torch.ones(shape),
            )
            score.backward()
            scores.append(score)
            gradients.append((loc.grad, scale.grad))
        torch.testing.assert_close(scores[0], scores[1])
        for before, after in zip(gradients[0], gradients[1], strict=True):
            torch.testing.assert_close(before, after)
    parameters = dict(
        loc=torch.zeros(3, 28, 3),
        scale=torch.ones(3, 28, 3),
        target=torch.zeros(3, 28, 3),
        weights=torch.ones(1, 28, 3),
    )
    torch.testing.assert_close(
        compiled._weighted_masked_distribution_loss_from_weights(**parameters),
        eager._weighted_masked_distribution_loss_from_weights(**parameters),
    )
