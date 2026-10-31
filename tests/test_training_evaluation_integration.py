"""Bounded local acceptance of the training and evaluation pipeline."""

import csv
import hashlib
import json
import math

import pytest

from models.configs import OutputConfig, TrainingParams
from scripts.workflow.run_support import evaluate_best
from tests.dataset_test_helpers import (
    make_epi_config,
    make_tiny_epi_dataset,
    write_zarr_dataset,
)
from training.epiforecaster_trainer import EpiForecasterTrainer


@pytest.mark.integration
@pytest.mark.epiforecaster
@pytest.mark.parametrize("mobility_enabled", [False, True])
def test_train_checkpoint_and_evaluate_synthetic_dataset(tmp_path, mobility_enabled):
    dataset_path = tmp_path / "tiny.zarr"
    write_zarr_dataset(make_tiny_epi_dataset(periods=20, num_nodes=8), dataset_path)
    config = make_epi_config(
        dataset_path,
        model_options={
            "gnn_module": "gcn" if mobility_enabled else None,
            "type": {
                "cases": True,
                "regions": False,
                "biomarkers": True,
                "mobility": mobility_enabled,
            }
        },
    )
    config.training = TrainingParams(
        epochs=1,
        max_batches=1,
        batch_size=2,
        device="cpu",
        enable_mixed_precision=False,
        num_workers=0,
        val_workers=0,
        test_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        plot_forecasts=False,
        early_stopping_patience=None,
        run_final_evaluation=False,
    )
    config.output = OutputConfig(
        log_dir=str(tmp_path / "training"),
        experiment_name="integration",
        checkpoint_frequency=1,
        wandb_mode="disabled",
    )

    trainer = EpiForecasterTrainer(config, model_id="model")
    trainer.run()

    model_dir = tmp_path / "training/integration/model"
    checkpoint = model_dir / "checkpoints/best_model.pt"
    assert checkpoint.is_file()
    evaluation_dir = tmp_path / "evaluation"
    manifest_path = evaluate_best(checkpoint, evaluation_dir, "test", [], False)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoint"] == str(checkpoint)
    assert {item["path"] for item in manifest["artifacts"]} >= {
        "test_summary.json",
        "test_node_metrics.csv",
        "test_node_metrics_per_head.csv",
    }
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    for artifact in manifest["artifacts"]:
        path = evaluation_dir / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    summary = json.loads((evaluation_dir / "test_summary.json").read_text())
    assert math.isfinite(summary["eval_loss"])
    assert all(
        summary["metrics"][f"effective_count_{head}"] > 0
        for head in ("ww", "hosp", "cases", "deaths")
    )
    with (evaluation_dir / "test_node_metrics_per_head.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["target"] for row in rows} == {
        "wastewater",
        "hospitalizations",
        "cases",
        "deaths",
    }
    assert all(int(row["observed_count"]) > 0 for row in rows)
