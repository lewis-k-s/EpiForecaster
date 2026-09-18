import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml

from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from evaluation import research_pipeline as pipeline
from models.configs import EpiForecasterConfig
from scripts.workflow import run_config, run_support


@pytest.fixture
def config():
    return EpiForecasterConfig.load(
        "configs/train_epifor_smoke.yaml",
        overrides=[
            "training.split_strategy=research",
            "training.research_role_manifest_path=configs/research/loss_roles_v1.json",
            "model.forecast_horizon=28",
        ],
    )


@pytest.fixture
def manifest():
    return json.loads(Path("configs/research/loss_roles_v1.json").read_text())


@pytest.mark.parametrize("crossval", [False, True])
def test_fit_cases_excludes_later_labels_and_held_out_nodes(
    monkeypatch, config, manifest, crossval
):
    dates = pd.to_datetime(["2020-09-29", "2020-09-30", "2020-10-01"])
    dataset = xr.Dataset(
        {
            "cases": (
                (TEMPORAL_COORD, REGION_COORD),
                [[1.0, np.nan], [-1.0, 4.0], [999.0, 999.0]],
            ),
            "cases_mask": (
                (TEMPORAL_COORD, REGION_COORD),
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            ),
        },
        coords={TEMPORAL_COORD: dates, REGION_COORD: ["a", "b"]},
        attrs={
            "log_transformed": True,
            "population_norm": True,
            "preprocessing_config_yaml": yaml.safe_dump(
                {
                    "smoothing": {"parameter_fit_end_date": "2020-09-30"},
                    "source_provenance": {
                        "causal_preprocessing_contract": "loss-research-v1"
                    },
                }
            ),
        },
    )
    monkeypatch.setattr(
        pipeline.EpiDataset, "load_canonical_dataset", lambda *a, **kw: dataset
    )
    monkeypatch.setattr(pipeline, "split_nodes_by_ratio", lambda **kw: ([1], [0], []))
    config.training.crossval_enabled = crossval
    result = pipeline.load_fit_cases(config, manifest)
    assert result.observed.tolist() == ([4.0] if crossval else [1.0, 4.0])
    assert result.source_id.tolist() == (["b"] if crossval else ["a", "b"])
    assert result.target_date.max() == "2020-09-30"
    dataset.attrs["population_norm"] = False
    with pytest.raises(ValueError, match="per-100k"):
        pipeline.load_fit_cases(config, manifest)


@pytest.mark.parametrize("fail", [False, True])
def test_research_evaluation_publishes_complete_attempt_and_restores(
    monkeypatch, tmp_path, config, fail
):
    workflow = yaml.safe_load(
        Path("workflow/config/mn5_loss_score_acceptance.yaml").read_text()
    )
    workflow["evaluation"].update(
        split="val", granular=True, research_outputs=True, overrides=[]
    )
    root = tmp_path / "run"
    root.mkdir()
    workflow_path = root / "config.yaml"
    workflow_path.write_text(yaml.safe_dump(workflow))
    checkpoint = root / "best_model.pt"
    torch.save({"config": config.to_dict()}, checkpoint)
    calls = []

    def evaluate(checkpoint_path, output, split, overrides, granular):
        calls.append(split)
        output.mkdir(parents=True)
        for suffix in [
            "summary.json",
            "node_metrics.csv",
            "node_metrics_per_head.csv",
            "granular.parquet",
        ]:
            (output / f"{split}_{suffix}").write_text(split)
        result = output / "evaluation_manifest.json"
        result.write_text("{}")
        (output / ".done").write_text("{}")
        return result

    def outputs(**kwargs):
        if fail:
            raise ValueError("invalid role data")
        for name in pipeline.RESEARCH_ARTIFACTS:
            (kwargs["output_dir"] / name).write_text("{}")

    monkeypatch.setattr(run_support, "evaluate_best", evaluate)
    monkeypatch.setattr(pipeline, "write_research_outputs", outputs)
    if fail:
        with pytest.raises(ValueError, match="invalid role"):
            run_support.evaluate_configured_model(workflow_path, root, checkpoint)
        assert not (root / "outputs/evaluation/.done").exists()
        assert not list((root / ".state/attempts").rglob("complete.json"))
        return
    result = run_support.evaluate_configured_model(workflow_path, root, checkpoint)
    record = json.loads(result.read_text())
    assert calls == ["val", "test"]
    assert record["data_roles"] == {"val": "selection", "test": "calibration"}
    assert record["checkpoint_sha256"] == run_support.sha256_file(checkpoint)
    assert set(run_config.RESEARCH_OUTPUT_FILES) <= {
        item["path"] for item in record["artifacts"]
    }
    for item in record["artifacts"]:
        assert run_support.sha256_file(result.parent / item["path"]) == item["sha256"]
    (result.parent / "research_summary.json").unlink()
    assert (
        run_support.evaluate_configured_model(workflow_path, root, checkpoint) == result
    )
    assert calls == ["val", "test"]
    assert (result.parent / "research_summary.json").exists()


@pytest.mark.parametrize(
    "change", [dict(research_outputs="yes"), dict(split="test"), dict(granular=False)]
)
def test_research_workflow_rejects_incomplete_contract(change):
    workflow = yaml.safe_load(
        Path("workflow/config/mn5_loss_score_acceptance.yaml").read_text()
    )
    workflow["evaluation"].update(split="val", granular=True, research_outputs=True)
    workflow["evaluation"].update(change)
    with pytest.raises(ValueError):
        run_config.validate_workflow_contract(workflow)
