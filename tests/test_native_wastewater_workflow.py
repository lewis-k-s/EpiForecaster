"""Immutable-workflow contract for the bounded native EDAR smoke."""

from pathlib import Path

from scripts.workflow.config_templates import load_config_template

from models.configs import EpiForecasterConfig
from scripts.workflow.run_config import (
    dvc_payload_paths,
    validate_workflow_contract,
    workflow_target,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "workflow/config/mn5_native_acceptance.yaml"


def test_native_wastewater_smoke_contract_and_inputs() -> None:
    workflow = load_config_template(WORKFLOW_PATH, variant="wastewater_smoke")

    assert validate_workflow_contract(workflow) == "train"
    assert workflow_target(workflow) == "train"
    payloads = dvc_payload_paths(REPO_ROOT, workflow)
    assert Path("data/processed/native_support/obs_wastewater.zarr.dvc") in payloads
    assert (
        Path("data/processed/native_support/obs_deaths_abs_excess.zarr.dvc") in payloads
    )
    assert Path("data/processed/native_support/native_manifest_v4.json.dvc") in payloads

    training = workflow["training"]
    config = EpiForecasterConfig.load(
        REPO_ROOT / training["base_config"],
        overrides=training["overrides"],
        strict=True,
    )
    assert config.native.enabled
    assert config.native.wastewater_enabled
    assert config.training.device == "cuda"
    assert config.training.epochs == 1
    assert training["max_batches"] == 96
    assert config.output.granular_eval_format == "parquet"
    assert workflow["resources"]["allocation"]["qos"] == "acc_debug"
