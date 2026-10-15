"""Audit historical loss-screen checkpoints for full-coverage campaign reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from models.configs import EpiForecasterConfig
from scripts.hpo.ablation_worker import AblationCampaign, load_campaign

IDENTITY_PATTERN = re.compile(
    r"^(?P<family>normal|student_t)-(?P<objective>nll|crps)-m(?P<mobility>[01])"
    r"-f(?P<fold>\d+)-s(?P<seed>\d+)-r\d+$"
)
TRAINING_ONLY_EXCLUSIONS = {
    ("output",),
    ("training", "run_final_evaluation"),
    ("training", "plot_forecasts"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overrides(mapping: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[str]:
    values: list[str] = []
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            values.extend(_overrides(value, path))
        else:
            values.append(f"{'.'.join(path)}={json.dumps(value, separators=(',', ':'))}")
    return values


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _without_evaluation_controls(config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    for path in TRAINING_ONLY_EXCLUSIONS:
        cursor: Any = normalized
        for part in path[:-1]:
            if not isinstance(cursor, dict):
                break
            cursor = cursor.get(part)
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)
    return normalized


def _condition_name(match: re.Match[str]) -> str:
    mobility = "on" if match["mobility"] == "1" else "off"
    return f"{match['family']}:{match['objective']}:mobility_{mobility}"


def _campaign_config(
    campaign: AblationCampaign,
    *,
    condition: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    condition_overrides = campaign.conditions_by_name[condition].overrides
    merged = _deep_merge(campaign.fixed_overrides, condition_overrides)
    overrides = _overrides(merged)
    overrides.extend(
        [
            f"training.seed={seed}",
            f"training.crossval_fold_index={fold}",
            "output.write_granular_eval=true",
        ]
    )
    return EpiForecasterConfig.load(str(campaign.base_config), overrides=overrides).to_dict()


def _trained_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    configs = sorted(run_dir.glob("artifacts/training/*/*/config.yaml"))
    if len(configs) != 1:
        raise ValueError(f"{run_dir}: expected one hydrated training config, found {len(configs)}")
    parsed = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{configs[0]}: training config must be a mapping")
    return parsed, configs[0]


def audit_reuse(
    *,
    campaign: AblationCampaign,
    historical_root: Path,
    current_git_sha: str,
) -> dict[str, Any]:
    """Return immutable receipt records; never mutates historical artifacts."""
    expected_dataset = "b5b684d5b9f31369895589c413f39291"
    expected_region2vec = "e04ff23fe17dbbb245315798fd7cd887"
    records: list[dict[str, Any]] = []
    expected = {
        (condition.name, fold, seed)
        for condition in campaign.conditions
        for fold in campaign.fold_indices
        for seed in campaign.paired_model_seeds
    }
    for run_dir in sorted(path.parent for path in historical_root.glob("*/local_artifact_verification.json")):
        match = IDENTITY_PATTERN.fullmatch(run_dir.name)
        if match is None:
            continue
        condition = _condition_name(match)
        fold = int(match["fold"])
        seed = int(match["seed"])
        if (condition, fold, seed) not in expected:
            continue
        verification = json.loads(
            (run_dir / "local_artifact_verification.json").read_text(encoding="utf-8")
        )
        manifest = yaml.safe_load(
            (run_dir / "run_receipts/manifest.yaml").read_text(encoding="utf-8")
        )
        receipt = yaml.safe_load(
            (run_dir / "run_receipts/config.yaml").read_text(encoding="utf-8")
        )
        trained, trained_path = _trained_config(run_dir)
        checkpoint = next((run_dir / "artifacts/training").glob("*/*/checkpoints/best_model.pt"), None)
        reasons: list[str] = []
        if checkpoint is None or not checkpoint.is_file():
            reasons.append("checkpoint_not_materialized")
            checkpoint_sha256 = None
        else:
            checkpoint_sha256 = _sha256(checkpoint)
            if checkpoint_sha256 != verification.get("checkpoint_sha256"):
                reasons.append("checkpoint_hash_mismatch")
        if verification.get("status") != "passed" or not verification.get(
            "scientific_selection_eligible"
        ):
            reasons.append("historical_completion_receipt_not_scientific")
        identity = manifest.get("identity", {})
        dvc_hashes = identity.get("dvc_hashes", {})
        if dvc_hashes.get(
            "outputs/loss_research/loss-type-research-v1/real_loss_fit20200930_v1.nc.dvc"
        ) != expected_dataset:
            reasons.append("dataset_dvc_identity_mismatch")
        if dvc_hashes.get(
            "outputs/region_embeddings_no_mobility/region_embeddings.pt.dvc"
        ) != expected_region2vec:
            reasons.append("region2vec_dvc_identity_mismatch")
        if identity.get("git_sha") != current_git_sha:
            reasons.append("git_source_identity_mismatch")
        dirty_paths = receipt.get("source", {}).get("git_status", [])
        if any(not str(value).removeprefix(" M ").startswith(("docs/", "tex")) for value in dirty_paths):
            reasons.append("historical_dirty_training_source_unverified")
        expected_config = _campaign_config(
            campaign,
            condition=condition,
            fold=fold,
            seed=seed,
        )
        expected_identity = _without_evaluation_controls(expected_config)
        trained_identity = _without_evaluation_controls(trained)
        if expected_identity != trained_identity:
            reasons.append("hydrated_training_contract_mismatch")
        records.append(
            {
                "condition": condition,
                "fold": fold,
                "seed": seed,
                "status": "reusable" if not reasons else "rejected",
                "rejection_reasons": reasons,
                "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_sha256": checkpoint_sha256,
                "old_run_identity": manifest.get("run_id"),
                "old_run_path": str(run_dir),
                "source_bundle_sha256": identity.get("source_snapshot_sha256"),
                "git_sha": identity.get("git_sha"),
                "dataset_dvc_md5": dvc_hashes.get(
                    "outputs/loss_research/loss-type-research-v1/real_loss_fit20200930_v1.nc.dvc"
                ),
                "region2vec_dvc_md5": dvc_hashes.get(
                    "outputs/region_embeddings_no_mobility/region_embeddings.pt.dvc"
                ),
                "hydrated_training_config_sha256": hashlib.sha256(
                    json.dumps(expected_identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "historical_training_config": str(trained_path),
                "checkpoint_selection_metric": trained["training"]["selection_metric"],
                "completion_receipt": str(run_dir / "local_artifact_verification.json"),
                "completion_receipt_sha256": _sha256(
                    run_dir / "local_artifact_verification.json"
                ),
            }
        )
    observed = {(row["condition"], row["fold"], row["seed"]) for row in records}
    return {
        "schema_version": 1,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "current_git_sha": current_git_sha,
        "historical_root": str(historical_root),
        "expected_identities": len(expected),
        "historical_identities_found": len(records),
        "reusable_identities": sum(row["status"] == "reusable" for row in records),
        "missing_historical_identities": [
            {"condition": condition, "fold": fold, "seed": seed}
            for condition, fold, seed in sorted(expected - observed)
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/campaigns/loss_type_full_coverage_v2.yaml"),
    )
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=Path("outputs/loss_research/loss-type-research-v1/screen"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    current_git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = audit_reuse(
        campaign=campaign,
        historical_root=args.historical_root,
        current_git_sha=current_git_sha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{receipt['reusable_identities']} reusable / {receipt['historical_identities_found']} "
        f"historical identities; {receipt['expected_identities'] - receipt['reusable_identities']} new training identities"
    )


if __name__ == "__main__":
    main()
