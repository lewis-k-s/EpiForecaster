"""Audit historical loss-screen checkpoints for full-coverage campaign reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.hpo.ablation_worker import (
    AblationCampaign,
    hydrated_training_config,
    load_campaign,
    _without_evaluation_controls,
)

PROTOCOL_PATH = Path("configs/research/loss_protocol_v1.yaml")
IDENTITY_PATTERN = re.compile(
    r"^(?P<family>normal|student_t)-(?P<objective>nll|crps)-m(?P<mobility>[01])"
    r"-f(?P<fold>\d+)-s(?P<seed>\d+)(?:-r\d+)?$"
)
# Dirty paths that cannot change training behavior.
CLEAN_DIRTY_PREFIXES = ("docs/", "tex/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_name(match: re.Match[str]) -> str:
    mobility = "on" if match["mobility"] == "1" else "off"
    return f"{match['family']}:{match['objective']}:mobility_{mobility}"


def _dirty_training_paths(status_lines: list[str]) -> list[str]:
    """Return porcelain-status paths outside the prefixes that cannot affect training."""
    offending: list[str] = []
    for line in status_lines:
        # Porcelain v1: two status columns, one space, then the path. A rename
        # lists "old -> new"; both sides must be harmless.
        for path in str(line)[3:].split(" -> "):
            if not path.strip('"').startswith(CLEAN_DIRTY_PREFIXES):
                offending.append(path)
    return offending


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
    source_bundle_sha256: str,
    container_digest: str,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Return immutable receipt records; never mutates historical artifacts.

    ``source_bundle_sha256`` and ``container_digest`` are the identities of the
    deployed run that will reuse the checkpoints, not of this checkout.
    """
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    expected_dataset = str(protocol["data"]["research_dataset_dvc_md5"])
    expected_region2vec = str(protocol["data"]["region2vec_dvc_md5"])
    allowed_code = set(campaign.allowed_reuse_code_identities) | {source_bundle_sha256}
    expected_container = container_digest.removeprefix("sha256:")
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
        expected_config = hydrated_training_config(
            campaign, condition=condition, fold=fold, seed=seed
        )
        dataset_target = f"{expected_config['data']['dataset_path']}.dvc"
        region2vec_target = f"{expected_config['data']['region2vec_path']}.dvc"
        if dvc_hashes.get(dataset_target) != expected_dataset:
            reasons.append("dataset_dvc_identity_mismatch")
        if dvc_hashes.get(region2vec_target) != expected_region2vec:
            reasons.append("region2vec_dvc_identity_mismatch")
        historical_bundle = identity.get("source_snapshot_sha256")
        if historical_bundle not in allowed_code:
            reasons.append("source_bundle_identity_mismatch")
        historical_container = str(identity.get("container_digest", "")).removeprefix(
            "sha256:"
        )
        if historical_container != expected_container:
            reasons.append("container_digest_mismatch")
        if _dirty_training_paths(receipt.get("source", {}).get("git_status", [])):
            reasons.append("historical_dirty_training_source_unverified")
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
                "container_digest": identity.get("container_digest"),
                "dataset_dvc_md5": dvc_hashes.get(dataset_target),
                "region2vec_dvc_md5": dvc_hashes.get(region2vec_target),
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
        "expected_source_bundle_sha256": source_bundle_sha256,
        "expected_container_digest": expected_container,
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
    parser.add_argument(
        "--source-bundle-sha256",
        required=True,
        help="source_snapshot_sha256 of the deployed run that will reuse the checkpoints",
    )
    parser.add_argument(
        "--container-digest",
        required=True,
        help="Container SIF sha256 of the deployed run that will reuse the checkpoints",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    receipt = audit_reuse(
        campaign=campaign,
        historical_root=args.historical_root,
        source_bundle_sha256=args.source_bundle_sha256,
        container_digest=args.container_digest,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{receipt['reusable_identities']} reusable / {receipt['historical_identities_found']} "
        f"historical identities; {receipt['expected_identities'] - receipt['reusable_identities']} new training identities"
    )
    if receipt["historical_identities_found"] == 0:
        print(
            f"error: no historical run under {args.historical_root} matches the campaign",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
