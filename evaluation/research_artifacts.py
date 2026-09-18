"""Read complete, checksum-verified development research evaluation outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.research_roles import load_role_manifest
from scripts.workflow.run_config import RESEARCH_OUTPUT_FILES


@dataclass(frozen=True)
class ResearchEvidence:
    evaluation_manifest_sha256: str
    checkpoint_sha256: str
    role_manifest_sha256: str
    scores: pd.DataFrame
    events: pd.DataFrame
    summary: dict
    event_summary: dict
    calibration: dict


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _regular_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Artifact must be evaluation-relative: {relative}")
    result = root / path
    result.resolve().relative_to(root.resolve())
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"Artifact must be a regular file: {relative}")
    return result


def read_research_evidence(
    directory: Path,
    *,
    expected_checkpoint_sha256: str,
    role_manifest_path: Path,
) -> ResearchEvidence:
    """Verify local/synced outputs against a separately obtained checkpoint ID.

    The expected ID must come from the training/selection receipt, or a direct
    checkpoint hash. A self-reported evaluation hash alone is not that anchor.
    This reader does not fit calibration, recompute scores, or open holdout data.
    """
    if len(expected_checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_checkpoint_sha256
    ):
        raise ValueError("A checkpoint SHA-256 from the training receipt is required")
    manifest_path = _regular_file(directory, "evaluation_manifest.json")
    marker = json.loads(_regular_file(directory, ".done").read_text())
    manifest = json.loads(manifest_path.read_text())
    roles, role_hash = load_role_manifest(role_manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("research_outputs") is not True
        or manifest.get("split") != "val"
        or manifest.get("granular") is not True
        or manifest.get("data_roles") != {"val": "selection", "test": "calibration"}
        or manifest.get("role_manifest_sha256") != role_hash
        or manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
        or marker.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("Evaluation identity or protected role contract mismatch")
    required = {
        "val_summary.json",
        "val_node_metrics.csv",
        "val_node_metrics_per_head.csv",
        "val_granular.parquet",
        *RESEARCH_OUTPUT_FILES,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Evaluation has no artifact inventory")
    seen = set()
    for item in artifacts:
        relative = item["path"]
        if relative in seen:
            raise ValueError(f"Duplicate evaluation artifact: {relative}")
        seen.add(relative)
        if relative in required and item.get("required") is not True:
            raise ValueError(f"Research artifact is not declared required: {relative}")
        path = _regular_file(directory, relative)
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"Evaluation artifact checksum mismatch: {relative}")
    if required - seen:
        raise ValueError(f"Missing research artifacts: {sorted(required - seen)}")
    calibration = json.loads((directory / "research_calibration.json").read_text())
    content = {
        key: value for key, value in calibration.items() if key != "artifact_sha256"
    }
    encoded = json.dumps(
        content, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    role_encoded = json.dumps(
        roles, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    if (
        calibration.get("schema_version") != 1
        or calibration.get("fit_role") != "calibration"
        or calibration.get("artifact_sha256") != hashlib.sha256(encoded).hexdigest()
        or calibration.get("checkpoint_sha256") != expected_checkpoint_sha256
        or calibration.get("role_manifest_sha256")
        != hashlib.sha256(role_encoded).hexdigest()
    ):
        raise ValueError("Calibration artifact identity mismatch")
    summary = json.loads((directory / "research_summary.json").read_text())
    event_summary = json.loads((directory / "research_events.json").read_text())
    for record in [summary, event_summary]:
        if (
            record.get("schema_version") != 1
            or record.get("role") != "selection"
            or record.get("calibration_artifact_sha256")
            != calibration["artifact_sha256"]
        ):
            raise ValueError("Research summary identity mismatch")
    scores = pd.read_parquet(directory / "research_scored.parquet")
    events = pd.read_parquet(directory / "research_events.parquet")
    for frame in [scores, events]:
        if (
            frame.empty
            or not frame.role.eq("selection").all()
            or set(frame.control) != {"C0", "C1", "C2"}
        ):
            raise ValueError("Research rows have invalid roles or controls")
    return ResearchEvidence(
        evaluation_manifest_sha256=_sha256(manifest_path),
        checkpoint_sha256=expected_checkpoint_sha256,
        role_manifest_sha256=role_hash,
        scores=scores,
        events=events,
        summary=summary,
        event_summary=event_summary,
        calibration=calibration,
    )
