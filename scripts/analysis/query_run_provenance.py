"""Query portable run and DVC provenance for training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from training.training_dynamics import (
    DYNAMICS_ARTIFACT_NAME,
    DYNAMICS_MANIFEST_NAME,
)

PROVENANCE_SCHEMA_VERSION = 1
TRAINING_PROVENANCE_NAME = "training_provenance.json"
TRACE_GLOB = "*.pt.trace.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml", ".dvc"}:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return document


def _entity_directory(entity: Path) -> Path:
    resolved = entity.resolve(strict=True)
    return resolved if resolved.is_dir() else resolved.parent


def _find_run_manifest(entity_dir: Path) -> Path | None:
    for directory in (entity_dir, *entity_dir.parents):
        candidate = directory / "manifest.yaml"
        if candidate.is_file():
            return candidate
    return None


def _run_root(manifest_path: Path) -> Path:
    document = _load_mapping(manifest_path)
    configured = document.get("run", {}).get("root")
    if configured:
        configured_path = Path(str(configured))
        if configured_path.is_dir():
            return configured_path.resolve()
    return manifest_path.parent.resolve()


def _evidence_files(entity_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        entity_dir / DYNAMICS_ARTIFACT_NAME,
        entity_dir / DYNAMICS_MANIFEST_NAME,
        entity_dir / "training_manifest.json",
        *sorted(entity_dir.glob(TRACE_GLOB)),
    ]
    files = []
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        files.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return files


def _dvc_targets(
    run_manifest: dict[str, Any],
    data_manifest: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    recorded = run_manifest.get("dvc", {}).get("targets", {})
    if not isinstance(recorded, dict):
        raise ValueError("Run manifest dvc.targets must be a mapping")
    staged = (
        data_manifest.get("targets", {})
        if isinstance(data_manifest, dict)
        else {}
    )
    rows = []
    reconciliation = "matched"
    for pointer, identity in sorted(recorded.items()):
        if not isinstance(identity, dict):
            raise ValueError(f"Malformed DVC identity for {pointer}")
        staged_identity = staged.get(pointer)
        staged_md5 = (
            staged_identity.get("md5")
            if isinstance(staged_identity, dict)
            else None
        )
        role = str(identity.get("role", "unknown"))
        materialization = (
            "not_applicable"
            if role == "provenance"
            else "confirmed"
            if staged_md5 == identity.get("md5")
            else "not_recorded"
            if staged_md5 is None
            else "mismatch"
        )
        if materialization == "mismatch":
            reconciliation = "mismatch"
        elif materialization == "not_recorded" and reconciliation == "matched":
            reconciliation = "partial"
        rows.append(
            {
                "pointer": str(pointer),
                "output": identity.get("output"),
                "dvc_md5": identity.get("md5"),
                "bytes": identity.get("size"),
                "file_count": identity.get("nfiles"),
                "role": role,
                "materialization": materialization,
            }
        )
    return rows, reconciliation


def _dependency_contracts(run_root: Path) -> list[dict[str, Any]]:
    directory = run_root / ".state/dependencies"
    rows = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.json")):
        document = _load_mapping(path)
        rows.append(
            {
                "stage": document.get("stage", path.stem),
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "fingerprint": document.get("fingerprint"),
            }
        )
    return rows


def _retained_evidence(
    run_root: Path,
    evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    observed = {
        artifact["name"]: artifact["sha256"]
        for artifact in evidence
    }
    evidence_root = run_root / "evidence/training"
    if not evidence_root.is_dir():
        return None
    for path in sorted(evidence_root.glob("*/*/retention.json")):
        record = _load_mapping(path)
        recorded = record.get("artifacts", {})
        if not isinstance(recorded, dict):
            continue
        if all(
            isinstance(recorded.get(name), dict)
            and recorded[name].get("sha256") == digest
            for name, digest in observed.items()
        ):
            pointer = path.parent / str(record["dvc_pointer"])
            if not pointer.is_file():
                raise FileNotFoundError(
                    f"Retained evidence DVC pointer is missing: {pointer}"
                )
            return {
                "retention_class": "dvc_retained",
                "dvc_tracking": "recorded",
                "dvc_pointer": str(pointer.resolve()),
                "dvc_md5": record.get("dvc_md5"),
                "retention_record": str(path.resolve()),
            }
    return None


def collect_training_provenance(
    entity: Path,
    *,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Collect report-ready provenance without materializing DVC payloads."""
    entity_dir = _entity_directory(entity)
    portable = entity_dir / TRAINING_PROVENANCE_NAME
    portable_document = None
    if run_manifest_path is None and portable.is_file():
        document = _load_mapping(portable)
        if document.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise ValueError("Unsupported training provenance schema")
        portable_document = document

    manifest_path = (
        run_manifest_path.resolve(strict=True)
        if run_manifest_path is not None
        else _find_run_manifest(entity_dir)
    )
    evidence = _evidence_files(entity_dir)
    if manifest_path is None:
        if portable_document is not None:
            return portable_document
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "status": "partial",
            "run": None,
            "dvc": {
                "status": "not_available",
                "reconciliation": "not_available",
                "targets": [],
            },
            "dependencies": [],
            "training_evidence": {
                "retention_class": "run_scoped_hashed",
                "dvc_tracking": "not_recorded",
                "artifacts": evidence,
            },
            "report_artifacts": {"retention_class": "derived_replaceable"},
        }

    run_manifest = _load_mapping(manifest_path)
    run_root = _run_root(manifest_path)
    data_manifest_path = run_root / ".state/data_manifest.json"
    data_manifest = (
        _load_mapping(data_manifest_path)
        if data_manifest_path.is_file()
        else None
    )
    targets, reconciliation = _dvc_targets(run_manifest, data_manifest)
    retained = _retained_evidence(run_root, evidence)
    identity = run_manifest.get("identity", {})
    source = run_manifest.get("source", {})
    workflow = run_manifest.get("workflow", {})
    container = run_manifest.get("container", {})
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "resolved",
        "run": {
            "run_id": run_manifest.get("run_id"),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_snapshot_sha256": identity.get("source_snapshot_sha256"),
            "git_sha": identity.get("git_sha"),
            "git_dirty": source.get("dirty"),
            "snakemake_config_sha256": identity.get("snakemake_config_sha256"),
            "dispatch_request_sha256": identity.get("dispatch_request_sha256"),
            "container_digest": identity.get(
                "container_digest", container.get("digest")
            ),
            "workflow_tooling_key": identity.get("workflow_tooling_key"),
            "workflow_kind": identity.get("workflow_kind"),
            "workflow_target": identity.get("workflow_target"),
            "execution_adapter": workflow.get("execution_adapter"),
            "execution_attempt_id": workflow.get("execution_attempt_id"),
        },
        "dvc": {
            "status": "available",
            "reconciliation": reconciliation,
            "targets": targets,
        },
        "dependencies": _dependency_contracts(run_root),
        "training_evidence": {
            **(
                retained
                if retained is not None
                else {
                    "retention_class": "run_scoped_hashed",
                    "dvc_tracking": "not_recorded",
                }
            ),
            "artifacts": evidence,
        },
        "report_artifacts": {"retention_class": "derived_replaceable"},
    }


def provenance_table(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the reusable provenance contract for report tables."""
    rows = []
    run = document.get("run")
    if isinstance(run, dict):
        rows.extend(
            {
                "category": "run",
                "name": key,
                "identity": value,
                "role": None,
                "status": document["status"],
            }
            for key, value in run.items()
            if value is not None
        )
    rows.extend(
        [
            {
                "category": "training_evidence_bundle",
                "name": document["training_evidence"]["dvc_pointer"],
                "identity": document["training_evidence"].get("dvc_md5"),
                "role": document["training_evidence"]["retention_class"],
                "status": document["training_evidence"]["dvc_tracking"],
            }
        ]
        if document.get("training_evidence", {}).get("dvc_pointer")
        else []
    )
    rows.extend(
        {
            "category": "dvc_input",
            "name": target["pointer"],
            "identity": target["dvc_md5"],
            "role": target["role"],
            "status": target["materialization"],
        }
        for target in document.get("dvc", {}).get("targets", [])
    )
    rows.extend(
        {
            "category": "dependency",
            "name": dependency["stage"],
            "identity": dependency["fingerprint"],
            "role": "snakemake_stage",
            "status": "available",
        }
        for dependency in document.get("dependencies", [])
    )
    rows.extend(
        {
            "category": "training_evidence",
            "name": artifact["name"],
            "identity": artifact["sha256"],
            "role": document["training_evidence"]["retention_class"],
            "status": document["training_evidence"]["dvc_tracking"],
        }
        for artifact in document.get("training_evidence", {}).get("artifacts", [])
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    document = collect_training_provenance(
        args.entity,
        run_manifest_path=args.run_manifest,
    )
    rendered = json.dumps(
        document,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
