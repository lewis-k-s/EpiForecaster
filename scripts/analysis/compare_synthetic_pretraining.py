#!/usr/bin/env python
"""Build provenance-checked tables for the synthetic-pretraining experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TARGET_HEADS = {
    "ww": "wastewater",
    "hosp": "hospitalizations",
    "cases": "cases",
    "deaths": "deaths",
}
LATENT_PATTERNS = (
    re.compile(r"^(count)_(s|i|h|r|d)$"),
    re.compile(r"^(mae|rmse)_(s|i|h|r|d)_(fraction)$"),
    re.compile(r"^(mae_over_target_std|pearson)_(s|i|h|r|d)$"),
    re.compile(r"^(nonnegative_violation_fraction)_(s|i|h|r|d)$"),
)


@dataclass(frozen=True)
class RunSpec:
    stage: str
    dataset_role: str
    evaluation_summary: Path
    training_dynamics: Path
    training_dynamics_manifest: Path
    training_provenance: Path
    evaluation_manifest: Path


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    baseline: str
    candidate: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return document


def _resolve_input_path(base_dir: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {resolved}")
    return resolved


def _parse_spec(spec_path: Path) -> tuple[list[RunSpec], list[ComparisonSpec]]:
    document = _load_mapping(spec_path)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported synthetic-pretraining comparison schema")

    base_dir = spec_path.parent.resolve()
    raw_runs = document.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Comparison spec requires at least one run")

    runs: list[RunSpec] = []
    seen_stages: set[str] = set()
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        stage = str(raw_run.get("stage", "")).strip()
        dataset_role = str(raw_run.get("dataset_role", "")).strip()
        if not stage or not dataset_role:
            raise ValueError(f"runs[{index}] requires stage and dataset_role")
        if stage in seen_stages:
            raise ValueError(f"Duplicate stage in comparison spec: {stage}")
        seen_stages.add(stage)
        runs.append(
            RunSpec(
                stage=stage,
                dataset_role=dataset_role,
                evaluation_summary=_resolve_input_path(
                    base_dir,
                    raw_run.get("evaluation_summary"),
                    field=f"runs[{index}].evaluation_summary",
                ),
                training_dynamics=_resolve_input_path(
                    base_dir,
                    raw_run.get("training_dynamics"),
                    field=f"runs[{index}].training_dynamics",
                ),
                training_dynamics_manifest=_resolve_input_path(
                    base_dir,
                    raw_run.get("training_dynamics_manifest"),
                    field=f"runs[{index}].training_dynamics_manifest",
                ),
                training_provenance=_resolve_input_path(
                    base_dir,
                    raw_run.get("training_provenance"),
                    field=f"runs[{index}].training_provenance",
                ),
                evaluation_manifest=_resolve_input_path(
                    base_dir,
                    raw_run.get("evaluation_manifest"),
                    field=f"runs[{index}].evaluation_manifest",
                ),
            )
        )

    raw_comparisons = document.get("comparisons", [])
    if not isinstance(raw_comparisons, list):
        raise ValueError("comparisons must be a list")
    comparisons: list[ComparisonSpec] = []
    seen_names: set[str] = set()
    for index, raw_comparison in enumerate(raw_comparisons):
        if not isinstance(raw_comparison, dict):
            raise ValueError(f"comparisons[{index}] must be a mapping")
        comparison = ComparisonSpec(
            name=str(raw_comparison.get("name", "")).strip(),
            baseline=str(raw_comparison.get("baseline", "")).strip(),
            candidate=str(raw_comparison.get("candidate", "")).strip(),
        )
        if not comparison.name or not comparison.baseline or not comparison.candidate:
            raise ValueError(
                f"comparisons[{index}] requires name, baseline, and candidate"
            )
        if comparison.name in seen_names:
            raise ValueError(f"Duplicate comparison name: {comparison.name}")
        if comparison.baseline not in seen_stages:
            raise ValueError(
                f"Unknown comparison baseline: {comparison.baseline}"
            )
        if comparison.candidate not in seen_stages:
            raise ValueError(
                f"Unknown comparison candidate: {comparison.candidate}"
            )
        seen_names.add(comparison.name)
        comparisons.append(comparison)
    return runs, comparisons


def _numeric_scalar(value: Any, *, metric: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Metric {metric} is non-finite")
    return numeric


def _split_target_metric(metric: str) -> tuple[str, str] | None:
    for head, target in TARGET_HEADS.items():
        infix = f"_{head}_"
        if infix in metric:
            prefix, suffix = metric.split(infix, maxsplit=1)
            return target, f"{prefix}_{suffix}"
        ending = f"_{head}"
        if metric.endswith(ending):
            return target, metric[: -len(ending)]
    return None


def _split_latent_metric(metric: str) -> tuple[str, str] | None:
    body = metric.removeprefix("latent_")
    for pattern in LATENT_PATTERNS:
        match = pattern.fullmatch(body)
        if match is None:
            continue
        parts = list(match.groups())
        latent = parts.pop(1).upper()
        return latent, "_".join(parts)
    loss_match = re.fullmatch(r"loss_latent_(s|i|h|r|d)", metric)
    if loss_match is not None:
        return loss_match.group(1).upper(), "loss"
    return None


def _validate_run_artifacts(
    run: RunSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = _load_mapping(run.evaluation_summary)
    provenance = _load_mapping(run.training_provenance)
    evaluation_manifest = _load_mapping(run.evaluation_manifest)

    if summary.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation summary: {run.evaluation_summary}")
    if not isinstance(summary.get("metrics"), dict):
        raise ValueError(f"Evaluation summary has no metrics: {run.evaluation_summary}")
    if provenance.get("schema_version") != 1 or provenance.get("status") != "resolved":
        raise ValueError(
            f"Training provenance is not resolved: {run.training_provenance}"
        )
    if provenance.get("dvc", {}).get("reconciliation") != "matched":
        raise ValueError(
            f"DVC provenance is not reconciled: {run.training_provenance}"
        )
    if evaluation_manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported evaluation manifest: {run.evaluation_manifest}"
        )
    if summary.get("checkpoint") != evaluation_manifest.get("checkpoint"):
        raise ValueError(
            f"Checkpoint path mismatch for stage {run.stage}: "
            f"summary={summary.get('checkpoint')!r}, "
            f"manifest={evaluation_manifest.get('checkpoint')!r}"
        )

    dynamics_sha256 = _sha256_file(run.training_dynamics)
    dynamics_manifest = _load_mapping(run.training_dynamics_manifest)
    if dynamics_manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported dynamics manifest for stage {run.stage}"
        )
    if dynamics_manifest.get("artifact") != run.training_dynamics.name:
        raise ValueError(
            f"Dynamics manifest names the wrong artifact for stage {run.stage}"
        )
    if dynamics_manifest.get("artifact_sha256") != dynamics_sha256:
        raise ValueError(
            f"Dynamics manifest hash mismatch for stage {run.stage}"
        )
    observed_row_count = sum(
        1
        for line in run.training_dynamics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if dynamics_manifest.get("row_count") != observed_row_count:
        raise ValueError(
            f"Dynamics manifest row count mismatch for stage {run.stage}"
        )
    recorded_evidence = provenance.get("training_evidence", {}).get("artifacts", [])
    recorded_dynamics = next(
        (
            artifact
            for artifact in recorded_evidence
            if isinstance(artifact, dict)
            and artifact.get("name") == run.training_dynamics.name
        ),
        None,
    )
    if recorded_dynamics is None:
        raise ValueError(
            f"Training dynamics are absent from provenance for stage {run.stage}"
        )
    if recorded_dynamics.get("sha256") != dynamics_sha256:
        raise ValueError(
            f"Training dynamics hash mismatch for stage {run.stage}"
        )
    dynamics_manifest_sha256 = _sha256_file(run.training_dynamics_manifest)
    recorded_dynamics_manifest = next(
        (
            artifact
            for artifact in recorded_evidence
            if isinstance(artifact, dict)
            and artifact.get("name") == run.training_dynamics_manifest.name
        ),
        None,
    )
    if recorded_dynamics_manifest is None:
        raise ValueError(
            f"Dynamics manifest is absent from provenance for stage {run.stage}"
        )
    if recorded_dynamics_manifest.get("sha256") != dynamics_manifest_sha256:
        raise ValueError(
            f"Dynamics manifest provenance hash mismatch for stage {run.stage}"
        )

    summary_sha256 = _sha256_file(run.evaluation_summary)
    recorded_summary = next(
        (
            artifact
            for artifact in evaluation_manifest.get("artifacts", [])
            if isinstance(artifact, dict)
            and artifact.get("path") == run.evaluation_summary.name
        ),
        None,
    )
    if recorded_summary is None:
        raise ValueError(
            f"Evaluation summary is absent from manifest for stage {run.stage}"
        )
    if recorded_summary.get("sha256") != summary_sha256:
        raise ValueError(
            f"Evaluation summary hash mismatch for stage {run.stage}"
        )
    checkpoint_sha256 = evaluation_manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ) is None:
        raise ValueError(f"Invalid checkpoint hash for stage {run.stage}")
    return summary, provenance, evaluation_manifest


def _extract_evaluation_rows(
    run: RunSpec,
    summary: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    evaluation_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    split = str(summary["split"])

    for metric, raw_value in sorted(summary["metrics"].items()):
        value = _numeric_scalar(raw_value, metric=metric)
        if value is None:
            continue
        common = {
            "stage": run.stage,
            "dataset_role": run.dataset_role,
            "split": split,
            "value": value,
        }
        if metric.startswith("pathway_"):
            pathway_body = metric.removeprefix("pathway_")
            head, separator, normalized_metric = pathway_body.partition("_")
            if not separator or head not in TARGET_HEADS:
                structural_rows.append(
                    {**common, "metric": metric}
                )
                continue
            pathway_rows.append(
                {
                    **common,
                    "target": TARGET_HEADS[head],
                    "metric": normalized_metric,
                }
            )
            continue

        latent_metric = _split_latent_metric(metric)
        if latent_metric is not None:
            latent, normalized_metric = latent_metric
            latent_rows.append(
                {
                    **common,
                    "latent": latent,
                    "metric": normalized_metric,
                }
            )
            continue
        if metric.startswith("latent_"):
            structural_rows.append({**common, "metric": metric})
            continue

        target_metric = _split_target_metric(metric)
        target = "all"
        normalized_metric = metric
        if target_metric is not None:
            target, normalized_metric = target_metric
        evaluation_rows.append(
            {
                **common,
                "target": target,
                "metric": normalized_metric,
            }
        )
    return evaluation_rows, pathway_rows, latent_rows, structural_rows


def _extract_pathway_gradient_rows(run: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        run.training_dynamics.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        document = json.loads(line)
        if not isinstance(document, dict):
            raise ValueError(
                f"Training dynamics line {line_number} is not a mapping: "
                f"{run.training_dynamics}"
            )
        step = document.get("step")
        epoch = document.get("epoch")
        for key, raw_value in sorted(document.items()):
            if not key.startswith("pathgrad_"):
                continue
            value = _numeric_scalar(raw_value, metric=key)
            if value is None:
                continue
            body = key.removeprefix("pathgrad_")
            target = "all"
            metric = body
            if body.startswith("sidecar_"):
                metric = body
            else:
                head, separator, remainder = body.partition("_")
                if separator and head in TARGET_HEADS:
                    target = TARGET_HEADS[head]
                    metric = remainder
            rows.append(
                {
                    "stage": run.stage,
                    "dataset_role": run.dataset_role,
                    "step": step,
                    "epoch": epoch,
                    "target": target,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_rows(
    comparisons: list[ComparisonSpec],
    runs: list[RunSpec],
    table_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    runs_by_stage = {run.stage: run for run in runs}
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        baseline_run = runs_by_stage[comparison.baseline]
        candidate_run = runs_by_stage[comparison.candidate]
        if baseline_run.dataset_role != candidate_run.dataset_role:
            raise ValueError(
                f"Comparison {comparison.name} crosses dataset roles: "
                f"{baseline_run.dataset_role} vs {candidate_run.dataset_role}"
            )
        for table_name, records in table_rows.items():
            baseline = {
                (row["split"], row.get("target", "all"), row["metric"]): row["value"]
                for row in records
                if row["stage"] == comparison.baseline
            }
            candidate = {
                (row["split"], row.get("target", "all"), row["metric"]): row["value"]
                for row in records
                if row["stage"] == comparison.candidate
            }
            for key in sorted(baseline.keys() & candidate.keys()):
                split, target, metric = key
                baseline_value = float(baseline[key])
                candidate_value = float(candidate[key])
                rows.append(
                    {
                        "comparison": comparison.name,
                        "table": table_name,
                        "dataset_role": baseline_run.dataset_role,
                        "split": split,
                        "target": target,
                        "metric": metric,
                        "baseline_stage": comparison.baseline,
                        "candidate_stage": comparison.candidate,
                        "baseline_value": baseline_value,
                        "candidate_value": candidate_value,
                        "candidate_minus_baseline": (
                            candidate_value - baseline_value
                        ),
                        "candidate_over_baseline": (
                            candidate_value / baseline_value
                            if baseline_value != 0.0
                            else None
                        ),
                    }
                )
    return rows


def generate_comparison(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate run evidence and write deterministic comparison tables."""
    resolved_spec = spec_path.resolve(strict=True)
    runs, comparisons = _parse_spec(resolved_spec)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for run in runs:
        summary, provenance, evaluation_manifest = _validate_run_artifacts(run)
        extracted = _extract_evaluation_rows(run, summary)
        evaluation_rows.extend(extracted[0])
        pathway_rows.extend(extracted[1])
        latent_rows.extend(extracted[2])
        structural_rows.extend(extracted[3])
        gradient_rows.extend(_extract_pathway_gradient_rows(run))

        run_identity = provenance["run"]
        dvc_targets = provenance["dvc"]["targets"]
        inventory_rows.append(
            {
                "stage": run.stage,
                "dataset_role": run.dataset_role,
                "run_id": run_identity.get("run_id"),
                "git_sha": run_identity.get("git_sha"),
                "source_snapshot_sha256": run_identity.get(
                    "source_snapshot_sha256"
                ),
                "checkpoint": summary.get("checkpoint"),
                "checkpoint_sha256": evaluation_manifest.get(
                    "checkpoint_sha256"
                ),
                "evaluation_summary": str(run.evaluation_summary),
                "evaluation_summary_sha256": _sha256_file(
                    run.evaluation_summary
                ),
                "training_dynamics": str(run.training_dynamics),
                "training_dynamics_sha256": _sha256_file(
                    run.training_dynamics
                ),
                "training_dynamics_manifest": str(
                    run.training_dynamics_manifest
                ),
                "training_dynamics_manifest_sha256": _sha256_file(
                    run.training_dynamics_manifest
                ),
                "training_provenance": str(run.training_provenance),
                "training_provenance_sha256": _sha256_file(
                    run.training_provenance
                ),
                "evaluation_manifest": str(run.evaluation_manifest),
                "evaluation_manifest_sha256": _sha256_file(
                    run.evaluation_manifest
                ),
                "dvc_targets_json": json.dumps(
                    dvc_targets, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    table_rows = {
        "evaluation": evaluation_rows,
        "pathway": pathway_rows,
    }
    comparison_rows = _comparison_rows(
        comparisons,
        runs,
        table_rows,
    )
    outputs = {
        "run_inventory.csv": (
            inventory_rows,
            [
                "stage",
                "dataset_role",
                "run_id",
                "git_sha",
                "source_snapshot_sha256",
                "checkpoint",
                "checkpoint_sha256",
                "evaluation_summary",
                "evaluation_summary_sha256",
                "training_dynamics",
                "training_dynamics_sha256",
                "training_dynamics_manifest",
                "training_dynamics_manifest_sha256",
                "training_provenance",
                "training_provenance_sha256",
                "evaluation_manifest",
                "evaluation_manifest_sha256",
                "dvc_targets_json",
            ],
        ),
        "evaluation_metrics.csv": (
            evaluation_rows,
            ["stage", "dataset_role", "split", "target", "metric", "value"],
        ),
        "pathway_metrics.csv": (
            pathway_rows,
            ["stage", "dataset_role", "split", "target", "metric", "value"],
        ),
        "latent_metrics.csv": (
            latent_rows,
            ["stage", "dataset_role", "split", "latent", "metric", "value"],
        ),
        "latent_structure_metrics.csv": (
            structural_rows,
            ["stage", "dataset_role", "split", "metric", "value"],
        ),
        "pathway_gradients.csv": (
            gradient_rows,
            [
                "stage",
                "dataset_role",
                "step",
                "epoch",
                "target",
                "metric",
                "value",
            ],
        ),
        "comparison_deltas.csv": (
            comparison_rows,
            [
                "comparison",
                "table",
                "dataset_role",
                "split",
                "target",
                "metric",
                "baseline_stage",
                "candidate_stage",
                "baseline_value",
                "candidate_value",
                "candidate_minus_baseline",
                "candidate_over_baseline",
            ],
        ),
    }
    for name, (rows, fieldnames) in outputs.items():
        _write_csv(output_dir / name, rows, fieldnames)

    artifact_rows = []
    for name in sorted(outputs):
        path = output_dir / name
        artifact_rows.append(
            {
                "path": name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "spec": str(resolved_spec),
        "spec_sha256": _sha256_file(resolved_spec),
        "stages": [run.stage for run in runs],
        "comparisons": [comparison.name for comparison in comparisons],
        "artifacts": artifact_rows,
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate_comparison(args.spec, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
