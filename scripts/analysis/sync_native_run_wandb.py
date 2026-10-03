"""Import a completed source-native MN5 run into W&B after training.

This is a post-hoc import. It never changes the immutable MN5 run or claims
that W&B recorded live training telemetry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

EPOCH_RE = re.compile(
    r"native epoch (?P<epoch>\d+)/(?:\d+) "
    r"mean_train_loss=(?P<train>[0-9.eE+-]+) "
    r"val_loss=(?P<val>[0-9.eE+-]+) "
    r"selection=(?P<metric>[a-z_]+)=(?P<score>[0-9.eE+-]+)"
    r"(?: val_joint_loss=(?P<joint>[0-9.eE+-]+))?"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_evidence(evidence: Path) -> dict:
    model = evidence / "model"
    manifest = _read_json(model / "training_manifest.json")
    dynamics_manifest = _read_json(model / "training_dynamics_manifest.json")
    provenance = _read_json(model / "training_provenance.json")
    dynamics_path = model / "training_dynamics.jsonl"
    expected = manifest["training_dynamics"]
    if expected["status"] != "complete":
        raise ValueError("Training dynamics are not complete")
    for path, sha in (
        (dynamics_path, expected["artifact_sha256"]),
        (model / "training_dynamics_manifest.json", expected["manifest_sha256"]),
    ):
        if _sha256(path) != sha:
            raise ValueError(f"SHA-256 mismatch: {path}")

    dynamics = [json.loads(line) for line in dynamics_path.read_text().splitlines()]
    if len(dynamics) != expected["row_count"]:
        raise ValueError("Training dynamics row count mismatch")
    if not all(row["schema_version"] == 1 for row in dynamics):
        raise ValueError("Unsupported training dynamics schema")
    steps = [row["step"] for row in dynamics]
    if steps != sorted(set(steps)):
        raise ValueError("Training steps are not strictly increasing")

    summary = dynamics_manifest["summary"]
    if "val_wis_trace" in summary:
        epochs = summary["train_epoch_trace"]
        if [row["epoch"] for row in epochs] != list(
            range(1, summary["total_epochs"] + 1)
        ):
            raise ValueError("Epoch summary sequence mismatch")
        steps_by_epoch = {row["epoch"]: row["step"] for row in epochs}
        if not epochs or epochs[-1]["step"] != summary["global_step"]:
            raise ValueError("Epoch summary step count mismatch")
        validation = summary["validation_epoch_trace"]
        if [row["epoch"] for row in validation] != list(steps_by_epoch):
            raise ValueError("Validation epoch trace is incomplete")
        if [row["val_wis"] for row in validation] != summary["val_wis_trace"]:
            raise ValueError("Validation WIS trace mismatch")
        for split, trace in (
            ("val", validation),
            ("test", summary["test_epoch_trace"]),
        ):
            if split == "test" and len(trace) > 1:
                raise ValueError("Test must evaluate only the selected checkpoint")
            for row in trace:
                if split == "test" and "selection_score" in row:
                    raise ValueError("Test trace must not contain selection scores")
                if row["step"] != steps_by_epoch.get(row["epoch"]):
                    raise ValueError(f"Invalid {split} trace step")
                if split == "val":
                    definition = row["selection_metric_definition"]
                    for key in ("metric", "aggregation", "direction"):
                        if (
                            definition[key]
                            != summary["selection_metric_definition"][key]
                        ):
                            raise ValueError("Selection metric definition mismatch")
                if split == "test" and (
                    row["epoch"] != summary["selected_epoch"]
                    or row["step"] != summary["selected_step"]
                ):
                    raise ValueError("Test must evaluate only the selected checkpoint")
        selected = validation[summary["selected_epoch"] - 1]
        if selected["selection_score"] != summary["selection_score"]:
            raise ValueError("Selected validation score mismatch")
    else:
        log_text = (evidence / "logs" / "train_model.log").read_text()
        epochs = [match.groupdict() for match in EPOCH_RE.finditer(log_text)]
        if len(epochs) != summary["total_epochs"]:
            raise ValueError("Epoch summary count mismatch")
        if [int(row["epoch"]) for row in epochs] != list(
            range(1, summary["total_epochs"] + 1)
        ):
            raise ValueError("Epoch summary sequence mismatch")
        for index, row in enumerate(epochs):
            if abs(float(row["val"]) - summary["val_loss_trace"][index]) > 1e-5:
                raise ValueError(
                    "Log and dynamics manifest disagree on validation loss"
                )
            if summary.get("val_loss_metric") == "mean_target_wis":
                if row["metric"] != "mean_target_wis" or row["joint"] is None:
                    raise ValueError(
                        "WIS validation requires a separate joint-loss record"
                    )
                if abs(float(row["score"]) - float(row["val"])) > 1e-5:
                    raise ValueError(
                        "Primary validation loss disagrees with WIS selection"
                    )
                if (
                    abs(float(row["joint"]) - summary["val_joint_loss_trace"][index])
                    > 1e-5
                ):
                    raise ValueError("Log and dynamics manifest disagree on joint loss")
            else:
                # Immutable historical runs used val_loss for the joint objective.
                row["joint"] = row["val"]
        steps_per_epoch = summary["global_step"] // summary["total_epochs"]
        if steps_per_epoch * summary["total_epochs"] != summary["global_step"]:
            raise ValueError("Cannot map epoch summaries to optimizer steps")
        for split, key in (
            ("val", "validation_epoch_trace"),
            ("test", "test_epoch_trace"),
        ):
            trace = summary.get(key, [])
            numbered_epochs = [row["epoch"] for row in trace]
            if numbered_epochs != sorted(set(numbered_epochs)):
                raise ValueError(f"{split} epoch trace is not strictly increasing")
            if (
                split == "val"
                and key in summary
                and numbered_epochs != list(range(1, summary["total_epochs"] + 1))
            ):
                raise ValueError("Validation epoch trace is incomplete")
            for row in trace:
                epoch = row["epoch"]
                if not 1 <= epoch <= summary["total_epochs"]:
                    raise ValueError(f"Invalid {split} trace epoch: {epoch}")
                if row["step"] != epoch * steps_per_epoch:
                    raise ValueError(f"Invalid {split} trace step at epoch {epoch}")
                if (
                    split == "val"
                    and abs(row["joint_loss"] - float(epochs[epoch - 1]["joint"]))
                    > 1e-5
                ):
                    raise ValueError(
                        "Validation epoch trace disagrees with trainer log"
                    )
                if split == "val" and (
                    row["selection_metric"] != epochs[epoch - 1]["metric"]
                    or abs(row["selection_score"] - float(epochs[epoch - 1]["score"]))
                    > 1e-5
                ):
                    raise ValueError(
                        "Validation selection trace disagrees with trainer log"
                    )

    metrics = {}
    for split in ("val", "test"):
        path = model / f"{split}_native_aggregate_metrics.csv"
        with path.open(newline="") as stream:
            metrics[split] = list(csv.DictReader(stream))
        if not any(row["aggregation"] == "head_all" for row in metrics[split]):
            raise ValueError(f"No head_all rows in {path}")

    return {
        "run_id": provenance["run"]["run_id"],
        "provenance": provenance,
        "manifest": manifest,
        "dynamics_manifest": dynamics_manifest,
        "dynamics": dynamics,
        "epochs": epochs,
        "metrics": metrics,
    }


def build_history(data: dict) -> dict[int, dict[str, float | int]]:
    """Map canonical sampled and epoch metrics to one W&B step series."""
    summary = data["dynamics_manifest"]["summary"]
    native = "val_wis_trace" in summary
    steps_per_epoch = summary["global_step"] // summary["total_epochs"]
    history: dict[int, dict[str, float | int]] = {}
    for row in data["dynamics"]:
        history[row["step"]] = {
            "epoch": row["epoch"] + 1,
            "train/train_objective": row["train_objective"]
            if native
            else row["train_loss"],
            "train/grad_norm": row["grad_norm"],
        }
    for row in data["epochs"]:
        step = row["step"] if native else int(row["epoch"]) * steps_per_epoch
        values = history.setdefault(step, {})
        values["epoch"] = int(row["epoch"])
        values["train/epoch_train_objective"] = (
            row["train_objective"] if native else float(row["train"])
        )
        if not native:
            values["val/training_objective_diagnostic"] = float(row["joint"])
            values["val/selection_score"] = float(row["score"])
            if row["metric"] == "mean_target_wis":
                values["val/val_wis"] = float(row["score"])
    for split, key in (("val", "validation_epoch_trace"), ("test", "test_epoch_trace")):
        for row in summary.get(key, []):
            values = history.setdefault(row["step"], {})
            values["epoch"] = row["epoch"]
            for metric, value in row.items():
                if metric in {"epoch", "step", "selection_metric"}:
                    continue
                if isinstance(value, int | float) and math.isfinite(value):
                    name = (
                        "training_objective_diagnostic"
                        if metric == "joint_loss"
                        else metric
                    )
                    values[f"{split}/{name}"] = value
    return history


def upload(evidence: Path, data: dict, entity: str, project: str) -> str:
    import wandb

    run_id = data["run_id"]
    wandb_id = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    api = wandb.Api(timeout=20)
    try:
        existing = api.run(f"{entity}/{project}/{wandb_id}")
    except wandb.errors.CommError as error:
        if "not found" not in str(error).lower():
            raise
    else:
        raise ValueError(f"W&B run already exists: {existing.url}")

    provenance = data["provenance"]["run"]
    summary = data["dynamics_manifest"]["summary"]
    epochs = data["epochs"]
    if "val_wis_trace" in summary:
        previous_steps = [0] + [row["step"] for row in epochs[:-1]]
        steps_per_epoch = [
            row["step"] - previous
            for row, previous in zip(epochs, previous_steps, strict=True)
        ]
    else:
        steps_per_epoch = summary["global_step"] // summary["total_epochs"]
        if steps_per_epoch * summary["total_epochs"] != summary["global_step"]:
            raise ValueError("Cannot map historical epoch summaries to optimizer steps")
    history = build_history(data)

    run = wandb.init(
        entity=entity,
        project=project,
        mode="online",
        dir=str(evidence.parent),
        id=wandb_id,
        name=run_id,
        job_type="posthoc-native-import",
        tags=["source-native", "wastewater", "mn5", "posthoc-import"],
        config={
            "source_run_id": run_id,
            "source_git_sha": provenance["git_sha"],
            "source_snapshot_sha256": provenance["source_snapshot_sha256"],
            "source_container_digest": provenance["container_digest"],
            "source_manifest_sha256": provenance["manifest_sha256"],
            "training_overrides": data["manifest"]["overrides"],
            "epochs": summary["total_epochs"],
            "optimizer_steps": summary["global_step"],
            "steps_per_epoch": steps_per_epoch,
            "test_checkpoint_epochs": [
                row["epoch"] for row in summary.get("test_epoch_trace", [])
            ],
            "selection_metric_definition": summary.get("selection_metric_definition"),
            "telemetry_origin": "posthoc canonical artifacts",
        },
        notes=(
            "Post-hoc import of a completed immutable MN5 run. Sampled objective "
            "and gradient values come from canonical dynamics. New epoch scores "
            "come from the canonical manifest; historical runs use their recorded "
            "trainer log. Test never selects a checkpoint. No phase timing is inferred."
        ),
    )
    try:
        for step, values in sorted(history.items()):
            run.log(values, step=step)
        metric = summary["best_val_selection_metric"]
        run.summary[f"val/best_{metric}"] = summary["best_val_selection_score"]
        run.summary["val/selection_score"] = summary["best_val_selection_score"]
        if "val_wis_trace" in summary:
            run.summary["val/val_wis"] = summary["val_wis"]
            run.summary["val/best_epoch"] = summary["selected_epoch"]
            run.summary["selection_metric_definition"] = summary[
                "selection_metric_definition"
            ]
        else:
            run.summary["val/best_epoch"] = int(
                min(epochs, key=lambda row: float(row["score"]))["epoch"]
            )
        run.summary["source/run_id"] = run_id
        run.summary["source/workflow_status"] = "complete"
        tables = {}
        for split, rows in data["metrics"].items():
            for row in rows:
                if row["aggregation"] != "head_all":
                    continue
                target = row["target"]
                for metric in ("crps", "wis", "coverage_80", "observed_count"):
                    run.summary[f"{split}/{target}/{metric}"] = float(row[metric])
            horizons = [row for row in rows if row["aggregation"] == "head_horizon"]
            if horizons:
                columns = [
                    "target",
                    "horizon",
                    "observed_count",
                    "crps",
                    "wis",
                    "coverage_80",
                    "interval_width_80",
                ]
                table = wandb.Table(
                    columns=columns,
                    data=[
                        [
                            row["target"],
                            int(row["horizon"]),
                            int(row["observed_count"]),
                            float(row["crps"]),
                            float(row["wis"]),
                            float(row["coverage_80"]),
                            float(row["interval_width_80"]),
                        ]
                        for row in horizons
                    ],
                )
                tables[f"{split}/horizon_metrics"] = table
        if tables:
            run.log(tables, step=summary["global_step"] + 1)
        for path in (
            evidence / "config.yaml",
            evidence / "manifest.yaml",
            evidence / "logs" / "train_model.log",
            evidence / "model" / "training_manifest.json",
            evidence / "model" / "training_provenance.json",
            evidence / "model" / "training_dynamics.jsonl",
            evidence / "model" / "training_dynamics_manifest.json",
            evidence / "model" / "val_native_aggregate_metrics.csv",
            evidence / "model" / "test_native_aggregate_metrics.csv",
        ):
            run.save(str(path), base_path=str(evidence), policy="end")
        url = run.url
    finally:
        run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    data = read_evidence(args.evidence)
    print(
        f"Validated {data['run_id']}: {len(data['dynamics'])} dynamics rows, "
        f"{len(data['epochs'])} epochs, "
        f"{len(data['metrics']['test'])} test metric rows"
    )
    if args.upload:
        print(upload(args.evidence, data, args.entity, args.project))


if __name__ == "__main__":
    main()
