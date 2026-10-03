"""Build a deterministic single-run report after training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from dataviz.run_report import RunFigures
from scripts.analysis.render_run_report import render_report, sha256
from scripts.analysis.query_run_provenance import provenance_table
from scripts.analysis.run_report_inputs import (
    epoch_tables,
    gpu_tables,
    inspect_run,
    read_json,
)


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


class RunReport:
    def __init__(self, output: Path, run_id: str):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        (output / "tables").mkdir(exist_ok=True)
        self.bundle = {
            "schema_version": 2,
            "report_spec": "report_spec.json",
            "report_inputs": "report_inputs.json",
            "tables": {},
            "figures": {},
        }
        self.spec = {
            "schema_version": 1,
            "title": f"{run_id} · single-run report",
            "split_strategy": "region",
            "sections": [],
            "links": [
                {"path": "bundle_manifest.json", "label": "Evidence bundle"},
                {"path": "report_inputs.json", "label": "Input identities"},
            ],
        }

    def table(
        self, key: str, frame: pd.DataFrame, title: str, *, collapsed: bool = False
    ) -> dict:
        frame.to_csv(self.output / "tables" / f"{key}.csv", index=False)
        self.bundle["tables"][key] = f"tables/{key}.csv"
        return {"kind": "table", "key": key, "title": title, "collapsed": collapsed}

    def section(self, key: str, title: str, items: list[dict]) -> None:
        self.spec["sections"].append({"id": key, "title": title, "items": items})

    @staticmethod
    def figures(assets: dict[str, str], keys: list[str]) -> list[dict]:
        return [
            {"kind": "figure", "key": key, "title": key.replace("_", " ")}
            for key in keys
            if key in assets
        ]

    def finish(self, inputs: list[Path], state: str) -> Path:
        retained = []
        for index, source in enumerate(sorted(set(inputs))):
            item = {
                "source": str(source),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
            if source.suffix in {".json", ".yaml"}:
                name = f"evidence/{index:03d}_{source.name}"
                (self.output / "evidence").mkdir(exist_ok=True)
                shutil.copyfile(source, self.output / name)
                item["copy"] = name
            retained.append(item)
        write_json(self.output / "report_inputs.json", retained)
        write_json(self.output / "report_spec.json", self.spec)
        write_json(self.output / "bundle_manifest.json", self.bundle)
        document, receipt = render_report(self.output / "bundle_manifest.json")
        (self.output / "report.html").write_text(document)
        write_json(self.output / "report.render.json", receipt)
        paths = {
            "report.html",
            "report.render.json",
            "report_spec.json",
            "bundle_manifest.json",
            "report_inputs.json",
            *self.bundle["tables"].values(),
            *self.bundle["figures"].values(),
            *(item["copy"] for item in retained if "copy" in item),
        }
        manifest = self.output / "report_manifest.json"
        write_json(
            manifest,
            {
                "schema_version": 1,
                "state": state,
                "evidence_sha256": receipt["evidence_sha256"],
                "artifacts": [
                    {"path": path, "sha256": sha256(self.output / path)}
                    for path in sorted(paths)
                ],
            },
        )
        return manifest


def build_run_report(
    run_root: Path,
    output: Path,
    *,
    model_root: Path | None = None,
    evaluation_dir: Path | None = None,
) -> Path:
    evidence = inspect_run(
        run_root, model_root=model_root, evaluation_dir=evaluation_dir
    )
    report = RunReport(output, evidence["facts"]["run_id"])
    report.section(
        "status",
        "Run status",
        [
            report.table(
                "run_status",
                pd.DataFrame(
                    list(evidence["facts"].items()), columns=["field", "value"]
                ),
                "Completion and epoch counts",
            )
        ],
    )
    if evidence["errors"]:
        report.section(
            "errors",
            "Unavailable evidence",
            [
                report.table(
                    "errors", pd.DataFrame(evidence["errors"]), "Recorded error codes"
                )
            ],
        )
        return report.finish(evidence["inputs"], "incomplete")

    summary = evidence["summary"]
    evaluation_dir = evidence["evaluation_dir"]
    context = read_json(evaluation_dir / "val_summary.json").get("report_context", {})
    config = evidence["model_config"]
    strategy = context.get(
        "split_strategy",
        "temporal"
        if config.get("training", {}).get("split_strategy") == "temporal"
        else "region",
    )
    report.spec["split_strategy"] = strategy
    frames = epoch_tables(summary)
    selected_epoch = context.get("selected_epoch")
    test_summary = read_json(evaluation_dir / "test_summary.json")
    coverage = [
        {"evidence": f"{split} epoch trace", "recorded_rows": len(frame)}
        for split, frame in frames.items()
    ]
    coverage.append({"evidence": "GPU phase join", "recorded_rows": "not available"})
    plots = RunFigures(output)
    plots.epochs(
        frames,
        selected_epoch,
        test_summary.get("test_wis", test_summary.get("joint_wis")),
    )
    plots.wastewater(frames, selected_epoch)
    losses = report.figures(plots.assets, ["loss_epochs"])
    for split, frame in frames.items():
        if not frame.empty:
            losses.append(
                report.table(
                    f"{split}_epochs", frame, f"{split} epoch values", collapsed=True
                )
            )

    contract = context.get("split_contract", {})
    report.section(
        "split",
        "Temporal contract"
        if strategy == "temporal"
        else "Region / native-source contract",
        [
            report.table(
                "split_contract",
                pd.DataFrame(
                    [
                        (key, json.dumps(value, sort_keys=True))
                        for key, value in contract.items()
                    ],
                    columns=["field", "value"],
                ),
                "Saved split contract",
                collapsed=True,
            ),
            report.table(
                "evaluation_context",
                pd.DataFrame(
                    [
                        (key, value)
                        for key, value in context.items()
                        if key not in {"config", "split_contract"}
                    ],
                    columns=["field", "value"],
                ),
                "Checkpoint and window metadata",
            ),
        ]
        if context
        else [],
    )
    scores, horizons, checkpoint_items, example_items = [], [], [], []
    definition = summary.get("selection_metric_definition")
    if definition:
        checkpoint_items.append(
            report.table(
                "selection_rule",
                pd.DataFrame(
                    [
                        (key, json.dumps(value, sort_keys=True))
                        for key, value in definition.items()
                    ],
                    columns=["field", "value"],
                ),
                "Validation selection rule",
            )
        )
    wastewater_scores = []
    for split in ("val", "test"):
        split_summary = read_json(evaluation_dir / f"{split}_summary.json")
        metrics = split_summary.get("metrics", {})
        if "censored_count_ww" in metrics:
            wastewater_scores.append(
                {
                    "split": split,
                    **{
                        key: metrics.get(key)
                        for key in (
                            "wis_ww",
                            "censored_nll_ww",
                            "censored_count_ww",
                            "observed_count_ww",
                        )
                    },
                }
            )
        scalar = {
            key: value
            for key, value in split_summary.items()
            if not isinstance(value, dict | list)
        }
        checkpoint_items.append(
            report.table(
                f"{split}_summary",
                pd.DataFrame(list(scalar.items()), columns=["field", "value"]),
                f"{split} checkpoint summary",
                collapsed=True,
            )
        )
        aggregate_path = evaluation_dir / f"{split}_native_aggregate_metrics.csv"
        if aggregate_path.is_file():
            aggregate = pd.read_csv(aggregate_path, dtype={"source_id": str})
            scores.append(
                aggregate[aggregate.aggregation == "head_all"].assign(split=split)
            )
            horizons.append(
                aggregate[aggregate.aggregation == "head_horizon"].assign(split=split)
            )
        else:
            path = evaluation_dir / f"{split}_node_metrics_per_head.csv"
            if path.is_file():
                checkpoint_items.append(
                    report.table(
                        f"{split}_head_metrics",
                        pd.read_csv(path),
                        f"{split} saved head metrics",
                        collapsed=True,
                    )
                )
        selection_path = evaluation_dir / f"{split}_example_selection.csv"
        coverage.append(
            {
                "evidence": f"{split} native example exports",
                "recorded_rows": "available"
                if selection_path.is_file()
                else "not recorded",
            }
        )
        if selection_path.is_file():
            choices = pd.read_csv(selection_path, dtype={"source_id": str})
            example_items.append(
                report.table(
                    f"{split}_example_selection",
                    choices,
                    f"{split} source and origin selections",
                    collapsed=True,
                )
            )
            example_items.append(
                report.table(
                    f"{split}_example_availability",
                    pd.read_csv(evaluation_dir / f"{split}_example_availability.csv"),
                    f"{split} support and cohort counts",
                    collapsed=True,
                )
            )
            if not choices.empty:
                plots.examples(
                    split,
                    pd.read_parquet(evaluation_dir / f"{split}_example_rows.parquet"),
                    choices,
                    pd.read_parquet(
                        evaluation_dir / f"{split}_example_history.parquet"
                    ),
                    pd.read_csv(
                        evaluation_dir / f"{split}_native_excess_deaths.csv",
                        dtype={"abs_id": str},
                    ),
                    context,
                )
    if scores:
        all_scores = pd.concat(scores, ignore_index=True)
        columns = [
            "split",
            "target",
            "observed_count",
            "crps",
            "wis",
            "coverage_80",
            "coverage_95",
            "raw_count_mae",
        ]
        checkpoint_items.insert(
            0,
            report.table(
                "checkpoint_heads",
                all_scores[columns],
                "Selected checkpoint · head scores",
            ),
        )
        plots.heads(frames, all_scores, selected_epoch)
        plots.horizons(pd.concat(horizons, ignore_index=True))
        losses = (
            report.figures(
                plots.assets,
                [
                    "loss_epochs",
                    "heads_crps_epochs",
                    "heads_wis_epochs",
                    "heads_nll_epochs",
                    "wastewater_objectives",
                ],
            )
            + losses[1:]
        )
        checkpoint_items += report.figures(plots.assets, ["horizon_scores"])
    report.section("loss", "Loss and observation heads", losses)
    if wastewater_scores:
        checkpoint_items.append(
            report.table(
                "wastewater_companion",
                pd.DataFrame(wastewater_scores),
                "Wastewater: exact-target WIS and censored NLL with counts",
            )
        )
    report.section("checkpoint", "Selected checkpoint", checkpoint_items)
    provenance = evidence["model_root"] / "training_provenance.json"
    if provenance.is_file():
        report.section(
            "provenance",
            "Source and input identities",
            [
                report.table(
                    "provenance",
                    pd.DataFrame(provenance_table(read_json(provenance))),
                    "Recorded provenance",
                    collapsed=True,
                )
            ],
        )
    if example_items:
        report.section(
            "examples",
            "Fixed validation cohort across periods"
            if strategy == "temporal"
            else "Examples within each split",
            example_items
            + report.figures(
                plots.assets,
                [key for key in plots.assets if key.startswith("examples_")],
            ),
        )
    dynamics_manifest = read_json(
        evidence["model_root"] / "training_dynamics_manifest.json"
    )
    samples = pd.read_json(
        evidence["model_root"] / dynamics_manifest["artifact"], lines=True
    )
    plots.dynamics(samples)
    report.section(
        "dynamics",
        "Sampled dynamics",
        report.figures(plots.assets, ["sampled_loss_gradients"]),
    )
    gpu_items, gpu_summary, gpu_bands = [], [], []
    for index, path in enumerate(
        sorted((run_root / "outputs/gpu_telemetry").glob("*/manifest.json"))
    ):
        manifest = read_json(path)
        scope = {
            key: manifest[key]
            for key in (
                "execution_attempt_id",
                "job_id",
                "step_id",
                "worker_id",
                "status",
                "samples",
                "errors",
            )
        }
        gpu_items.append(
            report.table(
                f"gpu_collection_{index}",
                pd.DataFrame(list(scope.items()), columns=["field", "value"]),
                "Full worker invocation · collection identity",
                collapsed=True,
            )
        )
        evidence["inputs"].append(path)
        csv_path = path.parent / manifest["artifact"]
        if manifest["status"] == "complete" and csv_path.is_file():
            if sha256(csv_path) != manifest["artifact_sha256"]:
                raise ValueError(f"GPU telemetry hash mismatch: {csv_path}")
            gpu = pd.read_csv(csv_path, na_values=["N/A"])
            tables = gpu_tables(
                gpu,
                {
                    key: scope[key]
                    for key in (
                        "execution_attempt_id",
                        "job_id",
                        "step_id",
                        "worker_id",
                    )
                },
            )
            gpu_summary.append(tables[0])
            gpu_bands.append(tables[1])
            plots.gpu(f"gpu_telemetry_{index}", gpu)
            gpu_items += report.figures(plots.assets, [f"gpu_telemetry_{index}"])
            evidence["inputs"].append(csv_path)
    if gpu_summary:
        gpu_items.insert(
            0,
            report.table(
                "gpu_summary",
                pd.concat(gpu_summary, ignore_index=True),
                "Device sample statistics",
            ),
        )
        gpu_items.insert(
            1,
            report.table(
                "gpu_activity_bands",
                pd.concat(gpu_bands, ignore_index=True),
                "GPU activity bands",
            ),
        )
    coverage.append({"evidence": "GPU collections", "recorded_rows": len(gpu_summary)})
    if gpu_items:
        report.section("gpu", "GPU telemetry", gpu_items)
    report.section(
        "coverage",
        "Evidence coverage",
        [report.table("coverage", pd.DataFrame(coverage), "Recorded and missing data")],
    )
    report.bundle["figures"] = plots.assets
    return report.finish(evidence["inputs"], "complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()
    manifest = build_run_report(
        args.run_root,
        args.output_dir,
        model_root=args.model_root,
        evaluation_dir=args.evaluation_dir,
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
