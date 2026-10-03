"""Create a focused W&B report for one post-hoc native wastewater run.

Run with ``uv run --with wandb-workspaces python -m`` so this optional report
dependency does not alter the training environment.
"""

from __future__ import annotations

import argparse


def build_report(entity: str, project: str, run_id: str):
    import wandb_workspaces.reports.v2 as wr

    runset = wr.Runset(
        entity=entity,
        project=project,
        name="Completed MN5 source-native run",
        filters=f"Config('source_run_id') = '{run_id}'",
    )

    def grid(*panels):
        return wr.PanelGrid(runsets=[runset], panels=list(panels))

    return wr.Report(
        entity=entity,
        project=project,
        title=f"Native validation WIS ({run_id})",
        description=(
            "Primary validation WIS and training diagnostics from one completed, "
            "immutable MN5 source-native wastewater run."
        ),
        width="fluid",
        blocks=[
            wr.H1("Native wastewater model: performance and training"),
            wr.P(
                "This W&B run is a post-hoc import of saved native training "
                "and evaluation evidence. Missing telemetry remains missing. "
                "Checkpoint metrics use the saved selected checkpoint. Read "
                "the imported manifests and provenance for the run identities, "
                "sampling frequencies, selection rule, and evidence coverage."
            ),
            wr.H2("Selection and held-out performance"),
            grid(
                wr.ScalarChart(
                    title="Selected validation WIS · equal-head mean",
                    metric="val/val_wis",
                ),
                wr.ScalarChart(title="Selected epoch", metric="val/best_epoch"),
                wr.ScalarChart(
                    title="Test wastewater WIS", metric="test/wastewater/wis"
                ),
                wr.ScalarChart(
                    title="Test wastewater 80% coverage",
                    metric="test/wastewater/coverage_80",
                ),
                wr.ScalarChart(title="Test cases WIS", metric="test/cases/wis"),
                wr.ScalarChart(
                    title="Test hospitalizations WIS",
                    metric="test/hospitalizations/wis",
                ),
                wr.ScalarChart(title="Test deaths WIS", metric="test/deaths/wis"),
            ),
            wr.P(
                "WIS values are target-specific and are not directly "
                "comparable across targets with different scales. Older "
                "CRPS-selected runs do not have WIS selection histories. The "
                "validation and test horizon-metrics tables show the forecast "
                "horizon breakdown; 80% coverage is compared with nominal 0.80."
            ),
            wr.H2("Epoch-level convergence"),
            grid(
                wr.LinePlot(
                    title="Validation selection score",
                    x="epoch",
                    y=["val/selection_score"],
                    title_x="Epoch",
                    title_y="Configured selection score",
                ),
                wr.LinePlot(
                    title="Training objective",
                    x="epoch",
                    y=["train/epoch_train_objective"],
                    title_x="Epoch",
                    title_y="train_objective",
                ),
            ),
            wr.H2("Sampled optimizer dynamics"),
            grid(
                wr.LinePlot(
                    title="Sampled training objective",
                    x="Step",
                    y=["train/train_objective"],
                    title_x="Optimizer step",
                    title_y="train_objective",
                ),
                wr.LinePlot(
                    title="Gradient norm (every 100 steps)",
                    x="Step",
                    y=["train/grad_norm"],
                    title_x="Optimizer step",
                    title_y="Pre-clip gradient norm",
                ),
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report = build_report(args.entity, args.project, args.run_id)
    if args.save:
        report.save()
        print(report.url)
    else:
        report._to_model()
        print(f"Validated report layout with {len(report.blocks)} blocks")


if __name__ == "__main__":
    main()
