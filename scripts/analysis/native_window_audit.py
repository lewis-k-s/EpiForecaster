"""Audit window eligibility with the trainer's fixed source-set split contract."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from data.native_dataset import NativeDatasetConfig, NativeSupportDataset
from data.native_source_sets import load_source_set_manifest
from models.configs import EpiForecasterConfig
from training.native_trainer import source_set_manifest_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    template = yaml.safe_load(args.template.read_text())
    base = EpiForecasterConfig.from_file(template["training"]["base_config"])
    config = EpiForecasterConfig.apply_overrides(
        base, template["training"]["overrides"]
    )
    manifest = load_source_set_manifest(Path(config.native.fold_manifest))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, sources = [], []
    for length, horizon in [(28, 14), (60, 28)]:
        dataset = NativeSupportDataset(
            config.native.root,
            config.data.dataset_path,
            NativeDatasetConfig(
                input_window_length=length,
                forecast_horizon=horizon,
                window_stride=config.data.window_stride,
                min_history_observed_fraction=config.data.min_history_observed_fraction,
                min_horizon_observed_fraction=config.data.min_horizon_observed_fraction,
                confirmation_guard=config.native.confirmation_guard,
                spatial_encoder=config.native.spatial_encoder,
                od_source=config.native.od_source,
                od_min_flow_raw=config.native.od_min_flow_raw,
                od_self_flows=config.native.od_self_flows,
                mobility_enabled=config.native.mobility_enabled,
                wastewater_enabled=config.native.wastewater_enabled,
                wastewater_filename=config.native.wastewater_filename,
                include_vaccination=config.native.include_vaccination,
                include_od_context=True,
            ),
        )
        queries = dataset.queries
        heads = [dataset.head_at(i) for i in range(len(dataset))]
        inventory = pd.DataFrame(
            [
                {
                    "head": head,
                    "source_type": str(q.source_type),
                    "source_id": q.source_id,
                    "origin": q.forecast_origin,
                    "origin_date": str(dataset.date_grid[q.forecast_origin])[:10],
                }
                for q, head in zip(queries, heads, strict=True)
            ]
        )
        inventory.to_parquet(
            args.output_dir / f"eligible_{length}_{horizon}.parquet", index=False
        )
        print(f"{length}/{horizon}: {len(inventory)} eligible queries", flush=True)
        for fold in range(manifest.num_folds):
            positions = source_set_manifest_split(
                queries, heads, manifest, run_index=fold
            )
            for split, indices in zip(
                ["train", "validation", "test"], positions, strict=True
            ):
                selected = inventory.iloc[indices]
                for head, group in selected.groupby("head"):
                    summaries.append(
                        {
                            "history": length,
                            "horizon": horizon,
                            "run_index": fold,
                            "split": split,
                            "head": head,
                            "queries": len(group),
                            "sources": group.source_id.nunique(),
                            "origins": group.origin.nunique(),
                            "origin_min": int(group.origin.min()),
                            "origin_max": int(group.origin.max()),
                            "origin_date_min": group.origin_date.min(),
                            "origin_date_max": group.origin_date.max(),
                        }
                    )
                    for source, g in group.groupby("source_id"):
                        sources.append(
                            {
                                "history": length,
                                "horizon": horizon,
                                "run_index": fold,
                                "split": split,
                                "head": head,
                                "source_id": source,
                                "queries": len(g),
                            }
                        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "split_counts.csv", index=False)
    source_counts = pd.DataFrame(sources)
    source_counts.to_csv(args.output_dir / "source_counts.csv", index=False)
    keys = ["run_index", "split", "head"]
    old = summary[summary.history == 28].set_index(keys)
    new = summary[summary.history == 60].set_index(keys)
    comparison = new[["queries", "sources"]].join(
        old[["queries", "sources"]], lsuffix="_wide", rsuffix="_short"
    )
    comparison["query_retention"] = comparison.queries_wide / comparison.queries_short
    comparison["source_retention"] = comparison.sources_wide / comparison.sources_short
    comparison.to_csv(args.output_dir / "split_comparison.csv")
    for head, g in comparison.reset_index().groupby("head"):
        print(
            head,
            g.groupby("split").query_retention.agg(["min", "max"]).to_dict(),
            flush=True,
        )
    (args.output_dir / "audit_context.json").write_text(
        json.dumps(
            {
                "manifest": config.native.fold_manifest,
                "calendar": {
                    "validation_start_origin": manifest.validation_start_origin,
                    "test_start_origin": manifest.test_start_origin,
                },
                "thresholds": {
                    "history_fraction": config.data.min_history_observed_fraction,
                    "horizon_fraction": config.data.min_horizon_observed_fraction,
                },
                "scope": "same fixed source assignments and origin calendar; canonical trainer split; query counts are not independent sample counts",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
