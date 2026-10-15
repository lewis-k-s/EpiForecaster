from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from evaluation.ablation_probabilistic_analysis import (
    build_probabilistic_campaign_outputs,
)
from scripts.analysis.analyze_ablations import AblationRun
from scripts.hpo.ablation_worker import load_campaign


def _campaign(tmp_path: Path, mobility_path: Path) -> Path:
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps(
            {
                "num_folds": 2,
                "folds": [
                    {"fold_index": 0, "region_ids": ["r0"]},
                    {"fold_index": 1, "region_ids": ["r1"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    document = {
        "schema_version": 1,
        "campaign_id": "probabilistic-test",
        "stage": "ablation",
        "hypothesis": "probabilistic metrics retain exact support",
        "base_config": str(Path("configs/train_epifor_smoke.yaml").resolve()),
        "baseline_condition": "baseline",
        "fixed_overrides": {"training": {"epochs": 1}},
        "conditions": [
            {
                "name": "baseline",
                "hypothesis": "reference",
                "overrides": {},
                "allowed_difference_paths": [],
            },
            {
                "name": "mobility:off",
                "hypothesis": "mobility contrast",
                "overrides": {"model": {"type": {"mobility": False}}},
                "allowed_difference_paths": ["model.type.mobility"],
            },
        ],
        "paired_model_seeds": [52],
        "cross_validation": {
            "split_manifest": str(manifest),
            "fold_indices": [0, 1],
        },
        "metrics": {
            "primary": {"name": "mean_target_wis", "direction": "minimize"},
            "secondary": [],
            "safety": [],
            "diagnostic": [],
        },
        "equal_budget_policy": {"paths": ["training.epochs", "training.max_batches"]},
        "completeness_requirements": {
            "require_exact_identity_matrix": True,
            "require_terminal_trials": True,
            "require_finite_primary_metric": True,
            "require_full_manifest_folds": True,
        },
        "provenance_requirements": {
            "require_campaign_sha256": True,
            "require_output_identity": True,
        },
        "analysis": {
            "kind": "probabilistic_full_coverage",
            "granular_filename": "test_granular.parquet",
            "required_heads": [
                "wastewater",
                "hospitalizations",
                "cases",
                "deaths",
            ],
            "temporal_evaluation": {
                "scored_role": "calibration",
                "calibration_control": "C0",
                "confirmation": "preserved",
            },
            "mobility_context": {
                "kind": "incoming_flow_raw_tertiles",
                "path": str(mobility_path),
                "variable": "mobility",
                "date_start": "2020-05-01",
                "date_end": "2020-05-02",
            },
        },
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_run(
    root: Path, condition: str, fold: int, *, wis_offset: float
) -> AblationRun:
    run_dir = root / condition / f"fold-{fold}"
    run_dir.mkdir(parents=True)
    region = f"r{fold}"
    rows = []
    for target in ("wastewater", "hospitalizations", "cases", "deaths"):
        for horizon in (1, 2):
            rows.append(
                {
                    "split": "test",
                    "target": target,
                    "node_id": fold,
                    "region_id": region,
                    "window_start": 10,
                    "horizon": horizon,
                    "target_date": f"2021-01-0{horizon}",
                    "observed": 1.0,
                    "observed_mask": 1.0,
                    "observation_weight": 1.0,
                    "wis": 1.0 + wis_offset,
                    "crps": 2.0 + wis_offset,
                    "nll": 3.0 + wis_offset,
                    "abs_error": 4.0 + wis_offset,
                    "covered_50": True,
                    "covered_80": True,
                    "covered_95": False,
                    "interval_low_50": 0.0,
                    "interval_high_50": 1.0,
                    "interval_low_80": -1.0,
                    "interval_high_80": 2.0,
                    "interval_low_95": -2.0,
                    "interval_high_95": 3.0,
                }
            )
    pd.DataFrame(rows).to_parquet(run_dir / "test_granular.parquet", index=False)
    return AblationRun(
        ablation=condition,
        campaign_id="probabilistic-test",
        experiment_dir=root / condition,
        run_dir=run_dir,
        seed=52,
        fold=fold,
    )


def test_probabilistic_campaign_outputs_validate_coverage_and_support(
    tmp_path: Path,
) -> None:
    mobility_path = tmp_path / "mobility.zarr"
    xr.Dataset(
        {
            "mobility": (
                ("date", "origin", "target"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32),
            )
        },
        coords={
            "date": pd.to_datetime(["2020-05-01", "2020-05-02"]),
            "origin": ["origin"],
            "target": ["r0", "r1"],
        },
    ).to_zarr(mobility_path)
    campaign = load_campaign(_campaign(tmp_path, mobility_path))
    runs = {
        "baseline": [
            _write_run(tmp_path, "baseline", fold, wis_offset=0.0) for fold in (0, 1)
        ],
        "mobility:off": [
            _write_run(tmp_path, "mobility-off", fold, wis_offset=0.25)
            for fold in (0, 1)
        ],
    }
    for run in runs["mobility:off"]:
        object.__setattr__(run, "ablation", "mobility:off")

    outputs = build_probabilistic_campaign_outputs(campaign, runs)

    assert set(outputs) == {
        "campaign_completeness_manifest",
        "condition_scorecard",
        "condition_head_scorecard",
        "paired_fold_seed_deltas",
        "paired_region_deltas",
        "paired_head_horizon_deltas",
        "regional_effect_distribution",
        "mobility_context_summary",
        "calibration_interval_diagnostics",
        "invalid_prediction_support_audit",
    }
    assert len(outputs["campaign_completeness_manifest"]) == 4
    assert outputs["paired_region_deltas"]["mean_target_wis_delta"].eq(0.25).all()
    assert outputs["regional_effect_distribution"].iloc[0]["worsened_regions"] == 2
    assert (
        outputs["invalid_prediction_support_audit"]["source_measurements_independent"]
        .eq(False)
        .all()
    )


def test_probabilistic_campaign_rejects_unmatched_support(tmp_path: Path) -> None:
    mobility_path = tmp_path / "mobility.zarr"
    xr.Dataset(
        {"mobility": (("date", "origin", "target"), np.zeros((1, 1, 2)))},
        coords={
            "date": pd.to_datetime(["2020-05-01"]),
            "origin": ["origin"],
            "target": ["r0", "r1"],
        },
    ).to_zarr(mobility_path)
    campaign = load_campaign(_campaign(tmp_path, mobility_path))
    baseline = [
        _write_run(tmp_path, "baseline", fold, wis_offset=0.0) for fold in (0, 1)
    ]
    candidate = [
        _write_run(tmp_path, "mobility-off", fold, wis_offset=0.25) for fold in (0, 1)
    ]
    for run in candidate:
        object.__setattr__(run, "ablation", "mobility:off")
    frame = pd.read_parquet(candidate[0].run_dir / "test_granular.parquet")
    frame.loc[0, "observed"] = 99.0
    frame.to_parquet(candidate[0].run_dir / "test_granular.parquet", index=False)

    with pytest.raises(ValueError, match="observed values differ"):
        build_probabilistic_campaign_outputs(
            campaign,
            {"baseline": baseline, "mobility:off": candidate},
        )


def test_temporal_campaign_uses_all_regions_and_reports_blocks(tmp_path: Path) -> None:
    mobility_path = tmp_path / "mobility.zarr"
    xr.Dataset(
        {"mobility": (("date", "origin", "target"), np.ones((2, 1, 2)))},
        coords={
            "date": pd.to_datetime(["2020-05-01", "2020-05-02"]),
            "origin": ["origin"],
            "target": ["r0", "r1"],
        },
    ).to_zarr(mobility_path)
    campaign_path = _campaign(tmp_path, mobility_path)
    document = yaml.safe_load(campaign_path.read_text())
    manifest = document.pop("cross_validation")["split_manifest"]
    document["completeness_requirements"]["require_full_manifest_folds"] = False
    document["analysis"].update(
        {
            "evaluation_design": "temporal",
            "region_manifest": manifest,
            "temporal_blocks": [
                {"label_start": "2021-01-01", "label_end": "2021-01-01"},
                {"label_start": "2021-01-02", "label_end": "2021-01-02"},
            ],
        }
    )
    campaign_path.write_text(yaml.safe_dump(document))
    runs = {}
    for condition, offset in (("baseline", 0.0), ("mobility:off", 0.25)):
        run = _write_run(tmp_path, condition, 0, wis_offset=offset)
        frame = pd.read_parquet(run.run_dir / "test_granular.parquet")
        other = frame.assign(node_id=1, region_id="r1")
        pd.concat([frame, other], ignore_index=True).to_parquet(
            run.run_dir / "test_granular.parquet", index=False
        )
        object.__setattr__(run, "fold", None)
        runs[condition] = [run]

    outputs = build_probabilistic_campaign_outputs(load_campaign(campaign_path), runs)

    assert (
        outputs["campaign_completeness_manifest"]["expected_region_count"].eq(2).all()
    )
    assert outputs["campaign_completeness_manifest"]["fold"].eq(-1).all()
    blocks = outputs["paired_temporal_block_horizon_deltas"]
    assert set(blocks["temporal_block"]) == {0, 1}
    assert blocks["wis_delta"].eq(0.25).all()
