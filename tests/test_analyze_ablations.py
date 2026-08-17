from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml

from optuna.distributions import CategoricalDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import create_trial

from scripts.analysis.analyze_ablations import (
    aggregate_ablation_metrics,
    build_neighborhood_context_report,
    build_run_context_report,
    collect_ablation_runs,
    compute_baseline_deltas,
    compute_seed_matched_baseline_deltas,
    get_run_fold,
    parse_experiment_name,
    validate_campaign_outputs,
    validate_campaign_trials,
)
from scripts.hpo.ablation_worker import load_campaign


def _write_run(
    root: Path,
    experiment: str,
    run_id: str,
    *,
    seed: int,
    epochs: int = 50,
    split_strategy: str = "node",
    learning_rate: float = 1e-3,
    metric_offset: float = 0.0,
) -> Path:
    run_dir = root / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training": {
            "seed": seed,
            "epochs": epochs,
            "split_strategy": split_strategy,
            "learning_rate": learning_rate,
        },
        "output": {
            "wandb_group": "group",
            "wandb_tags": ["mn5"],
            "experiment_name": experiment,
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    metrics = pd.DataFrame(
        [
            {
                "target": "cases",
                "mae_median": 1.0 + metric_offset,
                "rmse_median": 2.0 + metric_offset,
                "r2_median": 0.5 + metric_offset,
            }
        ]
    )
    metrics.to_csv(run_dir / "test_main_model_aggregate_metrics.csv", index=False)
    return run_dir


def test_parse_experiment_name_supports_new_and_legacy() -> None:
    assert parse_experiment_name("mn5_ablation__camp123__baseline") == (
        "camp123",
        "baseline",
    )
    assert parse_experiment_name("mn5_ablation_no_sir_loss") == (None, "no_sir_loss")
    assert parse_experiment_name("not_an_ablation") is None


def test_collect_ablation_runs_filters_by_campaign(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_a__baseline", "run_1", seed=42)
    _write_run(tmp_path, "mn5_ablation__camp_b__baseline", "run_2", seed=42)

    ablation_runs, campaigns = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_a",
    )

    assert campaigns == {"camp_a"}
    assert sorted(ablation_runs.keys()) == ["baseline"]
    assert len(ablation_runs["baseline"]) == 1
    assert ablation_runs["baseline"][0].campaign_id == "camp_a"


def test_aggregate_rejects_mixed_epochs(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_x__baseline", "run_1", seed=42, epochs=50)
    _write_run(tmp_path, "mn5_ablation__camp_x__baseline", "run_2", seed=43, epochs=10)

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_x",
    )
    with pytest.raises(ValueError, match="training.epochs mismatch"):
        aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)


def test_aggregate_rejects_non_seed_config_drift(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "mn5_ablation__camp_x__baseline",
        "run_1",
        seed=42,
        learning_rate=1e-3,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_x__baseline",
        "run_2",
        seed=43,
        learning_rate=5e-4,
    )

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_x",
    )
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)


def test_aggregate_accepts_seed_only_variation(tmp_path: Path) -> None:
    _write_run(tmp_path, "mn5_ablation__camp_y__baseline", "run_1", seed=42)
    _write_run(tmp_path, "mn5_ablation__camp_y__baseline", "run_2", seed=43)

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_y",
    )
    df = aggregate_ablation_metrics(ablation_runs, assert_consistent_config=True)

    assert len(df) == 1
    assert df.iloc[0]["folds"] == 2


def test_compute_baseline_deltas_includes_absolute_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "baseline",
                "target": "cases",
                "mae_mean": 2.0,
                "rmse_mean": 4.0,
                "r2_mean": 0.0,
            },
            {
                "model": "no_sir_loss",
                "target": "cases",
                "mae_mean": 3.0,
                "rmse_mean": 5.0,
                "r2_mean": 0.2,
            },
        ]
    )

    deltas = compute_baseline_deltas(df, baseline_name="baseline")
    row = deltas.iloc[0]

    assert row["mae_delta_abs"] == pytest.approx(1.0)
    assert row["rmse_delta_abs"] == pytest.approx(1.0)
    assert row["r2_delta_abs"] == pytest.approx(0.2)

    assert row["mae_delta_pct"] == pytest.approx(50.0)
    assert row["rmse_delta_pct"] == pytest.approx(25.0)
    assert "r2_delta_pct" not in deltas.columns


def test_build_neighborhood_context_report_labels_aggregation_methods(
    tmp_path: Path,
) -> None:
    experiments = {
        "baseline": (True, True, "mobility"),
        "mobility:off": (False, True, "mobility"),
        "mobility:spatial_knn": (True, True, "spatial_knn"),
        "mobility:spatial_queen": (True, True, "spatial_queen"),
        "context:off": (False, False, "mobility"),
        "context:spatial_knn": (True, False, "spatial_knn"),
        "context:spatial_queen": (True, False, "spatial_queen"),
    }
    for ablation, (
        mobility_enabled,
        regions_enabled,
        adjacency_source,
    ) in experiments.items():
        run_dir = _write_run(
            tmp_path,
            f"mn5_ablation__camp_neighborhood__{ablation}",
            "s42",
            seed=42,
        )
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        config["model"] = {
            "type": {"mobility": mobility_enabled, "regions": regions_enabled},
            "mobility_embedding_dim": 16,
            "graph_adjacency_source": adjacency_source,
            "max_neighbors": 20,
            "gnn_depth": 3,
            "gnn_hidden_dim": 32,
            "gnn_module": "gat",
            "input_window_length": 60,
            "forecast_horizon": 28,
        }
        config["data"] = {
            "dataset_path": "data/processed/real_with_id.zarr",
            "mobility_min_flow_raw": 0.0,
            "mobility_weight_clip_max": 8.0,
            "mobility_lags": [],
        }
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )

    ablation_runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_neighborhood",
    )

    run_context = build_run_context_report(ablation_runs)
    summary = build_neighborhood_context_report(run_context)
    methods = dict(
        zip(
            summary["model"],
            summary["neighborhood_aggregation_method"],
            strict=True,
        )
    )

    assert methods == {
        "baseline": "dynamic_mobility",
        "context:off": "disabled",
        "context:spatial_knn": "static_spatial_knn",
        "context:spatial_queen": "static_spatial_queen",
        "mobility:off": "disabled",
        "mobility:spatial_knn": "static_spatial_knn",
        "mobility:spatial_queen": "static_spatial_queen",
    }
    assert set(summary["seeds"]) == {"42"}


def test_end_to_end_campaign_aggregation(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__baseline",
        "run_1",
        seed=42,
        metric_offset=0.0,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__baseline",
        "run_2",
        seed=43,
        metric_offset=0.1,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__no_sir_loss",
        "run_1",
        seed=42,
        metric_offset=0.2,
    )
    _write_run(
        tmp_path,
        "mn5_ablation__camp_z__no_sir_loss",
        "run_2",
        seed=43,
        metric_offset=0.3,
    )
    # Different campaign should be excluded by filter.
    _write_run(
        tmp_path,
        "mn5_ablation__camp_other__baseline",
        "run_1",
        seed=42,
        metric_offset=9.0,
    )

    ablation_runs, campaigns = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id="camp_z",
    )
    assert campaigns == {"camp_z"}
    assert sorted(ablation_runs) == ["baseline", "no_sir_loss"]

    aggregated = aggregate_ablation_metrics(
        ablation_runs,
        assert_consistent_config=True,
    )
    deltas = compute_baseline_deltas(aggregated, baseline_name="baseline")

    assert set(aggregated["model"]) == {"baseline", "no_sir_loss"}
    assert (aggregated["folds"] == 2).all()
    assert len(deltas) == 1
    assert deltas.iloc[0]["model"] == "no_sir_loss"


def test_campaign_analysis_requires_exact_provenanced_paired_fold_identities(
    tmp_path: Path,
) -> None:
    base_config = Path("configs/train_epifor_smoke.yaml").resolve()
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(
        json.dumps({"num_folds": 2, "folds": []}) + "\n",
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_id": "paired-folds",
                "stage": "ablation",
                "hypothesis": "paired deltas retain fold identity",
                "base_config": str(base_config),
                "baseline_condition": "baseline",
                "fixed_overrides": {},
                "conditions": [
                    {
                        "name": "baseline",
                        "hypothesis": "reference",
                        "overrides": {},
                        "allowed_difference_paths": [],
                    },
                    {
                        "name": "mobility:off",
                        "hypothesis": "intervention",
                        "overrides": {"model": {"type": {"mobility": False}}},
                        "allowed_difference_paths": ["model.type.mobility"],
                    },
                ],
                "paired_model_seeds": [42, 43],
                "cross_validation": {
                    "split_manifest": str(split_manifest),
                    "fold_indices": [0, 1],
                },
                "metrics": {
                    "primary": {"name": "mae_median", "direction": "minimize"},
                    "secondary": [],
                    "safety": [],
                    "diagnostic": [],
                },
                "equal_budget_policy": {
                    "paths": ["training.epochs", "training.max_batches"]
                },
                "completeness_requirements": {
                    "require_exact_identity_matrix": True,
                    "require_terminal_trials": True,
                    "require_finite_primary_metric": True,
                },
                "provenance_requirements": {
                    "require_campaign_sha256": True,
                    "require_output_identity": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    campaign = load_campaign(campaign_path)
    journal = tmp_path / "study.journal"
    study = optuna.create_study(
        study_name="paired-folds",
        storage=JournalStorage(JournalFileBackend(str(journal))),
        direction="minimize",
    )
    distributions = {
        "condition": CategoricalDistribution(["baseline", "mobility:off"]),
        "seed": CategoricalDistribution([42, 43]),
        "fold": CategoricalDistribution([0, 1]),
    }
    for condition in ("baseline", "mobility:off"):
        for fold in (0, 1):
            for seed in (42, 43):
                run_dir = _write_run(
                    tmp_path,
                    f"mn5_ablation__paired-folds__{condition}",
                    f"{condition}-{fold}-{seed}",
                    seed=seed,
                    metric_offset=0.1 if condition == "mobility:off" else 0.0,
                )
                config = yaml.safe_load((run_dir / "config.yaml").read_text())
                config["training"]["crossval_enabled"] = True
                config["training"]["crossval_fold_index"] = fold
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(config),
                    encoding="utf-8",
                )
                (run_dir / "campaign_identity.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "campaign_sha256": campaign.sha256,
                            "campaign_id": campaign.campaign_id,
                            "condition": condition,
                            "seed": seed,
                            "fold": fold,
                        }
                    ),
                    encoding="utf-8",
                )
                study.add_trial(
                    create_trial(
                        params={"condition": condition, "seed": seed, "fold": fold},
                        distributions=distributions,
                        value=1.0,
                        user_attrs={
                            "campaign_sha256": campaign.sha256,
                            "campaign_id": campaign.campaign_id,
                            "campaign_condition": condition,
                            "campaign_seed": seed,
                            "campaign_fold": fold,
                            "optuna_value_purpose": "validation_selection_coordination",
                            "best_val_selection_metric": "validation_loss",
                            "best_val_selection_value": 1.0,
                        },
                    )
                )
    study.add_trial(
        create_trial(
            params={"condition": "baseline", "seed": 42, "fold": 0},
            distributions=distributions,
            state=optuna.trial.TrialState.FAIL,
        )
    )

    runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id=campaign.campaign_id,
    )
    validate_campaign_trials(campaign, journal_path=journal, study_name="paired-folds")
    validate_campaign_outputs(campaign, runs)
    pairwise, summary = compute_seed_matched_baseline_deltas(runs)

    assert len(pairwise) == 4
    assert set(pairwise["fold"]) == {0, 1}
    assert summary.iloc[0]["pairs"] == 4

    study.add_trial(
        create_trial(
            params={"condition": "baseline", "seed": 42, "fold": 0},
            distributions=distributions,
            state=optuna.trial.TrialState.RUNNING,
        )
    )
    with pytest.raises(ValueError, match="nonterminal trial state RUNNING"):
        validate_campaign_trials(
            campaign, journal_path=journal, study_name="paired-folds"
        )


def test_nonfolded_output_identity_uses_none_fold(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_id": "nonfolded",
                "stage": "ablation",
                "hypothesis": "non-folded campaigns have no fold identity",
                "base_config": str(Path("configs/train_epifor_smoke.yaml").resolve()),
                "baseline_condition": "baseline",
                "fixed_overrides": {"training": {"epochs": 1}},
                "conditions": [
                    {
                        "name": "baseline",
                        "hypothesis": "reference",
                        "overrides": {},
                        "allowed_difference_paths": [],
                    }
                ],
                "paired_model_seeds": [42],
                "metrics": {
                    "primary": {"name": "mae_median", "direction": "minimize"},
                    "secondary": [],
                    "safety": [],
                    "diagnostic": [],
                },
                "equal_budget_policy": {
                    "paths": ["training.epochs", "training.max_batches"]
                },
                "completeness_requirements": {
                    "require_exact_identity_matrix": True,
                    "require_terminal_trials": True,
                    "require_finite_primary_metric": True,
                },
                "provenance_requirements": {
                    "require_campaign_sha256": True,
                    "require_output_identity": True,
                },
            }
        ),
        encoding="utf-8",
    )
    campaign = load_campaign(campaign_path)
    run_dir = _write_run(
        tmp_path,
        "mn5_ablation__nonfolded__baseline",
        "seed-42",
        seed=42,
    )
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    config["training"]["crossval_enabled"] = False
    config["training"]["crossval_fold_index"] = 3
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "campaign_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_sha256": campaign.sha256,
                "campaign_id": campaign.campaign_id,
                "condition": "baseline",
                "seed": 42,
                "fold": None,
            }
        ),
        encoding="utf-8",
    )

    runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id=campaign.campaign_id,
    )

    assert get_run_fold(run_dir) is None
    validate_campaign_outputs(campaign, runs)


def test_campaign_outputs_reject_empty_or_missing_required_metric_columns(
    tmp_path: Path,
) -> None:
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_id": "empty-metrics",
                "stage": "ablation",
                "hypothesis": "metric contracts fail closed",
                "base_config": str(Path("configs/train_epifor_smoke.yaml").resolve()),
                "baseline_condition": "baseline",
                "fixed_overrides": {"training": {"epochs": 1}},
                "conditions": [
                    {
                        "name": "baseline",
                        "hypothesis": "reference",
                        "overrides": {},
                        "allowed_difference_paths": [],
                    }
                ],
                "paired_model_seeds": [42],
                "metrics": {
                    "primary": {"name": "mae_median", "direction": "minimize"},
                    "secondary": [],
                    "safety": [],
                    "diagnostic": [],
                },
                "equal_budget_policy": {
                    "paths": ["training.epochs", "training.max_batches"]
                },
                "completeness_requirements": {
                    "require_exact_identity_matrix": True,
                    "require_terminal_trials": True,
                    "require_finite_primary_metric": True,
                },
                "provenance_requirements": {
                    "require_campaign_sha256": True,
                    "require_output_identity": True,
                },
            }
        ),
        encoding="utf-8",
    )
    campaign = load_campaign(campaign_path)
    run_dir = _write_run(
        tmp_path,
        "mn5_ablation__empty-metrics__baseline",
        "seed-42",
        seed=42,
    )
    (run_dir / "campaign_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_sha256": campaign.sha256,
                "campaign_id": campaign.campaign_id,
                "condition": "baseline",
                "seed": 42,
                "fold": None,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(columns=["target", "mae_median"]).to_csv(
        run_dir / "test_main_model_aggregate_metrics.csv",
        index=False,
    )
    runs, _ = collect_ablation_runs(
        tmp_path,
        experiment_pattern="mn5_ablation__*__*",
        campaign_id=campaign.campaign_id,
    )

    with pytest.raises(ValueError, match="aggregate metric table is empty"):
        validate_campaign_outputs(campaign, runs)

    pd.DataFrame({"target": ["cases"]}).to_csv(
        run_dir / "test_main_model_aggregate_metrics.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="mae_median"):
        validate_campaign_outputs(campaign, runs)

    pd.DataFrame({"mae_median": [1.0]}).to_csv(
        run_dir / "test_main_model_aggregate_metrics.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="target"):
        validate_campaign_outputs(campaign, runs)

    pd.DataFrame({"target": ["wastewater"], "mae_median": [float("nan")]}).to_csv(
        run_dir / "test_main_model_aggregate_metrics.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="nonfinite primary metric"):
        validate_campaign_outputs(campaign, runs)

    campaign_document = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign_document["completeness_requirements"]["require_finite_primary_metric"] = (
        False
    )
    campaign_path.write_text(
        yaml.safe_dump(campaign_document),
        encoding="utf-8",
    )
    campaign = load_campaign(campaign_path)
    output_identity_path = run_dir / "campaign_identity.json"
    output_identity = json.loads(output_identity_path.read_text(encoding="utf-8"))
    output_identity["campaign_sha256"] = campaign.sha256
    output_identity_path.write_text(json.dumps(output_identity), encoding="utf-8")

    validate_campaign_outputs(campaign, runs)
