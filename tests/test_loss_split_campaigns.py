"""The loss comparison keeps temporal and spatial evaluation separate."""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from data.preprocess.config import PreprocessingConfig
from data.research_roles import research_time_ranges
from evaluation.ablation_probabilistic_analysis import campaign_analysis_spec
from scripts.hpo.ablation_worker import build_trial_queue, load_campaign


def _document(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def test_temporal_campaign_scores_four_blocks_without_region_folds() -> None:
    document = _document("configs/campaigns/loss_type_temporal_v1.yaml")
    campaign = load_campaign(Path("configs/campaigns/loss_type_temporal_v1.yaml"))
    roles = json.loads(
        Path("configs/research/loss_roles_temporal_v1.json").read_text()
    )["roles"]
    preprocessing = _document(
        "configs/production_only/preprocess_mn5_loss_type_temporal_v1.yaml"
    )

    assert campaign.fold_indices == ()
    assert len(build_trial_queue(campaign)) == 16
    assert document["fixed_overrides"]["training"]["split_strategy"] == "research"
    assert document["fixed_overrides"]["training"]["crossval_enabled"] is False
    assert (
        preprocessing["smoothing"]["parameter_fit_end_date"]
        == roles["fit"]["label_end"]
    )
    assert preprocessing["smoothing"]["parameter_fit_mode"] == "per_series"
    blocks = document["analysis"]["temporal_blocks"]
    assert len(blocks) == 4
    assert blocks[0]["label_start"] == roles["calibration"]["label_start"]
    assert blocks[-1]["label_end"] == roles["calibration"]["label_end"]
    for index, block in enumerate(blocks):
        start = date.fromisoformat(block["label_start"])
        end = date.fromisoformat(block["label_end"])
        assert (end - start).days == 27
        if index:
            assert start == date.fromisoformat(
                blocks[index - 1]["label_end"]
            ) + timedelta(days=1)
    assert (
        date.fromisoformat(roles["confirmation"]["label_start"])
        - date.fromisoformat(roles["calibration"]["label_end"])
    ).days == 29
    assert campaign_analysis_spec(campaign) is not None


def test_spatial_campaign_uses_full_folds_and_stateless_smoothing() -> None:
    document = _document("configs/campaigns/loss_type_spatial_v1.yaml")
    campaign = load_campaign(Path("configs/campaigns/loss_type_spatial_v1.yaml"))
    preprocessing = _document(
        "configs/production_only/preprocess_mn5_loss_type_spatial_v1.yaml"
    )

    assert campaign.fold_indices == tuple(range(10))
    assert len(build_trial_queue(campaign)) == 160
    assert document["fixed_overrides"]["training"]["split_strategy"] == "node"
    assert "research_role_manifest_path" not in document["fixed_overrides"]["training"]
    assert preprocessing["smoothing"]["parameter_fit_mode"] == "fixed"
    assert document["analysis"]["temporal_evaluation"]["scored_role"] == "all_dates"
    assert campaign_analysis_spec(campaign) is not None


def test_rolling_temporal_campaigns_have_distinct_causal_fit_cutoffs() -> None:
    windows = (
        ("20200821", "2020-11-19", "2020-12-16"),
        ("20200918", "2020-12-17", "2021-01-13"),
        ("20201016", "2021-01-14", "2021-02-10"),
        ("20201113", "2021-02-11", "2021-03-10"),
    )
    date_grid = pd.date_range("2020-03-01", "2021-05-09", freq="D").values
    datasets = set()
    for stamp, score_start, score_end in windows:
        path = Path(
            f"configs/campaigns/loss_type_temporal_fit{stamp}_student_t_crps_v2.yaml"
        )
        campaign = load_campaign(path)
        document = _document(str(path))
        roles = json.loads(
            Path(f"configs/research/loss_roles_temporal_fit{stamp}_v2.json").read_text()
        )
        assert len(build_trial_queue(campaign)) == 4
        assert campaign.fold_indices == ()
        assert research_time_ranges(date_grid, roles, input_window=60, horizon=28)
        assert roles["roles"]["fit"]["label_end"].replace("-", "") == stamp
        assert roles["roles"]["calibration"] == {
            "label_start": score_start,
            "label_end": score_end,
        }
        assert document["analysis"]["temporal_blocks"] == [
            roles["roles"]["calibration"]
        ]
        dataset_path = document["fixed_overrides"]["data"]["dataset_path"]
        assert dataset_path not in datasets
        datasets.add(dataset_path)
        if stamp != "20200821":
            config = PreprocessingConfig.from_file(
                f"configs/research/preprocess_loss_type_temporal_fit{stamp}_v2.yaml"
            )
            assert (
                config.smoothing.parameter_fit_end_date
                == roles["roles"]["fit"]["label_end"]
            )
            assert (
                config.source_provenance["parameter_fit_end_date"]
                == config.smoothing.parameter_fit_end_date
            )


def test_bounded_smoke_campaigns_have_one_pair_each() -> None:
    for design, version in (("temporal", 2), ("spatial", 3)):
        campaign = load_campaign(
            Path(f"configs/campaigns/loss_type_{design}_smoke_v{version}.yaml")
        )
        assert len(build_trial_queue(campaign, smoketest=True)) == 2
        assert campaign.primary_metric == "wis_median"
