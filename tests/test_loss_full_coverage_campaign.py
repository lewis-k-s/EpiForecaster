from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.hpo.ablation_worker import build_trial_queue, load_campaign
from scripts.workflow.run_config import validate_mn5_allocation


def test_loss_campaign_compiles_complete_paired_matrix() -> None:
    campaign = load_campaign(Path("configs/campaigns/loss_type_full_coverage_v2.yaml"))

    assert campaign.primary_metric == "mean_target_wis"
    assert campaign.fold_indices == tuple(range(10))
    assert campaign.paired_model_seeds == (52, 53)
    assert len(campaign.conditions) == 8
    assert len(build_trial_queue(campaign)) == 160


def test_rolling_interactive_binding_fits_interactive_contract() -> None:
    config = yaml.safe_load(
        Path("workflow/config/mn5_loss_type_full_coverage_v2.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert validate_mn5_allocation(config)["qos"] == "acc_interactive"
    config["ablation"]["workers"] = 3
    config["resources"]["allocation"]["tasks"] = 3
    config["resources"]["allocation"]["gres"] = "gpu:3"
    with pytest.raises(ValueError, match="40-CPU ceiling"):
        validate_mn5_allocation(config)
