"""Every checked-in campaign declaration loads under the stage contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hpo.ablation_worker import CAMPAIGN_STAGES, load_campaign

CAMPAIGNS = sorted(Path("configs/campaigns").glob("*.yaml"))


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda path: path.name)
def test_campaign_loads(path: Path) -> None:
    campaign = load_campaign(path)
    assert campaign.stage in CAMPAIGN_STAGES
    if campaign.stage == "ablation":
        assert len(campaign.paired_model_seeds) >= 2
        assert len(campaign.conditions) >= 2
        if campaign.has_folds:
            assert campaign.completeness_requirements["require_full_manifest_folds"]


def test_smoke_campaign_is_labelled_smoke() -> None:
    campaign = load_campaign(Path("configs/campaigns/ablation_smoke_v1.yaml"))
    assert campaign.stage == "ablation_smoke"
