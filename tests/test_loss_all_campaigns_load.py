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


@pytest.mark.parametrize(
    "name",
    [
        "loss_type_spatial_smoke_v3",
        "loss_type_temporal_smoke_v2",
        "loss_type_spatial_rolling90_10epoch_smoke_v1",
    ],
)
def test_smoke_campaigns_are_labelled_smoke(name: str) -> None:
    assert load_campaign(Path(f"configs/campaigns/{name}.yaml")).stage == (
        "ablation_smoke"
    )
