"""Checkpoint reuse is accepted only for matching code, container and inputs."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
import yaml

from scripts.hpo import ablation_worker as worker
from tests.test_ablation_worker import _write_campaign

BUNDLE = "a" * 64
OTHER_BUNDLE = "b" * 64
CONTAINER = "c" * 64
IDENTITY = ("baseline", 42, 0)


def _campaign(tmp_path: Path, **extra: object) -> worker.AblationCampaign:
    return worker.load_campaign(
        _write_campaign(tmp_path, folds=[0, 1], extra=dict(extra) or None)
    )


def _execution(
    campaign: worker.AblationCampaign, *, bundle: str = BUNDLE
) -> worker.ExecutionIdentity:
    hydrated = worker.hydrated_training_config(
        campaign, condition="baseline", fold=0, seed=42
    )
    outputs = {
        Path(str(hydrated["data"]["dataset_path"])).as_posix(): "d" * 32,
        Path(str(hydrated["data"]["region2vec_path"])).as_posix(): "e" * 32,
    }
    return worker.ExecutionIdentity(
        source_bundle_sha256=bundle,
        container_digest=CONTAINER,
        dvc_md5_by_output=outputs,
    )


def _record(campaign: worker.AblationCampaign, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "condition": IDENTITY[0],
        "seed": IDENTITY[1],
        "fold": IDENTITY[2],
        "status": "reusable",
        "source_bundle_sha256": BUNDLE,
        "container_digest": f"sha256:{CONTAINER}",
        "dataset_dvc_md5": "d" * 32,
        "region2vec_dvc_md5": "e" * 32,
        "hydrated_training_config_sha256": worker.hydrated_training_config_sha256(
            campaign, condition=IDENTITY[0], fold=0, seed=42
        ),
    }
    record.update(overrides)
    return record


def _receipt(
    tmp_path: Path, campaign: worker.AblationCampaign, records: list[dict[str, object]]
) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "campaign_sha256": campaign.sha256,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_matching_record_is_accepted_and_receipt_is_written(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    acceptance = tmp_path / "run" / "reuse_acceptance.json"

    identities = worker.load_reuse_identities(
        _receipt(tmp_path, campaign, [_record(campaign)]),
        campaign,
        _execution(campaign),
        acceptance,
    )

    assert identities == {IDENTITY}
    written = json.loads(acceptance.read_text(encoding="utf-8"))
    assert written["accepted"] == [
        {"condition": "baseline", "seed": 42, "fold": 0}
    ]
    assert written["rejected"] == []


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"container_digest": "sha256:" + "f" * 64}, "container_digest_mismatch"),
        ({"container_digest": None}, "container_digest_mismatch"),
        ({"dataset_dvc_md5": "0" * 32}, "dataset_dvc_md5_mismatch"),
        ({"region2vec_dvc_md5": "0" * 32}, "region2vec_dvc_md5_mismatch"),
        (
            {"hydrated_training_config_sha256": "0" * 64},
            "hydrated_training_config_mismatch",
        ),
        ({"source_bundle_sha256": OTHER_BUNDLE}, "source_bundle_not_allowed"),
        ({"source_bundle_sha256": None}, "source_bundle_not_allowed"),
        ({"status": "rejected"}, "audit_status_not_reusable"),
    ],
)
def test_identity_mismatch_is_rejected_and_listed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    override: dict[str, object],
    reason: str,
) -> None:
    campaign = _campaign(tmp_path)
    acceptance = tmp_path / "reuse_acceptance.json"

    with caplog.at_level("WARNING"):
        identities = worker.load_reuse_identities(
            _receipt(tmp_path, campaign, [_record(campaign, **override)]),
            campaign,
            _execution(campaign),
            acceptance,
        )

    assert identities == set()
    rejected = json.loads(acceptance.read_text(encoding="utf-8"))["rejected"]
    assert reason in rejected[0]["reasons"]
    assert "Rejected 1 of 1 checkpoint reuse records" in caplog.text


def test_code_identity_needs_explicit_campaign_allow_list(tmp_path: Path) -> None:
    allowed = _campaign(tmp_path, reuse={"allow_code_identities": [OTHER_BUNDLE]})
    record = _record(allowed, source_bundle_sha256=OTHER_BUNDLE)

    assert worker.load_reuse_identities(
        _receipt(tmp_path, allowed, [record]), allowed, _execution(allowed)
    ) == {IDENTITY}


def test_allow_list_must_hold_sha256_digests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _campaign(tmp_path, reuse={"allow_code_identities": ["not-a-digest"]})


def test_receipt_for_another_campaign_is_an_error(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    path = _receipt(tmp_path, campaign, [_record(campaign)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["campaign_sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="campaign identity"):
        worker.load_reuse_identities(path, campaign, _execution(campaign))


def test_undeclared_identity_is_an_error(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    record = _record(campaign, condition="not-declared")

    with pytest.raises(ValueError, match="undeclared identities"):
        worker.load_reuse_identities(
            _receipt(tmp_path, campaign, [record]), campaign, _execution(campaign)
        )


def test_reuse_without_run_identity_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    with pytest.raises(ValueError, match="executing run identity"):
        worker.load_reuse_identities(
            _receipt(tmp_path, campaign, [_record(campaign)]), campaign
        )


def test_no_receipt_means_no_reuse(tmp_path: Path) -> None:
    assert worker.load_reuse_identities(None, _campaign(tmp_path)) == set()


def test_record_reused_trials_adds_complete_trials_once(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    study = optuna.create_study()

    worker._record_reused_trials(study, campaign, {IDENTITY})
    worker._record_reused_trials(study, campaign, {IDENTITY})

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert worker.trial_identity(trial) == IDENTITY
    assert trial.user_attrs["campaign_sha256"] == campaign.sha256
    assert (
        trial.user_attrs["optuna_value_purpose"]
        == "provenance_verified_checkpoint_reuse"
    )


def test_rejected_record_is_queued_for_training(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    bad = _record(campaign, container_digest="sha256:" + "f" * 64)
    reuse = worker.load_reuse_identities(
        _receipt(tmp_path, campaign, [bad]), campaign, _execution(campaign)
    )
    study = optuna.create_study()

    worker.initialize_study_queue(
        study=study,
        campaign=campaign,
        lock_path=tmp_path / "queue.lock",
        reuse_identities=reuse,
    )

    states = {
        worker.trial_identity(trial): trial.state
        for trial in study.trials
        if worker.trial_identity(trial) == IDENTITY
    }
    assert states == {IDENTITY: optuna.trial.TrialState.WAITING}


def test_load_execution_identity_reads_run_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "identity": {
                    "source_snapshot_sha256": BUNDLE,
                    "container_digest": f"sha256:{CONTAINER}",
                },
                "dvc": {
                    "targets": {
                        "data/x.nc.dvc": {"md5": "d" * 32, "output": "data/x.nc"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    execution = worker.load_execution_identity(manifest)

    assert execution.source_bundle_sha256 == BUNDLE
    assert execution.container_digest == CONTAINER
    assert execution.dvc_md5_by_output == {"data/x.nc": "d" * 32}
