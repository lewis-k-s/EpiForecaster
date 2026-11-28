from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.workflow import pinned_inputs
from scripts.workflow.pinned_inputs import (
    EVALUATION,
    TRAINING_INIT,
    TRAINING_RESUME,
    TRAINING_RESUME_BEST,
    PinnedFile,
    bind_dirs,
    declared_inputs,
    pin_inputs,
    recorded_inputs,
    verified_input,
    verify_file,
)


def _source(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _train(**training: object) -> dict[str, object]:
    return {"workflow": {"kind": "train"}, "training": training}


def _write_manifest(run_root: Path, records: dict[str, dict[str, str]]) -> Path:
    manifest = run_root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": pinned_inputs.RUN_MANIFEST_SCHEMA_VERSION,
                pinned_inputs.MANIFEST_SECTION: records,
            }
        )
    )
    return manifest


def _ablation(source_root: Path, checkpoints: dict[str, Path]) -> dict[str, object]:
    campaign = source_root / "configs/campaigns/transfer.yaml"
    campaign.parent.mkdir(parents=True, exist_ok=True)
    campaign.write_text(
        yaml.safe_dump(
            {
                "campaign_id": "transfer",
                "init_checkpoints": {name: str(p) for name, p in checkpoints.items()},
                "conditions": [
                    {"name": f"ft-{name}", "init_checkpoint": name}
                    for name in checkpoints
                ]
                + [{"name": "scratch"}],
            }
        )
    )
    return {
        "workflow": {"kind": "ablation"},
        "ablation": {"campaign": "configs/campaigns/transfer.yaml"},
    }


def test_pin_copies_read_only_and_records_content_identity(tmp_path: Path) -> None:
    source = _source(tmp_path / "pretrain/best_model.pt", b"pretrained")
    run_root = tmp_path / "run"
    inputs = declared_inputs(_train(init_checkpoint=str(source)), tmp_path)

    pinned = pin_inputs(inputs, run_root)

    copy = run_root / "inputs/training_init_checkpoint.pt"
    sha256 = hashlib.sha256(b"pretrained").hexdigest()
    assert copy.read_bytes() == b"pretrained"
    assert not copy.stat().st_mode & 0o222
    assert pinned.identity == {"init_checkpoint_sha256": sha256}
    assert pinned.records == {
        TRAINING_INIT: {
            "configured_checkpoint": str(source),
            "resolved_checkpoint": str(source.resolve()),
            "pinned_checkpoint": str(copy),
            "pinned_checkpoint_relative": "inputs/training_init_checkpoint.pt",
            "checkpoint_sha256": sha256,
        }
    }


def test_repin_accepts_same_content_and_rejects_changed_source(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "eval/best_model.pt", b"model")
    run_root = tmp_path / "run"
    config = {"workflow": {"kind": "evaluate"}, "evaluation": {"checkpoint": source}}
    inputs = declared_inputs(config, tmp_path)
    first = pin_inputs(inputs, run_root)

    assert pin_inputs(inputs, run_root) == first

    source.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="conflicts with configured input"):
        pin_inputs(inputs, run_root)


def test_pin_rejects_a_symlinked_source(tmp_path: Path) -> None:
    target = _source(tmp_path / "real.pt", b"model")
    link = tmp_path / "link.pt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        pin_inputs(
            declared_inputs(_train(init_checkpoint=str(link)), tmp_path),
            tmp_path / "run",
        )


def test_verified_input_returns_the_pinned_copy(tmp_path: Path) -> None:
    source = _source(tmp_path / "pretrain/best_model.pt", b"pretrained")
    run_root = tmp_path / "run"
    (declared,) = declared_inputs(_train(init_checkpoint=str(source)), tmp_path)
    manifest = _write_manifest(run_root, pin_inputs([declared], run_root).records)

    pinned = verified_input(manifest, declared)

    assert pinned == PinnedFile(
        path=run_root / "inputs/training_init_checkpoint.pt",
        sha256=hashlib.sha256(b"pretrained").hexdigest(),
    )


def test_verified_input_rejects_a_tampered_copy(tmp_path: Path) -> None:
    source = _source(tmp_path / "pretrain/best_model.pt", b"pretrained")
    run_root = tmp_path / "run"
    (declared,) = declared_inputs(_train(init_checkpoint=str(source)), tmp_path)
    manifest = _write_manifest(run_root, pin_inputs([declared], run_root).records)
    copy = run_root / declared.relative
    copy.chmod(0o644)
    copy.write_bytes(b"substituted")

    with pytest.raises(ValueError, match="failed validation"):
        verified_input(manifest, declared)
    with pytest.raises(ValueError, match="failed validation"):
        verify_file(
            PinnedFile(path=copy, sha256=hashlib.sha256(b"pretrained").hexdigest()),
            label="init",
        )


def test_verified_input_rejects_a_changed_declaration(tmp_path: Path) -> None:
    source = _source(tmp_path / "a/best_model.pt", b"a")
    other = _source(tmp_path / "b/best_model.pt", b"a")
    run_root = tmp_path / "run"
    (declared,) = declared_inputs(_train(init_checkpoint=str(source)), tmp_path)
    manifest = _write_manifest(run_root, pin_inputs([declared], run_root).records)
    (redeclared,) = declared_inputs(_train(init_checkpoint=str(other)), tmp_path)

    with pytest.raises(ValueError, match="does not match its declaration"):
        verified_input(manifest, redeclared)

    (evaluation,) = declared_inputs(
        {"workflow": {"kind": "evaluate"}, "evaluation": {"checkpoint": source}},
        tmp_path,
    )
    with pytest.raises(ValueError, match="does not pin the evaluation checkpoint"):
        verified_input(manifest, evaluation)


def test_training_resume_pair_is_declared_bound_and_identified(
    tmp_path: Path,
) -> None:
    resume = _source(tmp_path / "prev/checkpoints/checkpoint_epoch_0002.pt", b"r")
    best = _source(tmp_path / "prev/checkpoints/best_model.pt", b"b")
    inputs = declared_inputs(
        _train(resume_checkpoint=str(resume), resume_best_checkpoint=str(best)),
        tmp_path,
    )

    assert [pinned.key for pinned in inputs] == [TRAINING_RESUME, TRAINING_RESUME_BEST]
    assert bind_dirs(inputs) == [tmp_path / "prev/checkpoints"]
    assert pin_inputs(inputs, tmp_path / "run").identity == {
        "resume_checkpoint_sha256": hashlib.sha256(b"r").hexdigest(),
        "resume_best_checkpoint_sha256": hashlib.sha256(b"b").hexdigest(),
    }


def test_campaign_init_checkpoints_are_pinned_per_campaign(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoints = {
        "synth": _source(tmp_path / "pretrain/synth.pt", b"synth"),
        "real": _source(tmp_path / "other/real.pt", b"real"),
    }
    inputs = declared_inputs(_ablation(source_root, checkpoints), source_root)

    # The campaign file is YAML with sorted keys, so declarations come back sorted.
    assert [pinned.key for pinned in inputs] == [
        "campaign_init/transfer/real",
        "campaign_init/transfer/synth",
    ]
    assert bind_dirs(inputs) == [tmp_path / "other", tmp_path / "pretrain"]
    run_root = tmp_path / "run"
    pinned = pin_inputs(inputs, run_root)
    assert (run_root / "inputs/campaign_init/transfer/synth.pt").read_bytes() == (
        b"synth"
    )
    assert pinned.identity == {
        "campaign_init_checkpoint_sha256": {
            "transfer": {
                "synth": hashlib.sha256(b"synth").hexdigest(),
                "real": hashlib.sha256(b"real").hexdigest(),
            }
        }
    }


def test_workflows_without_out_of_run_checkpoints_pin_nothing(
    tmp_path: Path,
) -> None:
    assert declared_inputs({"workflow": {"kind": "hpo"}}, tmp_path) == ()
    assert declared_inputs(_train(), tmp_path) == ()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_train(resume_checkpoint="/gpfs/r.pt"), "provided together"),
        (_train(resume_best_checkpoint="/gpfs/b.pt"), "provided together"),
        (_train(init_checkpoint="relative/b.pt"), "init_checkpoint must be an abs"),
        (
            _train(resume_checkpoint="/gpfs/r.pt", resume_best_checkpoint="rel/b.pt"),
            "resume_best_checkpoint must be an absolute",
        ),
        (
            _train(
                init_checkpoint="/gpfs/i.pt",
                resume_checkpoint="/gpfs/r.pt",
                resume_best_checkpoint="/gpfs/b.pt",
            ),
            "mutually exclusive",
        ),
        (_train(require_init_checkpoint=True), "init_checkpoint is required"),
        (_train(require_init_checkpoint="yes"), "must be a boolean"),
        (
            {"workflow": {"kind": "evaluate"}, "evaluation": {}},
            "evaluation.checkpoint must be an absolute",
        ),
    ],
)
def test_invalid_declarations_are_rejected(
    tmp_path: Path, config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        declared_inputs(config, tmp_path)


def test_recorded_inputs_reads_schema_1_manifests() -> None:
    record = {"checkpoint_sha256": "a" * 64}
    manifest = {
        "schema_version": 1,
        "training_input": record,
        "training_resume_input": {"resume": record, "best": record},
        "evaluation_input": record,
        "campaign_init_inputs": {"transfer": {"synth": record}},
    }

    assert recorded_inputs(manifest) == {
        TRAINING_INIT: record,
        TRAINING_RESUME: record,
        TRAINING_RESUME_BEST: record,
        EVALUATION: record,
        "campaign_init/transfer/synth": record,
    }
    with pytest.raises(ValueError, match="Unsupported run manifest"):
        recorded_inputs({"schema_version": 3})
