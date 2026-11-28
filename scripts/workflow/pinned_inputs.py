"""Pinned inputs: out-of-run checkpoints that a run copies in and identifies.

A workflow config declares absolute source paths: ``training.init_checkpoint``,
``training.resume_checkpoint`` with ``training.resume_best_checkpoint``,
``evaluation.checkpoint``, and the ``init_checkpoints`` of each ablation
campaign. This module owns the path from those declarations to a verified use:

* ``declared_inputs`` validates the declarations and gives each input its key,
  run-relative location and identity field;
* ``bind_dirs`` names the source directories the contained coordinator binds;
* ``pin_inputs`` copies, hashes and write-protects each input at prepare and
  returns the manifest section and identity entries;
* ``verified_input`` and ``verify_file`` re-hash a pinned copy before use.

It uses only the standard library and PyYAML, so render, the Snakefile and the
workers can import it without the training stack.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.hpo.campaign_init import read_campaign_init_checkpoints

# Run manifest schema with every pinned input in one ``pinned_inputs`` section.
# Schema 1 manifests hold them in four kind-specific sections.
RUN_MANIFEST_SCHEMA_VERSION = 2
MANIFEST_SECTION = "pinned_inputs"

TRAINING_INIT = "training_init"
TRAINING_RESUME = "training_resume"
TRAINING_RESUME_BEST = "training_resume_best"
EVALUATION = "evaluation"
CAMPAIGN_INIT_DIR = Path("inputs/campaign_init")

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PinnedInput:
    """One declared out-of-run checkpoint and its place in the run."""

    key: str
    label: str
    configured: Path
    relative: Path
    identity_path: tuple[str, ...]


@dataclass(frozen=True)
class PinnedFile:
    """A pinned copy and the SHA-256 that prepare recorded for it."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class PinnedRecords:
    """What prepare writes: the manifest section and the run identity entries."""

    records: dict[str, dict[str, str]]
    identity: dict[str, Any]


def campaign_init_input(campaign_id: str, name: str, configured: Path) -> PinnedInput:
    """The pinned input of one campaign-declared init checkpoint."""
    return PinnedInput(
        key=f"campaign_init/{campaign_id}/{name}",
        label=f"campaign {campaign_id} init checkpoint {name}",
        configured=configured,
        relative=CAMPAIGN_INIT_DIR / campaign_id / f"{name}.pt",
        identity_path=("campaign_init_checkpoint_sha256", campaign_id, name),
    )


def _absolute(value: object, *, field: str, purpose: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute GPFS path {purpose}")
    return path


def _training_inputs(training: Mapping[str, Any]) -> list[PinnedInput]:
    init_checkpoint = training.get("init_checkpoint")
    require_init_checkpoint = training.get("require_init_checkpoint", False)
    if not isinstance(require_init_checkpoint, bool):
        raise ValueError("training.require_init_checkpoint must be a boolean")
    if require_init_checkpoint and init_checkpoint is None:
        raise ValueError(
            "training.init_checkpoint is required by this workflow template"
        )
    resume_checkpoint = training.get("resume_checkpoint")
    resume_best_checkpoint = training.get("resume_best_checkpoint")
    if (resume_checkpoint is None) != (resume_best_checkpoint is None):
        raise ValueError(
            "training.resume_checkpoint and training.resume_best_checkpoint "
            "must be provided together"
        )
    inputs: list[PinnedInput] = []
    if init_checkpoint is not None:
        inputs.append(
            PinnedInput(
                key=TRAINING_INIT,
                label="training initialization checkpoint",
                configured=_absolute(
                    init_checkpoint,
                    field="training.init_checkpoint",
                    purpose="when provided",
                ),
                relative=Path("inputs/training_init_checkpoint.pt"),
                identity_path=("init_checkpoint_sha256",),
            )
        )
    if resume_checkpoint is not None:
        resume = _absolute(
            resume_checkpoint,
            field="training.resume_checkpoint",
            purpose="when provided",
        )
        best = _absolute(
            resume_best_checkpoint,
            field="training.resume_best_checkpoint",
            purpose="when provided",
        )
        if init_checkpoint is not None:
            raise ValueError(
                "training.resume_checkpoint is mutually exclusive with "
                "training.init_checkpoint and training.require_init_checkpoint"
            )
        inputs.extend(
            [
                PinnedInput(
                    key=TRAINING_RESUME,
                    label="training resume checkpoint",
                    configured=resume,
                    relative=Path("inputs/training_resume_checkpoint.pt"),
                    identity_path=("resume_checkpoint_sha256",),
                ),
                PinnedInput(
                    key=TRAINING_RESUME_BEST,
                    label="training resume best checkpoint",
                    configured=best,
                    relative=Path("inputs/training_resume_best_checkpoint.pt"),
                    identity_path=("resume_best_checkpoint_sha256",),
                ),
            ]
        )
    return inputs


def _evaluation_input(evaluation: Mapping[str, Any]) -> PinnedInput:
    checkpoint = evaluation.get("checkpoint")
    if checkpoint is None:
        raise ValueError(
            "evaluation.checkpoint must be an absolute GPFS path for an "
            "evaluation-only workflow"
        )
    return PinnedInput(
        key=EVALUATION,
        label="evaluation checkpoint",
        configured=_absolute(
            checkpoint,
            field="evaluation.checkpoint",
            purpose="for an evaluation-only workflow",
        ),
        relative=Path("inputs/evaluation_checkpoint.pt"),
        identity_path=("input_checkpoint_sha256",),
    )


def run_inputs(workflow_config: Mapping[str, Any]) -> tuple[PinnedInput, ...]:
    """Validate and return the inputs declared in the workflow config itself.

    These need no source tree, so template validation calls this directly.
    Campaign-declared inputs need the campaign files: see ``declared_inputs``.
    """
    kind = workflow_config["workflow"]["kind"]
    if kind == "train":
        return tuple(_training_inputs(workflow_config["training"]))
    if kind == "evaluate":
        return (_evaluation_input(workflow_config["evaluation"]),)
    return ()


def _campaign_paths(workflow_config: Mapping[str, Any]) -> list[str]:
    kind = workflow_config["workflow"]["kind"]
    if kind == "ablation":
        return [str(workflow_config["ablation"]["campaign"])]
    if kind == "ablation_suite":
        return [
            str(study["campaign"])
            for study in workflow_config["ablation_suite"]["studies"]
        ]
    return []


def declared_inputs(
    workflow_config: Mapping[str, Any], source_root: Path
) -> tuple[PinnedInput, ...]:
    """Validate and return every pinned input of a run, in a fixed order.

    ``source_root`` locates the campaign files of ablation workflows. Campaign
    declaration errors (unknown or unused names, relative paths) raise here.
    """
    inputs = list(run_inputs(workflow_config))
    for campaign in _campaign_paths(workflow_config):
        relative = Path(campaign)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"campaign must be a safe relative path: {campaign!r}")
        campaign_id, declared = read_campaign_init_checkpoints(source_root / relative)
        inputs.extend(
            campaign_init_input(campaign_id, name, path)
            for name, path in declared.items()
        )
    return tuple(inputs)


def bind_dirs(inputs: Iterable[PinnedInput]) -> list[Path]:
    """Source directories the contained coordinator must bind for prepare."""
    return sorted({pinned.configured.parent for pinned in inputs})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(pinned: PinnedInput, run_root: Path) -> dict[str, str]:
    configured = pinned.configured
    if configured.is_symlink():
        raise ValueError(f"Input {pinned.label} must not be a symlink: {configured}")
    resolved = configured.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(
            f"Input {pinned.label} must resolve to a regular file: {configured}"
        )
    sha256 = _sha256(resolved)
    target = run_root / pinned.relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise ValueError(
                f"Pinned {pinned.label} must be a regular non-symlink file: {target}"
            )
        if _sha256(target) != sha256:
            raise ValueError(
                f"Pinned {pinned.label} conflicts with configured input: {target}"
            )
    else:
        partial = target.with_suffix(".pt.partial")
        shutil.copyfile(resolved, partial)
        if _sha256(partial) != sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f"Input {pinned.label} changed while being pinned")
        partial.chmod(0o444)
        partial.replace(target)
    return {
        "configured_checkpoint": str(configured),
        "resolved_checkpoint": str(resolved),
        "pinned_checkpoint": str(target),
        "pinned_checkpoint_relative": str(pinned.relative),
        "checkpoint_sha256": sha256,
    }


def pin_inputs(inputs: Iterable[PinnedInput], run_root: Path) -> PinnedRecords:
    """Copy each input into the run read-only and record its SHA-256.

    Re-pinning the same content is a no-op; different content at a pinned
    location is rejected.
    """
    records: dict[str, dict[str, str]] = {}
    identity: dict[str, Any] = {}
    for pinned in inputs:
        record = _pin(pinned, run_root)
        records[pinned.key] = record
        cursor = identity
        for part in pinned.identity_path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[pinned.identity_path[-1]] = record["checkpoint_sha256"]
    return PinnedRecords(records=records, identity=identity)


def _schema1_records(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    if manifest.get("training_input") is not None:
        records[TRAINING_INIT] = manifest["training_input"]
    resume = manifest.get("training_resume_input")
    if isinstance(resume, Mapping):
        records[TRAINING_RESUME] = resume.get("resume")
        records[TRAINING_RESUME_BEST] = resume.get("best")
    if manifest.get("evaluation_input") is not None:
        records[EVALUATION] = manifest["evaluation_input"]
    campaigns = manifest.get("campaign_init_inputs")
    if isinstance(campaigns, Mapping):
        for campaign_id, named in campaigns.items():
            if isinstance(named, Mapping):
                for name, record in named.items():
                    records[f"campaign_init/{campaign_id}/{name}"] = record
    return records


def recorded_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Pinned-input records of a run manifest by key, for schema 1 and 2.

    Runs are immutable, so manifests written before schema 2 keep their
    kind-specific sections; analyses of those runs read them through here.
    """
    version = manifest.get("schema_version")
    if version == RUN_MANIFEST_SCHEMA_VERSION:
        section = manifest.get(MANIFEST_SECTION, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"Run manifest {MANIFEST_SECTION} must be a mapping")
        return dict(section)
    if version == 1:
        return _schema1_records(manifest)
    raise ValueError(f"Unsupported run manifest schema_version: {version!r}")


def verify_file(pinned: PinnedFile, *, label: str) -> None:
    """Re-hash a pinned copy just before it is used."""
    path = pinned.path
    if not path.is_file() or path.is_symlink() or _sha256(path) != pinned.sha256:
        raise ValueError(f"Pinned {label} failed validation: {path}")


def verified_input(manifest_path: Path, pinned: PinnedInput) -> PinnedFile:
    """Check the manifest record of one declared input and its pinned copy.

    The record must name the declared source and the layout location, and the
    copy must be a regular file with the recorded SHA-256. Paths are anchored
    at the manifest's directory, which is the in-container run root for
    containerized stages.
    """
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Run manifest must be a mapping: {manifest_path}")
    record = recorded_inputs(manifest).get(pinned.key)
    if not isinstance(record, Mapping):
        raise ValueError(f"Run manifest does not pin the {pinned.label}")
    if record.get("configured_checkpoint") != str(pinned.configured):
        raise ValueError(f"Pinned {pinned.label} does not match its declaration")
    if record.get("pinned_checkpoint_relative") != str(pinned.relative):
        raise ValueError(f"Pinned {pinned.label} is not at {pinned.relative}")
    sha256 = record.get("checkpoint_sha256")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError(f"Pinned {pinned.label} has an invalid recorded SHA-256")
    file = PinnedFile(path=manifest_path.parent / pinned.relative, sha256=sha256)
    verify_file(file, label=pinned.label)
    return file
