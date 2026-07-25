"""Durable, versioned training-dynamics artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

DYNAMICS_SCHEMA_VERSION = 1
DYNAMICS_ARTIFACT_NAME = "training_dynamics.jsonl"
DYNAMICS_MANIFEST_NAME = "training_dynamics_manifest.json"
DYNAMICS_TERMINAL_STATUSES = {
    "complete",
    "pruned",
    "failed",
    "nonfinite_halt",
}
_RESERVED_ROW_FIELDS = {
    "schema_version",
    "step",
    "epoch",
    "nonfinite_fields",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_dynamics_identity(model_root: Path) -> dict[str, Any]:
    """Validate and return a model root's training-dynamics identity."""
    model_root = model_root.resolve()
    manifest_path = model_root / DYNAMICS_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(
            f"Training dynamics manifest is missing or unsafe: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DYNAMICS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported training dynamics schema: {manifest.get('schema_version')!r}"
        )
    status = str(manifest.get("status"))
    if status not in DYNAMICS_TERMINAL_STATUSES:
        raise ValueError(f"Training dynamics are not terminal: {status!r}")

    artifact_path = model_root / str(manifest.get("artifact", ""))
    if (
        artifact_path.parent.resolve() != model_root
        or not artifact_path.is_file()
        or artifact_path.is_symlink()
    ):
        raise FileNotFoundError(
            f"Training dynamics artifact is missing or unsafe: {artifact_path}"
        )
    artifact_sha256 = sha256_file(artifact_path)
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError(
            "Training dynamics artifact SHA-256 mismatch: "
            f"{artifact_path}"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "status": status,
        "row_count": int(manifest.get("row_count", 0)),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _scalar_value(value: Any) -> bool | int | float | None:
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                "Training dynamics only support scalar tensors, "
                f"got shape {tuple(value.shape)}"
            )
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return int(value) if isinstance(value, int) else float(value)
    raise TypeError(
        "Training dynamics only support scalar numeric values, "
        f"got {type(value).__name__}"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    raise TypeError(
        "Training dynamics summary only supports JSON-compatible values, "
        f"got {type(value).__name__}"
    )


class TrainingDynamicsWriter:
    """Append training-step evidence and maintain its lifecycle manifest."""

    def __init__(
        self,
        model_root: Path,
        *,
        progress_log_frequency: int,
        grad_norm_log_frequency: int,
        mobility_enabled: bool,
        resume: bool = False,
    ) -> None:
        if progress_log_frequency < 1:
            raise ValueError("progress_log_frequency must be at least 1")
        if grad_norm_log_frequency < 1:
            raise ValueError("grad_norm_log_frequency must be at least 1")

        self.model_root = model_root.resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.model_root / DYNAMICS_ARTIFACT_NAME
        self.manifest_path = self.model_root / DYNAMICS_MANIFEST_NAME
        self.progress_log_frequency = progress_log_frequency
        self.grad_norm_log_frequency = grad_norm_log_frequency
        self.mobility_enabled = mobility_enabled
        self.started_at = _utc_now()
        self.row_count = 0
        self.first_step: int | None = None
        self.last_step: int | None = None

        if self.artifact_path.exists():
            if not resume:
                raise FileExistsError(
                    "Training dynamics artifact already exists; refuse to overwrite: "
                    f"{self.artifact_path}"
                )
            self._load_resume_state()
        elif resume:
            raise FileNotFoundError(
                "Cannot resume missing training dynamics artifact: "
                f"{self.artifact_path}"
            )
        else:
            if self.manifest_path.exists():
                raise FileExistsError(
                    "Training dynamics manifest already exists without its artifact; "
                    f"refuse to overwrite: {self.manifest_path}"
                )
            self.artifact_path.touch(exist_ok=False)

        self._write_manifest(status="running")

    def _load_resume_state(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Cannot resume without a training dynamics manifest: "
                f"{self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DYNAMICS_SCHEMA_VERSION:
            raise ValueError("Training dynamics schema does not match on resume")
        expected_identity = {
            "artifact": DYNAMICS_ARTIFACT_NAME,
            "progress_log_frequency": self.progress_log_frequency,
            "grad_norm_log_frequency": self.grad_norm_log_frequency,
            "features": {"mobility": self.mobility_enabled},
        }
        mismatches = {
            field: (manifest.get(field), expected)
            for field, expected in expected_identity.items()
            if manifest.get(field) != expected
        }
        if mismatches:
            raise ValueError(
                "Training dynamics identity does not match on resume: "
                f"{mismatches}"
            )
        if manifest.get("status") in DYNAMICS_TERMINAL_STATUSES:
            raise ValueError(
                "Cannot append to terminal training dynamics artifact: "
                f"{manifest.get('status')}"
            )

        with self.artifact_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows:
            steps = [int(row["step"]) for row in rows]
            if any(right <= left for left, right in zip(steps, steps[1:])):
                raise ValueError(
                    "Existing training dynamics steps are not strictly increasing"
                )
            self.row_count = len(rows)
            self.first_step = steps[0]
            self.last_step = steps[-1]
        if manifest.get("row_count") != self.row_count:
            raise ValueError(
                "Training dynamics row count does not match on resume: "
                f"manifest={manifest.get('row_count')!r}, artifact={self.row_count}"
            )
        self.started_at = str(manifest.get("started_at") or self.started_at)

    def append(
        self,
        *,
        step: int,
        epoch: int,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one normalized JSONL record."""
        if step < 0:
            raise ValueError(f"Training dynamics step must be non-negative: {step}")
        if self.last_step is not None and step <= self.last_step:
            raise ValueError(
                "Training dynamics steps must be strictly increasing: "
                f"last={self.last_step}, new={step}"
            )

        row: dict[str, Any] = {
            "schema_version": DYNAMICS_SCHEMA_VERSION,
            "step": step,
            "epoch": epoch,
        }
        nonfinite_fields: list[str] = []
        for key in sorted(values):
            if key in _RESERVED_ROW_FIELDS:
                raise ValueError(f"Reserved training dynamics field: {key}")
            scalar = _scalar_value(values[key])
            if isinstance(scalar, float) and not math.isfinite(scalar):
                row[key] = None
                nonfinite_fields.append(key)
            else:
                row[key] = scalar
        row["nonfinite_fields"] = nonfinite_fields

        with self.artifact_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")

        self.row_count += 1
        self.first_step = step if self.first_step is None else self.first_step
        self.last_step = step
        self._write_manifest(status="running")
        return row

    def finalize(
        self,
        status: str,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize the manifest while preserving partial artifacts."""
        if status not in DYNAMICS_TERMINAL_STATUSES:
            raise ValueError(f"Unsupported terminal dynamics status: {status}")
        return self._write_manifest(
            status=status,
            error_type=error_type,
            error_message=error_message,
            summary=summary,
        )

    def _write_manifest(
        self,
        *,
        status: str,
        error_type: str | None = None,
        error_message: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        terminal = status in DYNAMICS_TERMINAL_STATUSES
        manifest: dict[str, Any] = {
            "schema_version": DYNAMICS_SCHEMA_VERSION,
            "artifact": DYNAMICS_ARTIFACT_NAME,
            "status": status,
            "row_count": self.row_count,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "progress_log_frequency": self.progress_log_frequency,
            "grad_norm_log_frequency": self.grad_norm_log_frequency,
            "features": {"mobility": self.mobility_enabled},
            "started_at": self.started_at,
            "completed_at": _utc_now() if terminal else None,
            "artifact_sha256": (
                sha256_file(self.artifact_path)
                if terminal and self.artifact_path.is_file()
                else None
            ),
        }
        if error_type is not None:
            manifest["error"] = {
                "type": error_type,
                "message": error_message or "",
            }
        if terminal and summary is not None:
            manifest["summary"] = _json_safe(summary)
        _atomic_write_json(self.manifest_path, manifest)
        return manifest
