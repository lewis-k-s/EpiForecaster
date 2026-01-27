"""Shared skill output envelope and formatting utilities."""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any

SKILL_VERSION = "1.0"


@dataclass
class SkillError:
    """Structured error information."""

    type: str
    message: str
    context: dict[str, Any] | None = None


@dataclass
class SkillMeta:
    """Metadata about skill execution."""

    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    input_path: str | None = None
    additional: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    """Envelope for machine-readable skill output."""

    ok: bool
    name: str
    version: str = SKILL_VERSION
    data: Any = None
    warnings: list[str] = field(default_factory=list)
    error: SkillError | None = None
    meta: SkillMeta | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "ok": self.ok,
            "name": self.name,
            "version": self.version,
            "data": self.data,
            "warnings": self.warnings,
        }

        if self.error:
            result["error"] = {
                "type": self.error.type,
                "message": self.error.message,
            }
            if self.error.context:
                result["error"]["context"] = self.error.context

        if self.meta:
            result["meta"] = {
                "latency_ms": self.meta.latency_ms,
                "timestamp": self.meta.timestamp,
            }
            if self.meta.input_path:
                result["meta"]["input_path"] = self.meta.input_path
            if self.meta.additional:
                result["meta"].update(self.meta.additional)

        return result


class SkillOutputBuilder:
    """Builder for constructing skill outputs with timing."""

    def __init__(self, skill_name: str, input_path: str | None = None):
        self.skill_name = skill_name
        self.input_path = input_path
        self.start_time = perf_counter()
        self.warnings: list[str] = []
        self.meta_additional: dict[str, Any] = {}

    def add_warning(self, warning: str) -> "SkillOutputBuilder":
        """Add a warning message."""
        self.warnings.append(warning)
        return self

    def add_meta(self, key: str, value: Any) -> "SkillOutputBuilder":
        """Add additional metadata."""
        self.meta_additional[key] = value
        return self

    def success(self, data: Any) -> SkillOutput:
        """Create a successful output."""
        latency_ms = (perf_counter() - self.start_time) * 1000

        return SkillOutput(
            ok=True,
            name=self.skill_name,
            data=data,
            warnings=self.warnings,
            meta=SkillMeta(
                latency_ms=latency_ms,
                input_path=self.input_path,
                additional=self.meta_additional,
            ),
        )

    def error(
        self, error_type: str, message: str, context: dict[str, Any] | None = None
    ) -> SkillOutput:
        """Create an error output."""
        latency_ms = (perf_counter() - self.start_time) * 1000

        return SkillOutput(
            ok=False,
            name=self.skill_name,
            data=None,
            warnings=self.warnings,
            error=SkillError(type=error_type, message=message, context=context),
            meta=SkillMeta(
                latency_ms=latency_ms,
                input_path=self.input_path,
                additional=self.meta_additional,
            ),
        )


def format_output(output: SkillOutput, indent: int = 2) -> str:
    """Format skill output as JSON."""
    return json.dumps(output.to_dict(), indent=indent)


def print_output(output: SkillOutput, indent: int = 2, file=sys.stdout) -> None:
    """Print skill output to file (stdout by default)."""
    print(format_output(output, indent), file=file)
    if not output.ok:
        sys.exit(1)
