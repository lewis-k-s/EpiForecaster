from __future__ import annotations

from typing import Any

# Canonical key contract for compile-stable training step inputs.
COMPILED_BATCH_KEYS: tuple[str, ...] = (
    "HospHist",
    "DeathsHist",
    "CasesHist",
    "BioNode",
    "MobBatch",
    "Population",
    "TargetNode",
    "TargetRegionIndex",
    "TemporalCovariates",
    "WWTarget",
    "HospTarget",
    "CasesTarget",
    "DeathsTarget",
    "WWTargetMask",
    "HospTargetMask",
    "CasesTargetMask",
    "DeathsTargetMask",
)

# Same contract excluding non-tensor objects like MobBatch.
COMPILED_BATCH_TENSOR_KEYS: tuple[str, ...] = (
    "HospHist",
    "DeathsHist",
    "CasesHist",
    "BioNode",
    "Population",
    "TargetNode",
    "TargetRegionIndex",
    "TemporalCovariates",
    "WWTarget",
    "HospTarget",
    "CasesTarget",
    "DeathsTarget",
    "WWTargetMask",
    "HospTargetMask",
    "CasesTargetMask",
    "DeathsTargetMask",
)


def build_compiled_batch_view(batch_data: dict[str, Any]) -> dict[str, Any]:
    """Build a shape-stable, metadata-free view for compiled training paths."""
    return {key: batch_data[key] for key in COMPILED_BATCH_KEYS}
