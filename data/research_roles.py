"""Validate dated research roles before datasets can expose their labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROLE_ORDER = ("fit", "selection", "calibration", "confirmation")


def load_role_manifest(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported research role schema")
    if set(manifest.get("roles", {})) != set(ROLE_ORDER):
        raise ValueError("Research requires exactly four declared data roles")
    if manifest.get("source_audit_status") != "complete":
        raise ValueError("Research requires a completed source audit")
    return manifest, hashlib.sha256(raw).hexdigest()


def research_time_ranges(
    dates: np.ndarray, manifest: dict, *, input_window: int, horizon: int
) -> dict[str, tuple[int, int]]:
    """Map inclusive label dates to history-plus-label, end-exclusive ranges.

    Earlier role observations may supply history. No label window can cross a
    role boundary. At least one full forecast horizon separates label roles.
    """
    index = pd.DatetimeIndex(dates)
    if (
        input_window < 1
        or horizon < 1
        or index.empty
        or index.hasnans
        or not index.is_monotonic_increasing
        or index.has_duplicates
        or not (
            np.diff(index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
            == pd.Timedelta(days=1).value
        ).all()
    ):
        raise ValueError("Research roles require a complete ordered daily date grid")
    ranges = {}
    previous_end = None
    for role in ROLE_ORDER:
        bounds = manifest["roles"][role]
        start, end = (
            pd.Timestamp(bounds["label_start"]),
            pd.Timestamp(bounds["label_end"]),
        )
        if start not in index or end not in index or end < start:
            raise ValueError(f"Invalid or unavailable {role} dates")
        if previous_end is not None and (start - previous_end).days - 1 < horizon:
            raise ValueError(
                f"{role} needs a forecast-horizon gap after the prior role"
            )
        locations = index.get_indexer(pd.DatetimeIndex([start, end]))
        start_idx, end_idx = int(locations[0]), int(locations[1]) + 1
        if start_idx < input_window or end_idx - start_idx < horizon:
            raise ValueError(f"{role} lacks history or a full label horizon")
        ranges[role] = (start_idx - input_window, end_idx)
        previous_end = end
    return ranges


def validate_preprocessing_role(attrs: dict, manifest: dict) -> None:
    """Reject precomputed features whose fitted parameters used later labels."""
    raw = attrs.get("preprocessing_config_yaml")
    if not raw:
        raise ValueError("Research dataset lacks its preprocessing configuration")
    config = yaml.safe_load(raw)
    cutoff = config.get("smoothing", {}).get("parameter_fit_end_date")
    expected = manifest["roles"]["fit"]["label_end"]
    if cutoff is None or pd.Timestamp(cutoff) != pd.Timestamp(expected):
        raise ValueError("Preprocessing parameter fit cutoff must match the fit role")
    provenance = config.get("source_provenance", {})
    if provenance.get("causal_preprocessing_contract") != "loss-research-v1":
        raise ValueError("Research requires the audited causal preprocessing contract")
