"""Role-checked calibration artifacts for canonical granular forecasts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats

from evaluation.distribution_scores import (
    WIS_INTERVAL_LEVELS,
    distribution_wis,
    normal_crps,
    standard_wis_half_widths,
    student_t_crps,
)
from evaluation.forecast_events import fit_scale_multiplier

HEADS = ("wastewater", "hospitalizations", "cases", "deaths")
KEYS = ["target", "region_id", "window_start", "horizon"]


def _digest(value: dict | list) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def observed_role_forecasts(
    frame: pd.DataFrame, manifest: dict, *, role: str, horizon: int = 28
) -> pd.DataFrame:
    """Validate whole forecast windows, then select observed target rows."""
    required = set(KEYS) | {
        "role",
        "target_date",
        "observed",
        "observed_mask",
        "observation_weight",
        "distribution_loc",
        "distribution_scale",
        "distribution_family",
        "distribution_df",
    }
    if required - set(frame.columns):
        raise ValueError(
            f"Missing granular fields: {sorted(required - set(frame.columns))}"
        )
    if frame.empty or not frame.role.eq(role).all() or role not in manifest["roles"]:
        raise ValueError("Forecasts do not belong to the requested data role")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Forecast identities must be complete and unique")
    leads = frame.horizon.to_numpy(dtype=float)
    if (
        not np.isfinite(leads).all()
        or not ((leads >= 1) & (leads <= horizon) & (leads == np.floor(leads))).all()
    ):
        raise ValueError("Invalid forecast lead")
    dates = pd.to_datetime(frame.target_date, errors="raise")
    origins = dates - pd.to_timedelta(leads - 1, unit="D")
    bounds = manifest["roles"][role]
    if (
        dates.isna().any()
        or not (
            (origins >= pd.Timestamp(bounds["label_start"]))
            & (
                origins + pd.Timedelta(days=horizon - 1)
                <= pd.Timestamp(bounds["label_end"])
            )
        ).all()
    ):
        raise ValueError("Forecast label window crosses the declared role boundary")
    if not frame.observed_mask.isin([True, False]).all():
        raise ValueError("Observation masks must be Boolean")
    observed = frame.loc[frame.observed_mask.eq(True)].sort_values(KEYS).copy()
    if set(observed.target.unique()) != set(HEADS):
        raise ValueError("All four heads require observed support")
    values = observed[["observed", "distribution_loc", "distribution_scale"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (observed.distribution_scale <= 0).any():
        raise ValueError(
            "Observed probabilistic forecasts must be finite with positive scale"
        )
    if not np.allclose(observed.observation_weight.to_numpy(dtype=float), 1.0):
        raise ValueError("Research calibration requires binary observed-only weights")
    for _, head in observed.groupby("target"):
        if (
            head.distribution_family.nunique(dropna=False) != 1
            or head.distribution_df.nunique(dropna=False) != 1
        ):
            raise ValueError("A head cannot mix distribution identities")
        standard_wis_half_widths(
            str(head.distribution_family.iloc[0]), float(head.distribution_df.iloc[0])
        )
    return observed


def fit_research_calibrators(
    frame: pd.DataFrame, manifest: dict, *, checkpoint_sha256: str
) -> dict:
    observed = observed_role_forecasts(frame, manifest, role="calibration")
    if len(checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in checkpoint_sha256
    ):
        raise ValueError("A checkpoint SHA-256 identity is required")
    artifact = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "role_manifest_sha256": _digest(manifest),
        "fit_role": "calibration",
        "forecast_identity_sha256": _digest(
            observed[KEYS + ["target_date"]].astype(str).to_dict("records")
        ),
        "heads": {},
    }
    for target, head in observed.groupby("target"):
        family = str(head.distribution_family.iloc[0])
        df = float(head.distribution_df.iloc[0])
        args = [
            head[c].to_numpy(dtype=float)
            for c in ["distribution_loc", "distribution_scale", "observed"]
        ]
        artifact["heads"][target] = {
            "family": family,
            "df": df,
            "count": len(head),
            "unique_region_dates": len(
                head[["region_id", "target_date"]].drop_duplicates()
            ),
            "C0": {"multiplier": 1.0, "objective": "identity"},
            "C1": fit_scale_multiplier(
                *args, family=family, df=df, objective="coverage_80"
            ),
            "C2": fit_scale_multiplier(*args, family=family, df=df, objective="wis"),
        }
    artifact["artifact_sha256"] = _digest(artifact)
    return artifact


def score_research_calibrators(
    frame: pd.DataFrame,
    artifact: dict,
    manifest: dict,
    *,
    role: str,
    checkpoint_sha256: str,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate frozen controls on selection or frozen confirmation forecasts.

    Returned rows are canonical metric outputs. Reporting code must use these
    values rather than refit calibration or recompute metrics.
    """
    if role not in {"selection", "confirmation"}:
        raise ValueError("Calibration fit data cannot supply evaluation evidence")
    expected = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    if artifact.get("schema_version") != 1 or artifact.get(
        "artifact_sha256"
    ) != _digest(expected):
        raise ValueError("Calibration artifact integrity check failed")
    if artifact["checkpoint_sha256"] != checkpoint_sha256 or artifact[
        "role_manifest_sha256"
    ] != _digest(manifest):
        raise ValueError("Calibration artifact checkpoint or protocol mismatch")
    observed = observed_role_forecasts(frame, manifest, role=role)
    rows = []
    summary = {
        "schema_version": 1,
        "role": role,
        "calibration_artifact_sha256": artifact["artifact_sha256"],
        "controls": {},
    }
    for control in ["C0", "C1", "C2"]:
        summary["controls"][control] = {"heads": {}}
        for target, head in observed.groupby("target"):
            fitted = artifact["heads"][target]
            family, df = (
                str(head.distribution_family.iloc[0]),
                float(head.distribution_df.iloc[0]),
            )
            if family != fitted["family"] or df != fitted["df"]:
                raise ValueError(
                    "Evaluation and calibration distribution identities differ"
                )
            multiplier = float(fitted[control]["multiplier"])
            loc, scale, y = [
                torch.as_tensor(head[c].to_numpy(dtype=float))
                for c in ["distribution_loc", "distribution_scale", "observed"]
            ]
            scale = scale * multiplier
            widths = torch.as_tensor(standard_wis_half_widths(family, df))
            scored = head.copy()
            scored["control"] = control
            scored["calibrated_scale"] = scale.numpy()
            scored["wis"] = distribution_wis(loc, scale, y, widths).numpy()
            scored["crps"] = (
                normal_crps(loc, scale, y)
                if family == "normal"
                else student_t_crps(loc, scale, y, df)
            ).numpy()
            residual = ((y - loc) / scale).numpy()
            distribution = stats.norm if family == "normal" else stats.t(df)
            scored["pit"] = distribution.cdf(residual)
            scored["nll"] = -distribution.logpdf(residual) + np.log(scale.numpy())
            scored["absolute_error"] = np.abs((y - loc).numpy())
            scored["target_units"] = (
                "log1p_concentration" if target == "wastewater" else "log1p_per_100k"
            )
            scored["forecast_origin_date"] = (
                pd.to_datetime(scored.target_date)
                - pd.to_timedelta(scored.horizon, unit="D")
            ).dt.strftime("%Y-%m-%d")
            quantiles = sorted(
                {0.5}
                | {round((1 - level) / 2, 8) for level in WIS_INTERVAL_LEVELS}
                | {round((1 + level) / 2, 8) for level in WIS_INTERVAL_LEVELS}
            )
            for quantile in quantiles:
                scored[f"quantile_{round(1000 * quantile):03d}"] = (
                    loc.numpy() + scale.numpy() * float(distribution.ppf(quantile))
                )
            metrics = {
                key: float(scored[key].mean())
                for key in ["wis", "crps", "nll", "absolute_error"]
            }
            metrics["count"] = len(scored)
            metrics["unique_region_dates"] = len(
                scored[["region_id", "target_date"]].drop_duplicates()
            )
            for level in [50, 80, 95]:
                half_width = scale.numpy() * float(
                    distribution.ppf((1 + level / 100) / 2)
                )
                scored[f"covered_{level}"] = scored.absolute_error <= half_width
                scored[f"width_{level}"] = 2 * half_width
                metrics[f"coverage_{level}"] = float(scored[f"covered_{level}"].mean())
                metrics[f"width_{level}"] = float(scored[f"width_{level}"].mean())
            summary["controls"][control]["heads"][target] = metrics
            rows.append(scored)
        summary["controls"][control]["mean_target_wis"] = float(
            np.mean([m["wis"] for m in summary["controls"][control]["heads"].values()])
        )
    return pd.concat(rows, ignore_index=True), summary
