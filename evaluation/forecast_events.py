"""Canonical high-incidence proxy definitions, probabilities, and alert scores.

Inputs must come from the declared data roles. These are retrospective research
events, not clinical outbreak diagnoses. Unsupported ratios remain undefined.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import average_precision_score

from evaluation.distribution_scores import WIS_INTERVAL_LEVELS, standard_wis_half_widths


@dataclass(frozen=True)
class EventDefinition:
    head: str
    threshold: float
    quantile: float
    unique_source_observations: int
    fit_identity_sha256: str
    units: str = "log1p_per_100k"
    quantile_method: str = "higher"
    positive_if: str = "strictly_greater"
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def fit_event_definition(
    frame: pd.DataFrame, *, head: str, quantile: float = 0.9
) -> EventDefinition:
    """Fit from unique, observed fit-role source/date values only."""
    required = {
        "role",
        "target",
        "source_id",
        "target_date",
        "observed",
        "observed_mask",
    }
    if required - set(frame.columns):
        raise ValueError(
            f"Event fit data lack columns: {sorted(required - set(frame.columns))}"
        )
    if frame.empty or not frame["role"].eq("fit").all():
        raise ValueError("Event thresholds require nonempty fit-role data only")
    if not 0 < quantile < 1:
        raise ValueError("Event quantile must be between zero and one")
    selected = frame.loc[
        frame["target"].eq(head) & frame["observed_mask"].eq(True),
        ["source_id", "target_date", "observed"],
    ].copy()
    if selected.empty or selected[["source_id", "target_date"]].isna().any().any():
        raise ValueError("Observed event fit data require source/date identities")
    if selected["source_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Source identities cannot be empty")
    if not np.isfinite(selected["observed"].to_numpy(dtype=float)).all():
        raise ValueError("Observed event fit values must be finite")
    counts = selected.groupby(["source_id", "target_date"])["observed"].nunique()
    if counts.gt(1).any():
        raise ValueError(
            "Conflicting values for one source/date; resolve source units first"
        )
    unique = selected.drop_duplicates(["source_id", "target_date"]).sort_values(
        ["source_id", "target_date"]
    )
    encoded = unique.to_json(orient="records", date_format="iso", double_precision=15)
    return EventDefinition(
        head=head,
        threshold=float(
            np.quantile(
                unique["observed"].to_numpy(dtype=float), quantile, method="higher"
            )
        ),
        quantile=quantile,
        unique_source_observations=len(unique),
        fit_identity_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
    )


def exceedance_probability(
    loc: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    *,
    family: str,
    df: float = 5.0,
) -> np.ndarray:
    loc, scale = np.broadcast_arrays(
        np.asarray(loc, dtype=float), np.asarray(scale, dtype=float)
    )
    if (
        not np.isfinite(loc).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(threshold)
    ):
        raise ValueError(
            "Event probabilities require finite location, threshold, and positive scale"
        )
    z = (threshold - loc) / scale
    if family == "normal":
        return stats.norm.sf(z)
    if family == "student_t" and np.isfinite(df) and df > 2:
        return stats.t.sf(z, df)
    raise ValueError("Unsupported event distribution or degrees of freedom")


def _event_inputs(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError(
            "Event inputs must be aligned finite one-dimensional probabilities"
        )
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Event labels must be binary and observed")
    return p, y.astype(bool)


def event_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, threshold: float = 0.5
) -> dict:
    """Score a fixed rule. AP is not trapezoidal precision-recall area."""
    p, y = _event_inputs(probabilities, labels)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must lie in [0, 1]")
    predicted = p >= threshold
    tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
    fn, tn = int((~predicted & y).sum()), int((~predicted & ~y).sum())
    n = len(y)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    def fscore(beta: float) -> float | None:
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp
        return (1 + beta**2) * tp / denominator if denominator else None

    bins = []
    bucket = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        selected = bucket == i
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": int(selected.sum()),
                "mean_probability": float(p[selected].mean())
                if selected.any()
                else None,
                "event_frequency": float(y[selected].mean())
                if selected.any()
                else None,
            }
        )
    reasons = {}
    if not n:
        reasons["all"] = "no_eligible_observations"
    if not y.any():
        reasons.update(
            recall="no_positive_events", average_precision="no_positive_events"
        )
    if not predicted.any():
        reasons["precision"] = "no_predicted_events"
    return {
        "count": n,
        "positive_count": tp + fn,
        "alert_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "decision_threshold": threshold,
        "prevalence": float(y.mean()) if n else None,
        "brier": float(np.mean((p - y) ** 2)) if n else None,
        "average_precision": float(average_precision_score(y, p)) if y.any() else None,
        "precision": precision,
        "recall": recall,
        "f1": fscore(1),
        "f2": fscore(2),
        "false_alerts_per_100": 100 * fp / n if n else None,
        "reliability_bins": bins,
        "undefined_reasons": reasons,
    }


def fit_decision_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    precision_floor: float = 0.5,
    min_alerts: int = 20,
    min_positives: int = 20,
) -> dict:
    """Maximize calibration recall subject to precision/support; freeze afterward."""
    p, y = _event_inputs(probabilities, labels)
    if not 0 < precision_floor <= 1 or min_alerts < 1 or min_positives < 1:
        raise ValueError("Invalid operating-point constraints")
    if y.sum() < min_positives:
        return {"threshold": None, "reason": "insufficient_positive_support"}
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p) - 1]
    positives = np.cumsum(sorted_y)[ends]
    alerts = ends + 1
    eligible = (alerts >= min_alerts) & (positives / alerts >= precision_floor)
    if not eligible.any():
        return {"threshold": None, "reason": "no_feasible_operating_point"}
    # Ties: highest precision, then highest probability threshold.
    candidates = np.flatnonzero(eligible)
    best = max(
        candidates,
        key=lambda i: (int(positives[i]), positives[i] / alerts[i], sorted_p[ends[i]]),
    )
    return {
        "threshold": float(sorted_p[ends[best]]),
        "reason": None,
        "calibration_recall": float(positives[best] / y.sum()),
        "calibration_precision": float(positives[best] / alerts[best]),
        "calibration_alert_count": int(alerts[best]),
    }


def fit_scale_multiplier(
    loc: np.ndarray,
    scale: np.ndarray,
    target: np.ndarray,
    *,
    family: str,
    df: float = 5.0,
    objective: str = "coverage_80",
    bounds: tuple[float, float] = (0.25, 4.0),
) -> dict:
    """Fit a bounded calibration scalar. Call only on the calibration role."""
    loc, scale, target = (np.asarray(x, dtype=float) for x in (loc, scale, target))
    if (
        loc.ndim != 1
        or not loc.size
        or scale.shape != loc.shape
        or target.shape != loc.shape
        or not all(np.isfinite(x).all() for x in (loc, scale, target))
        or (scale <= 0).any()
    ):
        raise ValueError("Scale fitting requires aligned finite observed forecasts")
    if not 0 < bounds[0] < bounds[1] or not np.isfinite(bounds).all():
        raise ValueError("Scale multiplier bounds must be finite and positive")
    quantiles = np.asarray(standard_wis_half_widths(family, df))
    if objective == "coverage_80":
        standardized = np.abs(target - loc) / scale
        raw = float(
            np.quantile(standardized, 0.8, method="higher")
            / quantiles[WIS_INTERVAL_LEVELS.index(0.8)]
        )
        multiplier = float(np.clip(raw, *bounds))
    elif objective == "wis":
        alpha = 1 - np.asarray(WIS_INTERVAL_LEVELS)
        errors = np.abs(target - loc)

        def score(multiplier: float) -> float:
            width = multiplier * scale[:, None] * quantiles
            return float(
                np.mean(
                    (
                        0.5 * errors
                        + (alpha * width + np.maximum(errors[:, None] - width, 0)).sum(
                            1
                        )
                    )
                    / (len(quantiles) + 0.5)
                )
            )

        fit = optimize.minimize_scalar(score, bounds=bounds, method="bounded")
        if not fit.success:
            raise ValueError("WIS scale fitting did not converge")
        multiplier = min((bounds[0], bounds[1], float(fit.x)), key=score)
    else:
        raise ValueError("Unknown calibration objective")
    identity = hashlib.sha256(
        json.dumps(
            {"loc": loc.tolist(), "scale": scale.tolist(), "target": target.tolist()},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return {
        "multiplier": multiplier,
        "objective": objective,
        "count": len(target),
        "bounds": list(bounds),
        "at_bound": bool(np.isclose(multiplier, bounds).any()),
        "fit_values_sha256": identity,
        "family": family,
        "df": df,
    }
