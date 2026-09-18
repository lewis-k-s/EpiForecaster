"""High-incidence event outputs for frozen research calibration controls."""

from __future__ import annotations

import pandas as pd

from evaluation.forecast_events import (
    event_metrics,
    exceedance_probability,
    fit_decision_threshold,
    fit_event_definition,
)
from evaluation.research_calibration import observed_role_forecasts


def score_research_events(
    fit_cases: pd.DataFrame,
    calibration: pd.DataFrame,
    scored: pd.DataFrame,
    artifact: dict,
    manifest: dict,
    *,
    role: str = "selection",
    min_support: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Fit native municipality case thresholds; score on a separate role.

    Metrics count eligible forecast instances. Unique municipality/date counts
    accompany them; repeated windows are not independent source observations.
    """
    if role not in {"selection", "confirmation"}:
        raise ValueError("Event evidence requires selection or confirmation forecasts")
    fit_dates = pd.to_datetime(fit_cases.target_date, errors="raise")
    bounds = manifest["roles"]["fit"]
    if (
        fit_dates.isna().any()
        or not fit_dates.between(bounds["label_start"], bounds["label_end"]).all()
    ):
        raise ValueError("Event threshold data exceed the fit role's dates")
    cal = observed_role_forecasts(calibration, manifest, role="calibration")
    cal = cal.loc[cal.target.eq("cases")]
    rows = []
    summary = {
        "schema_version": 1,
        "role": role,
        "definition": "high_incidence_exceedance_research_proxy",
        "metric_unit": "eligible_forecast_instance",
        "calibration_artifact_sha256": artifact["artifact_sha256"],
        "hospitalizations": {
            "status": "unsupported",
            "reason": "source_aggregation_not_identified",
        },
        "definitions": {},
    }
    fitted_head = artifact["heads"]["cases"]
    for quantile in [0.9, 0.95]:
        definition = fit_event_definition(fit_cases, head="cases", quantile=quantile)
        name = f"q{int(100 * quantile)}"
        result = {"event_definition": definition.to_dict(), "controls": {}}
        summary["definitions"][name] = result
        labels_cal = cal.observed.to_numpy() > definition.threshold
        positive_unique = len(
            cal.loc[labels_cal, ["region_id", "target_date"]].drop_duplicates()
        )
        for control in ["C0", "C1", "C2"]:
            selected = scored.loc[scored.control.eq(control)].copy()
            selected = observed_role_forecasts(selected, manifest, role=role)
            selected = selected.loc[selected.target.eq("cases")].copy()
            if selected.calibrated_scale.isna().any():
                raise ValueError("Missing canonical calibrated scale")
            multiplier = fitted_head[control]["multiplier"]
            p_cal = exceedance_probability(
                cal.distribution_loc.to_numpy(),
                cal.distribution_scale.to_numpy() * multiplier,
                definition.threshold,
                family=fitted_head["family"],
                df=fitted_head["df"],
            )
            if positive_unique < min_support:
                operating = {
                    "threshold": None,
                    "reason": "insufficient_unique_positive_support",
                }
            else:
                operating = fit_decision_threshold(
                    p_cal, labels_cal, min_alerts=min_support, min_positives=min_support
                )
                if operating["threshold"] is not None:
                    unique_alerts = len(
                        cal.loc[
                            p_cal >= operating["threshold"],
                            ["region_id", "target_date"],
                        ].drop_duplicates()
                    )
                    operating["calibration_unique_alerts"] = unique_alerts
                    if unique_alerts < min_support:
                        operating.update(
                            threshold=None, reason="insufficient_unique_alert_support"
                        )
            operating["calibration_unique_positive_count"] = positive_unique
            selected["event_probability"] = exceedance_probability(
                selected.distribution_loc.to_numpy(),
                selected.calibrated_scale.to_numpy(),
                definition.threshold,
                family=fitted_head["family"],
                df=fitted_head["df"],
            )
            selected["event_label"] = selected.observed > definition.threshold
            selected["event_definition"] = name
            selected["value_threshold"] = definition.threshold
            selected["operating_probability_threshold"] = operating["threshold"]
            bands = {}
            for band, low, high in [
                ("all", 1, 28),
                ("days_1_7", 1, 7),
                ("days_8_14", 8, 14),
                ("days_15_21", 15, 21),
                ("days_22_28", 22, 28),
            ]:
                group = selected.loc[selected.horizon.between(low, high)]
                p, y = group.event_probability.to_numpy(), group.event_label.to_numpy()
                bands[band] = {
                    "unique_region_dates": len(
                        group[["region_id", "target_date"]].drop_duplicates()
                    ),
                    "fixed_0_5": event_metrics(p, y),
                    "operating": event_metrics(
                        p, y, threshold=float(operating["threshold"])
                    )
                    if operating["threshold"] is not None
                    else None,
                }
            result["controls"][control] = {
                "operating_rule": operating,
                "horizon_bands": bands,
            }
            rows.append(selected)
    return pd.concat(rows, ignore_index=True), summary
