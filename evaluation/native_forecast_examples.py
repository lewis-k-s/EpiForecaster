"""Select reproducible qualitative examples from canonical native evaluation rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_native_forecast_examples(
    rows: pd.DataFrame,
    *,
    input_window_length: int,
    support: pd.DataFrame | None = None,
    source_cohort: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select low, middle, and high source WIS examples for each head.

    Source scores average persisted observation-level WIS across overlapping
    forecast windows. Select a middle origin among each source's most complete
    windows. This is a qualitative sample, not a new test-score estimate.
    """
    required = {"target", "source_type", "source_id", "window_start", "wis"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Missing native example fields: {sorted(missing)}")
    if input_window_length < 1:
        raise ValueError("input_window_length must be positive")
    if rows.empty or not np.isfinite(rows["wis"]).all():
        raise ValueError("Native examples require observed rows with finite WIS")
    if support is not None:
        rows = rows.merge(
            support[support.eligible],
            on=["target", "source_id", "window_start"],
            validate="many_to_one",
        )
        if rows.empty:
            raise ValueError("No windows meet the history and horizon support rules")
    keys = ["target", "source_type", "source_id"]
    scores = rows.groupby(keys, sort=True)["wis"].mean().reset_index()
    selections: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for target, sources in scores.groupby("target", sort=True):
        if source_cohort is None:
            sources = sources.sort_values(["wis", "source_type", "source_id"])
            ranks = [
                ("low", 0),
                ("middle", len(sources) // 2),
                ("high", len(sources) - 1),
            ]
        else:
            # Keep validation source identities and rank labels across periods.
            sources = source_cohort[source_cohort.target == target].rename(
                columns={"source_mean_wis": "wis"}
            )
            ranks = list(zip(sources.example_rank, range(len(sources)), strict=True))
        used: set[tuple[str, str]] = set()
        for label, rank in ranks:
            source = sources.iloc[rank]
            identity = source.source_type, source.source_id
            if identity in used:
                continue
            used.add(identity)
            candidates = rows[
                (rows.target == target)
                & (rows.source_type == source.source_type)
                & (rows.source_id == source.source_id)
            ]
            if candidates.empty:
                continue
            counts = candidates.groupby("window_start").size()
            origins = sorted(counts[counts == counts.max()].index)
            if support is not None:
                complete = candidates[candidates.window_start.isin(origins)]
                best_history = complete.history_uncensored.max()
                origins = sorted(
                    complete[
                        complete.history_uncensored == best_history
                    ].window_start.unique()
                )
            window_start = int(origins[len(origins) // 2])
            selected = candidates[candidates.window_start == window_start].copy()
            selected["example_rank"] = label
            selected["origin_index"] = window_start + input_window_length - 1
            selected_rows.append(selected)
            selections.append(
                {
                    "target": target,
                    "source_type": source.source_type,
                    "source_id": source.source_id,
                    "example_rank": label,
                    "source_mean_wis": float(source.wis),
                    "window_start": window_start,
                    "origin_index": window_start + input_window_length - 1,
                    "observed_points": len(selected),
                }
            )
            if support is not None:
                selections[-1].update(
                    {
                        name: selected[name].iloc[0]
                        for name in (
                            "history_observed",
                            "history_uncensored",
                            "history_nonzero",
                            "forecast_nonzero",
                            "origin_date",
                        )
                    }
                )
    columns = [
        "target",
        "source_type",
        "source_id",
        "example_rank",
        "source_mean_wis",
        "window_start",
        "origin_index",
        "observed_points",
    ]
    selected = (
        pd.concat(selected_rows, ignore_index=True) if selected_rows else rows.iloc[:0]
    )
    return selected, pd.DataFrame(
        selections, columns=columns if not selections else None
    )
