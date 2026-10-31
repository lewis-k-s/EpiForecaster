"""Matched loss-screen evidence with conservative temporal support checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.research_calibration import HEADS, KEYS


def compare_scored_forecasts(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    block_days: int = 28,
    minimum_blocks: int = 4,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 852,
) -> dict:
    """Compare one matched fold/seed and one calibration control per input.

    All regions and heads for a target date stay together. Fewer than four
    disjoint temporal blocks provide point evidence only. This screening helper
    never promotes a model; event gates and final confirmation are separate.
    """
    if block_days < 28 or minimum_blocks < 4 or bootstrap_draws < 100:
        raise ValueError("Temporal evidence requires >=28-day blocks and >=4 blocks")
    keys = KEYS + ["target_date"]
    metrics = ["wis", "absolute_error", "covered_80", "covered_95"]
    for frame in [baseline, candidate]:
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("Paired evidence requires unique nonempty forecasts")
        if frame["control"].nunique() != 1 or frame["role"].nunique() != 1:
            raise ValueError("Compare one control and role at a time")
        if set(frame.target) != set(HEADS):
            raise ValueError("All heads require paired evidence")
        if not frame.observed_mask.eq(True).all():
            raise ValueError("Only observed canonical scores are eligible")
        if not np.isfinite(frame[metrics + ["observed"]].to_numpy(dtype=float)).all():
            raise ValueError("Paired scores must be finite")
        if not frame[["covered_80", "covered_95"]].isin([True, False]).all().all():
            raise ValueError("Canonical coverage must be Boolean")
    if baseline.role.iloc[0] != candidate.role.iloc[0] or baseline.role.iloc[0] not in {
        "selection",
        "confirmation",
    }:
        raise ValueError(
            "Paired evidence requires the same selection/confirmation role"
        )
    joined = baseline.merge(
        candidate,
        on=keys,
        suffixes=("_base", "_candidate"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Candidate and baseline must use exactly matched support")
    if not np.array_equal(
        joined.observed_base.to_numpy(), joined.observed_candidate.to_numpy()
    ):
        raise ValueError("Matched observations differ between candidate and baseline")
    if (joined.wis_base < 0).any() or (joined.wis_candidate < 0).any():
        raise ValueError("WIS must be nonnegative")
    heads = {}
    for target, group in joined.groupby("target"):
        heads[target] = {
            "count": len(group),
            "baseline_wis": float(group.wis_base.mean()),
            "candidate_wis": float(group.wis_candidate.mean()),
            "baseline_mae": float(group.absolute_error_base.mean()),
            "candidate_mae": float(group.absolute_error_candidate.mean()),
            "candidate_coverage_80": float(group.covered_80_candidate.mean()),
            "candidate_coverage_95": float(group.covered_95_candidate.mean()),
        }
    baseline_wis = float(np.mean([value["baseline_wis"] for value in heads.values()]))
    candidate_wis = float(np.mean([value["candidate_wis"] for value in heads.values()]))
    difference = candidate_wis - baseline_wis
    dates = pd.to_datetime(joined.target_date, errors="raise")
    if dates.isna().any():
        raise ValueError("Target dates must be complete")
    joined["block"] = (dates - dates.min()).dt.days // block_days
    # Sum/count preserves the original equal-head mean under block resampling.
    grouped = joined.groupby(["block", "target"])
    blocks = sorted(joined.block.unique())
    sums = grouped[["wis_base", "wis_candidate"]].sum()
    counts = grouped.size()
    support = all(
        (block, target) in counts.index for block in blocks for target in HEADS
    )
    interval = None
    reason = None
    if len(blocks) < minimum_blocks:
        reason = "insufficient_temporal_blocks"
    elif not support:
        reason = "missing_head_support_in_temporal_blocks"
    else:
        totals = np.array(
            [
                [np.asarray(sums.loc[(block, head)], dtype=float) for head in HEADS]
                for block in blocks
            ]
        )
        sizes = np.array(
            [[counts.loc[(block, head)] for head in HEADS] for block in blocks]
        )
        rng = np.random.default_rng(bootstrap_seed)
        selected = rng.integers(0, len(blocks), size=(bootstrap_draws, len(blocks)))
        scores = totals[selected].sum(axis=1) / sizes[selected].sum(axis=1)[..., None]
        differences = (scores[..., 1] - scores[..., 0]).mean(axis=1)
        interval = np.quantile(differences, [0.025, 0.975]).tolist()
    return {
        "schema_version": 1,
        "role": str(baseline.role.iloc[0]),
        "baseline_control": str(baseline.control.iloc[0]),
        "candidate_control": str(candidate.control.iloc[0]),
        "heads": heads,
        "baseline_mean_target_wis": baseline_wis,
        "candidate_mean_target_wis": candidate_wis,
        "wis_difference_candidate_minus_baseline": difference,
        "relative_wis_improvement": -difference / baseline_wis
        if baseline_wis >= 1e-6
        else None,
        "absolute_wis_improvement": -difference,
        "uncertainty": {
            "method": "paired_disjoint_target_date_block_bootstrap_all_regions_and_heads",
            "block_days": block_days,
            "blocks": len(blocks),
            "minimum_blocks": minimum_blocks,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "difference_interval_95": interval,
            "unsupported_reason": reason,
        },
        "promotion_status": "not_assessed_screening_evidence_only",
    }
