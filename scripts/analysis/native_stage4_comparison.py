"""Compare matched native Stage-4 evaluation runs.

Native schema-v4 granular files are paired within fold and seed. Every direct
comparison uses the intersection of source, target date, and horizon identities.
The historical municipal study is exported as a separate reference table; its
scores are never merged with native scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.granular_export import GRANULAR_SCHEMA_VERSION

FACTORS = ("spatial_encoder", "mobility_enabled", "objective", "od_source")
VALID_FACTORS = {
    "spatial_encoder": {"pooled", "hetero"},
    "mobility_enabled": {False, True},
    "objective": {"nll", "crps"},
    "od_source": {"mitma", "municipal"},
}
IDENTITY_COLUMNS = ["target", "source_type", "source_id", "target_date", "horizon"]
METRICS = [
    "wis",
    "wis_dispersion",
    "wis_underprediction",
    "wis_overprediction",
    "wis_point",
    "abs_error",
    "abs_error_raw",
    "crps",
    "nll",
    "covered_50",
    "covered_80",
    "covered_95",
]
COVERAGE_METRICS = {"covered_50", "covered_80", "covered_95"}
HISTORICAL_RELATIVE_PATH = Path(
    "source_resolution_bundle/source_resolution_analysis/"
    "head_wis_decomposition_paired.csv"
)
CAVEATS = [
    "Native scores are direct-source schema-v4 evaluation results from retrained models.",
    "Historical scores use municipal labels and a static source-resolution diagnostic; their magnitudes are not directly comparable with native scores.",
    "Historical and native score values are kept in separate tables and are never pooled.",
    "Repeated origins and overlapping geographic supports are not treated as independent replications; fold-seed rows remain the replication unit.",
    "Weekly hospitalization horizons are reporting weeks, while cases and deaths use daily horizons.",
]


@dataclass(frozen=True)
class NativeRun:
    """One validated native granular export and its condition metadata."""

    path: Path
    frame: pd.DataFrame
    spatial_encoder: str
    mobility_enabled: bool
    objective: str
    od_source: str
    fold: int
    seed: int
    metadata: dict[str, Any]

    @property
    def factors(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FACTORS}


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(f"{path.suffix}.meta.json")
    if not sidecar.exists():
        raise ValueError(f"missing granular metadata sidecar: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _read_granular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path,
            dtype={"source_type": "string", "source_id": "string"},
        )
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        frame["source_type"] = frame["source_type"].astype("string")
        frame["source_id"] = frame["source_id"].astype("string")
    else:
        raise ValueError(f"unsupported granular format: {path}")
    required = set(IDENTITY_COLUMNS + METRICS + ["observed"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing schema-v4 columns: {missing}")
    observed = frame[frame["observed"].notna()].copy()
    duplicated = observed.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicated.any():
        raise ValueError(f"{path} has duplicate native comparison identities")
    return observed


def load_native_runs(paths: Iterable[Path]) -> list[NativeRun]:
    """Load and validate a complete condition matrix for each fold and seed."""
    runs: list[NativeRun] = []
    for path in sorted(paths):
        metadata = _read_sidecar(path)
        if str(metadata.get("schema_version")) != GRANULAR_SCHEMA_VERSION:
            raise ValueError(f"{path} is not granular schema v4")
        values = {name: metadata.get(name) for name in FACTORS}
        if not isinstance(values["mobility_enabled"], bool):
            raise ValueError(f"{path}: mobility_enabled metadata must be boolean")
        for name, allowed in VALID_FACTORS.items():
            if values[name] not in allowed:
                raise ValueError(f"{path}: invalid {name}={values[name]!r}")
        if metadata.get("fold") is None or metadata.get("seed") is None:
            raise ValueError(f"{path}: fold and seed metadata are required")
        runs.append(
            NativeRun(
                path=path,
                frame=_read_granular(path),
                fold=int(metadata["fold"]),
                seed=int(metadata["seed"]),
                metadata=metadata,
                **values,
            )
        )
    if not runs:
        raise ValueError("no native granular files found")
    keys = [
        (run.fold, run.seed, *(run.factors[name] for name in FACTORS)) for run in runs
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate native condition for the same fold and seed")
    expected = 1
    for values in VALID_FACTORS.values():
        expected *= len(values)
    counts = pd.Series([(run.fold, run.seed) for run in runs]).value_counts()
    incomplete = {
        str(key): int(value) for key, value in counts.items() if value != expected
    }
    if incomplete:
        raise ValueError(
            f"each fold-seed requires {expected} Stage-4 conditions; got {incomplete}"
        )
    return runs


def discover_native_paths(root: Path) -> list[Path]:
    """Find granular files while excluding aggregate exports."""
    paths = list(root.rglob("*granular.csv")) + list(root.rglob("*granular.parquet"))
    return sorted(path for path in paths if path.is_file())


def _condition_index(runs: list[NativeRun]) -> dict[tuple[Any, ...], NativeRun]:
    return {
        (run.fold, run.seed, *(run.factors[name] for name in FACTORS)): run
        for run in runs
    }


def _pair_specs(runs: list[NativeRun]) -> list[tuple[str, NativeRun, NativeRun]]:
    """Build encoder, mobility, and mobility-on OD comparisons."""
    index = _condition_index(runs)
    pairs: list[tuple[str, NativeRun, NativeRun]] = []
    for run in runs:
        factors = run.factors
        if run.spatial_encoder == "pooled":
            candidate = dict(factors, spatial_encoder="hetero")
            pairs.append(("encoder", run, index[_index_key(run, candidate)]))
        if not run.mobility_enabled:
            candidate = dict(factors, mobility_enabled=True)
            pairs.append(("mobility", run, index[_index_key(run, candidate)]))
        if run.mobility_enabled and run.od_source == "municipal":
            candidate = dict(factors, od_source="mitma")
            pairs.append(("od_context", run, index[_index_key(run, candidate)]))
    return pairs


def _index_key(run: NativeRun, factors: dict[str, Any]) -> tuple[Any, ...]:
    return (run.fold, run.seed, *(factors[name] for name in FACTORS))


def _identity_hash(frame: pd.DataFrame) -> str:
    text = "\n".join(
        "|".join(str(value) for value in row)
        for row in frame[IDENTITY_COLUMNS]
        .sort_values(IDENTITY_COLUMNS)
        .itertuples(index=False, name=None)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_native_runs(runs: list[NativeRun]) -> pd.DataFrame:
    """Return per-head, per-horizon fold-seed results on common identities."""
    output: list[dict[str, Any]] = []
    for comparison, baseline, candidate in _pair_specs(runs):
        merged = baseline.frame.merge(
            candidate.frame,
            on=IDENTITY_COLUMNS,
            how="inner",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if merged.empty:
            raise ValueError(
                f"no common source/date rows for {comparison}: "
                f"{baseline.path} vs {candidate.path}"
            )
        for (target, horizon), group in merged.groupby(
            ["target", "horizon"], sort=True, dropna=False
        ):
            baseline_scope = baseline.frame[
                (baseline.frame["target"] == target)
                & (baseline.frame["horizon"] == horizon)
            ]
            candidate_scope = candidate.frame[
                (candidate.frame["target"] == target)
                & (candidate.frame["horizon"] == horizon)
            ]
            row: dict[str, Any] = {
                "comparison": comparison,
                "fold": baseline.fold,
                "seed": baseline.seed,
                "objective": baseline.objective,
                "target": target,
                "horizon": int(horizon),
                "baseline_encoder": baseline.spatial_encoder,
                "candidate_encoder": candidate.spatial_encoder,
                "baseline_mobility": baseline.mobility_enabled,
                "candidate_mobility": candidate.mobility_enabled,
                "baseline_od_source": baseline.od_source,
                "candidate_od_source": candidate.od_source,
                "baseline_rows": len(baseline_scope),
                "candidate_rows": len(candidate_scope),
                "common_rows": len(group),
                "common_sources": group[["source_type", "source_id"]]
                .drop_duplicates()
                .shape[0],
                "common_dates": group["target_date"].nunique(),
                "common_identity_sha256": _identity_hash(group),
                "baseline_parameters": baseline.metadata.get("model_parameters"),
                "candidate_parameters": candidate.metadata.get("model_parameters"),
                "baseline_optimizer_steps": baseline.metadata.get("optimizer_steps"),
                "candidate_optimizer_steps": candidate.metadata.get("optimizer_steps"),
                "baseline_wall_time_s": baseline.metadata.get(
                    "wall_time_s_before_final_eval"
                ),
                "candidate_wall_time_s": candidate.metadata.get(
                    "wall_time_s_before_final_eval"
                ),
            }
            for metric in METRICS:
                source_values = group.groupby(["source_type", "source_id"], sort=True)[
                    [f"{metric}_baseline", f"{metric}_candidate"]
                ].mean()
                if metric == "abs_error_raw":
                    baseline_value = float(group[f"{metric}_baseline"].mean())
                    candidate_value = float(group[f"{metric}_candidate"].mean())
                else:
                    baseline_value = float(source_values[f"{metric}_baseline"].mean())
                    candidate_value = float(source_values[f"{metric}_candidate"].mean())
                row[f"{metric}_baseline"] = baseline_value
                row[f"{metric}_candidate"] = candidate_value
                row[f"{metric}_change"] = (
                    candidate_value - baseline_value
                    if metric in COVERAGE_METRICS
                    else baseline_value - candidate_value
                )
                source_change = (
                    source_values[f"{metric}_candidate"]
                    - source_values[f"{metric}_baseline"]
                    if metric in COVERAGE_METRICS
                    else source_values[f"{metric}_baseline"]
                    - source_values[f"{metric}_candidate"]
                )
                row[f"{metric}_source_change_std"] = float(source_change.std(ddof=1))
            output.append(row)
    return pd.DataFrame(output).sort_values(
        ["comparison", "objective", "target", "horizon", "fold", "seed"],
        kind="stable",
    )


def load_historical_reference(study_root: Path) -> pd.DataFrame:
    """Load only production-supported historical Normal NLL/CRPS rows."""
    path = study_root / HISTORICAL_RELATIVE_PATH
    frame = pd.read_csv(path, dtype={"family": "string", "objective": "string"})
    filtered = frame[
        (frame["family"] == "normal") & frame["objective"].isin(["nll", "crps"])
    ].copy()
    if filtered.empty:
        raise ValueError(f"historical reference has no supported rows: {path}")
    filtered.insert(0, "reference_scope", "historical_municipal_static_diagnostic")
    return filtered


def _variability_summary(table: pd.DataFrame) -> list[dict[str, Any]]:
    groups = ["comparison", "objective", "target", "horizon"]
    summary = (
        table.groupby(groups, sort=True)
        .agg(
            fold_seed_pairs=("common_rows", "size"),
            common_rows=("common_rows", "sum"),
            wis_gain_mean=("wis_change", "mean"),
            wis_gain_std=("wis_change", "std"),
            transformed_abs_error_gain_mean=("abs_error_change", "mean"),
            raw_abs_error_gain_mean=("abs_error_raw_change", "mean"),
        )
        .reset_index()
    )
    return json.loads(summary.to_json(orient="records"))


def run_comparison(
    *, native_paths: Iterable[Path], historical_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Run the complete analysis and write its machine-readable artifacts."""
    runs = load_native_runs(native_paths)
    native_table = compare_native_runs(runs)
    historical = load_historical_reference(historical_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / "native_stage4_paired_metrics.csv"
    historical_path = output_dir / "historical_municipal_reference.csv"
    summary_path = output_dir / "native_stage4_summary.json"
    native_table.to_csv(native_path, index=False)
    historical.to_csv(historical_path, index=False)
    summary = {
        "schema": "native_stage4_comparison.v1",
        "native_table": str(native_path),
        "historical_reference_table": str(historical_path),
        "native_run_count": len(runs),
        "fold_seed_pairs": sorted({f"{run.fold}:{run.seed}" for run in runs}),
        "common_subset_identity": IDENTITY_COLUMNS,
        "metric_change_convention": {
            "lower_is_better": "baseline_minus_candidate",
            "coverage": "candidate_minus_baseline",
        },
        "metric_aggregation": {
            "transformed_rate_and_coverage": "mean of per-source means",
            "raw_count_abs_error": "mean over common observed rows",
            "source_variability": "sample standard deviation of paired per-source changes",
            "replication_unit": "fold-seed",
        },
        "mobility_off_neutralized_inputs": [
            "od_context_zero_filled",
            "od_graph_messages",
        ],
        "native_variability": _variability_summary(native_table),
        "historical_filter": {
            "family": "normal",
            "objectives": ["nll", "crps"],
            "row_count": len(historical),
        },
        "caveats": CAVEATS,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=Path("outputs/loss_research/loss-type-full-coverage-v2"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reports/native_stage4")
    )
    args = parser.parse_args(argv)
    summary = run_comparison(
        native_paths=discover_native_paths(args.runs_root),
        historical_root=args.historical_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
