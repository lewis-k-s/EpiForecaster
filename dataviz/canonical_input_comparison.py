"""Compare canonical input series side by side across multiple processed datasets.

This module loads one or more canonical Zarr datasets, filters each to a single
run_id, discovers overlapping region IDs and dates, selects representative
region/window pairs, and renders the canonical input series (cases, biomarkers,
hospitalizations, deaths) for each source in a side-by-side layout.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from data.epi_dataset import EpiDataset
from data.preprocess.config import REGION_COORD, TEMPORAL_COORD
from dataviz.input_series_plots import _plot_biomarkers, _plot_single_series
from utils.logging import setup_logging, suppress_zarr_warnings

suppress_zarr_warnings()
logger = logging.getLogger(__name__)

COMPARISON_TARGETS = ("cases", "hospitalizations", "deaths", "wastewater")
BIOMARKER_VARIANTS = ("N1", "N2", "IP4")


@dataclass(frozen=True)
class SourceSpec:
    dataset_path: Path
    run_id: str
    label: str


@dataclass(frozen=True)
class CanonicalSource:
    spec: SourceSpec
    dataset: xr.Dataset
    region_ids: tuple[str, ...]
    dates: pd.DatetimeIndex
    region_index: dict[str, int]
    source_region_ids: frozenset[str]
    overlap_start_idx: int
    overlap_end_idx: int


@dataclass(frozen=True)
class ComparisonSelection:
    region_id: str
    window_start: int
    window_end: int
    window_dates: pd.DatetimeIndex
    score: float


def parse_source_spec(raw: str) -> SourceSpec:
    """Parse `<dataset_path>[:run_id[:label]]` into a source specification."""
    parts = raw.split(":")
    if not parts or not parts[0]:
        raise ValueError(f"Invalid source spec: {raw!r}")

    dataset_path = Path(parts[0]).expanduser()
    if not dataset_path.suffix:
        raise ValueError(
            f"Source path must point to a Zarr store, got {dataset_path!s}."
        )

    if len(parts) >= 2 and parts[1]:
        run_id = parts[1]
    else:
        runs = EpiDataset.discover_available_runs(dataset_path)
        if len(runs) != 1:
            raise ValueError(
                f"Dataset {dataset_path} has multiple run_ids {runs}; specify one explicitly."
            )
        run_id = runs[0]

    label = parts[2] if len(parts) >= 3 and parts[2] else f"{dataset_path.stem}:{run_id}"
    return SourceSpec(dataset_path=dataset_path, run_id=run_id, label=label)


def discover_default_sources(processed_dir: Path) -> list[SourceSpec]:
    """Discover a sensible default comparison from `data/processed`."""
    zarr_paths = sorted(processed_dir.glob("*.zarr"))
    if not zarr_paths:
        raise ValueError(f"No Zarr datasets found under {processed_dir}")

    specs: list[SourceSpec] = []
    for path in zarr_paths:
        runs = EpiDataset.discover_available_runs(path)
        if len(runs) == 1:
            specs.append(
                SourceSpec(
                    dataset_path=path,
                    run_id=runs[0],
                    label=f"{path.stem}:{runs[0]}",
                )
            )
        elif runs:
            specs.append(
                SourceSpec(
                    dataset_path=path,
                    run_id=runs[0],
                    label=f"{path.stem}:{runs[0]}",
                )
            )
    if len(specs) < 2:
        raise ValueError(
            f"Need at least two processed datasets in {processed_dir} to compare."
        )
    return specs[:2]


def load_canonical_source(spec: SourceSpec) -> CanonicalSource:
    dataset = EpiDataset.load_canonical_dataset(
        aligned_data_path=spec.dataset_path,
        run_id=spec.run_id,
        run_id_chunk_size=1,
    )

    region_ids = tuple(str(v) for v in dataset[REGION_COORD].values.tolist())
    dates = pd.DatetimeIndex(pd.to_datetime(dataset[TEMPORAL_COORD].values))

    return CanonicalSource(
        spec=spec,
        dataset=dataset,
        region_ids=region_ids,
        dates=dates,
        region_index={region_id: idx for idx, region_id in enumerate(region_ids)},
        source_region_ids=frozenset(region_ids),
        overlap_start_idx=0,
        overlap_end_idx=len(dates),
    )


def close_sources(sources: list[CanonicalSource]) -> None:
    for source in sources:
        source.dataset.close()


def restrict_sources_to_shared_time_range(
    sources: list[CanonicalSource],
) -> tuple[list[CanonicalSource], pd.DatetimeIndex]:
    overlap_start = max(source.dates[0] for source in sources)
    overlap_end = min(source.dates[-1] for source in sources)
    if overlap_start > overlap_end:
        raise ValueError("Sources do not have any overlapping dates.")

    overlap_dates = pd.date_range(overlap_start, overlap_end, freq="D")
    if len(overlap_dates) == 0:
        raise ValueError("No daily overlap found across the selected sources.")

    restricted: list[CanonicalSource] = []
    for source in sources:
        start_idx = int(source.dates.get_loc(overlap_start))
        end_idx = int(source.dates.get_loc(overlap_end)) + 1
        restricted.append(
            CanonicalSource(
                spec=source.spec,
                dataset=source.dataset,
                region_ids=source.region_ids,
                dates=source.dates,
                region_index=source.region_index,
                source_region_ids=source.source_region_ids,
                overlap_start_idx=start_idx,
                overlap_end_idx=end_idx,
            )
        )
    return restricted, overlap_dates


def common_region_ids(
    sources: list[CanonicalSource],
    *,
    requested_region_ids: list[str] | None = None,
    require_biomarker_source: bool = True,
) -> list[str]:
    if not sources:
        return []

    common_ids = set(sources[0].source_region_ids)
    for source in sources[1:]:
        common_ids &= source.source_region_ids

    if require_biomarker_source:
        for source in sources:
            if "edar_has_source" not in source.dataset:
                continue
            mask = source.dataset["edar_has_source"]
            if "run_id" in mask.dims:
                mask = mask.squeeze(drop=True)
            valid_ids = {
                str(region_id)
                for region_id, has_source in zip(
                    source.dataset[REGION_COORD].values,
                    mask.values.astype(bool),
                    strict=False,
                )
                if bool(has_source)
            }
            common_ids &= valid_ids

    ordered_common = sorted(common_ids)
    if requested_region_ids:
        missing = [region_id for region_id in requested_region_ids if region_id not in common_ids]
        if missing:
            raise ValueError(
                f"Requested region IDs are not available in every source: {missing}"
            )
        return requested_region_ids

    return ordered_common


def select_comparison_windows(
    sources: list[CanonicalSource],
    *,
    overlap_dates: pd.DatetimeIndex,
    num_samples: int,
    window_length: int,
    requested_region_ids: list[str] | None = None,
    requested_window_start: int | None = None,
    requested_window_date: str | None = None,
    require_biomarker_source: bool = True,
    seed: int = 42,
    max_candidate_starts: int = 24,
) -> list[ComparisonSelection]:
    region_ids = common_region_ids(
        sources,
        requested_region_ids=requested_region_ids,
        require_biomarker_source=require_biomarker_source,
    )
    if not region_ids:
        raise ValueError("No common region IDs found across the selected sources.")

    if len(overlap_dates) < window_length:
        raise ValueError(
            f"Window length {window_length} exceeds overlapping date span {len(overlap_dates)}."
        )

    if requested_window_date is not None:
        start_timestamp = pd.Timestamp(requested_window_date)
        if start_timestamp not in overlap_dates:
            raise ValueError(
                f"Requested window start date {requested_window_date} is outside the shared overlap."
            )
        requested_window_start = int(overlap_dates.get_loc(start_timestamp))

    max_start = len(overlap_dates) - window_length
    if requested_window_start is not None:
        if requested_window_start < 0 or requested_window_start > max_start:
            raise ValueError(
                f"Requested window_start {requested_window_start} must be between 0 and {max_start}."
            )
        candidate_starts = [requested_window_start]
    else:
        all_starts = np.arange(max_start + 1, dtype=np.int64)
        if len(all_starts) > max_candidate_starts:
            sampled_positions = np.linspace(
                0,
                len(all_starts) - 1,
                num=max_candidate_starts,
                dtype=int,
            )
            candidate_starts = all_starts[sampled_positions].tolist()
        else:
            candidate_starts = all_starts.tolist()

    rng = np.random.default_rng(seed)
    if not requested_region_ids:
        if len(region_ids) > max(num_samples * 16, 64):
            sampled = rng.choice(
                np.asarray(region_ids, dtype=object),
                size=max(num_samples * 16, 64),
                replace=False,
            )
            region_ids = [str(region_id) for region_id in sampled.tolist()]

    region_order = {region_id: idx for idx, region_id in enumerate(region_ids)}
    score_matrix = np.zeros((len(candidate_starts), len(region_ids)), dtype=np.float32)
    start_to_position = {start: idx for idx, start in enumerate(candidate_starts)}
    full_start_indices = np.asarray(candidate_starts, dtype=np.int64)

    for source in sources:
        source_region_indices = [source.region_index[region_id] for region_id in region_ids]
        time_slice = slice(source.overlap_start_idx, source.overlap_end_idx)

        target_masks: dict[str, np.ndarray] = {}
        for target in COMPARISON_TARGETS:
            if target == "wastewater":
                variant_masks: list[np.ndarray] = []
                for variant in BIOMARKER_VARIANTS:
                    var_name = f"edar_biomarker_{variant}_mask"
                    if var_name not in source.dataset:
                        continue
                    variant_masks.append(
                        source.dataset[var_name]
                        .isel(
                            {
                                TEMPORAL_COORD: time_slice,
                                REGION_COORD: source_region_indices,
                            }
                        )
                        .values.astype(np.float32)
                    )
                if variant_masks:
                    stacked = np.stack(variant_masks, axis=0)
                    target_masks[target] = np.any(stacked > 0, axis=0).astype(np.float32)
                continue

            mask_name = {
                "cases": "cases_mask",
                "hospitalizations": "hospitalizations_mask",
                "deaths": "deaths_mask",
            }[target]
            if mask_name in source.dataset:
                target_masks[target] = (
                    source.dataset[mask_name]
                    .isel(
                        {
                            TEMPORAL_COORD: time_slice,
                            REGION_COORD: source_region_indices,
                        }
                    )
                    .values.astype(np.float32)
                )

        for mask in target_masks.values():
            observed = (mask > 0).astype(np.float32)
            cumsum = np.concatenate(
                [
                    np.zeros((1, observed.shape[1]), dtype=np.float32),
                    np.cumsum(observed, axis=0),
                ],
                axis=0,
            )
            all_window_counts = cumsum[window_length:] - cumsum[:-window_length]
            score_matrix += all_window_counts[full_start_indices]

    candidates: list[ComparisonSelection] = []
    for region_id in region_ids:
        region_pos = region_order[region_id]
        best_idx = int(np.argmax(score_matrix[:, region_pos]))
        best_start = candidate_starts[best_idx]
        best: ComparisonSelection | None = None
        score = float(score_matrix[start_to_position[best_start], region_pos])
        selection = ComparisonSelection(
            region_id=region_id,
            window_start=best_start,
            window_end=best_start + window_length,
            window_dates=overlap_dates[best_start : best_start + window_length],
            score=score,
        )
        if best is None or selection.score > best.score:
            best = selection
        if best is not None:
            candidates.append(best)

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.region_id,
            item.window_start,
        )
    )
    return candidates[: min(num_samples, len(candidates))]


def _extract_window_sample(
    source: CanonicalSource,
    selection: ComparisonSelection,
) -> dict[str, Any]:
    region_idx = source.region_index[selection.region_id]
    t0 = source.overlap_start_idx + selection.window_start
    t1 = t0 + (selection.window_end - selection.window_start)
    window_dates = selection.window_dates
    def _series(name: str) -> np.ndarray:
        return (
            source.dataset[name]
            .isel({TEMPORAL_COORD: slice(t0, t1), REGION_COORD: region_idx})
            .values.astype(np.float32)
            .reshape(-1)
        )

    sample: dict[str, Any] = {
        "source_label": source.spec.label,
        "region_id": selection.region_id,
        "node_label": selection.region_id,
        "node_id": region_idx,
        "window_start": selection.window_start,
        "window_dates": window_dates,
        "cases_series": _series("cases"),
        "cases_age": _series("cases_age"),
        "cases_obs_mask_full": _series("cases_mask"),
        "hosp_series": _series("hospitalizations"),
        "hosp_age": _series("hospitalizations_age"),
        "hosp_obs_mask_full": _series("hospitalizations_mask"),
        "deaths_series": _series("deaths"),
        "deaths_age": _series("deaths_age"),
        "deaths_obs_mask_full": _series("deaths_mask"),
        "comparison_score": selection.score,
    }

    biomarkers: dict[str, dict[str, np.ndarray]] = {}
    ww_values: list[np.ndarray] = []
    ww_masks: list[np.ndarray] = []
    for variant in BIOMARKER_VARIANTS:
        base = f"edar_biomarker_{variant}"
        if base not in source.dataset:
            continue
        values = _series(base)
        mask = _series(f"{base}_mask")
        age = _series(f"{base}_age")
        censor_name = f"{base}_censor"
        censor = _series(censor_name) if censor_name in source.dataset else np.zeros_like(values)
        biomarkers[variant] = {
            "value": values,
            "mask": mask,
            "censor": censor,
            "age": age,
        }
        ww_values.append(values)
        ww_masks.append(mask)
    sample["biomarkers"] = biomarkers

    if ww_values:
        stacked_values = np.stack(ww_values, axis=0)
        stacked_masks = np.stack(ww_masks, axis=0) > 0
        valid_counts = stacked_masks.sum(axis=0)
        weighted_sum = np.where(stacked_masks, stacked_values, 0.0).sum(axis=0)
        ww_series = np.divide(
            weighted_sum,
            np.where(valid_counts > 0, valid_counts, 1),
        ).astype(np.float32)
        ww_series = np.where(valid_counts > 0, ww_series, np.nan).astype(np.float32)
        sample["ww_series"] = ww_series
        sample["ww_obs_mask_full"] = (valid_counts > 0).astype(np.float32)

    return sample


def make_canonical_input_comparison_figure(
    grouped_samples: list[list[dict[str, Any]]],
    *,
    window_length: int,
) -> Any:
    if not grouped_samples:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    n_rows = len(grouped_samples)
    n_sources = len(grouped_samples[0])
    n_series = 4

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_sources * n_series,
        figsize=(5.0 * n_sources * n_series, 3.4 * n_rows),
        sharex=True,
        squeeze=False,
    )

    colors = {
        "cases": "#1f77b4",
        "hosp": "#d62728",
        "deaths": "#9467bd",
    }
    t = np.arange(window_length)

    for row_idx, source_samples in enumerate(grouped_samples):
        per_row_ranges: dict[str, list[np.ndarray]] = {
            "cases": [],
            "biomarkers": [],
            "hosp": [],
            "deaths": [],
        }
        for sample in source_samples:
            per_row_ranges["cases"].append(np.asarray(sample["cases_series"], dtype=np.float32))
            per_row_ranges["hosp"].append(np.asarray(sample["hosp_series"], dtype=np.float32))
            per_row_ranges["deaths"].append(np.asarray(sample["deaths_series"], dtype=np.float32))
            ww_series = sample.get("ww_series")
            if ww_series is not None:
                per_row_ranges["biomarkers"].append(np.asarray(ww_series, dtype=np.float32))
            for channels in sample.get("biomarkers", {}).values():
                per_row_ranges["biomarkers"].append(
                    np.asarray(channels["value"], dtype=np.float32)
                )

        row_limits: dict[str, tuple[float, float] | None] = {}
        for key, arrays in per_row_ranges.items():
            if not arrays:
                row_limits[key] = None
                continue
            merged = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if np.isfinite(arr).any()])
            if merged.size == 0:
                row_limits[key] = None
                continue
            ymin = float(np.nanmin(merged))
            ymax = float(np.nanmax(merged))
            if ymax <= ymin:
                ymax = ymin + 1.0
            pad = (ymax - ymin) * (0.2 if key == "biomarkers" else 0.12)
            row_limits[key] = (ymin - pad, ymax + pad)

        for source_idx, sample in enumerate(source_samples):
            col_offset = source_idx * n_series

            ax_cases = axes[row_idx, col_offset]
            _plot_single_series(
                ax=ax_cases,
                series=sample["cases_series"],
                age=sample["cases_age"],
                observed_mask_full=sample["cases_obs_mask_full"],
                input_window_length=window_length,
                horizon_length=0,
                t=t,
                color=colors["cases"],
                label="Cases",
            )
            ax_cases.set_ylabel("log1p per-100k", fontsize=9)
            if row_limits["cases"] is not None:
                ax_cases.set_ylim(*row_limits["cases"])

            ax_bio = axes[row_idx, col_offset + 1]
            _plot_biomarkers(
                ax=ax_bio,
                biomarkers=sample.get("biomarkers", {}),
                ww_series=sample.get("ww_series"),
                ww_obs_mask_full=sample.get("ww_obs_mask_full"),
                input_window_length=window_length,
                horizon_length=0,
                t=t,
            )
            if row_limits["biomarkers"] is not None:
                ax_bio.set_ylim(*row_limits["biomarkers"])

            ax_hosp = axes[row_idx, col_offset + 2]
            _plot_single_series(
                ax=ax_hosp,
                series=sample["hosp_series"],
                age=sample["hosp_age"],
                observed_mask_full=sample["hosp_obs_mask_full"],
                input_window_length=window_length,
                horizon_length=0,
                t=t,
                color=colors["hosp"],
                label="Hosp",
            )
            if row_limits["hosp"] is not None:
                ax_hosp.set_ylim(*row_limits["hosp"])

            ax_deaths = axes[row_idx, col_offset + 3]
            _plot_single_series(
                ax=ax_deaths,
                series=sample["deaths_series"],
                age=sample["deaths_age"],
                observed_mask_full=sample["deaths_obs_mask_full"],
                input_window_length=window_length,
                horizon_length=0,
                t=t,
                color=colors["deaths"],
                label="Deaths",
            )
            if row_limits["deaths"] is not None:
                ax_deaths.set_ylim(*row_limits["deaths"])

            if row_idx == 0:
                axes[row_idx, col_offset].set_title(
                    f"{sample['source_label']}\nCases",
                    fontsize=10,
                    fontweight="semibold",
                )
                axes[row_idx, col_offset + 1].set_title(
                    f"{sample['source_label']}\nBiomarkers",
                    fontsize=10,
                    fontweight="semibold",
                )
                axes[row_idx, col_offset + 2].set_title(
                    f"{sample['source_label']}\nHospitalizations",
                    fontsize=10,
                    fontweight="semibold",
                )
                axes[row_idx, col_offset + 3].set_title(
                    f"{sample['source_label']}\nDeaths",
                    fontsize=10,
                    fontweight="semibold",
                )

        first = source_samples[0]
        row_title = (
            f"region={first['region_id']} | "
            f"{first['window_dates'][0].date()} -> {first['window_dates'][-1].date()} | "
            f"score={first['comparison_score']:.0f}"
        )
        axes[row_idx, 0].text(
            0.0,
            1.15,
            row_title,
            transform=axes[row_idx, 0].transAxes,
            fontsize=10,
            fontweight="semibold",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Window day", fontsize=9)

    plt.tight_layout()
    return fig


def generate_canonical_input_comparison(
    *,
    source_specs: list[SourceSpec],
    output_dir: Path,
    num_samples: int = 4,
    window_length: int = 84,
    region_ids: list[str] | None = None,
    window_start: int | None = None,
    window_start_date: str | None = None,
    require_biomarker_source: bool = True,
    seed: int = 42,
) -> tuple[list[ComparisonSelection], dict[str, Path]]:
    sources = [load_canonical_source(spec) for spec in source_specs]
    try:
        restricted_sources, overlap_dates = restrict_sources_to_shared_time_range(sources)
        selections = select_comparison_windows(
            restricted_sources,
            overlap_dates=overlap_dates,
            num_samples=num_samples,
            window_length=window_length,
            requested_region_ids=region_ids,
            requested_window_start=window_start,
            requested_window_date=window_start_date,
            require_biomarker_source=require_biomarker_source,
            seed=seed,
        )
        grouped_samples = [
            [_extract_window_sample(source, selection) for source in restricted_sources]
            for selection in selections
        ]
        figure = make_canonical_input_comparison_figure(
            grouped_samples,
            window_length=window_length,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}
        if figure is not None:
            import matplotlib.pyplot as plt

            figure_path = output_dir / "canonical_input_comparison.png"
            figure.savefig(figure_path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            artifacts["figure"] = figure_path

        summary_df = pd.DataFrame(
            [
                {
                    "region_id": selection.region_id,
                    "window_start": selection.window_start,
                    "window_end": selection.window_end,
                    "start_date": selection.window_dates[0],
                    "end_date": selection.window_dates[-1],
                    "score": selection.score,
                }
                for selection in selections
            ]
        )
        summary_path = output_dir / "canonical_input_comparison_windows.csv"
        summary_df.to_csv(summary_path, index=False)
        artifacts["summary"] = summary_path

        return selections, artifacts
    finally:
        close_sources(sources)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare canonical input series side by side across processed datasets."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Source spec formatted as <dataset_path>[:run_id[:label]]. "
            "Provide at least two sources, or omit to auto-discover from data/processed."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory used for default source discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/canonical_input_comparison"),
        help="Directory where comparison artifacts are written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of region/window comparisons to render.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=84,
        help="Number of days per comparison window.",
    )
    parser.add_argument(
        "--region-ids",
        type=str,
        default=None,
        help="Comma-separated canonical region IDs to compare.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=None,
        help="Window start offset within the shared overlap.",
    )
    parser.add_argument(
        "--window-start-date",
        type=str,
        default=None,
        help="Window start date within the shared overlap, e.g. 2020-09-01.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sub-sampling candidate regions.",
    )
    parser.add_argument(
        "--include-non-source-regions",
        action="store_true",
        help="Allow comparison regions that do not have biomarker source coverage in every dataset.",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)

    if args.source:
        source_specs = [parse_source_spec(raw) for raw in args.source]
    else:
        source_specs = discover_default_sources(args.processed_dir)

    if len(source_specs) < 2:
        raise ValueError("Need at least two sources for side-by-side comparison.")

    region_ids = None
    if args.region_ids:
        region_ids = [region_id.strip() for region_id in args.region_ids.split(",") if region_id.strip()]

    _, artifacts = generate_canonical_input_comparison(
        source_specs=source_specs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        window_length=args.window_length,
        region_ids=region_ids,
        window_start=args.window_start,
        window_start_date=args.window_start_date,
        require_biomarker_source=not args.include_non_source_regions,
        seed=args.seed,
    )

    for name, path in artifacts.items():
        logger.info("Wrote %s to %s", name, path)


if __name__ == "__main__":
    main()
