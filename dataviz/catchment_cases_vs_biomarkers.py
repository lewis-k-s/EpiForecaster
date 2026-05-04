"""Plot catchment-level canonical cases against EDAR biomarker series."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.append(sys_path)

from data.preprocess.config import (  # noqa: E402
    REGION_COORD,
    TEMPORAL_COORD,
    PreprocessingConfig,
)
from data.preprocess.processors.edar_processor import EDARProcessor  # noqa: E402
from utils.plotting import Style, format_date_axis, save_figure  # noqa: E402

logger = logging.getLogger(__name__)


def load_mapping(region_metadata: Path) -> xr.DataArray:
    mapping = xr.open_dataarray(region_metadata).fillna(0)
    if "home" in mapping.dims:
        mapping = mapping.rename({"home": REGION_COORD})
    mapping = mapping.assign_coords(
        edar_id=mapping["edar_id"].astype(str),
        **{REGION_COORD: mapping[REGION_COORD].astype(str)},
    )
    return mapping.where(mapping > 0, 0)


def aggregate_cases_to_catchments(
    dataset: xr.Dataset,
    mapping: xr.DataArray,
) -> xr.Dataset:
    cases = dataset["cases"]
    if "run_id" in cases.dims:
        cases = cases.sel(run_id="real")
    cases = cases.transpose(TEMPORAL_COORD, REGION_COORD)

    if "cases_mask" in dataset:
        cases_mask = dataset["cases_mask"]
        if "run_id" in cases_mask.dims:
            cases_mask = cases_mask.sel(run_id="real")
        cases_mask = cases_mask.transpose(TEMPORAL_COORD, REGION_COORD).astype(bool)
    else:
        cases_mask = cases.notnull()

    cases_aligned, mapping_aligned = xr.align(cases, mapping, join="inner")
    mask_aligned = cases_mask.sel({REGION_COORD: cases_aligned[REGION_COORD]})

    weighted_sum = xr.dot(
        cases_aligned.where(mask_aligned).fillna(0),
        mapping_aligned,
        dim=REGION_COORD,
    )
    observed_weight = xr.dot(
        mask_aligned.astype(float),
        mapping_aligned,
        dim=REGION_COORD,
    )
    catchment_cases = weighted_sum / observed_weight.where(observed_weight > 0)
    catchment_cases.name = "cases"

    catchment_mask = observed_weight > 0
    catchment_mask.name = "cases_mask"
    return xr.Dataset({"cases": catchment_cases, "cases_mask": catchment_mask})


def load_site_biomarkers(
    config: PreprocessingConfig,
    dates: pd.DatetimeIndex,
    catchment_ids: list[str],
    variants: list[str] | None,
) -> xr.Dataset:
    processor = EDARProcessor(config)
    flow_xr, censor_xr = processor.process_site_level(config.wastewater_file)

    flow = flow_xr.sel(run_id="real").assign_coords(
        edar_id=flow_xr["edar_id"].astype(str)
    )
    censor = censor_xr.sel(run_id="real").assign_coords(
        edar_id=censor_xr["edar_id"].astype(str)
    )

    if variants is None:
        selected_variants = [str(v) for v in flow["variant"].values.tolist()]
    else:
        selected_variants = variants

    flow = flow.sel(edar_id=catchment_ids, variant=selected_variants)
    censor = censor.sel(edar_id=catchment_ids, variant=selected_variants)

    flow = flow.reindex({TEMPORAL_COORD: dates})
    censor = censor.reindex({TEMPORAL_COORD: dates}).fillna(2)

    values = np.log1p(flow.clip(min=0))
    values.name = "biomarker"
    mask = ((censor < 1.5) & (flow > 0)).fillna(False)
    mask.name = "biomarker_mask"
    return xr.Dataset({"biomarker": values, "biomarker_mask": mask})


def select_catchments(
    biomarker_ds: xr.Dataset,
    requested_ids: list[str] | None,
    max_catchments: int,
) -> list[str]:
    available = [str(v) for v in biomarker_ds["edar_id"].values.tolist()]
    if requested_ids:
        missing = sorted(set(requested_ids) - set(available))
        if missing:
            raise ValueError(f"Unknown catchment ids: {missing}")
        return requested_ids

    coverage = biomarker_ds["biomarker_mask"].sum(dim=(TEMPORAL_COORD, "variant"))
    ranked = coverage.sortby(coverage, ascending=False)
    return [str(v) for v in ranked["edar_id"].values[:max_catchments].tolist()]


def build_plot_dataframe(
    cases_ds: xr.Dataset,
    biomarker_ds: xr.Dataset,
    catchment_ids: list[str],
) -> pd.DataFrame:
    cases = cases_ds.sel(edar_id=catchment_ids)
    cases_df = cases["cases"].to_dataframe().reset_index()
    cases_mask_df = cases["cases_mask"].to_dataframe().reset_index()
    cases_df = cases_df.merge(
        cases_mask_df,
        on=[TEMPORAL_COORD, "edar_id"],
        how="left",
    )
    cases_df = cases_df.rename(columns={"cases": "value", "cases_mask": "mask"})
    cases_df["series"] = "cases"
    cases_df["variant"] = "cases"

    biomarker = biomarker_ds.sel(edar_id=catchment_ids)
    biomarker_df = biomarker["biomarker"].to_dataframe().reset_index()
    biomarker_mask_df = biomarker["biomarker_mask"].to_dataframe().reset_index()
    biomarker_df = biomarker_df.merge(
        biomarker_mask_df,
        on=[TEMPORAL_COORD, "edar_id", "variant"],
        how="left",
    )
    biomarker_df = biomarker_df.rename(
        columns={"biomarker": "value", "biomarker_mask": "mask"}
    )
    biomarker_df["series"] = "biomarker"

    columns = [TEMPORAL_COORD, "edar_id", "series", "variant", "value", "mask"]
    return pd.concat(
        [cases_df[columns], biomarker_df[columns]],
        ignore_index=True,
    ).sort_values(["edar_id", "series", "variant", TEMPORAL_COORD])


def plot_catchment_series(plot_df: pd.DataFrame, output_path: Path) -> None:
    catchment_ids = plot_df["edar_id"].drop_duplicates().tolist()
    n_rows = len(catchment_ids)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(15, max(4, 3.2 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    cases_color = "tab:blue"
    for ax, catchment_id in zip(axes[:, 0], catchment_ids, strict=False):
        catchment_df = plot_df[plot_df["edar_id"] == catchment_id]
        cases_df = catchment_df[
            (catchment_df["series"] == "cases") & catchment_df["mask"].astype(bool)
        ]
        bio_df = catchment_df[
            (catchment_df["series"] == "biomarker")
            & catchment_df["mask"].astype(bool)
        ]

        sns.lineplot(
            data=cases_df,
            x=TEMPORAL_COORD,
            y="value",
            ax=ax,
            color=cases_color,
            linewidth=1.8,
            label="cases",
            errorbar=None,
        )
        ax.set_ylabel("Cases log1p(per-100k)", color=cases_color)
        ax.tick_params(axis="y", labelcolor=cases_color)
        ax.set_title(f"EDAR catchment {catchment_id}")
        ax.grid(True, alpha=0.25)

        twin = ax.twinx()
        if not bio_df.empty:
            variants = bio_df["variant"].drop_duplicates().tolist()
            biomarker_palette = dict(
                zip(
                    variants,
                    sns.color_palette("Dark2", n_colors=len(variants)),
                    strict=False,
                )
            )
            sns.lineplot(
                data=bio_df,
                x=TEMPORAL_COORD,
                y="value",
                hue="variant",
                ax=twin,
                palette=biomarker_palette,
                linewidth=1.4,
                errorbar=None,
            )
        twin.set_ylabel("Biomarker log1p")

        handles, labels = ax.get_legend_handles_labels()
        bio_handles, bio_labels = twin.get_legend_handles_labels()
        if twin.legend_ is not None:
            twin.legend_.remove()
        if handles or bio_handles:
            ax.legend(
                handles + bio_handles,
                labels + bio_labels,
                loc="upper left",
                ncol=min(4, max(1, len(labels + bio_labels))),
            )

        format_date_axis(ax)

    axes[-1, 0].set_xlabel("Date")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=Style.DPI, log_msg="Saved catchment plot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/real_with_id.zarr"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocess_real_holt.yaml"),
    )
    parser.add_argument(
        "--region-metadata",
        type=Path,
        default=Path("data/files/edar_muni_edges.nc"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/preprocess/catchment_cases_vs_biomarkers"),
    )
    parser.add_argument("--catchment-id", nargs="*", default=None)
    parser.add_argument("--max-catchments", type=int, default=8)
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    config = PreprocessingConfig.from_file(args.config)
    dataset = xr.open_zarr(args.dataset_path)
    dates = pd.DatetimeIndex(pd.to_datetime(dataset[TEMPORAL_COORD].values))

    mapping = load_mapping(args.region_metadata)
    cases_ds = aggregate_cases_to_catchments(dataset, mapping)
    biomarker_ds = load_site_biomarkers(
        config,
        dates,
        [str(v) for v in mapping["edar_id"].values.tolist()],
        args.variants,
    )
    catchment_ids = select_catchments(
        biomarker_ds,
        args.catchment_id,
        args.max_catchments,
    )
    plot_df = build_plot_dataframe(cases_ds, biomarker_ds, catchment_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "catchment_cases_vs_biomarkers.csv"
    fig_path = args.output_dir / "catchment_cases_vs_biomarkers.png"
    plot_df.to_csv(csv_path, index=False)
    logger.info("Saved plot data to %s", csv_path)
    plot_catchment_series(plot_df, fig_path)


if __name__ == "__main__":
    main()
