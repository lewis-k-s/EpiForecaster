#!/usr/bin/env python
"""Plot train, validation, and test membership for cross-validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from utils.plotting import Palette


ROLE_COLORS = {
    "train": Palette.SPLIT["train"],
    "validation": Palette.SPLIT["val"],
    "test": Palette.SPLIT["test"],
}
OUTSIDE_COLOR = "#E3E5E8"
BOUNDARY_COLOR = "#FFFFFF"


def plot_split_choropleth(
    *,
    membership_path: Path,
    geojson_path: Path,
    strategy: str,
    output_path: Path,
) -> None:
    """Render one train/validation/test choropleth for every split."""
    membership = pd.read_csv(membership_path, dtype={"region_id": str})
    membership = membership[membership["strategy"] == strategy].copy()
    if membership.empty:
        available = sorted(pd.read_csv(membership_path)["strategy"].unique())
        raise ValueError(
            f"Strategy {strategy!r} not found; available strategies: {available}"
        )

    duplicate_roles = membership.duplicated(["split_id", "region_id"], keep=False)
    if duplicate_roles.any():
        duplicates = membership.loc[duplicate_roles, ["split_id", "region_id", "role"]]
        raise ValueError(
            "Each region must have exactly one role per split; duplicates found:\n"
            f"{duplicates.head()}"
        )

    regions = gpd.read_file(geojson_path)
    if "id" not in regions:
        raise ValueError("GeoJSON must contain an 'id' property")
    regions["region_id"] = regions["id"].astype(str)

    geometry_ids = set(regions["region_id"])
    missing_ids = sorted(set(membership["region_id"]) - geometry_ids)
    if missing_ids:
        raise ValueError(f"Missing geometries for region IDs: {missing_ids[:10]}")

    split_ids = membership["split_id"].drop_duplicates().tolist()
    if len(split_ids) != 10:
        raise ValueError(f"Expected 10 folds, found {len(split_ids)}")
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(16, 8.8),
        constrained_layout=True,
    )

    for axis, split_id in zip(axes.flat, split_ids, strict=True):
        split_membership = membership[membership["split_id"] == split_id][
            ["region_id", "role"]
        ]
        mapped = regions.merge(split_membership, on="region_id", how="left")
        mapped["_fill"] = mapped["role"].map(ROLE_COLORS).fillna(OUTSIDE_COLOR)
        mapped.plot(
            ax=axis,
            color=mapped["_fill"],
            edgecolor=BOUNDARY_COLOR,
            linewidth=0.18,
        )

        role_counts = split_membership["role"].value_counts()
        axis.set_title(
            f"{split_id}\n"
            f"test {role_counts.get('test', 0)}  ·  "
            f"validation {role_counts.get('validation', 0)}  ·  "
            f"train {role_counts.get('train', 0)}",
            fontsize=10,
        )
        axis.set_axis_off()

    legend_handles = [
        Patch(facecolor=ROLE_COLORS["test"], label="Test"),
        Patch(facecolor=ROLE_COLORS["validation"], label="Validation"),
        Patch(facecolor=ROLE_COLORS["train"], label="Training"),
        Patch(facecolor=OUTSIDE_COLOR, label="Outside valid-target universe"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"{strategy}: spatial distribution by fold",
        fontsize=16,
        fontweight="semibold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_split_choropleth(
        membership_path=args.membership,
        geojson_path=args.geojson,
        strategy=args.strategy,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
