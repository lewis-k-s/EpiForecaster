"""Audit native source-fold composition for Stage-4 fold selection (WP-C prep).

Reads the audited ``native_source_folds.json`` manifest together with
``native_graph.zarr`` and reports, per fold: held-out source counts per type,
observed-supervised source counts, held-out audited population and its share
of each type's observed population, the manifest's owned overlap population,
and the geographic spread of held-out municipalities across the four Catalan
provinces (INE prefix 08 Barcelona, 17 Girona, 25 Lleida, 43 Tarragona).

The audit is selection evidence only; it never modifies the manifest.

Usage:
    uv run python scripts/analysis/audit_native_folds.py \
        --root data/processed/native_support \
        --output-dir outputs/reports/native_fold_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROVINCES = {"08": "barcelona", "17": "girona", "25": "lleida", "43": "tarragona"}


def _graph_population(root: Path) -> dict[tuple[str, str], tuple[float, str]]:
    """Return ``(audited population, status)`` per typed graph node."""
    graph = xr.open_zarr(root / "native_graph.zarr")
    table: dict[tuple[str, str], tuple[float, str]] = {}
    for node_type, source_id, population, status in zip(
        graph["node_type"].values.tolist(),
        graph["source_id"].values.tolist(),
        graph["population"].values.tolist(),
        graph["status"].values.tolist(),
        strict=True,
    ):
        table[(str(node_type), str(source_id))] = (
            float(population) if np.isfinite(population) else 0.0,
            str(status),
        )
    graph.close()
    return table


def _observed_population(
    table: dict[tuple[str, str], tuple[float, str]], node_type: str
) -> float:
    return sum(
        population
        for (source_type, _), (population, status) in table.items()
        if source_type == node_type and status == "observed"
    )


def build_audit(root: Path) -> tuple[pd.DataFrame, dict]:
    """Return per-fold composition rows plus a summary block."""
    manifest = json.loads(
        (root / "native_source_folds.json").read_text(encoding="utf-8")
    )
    table = _graph_population(root)
    totals = {
        node_type: _observed_population(table, node_type)
        for node_type in ("municipality", "abs", "comarca")
    }
    audit_block = manifest.get("audit", {})
    overlap = audit_block.get("fold_owned_overlap_pop", {}).get("abs")
    overlap_values: list[float] = (
        [float(value) for value in overlap] if isinstance(overlap, list) else []
    )

    rows: list[dict[str, object]] = []
    for fold in manifest["folds"]:
        index = int(fold["fold_index"])
        municipalities = [str(value) for value in fold["municipalities"]]
        abs_ids = [str(value) for value in fold["abs"]]
        comarcas = [str(value) for value in fold["comarcas"]]

        province_counts = {name: 0 for name in PROVINCES.values()}
        for municipality in municipalities:
            province_counts[PROVINCES.get(municipality[:2], "other")] += 1

        abs_entries = [
            table[("abs", value)] for value in abs_ids if ("abs", value) in table
        ]
        comarca_entries = [
            table[("comarca", value)]
            for value in comarcas
            if ("comarca", value) in table
        ]
        abs_observed = [entry for entry in abs_entries if entry[1] == "observed"]
        comarca_observed = [
            entry for entry in comarca_entries if entry[1] == "observed"
        ]

        abs_pop = sum(entry[0] for entry in abs_observed)
        comarca_pop = sum(entry[0] for entry in comarca_observed)
        overlap_value = overlap_values[index] if index < len(overlap_values) else None
        rows.append(
            {
                "fold": index,
                "municipalities": len(municipalities),
                "abs": len(abs_ids),
                "abs_observed": len(abs_observed),
                "comarcas": len(comarcas),
                "comarca_observed": len(comarca_observed),
                "abs_pop": round(abs_pop, 1),
                "abs_pop_share": round(abs_pop / totals["abs"], 4)
                if totals["abs"]
                else None,
                "comarca_pop": round(comarca_pop, 1),
                "comarca_pop_share": (
                    round(comarca_pop / totals["comarca"], 4)
                    if totals["comarca"]
                    else None
                ),
                "abs_overlap_pop_manifest": (
                    round(overlap_value, 1) if overlap_value is not None else None
                ),
                **{f"muni_{name}": count for name, count in province_counts.items()},
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "schema": "native_fold_audit.v1",
        "manifest_schema": manifest.get("schema"),
        "derivation": manifest.get("derivation"),
        "manifest_audit_counts": audit_block.get("counts"),
        "observed_population_totals": {
            key: round(value, 1) for key, value in totals.items()
        },
        "fold_size_spread": audit_block.get("fold_size_spread"),
        "unassigned_observed": audit_block.get("unassigned_observed"),
    }
    return frame, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Native fold composition audit (WP-C selection evidence)."
    )
    parser.add_argument("--root", default="data/processed/native_support")
    parser.add_argument("--output-dir", default="outputs/reports/native_fold_audit")
    args = parser.parse_args(argv)

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, summary = build_audit(root)

    csv_path = output_dir / "native_fold_audit.csv"
    json_path = output_dir / "native_fold_audit.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({**summary, "folds": frame.to_dict(orient="records")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"\nwrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
