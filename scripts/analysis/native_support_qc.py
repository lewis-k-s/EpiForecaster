"""Native-support QC report (slice MAN-1; plan §11.3, §12.2).

Builds ``native_manifest_v2.json`` (:mod:`data.preprocess.native_sources.manifest`)
and a reconciliation report over the built native-support artifacts:

- ABS case-strata ``stratum_pop_sum`` weekly lower bound vs the audited ABS
  population in ``native_graph`` (ratios; >1 is a warning with numbers);
- comarca population closure (``obs_deaths`` member-sum vs graph nodes);
- MITMA zone-count reconciliation (442 prefix / 454 G1 diagnostics /
  458 transform attr vs the built 456);
- OD mass retention per policy on both OD stores, read from the store attrs
  and never recomputed;
- observed-source coverage vs geometry counts (377 vs 379 ABS, 42/43
  comarca, municipal observed vs graph, and the registry codes without graph
  presence, including ``abs:302``);
- the v1<->v2 MITMA population crosswalk gap (107 NaN zones) as a recorded
  open item.

Tolerance violations are warnings carrying numbers; the only hard errors are
missing inputs or hashes (raised by the manifest builder). Run:

    uv run python scripts/analysis/native_support_qc.py --root data/processed/native_support
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.native_registry import (
    build_abs_registry,
    build_comarca_registry,
    build_municipal_registry,
)
from data.preprocess.native_sources.manifest import (
    CONFIRMATION_PERIOD,
    MANIFEST_NAME,
    SCHEMA as MANIFEST_SCHEMA,
    InputPaths,
    build_manifest,
    write_manifest,
)

SCHEMA = "native_support_qc.v1"
DEFAULT_REPORT = Path("outputs/native_support_qc_report.json")
DEFAULT_REAL_WITH_ID = Path("data/processed/real_with_id.zarr")

RATIO_TOL = 1e-6
CLOSURE_TOL = 1e-6
RETENTION_ANCHOR_TOL = 1e-9

#: Frozen built-store references (plan §11.3 od_mitma entry, verified).
EXPECTED_MITMA_ZONE_COUNTS = {
    "prefix_verified": 442,
    "g1_diagnostics": 454,
    "transform_attr": 458,
    "built": 456,
}

_ABS_SECTION = "ABS stratum-pop (weekly max) vs audited graph population"


def load_graph_nodes(graph: xr.Dataset) -> pd.DataFrame:
    """Node table as a plain frame: node_type, source_id, population."""
    return pd.DataFrame(
        {
            "node_type": np.asarray(graph["node_type"].values, dtype=str),
            "source_id": np.asarray(graph["source_id"].values, dtype=str),
            "population": np.asarray(graph["population"].values, dtype=float),
        }
    )


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def abs_population_ratios(nodes: pd.DataFrame, admissions: xr.Dataset) -> dict:
    """Per-ABS ratio of max weekly ``stratum_pop_sum`` to graph population.

    ``stratum_pop_sum`` sums case-reported strata populations only, so it is
    a weekly lower bound on the audited ABS denominator (§10.3(4)); ratios
    above 1 are tolerance violations (warnings), never recomputed masks.
    """
    abs_pop = nodes[nodes["node_type"] == "abs"].set_index("source_id")["population"]
    abs_ids = [str(v) for v in admissions["abs_id"].values]
    weekly = np.asarray(admissions["stratum_pop_sum"].values, dtype=float)
    weekly_max = np.where(np.isfinite(weekly), weekly, -np.inf).max(axis=0)

    ratios: dict[str, float] = {}
    missing: list[str] = []
    for k, abs_id in enumerate(abs_ids):
        pop = abs_pop.get(abs_id)
        if pop is None or not np.isfinite(pop) or pop <= 0:
            missing.append(abs_id)
            continue
        if weekly_max[k] > -np.inf:
            ratios[abs_id] = float(weekly_max[k] / pop)
    above_one = {k: v for k, v in ratios.items() if v > 1.0 + RATIO_TOL}
    worst = dict(sorted(above_one.items(), key=lambda kv: -kv[1])[:5])
    return {
        "method": _ABS_SECTION,
        "n_abs_observed": len(abs_ids),
        "n_with_graph_population": len(ratios),
        "abs_without_graph_population": sorted(missing),
        "ratio": _stats(list(ratios.values())) if ratios else None,
        "n_ratio_above_one": len(above_one),
        "worst_above_one": worst,
        "tolerance": f"ratio <= 1 + {RATIO_TOL:g} expected; violations warn",
    }


def comarca_population_closure(nodes: pd.DataFrame, deaths: xr.Dataset) -> dict:
    """``obs_deaths`` member-sum population vs graph comarca populations."""
    com_pop = nodes[nodes["node_type"] == "comarca"].set_index("source_id")[
        "population"
    ]
    ids = [str(v) for v in deaths["comarca_id"].values]
    pop = np.asarray(deaths["population"].values, dtype=float)
    if pop.ndim == 2:  # defensive: static denominators are 1-D in the store
        pop = np.where(np.isfinite(pop), pop, -np.inf).max(axis=0)

    rel_errors: dict[str, float] = {}
    unmatched: list[str] = []
    for k, comarca_id in enumerate(ids):
        graph_pop = com_pop.get(comarca_id)
        if (
            graph_pop is None
            or not np.isfinite(graph_pop)
            or graph_pop <= 0
            or not np.isfinite(pop[k])
        ):
            unmatched.append(comarca_id)
            continue
        rel_errors[comarca_id] = float(abs(pop[k] - graph_pop) / graph_pop)
    mismatches = {k: v for k, v in rel_errors.items() if v > CLOSURE_TOL}
    return {
        "method": "obs_deaths population (member-municipality sum) vs native_graph comarca nodes",
        "n_store_columns": len(ids),
        "n_matched": len(rel_errors),
        "unmatched_store_columns": sorted(unmatched),
        "max_abs_rel_error": max(rel_errors.values()) if rel_errors else None,
        "n_mismatch": len(mismatches),
        "worst_mismatch": dict(sorted(mismatches.items(), key=lambda kv: -kv[1])[:5]),
        "tolerance": f"rel err <= {CLOSURE_TOL:g}; violations warn",
    }


def mitma_zone_counts(od: xr.Dataset) -> dict:
    """Zone-count reconciliation straight from the built store."""
    zone_pop = np.asarray(od["zone_population"].values, dtype=float)
    return {
        "built": int(od.sizes["zone"]),
        "date_count": int(od.sizes["date"]),
        "zone_count_shp_total": od.attrs.get("zone_count_shp_total"),
        "reference_zone_counts": {
            k: int(v) for k, v in (od.attrs.get("reference_zone_counts") or {}).items()
        },
        "zone_population_nan": int(np.isnan(zone_pop).sum()),
    }


def normalize_retention(attrs: dict) -> dict:
    """Normalize both store attr layouts into one per-policy map.

    ``od_mitma`` stores flat ``retained_mass_fraction_<policy>_minflow<n>``
    keys; ``od_municipal`` nests a ``retained_mass_fraction`` dict keyed by
    ``<policy>__min_flow_raw_<n>``.
    """
    policies: dict[str, dict] = {}

    def _entry(policy: str, suffix: str) -> dict:
        return policies.setdefault(f"{policy}|min_flow_raw={float(suffix):g}", {})

    for key, value in attrs.items():
        flat = re.fullmatch(
            r"retained_mass_fraction_(retain|drop)_minflow([\d.]+)", key
        )
        if flat:
            entry = _entry(flat[1], flat[2])
            entry["retained_mass_fraction"] = float(value)
            entry["edge_count"] = attrs.get(f"edge_count_{flat[1]}_minflow{flat[2]}")
            entry["retained_mass"] = attrs.get(
                f"retained_mass_{flat[1]}_minflow{flat[2]}"
            )
    nested = attrs.get("retained_mass_fraction")
    if isinstance(nested, dict):
        for key, value in nested.items():
            dicted = re.fullmatch(r"(retain|drop)__min_flow_raw_([\d.]+)", key)
            if dicted:
                entry = _entry(dicted[1], dicted[2])
                entry["retained_mass_fraction"] = float(value)
                entry["edge_count"] = (attrs.get("edge_count") or {}).get(key)
    out = {"policies": dict(sorted(policies.items()))}
    for extra in ("diag_mass_fraction", "mass_in_zone_set_fraction", "total_raw_mass"):
        if attrs.get(extra) is not None:
            out[extra] = float(attrs[extra])
    return out


def od_mass_retention(mitma_attrs: dict, municipal_attrs: dict | None) -> dict:
    """Retention per policy from store attrs only (never recomputed, §12.2)."""
    return {
        "method": "read from store attrs; not recomputed",
        "od_mitma": normalize_retention(mitma_attrs),
        "od_municipal": normalize_retention(municipal_attrs)
        if municipal_attrs
        else None,
    }


def observed_coverage(
    nodes: pd.DataFrame,
    admissions: xr.Dataset,
    deaths: xr.Dataset,
    registry_ids: dict[str, set[str]],
    municipal_observed_ids: set[str] | None,
) -> dict:
    """Observed-source coverage vs geometry (graph) counts, plus registry codes
    with no graph presence (plan §7 sampling surface; GRF-1 warnings)."""
    graph_ids = {
        t: set(nodes[nodes["node_type"] == t]["source_id"])
        for t in ("municipality", "abs", "comarca", "mitma")
    }
    abs_obs = {str(v) for v in admissions["abs_id"].values}
    com_obs = {str(v) for v in deaths["comarca_id"].values}
    without_graph = {
        t: sorted(registry_ids.get(t, set()) - graph_ids[t])
        for t in ("municipality", "abs", "comarca")
    }
    return {
        "abs": {
            "observed_store_columns": len(abs_obs),
            "graph_nodes": len(graph_ids["abs"]),
            "store_not_in_graph": sorted(abs_obs - graph_ids["abs"]),
            "graph_not_observed": sorted(graph_ids["abs"] - abs_obs),
        },
        "comarca": {
            "observed_store_columns": len(com_obs),
            "observed_mapped": len(com_obs & graph_ids["comarca"]),
            "graph_nodes": len(graph_ids["comarca"]),
            "store_not_in_graph": sorted(com_obs - graph_ids["comarca"]),
            "graph_not_observed": sorted(graph_ids["comarca"] - com_obs),
        },
        "municipality": {
            "observed": None
            if municipal_observed_ids is None
            else len(municipal_observed_ids),
            "graph_nodes": len(graph_ids["municipality"]),
            "graph_not_observed": (
                None
                if municipal_observed_ids is None
                else len(graph_ids["municipality"] - municipal_observed_ids)
            ),
        },
        "registry_codes_without_graph_presence": {
            **without_graph,
            "total": sum(len(v) for v in without_graph.values()),
        },
    }


def load_registry_ids(spec: InputPaths) -> dict[str, set[str]]:
    """Registry id sets per source type via the public REG-1 builders."""
    abs_reg, _ = build_abs_registry(spec.abs_csv)
    com_reg, _ = build_comarca_registry(
        spec.deaths_csv, spec.mpis_csv, spec.population_csv, spec.reference_csv
    )
    mun_reg, _ = build_municipal_registry(spec.population_csv, spec.reference_csv)
    return {
        "abs": {str(r.source_id) for r in abs_reg},
        "comarca": {str(r.source_id) for r in com_reg},
        "municipality": {str(r.source_id) for r in mun_reg},
    }


def _load_real_with_id_ids(path: Path) -> set[str] | None:
    """Observed municipal ids: the cases store ``region_id`` coord only."""
    if not Path(path).is_dir():
        return None
    with xr.open_zarr(path, consolidated=False) as store:  # type: ignore[arg-type]
        return {str(v) for v in store["region_id"].values}


def build_report(
    root: str | Path,
    inputs: InputPaths | None = None,
    real_with_id: Path = DEFAULT_REAL_WITH_ID,
    municipal_observed_ids: set[str] | None = None,
    manifest: dict | None = None,
    registry_ids: dict[str, set[str]] | None = None,
) -> dict:
    """Build the manifest, run every QC section, and assemble the report.

    Raises (hard error) on missing inputs/hashes via the manifest builder;
    all tolerance findings are recorded warnings. ``manifest`` may carry a
    prebuilt manifest dict to avoid hashing the inputs twice; ``registry_ids``
    may carry REG-1 registry id sets to skip CSV re-parsing.
    """
    spec = inputs or InputPaths()
    root = Path(root)
    if manifest is None:
        manifest = build_manifest(root, spec)

    graph = xr.open_zarr(root / "native_graph.zarr", consolidated=False)
    admissions = xr.open_zarr(root / "obs_admissions.zarr", consolidated=False)
    deaths = xr.open_zarr(root / "obs_deaths.zarr", consolidated=False)
    od_mitma = xr.open_zarr(root / "od_mitma.zarr", consolidated=False)
    od_municipal = xr.open_zarr(root / "od_municipal.zarr", consolidated=False)
    nodes = load_graph_nodes(graph)

    if municipal_observed_ids is None:
        municipal_observed_ids = _load_real_with_id_ids(real_with_id)

    abs_check = abs_population_ratios(nodes, admissions)
    closure = comarca_population_closure(nodes, deaths)
    zones = mitma_zone_counts(od_mitma)
    retention = od_mass_retention(od_mitma.attrs, od_municipal.attrs)
    if registry_ids is None:
        registry_ids = load_registry_ids(spec)
    coverage = observed_coverage(
        nodes, admissions, deaths, registry_ids, municipal_observed_ids
    )

    warnings: list[str] = []
    if abs_check["n_ratio_above_one"]:
        top = ", ".join(
            f"{k}: {v:.3f}" for k, v in abs_check["worst_above_one"].items()
        )
        warnings.append(
            f"ABS stratum-pop lower bound exceeds graph population for "
            f"{abs_check['n_ratio_above_one']}/{abs_check['n_with_graph_population']} ABS "
            f"(worst: {top})"
        )
    if closure["n_mismatch"]:
        warnings.append(
            f"comarca population closure mismatch for {closure['n_mismatch']}/"
            f"{closure['n_matched']} comarcas, max rel err {closure['max_abs_rel_error']:.3e}"
        )
    for key, expected in EXPECTED_MITMA_ZONE_COUNTS.items():
        actual = (
            zones["built"]
            if key == "built"
            else zones["reference_zone_counts"].get(key)
        )
        if actual != expected:
            warnings.append(
                f"MITMA zone count {key}: built/report {actual} != expected {expected}"
            )
    for store in ("od_mitma", "od_municipal"):
        anchor = retention[store]["policies"].get("retain|min_flow_raw=0", {})
        fraction = anchor.get("retained_mass_fraction")
        if fraction is not None and abs(fraction - 1.0) > RETENTION_ANCHOR_TOL:
            warnings.append(
                f"{store} retain/min_flow_raw=0 retention {fraction:.12f} != 1.0"
            )
    registry_missing = coverage["registry_codes_without_graph_presence"]
    if registry_missing["total"]:
        listed = "; ".join(
            f"{t}: {ids}" for t, ids in registry_missing.items() if t != "total" and ids
        )
        warnings.append(
            f"{registry_missing['total']} registry codes without graph presence ({listed})"
        )
    if municipal_observed_ids is None:
        warnings.append(f"municipal observed ids unavailable: {real_with_id} not found")

    open_items = [
        (
            f"mitma v1<->v2 population crosswalk gap: {zones['zone_population_nan']}/"
            f"{zones['built']} od_mitma zones have NaN zone_population (frozen "
            "exact-distrito join; the mitma_v2 poblacion.csv zonification is finer "
            "than v1 and _AM districts do not aggregate exactly)"
        )
    ]
    return {
        "schema": SCHEMA,
        "manifest": {
            "path": str(root / MANIFEST_NAME),
            "schema": MANIFEST_SCHEMA,
            "code_stamp_sha256": manifest["code_stamp"]["package_sha256"],
            "confirmation_period": list(CONFIRMATION_PERIOD),
        },
        "abs_population": abs_check,
        "comarca_population_closure": closure,
        "mitma_zone_counts": {"expected": dict(EXPECTED_MITMA_ZONE_COUNTS), **zones},
        "od_mass_retention": retention,
        "observed_coverage": coverage,
        "open_items": open_items,
        "warnings": warnings,
        "errors": [],
    }


def format_report(report: dict) -> str:
    """Human-readable headline summary (stdout of the CLI)."""
    abs_check = report["abs_population"]
    closure = report["comarca_population_closure"]
    zones = report["mitma_zone_counts"]
    coverage = report["observed_coverage"]
    lines = [
        f"native_support QC ({report['schema']})",
        (
            f"  manifest: {report['manifest']['path']} "
            f"(code stamp {report['manifest']['code_stamp_sha256'][:12]})"
        ),
        (
            f"  confirmation_period: {report['manifest']['confirmation_period']} "
            "(scoring this window requires the explicit guard flag)"
        ),
        (
            f"  ABS stratum-pop/graph-pop ratio: n={abs_check['ratio']['n']}, "
            f"median={abs_check['ratio']['median']:.3f}, max={abs_check['ratio']['max']:.3f}; "
            f"{abs_check['n_ratio_above_one']} above 1+{RATIO_TOL:g}"
            + (
                f" (worst {abs_check['worst_above_one']})"
                if abs_check["worst_above_one"]
                else ""
            )
        ),
        (
            f"  comarca population closure: n={closure['n_matched']}, "
            f"max rel err {closure['max_abs_rel_error']:.3e}, "
            f"unmatched store columns {closure['unmatched_store_columns']}"
        ),
        (
            f"  MITMA zones: built {zones['built']} vs references "
            f"{zones['reference_zone_counts']} (expected {zones['expected']})"
        ),
        f"  open items ({len(report['open_items'])}): {report['open_items'][0]}",
    ]
    for store in ("od_mitma", "od_municipal"):
        policies = report["od_mass_retention"][store]["policies"]
        joined = ", ".join(
            f"{key}={entry['retained_mass_fraction']:.6f}"
            for key, entry in policies.items()
        )
        lines.append(f"  OD mass retention {store}: {joined}")
    lines.append(
        f"  coverage: ABS obs {coverage['abs']['observed_store_columns']} vs graph "
        f"{coverage['abs']['graph_nodes']}; comarca observed-mapped "
        f"{coverage['comarca']['observed_mapped']} vs graph {coverage['comarca']['graph_nodes']}; "
        f"municipal observed {coverage['municipality']['observed']} vs graph "
        f"{coverage['municipality']['graph_nodes']}; registry codes without graph presence: "
        f"{coverage['registry_codes_without_graph_presence']['total']}"
    )
    lines.append(f"  warnings ({len(report['warnings'])}):")
    lines.extend(f"    - {w}" for w in report["warnings"])
    lines.append(f"  errors ({len(report['errors'])})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/processed/native_support")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--dasymetric-root", default=str(InputPaths.dasymetric_root))
    parser.add_argument("--real-with-id", default=str(DEFAULT_REAL_WITH_ID))
    args = parser.parse_args(argv)

    spec = InputPaths(dasymetric_root=Path(args.dasymetric_root))
    manifest = build_manifest(args.root, spec)
    report = build_report(args.root, spec, Path(args.real_with_id), manifest=manifest)
    write_manifest(manifest, Path(args.root) / MANIFEST_NAME)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(format_report(report))
    print(f"report: {report_path}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
