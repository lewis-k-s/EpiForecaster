"""Municipal-projection OD store builder (slice MOB-1).

Builds ``data/processed/native_support/od_municipal.zarr`` — the comparability
artifact of ``plan_heterogeneous_graph_refactor.md`` §11.3 — from the in-repo
``data/files/mobility.zarr`` dasymetric MITMA→municipality projection
(451 days × 963 × 963 municipal zones, raw float64 trips). Comparability
artifact only: consumed by the projected-context Stage-4 condition.

Stored schema (§11.3):
- dims ``(date, origin, target)``; direction is origin→target and is never
  symmetrized (§3, §11.7).
- ``mobility`` = ``log1p(clip(raw_trips, 0))`` stored as float16. No
  self-flow removal and no flow threshold is applied at rest; policies are
  recorded as audit attrs and applied at consumption.
- ``zone_id`` — string zone registry taken from the origin coordinate.
- ``zone_in_cases_footprint`` — bool coverage mask: zone present in the
  ``region_id`` coordinate of ``data/processed/real_with_id.zarr``
  (§10.3(7): full observation support distinct from the 768-zone study
  footprint).
- attrs (JSON-safe): ``diag_mass_fraction``, ``retained_mass_fraction`` and
  ``edge_count`` per policy ``{retain, drop} × {0.0, 10.0}`` in raw-trip
  space, ``zone_count_reconciliation`` vs
  ``municipality_id_reference.csv`` ``muni_code_5digit``, plus provenance.

Policy semantics mirror the repo edge-selection convention
(``data/mobility_preprocessor.py``): an entry is retained when
``raw > 0`` and ``raw >= min_flow_raw``; ``drop`` additionally clears the
origin==target diagonal. ``min_flow_raw=0.0`` therefore keeps positive
flows only (cf. ``configs/production_only/train_epifor_mn5_synth_pretrain.yaml``).

CLI::

    python -m data.preprocess.native_sources.od_municipal \\
        --source data/files/mobility.zarr \\
        --out data/processed/native_support/od_municipal.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from utils.logging import suppress_zarr_warnings

suppress_zarr_warnings()

DEFAULT_SOURCE = Path("data/files/mobility.zarr")
DEFAULT_OUTPUT = Path("data/processed/native_support/od_municipal.zarr")
DEFAULT_REFERENCE_CSV = Path("data/files/municipality_id_reference.csv")
DEFAULT_CASES_STORE = Path("data/processed/real_with_id.zarr")

#: Consumption-time policies audited in raw-trip space (§11.6 defaults).
SELF_FLOW_POLICIES: tuple[str, ...] = ("retain", "drop")
MIN_FLOW_RAW_POLICIES: tuple[float, ...] = (0.0, 10.0)

POLICY_DESCRIPTION = (
    "keep = (raw > 0) & (raw >= min_flow_raw); 'drop' also clears the "
    "origin==target diagonal; computed in raw-trip space, applied at "
    "consumption only - nothing is masked in the stored tensor"
)


def policy_key(self_flows: str, min_flow_raw: float) -> str:
    """Deterministic policy key, e.g. ``retain__min_flow_raw_10.0``."""
    return f"{self_flows}__min_flow_raw_{min_flow_raw}"


def _as_string_array(values: np.ndarray) -> np.ndarray:
    """Object-dtype array of Python strings (zarr v3 VLenUTF8-safe)."""
    return np.asarray([str(v) for v in np.asarray(values)], dtype=object)


def load_raw_od(source: str | Path) -> xr.Dataset:
    """Open the raw municipal-projection OD store lazily and validate it."""
    ds = xr.open_zarr(source)  # type: ignore[arg-type]
    expected = {"date", "origin", "target"}
    if set(ds.dims) != expected:
        raise ValueError(
            f"expected dims {sorted(expected)} in {source}, got {dict(ds.sizes)}"
        )
    if "mobility" not in ds.data_vars:
        raise ValueError(f"'mobility' variable missing from {source}")
    if not np.issubdtype(ds["mobility"].dtype, np.number):
        raise ValueError(f"'mobility' must be numeric, got {ds['mobility'].dtype}")
    origin = _as_string_array(ds["origin"].values)
    target = _as_string_array(ds["target"].values)
    if not np.array_equal(origin, target):
        raise ValueError("origin and target coordinates must match (square OD)")
    return ds.assign_coords(origin=("origin", origin), target=("target", target))


def _diagonal_mask(raw: xr.DataArray) -> xr.DataArray:
    n = raw.sizes["origin"]
    return xr.DataArray(np.eye(n, dtype=bool), dims=("origin", "target"))


def _policy_keep_mask(
    raw: xr.DataArray, self_flows: str, min_flow_raw: float
) -> xr.DataArray:
    keep = (raw > 0) & (raw >= min_flow_raw)
    if self_flows == "drop":
        keep = keep & ~_diagonal_mask(raw)
    elif self_flows != "retain":
        raise ValueError(f"unknown self_flows policy: {self_flows!r}")
    return keep


def compute_od_audit(
    raw: xr.DataArray,
    self_flow_policies: tuple[str, ...] = SELF_FLOW_POLICIES,
    min_flow_policies: tuple[float, ...] = MIN_FLOW_RAW_POLICIES,
) -> dict:
    """Raw-trip-space construction checks (§3) as a JSON-safe attrs dict.

    ``diag_mass_fraction`` is the mean over dates of the per-date diagonal
    share of total raw mass. Per policy, ``retained_mass_fraction`` is the
    fraction of total raw mass kept after masking and ``edge_count`` is the
    number of retained (date, origin, target) entries summed over all dates.
    Reductions stay lazy on dask-backed inputs and evaluate in one
    ``xr.compute`` pass (chunk-streamed; never materializes the tensor).
    """
    diag = _diagonal_mask(raw)
    total_mass = raw.sum()
    per_date_total = raw.sum(dim=("origin", "target"))
    per_date_diag = (raw * diag).sum(dim=("origin", "target"))
    lazy: dict[str, xr.DataArray] = {
        "diag_mass_fraction": (per_date_diag / per_date_total).mean(dim="date"),
        "total_raw_mass": total_mass,
    }
    for self_flows in self_flow_policies:
        for min_flow_raw in min_flow_policies:
            keep = _policy_keep_mask(raw, self_flows, min_flow_raw)
            key = policy_key(self_flows, min_flow_raw)
            lazy[f"retained__{key}"] = (raw * keep).sum() / total_mass
            lazy[f"edges__{key}"] = keep.sum()
    computed = dict(zip(lazy, dask.compute(*lazy.values()), strict=True))

    audit: dict = {
        "diag_mass_fraction": float(computed["diag_mass_fraction"]),
        "total_raw_mass": float(computed["total_raw_mass"]),
        "retained_mass_fraction": {},
        "edge_count": {},
    }
    for self_flows in self_flow_policies:
        for min_flow_raw in min_flow_policies:
            key = policy_key(self_flows, min_flow_raw)
            audit["retained_mass_fraction"][key] = float(
                computed[f"retained__{key}"]
            )
            audit["edge_count"][key] = int(computed[f"edges__{key}"])
    return audit


def reconcile_zone_counts(
    zone_ids: np.ndarray, reference_csv: str | Path
) -> dict:
    """Match OD zone IDs against ``muni_code_5digit`` municipal codes."""
    reference = pd.read_csv(reference_csv, dtype={"muni_code_5digit": str})
    reference_ids = set(reference["muni_code_5digit"].dropna().astype(str))
    od_ids = {str(z) for z in np.asarray(zone_ids)}
    od_unmatched = sorted(od_ids - reference_ids)
    reference_unmatched = sorted(reference_ids - od_ids)
    return {
        "reference_column": "muni_code_5digit",
        "od_zone_count": len(od_ids),
        "reference_count": len(reference_ids),
        "matched": len(od_ids & reference_ids),
        "od_unmatched_count": len(od_unmatched),
        "od_unmatched_ids": od_unmatched,
        "reference_unmatched_count": len(reference_unmatched),
        "reference_unmatched_ids": reference_unmatched,
    }


def cases_footprint_mask(
    zone_ids: np.ndarray, cases_store: str | Path
) -> np.ndarray:
    """Bool mask: zone present in the cases store ``region_id`` coordinate.

    Opens the store lazily and reads only that coordinate array — never the
    case tensor (§10.3(7)).
    """
    with xr.open_zarr(cases_store) as cases:  # type: ignore[arg-type]
        region_ids = {str(v) for v in cases["region_id"].values}
    return np.asarray(
        [str(z) in region_ids for z in np.asarray(zone_ids)], dtype=bool
    )


def build_od_municipal_store(
    source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
    reference_csv: str | Path = DEFAULT_REFERENCE_CSV,
    cases_store: str | Path = DEFAULT_CASES_STORE,
) -> dict:
    """Build the §11.3 ``od_municipal.zarr`` store; return its audit attrs."""
    raw_ds = load_raw_od(source)
    raw = raw_ds["mobility"]
    zone_ids = raw_ds["origin"].values
    dates = raw_ds["date"].values

    audit = compute_od_audit(raw)
    audit["zone_count_reconciliation"] = reconcile_zone_counts(
        zone_ids, reference_csv
    )
    audit.update(
        {
            "od_source": "municipal",
            "source_path": str(source),
            "transform": "mobility = log1p(clip(raw_trips, 0)) stored as float16",
            "policy_semantics": POLICY_DESCRIPTION,
            "policies": {
                "self_flows": list(SELF_FLOW_POLICIES),
                "min_flow_raw": list(MIN_FLOW_RAW_POLICIES),
            },
            "date_count": int(raw.sizes["date"]),
            "date_range": [
                pd.Timestamp(dates[0]).date().isoformat(),
                pd.Timestamp(dates[-1]).date().isoformat(),
            ],
        }
    )

    footprint = cases_footprint_mask(zone_ids, cases_store)
    stored = xr.Dataset(
        data_vars={
            # Version-stable clip: xarray >=2025.01 removed clip(min=, max=).
            "mobility": np.log1p(xr.where(raw < 0, 0, raw)).astype("float16"),
            "zone_id": ("zone", zone_ids),
            "zone_in_cases_footprint": ("zone", footprint),
        },
        attrs=audit,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stored.to_zarr(output, mode="w")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the municipal-projection OD store (MOB-1, §11.3)."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--cases-store", type=Path, default=DEFAULT_CASES_STORE)
    args = parser.parse_args(argv)
    audit = build_od_municipal_store(
        source=args.source,
        output=args.out,
        reference_csv=args.reference_csv,
        cases_store=args.cases_store,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
