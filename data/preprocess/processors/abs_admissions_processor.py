"""
Processor for native ABS (Àrees Bàsiques de Salut) weekly admissions data.

Builds the ``obs_admissions`` dataset (plan §11.3) from the raw Catalan
surveillance CSV (``COVID-19__Persones_hospitalitzades.csv``), whose rows are
per (epi-week, ABS, sex, age group).

Weekly intervals are native to this source: no smoothing, no daily
interpolation, no transforms at rest.

- ``value``: weekly summed raw admissions (float32).
- ``mask``: true observation only — a reported zero-case week is value 0
  with mask 1; unreported ABS-weeks are value 0 with mask 0.
- ``age``: reporting opportunities since the last reported week, capped at
  14; weeks before an ABS's first report are 14.
- ``stratum_pop_sum``: weekly sum of the case-only per-stratum ``poblacio``
  over reported strata — a lower bound on ABS population, QC only, NOT a
  denominator (§10.3(4)); the audited denominator arrives at native_graph
  assembly.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import TEMPORAL_COORD, PreprocessingConfig

ABS_FILE = "COVID-19__Persones_hospitalitzades.csv"

ABS_ID_COORD = "abs_id"

# Reporting-opportunity cap, mirroring the daily convention of the
# hospitalizations processor adapted to weeks.
MAX_REPORTING_GAP_WEEKS = 14

# Relative deviation of a week's stratum_pop_sum from the ABS's window
# median above which the case-only reporting noise is flagged.
POP_DEVIATION_TOLERANCE = 0.01


class ABSAdmissionsProcessor:
    """
    Converts raw weekly ABS admission strata to a weekly xarray Dataset.

    Processing pipeline:
    1. Load the raw (epi-week, ABS, sex, age group) CSV, dropping only the
       structural sentinel triple (never ``index_socioeconomic == -1``:
       real newer ABS carry it with genuine admissions, §10.3(4)).
    2. Aggregate strata to ABS-weeks (admissions and stratum_pop_sum sums).
    3. Place onto the complete Monday-week grid inside the configured
       window, marking unreported ABS-weeks as missing.
    4. Emit dims ``(date=week_start, abs_id)`` with an aligned ``week_end``
       coordinate and ``value``/``mask``/``age``/``stratum_pop_sum``.
    """

    ABS_FILE = "COVID-19__Persones_hospitalitzades.csv"

    COLUMN_MAPPING = {
        "data_inici": "week_start",  # DD/MM/YYYY, always a Monday
        "data_final": "week_end",  # DD/MM/YYYY, always the same week's Sunday
        "codi_abs": "abs_id",  # 3-char zero-padded string
        "sexe": "sex",
        "grup_edat": "age_group",
        "casos": "admissions",
        "poblacio": "stratum_population",  # per (sex, age_group) stratum, NOT per ABS
    }

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the ABS admissions processor.

        Args:
            config: Preprocessing configuration with date ranges
        """
        self.config = config

    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw ABS admissions from CSV, dropping sentinel rows."""
        abs_file = data_dir / self.ABS_FILE

        if not abs_file.exists():
            raise FileNotFoundError(f"ABS admissions file not found: {abs_file}")

        print(f"  Loading ABS admissions from {abs_file}")

        # String dtype for the code columns preserves leading zeros
        df = pd.read_csv(
            abs_file,
            dtype={"codi_abs": str, "codi_regio": str, "codi_ambit": str},
        )

        # Sentinel filter on the ORIGINAL columns (§10.3(4)): the structural
        # triple only. The three conditions coincide exactly on the real
        # file (311 Catalonia-total rows); OR-semantics keeps the filter
        # robust to any one of them appearing alone.
        sentinel = (
            (df["codi_abs"] == "000")
            | (df["codi_regio"] == "00")
            | (df["sexe"] == "No disponible")
            | (df["grup_edat"] == "No disponible")
        )
        df = df[~sentinel]

        df = df.drop(columns=["codi_regio", "codi_ambit"], errors="ignore")
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Drop rows without a usable ABS code
        df = df[df["abs_id"].notna() & (df["abs_id"] != "")]
        df["abs_id"] = df["abs_id"].astype(str)

        # Parse week boundaries (DD/MM/YYYY format)
        for col in ("week_start", "week_end"):
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")

        df = df.dropna(subset=["week_start", "week_end", "admissions"])

        # Remove negative values
        df = df[df["admissions"] >= 0]

        print(f"  Loaded {len(df):,} ABS admission records")
        print(f"  Week range: {df['week_start'].min()} to {df['week_start'].max()}")
        print(f"  Unique ABS: {df['abs_id'].nunique()}")

        return df

    def _aggregate_to_abs_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sex x age-group strata to ABS-week sums."""
        aggregated = (
            df.groupby(["week_start", "abs_id"], dropna=False)
            .agg(
                admissions=("admissions", "sum"),
                stratum_pop_sum=("stratum_population", "sum"),
            )
            .reset_index()
        )

        self._warn_on_population_drift(aggregated)

        print(f"  Aggregated to {len(aggregated)} ABS-week records")
        return aggregated

    def _warn_on_population_drift(self, aggregated: pd.DataFrame) -> None:
        """
        Flag stratum_pop_sum noise against each ABS's window median.

        Per-stratum ``poblacio`` is case-only (only case-bearing strata are
        reported), so the weekly sum is a noisy lower bound of the ABS
        population rather than a stable denominator (§10.3(4)).
        """
        deviating_abs = 0
        deviating_weeks = 0
        for _, sums in aggregated.groupby("abs_id")["stratum_pop_sum"]:
            median = sums.median()
            if median <= 0:
                continue
            deviation = (sums - median).abs() / median
            n_deviating = int((deviation > POP_DEVIATION_TOLERANCE).sum())
            if n_deviating:
                deviating_abs += 1
                deviating_weeks += n_deviating

        if deviating_weeks:
            warnings.warn(
                f"stratum_pop_sum deviates >{POP_DEVIATION_TOLERANCE:.0%} from "
                f"the ABS window median for {deviating_abs} ABS across "
                f"{deviating_weeks} ABS-weeks; per-stratum poblacio is "
                "case-only, so the sum is a lower bound (QC only, not a "
                "denominator)",
                stacklevel=2,
            )

    def _build_week_grid(self) -> pd.DatetimeIndex:
        """Complete Monday weeks fully inside the configured window."""
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)

        # First Monday on or after the window start (Monday=0)
        first = start + pd.Timedelta(days=(7 - start.weekday()) % 7)

        grid = pd.date_range(start=first, end=end, freq="W-MON")
        # A week is inside the window only if its Sunday (start + 6d) fits
        return grid[grid + pd.Timedelta(days=6) <= end]

    def _compute_age(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        Reporting opportunities since the last reported week, capped at 14.

        Mirrors the daily convention of the hospitalizations processor:
        age = 1 on a reported week, +1 per subsequent unreported week,
        clipped at 14; weeks before an ABS's first report are 14.
        """
        mask = np.asarray(mask_2d, dtype=bool)
        n_weeks = mask.shape[0]

        rows = np.arange(n_weeks)[:, None]
        group_id = np.cumsum(mask, axis=0)
        # Row that started each row's current reported-run (0 before any
        # report; those rows are overridden to the cap below).
        run_start = np.maximum.accumulate(np.where(mask, rows, 0), axis=0)
        age = rows - run_start + 1
        age = np.where(group_id == 0, MAX_REPORTING_GAP_WEEKS, age)
        return np.minimum(age, MAX_REPORTING_GAP_WEEKS).astype(np.uint8)

    def process(
        self,
        data_dir: str | Path,
    ) -> xr.Dataset:
        """
        Process ABS admission data into a weekly xarray Dataset.

        Args:
            data_dir: Directory containing the ABS admissions CSV file

        Returns:
            xarray Dataset with dims (date, abs_id) and variables:
            - value: weekly summed raw admissions (float32)
            - mask: true observation only (bool)
            - age: reporting opportunities since last report (uint8, cap 14)
            - stratum_pop_sum: case-only stratum population sum (float32)
        """
        print("Processing native ABS admissions data")

        data_dir = Path(data_dir)

        raw_df = self._load_raw_data(data_dir)

        if raw_df.empty:
            print("  Warning: No ABS admissions data found")
            return self._create_empty_dataset()

        # Keep only whole weeks inside the configured window
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        raw_df = raw_df[(raw_df["week_start"] >= start) & (raw_df["week_end"] <= end)]

        if raw_df.empty:
            print("  Warning: No ABS admissions data in specified date range")
            return self._create_empty_dataset()

        weekly_df = self._aggregate_to_abs_week(raw_df)

        grid = self._build_week_grid()
        abs_ids = sorted(weekly_df["abs_id"].unique())

        value_pivot = (
            weekly_df.pivot(index="week_start", columns="abs_id", values="admissions")
            .reindex(index=grid, columns=abs_ids)
            .astype(np.float64)
        )
        pop_pivot = (
            weekly_df.pivot(
                index="week_start", columns="abs_id", values="stratum_pop_sum"
            )
            .reindex(index=grid, columns=abs_ids)
            .astype(np.float64)
        )

        mask_2d = value_pivot.notna().to_numpy()
        values = value_pivot.fillna(0.0).to_numpy()
        stratum_pop_sum = pop_pivot.fillna(0.0).to_numpy()
        age = self._compute_age(mask_2d)

        result_ds = xr.Dataset(
            {
                "value": ((TEMPORAL_COORD, ABS_ID_COORD), values.astype(np.float32)),
                "mask": ((TEMPORAL_COORD, ABS_ID_COORD), mask_2d),
                "age": ((TEMPORAL_COORD, ABS_ID_COORD), age),
                "stratum_pop_sum": (
                    (TEMPORAL_COORD, ABS_ID_COORD),
                    stratum_pop_sum.astype(np.float32),
                ),
            },
            coords={
                TEMPORAL_COORD: grid,
                "week_end": (TEMPORAL_COORD, grid + pd.Timedelta(days=6)),
                ABS_ID_COORD: abs_ids,
            },
            attrs=self._dataset_attrs(n_raw_rows=len(raw_df)),
        )

        n_observed = int(mask_2d.any(axis=0).sum())
        print(
            f"  Processed admissions: {result_ds.sizes[TEMPORAL_COORD]} weeks x "
            f"{result_ds.sizes[ABS_ID_COORD]} ABS"
        )
        print(f"  Observed ABS (any reported week): {n_observed}")
        print(f"  Total admissions: {float(result_ds['value'].sum()):.0f}")

        return result_ds

    def _create_empty_dataset(self) -> xr.Dataset:
        """Create empty dataset with proper structure when no data available."""
        grid = self._build_week_grid()
        shape = (len(grid), 0)

        return xr.Dataset(
            {
                "value": (
                    (TEMPORAL_COORD, ABS_ID_COORD),
                    np.zeros(shape, dtype=np.float32),
                ),
                "mask": (
                    (TEMPORAL_COORD, ABS_ID_COORD),
                    np.zeros(shape, dtype=bool),
                ),
                "age": (
                    (TEMPORAL_COORD, ABS_ID_COORD),
                    np.full(shape, MAX_REPORTING_GAP_WEEKS, dtype=np.uint8),
                ),
                "stratum_pop_sum": (
                    (TEMPORAL_COORD, ABS_ID_COORD),
                    np.zeros(shape, dtype=np.float32),
                ),
            },
            coords={
                TEMPORAL_COORD: grid,
                "week_end": (TEMPORAL_COORD, grid + pd.Timedelta(days=6)),
                ABS_ID_COORD: [],
            },
            attrs=self._dataset_attrs(n_raw_rows=0),
        )

    def _dataset_attrs(self, n_raw_rows: int) -> dict[str, str | int]:
        """Provenance and schema attributes for the obs_admissions store."""
        return {
            "schema": "obs_admissions v1 (plan §11.3)",
            "source_file": self.ABS_FILE,
            "raw_rows_retained": n_raw_rows,
            "sentinel_filter": (
                "structural triple: codi_abs=='000' OR codi_regio=='00' OR "
                "'No disponible'; index_socioeconomic==-1 retained (real "
                "newer ABS carry it with genuine admissions, §10.3(4))"
            ),
            "week_convention": "epi weeks Monday (date) to Sunday (week_end)",
            "population_caveat": (
                "stratum_pop_sum sums case-only per-stratum poblacio over "
                "reported strata: a weekly lower bound on ABS population, "
                "QC only, NOT a denominator (§10.3(4))"
            ),
        }


def to_obs_store(ds: xr.Dataset, path: str | Path) -> None:
    """
    Write an obs store as zarr v3 per plan §11.3.

    Falls back to the plain ``to_zarr`` call for xarray versions without
    the ``zarr_format`` kwarg.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds.to_zarr(path, mode="w", zarr_format=3)
    except TypeError:
        ds.to_zarr(path, mode="w")
