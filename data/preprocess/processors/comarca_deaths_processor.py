"""
Native comarca deaths processor for the obs_deaths store (plan §11.3).

Unlike ``deaths_processor.py`` — which consumes pre-projected municipal
deaths (weighted by polygon overlap) and applies Kalman smoothing — this
processor keeps deaths on their native comarca support:

- parses the raw Generalitat registry CSV (comarca x sex daily rows),
- aggregates sexes into one daily count per comarca,
- emits ``value`` / ``mask`` / ``age`` / ``population`` on dims
  ``(date, comarca_id)`` per the §11.3 ``obs_deaths`` schema,
- stores raw counts at rest: no per-100k/log transform and no smoothing
  (smoothing belongs to the legacy municipal path only).

Sentinel ruling (orchestrator): ``Codi Comarca == "00"`` ("No classificat")
rows carry genuine reported counts, so they are kept as a regular
``comarca_id`` column with ``population = NaN`` — the pseudo-comarca has no
municipality membership in ``mpiscatalunya.csv``, so no denominator is
fabricated for it. The comarca source registry (REG-1) excludes "00" from
its territory; downstream eligibility (native graph assembly) decides
whether the pseudo-comarca is queryable. The ``unmapped_comarca_ids`` /
``unobserved_mapped_comarca_ids`` dataset attrs record the reconciliation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import TEMPORAL_COORD, PreprocessingConfig
from .municipality_mapping_processor import MunicipalityMappingProcessor

COMARCA_COORD = "comarca_id"
AGE_CAP = 14


def to_obs_store(ds: xr.Dataset, path: str | Path) -> Path:
    """Write an obs_deaths dataset to a zarr v3 store (plan §11.3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=3)
    return path


class ComarcaDeathsProcessor:
    """
    Converts Catalonia COVID-19 deaths data to a native comarca xarray Dataset.

    Deaths are reported at comarca level (per sex). This processor:
    - Loads the raw comarca deaths registry CSV
    - Aggregates sexes into one daily count per comarca
    - Keeps absent (date, comarca) rows missing (mask 0, value NaN, §11.7)
    - Computes comarca population as the sum of member municipal populations
    """

    DEATHS_FILE = (
        "Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    )
    MAPPING_FILE = MunicipalityMappingProcessor.MUNICIPALITY_MAPPING_FILE
    POPULATION_FILE = "fl_population_por_municipis.csv"

    COLUMN_MAPPING = {
        "Data defunció": "date",
        "Codi Comarca": "comarca_code",
        "Comarca": "comarca_name",
        "Codi Sexe": "sex_code",
        "Sexe": "sex_name",
        "Nombre defuncions": "deaths",
    }

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the comarca deaths processor."""
        self.config = config

    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw comarca deaths data from CSV."""
        deaths_file = data_dir / self.DEATHS_FILE

        if not deaths_file.exists():
            raise FileNotFoundError(f"Deaths file not found: {deaths_file}")

        print(f"  Loading comarca deaths from {deaths_file}")

        # Load the CSV file with string dtype for comarca codes to preserve leading zeros
        df = pd.read_csv(deaths_file, dtype={"Codi Comarca": str})

        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)

        # Drop rows where comarca_code is NaN or empty string
        df = df[df["comarca_code"].notna() & (df["comarca_code"] != "")]

        # Comarca codes are 2-char zero-padded strings
        df["comarca_code"] = df["comarca_code"].astype(str).str.strip()

        # Parse dates from DD/MM/YYYY format
        df["date"] = (
            pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
            .dt.tz_localize(None)
            .dt.floor("D")
        )

        # Remove invalid rows
        df = df.dropna(subset=["date", "deaths"])

        # Remove negative deaths if any
        df = df[df["deaths"] >= 0]

        print(f"  Loaded {len(df):,} death records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique comarcas: {df['comarca_code'].nunique()}")

        return df

    def _aggregate_sexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-sex rows into one daily count per comarca.

        Rows with an unclassified sex (NaN ``sex_code``, e.g. the "00"
        sentinel) carry real counts and are included in the sums.
        """
        aggregated = (
            df.groupby(["date", "comarca_code"], dropna=False)
            .agg(deaths=("deaths", "sum"), comarca_name=("comarca_name", "first"))
            .reset_index()
        )

        return aggregated

    def _compute_population(self, data_dir: Path) -> pd.Series:
        """Compute comarca populations as sums of member municipal populations.

        Comarcas without mapping membership (e.g. the "00" sentinel) sum to
        NaN via ``min_count=1`` — no fabricated denominator.
        """
        population_file = data_dir / self.POPULATION_FILE

        if not population_file.exists():
            raise FileNotFoundError(f"Population file not found: {population_file}")

        mapping = MunicipalityMappingProcessor(data_dir).load_mapping()

        population = pd.read_csv(
            population_file,
            usecols=["id", "d.population"],
            dtype={"id": str},
        ).rename(columns={"d.population": "population"})
        population["id"] = population["id"].str.strip()

        # Mapping codes are 6-char Idescat codes; the first 5 chars are the
        # municipal id used by the population file (verified join key).
        merged = mapping.merge(
            population,
            how="left",
            left_on=mapping["municipality_code"].str[:5],
            right_on="id",
        )

        comarca_population = merged.groupby("comarca_code")["population"].sum(
            min_count=1
        )
        comarca_population.index.name = "comarca_code"

        return comarca_population

    @staticmethod
    def _age_from_mask(mask: np.ndarray) -> np.ndarray:
        """Compute the age channel from an observation mask.

        age = 1 on an observation day, k + 1 at k days after the last
        observation, AGE_CAP before the first observation (verified against
        the canonical municipal store convention).
        """
        idx = np.arange(mask.shape[0], dtype=np.int64)[:, None]
        last_obs = np.where(mask, idx, -1)
        np.maximum.accumulate(last_obs, axis=0, out=last_obs)
        age = idx - last_obs + 1
        age = np.where(last_obs < 0, AGE_CAP, age)
        return np.clip(age, 1, AGE_CAP).astype(np.uint8)

    def _dataset_attrs(
        self, file_codes: set[str], mapping_codes: set[str]
    ) -> dict[str, str | int]:
        """Build the obs_deaths provenance/metadata attrs (plan §11.3)."""
        return {
            "schema": "obs_deaths",
            "schema_version": 1,
            "source_type": "comarca",
            "head": "deaths",
            "source_file": self.DEATHS_FILE,
            "reporting_cadence_days": 1,
            "absent_row_policy": (
                "absent (date, comarca) rows are missing: mask 0, value NaN; "
                "genuine reported zeros are value 0 with mask 1 (plan §11.7)"
            ),
            "age_semantics": (
                "days since last reported row plus 1, capped at 14; "
                "14 before the first reported row"
            ),
            "population_source": (
                "sum of fl_population_por_municipis.csv d.population over "
                "mpiscatalunya.csv comarca membership"
            ),
            "transforms": "none: raw counts at rest (plan §11.3)",
            "unmapped_comarca_ids": ",".join(sorted(file_codes - mapping_codes)),
            "unobserved_mapped_comarca_ids": ",".join(
                sorted(mapping_codes - file_codes)
            ),
        }

    def process(self, data_dir: str | Path) -> xr.Dataset:
        """
        Process comarca deaths data into an obs_deaths xarray Dataset.

        Args:
            data_dir: Directory containing the raw input CSV files

        Returns:
            xarray Dataset with value/mask/age/population on dims
            (date, comarca_id)
        """
        print("Processing Catalonia comarca deaths data")

        data_dir = Path(data_dir)

        # Load raw data
        raw = self._load_raw_data(data_dir)
        file_codes = set(raw["comarca_code"].unique())
        mapping_codes = set(
            MunicipalityMappingProcessor(data_dir)
            .load_mapping()["comarca_code"]
            .unique()
        )

        # Filter to config date range
        windowed = raw[
            (raw["date"] >= self.config.start_date)
            & (raw["date"] <= self.config.end_date)
        ]

        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )

        if windowed.empty:
            print("  Warning: No comarca death data in specified date range")
            shape = (len(date_range), 0)
            return xr.Dataset(
                {
                    "value": (
                        (TEMPORAL_COORD, COMARCA_COORD),
                        np.full(shape, np.nan, dtype="float32"),
                    ),
                    "mask": (
                        (TEMPORAL_COORD, COMARCA_COORD),
                        np.zeros(shape, dtype=bool),
                    ),
                    "age": (
                        (TEMPORAL_COORD, COMARCA_COORD),
                        self._age_from_mask(np.zeros(shape, dtype=bool)),
                    ),
                    "population": ((COMARCA_COORD,), np.zeros(0, dtype="float32")),
                },
                coords={
                    TEMPORAL_COORD: date_range,
                    COMARCA_COORD: np.array([], dtype=object),
                    "comarca_name": ((COMARCA_COORD,), np.array([], dtype=object)),
                },
                attrs=self._dataset_attrs(
                    file_codes=file_codes, mapping_codes=mapping_codes
                ),
            )

        # Aggregate sexes to comarca-day
        aggregated = self._aggregate_sexes(windowed)

        # Sparse pivot (NaN where unobserved) on the complete daily grid,
        # sorted comarca ids for deterministic coord ordering
        pivot = aggregated.pivot(
            index="date", columns="comarca_code", values="deaths"
        ).reindex(date_range)
        pivot = pivot.sort_index(axis=1)
        pivot.columns.name = COMARCA_COORD
        pivot.index.name = TEMPORAL_COORD

        value = pivot.to_numpy(dtype="float32")
        mask = ~np.isnan(value)
        age = self._age_from_mask(mask)
        population = (
            self._compute_population(data_dir)
            .reindex(pivot.columns)
            .to_numpy(dtype="float32")
        )
        comarca_names = (
            aggregated.groupby("comarca_code")["comarca_name"]
            .first()
            .reindex(pivot.columns)
        )

        result_ds = xr.Dataset(
            {
                "value": ((TEMPORAL_COORD, COMARCA_COORD), value),
                "mask": ((TEMPORAL_COORD, COMARCA_COORD), mask),
                "age": ((TEMPORAL_COORD, COMARCA_COORD), age),
                "population": ((COMARCA_COORD,), population),
            },
            coords={
                TEMPORAL_COORD: date_range,
                COMARCA_COORD: pivot.columns.astype(str),
                "comarca_name": (
                    (COMARCA_COORD,),
                    comarca_names.to_numpy(dtype=object),
                ),
            },
            attrs=self._dataset_attrs(
                file_codes=file_codes, mapping_codes=mapping_codes
            ),
        )

        print(
            f"  Processed comarca deaths: {result_ds.sizes[TEMPORAL_COORD]} dates x "
            f"{result_ds.sizes[COMARCA_COORD]} comarcas"
        )
        print(f"  Total observed deaths: {int(np.nansum(value))}")

        return result_ds
