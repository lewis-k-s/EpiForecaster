"""
Processor for Catalonia COVID-19 vaccination coverage by municipality.

Data source:
Vacunacio per al COVID-19: dosis administrades per municipi
https://analisi.transparenciacatalunya.cat/d/irki-p3c7
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from ..utils import load_csv_with_string_ids


class VaccinationProcessor:
    """Convert daily administered first-dose counts to cumulative coverage rates."""

    COLUMN_MAPPING = {
        "municipi_codi": "municipality_code",
        "data": "date",
        "dosi": "dose",
        "fabricant": "manufacturer",
        "no_vacunat": "not_vaccinated",
        "recompte": "count",
    }

    DTYPES = {
        "municipi_codi": str,
        "municipi": str,
        "data": str,
        "dosi": str,
        "fabricant": str,
        "no_vacunat": str,
        "recompte": str,
    }

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def _load_raw_data(self, vaccination_file: str | Path) -> pd.DataFrame:
        path = Path(vaccination_file)
        if not path.exists():
            raise FileNotFoundError(f"Vaccination file not found: {path}")

        print(f"  Loading vaccinations from {path}")
        df = pd.read_csv(path, dtype=self.DTYPES)
        df = df.rename(columns=self.COLUMN_MAPPING)

        required = {"municipality_code", "date", "dose", "manufacturer", "count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required vaccination columns: {sorted(missing)}. "
                f"Available columns: {sorted(df.columns)}"
            )

        df = df[df["municipality_code"].notna() & (df["municipality_code"] != "")]
        df["municipality_code"] = df["municipality_code"].astype(str)
        df = df[~df["municipality_code"].str.lower().isin({"nan", "none"})]
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df["date"] = df["date"].dt.floor("D")
        df["dose"] = pd.to_numeric(df["dose"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df = df.dropna(subset=["municipality_code", "date", "dose", "count"])
        df = df[df["count"] >= 0]

        print(f"  Loaded {len(df):,} vaccination records")
        if not df.empty:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Unique municipalities: {df['municipality_code'].nunique()}")
        return df

    def _load_population(self) -> pd.Series:
        population = load_csv_with_string_ids(
            self.config.population_file,
            usecols=["id", "d.population"],
        )
        population = population.rename(
            columns={"id": "municipality_code", "d.population": "population"}
        )
        population = population[
            population["municipality_code"].notna()
            & (population["municipality_code"] != "")
        ]
        population["municipality_code"] = population["municipality_code"].astype(str)
        population = population[
            ~population["municipality_code"].str.lower().isin({"nan", "none"})
        ]
        population["population"] = pd.to_numeric(
            population["population"], errors="coerce"
        )
        population = population.dropna(subset=["municipality_code", "population"])
        population = population[population["population"] > 0]
        return population.set_index("municipality_code")["population"]

    def _aggregate_first_doses(self, df: pd.DataFrame) -> pd.DataFrame:
        vaccinated = df[df["dose"] == 1].copy()
        if "not_vaccinated" in vaccinated.columns:
            vaccinated = vaccinated[vaccinated["not_vaccinated"].isna()]
        vaccinated = vaccinated[vaccinated["manufacturer"] != "No administrada"]

        aggregated = (
            vaccinated.groupby(["date", "municipality_code"], dropna=False)["count"]
            .sum()
            .reset_index()
            .rename(columns={"count": "first_doses"})
        )
        print(f"  Aggregated to {len(aggregated):,} municipality-day first-dose rows")
        return aggregated

    def _build_dataset(self, first_doses: pd.DataFrame) -> xr.Dataset:
        date_range = pd.date_range(
            start=self.config.start_date, end=self.config.end_date, freq="D"
        )
        population = self._load_population()

        if first_doses.empty:
            region_ids = population.index.astype(str)
            daily = pd.DataFrame(0.0, index=date_range, columns=region_ids)
            observed = pd.DataFrame(False, index=date_range, columns=region_ids)
        else:
            daily = first_doses.pivot_table(
                index="date",
                columns="municipality_code",
                values="first_doses",
                aggfunc="sum",
            )
            daily.columns = daily.columns.astype(str)
            all_regions = population.index.union(daily.columns).astype(str)
            daily = daily.reindex(index=date_range, columns=all_regions)
            observed = daily.notna()
            daily = daily.fillna(0.0)

        cumulative = daily.cumsum(axis=0)
        population = population.reindex(cumulative.columns)
        rate = cumulative.divide(population, axis="columns")
        rate = rate.clip(lower=0.0, upper=1.0).fillna(0.0)

        age = self._compute_age_from_mask(observed.reindex_like(rate).fillna(False))

        rate.index.name = TEMPORAL_COORD
        rate.columns.name = REGION_COORD
        observed.index.name = TEMPORAL_COORD
        observed.columns.name = REGION_COORD
        age.index.name = TEMPORAL_COORD
        age.columns.name = REGION_COORD

        rate_da = xr.DataArray(
            rate.values.astype(np.float32),
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={TEMPORAL_COORD: rate.index, REGION_COORD: rate.columns.astype(str)},
            name="vaccination_rate",
            attrs={
                "description": "Cumulative first-dose COVID-19 vaccination coverage",
                "source": "https://analisi.transparenciacatalunya.cat/d/irki-p3c7",
            },
        ).expand_dims(run_id=["real"])

        mask_da = xr.DataArray(
            observed.values.astype(bool),
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: observed.index,
                REGION_COORD: observed.columns.astype(str),
            },
            name="vaccination_rate_mask",
        ).expand_dims(run_id=["real"])

        age_da = xr.DataArray(
            age.values.astype(np.uint8),
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={TEMPORAL_COORD: age.index, REGION_COORD: age.columns.astype(str)},
            name="vaccination_rate_age",
        ).expand_dims(run_id=["real"])

        print(
            f"  Processed vaccination_rate: {rate_da.sizes[TEMPORAL_COORD]} dates x "
            f"{rate_da.sizes[REGION_COORD]} regions"
        )
        return xr.Dataset(
            {
                "vaccination_rate": rate_da,
                "vaccination_rate_mask": mask_da,
                "vaccination_rate_age": age_da,
            }
        )

    @staticmethod
    def _compute_age_from_mask(mask: pd.DataFrame, max_age: int = 14) -> pd.DataFrame:
        age = pd.DataFrame(max_age, index=mask.index, columns=mask.columns, dtype=np.uint8)
        for column in mask.columns:
            last_seen = None
            values = mask[column].to_numpy(dtype=bool)
            for idx, observed in enumerate(values):
                if observed:
                    last_seen = idx
                    age.iat[idx, age.columns.get_loc(column)] = 1
                elif last_seen is not None:
                    age.iat[idx, age.columns.get_loc(column)] = min(
                        idx - last_seen + 1, max_age
                    )
        return age

    def process(self, vaccination_file: str | Path) -> xr.Dataset:
        print("Processing vaccination data")
        raw = self._load_raw_data(vaccination_file)
        first_doses = self._aggregate_first_doses(raw)
        return self._build_dataset(first_doses)
