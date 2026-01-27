# Catalonia Open Data Preprocessing Implementation Guide

This specification provides complete, actionable code for implementing a new preprocessing pipeline using official Catalonia open data sources. The goal is to replace the current flowmaps-based cases data with more defensible official sources.

## Executive Summary

**Objective**: Create a new `CasesProcessor` that ingests official Catalonia open data CSV files and produces the same xarray output format as the existing processor.

**Key Challenge**: The raw data sources have different spatial resolutions (municipality vs comarca) and require careful mapping to the canonical `region_id` coordinate used throughout the pipeline.

**Output Format**: `xr.Dataset` with dims `(date, region_id)` containing a `cases` variable matching the existing `CasesProcessor` output.

---

## 1. Raw Data Files

Located at `/Volumes/HUBSSD/code/EpiForecaster/data/files/`:

| File | Rows | Description | Spatial Resolution | Temporal Resolution |
|------|------|-------------|-------------------|---------------------|
| `Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv` | 523,565 | COVID-19 cases | Municipality | Daily |
| `Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv` | 10,900 | Deaths | Comarca | Daily |
| `COVID-19__Persones_hospitalitzades.csv` | 163,238 | Hospitalizations (SIVIC) | Health region | Weekly |
| `censph540mun.csv` | 69,770 | Population | Municipality | Annual snapshot |
| `mpiscatalunya.csv` | 947 | Municipality geometry/comarca mapping | Municipality | Static |

---

## 2. Column Mappings and Data Types

### 2.1 Cases Data (`Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv`)

**Raw header**:
```
TipusCasData,ComarcaCodi,ComarcaDescripcio,MunicipiCodi,MunicipiDescripcio,DistricteCodi,DistricteDescripcio,SexeCodi,SexeDescripcio,TipusCasDescripcio,NumCasos
```

**Column rename mapping**:
```python
CASES_COLUMN_MAPPING = {
    "TipusCasData": "date",                    # Case date (symptom onset)
    "MunicipiCodi": "municipality_code",       # Municipality code (preserve leading zeros)
    "MunicipiDescripcio": "municipality_name",
    "ComarcaCodi": "comarca_code",             # Comarca code (preserve leading zeros)
    "ComarcaDescripcio": "comarca_name",
    "SexeCodi": "sex_code",                    # 0=Home, 1=Dona, etc.
    "SexeDescripcio": "sex",
    "TipusCasDescripcio": "test_type",         # PCR, TAR, ELISA, etc.
    "NumCasos": "n_cases"
}
```

**Data types** (CRITICAL: preserve leading zeros):
```python
CASES_DTYPES = {
    "TipusCasData": str,           # DD/MM/YYYY format
    "MunicipiCodi": str,           # String to preserve leading zeros (e.g., "08019")
    "ComarcaCodi": str,            # String to preserve leading zeros
    "SexeCodi": str,
    "NumCasos": int
}
```

**Test type categorization** (for inferred testing rate):
```python
TEST_TYPE_CATEGORIES = {
    "Positiu TAR": "TAR",
    "Positiu per Test Ràpid": "TAR",
    "Positiu PCR": "PCR",
    "PCR probable": "PCR",
    "Positiu per ELISA": "ELISA",
    "Epidemiològic": "EPI"
}
```

**Sample rows**:
```csv
08/10/2020,21,MARESME,08121,MATARÓ,,No classificat,1,Dona,Positiu per Test Ràpid,1
05/07/2022,23,NOGUERA,25240,VALLFOGONA DE BALAGUER,,No classificat,1,Dona,Positiu TAR,2
21/05/2022,13,BARCELONES,08019,BARCELONA,03,SANTS-MONTJUÏC,1,Dona,Positiu PCR,4
```

### 2.2 Deaths Data (`Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv`)

**Raw header**:
```
Data defunció,Codi Comarca,Comarca,Codi Sexe,Sexe,Nombre defuncions
```

**Column rename mapping**:
```python
DEATHS_COLUMN_MAPPING = {
    "Data defunció": "date_of_death",
    "Codi Comarca": "comarca_code",            # Comarca code (preserve leading zeros)
    "Comarca": "comarca_name",
    "Codi Sexe": "sex_code",
    "Sexe": "sex",
    "Nombre defuncions": "n_deaths"
}
```

**Data types**:
```python
DEATHS_DTYPES = {
    "Data defunció": str,                     # DD/MM/YYYY format
    "Codi Comarca": str,                       # String to preserve leading zeros
    "Nombre defuncions": int
}
```

### 2.3 Municipality to Comarca Mapping (`mpiscatalunya.csv`)

**File structure** (has 4-line header):
```csv
﻿Municipis 
Institut d'Estadística de Catalunya
https://www.idescat.cat/codis/?id=50&n=9&lang=en
Codi,Nom,Codi comarca,Nom comarca,
250019,Abella de la Conca,25,Pallars Jussà,
080018,Abrera,11,Baix Llobregat,
```

**Loading code** (skip header lines):
```python
muni_to_comarca = pd.read_csv(
    "mpiscatalunya.csv",
    skiprows=4,
    names=["municipality_code", "municipality_name", "comarca_code", "comarca_name"],
    usecols=["municipality_code", "municipality_name", "comarca_code", "comarca_name"],
    dtype={"municipality_code": str, "comarca_code": str}
)
```

---

## 3. Preprocessing Code Implementation

### 3.1 Municipality to Comarca Mapping Processor

Create new file: `/Volumes/HUBSSD/code/EpiForecaster/data/preprocess/processors/municipality_mapping_processor.py`

```python
"""
Processor for municipality to comarca mapping data.

This module handles the loading and processing of the municipality-to-comarca
mapping table for spatial alignment across data sources.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD


class MunicipalityMappingProcessor:
    """
    Loads and processes municipality to comarca mapping data.
    
    The mapping is essential for:
    - Converting comarca-level data (deaths) to municipality-level
    - Validating spatial coverage across data sources
    - Creating population-weighted allocations
    """
    
    MUNICIPALITY_MAPPING_FILE = "mpiscatalunya.csv"
    
    # File has 4-line header before the actual column header
    HEADER_ROWS = 4
    
    def __init__(self, data_dir: str | Path):
        """
        Initialize the mapping processor.
        
        Args:
            data_dir: Directory containing mpiscatalunya.csv
        """
        self.data_dir = Path(data_dir)
        
    def load_mapping(self) -> pd.DataFrame:
        """
        Load municipality to comarca mapping from CSV.
        
        Returns:
            DataFrame with columns:
            - municipality_code: str (5-digit code with leading zeros)
            - municipality_name: str
            - comarca_code: str (2-digit code with leading zeros)
            - comarca_name: str
        """
        mapping_file = self.data_dir / self.MUNICIPALITY_MAPPING_FILE
        
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"Municipality mapping file not found: {mapping_file}"
            )
        
        df = pd.read_csv(
            mapping_file,
            skiprows=self.HEADER_ROWS,
            usecols=[0, 1, 2, 3],  # Codi, Nom, Codi comarca, Nom comarca
            names=[
                "municipality_code",
                "municipality_name",
                "comarca_code",
                "comarca_name"
            ],
            dtype={
                "municipality_code": str,
                "comarca_code": str
            }
        )
        
        # Clean up any trailing commas in data
        df["municipality_code"] = df["municipality_code"].str.strip()
        df["comarca_code"] = df["comarca_code"].str.strip()
        
        # Validate no missing codes
        if df["municipality_code"].isna().any():
            raise ValueError("Missing municipality codes in mapping data")
        if df["comarca_code"].isna().any():
            raise ValueError("Missing comarca codes in mapping data")
            
        print(f"  Loaded {len(df)} municipality -> comarca mappings")
        print(f"  Unique comarcas: {df['comarca_code'].nunique()}")
        
        return df
    
    def get_comarca_for_municipalities(
        self, 
        municipality_codes: list[str] | np.ndarray
    ) -> dict[str, str]:
        """
        Get comarca code for each municipality code.
        
        Args:
            municipality_codes: List of municipality codes
            
        Returns:
            Dictionary mapping municipality_code -> comarca_code
        """
        mapping_df = self.load_mapping()
        
        # Create lookup dictionary
        lookup = dict(zip(
            mapping_df["municipality_code"],
            mapping_df["comarca_code"]
        ))
        
        # Check for missing municipalities
        missing = set(municipality_codes) - set(lookup.keys())
        if missing:
            print(f"  Warning: {len(missing)} municipalities not in mapping")
            
        return lookup
```

### 3.2 New Catalonia Cases Processor

Create new file: `/Volumes/HUBSSD/code/EpiForecaster/data/preprocess/processors/catalonia_cases_processor.py`

```python
"""
Processor for Catalonia official COVID-19 case data.

This module handles the conversion of official Catalonia open data CSV files
into canonical xarray formats. It processes daily case counts at municipality
level and produces output compatible with the existing pipeline.

Data source: Registre de casos de COVID-19 a Catalunya per municipi i sexe
https://datos.gob.es/ca/catalogo/a09002970-registro-de-test-de-covid-19-realizados-en-catalunya-segregacion-por-sexo-y-municipio
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig


class CataloniaCasesProcessor:
    """
    Converts Catalonia official COVID case data to xarray Dataset.
    
    This processor handles:
    - Loading cases from official Catalonia CSV
    - Preserving leading zeros in municipality codes
    - Converting DD/MM/YYYY dates to datetime
    - Aggregating across sex and test type
    - Optional test-type-specific processing (PCR vs TAR)
    
    Output format matches existing CasesProcessor for drop-in compatibility.
    """
    
    CASES_FILE = "Registre_de_casos_de_COVID-19_a_Catalunya_per_municipi_i_sexe.csv"
    
    # Column mapping from Catalan to English
    COLUMN_MAPPING = {
        "TipusCasData": "date",
        "MunicipiCodi": "municipality_code",
        "TipusCasDescripcio": "test_type",
        "NumCasos": "n_cases"
    }
    
    # Data types to preserve leading zeros
    DTYPES = {
        "TipusCasData": str,
        "MunicipiCodi": str,  # Critical: preserve leading zeros
        "ComarcaCodi": str,
        "SexeCodi": str,
        "NumCasos": int
    }
    
    # Test type categorization for inferred testing rate
    TEST_TYPE_CATEGORIES = {
        "Positiu TAR": "TAR",
        "Positiu per Test Ràpid": "TAR",
        "Positiu PCR": "PCR",
        "PCR probable": "PCR",
        "Positiu per ELISA": "ELISA",
        "Epidemiològic": "EPI"
    }
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize the Catalonia cases processor."""
        self.config = config
        self.validation_options = config.validation_options
        
    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw cases data from CSV."""
        cases_file = data_dir / self.CASES_FILE
        
        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        
        print(f"  Loading cases from {cases_file}")
        
        # Read with explicit dtypes to preserve leading zeros
        df = pd.read_csv(
            cases_file,
            dtype=self.DTYPES,
            usecols=list(self.DTYPES.keys()) + ["TipusCasDescripcio"]
        )
        
        # Rename columns to English
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Parse dates from DD/MM/YYYY format
        df["date"] = pd.to_datetime(
            df["date"],
            format="%d/%m/%Y",
            errors="coerce"
        ).dt.tz_localize(None).dt.floor("D")
        
        # Remove invalid rows
        df = df.dropna(subset=["date", "municipality_code", "n_cases"])
        
        # Filter to valid cases only (remove empty municipality codes)
        df = df[df["municipality_code"] != ""]
        
        # Remove negative cases if any
        df = df[df["n_cases"] >= 0]
        
        print(f"  Loaded {len(df):,} case records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique municipalities: {df['municipality_code'].nunique()}")
        
        return df
    
    def _categorize_test_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized test type category column."""
        df["test_category"] = df["test_type"].map(self.TEST_TYPE_CATEGORIES)
        # Keep original test type as is for unmapped categories
        df["test_category"] = df["test_category"].fillna(df["test_type"])
        return df
    
    def _aggregate_by_municipality(
        self,
        df: pd.DataFrame,
        by_test_type: bool = False
    ) -> pd.DataFrame:
        """Aggregate cases to municipality-date level."""
        group_cols = ["date", "municipality_code"]
        if by_test_type:
            group_cols.append("test_category")
        
        aggregated = (
            df.groupby(group_cols, dropna=False)["n_cases"]
            .sum()
            .reset_index()
        )
        
        return aggregated
    
    def process(
        self,
        data_dir: str | Path,
        by_test_type: bool = False
    ) -> xr.Dataset:
        """
        Process Catalonia cases data into canonical xarray Dataset.
        
        Args:
            data_dir: Directory containing cases CSV file
            by_test_type: If True, return separate variables for PCR/TAR cases
            
        Returns:
            xarray Dataset with structure:
            - cases: total cases (date, region_id)
            - cases_pcr: PCR cases (optional, if by_test_type=True)
            - cases_tar: TAR cases (optional, if by_test_type=True)
        """
        print("Processing Catalonia official case data")
        
        data_dir = Path(data_dir)
        
        # Load raw data
        cases_df = self._load_raw_data(data_dir)
        
        # Add test type categories
        cases_df = self._categorize_test_types(cases_df)
        
        # Filter to config date range
        cases_df = cases_df[
            (cases_df["date"] >= self.config.start_date) &
            (cases_df["date"] <= self.config.end_date)
        ]
        
        if cases_df.empty:
            raise ValueError("No case data in specified date range")
        
        # Aggregate to municipality level
        aggregated = self._aggregate_by_municipality(cases_df, by_test_type=by_test_type)
        
        # Create pivot table: date x municipality_code
        if by_test_type:
            # Create separate pivot for each test type
            pivot_pcr = aggregated[
                aggregated["test_category"] == "PCR"
            ].pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0
            )
            
            pivot_tar = aggregated[
                aggregated["test_category"] == "TAR"
            ].pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0
            )
            
            # Total cases pivot
            pivot_total = aggregated.pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0
            )
        else:
            pivot_total = aggregated.pivot_table(
                index="date",
                columns="municipality_code",
                values="n_cases",
                aggfunc="sum",
                fill_value=0
            )
        
        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq="D"
        )
        pivot_total = pivot_total.reindex(date_range, fill_value=0)
        
        # Rename municipality_code to region_id for compatibility
        pivot_total.columns.name = REGION_COORD
        pivot_total.index.name = TEMPORAL_COORD
        
        # Convert to xarray
        cases_da = xr.DataArray(
            pivot_total.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot_total.index,
                REGION_COORD: pivot_total.columns.astype(str)
            }
        )
        
        # Build result dataset
        result_ds = xr.Dataset({"cases": cases_da})
        
        if by_test_type:
            # Reindex and add test-type-specific variables
            pivot_pcr = pivot_pcr.reindex(date_range, fill_value=0)
            pivot_tar = pivot_tar.reindex(date_range, fill_value=0)
            
            pivot_pcr.columns.name = REGION_COORD
            pivot_tar.columns.name = REGION_COORD
            
            cases_pcr = xr.DataArray(
                pivot_pcr.values,
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={
                    TEMPORAL_COORD: pivot_pcr.index,
                    REGION_COORD: pivot_pcr.columns.astype(str)
                }
            )
            
            cases_tar = xr.DataArray(
                pivot_tar.values,
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={
                    TEMPORAL_COORD: pivot_tar.index,
                    REGION_COORD: pivot_tar.columns.astype(str)
                }
            )
            
            result_ds["cases_pcr"] = cases_pcr
            result_ds["cases_tar"] = cases_tar
        
        print(f"  Processed cases: {cases_da.sizes[TEMPORAL_COORD]} dates x {cases_da.sizes[REGION_COORD]} regions")
        print(f"  Total cases: {int(cases_da.sum())}")
        
        return result_ds
```

### 3.3 Deaths Processor (Comarca-level)

Create new file: `/Volumes/HUBSSD/code/EpiForecaster/data/preprocess/processors/deaths_processor.py`

```python
"""
Processor for Catalonia COVID-19 deaths data.

This module handles the conversion of deaths data from official Catalonia
open data sources. Deaths are reported at comarca level.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import REGION_COORD, TEMPORAL_COORD, PreprocessingConfig
from .municipality_mapping_processor import MunicipalityMappingProcessor


class DeathsProcessor:
    """
    Converts Catalonia deaths data to xarray Dataset.
    
    Deaths are reported at comarca level. This processor:
    - Loads deaths data
    - Optionally allocates deaths to municipalities using population weights
    - Returns xarray Dataset with deaths variable
    """
    
    DEATHS_FILE = "Registre_de_defuncions_per_COVID-19_a_Catalunya_per_comarca_i_sexe.csv"
    
    COLUMN_MAPPING = {
        "Data defunció": "date_of_death",
        "Codi Comarca": "comarca_code",
        "Nombre defuncions": "n_deaths"
    }
    
    DTYPES = {
        "Data defunció": str,  # DD/MM/YYYY format
        "Codi Comarca": str,   # Preserve leading zeros
        "Nombre defuncions": int
    }
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize the deaths processor."""
        self.config = config
        
    def _load_raw_data(self, data_dir: Path) -> pd.DataFrame:
        """Load raw deaths data from CSV."""
        deaths_file = data_dir / self.DEATHS_FILE
        
        if not deaths_file.exists():
            raise FileNotFoundError(f"Deaths file not found: {deaths_file}")
        
        print(f"  Loading deaths from {deaths_file}")
        
        df = pd.read_csv(
            deaths_file,
            dtype=self.DTYPES,
            usecols=list(self.COLUMN_MAPPING.keys())
        )
        
        # Rename columns
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Parse dates from DD/MM/YYYY format
        df["date_of_death"] = pd.to_datetime(
            df["date_of_death"],
            format="%d/%m/%Y",
            errors="coerce"
        ).dt.tz_localize(None).dt.floor("D")
        
        # Rename to standard date column
        df = df.rename(columns={"date_of_death": "date"})
        
        # Remove invalid rows
        df = df.dropna(subset=["date", "comarca_code", "n_deaths"])
        
        # Remove negative deaths if any
        df = df[df["n_deaths"] >= 0]
        
        print(f"  Loaded {len(df):,} death records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique comarcas: {df['comarca_code'].nunique()}")
        
        return df
    
    def _aggregate_to_comarca_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate deaths to comarca-date level (summing across sex)."""
        aggregated = (
            df.groupby(["date", "comarca_code"], dropna=False)["n_deaths"]
            .sum()
            .reset_index()
        )
        
        return aggregated
    
    def _allocate_to_municipalities(
        self,
        comarca_deaths: pd.DataFrame,
        population_df: pd.DataFrame,
        mapping_processor: MunicipalityMappingProcessor
    ) -> pd.DataFrame:
        """
        Allocate comarca-level deaths to municipalities using population weights.
        
        Args:
            comarca_deaths: DataFrame with deaths per comarca per day
            population_df: DataFrame with population per municipality
            mapping_processor: For municipality->comarca mapping
            
        Returns:
            DataFrame with deaths per municipality per day
        """
        # Load mapping
        mapping = mapping_processor.load_mapping()
        
        # Merge mapping with population
        merged = mapping.merge(
            population_df,
            left_on="municipality_code",
            right_on=REGION_COORD,
            how="left"
        )
        
        # Calculate comarca population for weighting
        merged["comarca_population"] = merged.groupby("comarca_code")["population"].transform("sum")
        merged["weight"] = merged["population"] / merged["comarca_population"]
        
        # Merge with deaths data
        deaths_with_weights = comarca_deaths.merge(
            merged[["comarca_code", "municipality_code", "weight"]],
            on="comarca_code",
            how="left"
        )
        
        # Allocate deaths
        deaths_with_weights["n_deaths_allocated"] = (
            deaths_with_weights["n_deaths"] * deaths_with_weights["weight"]
        )
        
        # Aggregate to municipality level
        municipal_deaths = (
            deaths_with_weights.groupby(["date", "municipality_code"])["n_deaths_allocated"]
            .sum()
            .reset_index()
            .rename(columns={"n_deaths_allocated": "n_deaths"})
        )
        
        return municipal_deaths
    
    def process(
        self,
        data_dir: str | Path,
        population_df: pd.DataFrame | None = None,
        allocate_to_municipalities: bool = False,
        mapping_processor: MunicipalityMappingProcessor | None = None
    ) -> xr.Dataset:
        """
        Process deaths data into xarray Dataset.
        
        Args:
            data_dir: Directory containing deaths CSV file
            population_df: Optional population data for municipality allocation
            allocate_to_municipalities: If True, allocate deaths to municipalities
            mapping_processor: Required if allocate_to_municipalities=True
            
        Returns:
            xarray Dataset with deaths variable
        """
        print("Processing Catalonia deaths data")
        
        if allocate_to_municipalities and mapping_processor is None:
            raise ValueError(
                "mapping_processor required when allocate_to_municipalities=True"
            )
        
        data_dir = Path(data_dir)
        
        # Load raw data
        deaths_df = self._load_raw_data(data_dir)
        
        # Filter to config date range
        deaths_df = deaths_df[
            (deaths_df["date"] >= self.config.start_date) &
            (deaths_df["date"] <= self.config.end_date)
        ]
        
        if deaths_df.empty:
            print("  Warning: No death data in specified date range")
            # Return empty dataset with proper structure
            date_range = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date,
                freq="D"
            )
            deaths_da = xr.DataArray(
                np.zeros((len(date_range), 0)),
                dims=[TEMPORAL_COORD, REGION_COORD],
                coords={
                    TEMPORAL_COORD: date_range,
                    REGION_COORD: []
                }
            )
            return xr.Dataset({"deaths": deaths_da})
        
        # Aggregate to comarca-day
        comarca_deaths = self._aggregate_to_comarca_day(deaths_df)
        
        # Optionally allocate to municipalities
        if allocate_to_municipalities and population_df is not None:
            processed_deaths = self._allocate_to_municipalities(
                comarca_deaths,
                population_df,
                mapping_processor  # type: ignore
            )
            region_col = "municipality_code"
        else:
            # Keep at comarca level
            processed_deaths = comarca_deaths.rename(
                columns={"comarca_code": "region_id"}
            )
            region_col = "region_id"
        
        # Create pivot table
        pivot = processed_deaths.pivot_table(
            index="date",
            columns=region_col,
            values="n_deaths",
            aggfunc="sum",
            fill_value=0
        )
        
        # Reindex to complete date range
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq="D"
        )
        pivot = pivot.reindex(date_range, fill_value=0)
        
        # Rename to standard coordinate names
        pivot.columns.name = REGION_COORD
        pivot.index.name = TEMPORAL_COORD
        
        # Convert to xarray
        deaths_da = xr.DataArray(
            pivot.values,
            dims=[TEMPORAL_COORD, REGION_COORD],
            coords={
                TEMPORAL_COORD: pivot.index,
                REGION_COORD: pivot.columns.astype(str)
            }
        )
        
        result_ds = xr.Dataset({"deaths": deaths_da})
        
        print(f"  Processed deaths: {deaths_da.sizes[TEMPORAL_COORD]} dates x {deaths_da.sizes[REGION_COORD]} regions")
        print(f"  Total deaths: {int(deaths_da.sum())}")
        
        return result_ds
```

---

## 4. Integration with Existing Pipeline

### 4.1 Modified Pipeline Main File

Update `/Volumes/HUBSSD/code/EpiForecaster/data/preprocess/pipeline.py`:

```python
# In OfflinePreprocessingPipeline.__init__:

# Add processor selection based on config
use_catalonia = getattr(config, 'use_catalonia_processor', False)

if use_catalonia:
    from .processors.catalonia_cases_processor import CataloniaCasesProcessor
    from .processors.deaths_processor import DeathsProcessor
    from .processors.municipality_mapping_processor import MunicipalityMappingProcessor
    
    self.processors = {
        "mobility": MobilityProcessor(self.config),
        "cases": CataloniaCasesProcessor(self.config),
        "deaths": DeathsProcessor(self.config),
        "mapping": MunicipalityMappingProcessor(self.config.data_dir),
        "alignment": AlignmentProcessor(self.config),
    }
else:
    # Use existing flowmaps-based processors
    self.processors = {
        "mobility": MobilityProcessor(self.config),
        "cases": CasesProcessor(self.config),
        "edar": EDARProcessor(self.config),
        "alignment": AlignmentProcessor(self.config),
    }
```

---

## 5. Data Type Specifications and Edge Cases

### 5.1 Leading Zero Preservation

**Problem**: Municipality and comarca codes have leading zeros (e.g., "08019" for Barcelona). Pandas will convert these to integers by default, losing the leading zero.

**Solution**: Always specify `dtype=str` for these columns when reading CSV files.

```python
# CORRECT:
df = pd.read_csv("file.csv", dtype={"MunicipiCodi": str})

# INCORRECT:
df = pd.read_csv("file.csv")  # MunicipiCodi becomes 8019 instead of "08019"
```

### 5.2 Date Format Conversion

Catalonia CSVs use DD/MM/YYYY format:

```python
# Parse dates correctly
df["date"] = pd.to_datetime(
    df["TipusCasData"],  # or "Data defunció" for deaths
    format="%d/%m/%Y",
    errors="coerce"
)
```

### 5.3 Missing Values in Spatial Data

The cases data includes rows with empty `MunicipiCodi` (aggregated entries):

```python
# Filter out these rows
df = df[df["municipality_code"] != ""]
df = df.dropna(subset=["municipality_code"])
```

---

## 6. Implementation Checklist

- [ ] Create `municipality_mapping_processor.py`
- [ ] Create `catalonia_cases_processor.py`
- [ ] Create `deaths_processor.py`
- [ ] Add unit tests in `tests/test_catalonia_processors.py`
- [ ] Update `pipeline.py` to support processor selection
- [ ] Create YAML config template `catalonia_official.yaml`
- [ ] Run tests on sample data
- [ ] Validate output against existing flowmaps-based datasets
- [ ] Update documentation (EPIFORECASTER.md)
- [ ] Add deprecation notice for old flowmaps-based cases processor

---

## 7. References

- [Datos.gob.es - COVID-19 Cases Registry](https://datos.gob.es/ca/catalogo/a09002970-registro-de-test-de-covid-19-realizados-en-catalunya-segregacion-por-sexo-y-municipio)
- [Datos.gob.es - COVID-19 Deaths Registry](https://datos.gob.es/ca/catalogo/a09002970-registro-de-defunciones-por-covid-19-en-catalunya-segregacion-por-sexo-y-comarca)
- [ICGC Administrative Boundaries](https://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Cartographic-geoinformation/Administrative-boundaries)
- [Transparencia Catalunya API](https://analisi.transparenciacatalunya.cat/)
