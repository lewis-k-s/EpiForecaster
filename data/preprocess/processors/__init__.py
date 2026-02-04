"""
Individual data processors for the offline preprocessing pipeline.

This package contains processors for converting different data sources into
canonical tensor formats. Each processor handles a specific data type:

- MobilityProcessor: NetCDF mobility data → graph tensors
- CasesProcessor: COVID case CSV → aligned case tensors
- EDARProcessor: Wastewater biomarker data → normalized tensors
- AlignmentProcessor: Multi-dataset temporal and spatial alignment
- CataloniaCasesProcessor: Catalonia official COVID case data → xarray tensors
- DeathsProcessor: Catalonia COVID deaths data → xarray tensors
- HospitalizationsProcessor: Catalonia COVID hospitalizations → daily municipality-level tensors
- MunicipalityMappingProcessor: Municipality to comarca mapping table with ABS support
- SyntheticProcessor: Bundled synthetic zarr data → extracted components

These processors are used by the main preprocessing pipeline to convert
raw data into the canonical EpiBatch format.
"""

from .alignment_processor import AlignmentProcessor
from .cases_processor import CasesProcessor
from .catalonia_cases_processor import CataloniaCasesProcessor
from .edar_processor import EDARProcessor
from .deaths_processor import DeathsProcessor
from .hospitalizations_processor import HospitalizationsProcessor
from .mobility_processor import MobilityProcessor
from .municipality_mapping_processor import MunicipalityMappingProcessor
from .synthetic_processor import SyntheticProcessor

__all__ = [
    "MobilityProcessor",
    "CasesProcessor",
    "EDARProcessor",
    "AlignmentProcessor",
    "CataloniaCasesProcessor",
    "DeathsProcessor",
    "HospitalizationsProcessor",
    "MunicipalityMappingProcessor",
    "SyntheticProcessor",
]
