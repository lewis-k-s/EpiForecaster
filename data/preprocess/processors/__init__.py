"""
Individual data processors for the offline preprocessing pipeline.

This package contains processors for converting different data sources into
canonical tensor formats. Each processor handles a specific data type:

- MobilityProcessor: NetCDF mobility data → graph tensors
- CataloniaCasesProcessor: COVID case CSV → xarray tensors with mask/age channels
- EDARProcessor: Wastewater biomarker data → normalized tensors
- AlignmentProcessor: Multi-dataset temporal and spatial alignment
- DeathsProcessor: Catalonia COVID deaths data → xarray tensors
- HospitalizationsProcessor: Catalonia COVID hospitalizations → daily municipality-level tensors
- MunicipalityMappingProcessor: Municipality to comarca mapping table with ABS support
- SyntheticProcessor: Bundled synthetic zarr data → extracted components

These processors are used by the main preprocessing pipeline to convert
raw data into the canonical EpiBatch format.
"""

from .alignment_processor import AlignmentProcessor
from .catalonia_cases_processor import CataloniaCasesProcessor
from .edar_processor import EDARProcessor
from .deaths_processor import DeathsProcessor
from .hospitalizations_processor import HospitalizationsProcessor
from .mobility_processor import MobilityProcessor
from .municipality_mapping_processor import MunicipalityMappingProcessor
from .synthetic_processor import SyntheticProcessor

__all__ = [
    "MobilityProcessor",
    "CataloniaCasesProcessor",
    "EDARProcessor",
    "AlignmentProcessor",
    "DeathsProcessor",
    "HospitalizationsProcessor",
    "MunicipalityMappingProcessor",
    "SyntheticProcessor",
]
