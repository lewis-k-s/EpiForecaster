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
- MunicipalityMappingProcessor: Municipality to comarca mapping table

These processors are used by the main preprocessing pipeline to convert
raw data into the canonical EpiBatch format.
"""

from .alignment_processor import AlignmentProcessor
from .cases_processor import CasesProcessor
from .catalonia_cases_processor import CataloniaCasesProcessor
from .edar_processor import EDARProcessor
from .deaths_processor import DeathsProcessor
from .mobility_processor import MobilityProcessor
from .municipality_mapping_processor import MunicipalityMappingProcessor

__all__ = [
    "MobilityProcessor",
    "CasesProcessor",
    "EDARProcessor",
    "AlignmentProcessor",
    "CataloniaCasesProcessor",
    "DeathsProcessor",
    "MunicipalityMappingProcessor",
]
