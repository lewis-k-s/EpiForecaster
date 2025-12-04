"""
Individual data processors for the offline preprocessing pipeline.

This package contains processors for converting different data sources into
canonical tensor formats. Each processor handles a specific data type:

- MobilityProcessor: NetCDF mobility data → graph tensors
- CasesProcessor: COVID case CSV → aligned case tensors
- EDARProcessor: Wastewater biomarker data → normalized tensors
- AlignmentProcessor: Multi-dataset temporal and spatial alignment

These processors are used by the main preprocessing pipeline to convert
raw data into the canonical EpiBatch format.
"""

from .alignment_processor import AlignmentProcessor
from .cases_processor import CasesProcessor
from .edar_processor import EDARProcessor
from .mobility_processor import MobilityProcessor

__all__ = [
    "MobilityProcessor",
    "CasesProcessor",
    "EDARProcessor",
    "AlignmentProcessor",
]
