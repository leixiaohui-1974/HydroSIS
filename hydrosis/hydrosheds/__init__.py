"""HydroSHEDS backend integration for automated basin processing.

Provides lightweight client utilities to download/reference HydroSHEDS/HydroBASINS
datasets, processing helpers to standardize basin boundaries, and a simple
batch pipeline to generate outputs, metadata, and reports.
"""

from .client import HydroSHEDSClient
from .processor import HydroSHEDSProcessor
from .pipeline import HydroSHEDSPipeline

__all__ = [
    "HydroSHEDSClient",
    "HydroSHEDSProcessor",
    "HydroSHEDSPipeline",
]