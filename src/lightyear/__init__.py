"""Lightyear: learn a spatial omics dataset once, then query it through an agent."""
__version__ = "0.5.0"
from .store import Store, load, build_store
from .ingest import ingest

__all__ = ["Store", "load", "build_store", "ingest"]
