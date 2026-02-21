"""
PeakIntegrate.src — Public API.

Example::

    from PeakIntegrate.src import Experiment, load_experiment
    exp = load_experiment("path/to/tables", "path/to/chrom_data.h5")
"""

from PeakIntegrate.src.models import (
    PickedPeak,
    EIC,
    Chromatogram,
    Experiment,
)
from PeakIntegrate.src.loader import load_experiment
from PeakIntegrate.src.integration import integrate_experiment
from PeakIntegrate.src.config import (
    CompoundDef,
    load_compounds,
    resolve_target_compounds,
    get_base_compounds,
)

__all__ = [
    "PickedPeak",
    "EIC",
    "Chromatogram",
    "Experiment",
    "load_experiment",
    "integrate_experiment",
    "CompoundDef",
    "load_compounds",
    "resolve_target_compounds",
    "get_base_compounds",
]
