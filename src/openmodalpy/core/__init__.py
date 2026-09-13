"""Core utilities for OpenModalPy."""

from openmodalpy.core.base import BaseAnalyzer
from openmodalpy.core.config import (
    FFT_BACKEND,
    FIG_DPI,
    FIGURES_DIR,
    RESULTS_DIR,
)
from openmodalpy.core.provenance import collect_provenance
from openmodalpy.core.results import AnalysisResults, read_results, write_results

# decomposition is imported as a submodule (openmodalpy.core.decomposition)
# so analyzers can `from openmodalpy.core import decomposition`.

__all__ = [
    "BaseAnalyzer",
    "AnalysisResults",
    "read_results",
    "write_results",
    "collect_provenance",
    "FFT_BACKEND",
    "FIG_DPI",
    "RESULTS_DIR",
    "FIGURES_DIR",
]
