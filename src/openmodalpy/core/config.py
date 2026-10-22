"""
Central configuration for modal decomposition analysis.

NOTE: ALL imports are available here and this is imported in utils.py
so we only need to import utils in other files.
"""

import os

from fftkit import DEFAULT_BACKEND as _FFTKIT_BACKEND

os.environ["OS_ACTIVITY_MODE"] = "disable"  # suppress macOS IMKClient logs
"""
Configuration and shared imports for modal decomposition tools.
"""

# Default directories - organized by case, then method
# Structure: results/{case}/{method}/, figures/{case}/{method}/
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

# Figure saving options
FIG_DPI = 500
FIG_FORMAT = "png"  # or "pdf"

# FFT backend selection is owned by fftkit, which probes every backend at import
# time and degrades to an available one rather than failing. Re-exported here so
# the name openmodalpy reports is always the one fftkit actually dispatches to.
# Override with the FFTKIT_BACKEND env var.
FFT_BACKEND = _FFTKIT_BACKEND

CMAP_SEQ = "viridis"  # Sequential colormap for general use
CMAP_DIV = "RdBu_r"  # Diverging colormap for signed data

# Default window type for FFT
WINDOW_TYPE = "hamming"
WINDOW_NORM = "power"

# Other global options can be added here as needed
