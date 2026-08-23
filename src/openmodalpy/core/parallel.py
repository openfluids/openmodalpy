#!/usr/bin/env python3
"""
Simple Parallel Utilities for Modal Decomposition Analysis

This module provides optimized implementations using vectorized NumPy and
high-performance BLAS routines. OpenMP and Numba are no longer required; the
functions run on any standard Python installation.

Author: Modal Decomposition Team
"""

import logging

import numpy as np
from threadpoolctl import threadpool_info

from openmodalpy.core.welch import windowed_block_fft

logger = logging.getLogger(__name__)

# OpenMP support was removed. All routines rely on NumPy vectorization and the
# underlying BLAS implementation.
OPENMP_AVAILABLE = False
PARALLEL_AVAILABLE = True


def calculate_polar_weights_optimized(
    x: np.ndarray, y: np.ndarray, n_space: int | None = None
) -> np.ndarray:
    """
    Calculate integration weights for 2D cylindrical grid.

    This function uses a fully vectorized NumPy implementation that works on any
    platform without special dependencies.

    With ``n_space`` set and ``x``/``y`` both 1-D of length ``n_space``, the
    coordinates are scattered points and the weight per point is its radius,
    ``w_i = |y_i|`` (no cell measure), matching ``calculate_polar_weights``.

    Parameters:
    -----------
    x : np.ndarray
        Axial coordinates
    y : np.ndarray
        Radial coordinates
    n_space : int | None
        Number of snapshot columns; set to read ``x``/``y`` as scattered points.

    Returns:
    --------
    np.ndarray
        Integration weights, shape (Nx * Ny, 1)
    """
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.abs(y).reshape(int(n_space), 1)
    return _calculate_weights_numpy(x, y)


def _calculate_weights_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation of polar weights."""
    Nx, Ny = len(x), len(y)

    # Calculate y-direction (r-direction) integration weights (Wy) - vectorized
    Wy = np.zeros(Ny)

    if Ny > 1:
        # First point (centerline)
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2

        # Middle points - vectorized
        if Ny > 2:
            y_mid_left = (y[:-2] + y[1:-1]) / 2
            y_mid_right = (y[1:-1] + y[2:]) / 2
            Wy[1:-1] = np.pi * (y_mid_right**2 - y_mid_left**2)

        # Last point
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[-1] = np.pi * (y[-1] ** 2 - y_mid_left**2)
    else:
        Wy[0] = np.pi * y[0] ** 2

    # Calculate x-direction integration weights (Wx) - vectorized
    Wx = np.zeros(Nx)

    if Nx > 1:
        # First point
        Wx[0] = (x[1] - x[0]) / 2

        # Middle points - vectorized
        if Nx > 2:
            Wx[1:-1] = (x[2:] - x[:-2]) / 2

        # Last point
        Wx[-1] = (x[-1] - x[-2]) / 2
    else:
        Wx[0] = 1.0

    # Combine weights using outer product (much faster than loops)
    W = np.outer(Wx, Wy).flatten()

    return W.reshape(-1, 1)


# Placeholder function maintained for backward compatibility. It simply calls
# the NumPy implementation as OpenMP acceleration has been removed.
def _calculate_weights_openmp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _calculate_weights_numpy(x, y)


def blocksfft_optimized(
    q: np.ndarray,
    nfft: int,
    nblocks: int,
    novlap: int,
    blockwise_mean: bool = False,
    normvar: bool = False,
    window_norm: str = "power",
    window_type: str = "hamming",
) -> np.ndarray:
    """
    Optimized blocked FFT computation.

    This function uses the best available linear algebra backend (BLAS/LAPACK)
    and optimized memory access patterns for better performance.

    Parameters:
    -----------
    q : np.ndarray
        Input data [time, space]
    nfft : int
        Number of FFT points
    nblocks : int
        Number of blocks
    novlap : int
        Number of overlapping points between blocks
    blockwise_mean : bool
        Subtract blockwise mean if True
    normvar : bool
        If True, divide each block pointwise in space by its variance
        (unbiased, ``ddof=1``), matching ``spod_matlab`` (``opts.normvar``)
        and PySPOD (``normalize_data``). This does **not** produce unit
        variance and is therefore scale-dependent: scaling the input by
        ``c`` scales the normalized block by ``1/c``. Values below
        ``4*eps`` are clamped to 1. Implementation option, not a step in
        Towne, Schmidt & Colonius (2018). Defaults to False.
    window_norm : str
        Window normalization type ('amplitude' or 'power')
    window_type : str
        Window type. Use 'sine' for the custom sine window or any name
        recognized by ``scipy.signal.get_window`` (periodic / fftbins=True).

    Returns:
    --------
    np.ndarray
        FFT coefficients [freq, space, block]

    Notes
    -----
    Block starts are ``iblk * (nfft - novlap)`` with no end-of-record clamp.
    Callers must pass an ``nblocks`` that fits; oversize requests raise
    ``ValueError`` rather than re-using trailing samples.

    Implementation is the shared loop in :func:`openmodalpy.core.welch.windowed_block_fft`.
    """
    return windowed_block_fft(
        q,
        nfft,
        nblocks,
        novlap,
        blockwise_mean=blockwise_mean,
        normvar=normvar,
        window_norm=window_norm,
        window_type=window_type,
    )


def spod_single_frequency_optimized(
    qhat: np.ndarray,
    w: np.ndarray,
    nblocks: int,
    dst: float,
    num_modes: int | None = None,
    return_psi: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-frequency SPOD via the shared eigenproblem body.

    Thin wrapper around ``decomposition.spod_single_frequency``. Threading /
    BLAS setup for this path (if any) stays here; the algorithm does not.
    """
    # Late import avoids a parallel → decomposition → base cycle at module load.
    from openmodalpy.core.decomposition import spod_single_frequency

    return spod_single_frequency(
        qhat,
        nblocks,
        dst,
        w,
        num_modes=num_modes,
        return_psi=return_psi,
    )


def get_threadpool_summary() -> str:
    """Return a short description of active thread pools."""
    try:
        pools = threadpool_info()
        return (
            ", ".join(f"{p.get('prefix', '')}{p.get('internal_api')}={p.get('num_threads')}" for p in pools) or "none"
        )
    except Exception:
        return "unavailable"
