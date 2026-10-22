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

logger = logging.getLogger(__name__)

# OpenMP support was removed. All routines rely on NumPy vectorization and the
# underlying BLAS implementation.
OPENMP_AVAILABLE = False
PARALLEL_AVAILABLE = True


def calculate_polar_weights_optimized(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, n_space: int | None = None
) -> np.ndarray:
    """
    Calculate integration weights for 2D cylindrical grid.

    This function uses a fully vectorized NumPy implementation that works on any
    platform without special dependencies.

    With ``n_space`` set and ``x``/``y`` both 1-D of length ``n_space``, the
    coordinates are scattered points and the weight per point is its radius,
    ``w_i = |y_i|`` (no cell measure), matching ``calculate_polar_weights``.
    ``z`` is ignored in that branch.

    With ``z`` given (and not scattered), ``z`` is a 1-D azimuth axis theta in
    radians; see ``calculate_polar_weights`` for the sector-weight definition.
    Flattened in the order documented in ``calculate_cell_volume_weights``.

    Parameters:
    -----------
    x : np.ndarray
        Axial coordinates
    y : np.ndarray
        Radial coordinates
    z : np.ndarray | None
        Azimuth axis theta in radians, or None for the 2-D (x, r) grid.
    n_space : int | None
        Number of snapshot columns; set to read ``x``/``y`` as scattered points.

    Returns:
    --------
    np.ndarray
        Integration weights, shape (Nx * Ny, 1) or (Nx * Ny * Ntheta, 1)
    """
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.abs(y).reshape(int(n_space), 1)
    return _calculate_weights_numpy(x, y, z=z)


def _calculate_weights_numpy(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
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

    if z is None:
        # Combine weights: (Ny, Nx) outer product, flattened C-order.
        W = np.outer(Wy, Wx).flatten()
        return W.reshape(-1, 1)

    # 3-D polar: fold in the azimuth sector fraction and flatten (theta, r, x).
    # Imported lazily: base.py imports this module at load time, so importing
    # base at this module's top level would be circular.
    from openmodalpy.core.weights import _polar_theta_sector_fractions

    theta_fraction = _polar_theta_sector_fractions(np.asarray(z, dtype=np.float64))
    volumes_2d = np.outer(Wy, Wx)  # (Ny, Nx)
    volumes = theta_fraction[:, None, None] * volumes_2d[None, :, :]  # (Ntheta, Ny, Nx)
    return volumes.reshape(-1, 1)


def get_threadpool_summary() -> str:
    """Return a short description of active thread pools."""
    try:
        pools = threadpool_info()
        return (
            ", ".join(f"{p.get('prefix', '')}{p.get('internal_api')}={p.get('num_threads')}" for p in pools) or "none"
        )
    except Exception:
        return "unavailable"
