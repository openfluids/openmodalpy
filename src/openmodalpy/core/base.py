#!/usr/bin/env python3
"""
Common utilities for modal decomposition methods.

All imports are centralized here to keep the code clean and consistent.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast, overload

import h5py
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.linalg import svds

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

from openmodalpy.core.config import (
    CMAP_DIV,
    FFT_BACKEND,
)
from openmodalpy.core.io import load_data as di_load_data
from openmodalpy.core.io import load_jetles_data as di_load_jetles_data
from openmodalpy.core.io import load_mat_data as di_load_mat_data
from openmodalpy.core.threads import apply_blas_limit
from openmodalpy.core.welch import sine_window as sine_window  # re-export; body in welch.py
from openmodalpy.core.welch import welch_nblocks, windowed_block_fft

# Welch block FFT and helpers live in welch.py (single implementation) so base
# remains importable when the parallel stack fails to load.
from openmodalpy.specs import display_name_for

try:
    from openmodalpy.core.parallel import (
        PARALLEL_AVAILABLE,
        calculate_polar_weights_optimized,
        get_threadpool_summary,
        spod_single_frequency_optimized,
    )
except ImportError:
    PARALLEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Route iterative SVD (ARPACK via ``svds``) only when rank is a small fraction
# of the smaller dimension *and* that dimension is large enough for it to pay.
# Values from machine measurements (see ``use_iterative_svd`` docstring).
ARPACK_MAX_RANK_FRACTION = 0.05
ARPACK_MIN_DIM = 256


def use_iterative_svd(min_dim: int, rank: int) -> bool:
    """Whether to use ARPACK (``svds``) rather than dense SVD for a reduced SVD.

    Returns True only when both hold:

    - ``rank < ARPACK_MAX_RANK_FRACTION * min_dim``
    - ``min_dim >= ARPACK_MIN_DIM``

    Measured on this machine, dense vs ARPACK, decaying spectrum:

    - ``X(20000, 2000)``: k=10 ARPACK 3.7× faster; k=100 break-even; k=500
      dense 13× faster.
    - ``X(20000, 5000)``: k=10 ARPACK 15.8× faster and ~5 MB peak vs ~1000 MB;
      k=100 ARPACK 3.2× faster.

    Memory is the real argument at large ``min_dim``: dense peaks at
    ``O(m * min_dim)``. The fraction 0.05 sits at the break-even neighbourhood
    for the first case; a looser 1/4 threshold was measured about five times
    too permissive. Near-full-rank requests (e.g. POD/ST-POD with
    ``k = n_min - 1``) therefore stay on dense SVD — the regime where iterative
    solvers are worst.
    """
    return rank < ARPACK_MAX_RANK_FRACTION * min_dim and min_dim >= ARPACK_MIN_DIM


def randomized_svd(
    X: np.ndarray,
    rank: int,
    *,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leading *rank* singular triplets via a randomized (Halko) SVD.

    Standard Halko range finder with oversampling and re-orthonormalised power
    iterations (Halko, Martinsson & Tropp 2011, Algorithms 4.4 and 5.1): draw a
    Gaussian test matrix with ``rank + n_oversamples`` columns, QR the sketch,
    run ``n_power_iterations`` re-orthonormalised power iterations, project,
    dense-SVD the small projected matrix, and lift back. Returns ``(u, s, vh)``
    with descending ``s``, same convention as :func:`compute_reduced_svd`.

    Accuracy depends entirely on how fast the singular spectrum decays. This is
    an opt-in route — never selected by ``method="auto"`` — because a slowly
    decaying spectrum is NOT suitable: the leading singular values can be
    materially wrong, and for POD that means the energy percentages would be
    visibly wrong.

    Measured max relative error on the leading singular values
    (``m=4000``, ``n=800``, ``k=20``, planted spectra, this machine):

    - fast ``0.70^j`` — ``1.3e-15`` at 2 power iterations
    - medium ``0.95^j`` — ``5.9e-03`` at 2, ``2.3e-08`` at 8
    - algebraic ``1/j`` — ``8.9e-03`` at 2, ``2.4e-06`` at 8
    - slow ``0.999^j`` — ``8.7e-02`` at 2 and still ``2.4e-02`` at 8

    Runs under the process-wide BLAS thread policy (see ``core.threads``).
    """
    X = np.asarray(X)
    m, n = X.shape
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > min(m, n):
        raise ValueError(f"rank={rank} exceeds min(X.shape)={min(m, n)}")
    n_random = min(rank + n_oversamples, min(m, n))

    with apply_blas_limit():
        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((n, n_random))
        # Y = X @ Omega; Q spans an approximate range of X.
        q, _ = np.linalg.qr(X @ omega, mode="reduced")
        # Re-orthonormalised power iterations (Alg. 4.4): alternate X.T and X
        # with a QR at each half-step so the basis does not lose rank to
        # floating-point growth of the dominant directions.
        for _ in range(n_power_iterations):
            q, _ = np.linalg.qr(X.T @ q, mode="reduced")
            q, _ = np.linalg.qr(X @ q, mode="reduced")
        # Project, dense-SVD the small matrix, lift left vectors.
        b = q.T @ X
        ub, s, vh = np.linalg.svd(b, full_matrices=False)
        u = q @ ub
        return u[:, :rank], s[:rank], vh[:rank, :]


def compute_reduced_svd(
    X: np.ndarray,
    rank: int,
    v0_seed: int = 0,
    *,
    method: Literal["auto", "dense", "iterative", "randomized"] = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return leading *rank* singular triplets, using truncated SVD when it pays.

    Parameters
    ----------
    method : {"auto", "dense", "iterative", "randomized"}, default "auto"
        Route selection. ``"auto"`` consults :func:`use_iterative_svd` and
        chooses dense or iterative only — it never selects randomized. The
        other three force their route. Randomized is opt-in because its
        accuracy depends on spectral decay (see :func:`randomized_svd`).

    When ``method="auto"``, routing is decided solely by
    :func:`use_iterative_svd` (rank/min_dim ratio and a minimum dimension).
    Dense SVD is used otherwise — including for near-full-rank requests on
    large matrices, where ARPACK is slower.

    Results can differ across the routing threshold by design (different
    algorithms). Do not retune the constants without re-checking the
    measurements recorded on :func:`use_iterative_svd`.

    Runs under the process-wide BLAS thread policy (see ``core.threads``).
    """
    if method not in ("auto", "dense", "iterative", "randomized"):
        raise ValueError(f"Unknown method {method!r}. Accepted: 'auto', 'dense', 'iterative', 'randomized'.")

    if method == "randomized":
        return randomized_svd(X, rank, seed=v0_seed)

    with apply_blas_limit():
        min_dim = min(X.shape)
        use_iter = use_iterative_svd(min_dim, rank) if method == "auto" else method == "iterative"
        if use_iter:
            # Local deterministic start vector — never reseed the caller's global RNG.
            v0 = np.random.default_rng(v0_seed).standard_normal(min_dim)
            u, s, vh = svds(X, k=rank, v0=v0)
            order = np.argsort(s)[::-1]
            return u[:, order], s[order], vh[order, :]
        return np.linalg.svd(X, full_matrices=False)


# Relative band used when (1) choosing the pivot index for mode sign/phase and
# (2) grouping |λ| ties for canonical DMD spectrum order. Sits above typical
# cross-build eigenvector noise (~1e-14 to 1e-15 relative) and far below any
# physically meaningful difference between two peaks. Moves the sign
# discontinuity out of the last-bit regime; perfect uniqueness for exactly
# degenerate peaks is impossible (phi and -phi are both valid), so the band
# relocates the ambiguity rather than removing it.
CANONICAL_TIE_RTOL = 1e-12


def canonical_eigenvalue_order(eigvals: ArrayLike) -> np.ndarray:
    """Permutation that puts ``eigvals`` in canonical DMD spectrum order.

    Primary key: ``|λ|`` descending. A group is every run of that order whose
    magnitude agrees with the group's first (largest) member within
    ``CANONICAL_TIE_RTOL``, never merely with its neighbour. Within a group
    the order is lexicographic ``(Re, Im)`` ascending, which is continuous
    across the negative real axis where ``np.angle`` jumps at ``±π``.

    The returned index array ``idx`` satisfies ``eigvals[idx]`` is the
    canonical order. Empty input yields an empty integer array. DMD applies
    this permutation to the full spectrum and only then truncates.
    """
    eigvals = np.asarray(eigvals).reshape(-1)
    n = int(eigvals.size)
    if n == 0:
        return np.array([], dtype=int)

    mag = np.abs(eigvals).astype(float, copy=False)
    # Stable so exact-tie group membership does not depend on the platform sort.
    order = np.argsort(-mag, kind="stable")
    out = np.empty(n, dtype=int)
    i = 0
    while i < n:
        j = i + 1
        peak = float(mag[int(order[i])])
        while j < n:
            other = float(mag[int(order[j])])
            scale = max(abs(peak), abs(other))
            if scale == 0.0 or abs(peak - other) <= CANONICAL_TIE_RTOL * scale:
                j += 1
            else:
                break
        group = order[i:j]
        group_eigs = eigvals[group]
        tie_key = np.lexsort((np.imag(group_eigs), np.real(group_eigs)))
        out[i:j] = group[tie_key]
        i = j
    return out


def canonical_pivot_index(col: ArrayLike) -> int:
    """Lowest index whose magnitude is within ``CANONICAL_TIE_RTOL`` of max |col|.

    Empty or all-zero columns return 0. Shared by ``canonicalize_modes`` and the
    tests/probes that check the same invariant.
    """
    mag = np.abs(np.asarray(col))
    if mag.size == 0:
        return 0
    m = float(mag.max())
    if m == 0.0:
        return 0
    return int(np.argmax(mag >= (1.0 - CANONICAL_TIE_RTOL) * m))


@overload
def canonicalize_modes(modes: ArrayLike, coeffs: None = None) -> tuple[np.ndarray, None]: ...
@overload
def canonicalize_modes(modes: ArrayLike, coeffs: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...
def canonicalize_modes(modes: ArrayLike, coeffs: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Scale each mode so its band-pivot entry is real and positive.

    LAPACK leaves eigenvector sign (real) and phase (complex) free. For each
    mode column *k*, take the lowest index whose magnitude sits inside a relative
    band of the column maximum
    ``i = argmax(|col| >= (1 - CANONICAL_TIE_RTOL) * max|col|)`` and the entry
    ``v = modes[i, k]``. If the column is all zeros it is left alone; otherwise
    scale by ``s = conj(v) / |v|`` so that entry becomes ``|v|``. Coefficients
    receive the same factor ``s`` so they remain the projection of the data onto
    the modes. Real inputs give ``s = ±1`` and the same rule covers both cases.

    The band moves the sign-flip discontinuity away from single-ulp noise; it
    does not remove the ambiguity. For an exactly antisymmetric mode, ``phi``
    and ``-phi`` are both valid, so any rule must break the tie by some
    comparison, and every comparison has a discontinuity somewhere.

    Integer-dtype inputs are promoted to float64 before scaling, since the
    scale factor is not an integer. Float and complex inputs keep their own
    dtype. Either way the returned arrays are new copies, so the caller's
    array is never touched.
    """
    modes = np.asarray(modes)
    if not np.issubdtype(modes.dtype, np.inexact):
        modes = modes.astype(np.float64)
    if coeffs is not None:
        coeffs = np.asarray(coeffs)
        if not np.issubdtype(coeffs.dtype, np.inexact):
            coeffs = coeffs.astype(np.float64)
        n_cols = modes.shape[1] if modes.ndim >= 2 else 0
        if coeffs.ndim < 2 or coeffs.shape[1] != n_cols:
            raise ValueError(
                f"coeffs shape {coeffs.shape} does not match modes shape "
                f"{modes.shape}: coeffs.shape[1] must equal modes.shape[1]."
            )

    if modes.size == 0 or modes.ndim < 2 or modes.shape[1] == 0:
        return modes, coeffs

    modes = modes.copy()
    if coeffs is not None:
        coeffs = coeffs.copy()

    for k in range(modes.shape[1]):
        col = modes[:, k]
        if not np.all(np.isfinite(col)):
            raise ValueError(
                f"Mode column {k} contains non-finite entries (NaN or inf); "
                "refusing to choose a pivot that would poison the column."
            )
        if float(np.abs(col).max()) == 0:
            continue
        i = canonical_pivot_index(col)
        v = col[i]
        s = np.conj(v) / np.abs(v)
        modes[:, k] *= s
        if coeffs is not None:
            coeffs[:, k] *= s

    return modes, coeffs


def get_num_threads() -> int:
    """Return thread count from ``OMP_NUM_THREADS`` or ``os.cpu_count()``."""
    env = os.environ.get("OMP_NUM_THREADS")
    try:
        val = int(env) if env is not None else None
    except (TypeError, ValueError):
        val = None
    if val is not None and val > 0:
        return val
    cpu = os.cpu_count() or 1
    return cpu


T = TypeVar("T")
R = TypeVar("R")


def parallel_map(func: Callable[[T], R], iterable: Iterable[T], threads: int | None = None) -> list[R]:
    """Map function over iterable using threads."""
    threads = threads or get_num_threads()
    if threads <= 1:
        return [func(x) for x in iterable]
    results = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(func, x) for x in iterable]
        for f in futures:
            results.append(f.result())
    return results


def make_result_filename(root: str, nfft: int, overlap: float, Ns: int, analysis: str) -> str:
    """
    Generate a harmonized result filename for analysis outputs.
    Args:
        root (str): Base name of the dataset (no extension)
        nfft (int): FFT block size
        overlap (float): Overlap fraction (0-1)
        Ns (int): Number of snapshots
        analysis (str): Analysis type (e.g., 'spod', 'bsmd')
    Returns:
        str: Result filename (always .hdf5)
    """
    return f"{root}_Nfft{nfft}_ovlap{overlap}_{Ns}snapshots_{analysis}.hdf5"


_QHAT_STAMP_ATTR_PREFIX = "_fftcache_"


def _qhat_content_digest(q: np.ndarray) -> str:
    """Return a blake2b digest of ``q``'s raw bytes (plus shape/dtype).

    A full hash is O(n), i.e. cheaper than the O(n log n) FFT it lets us
    avoid recomputing, so hashing the exact content is affordable here and a
    sampled/strided checksum is not worth the false-negative risk of two
    different arrays colliding.
    """
    arr = np.ascontiguousarray(q)
    # ``arr.data`` is a memoryview onto the array's own buffer, so hashing it copies
    # nothing. ``tobytes()`` would duplicate the whole snapshot matrix just to hash it.
    h = hashlib.blake2b(arr.data.cast("B"), digest_size=16)
    h.update(str(arr.shape).encode())
    h.update(arr.dtype.str.encode())
    return h.hexdigest()


def _qhat_cache_stamp(analyzer: BaseAnalyzer, q: np.ndarray) -> dict[str, str | float | int | bool]:
    """Return the parameters that determine the FFT blocks produced for ``q``.

    Note: ``spatial_weight_type`` is deliberately excluded. ``blocksfft`` (see
    below) only ever receives ``q``, ``nfft``, ``nblocks``, ``novlap``,
    ``blockwise_mean``, ``normvar``, ``window_norm`` and ``window_type`` — the
    spatial weights are applied later, in the SPOD/BSMD eigenproblem, never in
    the FFT block computation. So it cannot affect ``qhat`` and does not need
    to be stamped.
    """
    return {
        "window_type": str(getattr(analyzer, "window_type", "hamming")),
        "window_norm": str(getattr(analyzer, "window_norm", "power")),
        "overlap": float(analyzer.overlap),
        "nfft": int(analyzer.nfft),
        "blockwise_mean": bool(getattr(analyzer, "blockwise_mean", False)),
        "normvar": bool(getattr(analyzer, "normvar", False)),
        "q_digest": _qhat_content_digest(q),
    }


def _write_qhat_stamp(h5file: h5py.File, analyzer: BaseAnalyzer, q: np.ndarray) -> None:
    """Stamp the parameters that produced ``qhat`` into ``h5file``'s attrs."""
    for key, value in _qhat_cache_stamp(analyzer, q).items():
        h5file.attrs[f"{_QHAT_STAMP_ATTR_PREFIX}{key}"] = value


def _verify_qhat_stamp(h5file: h5py.File, analyzer: BaseAnalyzer, q: np.ndarray) -> bool:
    """Return whether ``h5file``'s stamped FFT parameters match ``analyzer``/``q``.

    On any mismatch — or an absent stamp (e.g. a cache file from an older
    build) — this returns False rather than raising. That is the opposite
    policy from result files (which must raise on staleness): FFT blocks are
    cheaply re-derivable from the raw data, so silently recomputing them is
    correct and non-destructive. Do not "harmonise" this with the stricter
    policy used for saved results/modes.
    """
    expected = _qhat_cache_stamp(analyzer, q)
    for key, exp_value in expected.items():
        attr_name = f"{_QHAT_STAMP_ATTR_PREFIX}{key}"
        if attr_name not in h5file.attrs:
            logger.warning("FFT cache stamp missing '%s' (older cache file) — recomputing FFT blocks.", key)
            return False
        actual = h5file.attrs[attr_name]
        if isinstance(exp_value, bool):
            actual = bool(actual)
        elif isinstance(exp_value, int):
            actual = int(actual)
        elif isinstance(exp_value, float):
            actual = float(actual)
        else:
            actual = str(actual)
        if actual != exp_value:
            logger.warning(
                "FFT cache stamp mismatch on '%s': cached=%r != current=%r — recomputing FFT blocks.",
                key,
                actual,
                exp_value,
            )
            return False
    return True


def _hdf5_write_mode(path: str) -> str:
    """Return ``"a"`` if ``path`` is a readable HDF5 file, else ``"w"``.

    File existence is the wrong predicate: a truncated or otherwise corrupt
    cache still exists on disk, so ``os.path.exists`` would open it in append
    mode and die with an uncaught ``OSError``.

    ``h5py.is_hdf5`` is the first filter (False for a missing path → ``"w"``).
    It is not sufficient alone: a truncated file often still carries a valid
    HDF5 signature at offset 0, so ``is_hdf5`` returns True even though any
    open in ``"a"``/``"r"`` raises. Probe a read-only open and only then
    return ``"a"``; on ``OSError`` return ``"w"`` so the caller overwrites.
    """
    if not h5py.is_hdf5(path):
        return "w"
    try:
        with h5py.File(path, "r"):
            pass
    except OSError:
        return "w"
    return "a"


def print_summary(analysis: str, results_dir: str, figures_dir: str) -> None:
    """Log a short summary of where results and figures were saved."""
    logger.info("%s analysis finished", analysis)
    logger.info("Results: %s", results_dir)
    logger.info("Figures: %s", figures_dir)


def compute_aspect_ratio(x_coords: np.ndarray, y_coords: np.ndarray) -> float | Literal["auto"]:
    """Return ``dy/dx`` if coordinates are 1D vectors, else ``'auto'``."""
    if hasattr(x_coords, "ndim") and hasattr(y_coords, "ndim"):
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            dx = float(x_coords.max() - x_coords.min())
            dy = float(y_coords.max() - y_coords.min())
            if dx > 0 and dy > 0:
                return dy / dx
    return "auto"


def get_aspect_ratio(data: dict) -> Union[float, str]:
    """Return aspect ratio for ``data`` using available coordinates."""
    x = data.get("x", [])
    y = data.get("y", [])
    return compute_aspect_ratio(x, y)


def get_robust_clim(data: np.ndarray, method: str = "percentile", sigma: float = 2.5) -> tuple:
    """Compute robust colormap limits that reduce the effect of outliers.

    Parameters
    ----------
    data : ndarray
        Data array (can contain NaNs which will be ignored)
    method : str
        'percentile' : Use 2nd and 98th percentiles
        'sigma' : Use median ± sigma * MAD (median absolute deviation)
        'minmax' : Use global min/max (no robustness)
    sigma : float
        Number of standard deviations for 'sigma' method

    Returns
    -------
    vmin, vmax : float
        Colormap limits
    """
    arr = np.asarray(data)
    if arr.size == 0:
        return -1.0, 1.0
    # Fancy-indexing with isfinite always copies, even when every value is finite.
    # Mode volumes are finite; keep that path allocation-light. Still drop NaN and
    # +/-Inf when present (np.nanpercentile keeps Inf, so it is not a substitute).
    if np.isfinite(arr).all():
        data_clean = arr
    else:
        flat = arr.ravel()
        data_clean = flat[np.isfinite(flat)]
        if data_clean.size == 0:
            return -1.0, 1.0

    if method == "percentile":
        vmin, vmax = np.percentile(data_clean, [2, 98])
    elif method == "sigma":
        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        # MAD to std: std ≈ 1.4826 * MAD
        std_estimate = 1.4826 * mad
        vmin = median - sigma * std_estimate
        vmax = median + sigma * std_estimate
    else:  # minmax
        vmin, vmax = data_clean.min(), data_clean.max()

    # Ensure symmetric for diverging colormaps
    abs_max = max(abs(vmin), abs(vmax))
    if not np.isfinite(abs_max) or abs_max == 0.0:
        return -1.0, 1.0
    return -abs_max, abs_max


def get_fig_aspect_ratio(data: dict, clamp_low: float = 0.3, clamp_high: float = 5.0) -> float:
    """Return physical domain aspect ratio (dx/dy) with reasonable clamping for figure sizing.

    Computes aspect from physical extent if coordinates available, otherwise from Nx/Ny.
    Clamps to [0.3, 5.0] to avoid extremely distorted figures while preserving physical proportions.
    """
    x_coords = data.get("x")
    y_coords = data.get("y")

    # Try to compute from physical extent first
    if x_coords is not None and y_coords is not None:
        try:
            x_arr = np.asarray(x_coords)
            y_arr = np.asarray(y_coords)
            dx = float(x_arr.max() - x_arr.min())
            dy = float(y_arr.max() - y_arr.min())
            if dx > 0 and dy > 0:
                aspect = dx / dy
                return max(clamp_low, min(aspect, clamp_high))
        except (ValueError, TypeError):
            pass

    # Fall back to grid point ratio
    nx = int(data.get("Nx", 1))
    ny = int(data.get("Ny", 1))
    if ny <= 0:
        aspect = 1.0
    else:
        aspect = nx / ny
    return max(clamp_low, min(aspect, clamp_high))


def get_plot_style(data: dict, section: str = "spatial") -> dict[str, Any]:
    """Return plot-style overrides stored in data metadata."""
    metadata = data.get("metadata", {})
    plot_style = metadata.get("plot_style", {})
    if not isinstance(plot_style, dict):
        return {}
    section_style = plot_style.get(section)
    if isinstance(section_style, dict):
        return section_style
    return plot_style


def format_mode_title(data: dict, mode_index: int, default: str) -> str:
    """Format a mode title using optional metadata-driven templates."""
    style = get_plot_style(data)
    template = style.get("title_template")
    if not template:
        return default
    mode_number = mode_index + 1
    return template.format(mode=mode_number, m=mode_number)


def style_spatial_axes(
    ax: Axes,
    data: dict,
    *,
    x_coords: ArrayLike | None = None,
    y_coords: ArrayLike | None = None,
    equal_default: bool = True,
) -> None:
    """Apply metadata-driven styling to a 2D spatial axis."""
    style = get_plot_style(data)
    figure_facecolor = style.get("figure_facecolor")
    axes_facecolor = style.get("axes_facecolor", style.get("facecolor"))
    if figure_facecolor:
        ax.figure.patch.set_facecolor(figure_facecolor)
    if axes_facecolor:
        ax.set_facecolor(axes_facecolor)

    axis_labels = style.get("axis_labels", {})
    ax.set_xlabel(axis_labels.get("x", r"$x/D$"))
    ax.set_ylabel(axis_labels.get("y", r"$y/D$"))

    aspect = style.get("aspect")
    if aspect == "equal":
        ax.set_aspect("equal", "box")
    elif aspect == "auto":
        ax.set_aspect("auto")
    elif aspect is not None:
        ax.set_aspect(aspect)
    elif equal_default:
        ax.set_aspect("equal", "box")
    else:
        ax.set_aspect("auto")

    if x_coords is not None:
        x_arr = np.asarray(x_coords)
        x_limits = style.get("xlim", [float(np.min(x_arr)), float(np.max(x_arr))])
        ax.set_xlim(*x_limits)
    elif "xlim" in style:
        ax.set_xlim(*style["xlim"])

    if y_coords is not None:
        y_arr = np.asarray(y_coords)
        y_limits = style.get("ylim", [float(np.min(y_arr)), float(np.max(y_arr))])
        ax.set_ylim(*y_limits)
    elif "ylim" in style:
        ax.set_ylim(*style["ylim"])

    grid_style = style.get("grid", {})
    if isinstance(grid_style, dict):
        grid_enabled = grid_style.get("enabled", True)
        grid_kwargs = {
            "linestyle": grid_style.get("linestyle", "--"),
            "alpha": grid_style.get("alpha", 0.3),
            "color": grid_style.get("color"),
        }
    else:
        grid_enabled = bool(grid_style) if style.get("grid") is not None else True
        grid_kwargs = {"linestyle": "--", "alpha": 0.3, "color": None}
    if grid_enabled:
        if grid_kwargs["color"] is None:
            del grid_kwargs["color"]
        ax.grid(True, **grid_kwargs)
    else:
        ax.grid(False)


def add_inset_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
    data: dict,
    *,
    ticks: Sequence[float] | None = None,
    ticklabels: Sequence[str] | None = None,
    fmt: str = "%.2f",
) -> Colorbar | None:
    """Add a compact, metadata-driven inset colorbar to an axis."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    style = get_plot_style(data)
    cbar_style = style.get("colorbar", {})
    if cbar_style.get("enabled", True) is False:
        return None

    location = cbar_style.get("location", "top_inset")
    orientation = cbar_style.get("orientation", "horizontal")
    if location == "top_inset":
        cax = inset_axes(
            ax,
            width=cbar_style.get("width", "24%"),
            height=cbar_style.get("height", "6%"),
            loc=cbar_style.get("loc", "upper right"),
            borderpad=cbar_style.get("borderpad", 2.0),
        )
        cb = fig.colorbar(mappable, cax=cax, orientation=orientation, format=fmt)
    else:
        cb = fig.colorbar(mappable, ax=ax, shrink=cbar_style.get("shrink", 0.8), format=fmt)
        cax = cb.ax

    cb.ax.tick_params(
        labelsize=cbar_style.get("tick_fontsize", 8),
        pad=cbar_style.get("tick_pad", 1),
        colors=cbar_style.get("tick_color", "black"),
    )
    if orientation == "horizontal":
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cax.patch.set_facecolor(cbar_style.get("facecolor", "white"))
    cax.patch.set_alpha(cbar_style.get("alpha", 0.95))
    return cb


def subset_volume_focus_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply optional metadata-driven volume cropping to a 3D scalar field."""
    values = np.asarray(field_3d)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D field, got shape {values.shape}.")

    x_arr = np.asarray(x_coords)
    y_arr = np.asarray(y_coords)
    z_arr = np.asarray(z_coords)
    nx, ny, nz = values.shape
    if x_arr.shape[0] != nx or y_arr.shape[0] != ny or z_arr.shape[0] != nz:
        raise ValueError(
            f"Coordinate lengths {(x_arr.shape[0], y_arr.shape[0], z_arr.shape[0])} do not match field shape {values.shape}."
        )

    style = get_plot_style(data, section="volume")

    def _axis_subset(arr: np.ndarray, limits: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            mask = np.ones(arr.shape[0], dtype=bool)
            return arr, mask
        lo = max(float(np.min(arr)), float(limits[0]))
        hi = min(float(np.max(arr)), float(limits[1]))
        mask = (arr >= lo) & (arr <= hi)
        if not np.any(mask):
            raise ValueError(f"Requested volume limits {limits} do not overlap the available coordinate range.")
        return arr[mask], mask

    x_focus, x_mask = _axis_subset(x_arr, style.get("xlim"))
    y_focus, y_mask = _axis_subset(y_arr, style.get("ylim"))
    z_focus, z_mask = _axis_subset(z_arr, style.get("zlim"))
    # np.ix_ fancy indexing always copies. When no axis is cropped (the default),
    # return the input array as a view so the uncropped path pays no volume copy.
    if x_mask.all() and y_mask.all() and z_mask.all():
        focused = values
    else:
        focused = values[np.ix_(x_mask, y_mask, z_mask)]
    return focused, x_focus, y_focus, z_focus


def resolve_volume_layout(data: dict, mode_size: int) -> tuple[int, int, int, int] | None:
    """Return `(Nx, Ny, Nz, multiplier)` when `mode_size` matches a 3D layout."""
    nx = int(data.get("Nx", 0) or 0)
    ny = int(data.get("Ny", 0) or 0)
    z_value = data.get("Nz")
    if z_value is None:
        z_coords = data.get("z")
        nz = int(len(z_coords)) if z_coords is not None else 1
    else:
        nz = int(z_value)
    nz = max(nz, 1)
    physical_nspace = nx * ny * nz
    if nx <= 1 or ny <= 1 or nz <= 1 or physical_nspace <= 0:
        return None
    if mode_size % physical_nspace != 0:
        return None
    return nx, ny, nz, mode_size // physical_nspace


def reshape_mode_to_volume(mode_values: np.ndarray, data: dict, *, block_index: int = 0) -> np.ndarray:
    """Reshape a flattened spatial mode into a 3D volume, selecting one block if needed.

    The flattened layout is the data contract (C-order, ``index =
    iz*Ny*Nx + iy*Nx + ix``); the returned array is indexed ``[ix, iy, iz]``
    because the PyVista slice plots downstream are built on
    ``RectilinearGrid(x, y, z)``.
    """
    mode_arr = np.asarray(mode_values)
    layout = resolve_volume_layout(data, mode_arr.size)
    if layout is None:
        raise ValueError(f"Mode of length {mode_arr.size} does not match a volumetric layout.")
    nx, ny, nz, multiplier = layout
    if not 0 <= block_index < multiplier:
        raise ValueError(f"Requested block_index={block_index} but multiplier={multiplier}.")
    blocks = mode_arr.reshape((multiplier, nz, ny, nx))  # contract C-order
    return blocks[block_index].transpose(2, 1, 0)


def plot_orthogonal_slices_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    output_path: str,
    title_prefix: str,
    data: dict,
    slice_indices: tuple[int, int, int] | None = None,
    scalar_name: str = "mode",
) -> None:
    """Render 3 orthogonal slices of a 3D scalar field with PyVista."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyVista is required for 3D slice plots. Install openmodalpy[viz3d] to enable 3D plotting."
        ) from exc

    values, x_arr, y_arr, z_arr = subset_volume_focus_3d(field_3d, x_coords, y_coords, z_coords, data)
    nx, ny, nz = values.shape

    if slice_indices is None:
        slice_indices = (nx // 2, ny // 2, nz // 2)
    ix, iy, iz = slice_indices

    vmin, vmax = get_robust_clim(values, method="percentile")
    center = [float(x_arr[ix]), float(y_arr[iy]), float(z_arr[iz])]

    grid = pv.RectilinearGrid(x_arr, y_arr, z_arr)
    grid.point_data[scalar_name] = values.flatten(order="F")

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600), border=False)
    plotter.set_background("white")

    slice_specs = [
        ("YZ", "x", [center[0], center[1], center[2]], plotter.view_yz, f"x = {center[0]:.3g}"),
        ("XZ", "y", [center[0], center[1], center[2]], plotter.view_xz, f"y = {center[1]:.3g}"),
        ("XY", "z", [center[0], center[1], center[2]], plotter.view_xy, f"z = {center[2]:.3g}"),
    ]

    for idx, (plane_name, normal, origin, view_fn, coord_label) in enumerate(slice_specs):
        plotter.subplot(0, idx)
        slc = grid.slice(normal=normal, origin=origin)
        plotter.add_mesh(
            slc,
            scalars=scalar_name,
            cmap=CMAP_DIV,
            clim=[vmin, vmax],
            show_scalar_bar=(idx == 2),
            scalar_bar_args={"title": "", "n_labels": 3},
        )
        plotter.add_text(f"{title_prefix}\n{plane_name} @ {coord_label}", font_size=10)
        view_fn()
        plotter.enable_parallel_projection()
        plotter.show_bounds(
            grid="front",
            location="outer",
            ticks="outside",
            xtitle="x",
            ytitle="y",
            ztitle="z",
            font_size=9,
            minor_ticks=False,
        )

    plotter.screenshot(output_path)
    plotter.close()
    logger.info("Saving figure %s", output_path)


def plot_isometric_slices_3d(
    field_3d: np.ndarray,
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    output_path: str,
    title_prefix: str,
    data: dict,
    slice_indices: tuple[int, int, int] | None = None,
    scalar_name: str = "mode",
) -> None:
    """Render positive/negative 3D isosurfaces in one isometric PyVista view."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyVista is required for 3D isometric plots. Install openmodalpy[viz3d] to enable 3D plotting."
        ) from exc

    values, x_arr, y_arr, z_arr = subset_volume_focus_3d(field_3d, x_coords, y_coords, z_coords, data)
    nx, ny, nz = values.shape

    vmin, vmax = get_robust_clim(values, method="percentile")
    grid = pv.RectilinearGrid(x_arr, y_arr, z_arr)
    grid.point_data[scalar_name] = values.flatten(order="F")
    abs_scale = max(abs(vmin), abs(vmax))
    if abs_scale <= 0:
        raise ValueError("Cannot build isosurfaces from a zero field.")
    iso_value = 0.45 * abs_scale

    positive = grid.contour(isosurfaces=[iso_value], scalars=scalar_name)
    negative = grid.contour(isosurfaces=[-iso_value], scalars=scalar_name)

    bounds_sources = [mesh.bounds for mesh in (positive, negative) if mesh.n_points]
    if bounds_sources:
        xmin = min(bounds[0] for bounds in bounds_sources)
        xmax = max(bounds[1] for bounds in bounds_sources)
        ymin = min(bounds[2] for bounds in bounds_sources)
        ymax = max(bounds[3] for bounds in bounds_sources)
        zmin = min(bounds[4] for bounds in bounds_sources)
        zmax = max(bounds[5] for bounds in bounds_sources)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    span_z = max(zmax - zmin, 1e-12)
    shock_xmax = xmin + 0.40 * span_x
    focus_bounds = (
        xmin,
        shock_xmax,
        ymin - 0.12 * span_y,
        ymax + 0.12 * span_y,
        zmin - 0.12 * span_z,
        zmax + 0.12 * span_z,
    )
    positive = positive.clip_box(bounds=focus_bounds, invert=False)
    negative = negative.clip_box(bounds=focus_bounds, invert=False)

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900), border=False)
    plotter.set_background("white")
    if positive.n_points:
        plotter.add_mesh(
            positive,
            color="#d1495b",
            opacity=0.62,
            smooth_shading=True,
            specular=0.2,
        )
    if negative.n_points:
        plotter.add_mesh(
            negative,
            color="#3a86ff",
            opacity=0.62,
            smooth_shading=True,
            specular=0.2,
        )

    bounds_sources = [mesh.bounds for mesh in (positive, negative) if mesh.n_points]
    if bounds_sources:
        xmin = min(bounds[0] for bounds in bounds_sources)
        xmax = max(bounds[1] for bounds in bounds_sources)
        ymin = min(bounds[2] for bounds in bounds_sources)
        ymax = max(bounds[3] for bounds in bounds_sources)
        zmin = min(bounds[4] for bounds in bounds_sources)
        zmax = max(bounds[5] for bounds in bounds_sources)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    span_z = max(zmax - zmin, 1e-12)
    pad_x = 0.12 * span_x
    pad_y = 0.12 * span_y
    pad_z = 0.12 * span_z
    focus_bounds = (
        xmin - pad_x,
        xmax + pad_x,
        ymin - pad_y,
        ymax + pad_y,
        zmin - pad_z,
        zmax + pad_z,
    )
    focus_center = (
        0.5 * (focus_bounds[0] + focus_bounds[1]),
        0.5 * (focus_bounds[2] + focus_bounds[3]),
        0.5 * (focus_bounds[4] + focus_bounds[5]),
    )
    max_span = max(
        focus_bounds[1] - focus_bounds[0],
        focus_bounds[3] - focus_bounds[2],
        focus_bounds[5] - focus_bounds[4],
    )

    plotter.add_mesh(
        pv.Box(bounds=focus_bounds),
        style="wireframe",
        color="black",
        line_width=1,
        opacity=0.35,
    )
    plotter.add_text(f"{title_prefix}\niso = ±{iso_value:.3g}", font_size=11)
    plotter.camera.focal_point = focus_center
    plotter.camera.position = (
        focus_center[0] + 1.45 * max_span,
        focus_center[1] + 0.55 * max_span,
        focus_center[2] - 1.35 * max_span,
    )
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.clipping_range = (1e-3, 50.0 * max_span)
    plotter.screenshot(output_path)
    plotter.close()
    logger.info("Saving figure %s", output_path)


def plot_modes_3d(
    kind: str,
    work_items: Iterable[Mapping[str, Any]],
    x_coords: ArrayLike | None,
    y_coords: ArrayLike | None,
    z_coords: ArrayLike | None,
    *,
    data: dict,
) -> None:
    """Dispatch a sequence of 3D mode plots to the slices or isometric renderer.

    Each work item is a mapping with required keys ``mode_3d``, ``output_path``,
    and ``title_prefix``. Optional ``scalar_name`` is forwarded when present so
    callers that omit it keep the renderer default.
    """
    if kind == "slices":
        plot_fn = plot_orthogonal_slices_3d
    elif kind == "isometric":
        plot_fn = plot_isometric_slices_3d
    else:
        raise ValueError(f"kind must be 'slices' or 'isometric', got {kind!r}")

    for item in work_items:
        kwargs = {
            "output_path": item["output_path"],
            "title_prefix": item["title_prefix"],
            "data": data,
        }
        if "scalar_name" in item:
            kwargs["scalar_name"] = item["scalar_name"]
        plot_fn(item["mode_3d"], x_coords, y_coords, z_coords, **kwargs)


# Re-export data loading functions
load_jetles_data = di_load_jetles_data
load_mat_data = di_load_mat_data
load_data = di_load_data


def generate_dummy_data_like_jetles(
    output_path: str,
    Ns: int = 100,
    Nx: int = 30,
    Ny: int = 20,
    dt: float = 0.01,
    f1: float = 5.0,
    f2: float = 2.0,
    noise_level: float = 0.05,
    save_mat: bool = False,
    seed: int = 0,
) -> str:
    """Create a small JetLES-like dataset with simple coherent content.

    This utility generates a synthetic pressure field composed of a few
    low-frequency modes rather than purely random noise.  It is intended for
    quick demonstrations when no real dataset is available.

    Parameters
    ----------
    output_path : str
        Path to the file to create.
    Ns : int, optional
        Number of snapshots (time samples).
    Nx : int, optional
        Number of points in the ``x`` direction.
    Ny : int, optional
        Number of points in the radial ``r`` direction.
    dt : float, optional
        Time step between snapshots.
    f1, f2 : float, optional
        Dominant temporal frequencies of the two synthetic modes.
    noise_level : float, optional
        Amplitude of added Gaussian noise relative to the signal.
    save_mat : bool, optional
        If ``True`` the file is created with ``.mat`` extension, otherwise an
        HDF5 ``.h5`` file is created.  The function does not require SciPy and
        always uses ``h5py`` for writing.
    seed : int, optional
        Seed for the local noise RNG. Same seed yields identical arrays.
    Returns
    -------
    str
        Path to the generated dummy file.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Coordinates stored as 2-D arrays as in the real dataset
    x = np.linspace(0.0, 1.0, Nx)[:, None]
    r = np.linspace(0.0, 1.0, Ny)[None, :]

    # Temporal vector
    t = np.arange(Ns) * dt

    # Simple spatial modes
    mode1 = np.sin(np.pi * x) * np.cos(np.pi * r)
    mode2 = np.cos(0.5 * np.pi * x) * np.sin(2.0 * np.pi * r)

    # Construct coherent pressure field (shape: Nx, Ny, Ns)
    signal = (
        np.sin(2 * np.pi * f1 * t)[:, None, None] * mode1[None, :, :]
        + 0.5 * np.sin(2 * np.pi * f2 * t)[:, None, None] * mode2[None, :, :]
    )

    rng = np.random.default_rng(seed)
    noise = noise_level * rng.standard_normal((Ns, Nx, Ny))
    p = np.transpose(signal + noise, (1, 2, 0))  # (Nx, Ny, Ns)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("p", data=p)
        f.create_dataset("x", data=x)
        f.create_dataset("r", data=r)
        f.create_dataset("dt", data=np.array([[dt]]))

    # Optionally save a ``.mat`` file for compatibility with some loaders
    if save_mat and not output_path.endswith(".mat"):
        mat_path = os.path.splitext(output_path)[0] + ".mat"
        with h5py.File(mat_path, "w") as f:
            f.create_dataset("p", data=p)
            f.create_dataset("x", data=x)
            f.create_dataset("r", data=r)
            f.create_dataset("dt", data=np.array([[dt]]))

    return output_path


def _polar_theta_sector_fractions(z: np.ndarray) -> np.ndarray:
    """Sector fraction Delta-theta / (2*pi) for the polar third axis.

    ``z`` is azimuth theta in radians, strictly increasing, spanning at most
    one revolution (``theta_max - theta_min <= 2*pi + 1e-9``). Theta is
    periodic, so each point's width also counts the gap that wraps back to
    the first point one revolution later: append a virtual point at
    ``theta[0] + 2*pi``, take the plain trapezoid widths of that extended
    axis with ``_trapezoid_widths``, then fold the last (virtual) width back
    onto the first point. A partition covering the full circle then sums to
    exactly 2*pi, whatever its spacing.

    Two full-revolution samplings are accepted: half-open (``endpoint=False``
    style, the wrap gap back to ``theta[0] + 2*pi`` is close to the regular
    interior spacing) and closed (``z`` includes both 0 and ``2*pi``, wrap
    gap 0 -- the plain trapezoid widths of ``z`` itself already sum to
    ``2*pi`` and are used directly, with no wraparound extension). A theta
    axis covering only part of the circle -- a wrap gap much larger than the
    largest interior spacing -- is refused: sector weights assume one full
    revolution, and a wedge needs an explicit ``spatial_weights=`` metric.

    Flattened in the order documented in ``calculate_cell_volume_weights``.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 1:
        raise ValueError(
            "z must be a 1-D azimuth theta axis in radians; got ndim="
            f"{z_arr.ndim}. The polar third axis is azimuth, not a mesh."
        )
    if z_arr.size == 0:
        raise ValueError("z is empty; cannot derive azimuth sector widths.")
    if z_arr.size > 1 and np.any(np.diff(z_arr) <= 0.0):
        raise ValueError(
            f"z is not strictly increasing (range [{z_arr[0]:g}, {z_arr[-1]:g}] "
            "rad). The polar third axis is azimuth theta in radians; sort it "
            "before calling."
        )
    theta_range = float(z_arr[-1] - z_arr[0]) if z_arr.size > 1 else 0.0
    if theta_range > 2.0 * np.pi + 1e-9:
        raise ValueError(
            f"z spans {theta_range:g} rad, more than one revolution (2*pi "
            f"rad ~= {2.0 * np.pi:g}). The polar third axis is azimuth theta "
            "in radians, not a Cartesian z; a Cartesian z passed by mistake "
            "must be caught here."
        )
    if z_arr.size == 1:
        return np.array([1.0])
    wrap_gap = float((z_arr[0] + 2.0 * np.pi) - z_arr[-1])
    max_interior_gap = float(np.diff(z_arr).max())
    if wrap_gap > 1.5 * max_interior_gap:
        raise ValueError(
            f"z covers only part of one revolution: range [{z_arr[0]:g}, "
            f"{z_arr[-1]:g}] rad, wrap gap {wrap_gap:g} rad back to "
            f"theta[0] + 2*pi versus largest interior spacing "
            f"{max_interior_gap:g} rad. Sector weights assume a full "
            "revolution; a partial wedge needs an explicit spatial_weights= "
            "metric instead."
        )
    if abs(wrap_gap) < 1e-9:
        # Closed sampling: z already includes both 0 and 2*pi, so the plain
        # trapezoid widths already sum to 2*pi -- no wraparound extension.
        w = _trapezoid_widths(z_arr, "z")
        return w / (2.0 * np.pi)
    z_ext = np.concatenate([z_arr, [z_arr[0] + 2.0 * np.pi]])
    w_ext = _trapezoid_widths(z_ext, "z")
    w = w_ext[:-1].copy()
    w[0] += w_ext[-1]
    return w / (2.0 * np.pi)


def calculate_polar_weights(
    x: np.ndarray,
    y: np.ndarray,
    z: ArrayLike | None = None,
    use_parallel: bool = True,
    n_space: int | None = None,
) -> np.ndarray:
    """Calculate integration weights for a 2D cylindrical grid (x, r).

    With ``n_space`` set and ``x``/``y`` both 1-D of length ``n_space``, the
    coordinates are read as scattered points rather than grid axes: the
    weight per point is its radius, ``w_i = r_i = |y_i|``. This is the
    cylindrical Jacobian at the point, not a cell measure — it carries no
    integration cell, same as the scattered branch of
    ``calculate_uniform_weights``. ``z`` is ignored in the scattered branch.

    With ``z`` given (and not scattered), ``z`` is a 1-D azimuth axis theta in
    radians and the weight per (x, r, theta) cell is the 2-D (x, r) weight
    times the sector fraction ``Delta-theta / (2*pi)`` (see
    ``_polar_theta_sector_fractions``). Flattened in the order documented in
    ``calculate_cell_volume_weights``.
    """
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.abs(y).reshape(int(n_space), 1)
    if use_parallel and PARALLEL_AVAILABLE:
        return calculate_polar_weights_optimized(x, y, z=z, n_space=n_space)
    # Support both 1-D and 2-D coordinate arrays
    x_line = x[:, 0] if x.ndim > 1 else x
    y_line = y[0, :] if y.ndim > 1 else y
    Nx = x_line.shape[0]
    Ny = y_line.shape[0]

    # Calculate y-direction (r-direction) integration weights (Wy)
    Wy = np.zeros((Ny, 1))

    # First point (centerline)
    if Ny > 1:
        y_mid_right = (y_line[0] + y_line[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y_line[0] ** 2

    # Middle points
    for i in range(1, Ny - 1):
        y_mid_left = (y_line[i - 1] + y_line[i]) / 2
        y_mid_right = (y_line[i] + y_line[i + 1]) / 2
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)

    # Last point
    if Ny > 1:
        y_mid_left = (y_line[-2] + y_line[-1]) / 2
        Wy[Ny - 1] = np.pi * (y_line[-1] ** 2 - y_mid_left**2)

    # Calculate x-direction integration weights (Wx)
    Wx = np.zeros((Nx, 1))

    # First point
    if Nx > 1:
        Wx[0] = (x_line[1] - x_line[0]) / 2
    else:
        Wx[0] = 1.0

    # Middle points
    for i in range(1, Nx - 1):
        Wx[i] = (x_line[i + 1] - x_line[i - 1]) / 2

    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x_line[Nx - 1] - x_line[Nx - 2]) / 2

    if z is None:
        # Combine weights: (Ny, Nx) outer product, flattened C-order.
        W = np.reshape(np.outer(Wy.ravel(), Wx.ravel()), (Nx * Ny, 1))
        return W

    # 3-D polar: fold in the azimuth sector fraction and flatten (theta, r, x).
    theta_fraction = _polar_theta_sector_fractions(np.asarray(z, dtype=np.float64))
    volumes_2d = np.outer(Wy.ravel(), Wx.ravel())  # (Ny, Nx)
    volumes = theta_fraction[:, None, None] * volumes_2d[None, :, :]  # (Ntheta, Ny, Nx)
    return volumes.reshape(-1, 1)


def calculate_uniform_weights(
    x: np.ndarray, y: np.ndarray, z: ArrayLike | None = None, n_space: int | None = None
) -> np.ndarray:
    """Return uniform weights for a Cartesian grid or a scattered point set.

    Returns an all-ones column of length ``n_space`` when the coordinates are
    scattered (1-D ``x`` and ``y``, ``len(x) == len(y) == n_space``), and the
    tensor product ``Nx*Ny*Nz`` otherwise. The two readings collide only when
    ``n == 1``; scattered is preferred then only if ``n_space`` says so.
    With ``n_space=None`` the result is always the tensor product (historical
    behaviour). Grid spacing / cell volumes are not applied; callers that need
    a domain integral must supply their own W. Flattened in the order
    documented in ``calculate_cell_volume_weights``.
    """
    if (
        n_space is not None
        and x.ndim == 1
        and y.ndim == 1
        and int(x.shape[0]) == int(n_space)
        and int(y.shape[0]) == int(n_space)
    ):
        return np.ones((int(n_space), 1))
    # Support both 1-D and 2-D coordinate arrays
    if x.ndim > 1:
        Nx, Ny = x.shape
    elif y.ndim > 1:
        Nx, Ny = y.shape
    else:
        Nx, Ny = x.shape[0], y.shape[0]
    if z is None:
        Nz = 1
    else:
        z_arr = np.asarray(z)
        Nz = int(z_arr.shape[0] if z_arr.ndim > 0 else 1)
    return np.ones((Nx * Ny * Nz, 1))


def _trapezoid_widths(a: np.ndarray, name: str) -> np.ndarray:
    """Trapezoid cell widths along one 1-D axis (half spacing at the ends).

    A single-point axis gets width 1.0 so an outer product across axes stays
    well defined (same convention as the polar helper).
    """
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D coordinate array to derive cell volumes; "
            f"got ndim={a.ndim}. Pass the axis coordinates, not a mesh."
        )
    if a.size == 0:
        raise ValueError(f"{name} is empty; cannot derive cell widths.")
    if a.size == 1:
        return np.ones(1)
    d = np.diff(a)
    if np.any(d <= 0.0):
        if np.all(d < 0.0):
            raise ValueError(
                f"{name} is strictly decreasing "
                f"({a[0]:g} -> {a[-1]:g}). Cell-volume weights require "
                f"strictly increasing coordinates; flip the axis first "
                f"(e.g. {name} = {name}[::-1] together with the matching "
                f"axis of q) instead of relying on a silent sort."
            )
        raise ValueError(
            f"{name} is not strictly increasing (first bad step near index "
            f"{int(np.argmax(d <= 0.0))}). Cell-volume weights refuse to "
            f"sort or repair coordinates; pass monotone 1-D {name}."
        )
    w = np.empty_like(a)
    w[0] = (a[1] - a[0]) / 2.0
    w[-1] = (a[-1] - a[-2]) / 2.0
    w[1:-1] = (a[2:] - a[:-2]) / 2.0
    return w


def calculate_cell_volume_weights(x: ArrayLike, y: ArrayLike, z: ArrayLike | None = None) -> np.ndarray:
    """Cell-volume weights for a (possibly stretched) Cartesian grid.

    Each axis contributes trapezoid cell widths (half the neighbouring
    spacing at the boundary points); the outer product across x, y[, z] gives
    one weight per cell, flattened in C-order to match the snapshot layout
    ``(Ns, Ny*Nx*Nz)``: ``index = ((iz*Ny + iy)*Nx + ix)`` (``iy*Nx + ix``
    in 2-D). Coordinates must be strictly increasing 1-D arrays; anything
    else raises ``ValueError`` — decreasing axes are told to flip, never
    sorted silently.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    wx = _trapezoid_widths(x_arr, "x")
    wy = _trapezoid_widths(y_arr, "y")
    volumes = np.outer(wy, wx)  # (Ny, Nx); C-order flatten -> iy*Nx + ix
    if z is not None:
        z_arr = np.asarray(z, dtype=np.float64)
        wz = _trapezoid_widths(z_arr, "z")
        volumes = wz[:, None, None] * volumes[None, :, :]  # (Nz, Ny, Nx)
    return volumes.reshape(-1, 1)


def blocksfft(
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
    Compute blocked FFT using Welch's method for CSD estimation.

    Parameters:
    q (np.ndarray): Input data [time, space]
    nfft (int): Number of FFT points
    nblocks (int): Number of blocks
    novlap (int): Number of overlapping points between blocks
    blockwise_mean (bool): Subtract blockwise mean if True
    normvar (bool): If True, divide each block pointwise in space by its variance
        (unbiased, ``ddof=1``), matching ``spod_matlab`` (``opts.normvar``) and
        PySPOD (``normalize_data``). This does **not** produce unit variance and
        is therefore scale-dependent: scaling the input by ``c`` scales the
        normalized block by ``1/c``. Values below ``4*eps`` are clamped to 1.
        Implementation option, not a step in Towne, Schmidt & Colonius (2018).
        Defaults to False.
    window_norm (str): Window normalization type ('amplitude' or 'power')
    window_type (str): Window type. Use 'sine' for the custom sine window or any
        name recognized by ``scipy.signal.get_window`` (e.g., 'hamming', 'hann',
        'blackman', etc.)

    Returns:
    q_hat (np.ndarray): FFT coefficients [freq, space, block]

    ---
    IMPORTANT:
    - This function assumes the FFT backend (numpy, scipy, pyfftw, etc.) does NOT normalize the FFT by default (which is true for standard backends).
    - If you use a backend or option that applies normalization (e.g., norm='ortho'), REMOVE the
      division by nfft in ``welch.windowed_block_fft`` to avoid double normalization.
    - SPOD callers pass ``dst`` into ``spod_function`` as a spectral weight. In this
      codebase ``dst`` is the Strouhal step (``St[1] - St[0] = df * L / U``), not
      the raw frequency resolution ``df = fs / nfft``. Reported SPOD eigenvalues
      therefore scale with ``U/L``; with the default L = U = 1 the two coincide.
    - Block starts are ``iblk * (nfft - novlap)`` with no end-of-record clamp.
      Callers must pass an ``nblocks`` that fits; oversize requests raise
      ``ValueError`` rather than re-using trailing samples.
    ---
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


def require_spatial_metric(weights: ArrayLike) -> None:
    """Raise ``ValueError`` if ``weights`` do not define an inner product.

    A metric that is not an inner product must not reach a solver. Isolated
    zeros among positive weights stay allowed -- a zero-measure cell
    contributes nothing (same as SPOD and BSMD). What is rejected: complex
    entries, non-finite entries, negative entries, and a zero total measure.
    Single definition, used by the decomposition seam, SPOD (via
    ``_coerce_spatial_weights``) and BSMD.
    """
    weights = np.asarray(weights)
    if np.iscomplexobj(weights):
        raise ValueError(
            "Spatial metric is complex. Casting it to real would silently discard the "
            "imaginary part and hand the solver a metric the caller never asked for."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            f"Spatial metric contains {np.count_nonzero(~np.isfinite(weights))} non-finite "
            "weight(s) (NaN or inf). This would otherwise surface much later as an "
            "unhelpful LAPACK error from inside the eigensolver."
        )
    if np.any(weights < 0):
        raise ValueError(
            f"Spatial metric contains {np.count_nonzero(weights < 0)} negative weight(s) "
            f"(most negative: {float(np.min(weights)):.6g}). A negative entry means the "
            "metric is not an inner product, so any energy computed from it is meaningless."
        )
    # Reached only when every weight is >= 0, so this means all of them are zero:
    # the domain has no measure, and the usual cause is worth naming.
    if np.sum(weights) <= 0:
        raise ValueError(
            f"Spatial metric has zero total measure ({weights.size} weights, all zero), so "
            "it defines no inner product. The usual cause is polar weights on a grid whose "
            "radial coordinate is 0: every annulus area is pi*r**2 = 0. Note the condition "
            "is r > 0, not Ny > 1 -- a single radial station at r > 0 is fine."
        )


def _coerce_spatial_weights(w: ArrayLike, expected_len: int) -> np.ndarray:
    """Accepted weight shapes -> 1-D vector of length ``expected_len``.

    Routes: 1-D; ``(n, 1)``; square matrix (its diagonal); non-square
    ``(n, k)`` row-major flatten; 3-D per-component stacked diagonals.

    A square (or 3-D stack of square planes) is always reduced to its
    diagonal: nothing off-diagonal is kept. The matrix is accepted (then
    reduced) only when it is numerically diagonal under the scale-invariant
    ratio

        r = max_{i != j} |W_ij| / sqrt(|W_ii| * |W_jj|)

    Reject when ``r > C * n * eps``, where ``n`` is the matrix side, ``eps``
    is machine epsilon of the array's own floating dtype (float64 for
    integer or object input), and ``C = 2``. At n = 4 that is a 3.8x margin
    over measured float32-after-arithmetic round-off (r ≈ 2.10 eps) and
    still rejects a float64 change of basis at 1e-14 (r ≈ 157 eps). A
    non-finite (NaN or inf) off-diagonal is rejected. A zero on the
    diagonal makes its row and column strict (any non-zero coupling gives
    r = inf). This package cannot represent off-diagonal coupling at all;
    pass ``np.diag(W)`` only if the diagonal is what was meant.
    """
    w = np.asarray(w)
    # C=2 → 3.8x margin over measured float32-after-arithmetic (r ≈ 2.10 eps).
    _diagonality_c = 2

    def _metric_eps(arr: np.ndarray) -> float:
        try:
            return float(np.finfo(arr.dtype).eps)
        except (TypeError, ValueError):
            return float(np.finfo(np.float64).eps)

    def _coupling_ratio(plane: np.ndarray) -> float:
        n = int(plane.shape[0])
        if n <= 1:
            return 0.0
        off_mask = ~np.eye(n, dtype=bool)
        off = plane[off_mask]
        abs_off = np.asarray(np.abs(off), dtype=np.float64)
        diag = np.asarray(np.abs(np.diag(plane)), dtype=np.float64)
        rows, cols = np.nonzero(off_mask)
        scale = np.sqrt(diag[rows] * diag[cols])
        ratio = np.zeros_like(abs_off)
        nz = scale > 0
        ratio[nz] = abs_off[nz] / scale[nz]
        ratio[~nz] = np.where(abs_off[~nz] > 0, np.inf, 0.0)
        return float(np.max(ratio))

    def _largest_rejected_offdiag(planes: list[np.ndarray]) -> float | None:
        # r > C * n * eps(dtype) per plane; non-finite off-diag => inf.
        worst: float | None = None
        for plane in planes:
            n = int(plane.shape[0])
            r = _coupling_ratio(plane)
            limit = _diagonality_c * n * _metric_eps(plane)
            if not np.isfinite(r) or r > limit:
                max_off = float(np.max(np.abs(plane - np.diag(np.diag(plane)))))
                worst = max_off if worst is None else max(worst, max_off)
        return worst

    def _reject_nondiagonal_square(max_off: float, shape: tuple[int, ...]) -> None:
        if len(shape) == 3:
            head = (
                "A 3-D stack of planes is read as stacked diagonals and cannot "
                f"represent off-diagonal coupling in an array of shape {shape}"
            )
        else:
            head = (
                "A square spatial metric is read as its diagonal and cannot "
                f"represent off-diagonal coupling in an array of shape {shape}"
            )
        raise ValueError(
            f"{head} (largest off-diagonal magnitude {max_off}). "
            "This package cannot represent off-diagonal coupling at all. "
            "If the diagonal is what was meant, pass np.diag(W)."
        )

    # Shape work only — do not cast to float yet. A complex array must reach
    # require_spatial_metric with its imaginary part intact; casting first would
    # emit ComplexWarning and hand the real part to the metric checks.
    if w.ndim == 3:
        if w.shape[0] != w.shape[1]:
            raise ValueError("weight array's first two dimensions must be equal")
        worst = _largest_rejected_offdiag([w[:, :, i] for i in range(w.shape[2])])
        if worst is not None:
            _reject_nondiagonal_square(worst, w.shape)
        w = np.stack([np.diag(w[:, :, i]) for i in range(w.shape[2])], axis=1)
    elif w.ndim == 2:
        if w.shape[0] == w.shape[1] and w.shape[1] != 1:
            worst = _largest_rejected_offdiag([w])
            if worst is not None:
                _reject_nondiagonal_square(worst, w.shape)
            w = np.diag(w)
        elif w.shape[1] > 1:
            w = w.reshape(-1)
        else:
            w = w.ravel()
    weights = np.asarray(w).reshape(-1)
    if weights.size != expected_len:
        raise ValueError(f"Weight vector length {weights.size} does not match n_space={expected_len}")
    if weights.dtype == object:
        # Same cast the return already performs. Doing it first lets
        # require_spatial_metric inspect a numeric object diagonal; np.isfinite
        # cannot read object dtype.
        weights = np.asarray(weights, dtype=float)
    require_spatial_metric(weights)
    return np.asarray(weights, dtype=float)


def _as_spatial_weight_column(w: ArrayLike, n_space: int | None = None) -> np.ndarray:
    """Accepted weight shapes -> column of shape ``(n_space, 1)``.

    ``n_space`` defaults to the length ``_coerce_spatial_weights`` would
    produce from ``w`` (vector length, column height, or square side).
    Pass it explicitly to reject a wrong-length input.
    """
    arr = np.asarray(w)
    if n_space is None:
        if arr.ndim == 3:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("weight array's first two dimensions must be equal")
            n_space = int(arr.shape[0] * arr.shape[2])
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[1] != 1:
            n_space = int(arr.shape[0])
        else:
            n_space = int(arr.size)
    return _coerce_spatial_weights(arr, int(n_space)).reshape(-1, 1)


def _reported_grid(data: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Return ``(Nx, Ny, Nz)`` when ``data`` carries a grid claim, else ``None``.

    A missing axis is extent 1. Absence of all three keys is not a grid —
    callers must not invent one to check against ``q.shape[1]``. A set of
    keys whose extents are all 1 also says nothing about extent (a leftover
    ``Nz: 1`` is not a 1×1×1 claim against a wider matrix). A lone ``Nx``
    that names a real extent is still a claim.
    """
    if not any(key in data for key in ("Nx", "Ny", "Nz")):
        return None

    def extent(name: str) -> int:
        if name not in data or data[name] is None:
            return 1
        return int(data[name])

    grid = extent("Nx"), extent("Ny"), extent("Nz")
    if grid == (1, 1, 1):
        return None
    return grid


def spod_function(
    qhat: np.ndarray,
    nblocks: int,
    dst: float,
    w: np.ndarray,
    return_psi: bool = False,
    use_parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SPOD modes and eigenvalues for a single frequency.
    Args:
        qhat (np.ndarray): FFT coefficients for this frequency [space, block].
        nblocks (int): Number of blocks.
        dst (float): Spectral weight used as ``1/sqrt(nblocks * dst)``. Callers in
            this package pass the Strouhal step (not necessarily ``df = fs/nfft``).
        w (np.ndarray): Spatial integration weights [space, 1].
        return_psi (bool): If True, also return psi (time coefficients).
    Returns:
        tuple: (phi, lambda_tilde[, psi])
            phi (np.ndarray): Spatial SPOD modes for this frequency [space, mode].
            lambda_tilde (np.ndarray): SPOD eigenvalues (energy) for this frequency [mode].
            psi (np.ndarray, optional): Time coefficients for this frequency [block, mode].
    """
    if use_parallel and PARALLEL_AVAILABLE:
        # Pass w through unchanged — same as the serial branch. The shared body
        # coerces and validates once; a pre-flatten here was a second check.
        return spod_single_frequency_optimized(
            qhat,
            w,
            nblocks,
            dst,
            return_psi=return_psi,
        )

    # Same eigenproblem as the parallel entry; body lives in decomposition.py.
    from openmodalpy.core.decomposition import spod_single_frequency

    return spod_single_frequency(qhat, nblocks, dst, w, return_psi=return_psi)


class BaseAnalyzer:
    """Base class for modal decomposition analyzers."""

    # Declared here so shared helpers (e.g. ``_resync_mode_count``) type-check.
    # Concrete analyzers populate them in ``__init__`` / after decomposition.
    n_modes_save: int
    modes: np.ndarray
    eigenvalues: np.ndarray
    time_coefficients: np.ndarray

    def __init__(
        self,
        file_path: str | None = None,
        nfft: int = 128,
        overlap: float = 0.5,
        results_dir: str = "./preprocess",
        figures_dir: str = "./figs",
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        use_parallel: bool = True,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            file_path (str | None): Path to data file. Optional when ``data``
                carries the loaded dataset instead.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks.
            results_dir (str): Directory to save results.
            figures_dir (str): Directory to save figures.
            data_loader (callable): Function to load data.
            spatial_weight_type (str | None): Type of spatial weighting
                (``None`` → ``"uniform"``, or ``"uniform"``, ``"polar"``,
                ``"prescribed"``, ``"cell_volume"``). ``"auto"`` is not accepted.
            spatial_weights: Optional array of spatial integration weights.
                When given, the weight type becomes ``"prescribed"`` and the
                vector is checked against the grid in ``load_and_preprocess``.
            data (dict | None): Already-loaded dataset following the data
                contract (``q``, ``x``, ``y``, ``dt``, ``Nx``, ``Ny``, ``Ns``;
                see DOC.md). Given instead of ``file_path`` — exactly one of
                the two is required. The dict is stored by reference, so one
                load can feed several analyzers.
        """
        self.file_path = file_path

        # Exactly one input source. ``data`` is the documented in-memory path;
        # assigning ``.data`` after construction stays as the legacy escape hatch.
        if data is not None:
            if file_path is not None:
                raise ValueError("Pass file_path or data, not both: an analyzer takes exactly one input source.")
            if not isinstance(data, dict) or not data:
                raise ValueError(
                    "data must be a non-empty dict following the data contract (q, x, y, dt, Nx, Ny, Ns; see DOC.md)."
                )
        elif file_path is None:
            raise ValueError("No input source: pass file_path (path to a data file) or data (the loaded dict).")
        self.nfft = nfft
        self.overlap = overlap
        self.results_dir = results_dir
        self.figures_dir = figures_dir

        # Set default data loader based on file type
        self.data_loader = data_loader or load_data
        self.use_parallel = use_parallel

        # Weight type / prescribed vector — one validation site for all analyzers.
        # None means "not specified" and resolves to uniform (same numeric default
        # as the former unconditional "auto" path). Keep None out of the conflict
        # check so spatial_weights=array with no type still prescribes a metric.
        accepted = ("uniform", "polar", "prescribed", "cell_volume")
        self._prescribed_spatial_weights: ArrayLike | None
        if spatial_weights is not None:
            if spatial_weight_type not in (None, "prescribed"):
                raise ValueError(
                    f"spatial_weights was given, but spatial_weight_type={spatial_weight_type!r} "
                    f"conflicts with it. Use spatial_weight_type='prescribed' (or omit / None), "
                    f"or drop spatial_weights. Accepted types: {', '.join(accepted)}."
                )
            self.spatial_weight_type = "prescribed"
            self._prescribed_spatial_weights = spatial_weights
        else:
            if spatial_weight_type is None:
                self.spatial_weight_type = "uniform"
            elif spatial_weight_type not in accepted:
                raise ValueError(
                    f"spatial_weight_type={spatial_weight_type!r} is not recognised. "
                    f"Accepted values: {', '.join(accepted)}."
                )
            elif spatial_weight_type == "prescribed":
                raise ValueError("spatial_weight_type='prescribed' requires a spatial_weights array.")
            else:
                self.spatial_weight_type = spatial_weight_type
            self._prescribed_spatial_weights = None

        # Calculated later
        self.novlap = int(overlap * nfft)
        self.data: dict[str, Any] = data if data is not None else {}
        self.W = np.array([])
        self.nblocks = 0
        self.fs = 0.0
        self.qhat = np.array([])
        # Set to a real path by compute_fft_blocks when a cache file is in use.
        # Declared here so it always exists and readers can test it for None.
        self._qhat_cache_path: str | None = None

        # Extract root name for output files. With in-memory data there is no
        # path to derive one from, so outputs are named after the analyzer.
        if file_path is not None:
            base = os.path.basename(file_path)
            root, ext = os.path.splitext(base)
            if ext == ".npz":
                npz_files = glob.glob(os.path.join(os.path.dirname(file_path), "*.npz"))
                if len(npz_files) > 1:
                    # Use directory name if multiple npz files were concatenated
                    self.data_root = os.path.basename(os.path.dirname(file_path))
                else:
                    self.data_root = root
            else:
                self.data_root = root
        else:
            self.data_root = getattr(self, "_METHOD_NAME", "analyzer")

        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_and_preprocess(self) -> None:
        """Load data and calculate weights."""
        # Load data from file only if not already provided. The constructor
        # guarantees a non-empty dict whenever ``data=`` was given, so an empty
        # dict here can only come from a legacy side-channel assignment, which
        # keeps its old reload semantics.
        if not self.data:
            self.data = self.data_loader(self.file_path)

        # data_loader is any (str) -> dict. If the dataset reports a grid, its
        # product must be the snapshot width. Skip when there is no claim
        # (no keys, or extents that are all 1) rather than inventing one.
        grid = _reported_grid(self.data)
        q_array = np.asarray(self.data["q"])
        # A q that is not (time, space) has no width to compare. Leave it to the
        # error it already raised downstream rather than adding an IndexError here.
        n_space = int(q_array.shape[1]) if q_array.ndim == 2 else None
        if grid is not None and n_space is not None:
            nx, ny, nz = grid
            grid_nspace = nx * ny * nz
            if grid_nspace != n_space:
                raise ValueError(
                    f"grid product Nx*Ny*Nz={grid_nspace} does not match "
                    f"q.shape[1]={n_space} (grid Nx={nx}, Ny={ny}, Nz={nz})"
                )

        # Calculate spatial weights. Every type is a column (n_space, 1).
        if self.spatial_weight_type == "prescribed":
            # n_space from the snapshot matrix (time × space); helpers check
            # length/shape and that the metric is an inner product.
            if n_space is None:
                n_space = int(np.asarray(self.data["q"]).shape[1])
            # Invariant from __init__: prescribed type always carries a vector.
            self.W = _as_spatial_weight_column(cast(ArrayLike, self._prescribed_spatial_weights), n_space)
            logger.info("Using prescribed spatial weights.")
        elif self.spatial_weight_type == "polar":
            self.W = _as_spatial_weight_column(
                calculate_polar_weights(
                    self.data["x"],
                    self.data["y"],
                    z=self.data.get("z"),
                    use_parallel=self.use_parallel,
                    n_space=n_space,
                )
            )
            logger.info("Using polar (cylindrical) spatial weights.")
        elif self.spatial_weight_type == "cell_volume":
            x_arr = np.asarray(self.data["x"])
            y_arr = np.asarray(self.data["y"])
            z_raw = self.data.get("z")
            z_arr = None if z_raw is None else np.asarray(z_raw)
            axes_1d = x_arr.ndim == 1 and y_arr.ndim == 1 and (z_arr is None or z_arr.ndim == 1)
            scattered = n_space is not None and x_arr.size == n_space and y_arr.size == n_space
            n_axes = x_arr.size * y_arr.size * (1 if z_arr is None else z_arr.size)
            if axes_1d and not scattered and n_space is not None and n_axes == n_space:
                # 1-D monotone axis coordinates describe a Cartesian grid.
                # Non-monotone axes raise from the helper; they are never sorted.
                self.W = _as_spatial_weight_column(calculate_cell_volume_weights(x_arr, y_arr, z_arr))
                logger.info("Using cell-volume spatial weights from the 1-D grid coordinates.")
            else:
                raise ValueError(
                    "spatial_weight_type='cell_volume' needs 1-D Cartesian axis "
                    f"coordinates whose sizes multiply to q.shape[1]={n_space}; got "
                    f"x.shape={x_arr.shape}, y.shape={y_arr.shape}"
                    + (f", z.shape={z_arr.shape}" if z_arr is not None else "")
                    + ". A scattered point set has no cells; pass explicit "
                    "spatial_weights= for a real metric."
                )
        else:
            self.W = _as_spatial_weight_column(
                calculate_uniform_weights(self.data["x"], self.data["y"], self.data.get("z"), n_space=n_space)
            )
            logger.info("Using uniform spatial weights (rectangular grid).")

        if n_space is not None and int(np.asarray(self.W).size) != n_space:
            raise ValueError(
                f"spatial metric length {int(np.asarray(self.W).size)} does not match "
                f"q.shape[1]={n_space}. The metric that enters the inner product "
                f"must have one weight per snapshot column. For scattered points "
                f"pass 1-D x and y of length {n_space}; for a Cartesian grid pass "
                f"the axis coordinates (and z when Nz > 1). Polar weights are 2-D "
                f"only (x, r) — they ignore z."
            )

        # Welch floor partitioning (scipy.signal.welch): drop the remainder so
        # every block is an independent ensemble member. Ceil + end-clamp re-uses
        # samples in the last block and biases SPOD/BSMD eigenvalues.
        # Shared helper with commands._apply_snapshot_limit (welch_nblocks).
        Ns = int(self.data["Ns"])
        self.nblocks = welch_nblocks(Ns, self.nfft, self.novlap)
        if Ns < self.nfft or self.nblocks < 1:
            raise ValueError(
                f"Cannot form Welch blocks: Ns={Ns}, nfft={self.nfft} "
                f"(overlap={self.overlap}, novlap={self.novlap}) yield "
                f"nblocks={self.nblocks}"
            )
        # Divide by the validated local, not the dict entry: a float32 or 0-d
        # array dt would otherwise give a slightly different fs than the value
        # just checked. self.data["dt"] is deliberately left as the loader set it.
        self.fs = 1 / self._require_dt()

        logger.info(
            "Data loaded: %s snapshots, %s×%s spatial points",
            self.data["Ns"],
            self.data.get("Nx"),
            self.data.get("Ny"),
        )
        if self.nfft > 1:
            logger.info(
                "FFT parameters: %s points, %s%% overlap, %s blocks [backend: %s]",
                self.nfft,
                self.overlap * 100,
                self.nblocks,
                FFT_BACKEND,
            )

    def _require_dt(self) -> float:
        """Return a validated positive finite timestep from ``self.data``.

        Does not write to ``self.data["dt"]``. Physics-bearing consumers must
        call this instead of ``self.data.get("dt", 1.0)``.
        """
        dt = self.data.get("dt") if self.data else None
        try:
            if dt is None:
                raise TypeError("dt is None")
            dt_f = float(dt)
        except (TypeError, ValueError):
            # Absent, None, or not a scalar (e.g. an array) — all the same failure
            # to the caller, so they get the same exception rather than a KeyError
            # or TypeError leaking out of a public method.
            raise ValueError(
                f"Missing or non-scalar timestep dt={dt!r} for data source {self.file_path!r}; "
                "provide a positive finite scalar dt in the data dict or loader."
            ) from None
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            raise ValueError(
                f"Invalid timestep dt={dt!r} for data source {self.file_path!r}; "
                "provide a positive finite scalar dt in the data dict or loader."
            )
        return dt_f

    def _require_fs(self) -> float:
        """Return a validated positive finite sampling rate from ``self.fs``.

        Does not write to ``self.fs``. Physics-bearing consumers (periodograms,
        ``rfftfreq`` axes, Nyquist limits) must call this instead of reading
        ``self.fs`` directly — the ``0.0`` default is a not-yet-computed
        sentinel, not a valid rate.
        """
        fs = self.fs
        try:
            fs_f = float(fs)
        except (TypeError, ValueError):
            raise ValueError(
                f"Missing or non-scalar sampling rate fs={fs!r} for data source {self.file_path!r}; "
                "provide a positive finite scalar fs (via load_and_preprocess from a valid dt)."
            ) from None
        if not np.isfinite(fs_f) or fs_f <= 0.0:
            raise ValueError(
                f"Invalid sampling rate fs={fs!r} for data source {self.file_path!r}; "
                "provide a positive finite scalar fs (via load_and_preprocess from a valid dt)."
            )
        return fs_f

    def _time_axis(self, n: int) -> tuple[np.ndarray, str]:
        """Return abscissa values and an honest x-axis label of length ``n``.

        Prefer an explicit time vector in ``self.data["t"]`` when present and
        long enough. Otherwise scale sample indices by a usable positive finite
        ``dt``. When no usable timestep exists, return sample indices and a
        label that does not claim units of time.

        Never raises for missing, ``None``, zero, or non-finite ``dt`` — this is
        a plot axis, not a physical claim. Callers that need Hz or growth rates
        must use :meth:`_require_dt` instead.
        """
        data = self.data if self.data else {}
        t = data.get("t")
        if t is not None:
            t_arr = np.asarray(t)
            if t_arr.size >= n:
                return np.asarray(t_arr[:n], dtype=float), "Time [s]"

        dt = data.get("dt")
        dt_f: float | None
        try:
            if dt is None:
                raise TypeError("dt is None")
            dt_f = float(dt)
        except (TypeError, ValueError):
            dt_f = None
        if dt_f is not None and np.isfinite(dt_f) and dt_f > 0.0:
            return np.arange(n, dtype=float) * dt_f, "Time [s]"

        return np.arange(n, dtype=float), "Sample index"

    def _fft_block_cache_path(self) -> str | None:
        """Return the per-analysis-type FFT cache path, or None to skip caching.

        Caching is tied to ``analysis_type`` so a plain ``BaseAnalyzer`` (used
        by quiet/library tests) keeps computing in memory only, while SPOD,
        BSMD and PSD-POD each write their own ``..._<type>.hdf5`` file.
        """
        analysis_type = getattr(self, "analysis_type", None)
        if not analysis_type:
            return None
        filename = make_result_filename(
            self.data_root,
            self.nfft,
            self.overlap,
            self.data.get("Ns", 0),
            analysis_type,
        )
        return os.path.join(self.results_dir, filename)

    def _sibling_fft_cache_paths(self, own_path: str) -> list[str]:
        """Other ``..._<analysis>.hdf5`` files in ``results_dir`` for the same params.

        All Welch-family analyzers produce bit-identical FFT blocks for matching
        parameters, so a stamp-matching sibling may be adopted. Lookup is
        confined to this analyzer's ``results_dir`` (not a global results constant).
        """
        if not getattr(self, "analysis_type", None) or not os.path.isdir(self.results_dir):
            return []
        stem = f"{self.data_root}_Nfft{self.nfft}_ovlap{self.overlap}_{self.data.get('Ns', 0)}snapshots_"
        own_name = os.path.basename(own_path)
        siblings: list[str] = []
        for name in sorted(os.listdir(self.results_dir)):
            if name == own_name:
                continue
            if name.startswith(stem) and name.endswith(".hdf5"):
                siblings.append(os.path.join(self.results_dir, name))
        return siblings

    def _try_load_fft_block_cache(self, cache_path: str) -> bool:
        """Load FFT blocks from ``cache_path`` when shape and stamp match.

        Returns True on a usable hit (sets ``qhat``, ``nblocks``, ``qhat_cached``).
        On missing file, missing dataset, stamp mismatch, or unreadable file,
        returns False so the caller recomputes. Read failures are fail-soft;
        write failures are not handled here.

        Fail-soft covers a malformed payload, not just an unreadable file. Since
        this now also opens files written by OTHER analyses, a neighbour's bad
        cache must never abort this run: a wrong-rank dataset, a group where a
        dataset belongs, or a stamp attribute that will not cast all recompute.
        """
        if not os.path.exists(cache_path):
            return False
        try:
            with h5py.File(cache_path, "r") as f:
                if "FFTBlocks" not in f:
                    return False
                qhat_cached = f["FFTBlocks"][:]
                if qhat_cached.ndim != 3 or qhat_cached.shape[0] != self.nfft // 2 + 1:
                    return False
                if not _verify_qhat_stamp(f, self, self.data["q"]):
                    return False
                self.qhat = qhat_cached
                self.nblocks = qhat_cached.shape[2]
                self.qhat_cached = True
                logger.info("Loaded cached FFT blocks from %s", cache_path)
                return True
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning(
                "Failed to load cached FFT blocks from %s (%s), recomputing.",
                cache_path,
                exc,
            )
            return False

    def _save_fft_block_cache(self, cache_path: str) -> None:
        """Write ``self.qhat`` and its parameter stamp to ``cache_path``."""
        os.makedirs(self.results_dir, exist_ok=True)
        mode = _hdf5_write_mode(cache_path)
        with h5py.File(cache_path, mode) as f:
            if "FFTBlocks" in f:
                del f["FFTBlocks"]
            f.create_dataset("FFTBlocks", data=self.qhat, compression="gzip")
            if mode == "w":
                for key, value in self._get_metadata().items():
                    f.attrs[key] = value
            _write_qhat_stamp(f, self, self.data["q"])
        logger.info("Saved FFT blocks to cache at %s", cache_path)

    def _on_fft_blocks_ready(self) -> None:
        """Hook after FFT blocks are available (cache hit or fresh compute).

        Subclasses such as BSMD use this for frequency axes and optional
        disk offload; the default is a no-op.
        """

    def compute_fft_blocks(self) -> None:
        """Compute blocked FFT using Welch's method, with optional disk cache.

        When ``analysis_type`` is set, blocks are loaded from / saved to a
        per-type HDF5 file under ``results_dir``. Stamp verification lives
        only here so SPOD, BSMD and PSD-POD share one implementation.

        Lookup order: this analyzer's own cache file, then other
        ``..._<analysis>.hdf5`` siblings in the same ``results_dir`` whose
        stamp matches. Adopting a sibling still writes only this analyzer's
        own file (a local copy for later runs).
        """
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        self.qhat_cached = False
        cache_path = self._fft_block_cache_path()
        if cache_path is not None:
            # BSMD save_results compares this path to decide append vs rewrite.
            self._qhat_cache_path = cache_path
            candidates = [cache_path, *self._sibling_fft_cache_paths(cache_path)]
            for path in candidates:
                if self._try_load_fft_block_cache(path):
                    if path != cache_path:
                        # Keep a copy under our own name so the next run hits
                        # the own-file path without depending on the sibling.
                        self._save_fft_block_cache(cache_path)
                    self._on_fft_blocks_ready()
                    return

        pools = get_threadpool_summary()
        logger.info(
            "Computing FFT with %s blocks on %s backend [pools: %s]",
            self.nblocks,
            FFT_BACKEND,
            pools,
        )
        self.qhat = blocksfft(
            self.data["q"],
            self.nfft,
            self.nblocks,
            self.novlap,
            blockwise_mean=getattr(self, "blockwise_mean", False),
            normvar=getattr(self, "normvar", False),
            window_norm=getattr(self, "window_norm", "power"),
            window_type=getattr(self, "window_type", "hamming"),
        )
        logger.info("FFT computation complete.")

        if cache_path is not None:
            self._save_fft_block_cache(cache_path)
        self._on_fft_blocks_ready()

    def save_results(self, filename: str | None = None) -> None:
        """Save results to HDF5 file with harmonized filename and format.

        Args:
            filename: Custom filename. If omitted, uses the harmonized scheme
                with ``self.analysis_type``.
        """
        from openmodalpy.core.results import write_results

        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                getattr(self, "analysis_type", "spod"),
            )
        save_path = os.path.join(self.results_dir, filename)
        logger.info("Saving results to %s", save_path)
        # Placeholder — subclasses write their full payload through write_results.
        write_results(
            save_path,
            {
                "x": self.data["x"],
                "y": self.data["y"],
                "W": self.W,
            },
            attrs={
                "nfft": self.nfft,
                "overlap": self.overlap,
                "nblocks": self.nblocks,
                "fs": self.fs,
            },
        )

    # Analysis-sequence seam.
    # Subclasses declare their perform entry and whether Welch blocks precede
    # it; plotting policy lives in _plot_run. Both the library entry point and
    # the CLI call run_analysis, so the two paths cannot drift again.
    _perform_name: str
    _needs_fft_blocks: bool = False

    def run_analysis(
        self,
        *,
        plots: bool = True,
        run_id: str | None = None,
        snapshot_limit: int | None = None,
        **perform_kwargs: Any,
    ) -> "BaseAnalyzer":
        """Run the full analysis sequence: load, decompose, save, plot.

        This is the single analysis sequence for both the library and the CLI.
        ``plots`` defaults to True (decision 2026-08-14); the CLI passes its
        config's generate_plots through explicitly. Method-specific keyword
        arguments go straight to this class's ``perform_*`` method.
        """
        display = display_name_for(getattr(self, "analysis_type", ""))
        logger.info("Starting %s analysis", display)
        start_time = time.time()
        self.load_and_preprocess()
        if snapshot_limit is not None:
            self._apply_snapshot_limit(snapshot_limit)
        if self._needs_fft_blocks:
            self.compute_fft_blocks()
        getattr(self, self._perform_name)(**perform_kwargs)
        self.save_results()
        if plots:
            self._plot_run(run_id=run_id)
        self._on_run_complete()
        logger.info(
            "%s analysis and plotting completed successfully in %.2f seconds.",
            display,
            time.time() - start_time,
        )
        return self

    def _apply_snapshot_limit(self, limit_value: int | str | None) -> None:
        """Optionally truncate the loaded snapshot matrix for heavy runs.

        Moved verbatim from the CLI layer now that the seam owns the sequence:
        same floor formula for Welch blocks, same guards.
        """
        if limit_value is None:
            return
        if "q" not in self.data:
            return
        limit = int(limit_value)
        q = self.data["q"]
        if limit < 2 or limit >= q.shape[0]:
            return
        self.data["q"] = q[:limit, :]
        self.data["Ns"] = limit
        if hasattr(self, "novlap") and hasattr(self, "nfft") and self.nfft > 1:
            # Same floor formula as BaseAnalyzer.load_and_preprocess (welch_nblocks).
            # The old ceil here overwrote a correct floor value and requested more
            # blocks than fit after truncation (e.g. Ns=400, nfft=128, ovl=0.5).
            n_snapshots = int(self.data["Ns"])
            nblocks = welch_nblocks(n_snapshots, self.nfft, self.novlap)
            if nblocks < 1:
                raise ValueError(
                    f"Cannot form Welch blocks: Ns={n_snapshots}, nfft={self.nfft} "
                    f"(novlap={self.novlap}) yield nblocks={nblocks}"
                )
            self.nblocks = nblocks

    def _plot_run(self, run_id: str | None = None) -> None:
        """Method-specific default figures after a completed run."""
        raise NotImplementedError(f"{type(self).__name__} must implement _plot_run")

    def _maybe_plot_volumetric_modes(
        self,
        *,
        plot_n_modes: int,
        slices_kwargs: dict[str, Any] | None = None,
        iso_kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Use analyzer-specific 3D plot hooks when volumetric data is present."""
        if int(self.data.get("Nz", 1)) <= 1:
            return False
        used = False
        if hasattr(self, "plot_modes_3d_slices"):
            kwargs = {"plot_n_modes": plot_n_modes}
            kwargs.update(slices_kwargs or {})
            self.plot_modes_3d_slices(**kwargs)
            used = True
        if hasattr(self, "plot_modes_3d_isometric"):
            kwargs = {"plot_n_modes": plot_n_modes}
            kwargs.update(iso_kwargs or {})
            self.plot_modes_3d_isometric(**kwargs)
            used = True
        return used

    def run(self, compute_fft: bool = True) -> BaseAnalyzer:
        """Prepare only: load and (optionally) build Welch blocks.

        This is not the analysis entry point — it performs no decomposition
        and saves nothing. The full sequence lives in :meth:`run_analysis`.
        """
        start_time = time.time()

        # Load data and calculate weights
        self.load_and_preprocess()

        # Compute FFT blocks if requested
        if compute_fft:
            self.compute_fft_blocks()

        end_time = time.time()
        logger.info("Completed in %.2f seconds.", end_time - start_time)

        return self

    def _on_run_complete(self) -> None:
        """Hook at the very end of run_analysis. Default: no-op.

        BSMD uses this to release disk-backed FFT resources.
        """

    def _resync_mode_count(self) -> None:
        """Lower ``n_modes_save`` to the solver's actual mode count and slice.
        The SVD/eigh solver may return fewer modes than the caller's cap after
        its relative cutoff. Keep the counter honest so save/plot paths never
        believe a wider array than exists. Slicing arrays already at that
        width is a no-op, so routes that truncate before calling this (ST-POD,
        multi-band mPOD) get the counter fixed and the arrays left alone.

        Call this only after the solver has assigned the three arrays. Mode
        count is read from ``eigenvalues`` alone, which is the solver's contract.
        Two-dimensional arrays are sliced to that width; a 1-D empty array
        (the rank-0 / pre-decomposition default) is left alone.
        """
        n_available_modes = len(self.eigenvalues)
        if self.n_modes_save > n_available_modes:
            logger.warning(
                "n_modes_save (%d) > available modes (%d). Using all available.",
                self.n_modes_save,
                n_available_modes,
            )
            self.n_modes_save = n_available_modes

        if self.modes.ndim == 2:
            self.modes = self.modes[:, : self.n_modes_save]
        self.eigenvalues = self.eigenvalues[: self.n_modes_save]
        if self.time_coefficients.ndim == 2:
            self.time_coefficients = self.time_coefficients[:, : self.n_modes_save]

    def _get_metadata(self) -> dict[str, Any]:
        """Return a dictionary of common metadata for saving results."""
        meta = {
            "analysis_type": getattr(self, "analysis_type", ""),
            # h5py cannot store None; an in-memory dataset has no file to name.
            "data_file": self.file_path if self.file_path is not None else "<in-memory data>",
            "nfft": self.nfft,
            "overlap": self.overlap,
            "nblocks": self.nblocks,
            "fs": self.fs,
            "dt": self.data.get("dt", 0),
            "Ns": self.data.get("Ns", 0),
            "Nx": self.data.get("Nx", 0),
            "Ny": self.data.get("Ny", 0),
            "Nz": self.data.get("Nz", 1),
            "spatial_weight_type": self.spatial_weight_type,
        }
        if "seed" in self.data:
            meta["data_seed"] = self.data["seed"]
        for attr in ["window_type", "window_norm", "blockwise_mean", "normvar"]:
            if hasattr(self, attr):
                meta[attr] = getattr(self, attr)
        meta.update(self._get_algorithm_metadata())
        return meta

    def _get_algorithm_metadata(self) -> dict:
        """Return analyzer-specific metadata describing the implemented contract."""
        return {}

    def release_memory(self) -> None:
        """Release large arrays to free memory."""
        attrs = [
            "data",
            "W",
            "qhat",
            "modes",
            "eigenvalues",
            "time_coefficients",
            "temporal_mean",
            "freq",
            "St",
            "amplitudes",
        ]
        for attr in attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    setattr(self, attr, np.array([]))
                elif isinstance(val, dict):
                    setattr(self, attr, {})
                else:
                    setattr(self, attr, None)
