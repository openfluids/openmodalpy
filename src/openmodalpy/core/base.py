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
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import h5py
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.linalg import svds

if TYPE_CHECKING:
    pass

from openmodalpy.core.config import (
    FFT_BACKEND,
)
from openmodalpy.core.io import derive_grid_and_snapshot_counts
from openmodalpy.core.io import load_data as di_load_data
from openmodalpy.core.io import load_jetles_data as di_load_jetles_data
from openmodalpy.core.io import load_mat_data as di_load_mat_data
from openmodalpy.core.threads import apply_blas_limit
from openmodalpy.core.weights import (
    _as_spatial_weight_column,
    calculate_cell_volume_weights,
    calculate_polar_weights,
    calculate_uniform_weights,
)
from openmodalpy.core.welch import sine_window as sine_window  # re-export; body in welch.py
from openmodalpy.core.welch import welch_nblocks, windowed_block_fft

# Welch block FFT and helpers live in welch.py (single implementation) so base
# remains importable when the parallel stack fails to load.
from openmodalpy.specs import display_name_for

try:
    from openmodalpy.core.parallel import get_threadpool_summary
except ImportError:
    pass

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
            # ARPACK does one matrix-vector product per iteration, so it reads
            # the whole matrix tens of times. A non-contiguous view makes every
            # one of those reads stride through memory, and the caller usually
            # hands one over: DMD passes X[:, :-1], which drops the last column
            # and leaves a view. One copy costs far less than the strided reads.
            # Measured on a delay-embedded double gyre, X[:, :-1] at
            # (80000, 396), rank 10, 1 BLAS thread: 2.792 s as a view against
            # 0.045 s to copy plus 0.338 s to solve.
            X = np.ascontiguousarray(X)
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


def canonical_tie_groups(magnitudes: ArrayLike) -> list[np.ndarray]:
    """Split indices into |value|-descending runs that tie under ``CANONICAL_TIE_RTOL``.

    A group is every run of the descending order whose magnitude agrees with
    the group's first (largest) member within ``CANONICAL_TIE_RTOL``, never
    merely with its neighbour. Each returned array holds the original indices
    of one group, in magnitude-descending order; groups are returned in
    descending order too. Empty input yields an empty list.
    """
    mag = np.abs(np.asarray(magnitudes)).astype(float, copy=False).reshape(-1)
    n = int(mag.size)
    if n == 0:
        return []

    # Stable so exact-tie group membership does not depend on the platform sort.
    order = np.argsort(-mag, kind="stable")
    groups: list[np.ndarray] = []
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
        groups.append(order[i:j])
        i = j
    return groups


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

    out = np.empty(n, dtype=int)
    i = 0
    for group in canonical_tie_groups(eigvals):
        group_eigs = eigvals[group]
        tie_key = np.lexsort((np.imag(group_eigs), np.real(group_eigs)))
        j = i + group.size
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


def validate_nfft_overlap(nfft: int, overlap: float) -> None:
    """Check that ``nfft`` is positive and ``overlap`` is in [0, 1).

    Shared by every analyzer that forms Welch FFT blocks (SPOD, BSMD),
    so the same input raises the same message everywhere.

    Args:
        nfft (int): Number of points per FFT block.
        overlap (float): Overlap fraction between blocks.

    Raises:
        ValueError: If ``overlap`` is not in [0, 1) or ``nfft`` is not positive.
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")
    if nfft <= 0:
        raise ValueError("NFFT must be positive.")


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
    """Log a short summary of where results and figures were saved.

    No longer called by ``run_analysis``; kept only for compatibility.
    """
    logger.info("%s analysis finished", analysis)
    logger.info("Results: %s", results_dir)
    logger.info("Figures: %s", figures_dir)


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
    - SPOD callers pass ``dst`` into ``decomposition.spod_single_frequency`` as a
      spectral weight. In this
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


def _fill_contract_counts(data: dict[str, Any], *, source: str) -> None:
    """Check the required contract keys, then fill the derived counts.

    This is the one rule for both ways a caller supplies data: a dict given to
    ``data=``, and the dict a ``data_loader=`` callable returns. Both must
    behave the same, because DOC.md documents them as the same plug-in point.

    ``q``, ``x`` and ``y`` are required, because the shapes come from them.
    ``dt`` is not checked here: ``_require_dt`` owns it, and says which dataset
    is short of one. ``Nx``, ``Ny``, ``Nz`` and ``Ns`` come from the array
    shapes when they are absent. A dict that already states a count keeps it.

    Parameters
    ----------
    data : dict[str, Any]
        The dataset. This function adds the missing counts to it.
    source : str
        Where the dict came from. Used for the error text only.

    Raises
    ------
    ValueError
        If a required key is absent. The message names every missing key.
    """
    required = ("q", "x", "y")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            f"data is missing required key(s): {', '.join(missing)}. "
            f"Required: q, x, y, dt. Derived when absent: Nx, Ny, Nz, Ns."
        )
    derived_keys = ("Nx", "Ny", "Nz", "Ns")
    if all(key in data for key in derived_keys):
        # Nothing to derive. Do not inspect the shapes: the checks in
        # load_and_preprocess own every mismatch message, and they say more
        # than this function can, naming the product and q.shape[1].
        return

    # Derive from the shapes alone: pass no stated counts. A count the caller
    # already stated stays untouched below.
    z_raw = data.get("z")
    nx, ny, nz, ns = derive_grid_and_snapshot_counts(
        np.asarray(data["q"]),
        np.asarray(data["x"]),
        np.asarray(data["y"]),
        np.asarray(z_raw) if z_raw is not None else None,
        {},
        source=source,
    )
    data.setdefault("Nx", nx)
    data.setdefault("Ny", ny)
    data.setdefault("Nz", nz)
    data.setdefault("Ns", ns)


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
        *,
        results_dir: str = "./preprocess",
        figures_dir: str = "./figs",
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the analyzer.

        Welch block parameters (``nfft``, ``overlap``) and ``use_parallel``
        are not generic: only SPOD, PSD-POD and BSMD form FFT blocks, and
        only BSMD runs work in parallel threads. Each of those three sets
        its own ``nfft``/``overlap`` attributes after calling this
        constructor; BSMD also keeps its own ``use_parallel`` attribute.
        Every other analyzer gets the dummy values below, which keep the
        shared filename and metadata helpers working without claiming a
        block size that was never used.

        Args:
            file_path (str | None): Path to data file. Optional when ``data``
                carries the loaded dataset instead.
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
                contract (``q``, ``x``, ``y``, ``dt`` required; ``Nx``,
                ``Ny``, ``Nz``, ``Ns`` derived when absent; see DOC.md).
                Given instead of ``file_path`` — exactly one of the two is
                required. The dict is stored by reference, so one load can
                feed several analyzers. The derived counts are written back
                into the given dict, so a second analyzer re-uses them
                instead of deriving them again.
        """
        self.file_path = file_path

        # Exactly one input source. ``data`` is the documented in-memory path;
        # assigning ``.data`` after construction stays as the legacy escape hatch.
        if data is not None:
            if file_path is not None:
                raise ValueError("Pass file_path or data, not both: an analyzer takes exactly one input source.")
            if not isinstance(data, dict) or not data:
                raise ValueError(
                    "data must be a non-empty dict following the data contract "
                    "(q, x, y, dt required; Nx, Ny, Nz, Ns derived when absent; see DOC.md)."
                )
            _fill_contract_counts(data, source="data")
        elif file_path is None:
            raise ValueError("No input source: pass file_path (path to a data file) or data (the loaded dict).")
        # Dummy Welch stamp for the six methods that never form FFT blocks:
        # SPOD, PSD-POD and BSMD overwrite these with their own nfft/overlap
        # right after this constructor returns.
        self.nfft = 1
        self.overlap = 0.0
        self.results_dir = results_dir
        self.figures_dir = figures_dir

        # Set default data loader based on file type
        self.data_loader = data_loader or load_data

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
        self.novlap = 0

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

    def load_and_preprocess(self) -> None:
        """Load data and calculate weights."""
        # Load data from file only if not already provided. The constructor
        # guarantees a non-empty dict whenever ``data=`` was given, so an empty
        # dict here can only come from a legacy side-channel assignment, which
        # keeps its old reload semantics.
        if not self.data:
            if self.file_path is None:
                raise ValueError("no file_path and no data were given")
            self.data = self.data_loader(self.file_path)
            # A custom (path) -> dict loader is the documented plug-in point,
            # so it must give the same result as a dict passed to data=.
            # Without this, a loader that returns only q, x, y and dt failed
            # later with a bare KeyError on Ns.
            _fill_contract_counts(self.data, source="data_loader")

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
                    # Weight computation is not the parallel path this bead
                    # touches (BSMD's use_parallel); keep the old default.
                    use_parallel=True,
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

        # fs is generic (every method may report a sampling rate); Welch block
        # counting is not — only SPOD, PSD-POD and BSMD form FFT blocks, and
        # each does so lazily in its own compute_fft_blocks() call.
        self.fs = 1 / self._require_dt()

        logger.info(
            "Data loaded: %s snapshots, %s×%s spatial points",
            self.data["Ns"],
            self.data.get("Nx"),
            self.data.get("Ny"),
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

    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return datasets and metadata to save.

        Returns a tuple of (datasets dict, attributes dict). Subclasses
        override to specify their result structure. The base implementation
        is a no-op stub; all seven analyzers and toy implement this.

        Datasets should be canonical names (lowercase); write_results
        handles None values. Attributes are arbitrary metadata
        (provenance is added automatically by write_results).
        """
        return {}, {}

    def _required_result_fields(self) -> tuple[str, ...]:
        """Name the datasets a real result file of this method must hold.

        The loader raises when one is absent, which keeps a wrong file loud:
        assigning only what is present would give empty arrays and a cheerful
        "results loaded" line. The default is empty, because some methods
        accept a partial file on purpose. BSMD reads a file with no ``modes1``
        to reach the weights, and DMD lowers its mode cap from an empty file.
        A subclass that wants the loud failure names its datasets here.
        """
        return ()

    def _result_filename(self) -> str:
        """Return the name of this method's result file.

        The base name holds the Welch block size and the overlap. A method
        that has no blocks gives a shorter name here. Save and load both
        call this method, so a method cannot write a file it cannot find.
        """
        return make_result_filename(
            self.data_root,
            self.nfft,
            self.overlap,
            self.data.get("Ns", 0),
            getattr(self, "analysis_type", "spod"),
        )

    def save_results(self, filename: str | None = None) -> None:
        """Save results to HDF5 using the unified writer.

        Calls _result_payload() to get the datasets and metadata,
        ensures results_dir exists, and writes through write_results.

        Parameters
        ----------
        filename
            Custom HDF5 filename. If None, uses harmonized scheme
            with data_root, nfft, overlap, analysis_type.
        """
        from openmodalpy.core.results import write_results

        if not filename:
            filename = self._result_filename()
        save_path = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        datasets, attrs = self._result_payload()
        name = display_name_for(getattr(self, "analysis_type", "spod"))
        own_logger = logging.getLogger(type(self).__module__)
        own_logger.info("Saving %s results to %s", name, save_path)
        write_results(save_path, datasets, attrs=attrs)
        self._after_write(save_path)
        own_logger.info("%s results saved to %s", name, save_path)

    def load_results(self, filename: str | None = None) -> None:
        """Load results from HDF5 using the unified reader.

        Reads the file through read_results and assigns arrays to instance.
        Subclasses may override to handle special post-processing.

        Parameters
        ----------
        filename
            Custom HDF5 filename. If None, uses the same harmonized scheme
            as save_results.
        """
        from openmodalpy.core.results import read_results

        if not filename:
            filename = self._result_filename()
        load_path = os.path.join(self.results_dir, filename)
        name = display_name_for(getattr(self, "analysis_type", "spod"))
        own_logger = logging.getLogger(type(self).__module__)
        own_logger.info("Loading %s results from %s", name, load_path)

        if not os.path.isfile(load_path):
            from openmodalpy.core.results import find_latest_result

            latest = find_latest_result(self.results_dir, f"*_{getattr(self, 'analysis_type', 'spod')}.hdf5")
            if latest:
                load_path = latest
                logger.info("[Auto-detect] Using available results file: %s", load_path)
            else:
                logger.error(
                    "No results file found in %s matching '*_%s.hdf5'. Run the analysis or call save_results first.",
                    self.results_dir,
                    getattr(self, "analysis_type", "spod"),
                )
                return

        res = read_results(load_path)
        missing = [field for field in self._required_result_fields() if getattr(res, field, None) is None]
        if missing:
            raise KeyError(f"{load_path} is not a {name} result file: missing {', '.join(missing)}")
        self._assign_loaded_results(res)
        own_logger.info("%s results loaded.", name)

    def _assign_loaded_results(self, res: Any) -> None:
        """Assign loaded AnalysisResults to instance.

        Default base implementation assigns standard fields (modes,
        eigenvalues, time_coefficients, coordinates, weights, etc.).
        Subclasses override to handle non-standard fields or post-processing.

        Parameters
        ----------
        res
            AnalysisResults object from read_results.
        """
        # Standard fields available in AnalysisResults.
        if res.modes is not None:
            self.modes = res.modes
        if res.eigenvalues is not None:
            self.eigenvalues = res.eigenvalues
        if res.time_coefficients is not None:
            self.time_coefficients = res.time_coefficients
        if res.W is not None:
            self.W = res.W
        if res.temporal_mean is not None:
            self.temporal_mean = res.temporal_mean

        # Coordinates and metadata.
        for coord_key in ("x", "y", "z"):
            value = getattr(res, coord_key, None)
            if value is not None:
                self.data[coord_key] = value
            elif coord_key in res.extra:
                self.data[coord_key] = res.extra[coord_key]

        # Metadata attributes.
        if "dt" in res.attrs:
            self.data["dt"] = res.attrs["dt"]
        if "Ns" in res.attrs:
            self.data["Ns"] = res.attrs["Ns"]
        if "Nspace" in res.attrs:
            self.data["Nspace"] = res.attrs["Nspace"]
        if "Nx" in res.attrs:
            self.data["Nx"] = int(res.attrs["Nx"])
        if "Ny" in res.attrs:
            self.data["Ny"] = int(res.attrs["Ny"])
        if "Nz" in res.attrs:
            self.data["Nz"] = int(res.attrs["Nz"])

    def _after_write(self, save_path: str) -> None:
        """Do extra work on the result file after the writer closes it.

        The default does nothing. SPOD writes its FFT cache stamp here, so
        that a later run can reuse the blocks instead of computing them again.
        """

    def _ensure_figures_dir_exists(self) -> None:
        """Create figures_dir if it does not exist (first plot write)."""
        os.makedirs(self.figures_dir, exist_ok=True)

    # Analysis-sequence seam
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
        if plots:
            logger.info(
                "%s analysis and plotting completed successfully in %.2f seconds.",
                display,
                time.time() - start_time,
            )
        else:
            logger.info(
                "%s analysis completed successfully in %.2f seconds (no figures written: plots=False).",
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
