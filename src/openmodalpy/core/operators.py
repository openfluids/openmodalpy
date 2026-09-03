"""The shared numerical steps every method builds on.

The reduced singular value decomposition and its routing live here, with the
block transform and the rules that give modes one canonical sign and order.
These steps hold no state and know nothing about the analyzers that call them.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.linalg import svds

from openmodalpy.core.threads import apply_blas_limit
from openmodalpy.core.welch import windowed_block_fft

# Route iterative SVD (ARPACK via ``svds``) only when rank is a small fraction
# of the smaller dimension *and* that dimension is large enough for it to pay.
# Values from machine measurements (see ``use_iterative_svd`` docstring).
ARPACK_MAX_RANK_FRACTION = 0.05
ARPACK_MIN_DIM = 256

# Relative band used when (1) choosing the pivot index for mode sign/phase and
# (2) grouping |λ| ties for canonical DMD spectrum order. Sits above typical
# cross-build eigenvector noise (~1e-14 to 1e-15 relative) and far below any
# physically meaningful difference between two peaks. Moves the sign
# discontinuity out of the last-bit regime; perfect uniqueness for exactly
# degenerate peaks is impossible (phi and -phi are both valid), so the band
# relocates the ambiguity rather than removing it.
CANONICAL_TIE_RTOL = 1e-12


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
