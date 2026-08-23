"""Lift / metric / weighted second-order operator seam.

POD, mPOD and ST-POD are the same weighted second-order decomposition applied
to different lifts of the raw snapshot data. This module names those pieces:

- a **lift** (``.kind``, ``.apply``) that maps centered snapshots into the
  space where the second-order problem is posed;
- a **spatial metric** that defines the inner product, with ``.tile(d)`` for
  delay-embedded (lifted) spaces;
- **``weighted_second_order``**, the single solver both the eigh and SVD
  operator routes go through;
- **``spod_single_frequency``**, the single-frequency SPOD eigenproblem.

Both solver routes drop modes at or below a relative cutoff built from the
same scale ``n_kernel * eps``:

- **eigh** — eigenvalue domain: drop ``lambda <= n_kernel * eps * lambda_max``
  (numerical rank of the correlation matrix that was factored; its
  conditioning is already squared).
- **svd** — singular-value domain: drop
  ``sigma <= n_kernel * eps * sigma_max`` (equivalently
  ``lambda <= (n_kernel * eps)**2 * lambda_max``). Applying the eigh floor
  to ``lambda`` here would discard genuine weak modes the SVD route exists
  to recover.

``n_keep`` still truncates the leading end after that filter.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import scipy.linalg

from openmodalpy.core.base import (
    _coerce_spatial_weights,
    canonicalize_modes,
    compute_reduced_svd,
    require_spatial_metric,
)
from openmodalpy.core.threads import apply_blas_limit


@runtime_checkable
class Lift(Protocol):
    """A named transformation of centered snapshots into an analysis space."""

    kind: str

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Return the lifted matrix (samples × lifted features)."""
        ...


class IdentityLift:
    """Centered snapshots as-is — the POD lift."""

    kind = "identity_centered_snapshots"

    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)


class DelayEmbeddingLift:
    """Block-Hankel (delay) lift — the ST-POD lift.

    Each output row is a stack of ``embedding_dim`` consecutive snapshots, so
    the returned matrix has shape ``(m, d * Nspace)`` with
    ``m = Ns - d + 1``.
    """

    kind = "delay_embedding"

    def __init__(self, embedding_dim: int):
        if embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
        self.embedding_dim = int(embedding_dim)

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"DelayEmbeddingLift expects 2D data, got shape {data.shape}")
        n_snapshots, n_space = data.shape
        d = self.embedding_dim
        if d >= n_snapshots:
            raise ValueError(f"embedding_dim ({d}) must be < number of snapshots ({n_snapshots})")
        m = n_snapshots - d + 1
        # col_idx[k, j] = j + k → snapshot row for delay block k, column j
        col_idx = np.arange(m)[np.newaxis, :] + np.arange(d)[:, np.newaxis]
        # (d, m, Nspace) → (m, d, Nspace) → (m, d*Nspace)
        stacked = data[col_idx].transpose(1, 0, 2).reshape(m, d * n_space)
        return np.ascontiguousarray(stacked)


class BandFilteredLift:
    """Temporal band-pass lift — the mPOD lift for one frequency band.

    When constructed without band edges the object still carries the paper's
    ``kind`` string for metadata; call ``apply`` only after setting the band
    (via constructor args) so the FFT mask is well-defined.
    """

    kind = "multiscale_filtered_snapshots"

    def __init__(
        self,
        f_low: float | None = None,
        f_high: float | None = None,
        dt: float | None = None,
        *,
        is_last: bool = False,
    ):
        self.f_low = f_low
        self.f_high = f_high
        self.dt = dt
        self.is_last = bool(is_last)

    def mask(self, n_snapshots: int) -> np.ndarray:
        """Boolean rfft-bin mask for this band.

        Exposed so a caller can detect an empty band without duplicating the
        half-open/closed edge convention that ``apply`` uses.
        """
        if self.f_low is None or self.f_high is None or self.dt is None:
            raise ValueError("BandFilteredLift requires f_low, f_high and dt (pass them to the constructor).")
        freq = np.fft.rfftfreq(int(n_snapshots), d=float(self.dt))
        if self.is_last:
            return (freq >= self.f_low) & (freq <= self.f_high)
        return (freq >= self.f_low) & (freq < self.f_high)

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        n_snapshots = data.shape[0]
        mask = self.mask(n_snapshots)
        qhat = np.fft.rfft(data, axis=0)
        qhat_band = np.zeros_like(qhat)
        qhat_band[mask, :] = qhat[mask, :]
        return np.real(np.fft.irfft(qhat_band, n=n_snapshots, axis=0))


class SpatialMetric:
    """Diagonal spatial inner-product weights (the POD/SPOD metric W).

    Holds a diagonal metric as a 1-D vector. A full square matrix or a 3-D
    weight array is rejected rather than flattened — pass ``np.diag(W)`` when
    the metric is diagonal, or hand the raw array to the analyzer weight path
    (which reads the diagonal / stacks per-component diagonals).
    """

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights)
        # Shape before metric checks: square and 3-D are ambiguous for a
        # diagonal-only container. A complex square then reports the shape,
        # not the complex entries — the caller cannot act on the latter until
        # the shape is right.
        if w.ndim == 3:
            raise ValueError(
                f"SpatialMetric holds a diagonal metric as a vector and cannot "
                f"represent a 3-D weight array of shape {w.shape}. Stack "
                f"np.diag of each component plane into a 1-D vector, or pass "
                f"the raw array to the analyzer weight path which does that."
            )
        if w.ndim == 2 and w.shape[0] == w.shape[1] and w.shape[0] > 1:
            raise ValueError(
                f"SpatialMetric never accepts a square matrix, diagonal or not "
                f"(got shape {w.shape}). Pass the diagonal itself as a 1-D "
                f"vector: np.diag(W)."
            )
        # Validate on the raw input before any real cast — complex would
        # otherwise truncate under ComplexWarning and store only the real part.
        require_spatial_metric(w)
        self.weights = np.asarray(w, dtype=float).reshape(-1)

    def tile(self, d: int) -> np.ndarray:
        """Repeat the metric ``d`` times for a delay-embedded space (I_d ⊗ W)."""
        if d < 1:
            raise ValueError(f"tile count must be >= 1, got {d}")
        return np.tile(self.weights, int(d))


def _as_weight_vector(metric: SpatialMetric | np.ndarray, n_space: int) -> np.ndarray:
    if isinstance(metric, SpatialMetric):
        metric = metric.weights
    return _coerce_spatial_weights(metric, n_space)


def apply_sqrt_metric(data: np.ndarray, metric: SpatialMetric | np.ndarray) -> np.ndarray:
    """Row (samples) x column (features) data scaled by sqrt of the metric weights.

    This is the only place the sqrt(W) weighting of a samples x features matrix
    is applied. ``_solve_eigh``, ``_solve_svd`` and ST-POD's total energy call this, so they
    stay in exact agreement.
    """
    weights = _as_weight_vector(metric, data.shape[1])
    return data * np.sqrt(weights)


def weighted_total_energy(data: np.ndarray, metric: SpatialMetric | np.ndarray) -> float:
    """Pre-truncation total energy: ||sqrt(W)-weighted data||_F^2 / n_samples."""
    data_weighted = apply_sqrt_metric(data, metric)
    return float(np.linalg.norm(data_weighted, "fro") ** 2 / data.shape[0])


# Row-centeredness discriminator for the SVD route when ``n_keep is None``.
# Statistic: max|mean over axis 0| / std(data). Real ST-POD delay lifts bottom
# out near 2e-3 (~2000x above this). Centered data stays below it through
# offsets ~1e9 x the fluctuation; beyond that detection is lost (see
# ``_solve_svd``). The two mistakes are not symmetric: a false "centered" on a
# delay lift drops a real mode (unacceptable), a false "not centered" keeps
# today's behaviour. When unsure, do not declare centered.
CENTERED_ROW_MEAN_RATIO = 1e-6


def _row_mean_to_std_ratio(data: np.ndarray) -> float:
    """max|column-mean| / std(data); infinite when the array has no spread.

    A constant array has no spread, so the ratio is mathematically infinite, not
    zero: every column mean equals the constant. Reporting zero would call it
    centered and tighten the cap, which costs a single-sample constant array its
    only mode. Infinity keeps it on the safe side of the asymmetry.
    """
    spread = float(np.std(data))
    if spread == 0.0:
        return float("inf")
    return float(np.max(np.abs(np.mean(data, axis=0))) / spread)


def _measures_as_centered(data: np.ndarray) -> bool:
    """True only when the row-mean / std statistic is clearly below threshold."""
    return _row_mean_to_std_ratio(data) < CENTERED_ROW_MEAN_RATIO


def _unweight_modes(weighted_modes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Recover physical modes from sqrt(W)-weighted ones.

    Division is taken only where ``weights > 0``; a zero-measure cell gets
    exactly 0 (no NaN/Inf from 0/0).
    """
    modes = np.zeros_like(weighted_modes)
    positive = weights > 0
    if np.any(positive):
        modes[positive] = weighted_modes[positive] / np.sqrt(weights[positive])[:, np.newaxis]
    return modes


def _working_eps(dtype: np.dtype | type) -> float:
    """Machine epsilon of a real working dtype (float64 fallback)."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return float(np.finfo(dt).eps)
    return float(np.finfo(np.float64).eps)


def _relative_floor(peak: float, n_kernel: int, dtype: np.dtype | type) -> float:
    """Shared relative floor scale: ``n_kernel * eps * peak``.

    Both route masks use this quantity. The eigh route applies it to
    eigenvalues; the SVD route applies it to singular values (see the
    two mask helpers). Never paste a second constant at a call site.
    """
    return float(n_kernel) * _working_eps(dtype) * float(peak)


def _significant_eigenvalue_mask(
    eigenvalues: np.ndarray,
    n_kernel: int,
) -> np.ndarray:
    """Keep eigenvalues above ``n_kernel * eps * lambda_max``.

    **Eigenvalue-domain** floor for the eigh route. This is the numerical
    rank of the correlation (Gram) matrix that was factored, not of the
    snapshot data. ``n_kernel`` is the dimension of that matrix
    (``n_samples`` on the temporal branch, ``n_space`` on the spatial
    branch). ``eps`` is machine epsilon of the working real dtype.

    The Gram matrix already has squared conditioning, so this floor lives
    in the eigenvalue domain. The SVD route must not reuse it on
    ``lambda = sigma**2 / n_samples`` — use
    :func:`_significant_singular_value_mask` instead.
    """
    real = np.asarray(np.real(eigenvalues))
    if real.size == 0:
        return np.zeros(0, dtype=bool)
    lam_max = float(np.max(real))
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return np.zeros(real.shape, dtype=bool)
    cutoff = _relative_floor(lam_max, n_kernel, real.dtype)
    return real > cutoff


def _significant_singular_value_mask(
    singular_values: np.ndarray,
    n_kernel: int,
) -> np.ndarray:
    """Keep singular values above ``n_kernel * eps * sigma_max``.

    **Singular-value-domain** floor for the SVD route. ``n_kernel`` has the
    same meaning as on the eigh path (dimension of the Gram matrix that
    *would* have been factored: ``min(n_samples, n_space)``). The relative
    scale ``n_kernel * eps`` is shared with
    :func:`_significant_eigenvalue_mask` via :func:`_relative_floor`; only
    the domain differs.

    Equivalently for ``lambda = sigma**2 / n_samples`` the floor is
    ``(n_kernel * eps)**2 * lambda_max``. A mode at singular-value ratio
    ``1e-10`` sits at eigenvalue ratio ``1e-20``: below the eigh floor
    (``n_kernel * eps``) but far above this squared one, so the SVD route
    keeps it.
    """
    sigma = np.asarray(np.real(singular_values))
    if sigma.size == 0:
        return np.zeros(0, dtype=bool)
    sig_max = float(np.max(sigma))
    if not np.isfinite(sig_max) or sig_max <= 0.0:
        return np.zeros(sigma.shape, dtype=bool)
    cutoff = _relative_floor(sig_max, n_kernel, sigma.dtype)
    return sigma > cutoff


def weighted_second_order(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    method: Literal["eigh", "svd"] = "eigh",
    n_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the weighted second-order problem on lifted data.

    Parameters
    ----------
    data
        Lifted snapshot matrix, shape ``(n_samples, n_space)``. Samples run
        along axis 0 (time, Fourier realizations, or Hankel columns).
    metric
        Spatial (or already-tiled lifted) weights as a ``SpatialMetric`` or
        a 1D array of length ``n_space``.
    method
        ``"eigh"`` — covariance / Gram kernel eigenproblem (POD, mPOD,
        PSD-POD). ``"svd"`` — weighted SVD of the data matrix (ST-POD); use
        this rather than squaring a Hankel matrix.
    n_keep
        If set, keep only the leading ``n_keep`` modes (POD / mPOD /
        ST-POD / PSD-POD). The eigh route computes the full spectrum
        and truncates afterwards; the svd route genuinely solves for
        ``k`` only. Callers that need the true total energy must
        compute it from the Frobenius identity themselves.

    Returns
    -------
    modes, eigenvalues, time_coefficients
        ``modes`` has shape ``(n_space, r)``, ``eigenvalues`` shape ``(r,)``,
        ``time_coefficients`` shape ``(n_samples, r)``. Rank-deficient input
        returns fewer than ``n_keep`` / ``n_modes_save`` modes — the honest
        count above the route's relative cutoff. Both routes always drop
        modes at or below their relative cutoff (eigenvalue domain on eigh,
        singular-value domain on svd).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"weighted_second_order expects 2D data, got shape {data.shape}")
    with apply_blas_limit():
        if method == "svd":
            return _solve_svd(data, metric, n_keep=n_keep)
        if method == "eigh":
            return _solve_eigh(
                data,
                metric,
                n_keep=n_keep,
            )
    raise ValueError(f"method must be 'eigh' or 'svd', got {method!r}")


def _solve_eigh(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_space = data.shape
    weights = _as_weight_vector(metric, n_space)
    data_weighted = apply_sqrt_metric(data, metric)

    # Complex ensembles (PSD-POD Fourier realizations) use the Hermitian
    # temporal kernel and the reconstruction that path has always used.
    if np.iscomplexobj(data):
        return _solve_eigh_complex(
            data,
            data_weighted,
            weights,
            n_samples,
            n_keep=n_keep,
        )

    use_temporal = n_samples < n_space
    if use_temporal:
        n_kernel = n_samples
        kernel = np.dot(data_weighted, data_weighted.T) / n_samples
        eigenvalues, vectors = scipy.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        vectors = vectors[:, order]
        # Recompute each eigenvalue as the Rayleigh quotient v.K.v. The
        # quotient is stationary at an eigenvector, so a first-order error in
        # the vector costs only a second-order error in the value: it is a
        # more accurate eigenvalue than the one LAPACK returns, which is what
        # makes the rank test below trustworthy for a value near the cutoff.
        # On well-conditioned data it agrees with LAPACK to the last bit; it can
        # differ by about 1e-15 relative when the kernel is poorly conditioned.
        # The spatial branch below and the complex path do the same thing.
        eigenvalues = np.sum(vectors * (kernel @ vectors), axis=0)

        keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
        eigenvalues = eigenvalues[keep]
        vectors = vectors[:, keep]
        if eigenvalues.size == 0:
            return (
                np.empty((n_space, 0)),
                np.empty((0,)),
                np.empty((n_samples, 0)),
            )

        if n_keep is not None:
            take = min(int(n_keep), eigenvalues.size)
            eigenvalues = eigenvalues[:take]
            vectors = vectors[:, :take]

        # After the relative filter every eigenvalue is strictly positive.
        safe = eigenvalues * n_samples
        normalization = 1.0 / np.sqrt(safe)
        weighted_modes = np.dot(data_weighted.T, vectors) * normalization
        modes = _unweight_modes(weighted_modes, weights)
        coeffs = vectors * np.sqrt(safe)
    else:
        n_kernel = n_space
        kernel = np.dot(data_weighted.T, data_weighted) / n_samples
        eigenvalues, weighted_modes = scipy.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        weighted_modes = weighted_modes[:, order]
        eigenvalues = np.sum(weighted_modes * (kernel @ weighted_modes), axis=0)

        keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
        eigenvalues = eigenvalues[keep]
        weighted_modes = weighted_modes[:, keep]
        if eigenvalues.size == 0:
            return (
                np.empty((n_space, 0)),
                np.empty((0,)),
                np.empty((n_samples, 0)),
            )

        if n_keep is not None:
            take = min(int(n_keep), eigenvalues.size)
            eigenvalues = eigenvalues[:take]
            weighted_modes = weighted_modes[:, :take]

        modes = _unweight_modes(weighted_modes, weights)
        coeffs = np.dot(data_weighted, weighted_modes)

    modes, coeffs = canonicalize_modes(np.real(modes), np.real(coeffs))
    return modes, np.real(eigenvalues), coeffs


def _solve_eigh_complex(
    data: np.ndarray,
    data_weighted: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermitian temporal-kernel path used by PSD-POD Fourier ensembles."""
    n_space = data.shape[1]
    n_kernel = n_samples
    kernel = (data_weighted @ data_weighted.conj().T) / n_samples
    # eigh on a Hermitian Gram matrix — same contract as the real POD path.
    eigenvalues, eigenvectors = scipy.linalg.eigh(kernel)
    order = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.sum(np.conj(eigenvectors) * (kernel @ eigenvectors), axis=0).real

    keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    if eigenvalues.size == 0:
        return (
            np.empty((n_space, 0), dtype=data.dtype),
            np.empty((0,)),
            np.empty((n_samples, 0), dtype=data.dtype),
        )

    if n_keep is not None:
        take = min(int(n_keep), eigenvalues.size)
        eigenvalues = eigenvalues[:take]
        eigenvectors = eigenvectors[:, :take]

    eigenvalues = np.real_if_close(eigenvalues)
    safe_eigs = np.real(eigenvalues)
    # Build from the sqrt(W)-weighted ensemble, then unweight — same policy as
    # the real eigh/svd routes. Where w > 0 the sqrt(w) cancels and the mode
    # matches the historical unweighted formula; where w == 0 the mode is 0.
    weighted_modes = (data_weighted.conj().T @ eigenvectors) / np.sqrt(safe_eigs * n_samples)
    modes = _unweight_modes(weighted_modes, weights)
    coeffs = data.conj() @ (weights[:, np.newaxis] * modes)
    modes, coeffs = canonicalize_modes(modes, coeffs)
    return modes, np.asarray(eigenvalues), coeffs


def _solve_svd(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted SVD route (ST-POD). ``data`` is samples × features.

    Caps the mode count at the matrix bound ``min(n_samples, n_space)`` when
    the input does not measure as row-centered. A delay-embedded window of a
    zero-mean series is not itself zero-mean, so the all-ones left-null that
    centering would create is not present and the full matrix bound applies.

    When ``n_keep is None``, centeredness is *measured* on the input via
    :data:`CENTERED_ROW_MEAN_RATIO` (max |column-mean| / std). If the input
    measures as centered, the cap tightens to ``min(n_samples - 1, n_space)``
    because row-centering nulls one direction. Callers that pass an explicit
    ``n_keep`` have declared their own bound; that path is untouched.

    Detection ceiling near offset 1e9. The statistic separates centered data
    from genuine delay lifts up to a mean offset of roughly 1e9 times the
    fluctuation. Beyond that the residual null no longer measures as centered
    and the route falls back to today's behaviour (keep the extra mode). At
    those offsets centering has already destroyed most significant digits, so
    the decomposition is meaningless regardless of the mode count — the fix is
    not universal past that ceiling. The asymmetric risk is intentional: a
    false "centered" on a delay lift would drop a real mode; a false "not
    centered" only preserves the prior junk-mode behaviour.
    """
    n_samples, n_space = data.shape
    weights = _as_weight_vector(metric, n_space)
    data_weighted = apply_sqrt_metric(data, metric)

    # n_kernel scales the singular-value floor: dimension of the Gram that
    # would have been factored (temporal if n_samples < n_space, else spatial).
    # Distinct from the mode-count cap below — do not reuse one for the other.
    n_kernel = min(n_samples, n_space)
    # Honest matrix rank bound. ST-POD's lifted matrix is full row rank. When
    # the caller leaves n_keep unset and the input measures as row-centered,
    # tighten by one (centering nulls a direction). Explicit n_keep is the
    # caller's bound and is never rewritten here.
    max_rank = max(min(n_samples, n_space), 0)
    if n_keep is None:
        if _measures_as_centered(data):
            max_rank = max(min(n_samples - 1, n_space), 0)
        k = max_rank
    else:
        k = min(int(n_keep), max_rank)
    if k < 1:
        return (
            np.empty((n_space, 0)),
            np.empty((0,)),
            np.empty((n_samples, 0)),
        )

    # SVD of the feature × sample matrix so left singular vectors are modes.
    u_full, sigma_full, vt_full = compute_reduced_svd(data_weighted.T, k)
    sigma = sigma_full[:k]
    u = u_full[:, :k]
    vt = vt_full[:k, :]

    # Singular-value-domain relative floor (not the eigh eigenvalue floor).
    keep = _significant_singular_value_mask(sigma, n_kernel)
    sigma = sigma[keep]
    u = u[:, keep]
    vt = vt[keep, :]
    if sigma.size == 0:
        return (
            np.empty((n_space, 0)),
            np.empty((0,)),
            np.empty((n_samples, 0)),
        )

    eigenvalues = (sigma**2) / n_samples
    modes = _unweight_modes(u, weights)
    coeffs = (vt * sigma[:, np.newaxis]).T
    modes, coeffs = canonicalize_modes(np.real(modes), np.real(coeffs))
    return modes, np.real(eigenvalues), coeffs


def spod_single_frequency(
    qhat: np.ndarray,
    nblocks: int,
    dst: float,
    w: np.ndarray,
    *,
    num_modes: int | None = None,
    return_psi: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPOD eigenproblem at one frequency (shared serial / parallel body).

    ``qhat`` is ``(n_space, n_blocks)``; ``dst`` is the spectral weight in
    ``1/sqrt(nblocks * dst)``. Optional ``num_modes`` truncates after sorting;
    ``return_psi`` also returns the block-space eigenvectors.
    """
    with apply_blas_limit():
        x = qhat / np.sqrt(nblocks * dst)
        w_col = _coerce_spatial_weights(w, qhat.shape[0]).reshape(-1, 1)
        xprime_w = np.conj(x).T * w_col.T  # X_f^H * W
        m = xprime_w @ x
        lambda_tilde, psi = np.linalg.eigh(m)
        idx = lambda_tilde.argsort()[::-1]
        lambda_tilde = lambda_tilde[idx]
        psi = psi[:, idx]
        if num_modes is not None:
            keep = min(int(num_modes), len(lambda_tilde))
            lambda_tilde = lambda_tilde[:keep]
            psi = psi[:, :keep]
        inv_sqrt_lambda = np.zeros_like(lambda_tilde)
        # n_kernel is the block dimension of the CSD matrix that was factored.
        mask = _significant_eigenvalue_mask(lambda_tilde, nblocks)
        inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
        phi = x @ (psi * inv_sqrt_lambda[np.newaxis, :])
        # Same unit factor on phi and psi so psi = X^H W phi Lambda^{-1/2} holds
        # after the LAPACK phase is fixed (return_psi True or False → same phi).
        phi, psi = canonicalize_modes(phi, psi)
        # Gram is PSD: a negative eigenvalue is roundoff. Clamp to zero —
        # abs() would report it as real energy with the wrong sign flipped.
        lambda_out = np.maximum(lambda_tilde, 0.0)
        if return_psi:
            return phi, lambda_out, psi
        return phi, lambda_out
