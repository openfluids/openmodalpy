#!/usr/bin/env python3
"""
Dynamic Mode Decomposition (exact DMD) implementation.

The current analyzer implements standard exact DMD on raw shifted snapshots
with Euclidean least squares. It does not currently apply the spatial metric
``W`` or mean subtraction inside ``perform_dmd()``.
"""

from __future__ import annotations

# Standard library imports
import logging
import os

import matplotlib

matplotlib.use("Agg")
import inspect
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt

from openmodalpy.core.results import AnalysisResults

logger = logging.getLogger(__name__)

# Suppress contour warnings when no levels can be plotted
warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
import numpy as np  # noqa: E402
from numpy.typing import ArrayLike, DTypeLike  # noqa: E402

from openmodalpy.core.base import (  # noqa: E402
    BaseAnalyzer,
    add_inset_colorbar,
    canonical_eigenvalue_order,
    compute_reduced_svd,
    format_mode_title,
    get_fig_aspect_ratio,
    make_result_filename,
    plot_modes_3d,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.config import (  # noqa: E402
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_DMD,
    RESULTS_DIR_DMD,
)
from openmodalpy.core.threads import apply_blas_limit  # noqa: E402


def _delay_embed(X: np.ndarray, d: int) -> np.ndarray:
    """Build delay-embedded (Hankel) matrix from snapshot matrix X.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Snapshot matrix with *n* spatial points and *m* time steps.
    d : int
        Stack depth for delay embedding.  ``d=1`` returns *X* unchanged.

    Returns
    -------
    ndarray, shape (n*d, m-d+1)
    """
    n, m = X.shape
    cols = m - d + 1
    out = np.empty((n * d, cols), dtype=X.dtype)
    for i in range(d):
        out[i * n : (i + 1) * n, :] = X[:, i : i + cols]
    return out


def _dmd_pinv_rcond(shape: Sequence[int], dtype: DTypeLike) -> float:
    """Return ``max(shape) * finfo(dtype).eps`` (numpy.linalg.pinv default).

    Used as the relative singular-value floor: keep ``s_j > rcond * s[0]``.
    Not a public constructor parameter.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        dt = np.dtype(dt.type(0).real.dtype)
    elif not np.issubdtype(dt, np.floating):
        dt = np.dtype(np.float64)
    return max(shape) * np.finfo(dt).eps


# Relative SVD / pinv cutoff policy: numpy.linalg.pinv rcond convention.
# rcond = max(M, N) * finfo(dtype).eps; rank keeps s_j > rcond * s[0].
# Shape/dtype-dependent — computed per call via _dmd_pinv_rcond. Not a user knob.
DMD_PINV_RCOND = _dmd_pinv_rcond


def svht_lambda(beta: float) -> float:
    """Gavish–Donoho (2014) optimal hard-threshold coefficient (known noise).

    For a *known* noise level ``sigma``, the hard threshold is
    ``tau = svht_lambda(beta) * sigma * sqrt(max(shape))`` — the square-root
    factor is part of the known-noise form and is easy to drop by mistake.
    When ``sigma`` is unknown and
    estimated by the median singular value, use :func:`svht_omega` instead —
    that is the coefficient used by the ``rank="svht"`` path.

    Parameters
    ----------
    beta : float
        Aspect ratio ``min(shape) / max(shape)`` of the snapshot-pair matrix,
        in ``(0, 1]``.

    Returns
    -------
    float
        ``lambda(beta)`` such that the hard threshold is
        ``tau = lambda(beta) * sigma * sqrt(max(shape))`` for known noise
        standard deviation ``sigma``.

    Notes
    -----
    ``lambda(1) = 4/sqrt(3) ≈ 2.309401`` is the square-matrix value, not a
    universal constant. As ``beta -> 0``, ``lambda -> sqrt(2)``.
    """
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"svht_lambda requires 0 < beta <= 1, got {beta!r}")
    return float(np.sqrt(2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + np.sqrt(beta * beta + 14.0 * beta + 1.0))))


def _mp_median(beta: float) -> float:
    """Median of the Marchenko–Pastur law with aspect ratio ``beta`` in (0, 1].

    Density ``sqrt((b-x)(x-a)) / (2*pi*beta*x)`` on ``[a, b]`` with
    ``a = (1-sqrt(beta))**2``, ``b = (1+sqrt(beta))**2``.  The endpoint
    square-root singularities are removed by the substitution
    ``x = a + (b-a)*sin**2(phi)`` so ``scipy.integrate.quad`` stays quiet
    under warnings-as-errors.
    """
    from scipy import integrate, optimize

    beta = float(beta)
    a = (1.0 - np.sqrt(beta)) ** 2
    b_edge = (1.0 + np.sqrt(beta)) ** 2
    if b_edge <= a:
        return float(a)
    span = b_edge - a

    def phi_of(x: float) -> float:
        # Map x in [a, b] to phi in [0, pi/2].
        t = (x - a) / span
        t = min(max(t, 0.0), 1.0)
        return float(np.arcsin(np.sqrt(t)))

    def integrand(phi: float) -> float:
        # Transformed density: integrable form without endpoint singularities.
        s2 = np.sin(phi) ** 2
        c2 = np.cos(phi) ** 2
        x = a + span * s2
        # rho(x) dx = (span**2 * sin^2 phi * cos^2 phi) / (pi * beta * x) d phi
        # (see Notes in Gavish–Donoho; factor matches the original density)
        if x == 0.0:
            # beta == 1 puts the lower edge at a = 0, so phi = 0 gives 0/0.
            # The limit is span/(pi*beta); quad does not evaluate the endpoint
            # today, but a stricter integrator would hit the NaN.
            return span / (np.pi * beta)
        return (span**2 * s2 * c2) / (np.pi * beta * x)

    def cdf_minus_half(t: float) -> float:
        if t <= a:
            return -0.5
        if t >= b_edge:
            return 0.5
        val, _ = integrate.quad(integrand, 0.0, phi_of(t), epsabs=1e-12)
        return val - 0.5

    lo = a + np.finfo(float).eps * span
    hi = b_edge - np.finfo(float).eps * span
    return float(optimize.brentq(cdf_minus_half, lo, hi))


def svht_omega(beta: float) -> float:
    """Gavish–Donoho (2014) optimal hard-threshold coefficient (unknown noise).

    When the noise level is unknown, it is estimated by the median singular
    value and the hard threshold is ``tau = svht_omega(beta) * median(s)``,
    with ``omega(beta) = lambda(beta) / sqrt(mu_beta)`` and ``mu_beta`` the
    median of the Marchenko–Pastur distribution at aspect ratio ``beta``.
    This is the coefficient used by the ``rank="svht"`` path.

    Parameters
    ----------
    beta : float
        Aspect ratio ``min(shape) / max(shape)`` of the snapshot-pair matrix,
        in ``(0, 1]``.

    Returns
    -------
    float
        ``omega(beta)`` such that ``tau = omega(beta) * median(singular values)``.

    Notes
    -----
    ``omega(1) ≈ 2.858`` is the square-matrix value. As ``beta -> 0``,
    ``omega -> sqrt(2)``.
    """
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"svht_omega requires 0 < beta <= 1, got {beta!r}")
    return float(svht_lambda(beta) / np.sqrt(_mp_median(beta)))


class DMDAnalyzer(BaseAnalyzer):
    """Exact Dynamic Mode Decomposition analyzer.

    The current implementation is intentionally narrow: raw snapshot pairs are
    regressed in Euclidean norm and the resulting modes are sorted by
    ``|lambda|``.
    """

    _METHOD_NAME = "dmd"

    def __init__(
        self,
        file_path: str | None = None,
        *,
        results_dir: str = RESULTS_DIR_DMD,
        figures_dir: str = FIGURES_DIR_DMD,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        n_modes_save: int = 10,
        rank: int | np.integer | Literal["svht", "energy"] | None = None,  # required: positive int | "svht" | "energy"
        energy_fraction: float = 0.999,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            spatial_weights=spatial_weights,
            data=data,
        )
        self.n_modes_save = n_modes_save
        if rank is None:
            raise ValueError(
                "DMD truncation rank is required; pass a positive int, 'svht', or 'energy'. "
                "There is no default: the previous silent default of n_modes_save coupled a "
                "plotting parameter to the operator rank."
            )
        self.rank = rank
        self.energy_fraction = float(energy_fraction)
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.analysis_type = "dmd"
        self.temporal_mean = np.array([])
        # Store modal amplitudes (|b|) after perform_dmd()
        self.amplitudes = np.array([])
        # Continuous-time eigenvalues: omega = log(lambda) / dt
        self.omega = np.array([])
        # Numerical rank used by the last perform_dmd() (after criterion + rcond)
        self.effective_rank = 0
        # Algorithm settings (written by perform_dmd, read by metadata)
        self._dmd_method = "ls"
        self._dmd_embedding_dim = 1
        self._dmd_named_variant = "dmd"

    def _svd_request_rank(self, shape: Sequence[int]) -> int:
        """How many singular triplets to request from ``compute_reduced_svd``.

        An explicit integer ``rank`` can use a truncated path; spectrum-based
        criteria (``"svht"``, ``"energy"``) need the full thin SVD of ``X1``.
        """
        max_r = min(shape)
        rank = self.rank
        if isinstance(rank, (int, np.integer)):
            if int(rank) < 1:
                raise ValueError(f"rank must be >= 1 when given as int, got {rank!r}")
            return min(int(rank), max_r)
        if rank in ("svht", "energy"):
            return max_r
        raise ValueError(f"Unknown rank {rank!r}; use a positive int, 'svht', or 'energy'.")

    def _resolve_rank(self, s: np.ndarray, shape: Sequence[int], rcond: float) -> tuple[int, int]:
        """Map singular values + ``self.rank`` to ``(effective_r, r_requested)``.

        Every path floors by the relative threshold ``s_j > rcond * s[0]``.
        ``r_requested`` is the criterion's target before that floor (used for
        the under-rank warning). ``n_modes_save`` is never consulted here —
        it only bounds how many modes are kept after sorting.
        """
        max_r = min(shape)
        if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
            return 0, max_r

        r_numeric = int(np.sum(s > (rcond * s[0])))
        rank = self.rank

        if isinstance(rank, (int, np.integer)):
            r_requested = min(int(rank), max_r)
            r = min(r_requested, r_numeric)
            return r, r_requested

        if rank == "svht":
            beta = min(shape) / max(shape)
            tau = svht_omega(beta) * float(np.median(s))
            r_svht = int(np.sum(s > tau))
            r = min(r_svht, r_numeric)
            return r, max(r_svht, 1) if r_svht > 0 else max_r

        if rank == "energy":
            frac = self.energy_fraction
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"energy_fraction must be in (0, 1], got {frac!r}")
            energy = np.cumsum(s.astype(np.float64) ** 2)
            total = float(energy[-1])
            if total <= 0.0 or not np.isfinite(total):
                return 0, max_r
            # Smallest r with cumulative s^2 fraction >= energy_fraction.
            r_energy = int(np.searchsorted(energy / total, frac, side="left") + 1)
            r_energy = min(max(r_energy, 1), max_r, s.size)
            r = min(r_energy, r_numeric)
            return r, r_energy

        raise ValueError(f"Unknown rank {rank!r}; use a positive int, 'svht', or 'energy'.")

    def perform_dmd(
        self,
        method: str = "ls",
        embedding_dim: int = 1,
        named_variant: str | None = None,
    ) -> None:
        """Compute DMD on raw shifted snapshots.

        Parameters
        ----------
        method : ``"ls"`` | ``"tls"``
            ``"ls"``  — standard exact DMD (least-squares).
            ``"tls"`` — total least-squares DMD. Its advantage on noisy data
            is an ``embedding_dim=1`` property; see ``embedding_dim`` below.
        embedding_dim : int, default 1
            Embedding depth.  ``embedding_dim=1`` is standard DMD (no delay lift);
            ``embedding_dim>1`` builds a Hankel matrix before forming snapshot pairs.
            Delay embedding repeats the same noise across the Hankel rows, and
            ``"tls"`` assumes the errors in the two snapshot matrices are
            independent. The TLS advantage therefore decays as ``embedding_dim`` grows:
            measured over 200 noisy seeds, TLS beat LS in 177/200 runs at
            ``embedding_dim=1`` but only 95/200 at ``embedding_dim=5``, where LS is better on
            average. Do not pick ``"tls"`` and a large ``embedding_dim`` together
            to fight noise. ``embedding_dim=1`` on DMD is accepted and does no
            delay embedding. ST-POD rejects ``embedding_dim=1``.

        Notes
        -----
        - Uses ``q[:-1]`` and ``q[1:]`` directly as the paired data.
        - Does not subtract the temporal mean.
        - Does not use the spatial metric ``self.W`` in the regression.
        - Sorts the full spectrum by descending ``|lambda|``, breaking
          ``|lambda|`` ties by ``(Re, Im)`` ascending, then truncates.
        - Truncation rank is controlled by ``self.rank`` (required: a positive
          int, ``"svht"``, or ``"energy"``). ``n_modes_save`` only bounds how
          many modes are kept after sorting and never sets the operator rank.
        """
        if method not in ("ls", "tls"):
            raise ValueError(f"Unknown method '{method}'; use 'ls' or 'tls'.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1.")

        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        q = self.data["q"]
        n_snapshots = q.shape[0]
        X = q.T  # (n_spatial, n_time)

        # Delay embedding (Hankel lift)
        if embedding_dim > 1:
            if embedding_dim > n_snapshots - 2:
                raise ValueError(
                    f"embedding_dim={embedding_dim} too large for n_snapshots={n_snapshots}; "
                    "need at least 2 snapshot pairs after embedding "
                    f"(max embedding_dim = {n_snapshots - 2})."
                )
            if embedding_dim >= n_snapshots // 2:
                warnings.warn(
                    f"embedding_dim={embedding_dim} is large relative to n_snapshots={n_snapshots}; "
                    "the effective snapshot count will be small.",
                    stacklevel=2,
                )
            X = _delay_embed(X, embedding_dim)

        X1 = X[:, :-1]
        X2 = X[:, 1:]

        r_svd = self._svd_request_rank(X1.shape)
        u, s, vh = compute_reduced_svd(X1, r_svd)
        rcond = DMD_PINV_RCOND(X1.shape, s.dtype if s.size else X1.dtype)
        r, r_requested = self._resolve_rank(s, X1.shape, rcond)
        # Order matters. The relative test always keeps s[0] (s[0] > rcond * s[0]
        # reduces to 1 > rcond), so r reaches 0 only when the spectrum itself is
        # unusable: empty, non-finite, or all zero. Bumping r back to 1 there would
        # divide by that unusable s[0] and surface as an opaque LinAlgError out of
        # np.linalg.eig, so the degenerate case has to return before the bump.
        if r < r_requested:
            # RuntimeWarning, not UserWarning: this reports a numerical property of
            # the data, not a misuse of the API. The caller asking for more modes
            # than the data supports is normal and expected -- the analytic cases
            # are rank-1 by construction -- so it belongs in the same category numpy
            # uses for conditioning and precision notices.
            warnings.warn(
                f"DMD effective rank {r} is below the requested {r_requested} "
                f"(relative singular-value threshold rcond={rcond:.3e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.effective_rank = int(r)
        if r == 0:
            self.eigenvalues = np.array([])
            self.omega = np.array([])
            self.modes = np.array([])
            self.time_coefficients = np.array([])
            self.amplitudes = np.array([])
            self._dmd_method = method
            self._dmd_embedding_dim = embedding_dim
            self._dmd_named_variant = named_variant or "dmd"
            self._resync_mode_count()
            return

        u_r = u[:, :r]
        s_r = s[:r]
        v_r = vh[:r].conj().T

        # Reduced operator, eig, and amplitude pinv all under one BLAS seam
        # (pinv runs an SVD).
        with apply_blas_limit():
            if method == "tls":
                Z = np.vstack([X1, X2])
                Uz, _, _ = compute_reduced_svd(Z, r)
                Uz = Uz[:, :r]
                n1 = X1.shape[0]
                U11 = Uz[:n1, :]
                U21 = Uz[n1:, :]
                # Project into the reduced basis so atilde is (r, r)
                u_r_H_U11 = u_r.conj().T @ U11
                atilde = (u_r.conj().T @ U21) @ np.linalg.pinv(u_r_H_U11, rcond=rcond)
            else:
                atilde = (u_r.conj().T @ X2 @ v_r) / s_r

            eigvals, w = np.linalg.eig(atilde)
            # Exact DMD mode recovery.  For TLS this is an approximation: the
            # eigenvalues benefit from the TLS operator, while the spatial modes
            # are projected through the LS basis.  This is standard practice;
            # see Hemati et al. (2017) for a discussion of TLS-DMD variants.
            modes = X2 @ (v_r / s_r) @ w

            # Continuous-time eigenvalues (guard against log(0))
            dt = self._require_dt()
            safe_eigvals = np.where(np.abs(eigvals) > 0, eigvals, np.finfo(float).tiny)
            omega = np.log(safe_eigvals.astype(complex)) / dt

            # Amplitudes and time dynamics (use original snapshot count)
            b = np.linalg.pinv(modes, rcond=rcond) @ X[:, 0]
        t = np.arange(n_snapshots)
        time_dynamics = (b[:, None] * eigvals[:, None] ** t).T

        idx = canonical_eigenvalue_order(eigvals)
        n_keep = min(self.n_modes_save, r)
        self.eigenvalues = eigvals[idx][:n_keep]
        self.omega = omega[idx][:n_keep]
        self.modes = modes[:, idx][:, :n_keep]
        self.time_coefficients = time_dynamics[:, idx][:, :n_keep]
        self.amplitudes = np.abs(b[idx][:n_keep])
        self._dmd_method = method
        self._dmd_embedding_dim = embedding_dim
        self._dmd_named_variant = named_variant or "dmd"
        self._resync_mode_count()

    def _get_algorithm_metadata(self) -> dict:
        """Describe the DMD contract currently implemented."""
        method = self._dmd_method
        embedding_dim = self._dmd_embedding_dim
        named_variant = self._dmd_named_variant
        variant = "tls_dmd" if method == "tls" else "exact_dmd"
        if named_variant == "hodmd":
            variant = "hodmd"
        elif named_variant == "tls_hodmd":
            variant = "tls_hodmd"
        elif embedding_dim > 1:
            variant = f"delay_embedded_{variant}"
        return {
            "lift_kind": "delay_embedding" if embedding_dim > 1 else "identity_paired_snapshots",
            "paired_data_contract": "raw_shifted_snapshots",
            "uses_mean_subtraction": False,
            "uses_spatial_metric_in_regression": False,
            "regression_norm": "frobenius",
            "mode_ranking": "abs_lambda_desc",
            "dmd_variant": variant,
            "dmd_named_variant": named_variant,
            "dmd_method": method,
            "dmd_embedding_dim": embedding_dim,
        }

    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return DMD datasets and metadata to save."""
        datasets: dict[str, Any] = {
            "eigenvalues": self.eigenvalues,
            "modes": self.modes,
            "time_coefficients": self.time_coefficients,
            "amplitudes": self.amplitudes,
            "x": self.data["x"],
            "y": self.data["y"],
        }
        if self.omega.size > 0:
            datasets["omega"] = self.omega
        if "z" in self.data and self.data["z"] is not None:
            datasets["z"] = self.data["z"]
        return datasets, self._get_metadata()

    def load_results(self, filename: str | None = None) -> None:
        """Load DMD results and restore state."""
        super().load_results(filename=filename)

        from openmodalpy.core.results import read_results

        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        path = os.path.join(self.results_dir, filename)

        res = read_results(path)
        # Validate required fields.
        if res.modes is None or res.eigenvalues is None or res.time_coefficients is None:
            missing = [
                n
                for n, v in (
                    ("modes", res.modes),
                    ("eigenvalues", res.eigenvalues),
                    ("time_coefficients", res.time_coefficients),
                )
                if v is None
            ]
            raise KeyError(f"{path} is not a DMD result file: missing {', '.join(missing)}")

        # Load amplitudes (backward compatible).
        if res.amplitudes is not None:
            self.amplitudes = res.amplitudes
        else:
            self.amplitudes = np.abs(self.eigenvalues)

        # Load continuous-time eigenvalues if present.
        if res.omega is not None:
            self.omega = res.omega

        # Restore DMD configuration from metadata.
        self._dmd_method = str(res.attrs.get("dmd_method", "ls"))
        self._dmd_embedding_dim = int(res.attrs.get("dmd_embedding_dim", 1))
        self._dmd_named_variant = str(res.attrs.get("dmd_named_variant", "dmd"))

    def _plot_run(self, run_id: str | None = None) -> None:
        """Default figures after run_analysis — the CLI dmd set.

        BEHAVIOUR CHANGE (v0.6.0): run_analysis plots by default now; the
        docstring used to promise no default plots.
        """
        self.plot_eigenvalues()
        if not self._maybe_plot_volumetric_modes(plot_n_modes=min(2, self.n_modes_save)):
            self.plot_modes(plot_n_modes=min(2, self.n_modes_save), modes_per_fig=2)
        self.plot_time_coefficients(n_coeffs_to_plot=min(2, self.n_modes_save))
        self.plot_cumulative_energy()

    def _assign_loaded_results(self, res: AnalysisResults) -> None:
        """Assign loaded results and cap n_modes_save."""
        super()._assign_loaded_results(res)

        # Cap n_modes_save to actual modes available (for narrow files loaded into wide cap).
        n_modes_available = self.modes.shape[1] if self.modes.ndim >= 2 else self.modes.size
        self.n_modes_save = min(self.n_modes_save, n_modes_available)

    def save_results(self, filename: str | None = None) -> None:
        """Save DMD results using the harmonized filename."""
        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        super().save_results(filename=filename)

    def _mode_freq(self, eigvals: np.ndarray) -> np.ndarray | None:
        """Return mode frequencies in Hz, or ``None`` when ``dt`` is unusable.

        Computes ``angle(eigvals) / (2π · dt)`` when ``self.data["dt"]`` is a
        positive finite scalar. Returns ``None`` (never raises) when ``dt`` is
        missing, ``None``, zero, or non-finite — suitable for optional title
        annotations. Physics-bearing plots must use :meth:`_require_dt` instead.
        """
        dt = self.data.get("dt") if self.data else None
        if dt is None:
            return None
        try:
            dt_f = float(dt)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            return None
        return np.angle(eigvals) / (2 * np.pi * dt_f)

    _perform_name = "perform_dmd"

    def plot_eigenvalues(self) -> None:
        os.makedirs(self.figures_dir, exist_ok=True)
        """Plot DMD eigenvalues in the complex plane."""
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot.")
            return
        plt.figure(figsize=(6, 6))
        plt.plot(self.eigenvalues.real, self.eigenvalues.imag, "bo")
        circle = plt.Circle((0, 0), 1.0, color="green", fill=False, linestyle="--")
        ax = plt.gca()
        ax.add_artist(circle)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title("DMD Eigenvalues (Complex Plane)")
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenvalues.png")
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", fname)

    def plot_eigenspectra(self) -> None:
        """Create composite spectra figure: eigenvalues circle, amplitude vs frequency and growth rate."""
        if self.eigenvalues.size == 0 or self.amplitudes.size == 0:
            logger.warning("No eigenvalue data to plot. Run perform_dmd() first.")
            return
        dt = self._require_dt()
        eigvals = self.eigenvalues
        amps = self.amplitudes
        amps_norm = amps / np.max(amps)
        freq = np.angle(eigvals) / (2 * np.pi * dt)
        growth = np.log(np.abs(eigvals)) / dt

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_complex = fig.add_subplot(gs[0, :])
        ax_freq = fig.add_subplot(gs[1, 0])
        ax_growth = fig.add_subplot(gs[1, 1])

        # Complex eigenvalue plot
        ax_complex.plot(eigvals.real, eigvals.imag, "o", mfc="none", mec="brown")
        # Annotate every eigenvalue with its mode number; mark mean explicitly
        for k, lam in enumerate(eigvals):
            label = f"{k + 1}"
            if np.isclose(lam, 1 + 0j, atol=1e-3):
                label += " (mean)"
            ax_complex.text(lam.real, lam.imag, f" {label}", fontsize=7, color="black")
        idx_mean = int(np.argmin(np.abs(eigvals - 1)))
        ax_complex.text(eigvals.real[idx_mean], eigvals.imag[idx_mean], "  mean", color="red", fontsize=8, va="center")
        # Annotate first few oscillatory modes with frequency
        for k in range(min(4, len(eigvals))):
            if k == idx_mean:
                continue
            ax_complex.text(eigvals.real[k], eigvals.imag[k], f"  f={freq[k]:.2f}", fontsize=7, color="black")
        unit_circle = plt.Circle((0, 0), 1.0, color="brown", fill=False, linewidth=1.0)
        ax_complex.add_patch(unit_circle)
        ax_complex.axhline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.axvline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax_complex.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax_complex.set_aspect("equal")
        ax_complex.set_title("DMD eigenvalues")

        # Amplitude vs frequency
        # Route through Callable[..., object] so the optional pre-3.8 matplotlib
        # kwarg ``use_line_collection`` is not checked against the current stub.
        stem_freq: Callable[..., object] = ax_freq.stem
        if "use_line_collection" in inspect.signature(ax_freq.stem).parameters:
            stem_freq(
                freq,
                amps_norm,
                linefmt="brown",
                markerfmt="ro",
                basefmt=" ",
                use_line_collection=True,
            )
        else:
            ax_freq.stem(freq, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ")
        for k, (x, y) in enumerate(zip(freq, amps_norm)):
            ax_freq.text(x, y, f" {k + 1}", fontsize=6, rotation=45, va="bottom")
        ax_freq.set_xlabel("frequency")
        ax_freq.set_ylabel("normalized amplitude")
        ax_freq.set_yscale("log")
        ax_freq.set_title("Amplitude vs frequency")

        # Amplitude vs growth rate
        stem_growth: Callable[..., object] = ax_growth.stem
        if "use_line_collection" in inspect.signature(ax_growth.stem).parameters:
            stem_growth(
                growth,
                amps_norm,
                linefmt="brown",
                markerfmt="ro",
                basefmt=" ",
                use_line_collection=True,
            )
        else:
            ax_growth.stem(growth, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ")
        for k, (x, y) in enumerate(zip(growth, amps_norm)):
            ax_growth.text(x, y, f" {k + 1}", fontsize=6, rotation=45, va="bottom")
        ax_growth.set_xlabel("growth rate")
        ax_growth.set_yscale("log")
        ax_growth.set_title("Amplitude vs growth rate")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fname_spec = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenspectra.png")
        fig.savefig(fname_spec, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", fname_spec)

    def plot_modes_detailed(
        self,
        plot_n_modes: int = 8,
        zero_phase_ref: bool = False,
        unwrap_phase: bool = False,
        ref_method: str = "max",
        show_cylinder: bool = False,
    ) -> None:
        """Plot real, imaginary, magnitude, and phase of several modes in a 4-row grid.

        Args:
            plot_n_modes: Number of modes to plot
            zero_phase_ref: If True, reference phase to 0
            unwrap_phase: If True, unwrap phase
            ref_method: Method for phase reference ('max' or 'mean')
            show_cylinder: If True, add cylinder mask at origin with radius 0.5
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_dmd() first.")
            return
        n_modes = min(plot_n_modes, self.modes.shape[1])
        if n_modes == 0:
            logger.warning("No modes available to plot.")
            return

        nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        if self.modes.shape[0] != nx * ny or nx <= 1 or ny <= 1:
            logger.warning("Detailed mode plotting supports 2D data only.")
            return

        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Optional frequency annotation; modes still plot without a usable dt
        freq = self._mode_freq(self.eigenvalues[:n_modes])

        fig, axes = plt.subplots(4, n_modes, figsize=(3 * n_modes * fig_aspect, 12), squeeze=False)
        row_labels = ["real", "imaginary", "magnitude", "phase"]
        cmaps = [CMAP_DIV, CMAP_DIV, CMAP_SEQ, "twilight"]

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
        else:
            x_mesh, y_mesh = x_coords, y_coords
        # Optionally create cylinder mask
        if show_cylinder:
            distance = np.sqrt(x_mesh**2 + y_mesh**2)
            cylinder_mask = distance <= 0.5
        else:
            cylinder_mask = None

        for m in range(n_modes):
            vec = self.modes[:, m]
            # Phase processing with optional reference and unwrapping
            phase_arr = np.angle(vec)
            if zero_phase_ref:
                if ref_method == "max":
                    phase0 = phase_arr[np.argmax(np.abs(vec))]
                else:  # 'mean'
                    phase0 = np.mean(phase_arr)
                phase_arr = (phase_arr - phase0 + np.pi) % (2 * np.pi) - np.pi
            if unwrap_phase:
                phase_arr = np.unwrap(phase_arr)
            comps = [vec.real, vec.imag, np.abs(vec), phase_arr]
            for r, comp in enumerate(comps):
                ax = axes[r, m]
                comp2d = comp.reshape((ny, nx))
                if cylinder_mask is not None:
                    comp_plot = np.ma.array(comp2d, mask=cylinder_mask)
                else:
                    comp_plot = comp2d
                if r == 2:
                    vmin, vmax = 0.0, np.nanmax(comp_plot)
                elif r == 3:
                    vmin, vmax = -np.pi, np.pi
                else:
                    vmin, vmax = np.nanmin(comp_plot), np.nanmax(comp_plot)
                # Ensure levels are valid and strictly increasing
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue  # skip if invalid
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-12  # tiny range to allow contouring
                levels = np.linspace(vmin, vmax, 21)
                cf = ax.contourf(x_mesh, y_mesh, comp_plot, levels=levels, cmap=cmaps[r], extend="both")
                # Add line contours only if range is significant
                if vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=levels[::4], colors="k", linewidths=0.4, alpha=0.4)

                # Add individual small colorbar inside the data area (upper right)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                cax = inset_axes(ax, width="15%", height="6%", loc="upper right", borderpad=3)
                cb = fig.colorbar(cf, cax=cax, orientation="horizontal", format="%.2f")
                cb.ax.tick_params(labelsize=8, pad=1, colors="black")
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position("top")
                # Set custom ticks: min, 0, max (except for magnitude which starts at 0)
                if r == 2:  # magnitude
                    cb.set_ticks([0, vmax / 2, vmax])
                    cb.set_ticklabels(["0", f"{vmax / 2:.2f}", f"{vmax:.2f}"])
                elif r == 3:  # phase
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels(["-π", "0", "π"])
                else:  # real and imaginary
                    cb.set_ticks([vmin, 0, vmax])
                    cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
                # Make colorbar background semi-transparent
                cax.patch.set_facecolor("black")
                cax.patch.set_alpha(0.7)

                # Optionally add cylinder overlay
                if show_cylinder:
                    cylinder = plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                    ax.add_patch(cylinder)
                # Phase zero-line overlay
                if r == 3 and vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=[0.0], colors="white", linewidths=0.6)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if m == 0:
                    ax.set_ylabel(row_labels[r])
                if r == 0:
                    # Column header annotations
                    if m == 0:
                        header = "1 (mean)"
                    elif freq is not None:
                        header = f"{m + 1} (f={freq[m]:.2f})"
                    else:
                        header = f"{m + 1}"
                    ax.set_title(header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fname_modes = os.path.join(self.figures_dir, f"{self.data_root}_dmd_modes_detailed_{n_modes}_{var_name}.png")
        fig.savefig(fname_modes, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", fname_modes)

    def plot_cumulative_energy(self) -> None:
        """Plot the cumulative energy captured by DMD modes (using |eigval|^2 as proxy)."""
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_dmd() first.")
            return
        # Use squared modulus as 'energy' proxy
        eigvals_abs2 = np.abs(self.eigenvalues) ** 2
        cumulative_energy = np.cumsum(eigvals_abs2) / np.sum(eigvals_abs2) * 100
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative |Eigval|^2 (%)")
        plt.title("Cumulative Energy of DMD Modes (|eigval|^2)")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_cumulative_energy.png")
        plt.savefig(fname, dpi=FIG_DPI * 0.8)
        plt.close()
        logger.info("Saving figure %s", fname)

    def plot_modes(self, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1, show_cylinder: bool = False) -> None:
        """Plot the spatial DMD modes (1D/2D, like POD).

        Args:
            plot_n_modes: Number of modes to plot
            modes_per_fig: Number of modes per figure
            show_cylinder: If True, add cylinder mask at origin with radius 0.5
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_dmd() first.")
            return
        n_modes = self.modes.shape[1]
        if plot_n_modes is not None:
            n_modes = min(plot_n_modes, n_modes, self.n_modes_save)
        if n_modes == 0:
            logger.warning("No modes available to plot.")
            return
        if resolve_volume_layout(self.data, self.modes.shape[0]) is not None:
            self.plot_modes_3d_slices(plot_n_modes=n_modes)
            return
        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        mode_size = self.modes.shape[0]
        physical_nspace = Nx * Ny
        lifted_delays = 1
        is_2d = False
        if Nx > 1 and Ny > 1:
            if mode_size == physical_nspace:
                is_2d = True
            elif physical_nspace > 0 and mode_size % physical_nspace == 0:
                lifted_delays = mode_size // physical_nspace
                is_2d = True
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Compute mode frequencies (Hz) for annotation purposes

        for start in range(0, n_modes, modes_per_fig):
            end = min(start + modes_per_fig, n_modes)
            ncols = end - start
            if is_2d:
                fig, axes = plt.subplots(
                    1,
                    ncols,
                    figsize=(4 * ncols * fig_aspect, 4),
                    squeeze=False,
                )
            else:
                fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3), squeeze=False)
            axes = axes.ravel()
            for j, i in enumerate(range(start, end)):
                ax = axes[j]
                mode = self.modes[:, i].real
                if is_2d:
                    if lifted_delays > 1:
                        mode_2d = mode.reshape((lifted_delays, Ny, Nx))[0]
                    else:
                        mode_2d = mode.reshape((Ny, Nx))
                    # Get meshgrid for plotting
                    if x_coords.ndim == 1 and y_coords.ndim == 1:
                        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
                    else:
                        x_mesh, y_mesh = x_coords, y_coords
                    # Optionally apply cylinder mask (always mask NaNs)
                    nan_mask = np.isnan(mode_2d)
                    if show_cylinder:
                        distance = np.sqrt(x_mesh**2 + y_mesh**2)
                        cylinder_mask = distance <= 0.5
                        combined_mask = nan_mask | cylinder_mask
                    else:
                        combined_mask = nan_mask
                    mode_plot = np.ma.array(mode_2d, mask=combined_mask)
                    mode_flat = mode_2d[~combined_mask]
                    # Guard against empty array (e.g., all points masked)
                    if mode_flat.size == 0:
                        logger.warning("Mode %d has no valid data points, skipping plot.", i)
                        continue
                    # Compute levels with robust limits
                    mode_clean = mode_flat[np.isfinite(mode_flat)]
                    vmin, vmax = np.percentile(mode_clean, [2, 98]) if len(mode_clean) > 0 else (0, 1)
                    levels = np.linspace(vmin, vmax, 21)
                    # Plot filled contour
                    cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")
                    # Contour lines
                    _ = ax.contour(x_mesh, y_mesh, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                    # Optionally add cylinder overlay
                    if show_cylinder:
                        cylinder = plt.Circle(
                            (0, 0), 0.5, fill=True, linewidth=0.5, zorder=3, facecolor="lightgray", edgecolor="black"
                        )
                        ax.add_patch(cylinder)
                    style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                    add_inset_colorbar(
                        fig,
                        ax,
                        cf,
                        self.data,
                        ticks=[vmin, 0, vmax],
                        ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
                    )
                else:
                    ax.plot(mode)
                    ax.set_xlabel("Spatial index")
                    ax.set_ylabel("Amplitude")
                delay_suffix = " (delay 0)" if is_2d and lifted_delays > 1 else ""
                ax.set_title(format_mode_title(self.data, i, default=f"DMD Mode {i + 1}{delay_suffix} [{var_name}]"))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
            fname = os.path.join(
                self.figures_dir,
                f"{self.data_root}_dmd_modes_{start + 1}_to_{end}_{var_name}.png",
            )
            fig.savefig(fname, dpi=FIG_DPI)
            plt.close(fig)
            logger.info("Saving figure %s", fname)

    def plot_modes_3d_slices(self, plot_n_modes: Optional[int] = 4, delay_idx: int = 0) -> None:
        """Plot orthogonal 3D slices for leading DMD/HODMD modes."""
        self._plot_modes_3d("slices", plot_n_modes=plot_n_modes, delay_idx=delay_idx)

    def plot_modes_3d_isometric(self, plot_n_modes: Optional[int] = 4, delay_idx: int = 0) -> None:
        """Plot 3D isosurfaces for leading DMD/HODMD modes."""
        self._plot_modes_3d("isometric", plot_n_modes=plot_n_modes, delay_idx=delay_idx)

    def _plot_modes_3d(self, kind: str, plot_n_modes: Optional[int] = 4, delay_idx: int = 0) -> None:
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_dmd() first.")
            return
        layout = resolve_volume_layout(self.data, self.modes.shape[0])
        if layout is None:
            logger.warning("plot_modes_3d_%s requires volumetric data.", kind)
            return
        _nx, _ny, _nz, lifted_delays = layout
        if delay_idx >= lifted_delays:
            raise ValueError(f"delay_idx={delay_idx} exceeds available lifted embedding_dim ({lifted_delays}).")
        n_modes = min(self.modes.shape[1], self.n_modes_save)
        if plot_n_modes is not None:
            n_modes = min(n_modes, plot_n_modes)
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        freq = self._mode_freq(self.eigenvalues[:n_modes])
        items = []
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.modes[:, mode_idx].real, self.data, block_index=delay_idx)
            delay_suffix = f" | delay={delay_idx}" if lifted_delays > 1 else ""
            if freq is not None:
                title = f"DMD Mode {mode_idx + 1} | f={freq[mode_idx]:.3g}{delay_suffix}"
            else:
                title = f"DMD Mode {mode_idx + 1}{delay_suffix}"
            output_path = os.path.join(self.figures_dir, f"{self.data_root}_dmd_mode_{mode_idx + 1}_{kind}.png")
            items.append(
                {
                    "mode_3d": mode_3d,
                    "output_path": output_path,
                    "title_prefix": title,
                    "scalar_name": "dmd_mode",
                }
            )
        plot_modes_3d(kind, items, x_coords, y_coords, z_coords, data=self.data)

    def plot_time_coefficients(self, n_coeffs_to_plot: int = 2) -> None:
        """Plot DMD temporal coefficients."""
        if self.time_coefficients.size == 0:
            logger.warning("No time coefficients to plot. Run perform_dmd() first.")
            return
        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1], self.n_modes_save)
        if n_coeffs_to_plot == 0:
            logger.warning("No coefficients available to plot.")
            return
        Ns_total = self.time_coefficients.shape[0]
        t, xlabel = self._time_axis(Ns_total)
        fig = plt.figure(figsize=(10, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 1, i + 1)
            coeff = np.asarray(self.time_coefficients[:Ns_total, i].real, dtype=float)
            finite = np.isfinite(coeff)
            if not np.any(finite):
                plt.text(0.5, 0.5, "No finite coefficients", ha="center", va="center", transform=plt.gca().transAxes)
                plt.axis("off")
                continue
            t_plot = t[finite]
            coeff_plot = coeff[finite]
            amp_scale = float(np.max(np.abs(coeff_plot)))
            ylabel = f"Amplitude Mode {i + 1}"
            if np.isfinite(amp_scale) and amp_scale > 1e50:
                coeff_plot = coeff_plot / amp_scale
                ylabel += " (normalized)"
            plt.plot(t_plot, coeff_plot, linewidth=1.0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Temporal Coefficient for DMD Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(t_plot.min(), t_plot.max())
            y_min = float(np.min(coeff_plot))
            y_max = float(np.max(coeff_plot))
            if np.isclose(y_min, y_max):
                margin = 1.0 if np.isclose(y_min, 0.0) else 0.05 * max(abs(y_min), 1.0)
                plt.ylim(y_min - margin, y_max + margin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_time_coeffs.png")
        fig.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def plot_reconstruction_error(self) -> None:
        """Plot the data reconstruction error using an increasing number of DMD modes."""
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            logger.warning("Data, modes, or time coefficients not available. Run perform_dmd() first.")
            return
        data_matrix = self.data["q"]
        # Guard: delay-embedded modes live in a lifted space incompatible with q
        if self.modes.shape[0] != data_matrix.shape[1]:
            logger.warning(
                "Reconstruction error plot is not available for delay-embedded DMD "
                "(mode dimension %d != spatial dimension %d).",
                self.modes.shape[0],
                data_matrix.shape[1],
            )
            return
        # DMD reconstruction: sum_k a_k(t) * phi_k
        n_modes_check = self.modes.shape[1]
        reconstruction_errors = []
        for k in range(1, n_modes_check + 1):
            reconstructed_data_k_modes = self.time_coefficients[:, :k] @ self.modes[:, :k].T
            error = np.linalg.norm(data_matrix - reconstructed_data_k_modes, "fro") / np.linalg.norm(data_matrix, "fro")
            reconstruction_errors.append(error * 100)
        mode_indices = np.arange(1, n_modes_check + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, reconstruction_errors, "s-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title("Data Reconstruction Error vs. Number of DMD Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_reconstruction_error.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)
