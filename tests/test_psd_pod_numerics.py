"""Numerical checks for the PSD-POD complex path (``_solve_eigh_complex``).

The CLI suite reaches this path but does not assert its arithmetic: dropping the
metric from the time coefficients, or dropping the ``1/sqrt(lambda*N)`` mode
normalization, still leaves that suite green.

NOTE: ``reference_psd_pod`` is a twin of ``_solve_eigh_complex`` (weighted
mode build + unweight, the openmodalpy-era zero-measure policy), kept as a
refactoring guard against the shared solver. It is a characterization test,
NOT an independent physics oracle — it only proves the shared path stays
consistent with this expression.

That limit was not theoretical. The twin carried the same conjugate on the
ensemble that the solver did, so both agreed on time coefficients that were
wrong. The last two tests in this file state the two properties a POD
expansion must have, and neither one reads the twin: the variance of the k-th
coefficient is the k-th eigenvalue, and the coefficients times the modes
return the ensemble. Both hold to about 1e-15 now and failed by a factor of
five before.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.decomposition import (
    SpatialMetric,
    _unweight_modes,
    weighted_second_order,
)


def reference_psd_pod(ensemble: np.ndarray, weights: np.ndarray, n_modes_save: int):
    """Twin of ``_solve_eigh_complex`` (weighted build + unweight policy).

    Characterization / refactoring guard only — not an independent oracle.
    Where w > 0 this is algebraically identical to the pre-refactor unweighted
    build (commands.py at 3102d9a); where w == 0 the mode value is exactly 0.
    """
    n_realizations = ensemble.shape[0]
    ensemble_weighted = ensemble * np.sqrt(weights)[np.newaxis, :]
    kernel = (ensemble_weighted @ ensemble_weighted.conj().T) / n_realizations
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues.real)[::-1]
    keep = min(n_modes_save, len(order))
    eigenvalues = np.real_if_close(eigenvalues[order][:keep])
    eigenvectors = eigenvectors[:, order][:, :keep]
    safe_eigs = np.maximum(np.real(eigenvalues), 1e-16)
    weighted_modes = (ensemble_weighted.conj().T @ eigenvectors) / np.sqrt(safe_eigs * n_realizations)
    modes = _unweight_modes(weighted_modes, weights)
    time_coefficients = ensemble @ (weights[:, np.newaxis] * modes)
    return modes, eigenvalues, time_coefficients


def _phase_invariant_overlap(modes: np.ndarray, ref_modes: np.ndarray) -> float:
    """Min |<u, u_ref>| / (|u| |u_ref|) over modes; complex vectors up to phase."""
    num = np.abs(np.sum(np.conj(modes) * ref_modes, axis=0))
    den = np.linalg.norm(modes, axis=0) * np.linalg.norm(ref_modes, axis=0)
    # Degenerate cases must FAIL, not score a perfect overlap: an empty basis or
    # a zero-norm mode is a broken solve, and defaulting them to 1.0 would make
    # this helper unable to detect exactly the failures it exists to catch.
    assert num.size > 0, "solver returned an empty mode set"
    assert np.all(den > 0), "a mode has zero norm; overlap is undefined"
    return float(np.min(num / den))


def _run_solver(ensemble: np.ndarray, weights: np.ndarray, n_keep: int):
    return weighted_second_order(
        ensemble,
        SpatialMetric(weights),
        method="eigh",
        n_keep=n_keep,
    )


def _assert_matches_reference(ensemble: np.ndarray, weights: np.ndarray, n_keep: int) -> None:
    ref_modes, ref_eigs, ref_coeffs = reference_psd_pod(ensemble, weights, n_keep)
    modes, eigs, coeffs = _run_solver(ensemble, weights, n_keep)

    # Shapes first: an empty or wrongly-truncated result must not reach the
    # value comparisons, several of which are vacuous on empty arrays.
    n_samples, n_space = ensemble.shape
    assert modes.shape == (n_space, n_keep)
    assert eigs.shape == (n_keep,)
    assert coeffs.shape == (n_samples, n_keep)

    np.testing.assert_allclose(np.real(eigs), np.real(ref_eigs), rtol=1e-10, atol=1e-12)
    assert _phase_invariant_overlap(modes, ref_modes) >= 1.0 - 1e-8
    np.testing.assert_allclose(np.abs(coeffs), np.abs(ref_coeffs), rtol=1e-8, atol=1e-10)


def _fourier_ensemble(seed: int = 4242) -> tuple[np.ndarray, np.ndarray]:
    """10 complex Fourier realizations over 6 spatial points."""
    rng = np.random.default_rng(seed)
    ensemble = rng.standard_normal((10, 6)) + 1j * rng.standard_normal((10, 6))
    weights = np.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5])
    return ensemble, weights


@pytest.mark.oracle
def test_manufactured_rank_one_ensemble_matches_closed_form():
    """Rank-1 ensemble a_n * phi has a closed-form leading eigenvalue and mode.

    Build ensemble[n] = a_n * phi for a known W-normalised real phi. Under the
    weighted temporal-kernel convention the kernel is rank-1 with leading
    eigenvalue (sum_n |a_n|^2) * ||phi||_W^2 / N; the leading mode recovers
    phi up to a global phase; residual modes sit at ~0 energy. Independent of
    the pre-refactor twin — a wrong sqrt(W) seam moves the eigenvalue off the
    closed form.
    """
    rng = np.random.default_rng(11)
    n_realizations = 8
    weights = np.array([0.5, 1.0, 2.0, 0.25, 3.0])
    phi_raw = rng.standard_normal(weights.size)
    phi_norm = np.sqrt(np.sum(phi_raw**2 * weights))
    phi = phi_raw / phi_norm
    # ||phi||_W^2 == 1 by construction
    assert np.isclose(np.sum(phi**2 * weights), 1.0)

    amplitudes = rng.standard_normal(n_realizations) + 1j * rng.standard_normal(n_realizations)
    ensemble = amplitudes[:, np.newaxis] * phi[np.newaxis, :]

    modes, eigenvalues, _coeffs = _run_solver(ensemble, weights, n_keep=n_realizations)

    expected_leading = float(np.sum(np.abs(amplitudes) ** 2) * 1.0 / n_realizations)
    # Relative Gram-rank filter keeps only the single significant eigenvalue;
    # residual directions sit at the noise floor and are not returned as modes.
    assert eigenvalues.size == 1
    assert modes.shape[1] == 1
    np.testing.assert_allclose(np.real(eigenvalues[0]), expected_leading, rtol=1e-10, atol=1e-12)

    # Leading mode recovers phi up to a global phase (real phi → sign only).
    mode0 = modes[:, 0]
    overlap = np.abs(np.vdot(mode0 * weights, phi)) / (
        np.sqrt(np.vdot(mode0 * weights, mode0).real) * np.sqrt(np.sum(phi**2 * weights))
    )
    assert float(overlap) >= 1.0 - 1e-8


@pytest.mark.characterization
def test_psd_pod_positive_nonuniform_metric():
    ensemble, weights = _fourier_ensemble()
    assert np.all(weights > 0)
    assert len(np.unique(weights)) == weights.size
    _assert_matches_reference(ensemble, weights, n_keep=4)


@pytest.mark.characterization
def test_psd_pod_isolated_zero_weight_station():
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    weights[2] = 0.0
    assert weights[2] == 0.0
    assert np.count_nonzero(weights == 0.0) == 1
    # Zero measure: the cell contributes nothing (exact sqrt(0) = 0). On this
    # ensemble the shared path still agrees with the reference within tol.
    _assert_matches_reference(ensemble, weights, n_keep=4)


def test_psd_pod_planted_garbage_at_zero_weight_station():
    """Masked station must report mode value 0 even if the raw data is garbage.

    Mirrors the zero-measure measurement: zero one station's weight, plant
    1e6 there, and require (i) exact-zero mode values at that station, (ii)
    spectrum matching the station-deleted reference, (iii) other stations
    matching the no-garbage run.
    """
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    station = 2
    weights[station] = 0.0
    n_keep = 4

    modes_clean, eigs_clean, coeffs_clean = _run_solver(ensemble, weights, n_keep)

    ensemble_garbage = ensemble.copy()
    ensemble_garbage[:, station] = 1e6
    modes_g, eigs_g, coeffs_g = _run_solver(ensemble_garbage, weights, n_keep)

    # (i) mode values at the masked station are exactly 0 for every kept mode
    assert modes_g.shape[1] == n_keep
    assert np.all(modes_g[station, :] == 0.0)

    # (ii) eigenvalues match deleting the station outright
    keep_stations = np.arange(ensemble.shape[1]) != station
    _, eigs_deleted, _ = _run_solver(
        ensemble[:, keep_stations],
        weights[keep_stations],
        n_keep,
    )
    np.testing.assert_allclose(np.real(eigs_g), np.real(eigs_deleted), rtol=1e-10, atol=1e-12)

    # (iii) other stations (and spectrum/coeffs) match the no-garbage run
    np.testing.assert_allclose(np.real(eigs_g), np.real(eigs_clean), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        modes_g[keep_stations, :],
        modes_clean[keep_stations, :],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(coeffs_g, coeffs_clean, rtol=1e-10, atol=1e-12)


def test_psd_pod_negative_weight_station_raises():
    """A negative weight is not a valid inner-product metric — raise ValueError.

    An isolated zero among positive weights is still allowed (it contributes
    nothing); a negative entry means the metric is not an inner product, so
    the solver refuses it before taking ``sqrt(W)``.
    """
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    weights[1] = -0.5
    assert weights[1] < 0.0

    with pytest.raises(ValueError, match="negative weight"):
        _run_solver(ensemble, weights, n_keep=4)


def test_the_coefficient_variance_is_the_eigenvalue() -> None:
    """The k-th eigenvalue is the mean energy of the k-th coefficient.

    This is what an eigenvalue of the POD kernel means, and it does not depend
    on any sign or phase convention, so it holds whatever the solver decides
    about either. It is the check the twin above cannot make, because the twin
    computes its coefficients the same way the solver does.
    """
    ensemble, weights = _fourier_ensemble()
    n_keep = 4

    _, eigenvalues, coefficients = _run_solver(ensemble, weights, n_keep)

    measured = (np.abs(coefficients) ** 2).mean(axis=0)
    np.testing.assert_allclose(measured, np.real(eigenvalues), rtol=1e-10)


def test_the_coefficients_and_modes_rebuild_the_ensemble() -> None:
    """A full set of coefficients and modes must return the field.

    ``n_keep`` is the number of realizations here, so the expansion is
    complete and the sum has to be exact to rounding. This is the second
    convention-free property, and it is the one that says the coefficients
    belong to these modes and not to their conjugates.
    """
    ensemble, weights = _fourier_ensemble()
    n_keep = ensemble.shape[0]

    modes, _, coefficients = _run_solver(ensemble, weights, n_keep)

    np.testing.assert_allclose(coefficients @ modes.conj().T, ensemble, atol=1e-10)


def test_the_modes_are_orthonormal_under_the_spatial_weight() -> None:
    """The modes must be orthonormal in the inner product the grid defines.

    Without this the eigenvalue is not an energy and the coefficient variance
    identity above would hold for the wrong reason.
    """
    ensemble, weights = _fourier_ensemble()
    n_keep = 4

    modes, _, _ = _run_solver(ensemble, weights, n_keep)

    gram = modes.conj().T @ (weights[:, np.newaxis] * modes)
    np.testing.assert_allclose(gram, np.eye(n_keep), atol=1e-10)
