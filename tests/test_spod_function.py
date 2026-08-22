import h5py
import numpy as np
import pytest

from openmodalpy import SPODAnalyzer
from openmodalpy.core.base import (
    _QHAT_STAMP_ATTR_PREFIX,
    PARALLEL_AVAILABLE,
    _qhat_cache_stamp,
    make_result_filename,
    spod_function,
)
from openmodalpy.core.decomposition import spod_single_frequency
from openmodalpy.core.parallel import spod_single_frequency_optimized
from tests.reference_helpers import reference_pivot_index


def test_spod_function_simple():
    qhat = np.array([[1.0], [0.0]], dtype=complex)
    w = np.ones((2, 1))
    phi, lam, psi = spod_function(qhat, nblocks=1, dst=1.0, w=w, return_psi=True)
    assert phi.shape == (2, 1)
    assert psi.shape == (1, 1)
    assert np.allclose(lam, 1.0)
    assert np.allclose(phi[:, 0], [1.0, 0.0])


def test_spod_function_per_component_weights():
    qhat = np.array([[1.0], [2.0]], dtype=complex)
    w = np.zeros((1, 1, 2))
    w[0, 0, 0] = 1.0
    w[0, 0, 1] = 2.0
    phi, lam, psi = spod_function(qhat, nblocks=1, dst=1.0, w=w, return_psi=True)
    assert phi.shape == (2, 1)
    assert psi.shape == (1, 1)
    assert np.allclose(lam, 9.0)
    assert np.allclose(phi[:, 0], [1 / 3, 2 / 3])


def _assert_spod_modes_canonical(modes: np.ndarray) -> None:
    """Each SPOD mode's band-pivot entry must be real and positive."""
    for k in range(modes.shape[1]):
        col = modes[:, k]
        if not np.any(np.abs(col) > 0):
            continue
        i = reference_pivot_index(col)
        v = col[i]
        assert float(np.real(v)) > 0
        assert abs(float(np.imag(v))) <= 1e-9 * max(float(np.abs(v)), 1e-30)


@pytest.mark.characterization
def test_spod_modes_deterministic_and_canonical():
    """Characterisation test of canonicalisation and phase invariance.

    This is not evidence about the eigenvalue magnitudes. SPOD modes are
    phase-fixed, route-stable, and spectrum-invariant. A global unit phase on
    ``qhat`` leaves the CSD matrix unchanged but multiplies the raw spatial
    modes by that phase. After canonicalization both inputs give the same phi
    (no ``np.abs``). Removing the call makes this fail.
    """
    rng = np.random.default_rng(42)
    n_space, nblocks = 10, 6
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))
    w = np.linspace(0.5, 2.0, n_space)
    dst = 0.15
    phase = np.exp(1j * 0.73)

    phi_a, lam_a = spod_single_frequency(qhat, nblocks, dst, w)
    phi_b, lam_b = spod_single_frequency(qhat, nblocks, dst, w)
    phi_phased, lam_phased = spod_single_frequency(qhat * phase, nblocks, dst, w)
    phi_psi, _, psi = spod_single_frequency(qhat, nblocks, dst, w, return_psi=True)
    phi_no_psi, _ = spod_single_frequency(qhat, nblocks, dst, w, return_psi=False)

    _assert_spod_modes_canonical(phi_a)
    np.testing.assert_allclose(phi_a, phi_b, rtol=0, atol=0)
    np.testing.assert_allclose(lam_a, lam_b, rtol=0, atol=0)
    np.testing.assert_allclose(phi_a, phi_phased, rtol=0, atol=1e-12)
    np.testing.assert_allclose(lam_a, lam_phased, rtol=0, atol=1e-12)
    np.testing.assert_allclose(phi_a, phi_psi, rtol=0, atol=0)
    np.testing.assert_allclose(phi_a, phi_no_psi, rtol=0, atol=0)
    assert psi is not None

    if PARALLEL_AVAILABLE:
        phi_opt, lam_opt = spod_single_frequency_optimized(qhat, w.reshape(-1, 1), nblocks, dst)
        np.testing.assert_allclose(phi_a, phi_opt, rtol=0, atol=0)
        np.testing.assert_allclose(lam_a, lam_opt, rtol=0, atol=0)

    # Spectrum from the CSD matrix alone — canonicalization must not move it.
    x = qhat / np.sqrt(nblocks * dst)
    m = (np.conj(x).T * w[np.newaxis, :]) @ x
    lam_ref = np.sort(np.linalg.eigvalsh(m))[::-1]
    np.testing.assert_allclose(lam_a, np.abs(lam_ref), rtol=0, atol=1e-12)

    # psi must carry the SAME factor as phi, not be left at its raw LAPACK
    # phase. Every assertion above passes if only phi is canonicalized, so
    # check the relation that ties them: X psi = phi * sqrt(lambda).
    significant = lam_a > 1e-12 * float(np.max(lam_a))
    np.testing.assert_allclose(
        (x @ psi)[:, significant],
        (phi_psi * np.sqrt(lam_a)[np.newaxis, :])[:, significant],
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("use_parallel", [False, True])
def test_spod_function_rejects_invalid_metric(use_parallel):
    """Negative weights and a zero-measure metric raise; isolated zeros stay allowed.

    Both the serial and optimized routes must refuse the same invalid metrics.
    An isolated zero among positive weights is still accepted: SPOD applies the
    weights as they are, so that cell contributes nothing to the CSD. The 1e-12
    floor is the POD seam's, not this one's.
    """
    if use_parallel and not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))
    w_neg = np.array([1.0, -0.5, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_zero = np.zeros((n_space, 1))
    w_iso = np.array([1.0, 0.0, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_ok = np.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5]).reshape(-1, 1)

    with pytest.raises(ValueError, match="negative weight"):
        spod_function(qhat, nblocks, 0.1, w_neg, use_parallel=use_parallel)
    with pytest.raises(ValueError, match="zero total measure"):
        spod_function(qhat, nblocks, 0.1, w_zero, use_parallel=use_parallel)

    phi_iso, lam_iso = spod_function(qhat, nblocks, 0.1, w_iso, use_parallel=use_parallel)
    phi_ok, lam_ok = spod_function(qhat, nblocks, 0.1, w_ok, use_parallel=use_parallel)
    assert phi_iso.shape[0] == n_space
    assert phi_ok.shape[0] == n_space
    assert np.all(np.isfinite(lam_iso))
    assert np.all(np.isfinite(lam_ok))


@pytest.mark.parametrize("use_parallel", [False, True])
def test_spod_function_rejects_nonfinite_metric(use_parallel):
    """NaN and inf weights raise the same way on both SPOD routes."""
    if use_parallel and not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))
    w_nan = np.array([1.0, np.nan, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_inf = np.array([1.0, np.inf, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

    with pytest.raises(ValueError, match="non-finite"):
        spod_function(qhat, nblocks, 0.1, w_nan, use_parallel=use_parallel)
    with pytest.raises(ValueError, match="non-finite"):
        spod_function(qhat, nblocks, 0.1, w_inf, use_parallel=use_parallel)


def test_spod_single_frequency_optimized_rejects_invalid_metric():
    """Direct call refuses negative, zero-measure, and non-finite metrics.

    Guards the path that already validates through the shared body so a future
    edit cannot drop that check without this test failing first.
    """
    if not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))

    with pytest.raises(ValueError, match="negative weight"):
        spod_single_frequency_optimized(qhat, np.array([1.0, -0.5, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1), nblocks, 0.1)
    with pytest.raises(ValueError, match="zero total measure"):
        spod_single_frequency_optimized(qhat, np.zeros((n_space, 1)), nblocks, 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        spod_single_frequency_optimized(qhat, np.array([1.0, np.nan, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1), nblocks, 0.1)


def test_spod_save_reapplies_the_fft_cache_stamp(tmp_path):
    """save_results rewrites the file then re-stamps the FFT-cache attrs.

    A full mode-"w" write clears the stamp that compute_fft_blocks left; the
    re-apply after write_results is what lets the next run reuse FFTBlocks.
    Without it the stamp is missing and the next run recomputes.
    """
    rng = np.random.default_rng(7)
    data = {
        "q": rng.standard_normal((8, 4)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Ns": 8,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=4,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.save_results()

    save_path = tmp_path / make_result_filename("dummy", 4, 0.0, 8, "spod")
    expected = _qhat_cache_stamp(analyzer, analyzer.data["q"])
    # Without this the loop below would assert nothing if the stamp ever
    # became empty, and the test would pass while checking nothing.
    assert expected, "the stamp is empty, so this test would verify nothing"
    with h5py.File(save_path, "r") as handle:
        # The stamp survives a full rewrite only because it is re-applied
        # afterwards. Assert the rewrite actually happened, so a save that did
        # nothing at all could not satisfy the stamp checks below by leaving
        # the compute-time stamp untouched.
        assert "modes" in handle and "eigenvalues" in handle
        for key, want in expected.items():
            attr = f"{_QHAT_STAMP_ATTR_PREFIX}{key}"
            assert attr in handle.attrs, f"missing stamp attr {attr}"
            got = handle.attrs[attr]
            if isinstance(want, bool):
                got = bool(got)
            elif isinstance(want, int):
                got = int(got)
            elif isinstance(want, float):
                got = float(got)
            else:
                got = str(got)
            assert got == want
