"""Metamorphic relations that must hold analytically across analyzers.

These are oracles without closed-form solutions: reductions, path identity, and
homogeneous scaling. Each test states WHY the relation is exact so a later
reader can distinguish a real regression from a wrong assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
)
from openmodalpy.core.weights import PARALLEL_AVAILABLE

# Without optimized parallel routines, use_parallel=True falls through to the
# serial path and serial-vs-parallel would compare serial to serial (vacuous green).
# Fail hard here — never skip — so the suite cannot pass vacuously.
assert PARALLEL_AVAILABLE is True, (
    "PARALLEL_AVAILABLE must be True: serial-vs-parallel tests would otherwise "
    "compare the serial path to itself and report a false green"
)

# Widely separated so a wrong power (alpha^1 or alpha^3) cannot sneak through.
_SCALE_ALPHAS = (1e-3, 1e4)
_RTOL_EXACT = 1e-10
_ATOL_EXACT = 1e-10


def _make_data(q: np.ndarray, dt: float = 0.1) -> dict:
    n_space = q.shape[1]
    return {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": dt,
        "Nx": n_space,
        "Ny": 1,
        "Ns": q.shape[0],
    }


def _independent_hankel(data_centered: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Build the block-Hankel matrix with plain numpy (no library helpers).

    For centered snapshots Q[t, x], column j is the stacked delays
    [Q[j], Q[j+1], ..., Q[j+d-1]] each transposed to a spatial column.
    Built here so the ST-POD check is an independent oracle, not a self-compare.
    """
    ns, n_space = data_centered.shape
    m = ns - embedding_dim + 1
    hankel = np.empty((embedding_dim * n_space, m), dtype=data_centered.dtype)
    for lag in range(embedding_dim):
        hankel[lag * n_space : (lag + 1) * n_space, :] = data_centered[lag : lag + m, :].T
    return hankel


# ---------------------------------------------------------------------------
# DMD: exact DMD at embedding_dim=1 vs independent numpy; embedding_dim=2 is a real lift
# ---------------------------------------------------------------------------


def test_dmd_embedding_dim_one_matches_independent_exact_dmd():
    """embedding_dim=1 must match exact DMD built with plain numpy in the test.

    WHY: exact DMD is the eigendecomposition of the projected companion
    A_tilde = U^H X2 V S^{-1} from the thin SVD of X1 = q.T[:, :-1]. Building
    that operator here with np.linalg (not dmd.py helpers) is an independent
    oracle: agreement proves perform_dmd(embedding_dim=1) implements exact DMD, not
    that two aliases of the same call match. Sorted by |lambda| descending.
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal((32, 4))
    data = _make_data(q)
    n_modes = 3

    analyzer = DMDAnalyzer(
        file_path="meta_dmd_d1",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes,
        rank=n_modes,  # explicit operator rank (matches independent reference)
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd(embedding_dim=1)

    # Independent exact-DMD reference (no library helpers).
    X = q.T  # (n_space, n_time) — same layout as dmd.py
    X1, X2 = X[:, :-1], X[:, 1:]
    r = min(n_modes, min(X1.shape))
    u, s, vh = np.linalg.svd(X1, full_matrices=False)
    u_r, s_r, v_r = u[:, :r], s[:r], vh[:r].conj().T
    a_tilde = (u_r.conj().T @ X2 @ v_r) / s_r  # U^H X2 V S^{-1}
    ref_eigs = np.linalg.eigvals(a_tilde)
    ref_eigs = ref_eigs[np.argsort(np.abs(ref_eigs))[::-1]][:n_modes]

    np.testing.assert_allclose(analyzer.eigenvalues, ref_eigs, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)


def test_dmd_embedding_dim_two_differs_from_one():
    """embedding_dim=2 must change the spectrum relative to embedding_dim=1.

    WHY: embedding_dim=1 regresses in R^{n_space}; embedding_dim=2 stacks consecutive
    snapshots into R^{2 n_space} before forming (X1, X2), so A_tilde is a
    different operator. On broadband random data the lifted eigenvalues are
    not a permutation of the plain ones — if the delay path silently
    degraded to the identity (the old tautology), this assert would fail.
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal((32, 4))
    data = _make_data(q)
    n_modes = 3

    common = dict(
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes,
    )
    d1 = DMDAnalyzer(file_path="meta_dmd_d1_vs", **common, rank=n_modes)
    d1.load_and_preprocess()
    d1.perform_dmd(embedding_dim=1)

    d2 = DMDAnalyzer(file_path="meta_dmd_d2", **common, rank=n_modes)
    d2.load_and_preprocess()
    d2.perform_dmd(embedding_dim=2)

    # Positive: embedding ran — spectra must differ on this data.
    assert d2._dmd_embedding_dim == 2
    assert not np.allclose(d2.eigenvalues, d1.eigenvalues, rtol=_RTOL_EXACT, atol=_ATOL_EXACT), (
        "embedding_dim=2 spectrum matched embedding_dim=1; delay path looks like a no-op"
    )


# ---------------------------------------------------------------------------
# ST-POD: guard at d=1; honest embedding at d=2 vs independent Hankel POD
# ---------------------------------------------------------------------------


def test_stpod_embedding_dim_guard_rejects_one():
    """embedding_dim=1 must raise — ST-POD is space-time, not plain POD.

    WHY: ST-POD's contract requires d >= 2 (stpod.py). The guard is the
    public API surface that prevents silently reducing to snapshot POD.
    """
    rng = np.random.default_rng(1)
    data = _make_data(rng.standard_normal((16, 3)))
    analyzer = STPODAnalyzer(
        file_path="meta_stpod_guard",
        embedding_dim=1,
        n_modes_save=2,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    with pytest.raises(ValueError, match=r"embedding_dim must be >= 2"):
        analyzer.perform_stpod()


def test_stpod_d2_matches_independent_hankel_pod():
    """ST-POD at d=2 equals POD on an independently built Hankel matrix.

    WHY: ST-POD is SVD of the block-Hankel of mean-centered snapshots with
    eigenvalues sigma^2 / m (m = number of Hankel columns). Building that
    Hankel in the test with plain numpy (not the library's embed helpers)
    is an independent oracle: agreement proves the analyzer implements the
    stated construction, not that two aliases of the same helper match.
    Modes are sign-canonicalized at the seam; the independent SVD is brought
    to the same convention before comparison.
    """
    from tests.reference_helpers import canonicalize_reference

    rng = np.random.default_rng(2)
    q = rng.standard_normal((24, 5))
    embedding_dim = 2
    n_modes = 4
    data = _make_data(q)

    analyzer = STPODAnalyzer(
        file_path="meta_stpod_d2",
        embedding_dim=embedding_dim,
        n_modes_save=n_modes,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.perform_stpod()

    # Independent reference: center → Hankel → sqrt(W) weighting → thin SVD
    # → lambda = sigma^2 / m. W is the metric the load built (cell volumes
    # for grid-shaped 1-D coordinates), tiled over the embedding blocks; modes are
    # divided back by sqrt(W) exactly as the solver unweights them.
    centered = q - np.mean(q, axis=0)
    hankel = _independent_hankel(centered, embedding_dim)
    m_cols = hankel.shape[1]
    w_lift = np.tile(np.asarray(analyzer.W).ravel(), embedding_dim)
    u, sigma, _vt = np.linalg.svd(np.sqrt(w_lift)[:, None] * hankel, full_matrices=False)
    ref_eigs = (sigma[:n_modes] ** 2) / m_cols
    ref_modes, _ = canonicalize_reference(u[:, :n_modes])
    ref_modes = ref_modes / np.sqrt(w_lift)[:, None]

    np.testing.assert_allclose(analyzer.eigenvalues, ref_eigs, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)
    np.testing.assert_allclose(analyzer.modes, ref_modes, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)


# ---------------------------------------------------------------------------
# serial == parallel for BSMD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nfft,n_space,triads",
    [
        (8, 4, [(0, 0, 0), (1, -1, 0), (1, 1, 2)]),
        (16, 4, [(0, 0, 0), (2, -2, 0), (3, 3, 6)]),
        (8, 6, [(0, 0, 0), (1, 1, 2), (2, -1, 1)]),
    ],
)
def test_bsmd_serial_parallel(tmp_path, nfft, n_space, triads):
    """BSMD static core serial and thread-parallel paths must agree.

    WHY: use_parallel only changes the execution schedule (ThreadPoolExecutor
    vs sequential loop over the same _compute_single_triad). Each triad is
    independent; the eigenpair of the bispectral correlation matrix does not
    depend on order, so results are identical — not statistically similar.
    """
    assert PARALLEL_AVAILABLE is True
    rng = np.random.default_rng(20 + nfft + n_space)
    ns = 3 * nfft
    q = rng.standard_normal((ns, n_space))
    data = _make_data(q, dt=1.0)

    common = dict(
        nfft=nfft,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=triads,
    )

    serial = BSMDAnalyzer(file_path=f"meta_bsmd_s_{nfft}_{n_space}", use_parallel=False, **common)
    serial.load_and_preprocess()
    serial.compute_fft_blocks()
    serial._perform_static_bsmd_core()

    parallel = BSMDAnalyzer(file_path=f"meta_bsmd_p_{nfft}_{n_space}", use_parallel=True, **common)
    parallel.load_and_preprocess()
    parallel.compute_fft_blocks()
    parallel._perform_static_bsmd_core()

    # Exact same triad arithmetic; only scheduling differs.
    np.testing.assert_allclose(parallel.eigenvalues, serial.eigenvalues, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.abs(parallel.modes1), np.abs(serial.modes1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.abs(parallel.modes2), np.abs(serial.modes2), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Scaling: q -> alpha*q  ⇒  lambda -> alpha^2 * lambda
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", _SCALE_ALPHAS)
def test_scaling_pod_eigenvalues_quadratic(alpha):
    """POD eigenvalues are quadratic forms of the snapshots.

    WHY: POD solves an eigenproblem on a second-moment (covariance / Gram)
    matrix built from mean-centered q. Scaling q by alpha scales that
    operator by alpha^2, hence every eigenvalue by alpha^2. Two alphas
    spanning 1e-3 and 1e4 expose a wrong power (alpha^1 or alpha^3).
    """
    rng = np.random.default_rng(30)
    q = rng.standard_normal((32, 4))
    n_modes = 3

    def eigs(field: np.ndarray) -> np.ndarray:
        a = PODAnalyzer(
            file_path="meta_scale_pod",
            data_loader=lambda _: _make_data(field),
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
        )
        a.load_and_preprocess()
        a.perform_pod()
        return a.eigenvalues

    base = eigs(q)
    scaled = eigs(alpha * q)
    np.testing.assert_allclose(scaled, (alpha**2) * base, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)


@pytest.mark.parametrize("alpha", _SCALE_ALPHAS)
def test_scaling_spod_eigenvalues_quadratic(alpha, tmp_path):
    """SPOD eigenvalues inherit the alpha^2 energy scaling of the CSD.

    WHY: SPOD eigenvalues are those of the weighted cross-spectral density
    of the block-FFT of q. FFT is linear in q, CSD is quadratic, so lambda
    scales as alpha^2. BSMD's distinct alpha^3 law is covered elsewhere.
    """
    rng = np.random.default_rng(31)
    q = rng.standard_normal((64, 4))
    nfft = 16

    def eigs(field: np.ndarray, tag: str) -> np.ndarray:
        a = SPODAnalyzer(
            file_path=f"meta_scale_spod_{tag}",
            nfft=nfft,
            overlap=0.0,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: _make_data(field),
            spatial_weight_type="uniform",
        )
        a.load_and_preprocess()
        a.compute_fft_blocks()
        a.perform_spod()
        return a.eigenvalues

    base = eigs(q, "1")
    scaled = eigs(alpha * q, f"a{alpha}")
    np.testing.assert_allclose(scaled, (alpha**2) * base, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)


@pytest.mark.parametrize("alpha", _SCALE_ALPHAS)
def test_scaling_mpod_eigenvalues_quadratic(alpha):
    """mPOD eigenvalues scale as alpha^2 (band-limited POD of q).

    WHY: mPOD is POD on band-filtered snapshots. Filtering is linear in q,
    POD eigenvalues are quadratic, so the composition is alpha^2. Edges are
    strictly interior to [0, Nyquist] so the multi-band filter path runs
    (full-span edges take the POD shortcut and would not test mPOD).
    """
    rng = np.random.default_rng(32)
    q = rng.standard_normal((32, 4))
    n_modes = 3
    # dt=0.1 → Nyquist 5 Hz; interior edge at 2 Hz forces multi-band filtering.
    band_edges = [0.0, 2.0, 5.0]

    def run(field: np.ndarray) -> MPODAnalyzer:
        a = MPODAnalyzer(
            file_path="meta_scale_mpod",
            data_loader=lambda _: _make_data(field),
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
            band_edges=band_edges,
        )
        a.load_and_preprocess()
        a.perform_mpod()
        return a

    base = run(q)
    # Positive: filtering path ran — more than one band contributed modes.
    assert len(np.unique(base.mode_band_indices)) > 1, (
        "mode_band_indices has a single band; mPOD took the full-band POD shortcut"
    )
    scaled = run(alpha * q)
    np.testing.assert_allclose(
        scaled.eigenvalues,
        (alpha**2) * base.eigenvalues,
        rtol=_RTOL_EXACT,
        atol=_ATOL_EXACT,
    )


@pytest.mark.parametrize("alpha", _SCALE_ALPHAS)
def test_scaling_stpod_eigenvalues_quadratic(alpha):
    """ST-POD eigenvalues scale as alpha^2 (SVD energy of the Hankel).

    WHY: The Hankel is linear in q; ST-POD stores sigma^2 / m from the SVD.
    sigma scales as |q| ~ alpha, so lambda ~ alpha^2. Same power as POD
    because ST-POD is POD on a linear lift of the snapshots.
    """
    rng = np.random.default_rng(33)
    q = rng.standard_normal((32, 4))
    n_modes = 3

    def eigs(field: np.ndarray) -> np.ndarray:
        a = STPODAnalyzer(
            file_path="meta_scale_stpod",
            embedding_dim=2,
            n_modes_save=n_modes,
            data_loader=lambda _: _make_data(field),
            spatial_weight_type="uniform",
        )
        a.load_and_preprocess()
        a.perform_stpod()
        return a.eigenvalues

    base = eigs(q)
    scaled = eigs(alpha * q)
    np.testing.assert_allclose(scaled, (alpha**2) * base, rtol=_RTOL_EXACT, atol=_ATOL_EXACT)
