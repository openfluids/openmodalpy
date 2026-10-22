import warnings

import h5py
import matplotlib
import numpy as np
import pytest

from openmodalpy import DMDAnalyzer
from openmodalpy.dmd import _delay_embed


def _exact_dmd_eigenvalues(q: np.ndarray, n_modes_save: int, rank=None) -> np.ndarray:
    """Independent exact-DMD reference.

    ``rank`` is the operator truncation (defaults to ``n_modes_save`` to match
    the pre-requirement migration). ``n_modes_save`` also bounds how many
    eigenvalues are returned after sorting.
    """
    x = q[:-1, :].T
    y = q[1:, :].T

    u, s, vh = np.linalg.svd(x, full_matrices=False)
    r_full = len(s)
    r_req = n_modes_save if rank is None else int(rank)
    r = min(r_req, r_full)
    u_r = u[:, :r]
    s_r = np.diag(s[:r])
    v_r = vh.conj().T[:, :r]

    atilde = u_r.conj().T @ y @ v_r @ np.linalg.inv(s_r)
    eigvals, _ = np.linalg.eig(atilde)
    idx = np.argsort(np.abs(eigvals))[::-1]
    n_keep = min(n_modes_save, r)
    return eigvals[idx][:n_keep]


def test_perform_dmd_simple():
    data = {
        "q": np.array([[1, 2], [2, 4], [4, 8]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=2, rank=2
    )
    analyzer.load_and_preprocess()
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd()
    # Snapshot pairs are rank-1 (col2 = 2 * col1); relative floor keeps one mode.
    assert analyzer.effective_rank == 1
    assert analyzer.modes.shape == (2, 1)
    assert analyzer.time_coefficients.shape == (3, 1)
    assert np.isclose(analyzer.eigenvalues[0], 2.0, atol=1e-6)


def test_plot_eigenspectra_stem_compat(monkeypatch, tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values."""
    rng = np.random.default_rng(10)
    data = {
        "q": rng.standard_normal((8, 4)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Ns": 8,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy.h5",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        rank=10,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    calls = []

    def stem_no_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None):
        calls.append("no_use")
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, "stem", stem_no_use)
    analyzer.plot_eigenspectra()
    assert "no_use" in calls

    calls.clear()

    def stem_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None, use_line_collection=True):
        calls.append("use_line_collection" if use_line_collection else "use")
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, "stem", stem_use)
    analyzer.plot_eigenspectra()
    assert "use_line_collection" in calls

    expected = tmp_path / "dummy_dmd_eigenspectra.png"
    assert expected.exists()


@pytest.mark.characterization
def test_dmd_uses_raw_shifted_snapshots_without_weighting():
    """Exact DMD on raw shifted snapshots, independent of a post-hoc W.

    This pins the paired-data contract (no mean subtraction; the regression
    matches the unweighted exact-DMD oracle). The two W vectors have the same
    mean, so a mean-preserving use of W would still match ``expected_raw`` —
    eigenvalues of a uniformly scaled metric are isospectral for exact DMD.
    The spatial-structure tripwire lives in
    ``test_prescribed_weights_change_the_eigenvalues``, which compares DMD
    modes under an equal-mean pair. This test stays about the oracle contract.
    """
    data = {
        "q": np.array(
            [
                [10.0, 1.0],
                [11.0, 2.0],
                [13.0, 4.0],
                [16.0, 8.0],
                [20.0, 16.0],
            ],
            dtype=float,
        ),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 5,
    }

    analyzer_a = DMDAnalyzer(
        file_path="dummy_a", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=2, rank=2
    )
    analyzer_a.load_and_preprocess()
    analyzer_a.W = np.array([1.0, 50.0])
    analyzer_a.perform_dmd()

    analyzer_b = DMDAnalyzer(
        file_path="dummy_b", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=2, rank=2
    )
    analyzer_b.load_and_preprocess()
    analyzer_b.W = np.array([50.0, 1.0])
    analyzer_b.perform_dmd()

    expected_raw = _exact_dmd_eigenvalues(data["q"], n_modes_save=2)
    expected_centered = _exact_dmd_eigenvalues(
        data["q"] - np.mean(data["q"], axis=0, keepdims=True),
        n_modes_save=2,
    )

    np.testing.assert_allclose(analyzer_a.eigenvalues, expected_raw, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(analyzer_b.eigenvalues, expected_raw, rtol=1e-10, atol=1e-10)
    assert not np.allclose(
        np.sort_complex(analyzer_a.eigenvalues),
        np.sort_complex(expected_centered),
        rtol=1e-8,
        atol=1e-8,
    )


def test_dmd_save_results_records_current_contract(tmp_path):
    data = {
        "q": np.array([[1, 2], [2, 4], [4, 8]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        rank=2,
    )
    analyzer.load_and_preprocess()
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd()
    analyzer.save_results("dmd_contract.hdf5")

    with h5py.File(tmp_path / "dmd_contract.hdf5", "r") as handle:
        assert handle.attrs["dmd_variant"] == "exact_dmd"
        assert handle.attrs["paired_data_contract"] == "raw_shifted_snapshots"
        assert not bool(handle.attrs["uses_mean_subtraction"])
        assert not bool(handle.attrs["uses_spatial_metric_in_regression"])
        assert handle.attrs["mode_ranking"] == "abs_lambda_desc"


# ---------------------------------------------------------------------------
# Helper: generate snapshots from a linear system x_{k+1} = A x_k
# ---------------------------------------------------------------------------


def _make_linear_snapshots(A, x0, n_steps):
    """Return q array of shape (n_steps+1, n_spatial)."""
    snapshots = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = A @ x
        snapshots.append(x.copy())
    return np.array(snapshots)


def _make_rank4_snapshots(n_steps=40):
    """Trajectory of a 4-state, two-frequency linear system — genuinely rank 4."""

    def _rot_scale(theta, radius):
        c, s = np.cos(theta), np.sin(theta)
        return radius * np.array([[c, -s], [s, c]])

    A = np.block(
        [
            [_rot_scale(0.3, 0.98), np.zeros((2, 2))],
            [np.zeros((2, 2)), _rot_scale(0.9, 0.9)],
        ]
    )
    # Excite both 2-state blocks so the trajectory spans all four dimensions.
    x0 = np.array([1.0, 0.0, 1.0, 0.0])
    return _make_linear_snapshots(A, x0, n_steps)


def _make_analyzer(q, n_modes_save=None, rank=None):
    """Shorthand to build a DMDAnalyzer from a snapshot array."""
    n_spatial = q.shape[1]
    if n_modes_save is None:
        n_modes_save = n_spatial
    # rank is required on DMDAnalyzer; default matches the pre-migration coupling.
    if rank is None:
        rank = n_modes_save
    data = {
        "q": q,
        "x": np.arange(n_spatial, dtype=float),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_spatial,
        "Ny": 1,
        "Ns": q.shape[0],
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes_save,
        rank=rank,
    )
    analyzer.load_and_preprocess()
    return analyzer


def _eig_set_err(got, want):
    """Max distance after matching spectra as sorted sets (order-independent)."""
    got = np.asarray(got)
    want = np.asarray(want)
    if got.size != want.size:
        return np.inf
    return float(np.max(np.abs(np.sort_complex(got) - np.sort_complex(want))))


# ---------------------------------------------------------------------------
# Delay embedding
# ---------------------------------------------------------------------------


def test_delay_embed_shape():
    """Smoke test: asserts execution and artifact only, not numerical values.

    Shape of the delay-embedded Hankel only; value content is covered by
    test_delay_embed_values and test_delay_embed_d1_identity.
    """
    rng = np.random.default_rng(11)
    X = rng.standard_normal((3, 10))
    d = 4
    Xd = _delay_embed(X, d)
    assert Xd.shape == (3 * d, 10 - d + 1)


def test_delay_embed_d1_identity():
    """With d=1, _delay_embed returns the input unchanged.

    Already adequate: array_equal against the input pins values, not just shape.
    """
    rng = np.random.default_rng(12)
    X = rng.standard_normal((5, 8))
    Xd = _delay_embed(X, 1)
    np.testing.assert_array_equal(Xd, X)


def test_delay_embed_values():
    """Verify the stacking order of _delay_embed."""
    # 2 spatial points, 5 time steps
    X = np.arange(10).reshape(2, 5).astype(float)
    Xd = _delay_embed(X, 3)
    # Row block 0: X[:, 0:3], block 1: X[:, 1:4], block 2: X[:, 2:5]
    expected = np.vstack([X[:, 0:3], X[:, 1:4], X[:, 2:5]])
    np.testing.assert_array_equal(Xd, expected)


# ---------------------------------------------------------------------------
# Omega (continuous-time eigenvalues)
# ---------------------------------------------------------------------------


def test_omega_returned():
    """perform_dmd recovers the continuous-time spectrum of the generating map.

    The snapshots are generated by iterating a KNOWN matrix A, so the true
    continuous-time eigenvalues are log(eig(A))/dt computed from A itself --
    never from the analyzer's own output. That makes this an independent check
    of both the discrete spectrum and its conversion, not a restatement of the
    implementation's formula. dt != 1 so the /dt scaling is not a no-op.
    """
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 30)
    analyzer = _make_analyzer(q)
    dt = 0.25
    analyzer.data["dt"] = dt
    analyzer.perform_dmd()

    assert analyzer.omega.size == analyzer.eigenvalues.size
    expected_omega = np.sort_complex(np.log(np.linalg.eigvals(A).astype(complex)) / dt)
    np.testing.assert_allclose(
        np.sort_complex(np.asarray(analyzer.omega).astype(complex)),
        expected_omega,
        rtol=1e-8,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Default backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.characterization
def test_default_args_match_original():
    """Default perform_dmd() gives identical eigenvalues to the reference helper."""
    data_q = np.array(
        [[10.0, 1.0], [11.0, 2.0], [13.0, 4.0], [16.0, 8.0], [20.0, 16.0]],
        dtype=float,
    )
    # Data is 2-D spatial; full numerical rank equals n_modes_save=2.
    expected = _exact_dmd_eigenvalues(data_q, n_modes_save=2, rank=2)

    analyzer = _make_analyzer(data_q, n_modes_save=2)
    analyzer.perform_dmd()  # defaults: method="ls", embedding_dim=1
    np.testing.assert_allclose(analyzer.eigenvalues, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# TLS-DMD
# ---------------------------------------------------------------------------


def test_tls_dmd_clean_data():
    """On noise-free data, TLS eigenvalues recover the true system eigenvalues."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    true_eigvals = np.sort(np.linalg.eigvals(A))
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 50)

    analyzer = _make_analyzer(q)
    analyzer.perform_dmd(method="tls")

    recovered = np.sort(analyzer.eigenvalues)
    np.testing.assert_allclose(recovered, true_eigvals, atol=1e-8)


def test_tls_noise_robustness():
    """TLS-DMD eigenvalues should be at least as close to ground truth as LS under noise."""
    rng = np.random.default_rng(42)
    A = np.array([[0.95, 0.05], [-0.05, 0.90]])
    true_eigvals = np.sort(np.linalg.eigvals(A))
    q_clean = _make_linear_snapshots(A, np.array([1.0, 0.5]), 80)

    noise_level = 0.05 * np.std(q_clean)
    q_noisy = q_clean + rng.normal(0, noise_level, q_clean.shape)

    # LS
    analyzer_ls = _make_analyzer(q_noisy)
    analyzer_ls.perform_dmd(method="ls")
    err_ls = np.linalg.norm(np.sort(analyzer_ls.eigenvalues) - true_eigvals)

    # TLS
    analyzer_tls = _make_analyzer(q_noisy)
    analyzer_tls.perform_dmd(method="tls")
    err_tls = np.linalg.norm(np.sort(analyzer_tls.eigenvalues) - true_eigvals)

    # TLS should not be worse (allow small tolerance for edge cases)
    assert err_tls <= err_ls * 1.1, f"TLS error {err_tls:.6e} exceeded LS error {err_ls:.6e} by more than 10%"


# ---------------------------------------------------------------------------
# Delay-embedded DMD
# ---------------------------------------------------------------------------


def test_dmd_with_embedding_dim():
    """DMD with delay embedding runs and returns valid eigenvalues."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd(embedding_dim=3)

    # Order-2 linear system → numerical rank 2 even after delay lift.
    assert analyzer.effective_rank == 2
    assert analyzer.eigenvalues.size == 2
    assert analyzer.modes.shape[0] == 2 * 3  # n_spatial * embedding_dim
    assert analyzer.omega.size == 2
    # All eigenvalues should be finite
    assert np.all(np.isfinite(analyzer.eigenvalues))


def test_hodmd_eigenvalue_oracle():
    """HODMD / TLS-HODMD recover eigvals(A) as sorted sets, both methods."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 50)
    true = np.linalg.eigvals(A)
    # For a diagonalizable A, |dlambda| <= cond(V) * ||dA||. Pipeline round-off
    # grows like the working dimension n_spatial * embedding_dim, so
    #   tol = C * cond(V) * eps * n_spatial * embedding_dim
    # with C a small integer. On this fixture (ls/tls, embedding_dim 2/3/5) the worst
    # observed set_err is 7.1e-16, and the tightest bound is the embedding_dim=2 one,
    # so C=4 leaves ~8.6x margin. Jittering the snapshots at ulp scale -- a
    # stand-in for a different BLAS summation order on another platform --
    # pushed the worst case only to 9.1e-16, still 15% of that bound.
    _, V = np.linalg.eig(A)
    n_spatial = A.shape[0]
    eps = np.finfo(float).eps
    C = 4
    # Both methods at embedding_dim=2, embedding_dim=3, embedding_dim=5 (nothing truncated).
    for method in ("ls", "tls"):
        for dim in (2, 3, 5):
            analyzer = _make_analyzer(q, n_modes_save=2, rank=2)
            analyzer.perform_dmd(method=method, embedding_dim=dim)
            tol = C * np.linalg.cond(V) * eps * n_spatial * dim
            np.testing.assert_allclose(
                np.sort_complex(analyzer.eigenvalues),
                np.sort_complex(true),
                rtol=0.0,
                atol=tol,
            )


def test_hodmd_rank_staircase():
    """Scalar observation of two oscillators: Hankel rank is min(d, 4)."""
    r1, w1 = 0.97, 0.3
    r2, w2 = 0.90, 1.1
    t = np.arange(80)
    y = (r1**t) * np.cos(w1 * t + 0.2) + 0.7 * (r2**t) * np.cos(w2 * t - 0.5)
    q = y.reshape(-1, 1)
    true4 = np.array(
        [
            r1 * np.exp(1j * w1),
            r1 * np.exp(-1j * w1),
            r2 * np.exp(1j * w2),
            r2 * np.exp(-1j * w2),
        ]
    )
    ranks = []
    # rank=4 requested at every depth so the cap is the Hankel's, not the argument.
    for d in (1, 2, 3, 4, 6):
        analyzer = _make_analyzer(q, n_modes_save=4, rank=4)
        analyzer.perform_dmd(embedding_dim=d)
        assert analyzer.effective_rank == min(d, 4)
        ranks.append(analyzer.effective_rank)
        rec = np.asarray(analyzer.eigenvalues)
        err = _eig_set_err(rec, true4)
        if d < 4:
            # Four modes are unreachable: the Hankel has only d rows.
            # `err == inf` on its own would only restate the rank assertion
            # above, since the helper returns inf whenever the counts differ.
            # It would still pass for an implementation that computed the true
            # 4-set and then truncated the output. So also check the poles that
            # DO come back are genuinely wrong, by their distance to the
            # nearest true pole: measured 0.22-0.32 here, where a truncating
            # implementation would give ~0.
            assert err == np.inf
            worst = max(float(np.min(np.abs(p - true4))) for p in rec)
            assert worst > 0.05
        else:
            # True 4-mode spectrum is recovered once d >= 4 (flatten at d=6).
            # 1e-10 is a recovered/not-recovered discriminator, not an accuracy
            # bound: the measured error is ~2e-15, and the accuracy claim lives
            # in test_hodmd_eigenvalue_oracle.
            assert err < 1e-10
    assert ranks == [1, 2, 3, 4, 4]


def test_hodmd_tls_vs_ls_median():
    """On noisy data, median TLS eigenvalue error is below median LS error."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    true = np.linalg.eigvals(A)
    # 60 snapshots, matching the measured median-over-seeds fixture.
    q_clean = _make_linear_snapshots(A, np.array([1.0, 0.5]), 59)
    noise_std = 3e-2 * np.max(np.abs(q_clean))
    err_ls = []
    err_tls = []
    for seed in range(25):
        rng = np.random.default_rng(seed)
        q_noisy = q_clean + rng.normal(0.0, noise_std, q_clean.shape)
        analyzer_ls = _make_analyzer(q_noisy, n_modes_save=2, rank=2)
        analyzer_ls.perform_dmd(method="ls", embedding_dim=1)
        analyzer_tls = _make_analyzer(q_noisy, n_modes_save=2, rank=2)
        analyzer_tls.perform_dmd(method="tls", embedding_dim=1)
        err_ls.append(_eig_set_err(analyzer_ls.eigenvalues, true))
        err_tls.append(_eig_set_err(analyzer_tls.eigenvalues, true))
    assert np.median(err_tls) < np.median(err_ls)


def test_tls_hodmd_differs_from_ls_at_depth():
    """TLS-HODMD is its own computation, not LS under a different name.

    Nothing else here can see this. On clean, fully observed data LS and TLS
    agree to ~1e-16, so the oracle stays green even if the TLS branch silently
    became LS whenever ``embedding_dim > 1``; the staircase runs the default LS path;
    and the median comparison runs ``embedding_dim=1``. Noise at depth is what
    separates the two formulas.
    """
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q_clean = _make_linear_snapshots(A, np.array([1.0, 0.5]), 59)
    noise_std = 3e-2 * np.max(np.abs(q_clean))
    for seed in range(5):
        rng = np.random.default_rng(seed)
        q_noisy = q_clean + rng.normal(0.0, noise_std, q_clean.shape)
        analyzer_ls = _make_analyzer(q_noisy, n_modes_save=2, rank=2)
        analyzer_ls.perform_dmd(method="ls", embedding_dim=3)
        analyzer_tls = _make_analyzer(q_noisy, n_modes_save=2, rank=2)
        analyzer_tls.perform_dmd(method="tls", embedding_dim=3)
        # Smallest separation measured over 25 seeds at embedding_dim=3 is 4.3e-3, so
        # 1e-3 leaves margin while still being far above any round-off.
        gap = _eig_set_err(analyzer_tls.eigenvalues, analyzer_ls.eigenvalues)
        assert gap > 1e-3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    A = np.eye(2) * 0.9
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 10)
    analyzer = _make_analyzer(q)
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.perform_dmd(method="bogus")


def test_invalid_embedding_dim_raises():
    A = np.eye(2) * 0.9
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 10)
    analyzer = _make_analyzer(q)
    with pytest.raises(ValueError, match="embedding_dim must be >= 1"):
        analyzer.perform_dmd(embedding_dim=0)


# ---------------------------------------------------------------------------
# Metadata reflects variant
# ---------------------------------------------------------------------------


def test_metadata_tls_embedding_dim(tmp_path):
    """Metadata should reflect TLS + delay embedding settings."""
    q = _make_rank4_snapshots(30)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="tls", embedding_dim=3)
    assert analyzer.effective_rank == 4

    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "delay_embedded_tls_dmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_embedding_dim"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_load_results_restores_variant_metadata(tmp_path):
    """Saved DMD variant metadata should survive a load/save round-trip."""
    q = _make_rank4_snapshots(30)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.results_dir = tmp_path
    analyzer.perform_dmd(method="tls", embedding_dim=3)
    assert analyzer.effective_rank == 4
    analyzer.save_results("dmd_variant_roundtrip.hdf5")

    reloaded = _make_analyzer(q, n_modes_save=4)
    reloaded.results_dir = tmp_path
    reloaded.load_results("dmd_variant_roundtrip.hdf5")

    meta = reloaded._get_algorithm_metadata()
    assert meta["dmd_variant"] == "delay_embedded_tls_dmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_embedding_dim"] == 3


def test_dmd_save_load_roundtrip_arrays(tmp_path):
    """DMD save → load restores eigenvalues, modes, coefficients, amplitudes exactly."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 30)

    analyzer = _make_analyzer(q, n_modes_save=2)
    analyzer.results_dir = tmp_path
    analyzer.perform_dmd()
    analyzer.save_results("dmd_array_roundtrip.hdf5")

    reloaded = _make_analyzer(q, n_modes_save=2)
    reloaded.results_dir = tmp_path
    reloaded.load_results("dmd_array_roundtrip.hdf5")

    np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
    np.testing.assert_array_equal(reloaded.modes, analyzer.modes)
    np.testing.assert_array_equal(reloaded.time_coefficients, analyzer.time_coefficients)
    np.testing.assert_array_equal(reloaded.amplitudes, analyzer.amplitudes)


# ---------------------------------------------------------------------------
# HODMD / TLS-HODMD named variant metadata
# ---------------------------------------------------------------------------


def test_hodmd_named_variant_metadata():
    """perform_dmd with named_variant='hodmd' sets the correct metadata."""
    q = _make_rank4_snapshots(40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="ls", embedding_dim=3, named_variant="hodmd")
    assert analyzer.effective_rank == 4

    assert analyzer._dmd_named_variant == "hodmd"
    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "hodmd"
    assert meta["dmd_named_variant"] == "hodmd"
    assert meta["dmd_method"] == "ls"
    assert meta["dmd_embedding_dim"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_tls_hodmd_named_variant_metadata():
    """perform_dmd with named_variant='tls_hodmd' sets the correct metadata."""
    q = _make_rank4_snapshots(40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="tls", embedding_dim=3, named_variant="tls_hodmd")
    assert analyzer.effective_rank == 4

    assert analyzer._dmd_named_variant == "tls_hodmd"
    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "tls_hodmd"
    assert meta["dmd_named_variant"] == "tls_hodmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_embedding_dim"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_hodmd_save_load_roundtrip(tmp_path):
    """HODMD named variant survives a save/load round-trip."""
    q = _make_rank4_snapshots(40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.results_dir = tmp_path
    analyzer.perform_dmd(method="ls", embedding_dim=3, named_variant="hodmd")
    assert analyzer.effective_rank == 4
    analyzer.save_results("hodmd_roundtrip.hdf5")

    reloaded = _make_analyzer(q, n_modes_save=4)
    reloaded.results_dir = tmp_path
    reloaded.load_results("hodmd_roundtrip.hdf5")

    assert reloaded._dmd_named_variant == "hodmd"
    meta = reloaded._get_algorithm_metadata()
    assert meta["dmd_variant"] == "hodmd"
    assert meta["dmd_method"] == "ls"
    assert meta["dmd_embedding_dim"] == 3


def test_hodmd_plot_modes_uses_2d_slice(monkeypatch, tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values.

    Delay-embedded DMD modes should be visualized as 2D maps, not 1D lines.
    """
    rng = np.random.default_rng(13)
    nx, ny = 4, 3
    n_space = nx * ny
    q = rng.standard_normal((40, n_space))
    data = {
        "q": q,
        "x": np.arange(nx, dtype=float),
        "y": np.arange(ny, dtype=float),
        "dt": 1.0,
        "Nx": nx,
        "Ny": ny,
        "Ns": q.shape[0],
        "metadata": {"var_name": "u"},
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        rank=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd(method="ls", embedding_dim=2, named_variant="hodmd")

    line_calls = {"count": 0}
    orig_plot = matplotlib.axes.Axes.plot

    def plot_wrapper(self, *args, **kwargs):
        line_calls["count"] += 1
        return orig_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot_wrapper)
    analyzer.plot_modes(plot_n_modes=2, modes_per_fig=2)

    assert line_calls["count"] == 0
    assert (tmp_path / "dummy_dmd_modes_1_to_2_u.png").exists()


def _rank3_kicked_snapshots(scale=1.0, kick=1e-3, seed=0):
    """Exactly rank-3 sequence with a kick only on the last snapshot (X2 ∉ range(X1))."""
    rng = np.random.default_rng(seed)
    nx, ny, ns, dt = 12, 10, 40, 0.1
    t = np.arange(ns) * dt
    p = rng.standard_normal((nx * ny, 3))
    a = np.column_stack(
        [
            np.cos(2 * np.pi * 0.5 * t),
            np.sin(2 * np.pi * 1.1 * t),
            np.exp(-0.3 * t),
        ]
    )
    q = a @ p.T
    q[-1] += kick * rng.standard_normal(nx * ny)
    return q * scale, nx, ny, dt


def test_dmd_rank_deficient_floors_singular_values():
    """Rank-3 data + final-snapshot kick must not amplify noise via tiny s_r."""
    q, nx, ny, dt = _rank3_kicked_snapshots()
    data = {
        "q": q,
        "x": np.arange(nx, dtype=float),
        "y": np.arange(ny, dtype=float),
        "dt": dt,
        "Nx": nx,
        "Ny": ny,
        "Ns": q.shape[0],
    }
    analyzer = DMDAnalyzer(
        file_path="dummy", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=10, rank=10
    )
    analyzer.load_and_preprocess()
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd()

    assert analyzer.effective_rank == 3
    assert analyzer.modes.shape[1] == 3
    assert float(np.abs(analyzer.eigenvalues).max()) <= 10.0
    assert float(np.abs(analyzer.modes).max()) <= 1e3
    assert np.all(np.isfinite(analyzer.eigenvalues))
    assert np.all(np.isfinite(analyzer.modes))


def test_dmd_effective_rank_scale_invariant():
    """Same dynamics at 1e-8 and 1e+8 must report the same effective rank."""
    ranks = []
    for scale in (1e-8, 1.0, 1e8):
        q, nx, ny, dt = _rank3_kicked_snapshots(scale=scale)
        data = {
            "q": q,
            "x": np.arange(nx, dtype=float),
            "y": np.arange(ny, dtype=float),
            "dt": dt,
            "Nx": nx,
            "Ny": ny,
            "Ns": q.shape[0],
        }
        analyzer = DMDAnalyzer(
            file_path="dummy", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=10, rank=10
        )
        analyzer.load_and_preprocess()
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()
        ranks.append(analyzer.effective_rank)
        assert float(np.abs(analyzer.eigenvalues).max()) <= 10.0

    assert ranks == [3, 3, 3]


def test_dmd_full_rank_path_silent_and_untruncated():
    """Well-conditioned data at an explicit rank keeps every mode and does not warn."""
    rng = np.random.default_rng(7)
    nsp = 120
    tt = np.arange(60) * 0.1
    q = np.column_stack([np.cos(2 * np.pi * f * tt) for f in (0.3, 0.7, 1.3, 1.9, 2.4)]) @ rng.standard_normal((5, nsp))
    q = q + 1e-2 * rng.standard_normal((60, nsp))
    data = {
        "q": q,
        "x": np.arange(12, dtype=float),
        "y": np.arange(10, dtype=float),
        "dt": 0.1,
        "Nx": 12,
        "Ny": 10,
        "Ns": q.shape[0],
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=5,
        rank=5,
    )
    analyzer.load_and_preprocess()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer.perform_dmd()
    rank_warnings = [w for w in caught if "effective rank" in str(w.message)]
    assert rank_warnings == []
    assert analyzer.effective_rank == 5
    assert analyzer.modes.shape[1] == 5


def test_dmd_degenerate_all_zero_returns_empty_without_raising():
    """An all-zero field has no usable spectrum: warn and return empty, do not crash.

    The relative test always keeps s[0] (s[0] > rcond * s[0] reduces to 1 > rcond),
    so effective_rank reaches 0 only when the spectrum itself is unusable. Before the
    guard was ordered correctly this divided by a zero s[0] and surfaced as
    ``LinAlgError: Array must not contain infs or NaNs`` out of np.linalg.eig.
    """
    q = np.zeros((20, 60))
    data = {
        "q": q,
        "x": np.arange(10, dtype=float),
        "y": np.arange(6, dtype=float),
        "dt": 0.1,
        "Nx": 10,
        "Ny": 6,
        "Ns": q.shape[0],
    }
    analyzer = DMDAnalyzer(
        file_path="dummy", data_loader=lambda _: data, spatial_weight_type="uniform", n_modes_save=5, rank=5
    )
    analyzer.load_and_preprocess()
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd()

    assert analyzer.effective_rank == 0
    assert analyzer.eigenvalues.size == 0
    assert analyzer.modes.size == 0
    assert analyzer.time_coefficients.size == 0


# ---------------------------------------------------------------------------
# rank vs n_modes_save (decoupled)
# ---------------------------------------------------------------------------


def test_n_modes_save_does_not_change_eigenvalues():
    """With rank fixed, changing only n_modes_save leaves eigenvalues bit-identical."""
    rng = np.random.default_rng(11)
    n_space, n_time = 80, 30
    phi, _ = np.linalg.qr(rng.standard_normal((n_space, 6)))
    lam = 0.95 - 0.02 * np.arange(6)
    t = np.arange(n_time)
    q = (phi @ (lam[:, None] ** t[None, :])).T
    q = q + 1e-4 * rng.standard_normal(q.shape)

    a = _make_analyzer(q, n_modes_save=4, rank=6)
    b = _make_analyzer(q, n_modes_save=12, rank=6)
    # Explicit rank: no DeprecationWarning; spectrum supports rank 6, so silent.
    a.perform_dmd()
    b.perform_dmd()

    assert a.effective_rank == b.effective_rank == 6
    k = min(a.eigenvalues.size, b.eigenvalues.size)
    assert k > 0
    assert np.array_equal(a.eigenvalues[:k], b.eigenvalues[:k])
    assert a.eigenvalues.size <= 4
    assert b.eigenvalues.size >= a.eigenvalues.size


def test_rank_parameter_changes_eigenvalues():
    """Changing rank must change the reduced operator / eigenvalues."""
    rng = np.random.default_rng(12)
    n_space, n_time = 80, 30
    phi, _ = np.linalg.qr(rng.standard_normal((n_space, 6)))
    lam = 0.95 - 0.02 * np.arange(6)
    t = np.arange(n_time)
    q = (phi @ (lam[:, None] ** t[None, :])).T
    q = q + 1e-4 * rng.standard_normal(q.shape)

    lo = _make_analyzer(q, n_modes_save=20, rank=2)
    hi = _make_analyzer(q, n_modes_save=20, rank=6)
    # Explicit rank; both ranks sit inside the numerical spectrum → silent.
    lo.perform_dmd()
    hi.perform_dmd()

    assert lo.effective_rank == 2
    assert hi.effective_rank == 6
    k = min(lo.eigenvalues.size, hi.eigenvalues.size)
    assert not np.array_equal(lo.eigenvalues[:k], hi.eigenvalues[:k])


def test_rank_required_refuses_none():
    """Omitting rank raises ValueError naming the allowed alternatives."""
    data = {
        "q": np.array([[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    with pytest.raises(ValueError, match=r"rank.*(svht|energy)"):
        DMDAnalyzer(
            file_path="dummy",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=2,
        )

    # Explicit rank past n_modes_save is allowed.
    rng = np.random.default_rng(13)
    n_space, n_time = 100, 40
    q = rng.standard_normal((n_time, n_space))
    b = _make_analyzer(q, n_modes_save=5, rank=12)
    b.perform_dmd()
    assert b.effective_rank == 12
    assert b.eigenvalues.size == 5


def test_svht_lambda_beta_dependent():
    from openmodalpy.dmd import svht_lambda

    # Gavish–Donoho formula: lambda(1)=4/sqrt(3); lambda decreases toward sqrt(2).
    # Compare against independent evaluation of the same closed form (1e-9).
    def ref(beta):
        beta = float(beta)
        return float(
            np.sqrt(2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + np.sqrt(beta * beta + 14.0 * beta + 1.0)))
        )

    for beta in (1.0, 0.5, 0.1, 0.01, 0.001):
        got = float(svht_lambda(beta))
        want = ref(beta)
        assert abs(got - want) < 1e-9, f"svht_lambda({beta}) = {got}, want {want}"

    # The loop above compares the implementation against a COPY of the same closed
    # form, so on its own it cannot detect a wrong formula. These two anchors are
    # independent published values, and they are what actually pins it:
    #   lambda(1) = 4/sqrt(3) exactly (the square-matrix case)
    #   lambda(beta) -> sqrt(2)  as beta -> 0
    assert abs(svht_lambda(1.0) - 4.0 / np.sqrt(3.0)) < 1e-12
    assert abs(svht_lambda(1e-12) - np.sqrt(2.0)) < 1e-6
    # 4/sqrt(3) is lambda(1), not a universal constant. Fluid snapshot matrices sit
    # near beta ~ 0.01, where using 2.309401 would set the threshold ~61% too high.
    assert abs(svht_lambda(0.01) - 4.0 / np.sqrt(3.0)) > 0.5
    assert svht_lambda(0.001) < svht_lambda(0.1) < svht_lambda(0.5) < svht_lambda(1.0)


def test_energy_rank_criterion():
    """rank='energy' keeps the smallest r whose cumulative s^2 meets the fraction."""
    rng = np.random.default_rng(14)
    # Plant a steep spectrum so energy_fraction=0.99 is well below full rank.
    n_space, n_time = 50, 20
    phi, _ = np.linalg.qr(rng.standard_normal((n_space, 8)))
    lam = 0.99 ** np.arange(8)
    amp = 10.0 ** (-0.5 * np.arange(8))
    t = np.arange(n_time)
    q = ((phi * amp) @ (lam[:, None] ** t[None, :])).T

    # Explicit large rank for the untruncated reference (not the deprecated default).
    full = _make_analyzer(q, n_modes_save=20, rank=20)
    energy = _make_analyzer(q, n_modes_save=20, rank="energy")
    energy.energy_fraction = 0.99
    # full asks for 20 but the plant spectrum only supports ~8 → RuntimeWarning.
    with pytest.warns(RuntimeWarning, match="effective rank"):
        full.perform_dmd()
    # energy criterion stops well inside the spectrum → silent.
    energy.perform_dmd()

    assert energy.effective_rank < full.effective_rank
    assert energy.effective_rank >= 1

    # Pin the rank by the criterion's DEFINING property rather than by re-running
    # searchsorted: the kept set must clear the fraction, and dropping one more mode
    # must fall below it. That is sufficiency plus minimality, which a copy of the
    # implementation's own arithmetic would not independently establish.
    s = np.linalg.svd(q.T[:, :-1], compute_uv=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    r = energy.effective_rank
    assert cum[r - 1] >= 0.99, f"rank {r} retains only {cum[r - 1]:.6f} of the energy, below the 0.99 asked for"
    if r > 1:
        assert cum[r - 2] < 0.99, f"rank {r} is not minimal: {r - 1} modes already retain {cum[r - 2]:.6f}"


def test_svht_flat_spectrum_returns_zero_effective_rank():
    """On a flat singular spectrum, SVHT returns effective_rank 0 with empty modes.

    Documented outcome when τ = ω(β)·median(s) exceeds σ₁ (see DOC.md, "svht"
    row): warn, leave eigenvalues and modes empty, do not invent a mode.

    The fixture is a flat spectrum, not a draw of i.i.d. noise, and that is
    deliberate. A flat spectrum makes median(s) == σ₁, so τ = ω(β)·σ₁ > σ₁ holds
    for every β and every size by construction — measured τ/σ₁ = 2.78, 2.47 and
    2.83 at 20×20, 40×30 and 64×64. It held under the old λ(β) coefficient too
    (2.28, 2.14, 2.30), which is why this fixture survived that fix unchanged:
    it does not depend on where the coefficient sits, only on it exceeding 1.
    A noise draw would pin the same outcome on a probability instead — see
    test_svht_pure_noise_returns_zero_effective_rank for that property and the
    sizes at which it is safe to assert.
    """
    rng = np.random.default_rng(0)
    n_space, n_time = 20, 20
    # Random orthogonal factors, unit singular values: white, no coherent signal.
    u, _, vt = np.linalg.svd(rng.standard_normal((n_space, n_time)), full_matrices=False)
    q = (u @ vt).T
    data = {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": 0.1,
        "Nx": n_space,
        "Ny": 1,
        "Ns": n_time,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=5,
        rank="svht",
    )
    analyzer.load_and_preprocess()
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer.perform_dmd()

    assert analyzer.effective_rank == 0
    assert analyzer.eigenvalues.size == 0
    assert analyzer.modes.size == 0
    assert analyzer.time_coefficients.size == 0


def test_svht_omega_matches_published_approximation():
    """svht_omega matches G&D's published cubic approximation within 1%."""
    from openmodalpy.dmd import svht_omega

    def approx(beta):
        b = float(beta)
        return 0.56 * b**3 - 0.95 * b**2 + 1.82 * b + 1.43

    for beta in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        got = float(svht_omega(beta))
        want = approx(beta)
        rel = abs(got - want) / want
        assert rel < 0.01, f"svht_omega({beta}) = {got}, approx = {want}, rel = {rel:.3%}"

    # Published square-matrix anchor: omega(1) ≈ 2.858
    assert abs(svht_omega(1.0) - 2.858) / 2.858 < 0.005


def test_svht_omega_limits_and_monotonic():
    """svht_omega -> sqrt(2) as beta -> 0 and is strictly increasing on (0, 1]."""
    from openmodalpy.dmd import svht_omega

    assert abs(svht_omega(1e-12) - np.sqrt(2.0)) < 1e-6
    betas = (0.001, 0.1, 0.5, 1.0)
    vals = [float(svht_omega(b)) for b in betas]
    assert vals[0] < vals[1] < vals[2] < vals[3]


def test_svht_pure_noise_returns_zero_effective_rank():
    """Pure i.i.d. Gaussian noise through rank='svht' yields effective_rank 0.

    The unknown-noise coefficient omega(beta) sets tau above the bulk edge of
    a pure-noise singular spectrum at realistic matrix sizes. With the known-
    noise lambda used against median(s), the same draws keep 1–3 noise modes.

    Sizes 40 and 100 only. Measured over 500 seeds: rank 0 comes out 100% of
    the time at both, but only 98.8% of the time at 20×20, where the matrix is
    small enough for edge fluctuations to push a single noise mode over tau.
    A fixed seed would hide that, so the small size is left out rather than
    pinned at a size where the claim is a probability instead of a property.
    """
    rng = np.random.default_rng(1)
    for n in (40, 100):
        q = rng.standard_normal((n, n))
        analyzer = _make_analyzer(q, n_modes_save=5, rank="svht")
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()
        assert analyzer.effective_rank == 0, f"pure noise {n}x{n}: effective_rank={analyzer.effective_rank}, want 0"


def test_svht_planted_rank3_signal_kept():
    """A rank-3 signal well above the noise floor is kept by rank='svht'."""
    rng = np.random.default_rng(42)
    n = 100
    # 100 x 100 snapshot matrix: three planted singular values + unit noise.
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    vt, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s_plant = np.zeros(n)
    s_plant[:3] = [100.0, 80.0, 60.0]
    q = (u @ np.diag(s_plant) @ vt.T + rng.standard_normal((n, n))).T
    analyzer = _make_analyzer(q, n_modes_save=10, rank="svht")
    analyzer.perform_dmd()
    assert analyzer.effective_rank == 3


# ---------------------------------------------------------------------------
# Canonical spectrum order (library-owned; must not depend on LAPACK emission)
# ---------------------------------------------------------------------------

_REAL_EIG = np.linalg.eig

# Hand-built linear map: one real pole at 0.95 and a quarter-turn pair ±0.8j.
# Canonical order is |λ| descending, then (Re, Im) ascending in a |λ| tie:
# 0.95, −0.8j, +0.8j. Written out as a literal so the assertion is not a
# helper round-trip.
_QUARTER_TURN_EXPECTED = np.array([0.95 + 0.0j, 0.0 - 0.8j, 0.0 + 0.8j])

# Two distinct conjugate pairs (not the quarter-turn map). Larger |λ| pair is
# weakly excited so |b| order is opposite |λ| order — a second independent
# sort of the companions is then visible.
_TWO_PAIR_EXPECTED = np.array(
    [
        0.98 * np.cos(0.4) - 1j * 0.98 * np.sin(0.4),
        0.98 * np.cos(0.4) + 1j * 0.98 * np.sin(0.4),
        0.85 * np.cos(1.1) - 1j * 0.85 * np.sin(1.1),
        0.85 * np.cos(1.1) + 1j * 0.85 * np.sin(1.1),
    ]
)


def _make_quarter_turn_snapshots(n_steps=40):
    A = np.array(
        [
            [0.95, 0.0, 0.0],
            [0.0, 0.0, -0.8],
            [0.0, 0.8, 0.0],
        ],
        dtype=float,
    )
    return _make_linear_snapshots(A, np.array([1.0, 1.0, 0.0]), n_steps)


def _make_two_pair_snapshots(n_steps=40):
    """Trajectory of two rotation-scaling blocks with |b| opposite |λ|."""

    def _rot_scale(theta, radius):
        c, s = np.cos(theta), np.sin(theta)
        return radius * np.array([[c, -s], [s, c]])

    A = np.block(
        [
            [_rot_scale(0.4, 0.98), np.zeros((2, 2))],
            [np.zeros((2, 2)), _rot_scale(1.1, 0.85)],
        ]
    )
    # Small seed on the larger-|λ| pair so |b| ranks the 0.85 pair first.
    return _make_linear_snapshots(A, np.array([0.05, 0.0, 1.0, 0.0]), n_steps)


def _assert_time_dynamics_match_eigenvalues(eigenvalues, time_coefficients, *, rtol=1e-8, atol=1e-10):
    """time_coefficients[t, k] / time_coefficients[0, k] ≈ eigenvalues[k] ** t."""
    eigs = np.asarray(eigenvalues)
    tc = np.asarray(time_coefficients)
    assert tc.ndim == 2 and tc.shape[1] == eigs.size
    t = np.arange(tc.shape[0])
    for k in range(eigs.size):
        scale = tc[0, k]
        assert abs(scale) > 0.0
        np.testing.assert_allclose(tc[:, k] / scale, eigs[k] ** t, rtol=rtol, atol=atol)


def _eig_as_is(a):
    return _REAL_EIG(a)


def _eig_reversed(a):
    vals, vecs = _REAL_EIG(a)
    return vals[::-1].copy(), vecs[:, ::-1].copy()


def _run_dmd_with_eig(q, eig_fn, monkeypatch, n_modes_save=None, rank=None):
    import openmodalpy.dmd as dmd_mod

    analyzer = _make_analyzer(q, n_modes_save=n_modes_save, rank=rank)
    monkeypatch.setattr(dmd_mod.np.linalg, "eig", eig_fn)
    analyzer.perform_dmd()
    return analyzer


def test_dmd_canonical_order_invariant_under_conjugate_emission_order(monkeypatch):
    """Both conjugate emission orders must return the same column order.

    ``np.linalg.eig`` on a non-Hermitian operator does not define pair order.
    HEAD sorts by ``|λ|`` alone, so reversing the emission swaps the pair and
    this test is RED. After the library owns the (Re, Im) tie-break, both
    runs agree, and eigenvalues / omega / modes / time_coefficients /
    amplitudes stay aligned with each other.
    """
    q = _make_quarter_turn_snapshots()
    forward = _run_dmd_with_eig(q, _eig_as_is, monkeypatch, n_modes_save=3, rank=3)
    reversed_run = _run_dmd_with_eig(q, _eig_reversed, monkeypatch, n_modes_save=3, rank=3)

    np.testing.assert_allclose(forward.eigenvalues, reversed_run.eigenvalues, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(forward.omega, reversed_run.omega, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(forward.modes, reversed_run.modes, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(forward.time_coefficients, reversed_run.time_coefficients, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(forward.amplitudes, reversed_run.amplitudes, rtol=0.0, atol=1e-12)

    # The five arrays are one permutation, not five independent sorts.
    np.testing.assert_allclose(
        forward.omega,
        np.log(forward.eigenvalues.astype(complex)),
        rtol=0.0,
        atol=1e-12,
    )
    assert forward.modes.shape[1] == forward.eigenvalues.size
    assert forward.time_coefficients.shape[1] == forward.eigenvalues.size
    assert forward.amplitudes.shape == forward.eigenvalues.shape
    # Bind companions to the eigenvalue index by the dynamics, not just by
    # emission-invariance and column counts. A second independent sort of
    # time_coefficients (or modes / amplitudes) would break this.
    _assert_time_dynamics_match_eigenvalues(forward.eigenvalues, forward.time_coefficients)
    _assert_time_dynamics_match_eigenvalues(reversed_run.eigenvalues, reversed_run.time_coefficients)
    np.testing.assert_allclose(forward.amplitudes, np.abs(forward.time_coefficients[0]), rtol=0.0, atol=1e-12)
    recon = forward.time_coefficients @ forward.modes.T
    np.testing.assert_allclose(recon, q, rtol=1e-10, atol=1e-10)


def test_dmd_tie_straddling_n_modes_save_keeps_same_set(monkeypatch):
    """A conjugate pair sitting on the n_modes_save cut keeps the same members."""
    q = _make_quarter_turn_snapshots()
    forward = _run_dmd_with_eig(q, _eig_as_is, monkeypatch, n_modes_save=2, rank=3)
    reversed_run = _run_dmd_with_eig(q, _eig_reversed, monkeypatch, n_modes_save=2, rank=3)

    assert forward.eigenvalues.size == 2
    assert reversed_run.eigenvalues.size == 2
    np.testing.assert_allclose(forward.eigenvalues, reversed_run.eigenvalues, rtol=0.0, atol=1e-12)
    # (Re, Im) ascending keeps −0.8j, not +0.8j.
    expected_kept = _QUARTER_TURN_EXPECTED[:2]
    np.testing.assert_allclose(forward.eigenvalues, expected_kept, rtol=0.0, atol=1e-12)


def test_dmd_canonical_band_inside_reorders_outside_keeps_magnitude(monkeypatch):
    """Library band is 1e-12: just inside is one group, just outside is not."""
    import openmodalpy.dmd as dmd_mod

    # Two-state full-rank trajectory so atilde is 2×2.
    q = _make_linear_snapshots(np.diag([0.9, 0.5]), np.array([1.0, 1.0]), 20)
    inside = np.array([1.0 + 0.0j, -(1.0 - 0.5e-12) + 0.0j])
    outside = np.array([1.0 + 0.0j, -(1.0 - 2.0e-12) + 0.0j])

    def _force(spectrum):
        spec = np.asarray(spectrum, dtype=np.complex128)

        def _eig(a):
            vals, vecs = _REAL_EIG(a)
            assert vals.size == spec.size
            return spec.copy(), vecs

        return _eig

    analyzer = _make_analyzer(q, n_modes_save=2, rank=2)
    monkeypatch.setattr(dmd_mod.np.linalg, "eig", _force(inside))
    analyzer.perform_dmd()
    # Same |λ| group: (Re, Im) puts the negative real first.
    assert analyzer.eigenvalues[0].real < analyzer.eigenvalues[1].real

    analyzer = _make_analyzer(q, n_modes_save=2, rank=2)
    monkeypatch.setattr(dmd_mod.np.linalg, "eig", _force(outside))
    analyzer.perform_dmd()
    # Distinct magnitudes: larger |λ| (positive real) stays first.
    assert analyzer.eigenvalues[0].real > 0.0
    assert analyzer.eigenvalues[0].real > analyzer.eigenvalues[1].real


def test_dmd_eigenvalues_match_handwritten_literal_order():
    """Analyzer output equals a hand-written expected vector, not a helper echo."""
    analyzer = _make_analyzer(_make_quarter_turn_snapshots(), n_modes_save=3, rank=3)
    analyzer.perform_dmd()
    np.testing.assert_allclose(analyzer.eigenvalues, _QUARTER_TURN_EXPECTED, rtol=0.0, atol=1e-12)


def test_dmd_two_pair_spectrum_binds_companions_by_time_dynamics():
    """A non-quarter-turn spectrum: companions follow λ**t, not |b| or energy.

    Two distinct conjugate pairs, weakly exciting the larger-|λ| pair so |b|
    order is opposite canonical order. Sorting time_coefficients / modes /
    amplitudes by |b| (or by column energy, when that key differs) must break
    the dynamics relation.
    """
    q = _make_two_pair_snapshots()
    analyzer = _make_analyzer(q, n_modes_save=4, rank=4)
    analyzer.perform_dmd()

    np.testing.assert_allclose(analyzer.eigenvalues, _TWO_PAIR_EXPECTED, rtol=0.0, atol=1e-12)
    _assert_time_dynamics_match_eigenvalues(analyzer.eigenvalues, analyzer.time_coefficients)
    np.testing.assert_allclose(analyzer.amplitudes, np.abs(analyzer.time_coefficients[0]), rtol=0.0, atol=1e-12)
    recon = analyzer.time_coefficients @ analyzer.modes.T
    np.testing.assert_allclose(recon, q, rtol=1e-10, atol=1e-10)

    energy = np.sum(np.abs(analyzer.modes) ** 2, axis=0)
    saw_second_sort = False
    t = np.arange(analyzer.time_coefficients.shape[0])
    for name, key in (("|b|", analyzer.amplitudes), ("column energy", energy)):
        perm = np.argsort(-np.asarray(key))
        if np.array_equal(perm, np.arange(perm.size)):
            continue
        saw_second_sort = True
        tc_perm = analyzer.time_coefficients[:, perm]
        holds = True
        for k in range(analyzer.eigenvalues.size):
            ratio = tc_perm[:, k] / tc_perm[0, k]
            if not np.allclose(ratio, analyzer.eigenvalues[k] ** t, rtol=1e-8, atol=1e-10):
                holds = False
                break
        assert not holds, f"sorting time_coefficients by {name} still satisfied λ**t"
        recon_perm = analyzer.time_coefficients @ analyzer.modes[:, perm].T
        assert not np.allclose(recon_perm, q, rtol=1e-6, atol=1e-6), (
            f"sorting modes by {name} still reconstructed the snapshots"
        )
        assert not np.allclose(
            analyzer.amplitudes[perm],
            np.abs(analyzer.time_coefficients[0]),
            rtol=1e-8,
            atol=1e-8,
        ), f"sorting amplitudes by {name} still matched |time_coefficients[0]|"
    assert saw_second_sort, "need a spectrum where |b| or column energy differs from canonical order"


def test_dmd_embedding_dim_parameter_accepts_new_name():
    """Check perform_dmd(embedding_dim=d) lifts modes to n_spatial * d rows."""
    q = _make_rank4_snapshots(20)
    n_spatial = q.shape[1]
    analyzer = _make_analyzer(q, n_modes_save=4, rank=4)
    analyzer.perform_dmd(embedding_dim=2)
    assert analyzer._dmd_embedding_dim == 2
    assert analyzer.modes.shape[0] == n_spatial * 2
    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_embedding_dim"] == 2
    assert meta["lift_kind"] == "delay_embedding"


def test_dmd_old_keyword_raises_typeerror():
    """Check perform_dmd rejects the old keyword with TypeError."""
    A = np.eye(2) * 0.9
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 10)
    analyzer = _make_analyzer(q, n_modes_save=2, rank=2)
    with pytest.raises(TypeError, match="delays"):
        analyzer.perform_dmd(delays=1)


def test_embedding_dim_one_accepted_by_dmd_rejected_by_stpod(tmp_path):
    """Check embedding_dim=1 is DMD's no-lift default and an ST-POD error."""
    from openmodalpy import STPODAnalyzer

    q = _make_rank4_snapshots(10)
    n_spatial = q.shape[1]
    dmd = _make_analyzer(q, n_modes_save=2, rank=2)
    dmd.perform_dmd(embedding_dim=1)
    assert dmd.modes.shape[0] == n_spatial
    assert dmd._get_algorithm_metadata()["lift_kind"] == "identity_paired_snapshots"

    data = {
        "q": np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]], dtype=float).T,
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 3,
        "Ny": 1,
        "Ns": 3,
    }
    stpod = STPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        embedding_dim=1,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    stpod.load_and_preprocess()
    with pytest.raises(ValueError, match="embedding_dim must be >= 2"):
        stpod.perform_stpod()
