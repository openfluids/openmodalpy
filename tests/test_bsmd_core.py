import logging
import warnings

import h5py
import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer
from openmodalpy.bsmd import ALL_TRIADS


def _make_analyzer(
    tmp_path,
    triads,
    nfft=4,
    Ns=10,
    Nspace=4,
    use_static=True,
    max_qhat_gb=None,
    use_parallel=False,
    file_path="dummy.h5",
):
    """Helper to build a BSMDAnalyzer with synthetic data."""
    rng = np.random.default_rng(20)
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    data = {
        "q": rng.standard_normal((Ns, Nspace)),
        "x": np.linspace(0, 1, Nx),
        "y": np.linspace(0, 1, Ny),
        "dt": 1.0,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    kwargs = {}
    if max_qhat_gb is not None:
        kwargs["max_qhat_gb"] = max_qhat_gb
    analyzer = BSMDAnalyzer(
        file_path=file_path,
        nfft=nfft,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=use_static,
        static_triads=triads,
        use_parallel=use_parallel,
        **kwargs,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


@pytest.mark.characterization
def test_static_bsmd_core_small(tmp_path):
    """Single zero-frequency triad: shapes plus a re-derived dominant eigenpair.

    Characterization, not an independent physical oracle: it re-builds the
    bispectral matrix C from qhat with the same formula the library uses
    (see test_compute_single_triad_matches_dominant_eigenpair_shortcut) and
    checks the analyzer's eigenvalue and modes against its dominant eigenpair.
    Arbitrary finite arrays of the right shape would still fail.
    """
    from tests.reference_helpers import canonicalize_reference

    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)])
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert analyzer.modes1.shape[0] == 1
    assert analyzer.modes1.shape[1] == 4

    # Independent dominant eigenpair of C for triad (0, 0, 0).
    q0 = analyzer._get_qhat_for_index(0)
    prod = q0 * q0
    c_matrix = (np.conj(q0).T @ (analyzer.W * prod)) / q0.shape[1]
    eigvals, eigvecs = np.linalg.eig(c_matrix)
    dom = int(np.argmax(np.abs(eigvals)))
    coeffs_col, _ = canonicalize_reference(eigvecs[:, dom].reshape(-1, 1))
    coeffs = coeffs_col[:, 0]
    ref_mode1 = q0 @ coeffs
    ref_mode2 = prod @ coeffs

    np.testing.assert_allclose(analyzer.eigenvalues[0], eigvals[dom], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(analyzer.modes1[0], ref_mode1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(analyzer.modes2[0], ref_mode2, rtol=1e-10, atol=1e-10)
    assert np.isfinite(analyzer.eigenvalues[0])
    assert np.linalg.norm(analyzer.modes1[0]) > 0.0


def test_negative_frequency_conjugate_symmetry(tmp_path):
    """Negative frequency bin indices are served via conjugate symmetry."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    # qhat has shape (nfft//2+1, Nspace, Nblocks) = (5, 4, Nblocks)
    assert analyzer.qhat.shape[0] == 5  # bins 0..4

    # Directly check the helper: qhat[-1] should equal conj(qhat[1])
    q_pos = analyzer._get_qhat_for_index(1)
    q_neg = analyzer._get_qhat_for_index(-1)
    np.testing.assert_array_equal(q_neg, np.conj(q_pos))

    # Run BSMD — should not crash with negative indices
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert not np.isnan(analyzer.eigenvalues[0])


def test_out_of_range_index_raises(tmp_path):
    """Triads with |p| > nfft//2 are unanalysable and raise ValueError."""
    analyzer = _make_analyzer(tmp_path, triads=[(99, -99, 0)], nfft=4, Ns=10)
    # nfft=4 → rfft bins 0..2; |p| = 99 exceeds nfft//2 = 2.
    with pytest.raises(ValueError, match=r"p=99"):
        analyzer._perform_static_bsmd_core()


def test_static_triads_default_is_none_and_resolves_to_copy(tmp_path):
    """static_triads=None resolves to a private copy of ALL_TRIADS."""
    analyzer = BSMDAnalyzer(
        file_path="default_triads",
        nfft=128,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: None,
    )
    assert list(analyzer.static_triads_list) == list(ALL_TRIADS)
    assert analyzer.static_triads_list is not ALL_TRIADS
    assert analyzer._static_triads_from_default is True


def test_default_triads_at_small_nfft_warn_and_filter(tmp_path):
    """Default ALL_TRIADS at nfft=8: filter + finite non-trivial spectrum.

    Kept triads stay in-range; every eigenvalue/mode must be finite; at least
    one |lambda| is strictly positive so a zeroed spectrum cannot pass.
    """
    rng = np.random.default_rng(20)
    ns, nspace = 64, 4
    data = {
        "q": rng.standard_normal((ns, nspace)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 0.1,
        "Nx": 2,
        "Ny": 2,
        "Ns": ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="default_nfft8",
        nfft=8,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer._perform_static_bsmd_core()
    msgs = [str(w.message) for w in caught]
    assert any("triad" in m.lower() for m in msgs), msgs
    limit = 8 // 2
    kept = list(analyzer.static_triads_list)
    assert kept, "default filter must keep at least one triad"
    assert all(all(abs(int(p)) <= limit for p in t) for t in kept)
    assert analyzer.eigenvalues.shape == (len(kept),)

    assert np.all(np.isfinite(analyzer.eigenvalues))
    assert np.all(np.isfinite(analyzer.modes1))
    assert np.all(np.isfinite(analyzer.modes2))
    # Spectrum is not the zero array — a wrong decomposition that zeros lambda fails.
    assert float(np.max(np.abs(analyzer.eigenvalues))) > 0.0
    assert float(np.linalg.norm(analyzer.modes1)) > 0.0
    assert float(np.linalg.norm(analyzer.modes2)) > 0.0


def test_user_out_of_range_triad_still_raises(tmp_path):
    """A user-supplied out-of-range triad remains a ValueError, not a filter."""
    analyzer = _make_analyzer(tmp_path, triads=[(8, -8, 0)], nfft=8, Ns=32)
    with pytest.raises(ValueError, match=r"p=8"):
        analyzer._perform_static_bsmd_core()


def test_user_mixed_list_names_every_offender(tmp_path):
    """A mixed valid/invalid user list names every out-of-range component once."""
    # nfft=8 → |p| <= 4; offenders in this list are 8, -8, 7, -5.
    triads = [(2, -2, 0), (8, -8, 0), (7, -5, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=32)
    with pytest.raises(ValueError) as excinfo:
        analyzer._perform_static_bsmd_core()
    msg = str(excinfo.value)
    for p in (8, -8, 7, -5):
        assert f"p={p}" in msg, f"expected p={p} in: {msg}"


def test_default_filter_to_empty_raises(tmp_path, monkeypatch):
    """Filtering the default list to nothing raises before any thread pool.

    use_parallel=True on purpose: an empty list also reaches
    ThreadPoolExecutor(max_workers=0). The message must name triads so that
    incidental executor error cannot stand in for this guard.
    """
    import openmodalpy.bsmd as bsmd

    monkeypatch.setattr(bsmd, "ALL_TRIADS", [(8, -8, 0), (7, -7, 0), (8, -7, 1)])
    rng = np.random.default_rng(20)
    ns, nspace = 64, 4
    data = {
        "q": rng.standard_normal((ns, nspace)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 0.1,
        "Nx": 2,
        "Ny": 2,
        "Ns": ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="empty_after_filter",
        nfft=8,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=True,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"(?i)triad") as excinfo:
            analyzer._perform_static_bsmd_core()
    msg = str(excinfo.value).lower()
    assert "cannot be analysed" in msg or "cannot be analyzed" in msg or "remain" in msg


def test_replaced_static_triads_list_treated_as_user(tmp_path):
    """A static_triads_list assigned after construction is user-supplied and raises."""
    rng = np.random.default_rng(20)
    ns, nspace = 64, 4
    data = {
        "q": rng.standard_normal((ns, nspace)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 0.1,
        "Nx": 2,
        "Ny": 2,
        "Ns": ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="replaced_list",
        nfft=8,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    # Values that also appear in ALL_TRIADS — origin, not value, decides the path.
    analyzer.static_triads_list = [(8, -8, 0)]
    with pytest.raises(ValueError, match=r"p=8"):
        analyzer._perform_static_bsmd_core()


def test_default_triads_truncated_loaded_bins_warn_and_filter(tmp_path):
    """Default list against a truncated loaded-bin count warns and filters, not raises."""
    rng = np.random.default_rng(20)
    ns, nspace = 600, 4
    data = {
        "q": rng.standard_normal((ns, nspace)),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 0.1,
        "Nx": 2,
        "Ny": 2,
        "Ns": ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="default_truncated",
        nfft=128,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    # Full rfft has 65 bins; shorten so |p| <= 4 (5 loaded bins).
    n_loaded = 5
    analyzer.qhat = analyzer.qhat[:n_loaded]
    assert analyzer._n_freq_bins == n_loaded
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer._perform_static_bsmd_core()
    msgs = [str(w.message) for w in caught]
    assert any("triad" in m.lower() for m in msgs), msgs
    limit = n_loaded - 1
    kept = list(analyzer.static_triads_list)
    assert kept, "default filter must keep at least one triad"
    assert all(all(abs(int(p)) <= limit for p in t) for t in kept)
    assert analyzer.eigenvalues.shape == (len(kept),)


def test_user_loaded_range_offender_raises(tmp_path):
    """A user-supplied list with a loaded-range (not rfft) offender raises ValueError."""
    analyzer = _make_analyzer(tmp_path, triads=[(20, 30, 50)], nfft=128, Ns=600)
    n_loaded = 10
    analyzer.qhat = analyzer.qhat[:n_loaded]
    assert analyzer._n_freq_bins == n_loaded
    # Inside nfft//2=64, outside loaded |p|<=9.
    with pytest.raises(ValueError, match=r"(loaded|p=20|p=30|p=50)"):
        analyzer._perform_static_bsmd_core()


def test_bsmd_rejects_invalid_spatial_metric(tmp_path):
    """Negative weights and a zero-measure metric raise ValueError once per analysis."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24, Nspace=4)
    analyzer.W = -np.abs(np.asarray(analyzer.W, dtype=float))
    with pytest.raises(ValueError, match="negative weight"):
        analyzer._perform_static_bsmd_core()

    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24, Nspace=4)
    analyzer.W = np.zeros_like(np.asarray(analyzer.W, dtype=float))
    with pytest.raises(ValueError, match="zero total measure"):
        analyzer._perform_static_bsmd_core()


def test_bsmd_rejects_nonfinite_spatial_metric(tmp_path):
    """NaN and inf weights raise before the eigenproblem runs."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24, Nspace=4)
    w = np.asarray(analyzer.W, dtype=float).copy()
    w.flat[0] = np.nan
    analyzer.W = w
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._perform_static_bsmd_core()

    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24, Nspace=4)
    w = np.asarray(analyzer.W, dtype=float).copy()
    w.flat[0] = np.inf
    analyzer.W = w
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._perform_static_bsmd_core()


def test_bsmd_accepts_isolated_zero_weight(tmp_path):
    """An isolated zero among positive weights is still a valid metric for BSMD.

    That cell contributes nothing to the inner product; the total measure stays
    positive. Pins this so a later over-strict check cannot reject it.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24, Nspace=4)
    w = np.asarray(analyzer.W, dtype=float).copy().reshape(-1)
    assert w.size >= 2
    w[0] = 0.0
    assert np.sum(w) > 0.0
    analyzer.W = w.reshape(analyzer.W.shape)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert np.all(np.isfinite(analyzer.eigenvalues))


def test_bsmd_polar_ny1_zero_measure_raises(tmp_path):
    """Polar weights on a single radial station at r=0 have zero total measure.

    The rejection happens as soon as the data is loaded, not later when the
    solver runs: load_and_preprocess builds the metric and checks it there.
    That is the earliest point the fault can be seen, and it is where the
    error text points the reader.
    """
    rng = np.random.default_rng(20)
    ns, nx, ny = 24, 4, 1
    data = {
        "q": rng.standard_normal((ns, nx * ny)),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="bsmd_polar_ny1",
        nfft=8,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="polar",
        use_static_triads=True,
        static_triads=[(1, 1, 2)],
        use_parallel=False,
    )
    with pytest.raises(ValueError, match="zero total measure"):
        analyzer.load_and_preprocess()


def test_nyquist_index_is_accepted(tmp_path):
    """|p| == nfft//2 is the last real rfft bin and must be analysed, not rejected."""
    analyzer = _make_analyzer(tmp_path, triads=[(4, -4, 0)], nfft=8, Ns=32)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)


def test_one_bin_past_nyquist_raises(tmp_path):
    """|p| == nfft//2 + 1 is the first unanalysable bin; the bound is exclusive there."""
    analyzer = _make_analyzer(tmp_path, triads=[(5, -5, 0)], nfft=8, Ns=32)
    with pytest.raises(ValueError, match=r"p=5"):
        analyzer._perform_static_bsmd_core()


def test_dynamic_triad_selection_raises(tmp_path):
    """Dynamic triad selection is unimplemented and must say so, not return empty arrays."""
    analyzer = _make_analyzer(tmp_path, triads=[], nfft=8, Ns=32, use_static=False)
    # Pin the public dispatch: use_static_triads=False must raise, not empty-return.
    assert analyzer.use_static_triads is False
    with pytest.raises(NotImplementedError):
        analyzer.perform_bsmd()


def test_result_count_follows_the_triads_not_the_blocks(tmp_path):
    """One result per triad, whatever the block count.

    BSMD assembles a ``(n_blocks, n_blocks)`` matrix per triad and keeps only
    the dominant eigenpair, so the number of results tracks the triad list and
    not the record length. SPOD does the opposite: its mode count IS the block
    count. This check keeps the class docstring honest about the difference.
    """
    triads = [(1, 1, 2), (1, -1, 0)]
    few = _make_analyzer(tmp_path / "few", triads=triads, nfft=8, Ns=24, Nspace=4)
    many = _make_analyzer(tmp_path / "many", triads=triads, nfft=8, Ns=96, Nspace=4)
    few.perform_bsmd()
    many.perform_bsmd()

    assert many.nblocks > few.nblocks, "the two runs must differ in block count"
    assert few.eigenvalues.shape == (len(triads),)
    assert many.eigenvalues.shape == (len(triads),)
    assert few.modes1.shape == many.modes1.shape


def test_perform_bsmd_static_path_shapes_and_nontrivial(tmp_path):
    """Public perform_bsmd() static-triad path yields shaped, non-trivial results."""
    triads = [(0, 0, 0), (1, -1, 0), (1, 1, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24, Nspace=4)
    analyzer.perform_bsmd()

    n_triads = len(triads)
    n_space = 4
    assert analyzer.modes1.shape == (n_triads, n_space)
    assert analyzer.modes2.shape == (n_triads, n_space)
    assert analyzer.eigenvalues.shape == (n_triads,)

    assert np.all(np.isfinite(analyzer.eigenvalues))
    assert np.all(np.isfinite(analyzer.modes1))
    assert np.all(np.isfinite(analyzer.modes2))
    assert not np.allclose(analyzer.modes1, 0.0)
    assert not np.allclose(analyzer.modes2, 0.0)
    assert not np.allclose(analyzer.eigenvalues, 0.0)

    # energy_map is a sparse (p1,p2) grid: unfilled slots are NaN by design;
    # occupied triad slots must be finite and not identically zero.
    assert analyzer.energy_map.size > 0
    finite = np.isfinite(analyzer.energy_map)
    assert np.any(finite), "energy map is all NaN (no triad landed)"
    assert not np.allclose(analyzer.energy_map[finite], 0.0)


def test_perform_bsmd_parallel_agrees_with_serial(tmp_path, caplog):
    """use_parallel=True must match the serial perform_bsmd path on the same data.

    Branch taken is pinned by the openmodalpy.bsmd logger message
    ("Thread-parallel BSMD ..."), not by re-deriving the use_parallel flag.
    """
    triads = [(0, 0, 0), (1, -1, 0), (1, 1, 2)]
    serial = _make_analyzer(
        tmp_path / "serial",
        triads=triads,
        nfft=8,
        Ns=24,
        Nspace=4,
        use_parallel=False,
        file_path="serial.h5",
    )
    with caplog.at_level(logging.INFO, logger="openmodalpy.bsmd"):
        serial.perform_bsmd()
    serial_msgs = [r.getMessage() for r in caplog.records if r.name == "openmodalpy.bsmd"]
    assert not any("Thread-parallel BSMD" in m for m in serial_msgs)
    caplog.clear()

    parallel = _make_analyzer(
        tmp_path / "parallel",
        triads=triads,
        nfft=8,
        Ns=24,
        Nspace=4,
        use_parallel=True,
        file_path="parallel.h5",
    )
    with caplog.at_level(logging.INFO, logger="openmodalpy.bsmd"):
        parallel.perform_bsmd()
    parallel_msgs = [r.getMessage() for r in caplog.records if r.name == "openmodalpy.bsmd"]
    assert any("Thread-parallel BSMD" in m for m in parallel_msgs)

    np.testing.assert_allclose(parallel.eigenvalues, serial.eigenvalues, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.abs(parallel.modes1), np.abs(serial.modes1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.abs(parallel.modes2), np.abs(serial.modes2), rtol=0, atol=1e-12)


def test_energy_map_keeps_triads_beyond_the_default_range(tmp_path):
    """A triad inside the rfft range but outside |p| <= 8 must still reach the energy map.

    The map used to be a fixed 17x17 grid centred on |p| = 8, so a valid triad at
    p = 12 (nfft = 32 gives 16 usable bins) was analysed and then silently dropped.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(12, -12, 0)], nfft=32, Ns=96)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    grid = analyzer.energy_map
    assert grid.shape == (25, 25), f"grid must span |p| = 12, got {grid.shape}"
    assert np.count_nonzero(np.isfinite(grid)) == 1, "the analysed triad is missing from the map"
    assert np.isfinite(grid[12 + 12, -12 + 12])


def test_triadic_constraint_violation_skipped(tmp_path):
    """Triads that violate p1+p2=p3 are skipped with NaN eigenvalue."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 1)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert np.isnan(np.abs(analyzer.eigenvalues[0]))


def test_triad_beyond_loaded_bins_raises_valueerror(tmp_path):
    """Triad inside nfft//2 but past the loaded qhat bins raises ValueError, not NaN.

    Simulates a stale/truncated FFT cache: full rfft would have nfft//2+1 bins,
    but qhat is shortened so early validation must use the real loaded length.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(20, 30, 50)], nfft=128, Ns=600)
    n_loaded = 10
    analyzer.qhat = analyzer.qhat[:n_loaded]
    assert analyzer._n_freq_bins == n_loaded
    # nfft//2 = 64 would accept |p|<=64; loaded bins only allow |p|<=9.
    with pytest.raises(ValueError, match=r"(10|9)"):
        analyzer._perform_static_bsmd_core()


def test_compute_single_triad_out_of_range_propagates_indexerror(tmp_path):
    """Out-of-range bin reads must not be laundered into a NaN eigenvalue."""
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)], nfft=128, Ns=600)
    analyzer.qhat = analyzer.qhat[:10]
    with pytest.raises(IndexError):
        analyzer._compute_single_triad(20, 30, 50)


def test_multiple_triads_with_negatives(tmp_path):
    """Multiple triads including negative bins all produce finite results."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (len(triads),)
    assert analyzer.modes1.shape == (len(triads), 4)
    assert analyzer.modes2.shape == (len(triads), 4)
    # All valid triads should produce finite eigenvalues
    for idx, (p1, p2, p3) in enumerate(triads):
        assert not np.isnan(analyzer.eigenvalues[idx]), f"Triad {(p1, p2, p3)} produced NaN"


def test_bispectral_correlation_uses_all_three_frequencies(tmp_path):
    """Verify that the bispectral correlation C involves Q1, Q2, AND Q3.

    Construct a case where Q3 is zeroed out.  If the algorithm correctly
    uses Q3 as B in C = A^H W B, all eigenvalues should be zero.
    """
    triads = [(1, 1, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    # Zero out qhat at bin 2 (= p3) → B = 0 → C = 0 → eigenvalue = 0
    analyzer.qhat[2, :, :] = 0.0
    analyzer._perform_static_bsmd_core()
    assert np.abs(analyzer.eigenvalues[0]) == pytest.approx(0.0, abs=1e-12)


def test_disk_backed_qhat_matches_ram(tmp_path):
    """Disk-backed mode (max_qhat_gb=0) produces identical results to RAM mode."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]

    # RAM mode (default) — helper seeds so both sides share identical data
    ram = _make_analyzer(tmp_path / "ram", triads=triads, nfft=8, Ns=24)
    ram._perform_static_bsmd_core()

    # Disk-backed mode: max_qhat_gb=0 forces offload on any qhat
    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    disk = _make_analyzer(disk_dir, triads=triads, nfft=8, Ns=24, max_qhat_gb=0)
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"

    disk._perform_static_bsmd_core()

    np.testing.assert_allclose(np.abs(disk.eigenvalues), np.abs(ram.eigenvalues), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes1), np.abs(ram.modes1), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes2), np.abs(ram.modes2), rtol=1e-12)
    disk.close()


def test_compute_single_triad_matches_dominant_eigenpair_shortcut(tmp_path):
    """The current BSMD core returns the dominant eigenpair of C.

    NOTE: this test re-derives the formula inline and therefore mirrors the
    implementation in src/openmodalpy/bsmd.py::_compute_single_triad -- it is
    a characterization test, NOT an independent oracle. It only proves the
    method is internally self-consistent with its own formula; it says nothing
    about whether that formula is physically correct. The independent oracle
    for correctness (manufactured phase-locked/control triads) lives in
    tests/test_bsmd_manufactured.py.
    """
    from tests.reference_helpers import canonicalize_reference

    analyzer = _make_analyzer(tmp_path, triads=[(1, 2, 3)], nfft=8, Ns=24, Nspace=2)
    analyzer.W = np.ones((2, 1), dtype=complex)

    q1 = np.ones((2, 2), dtype=complex)
    q2 = np.eye(2, dtype=complex)
    q3 = np.array([[0.0, 4.0], [0.0, 2.0]], dtype=complex)
    mapping = {1: q1, 2: q2, 3: q3}
    analyzer._get_qhat_for_index = lambda idx: mapping[idx]

    eigval, mode1, mode2 = analyzer._compute_single_triad(1, 2, 3)

    prod = q1 * q2
    c_matrix = (np.conj(q3).T @ (analyzer.W * prod)) / q1.shape[1]
    eigvals, eigvecs = np.linalg.eig(c_matrix)
    dom = np.argmax(np.abs(eigvals))
    coeffs_col, _ = canonicalize_reference(eigvecs[:, dom].reshape(-1, 1))
    coeffs = coeffs_col[:, 0]

    np.testing.assert_allclose(eigval, eigvals[dom], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mode1, q3 @ coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mode2, prod @ coeffs, rtol=1e-12, atol=1e-12)


def test_bsmd_modes_canonical_and_relative_phase_preserved(tmp_path, monkeypatch):
    """Canonicalize the shared eigenvector a; relative phase of the two modes stays.

    mode1 and mode2 inherit the same unit factor from a. Fixing each mode on
    its own pivot would apply two different factors and change conj(mode1)*mode2.
    The eigenvalue is the raw LAPACK value (phase of a does not enter it).

    Local LAPACK often already returns a real-positive pivot on this size of
    problem, so the free phase is injected through a unit factor on the
    eigenvectors; without the canonicalize call the modes keep that factor.
    """
    from tests.reference_helpers import canonicalize_reference

    analyzer = _make_analyzer(tmp_path, triads=[(1, 2, 3)], nfft=8, Ns=32, Nspace=4)
    analyzer.W = np.ones((4, 1), dtype=complex)

    rng = np.random.default_rng(7)
    q1 = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    q2 = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    q3 = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    mapping = {1: q1, 2: q2, 3: q3}
    analyzer._get_qhat_for_index = lambda idx: mapping[idx]

    phase = np.exp(1j * 1.1)
    real_eig = np.linalg.eig

    def phased_eig(c_matrix):
        vals, vecs = real_eig(c_matrix)
        return vals, vecs * phase

    monkeypatch.setattr(np.linalg, "eig", phased_eig)

    eigval_a, mode1_a, mode2_a = analyzer._compute_single_triad(1, 2, 3)
    eigval_b, mode1_b, mode2_b = analyzer._compute_single_triad(1, 2, 3)

    # Determinism: two calls, bit-comparable modes (no abs).
    np.testing.assert_allclose(mode1_a, mode1_b, rtol=0, atol=0)
    np.testing.assert_allclose(mode2_a, mode2_b, rtol=0, atol=0)
    np.testing.assert_allclose(eigval_a, eigval_b, rtol=0, atol=0)

    prod = q1 * q2
    c_matrix = (np.conj(q3).T @ (analyzer.W * prod)) / q1.shape[1]
    eigvals, eigvecs = np.linalg.eig(c_matrix)
    dom = np.argmax(np.abs(eigvals))
    a_phased = eigvecs[:, dom]
    a_can, _ = canonicalize_reference(a_phased.reshape(-1, 1))
    a_can = a_can[:, 0]
    assert not np.allclose(a_phased, a_can, rtol=0, atol=1e-10)

    mode1_phased = q3 @ a_phased
    mode2_phased = prod @ a_phased
    relative_phased = np.conj(mode1_phased) * mode2_phased
    relative_out = np.conj(mode1_a) * mode2_a

    # Modes follow the shared canonical coefficients, not the phased LAPACK vector.
    np.testing.assert_allclose(mode1_a, q3 @ a_can, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mode2_a, prod @ a_can, rtol=0, atol=1e-12)
    # Spectrum unchanged (eigenvalues are not scaled by the vector phase).
    np.testing.assert_allclose(eigval_a, eigvals[dom], rtol=0, atol=1e-12)
    # Same factor on both modes → relative product unchanged by canonicalization.
    np.testing.assert_allclose(relative_out, relative_phased, rtol=0, atol=1e-12)

    # Separate per-mode canonicalization would change that relative product.
    m1_sep, _ = canonicalize_reference(mode1_phased.reshape(-1, 1))
    m2_sep, _ = canonicalize_reference(mode2_phased.reshape(-1, 1))
    relative_sep = np.conj(m1_sep[:, 0]) * m2_sep[:, 0]
    assert not np.allclose(relative_sep, relative_phased, rtol=0, atol=1e-10)


def test_save_results_records_bsmd_approximation_contract(tmp_path):
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)], nfft=4, Ns=10)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_contract.hdf5")

    with h5py.File(tmp_path / "bsmd_contract.hdf5", "r") as handle:
        assert handle.attrs["bsmd_solver"] == "dominant_eigenpair_approximation"
        assert handle.attrs["bsmd_target_objective"] == "numerical_radius"
        assert handle.attrs["lift_kind"] == "triadic_spectral_product"
        assert bool(handle.attrs["uses_shared_triadic_coefficients"])
        assert handle.attrs["bispectrum_conjugation"] == "sum_frequency_conjugated"


def test_load_results_roundtrip_accepts_current_stamp(tmp_path):
    """A file written by the current build reloads without complaint."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_roundtrip.hdf5")

    reloaded = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    reloaded.load_results("bsmd_roundtrip.hdf5")
    np.testing.assert_allclose(reloaded.eigenvalues, analyzer.eigenvalues, rtol=1e-12)


def test_bsmd_save_load_roundtrip_arrays(tmp_path):
    """BSMD save → load restores eigenvalues and modes to machine precision."""
    triads = [(0, 0, 0), (1, -1, 0), (1, 1, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24, Nspace=4)
    analyzer.perform_bsmd()
    analyzer.save_results("bsmd_array_roundtrip.hdf5")

    reloaded = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24, Nspace=4)
    reloaded.load_results("bsmd_array_roundtrip.hdf5")

    np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
    np.testing.assert_array_equal(reloaded.modes1, analyzer.modes1)
    np.testing.assert_array_equal(reloaded.modes2, analyzer.modes2)
    np.testing.assert_array_equal(reloaded.triads, analyzer.triads)


def test_load_results_rejects_prefix_unconjugated_file(tmp_path):
    """Results written before the sum-frequency conjugation fix must not load.

    Such files carry eigenvalues and modes computed from E[X(f1)X(f2)X(f1+f2)]
    instead of the bispectrum E[X(f1)X(f2)X*(f1+f2)]; reloading them silently
    would hand the caller invalid numbers.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_stale.hdf5")

    # Simulate a file produced by a pre-fix build: the stamp did not exist.
    with h5py.File(tmp_path / "bsmd_stale.hdf5", "a") as handle:
        del handle.attrs["bispectrum_conjugation"]

    reloaded = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    with pytest.raises(ValueError, match="sum-frequency term was not conjugated"):
        reloaded.load_results("bsmd_stale.hdf5")


def _make_analyzer_without_fft(
    tmp_path,
    triads,
    nfft=8,
    Ns=32,
    Nspace=4,
    use_static=True,
):
    """Build a BSMDAnalyzer with weights set but qhat still empty (no FFT blocks)."""
    rng = np.random.default_rng(20)
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    data = {
        "q": rng.standard_normal((Ns, Nspace)),
        "x": np.linspace(0, 1, Nx),
        "y": np.linspace(0, 1, Ny),
        "dt": 1.0,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    analyzer = BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=nfft,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=use_static,
        static_triads=triads,
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    return analyzer


def test_empty_qhat_message_says_no_bins_loaded_user_triads(tmp_path):
    """User-supplied triads with no FFT blocks: message names empty load, not -1."""
    analyzer = _make_analyzer_without_fft(tmp_path, triads=[(1, 1, 2)])
    assert analyzer.qhat.size == 0
    with pytest.raises(ValueError) as excinfo:
        analyzer._perform_static_bsmd_core()
    msg = str(excinfo.value)
    assert "-1" not in msg
    assert "no frequency bins" in msg.lower() or "not loaded" in msg.lower()


def test_empty_qhat_message_says_no_bins_loaded_default_triads(tmp_path):
    """Default triad list with no FFT blocks: same honest empty-load message.

    Dropped triad indices can include -1 as a frequency bin; the check is that
    the bound phrase itself is never ``|p| <= -1``.
    """
    analyzer = _make_analyzer_without_fft(tmp_path, triads=None)
    assert analyzer.qhat.size == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            analyzer._perform_static_bsmd_core()
    msg = str(excinfo.value)
    assert "<= -1" not in msg
    assert "no frequency bins" in msg.lower() or "not loaded" in msg.lower()
    warn_text = " ".join(str(w.message) for w in caught)
    assert "<= -1" not in warn_text
    assert "no frequency bins" in warn_text.lower()


def test_perform_bsmd_forms_fft_blocks_on_first_use(tmp_path, caplog):
    """perform_bsmd forms its own FFT blocks when qhat is still empty.

    Previously this raised ValueError, forcing a separate hidden
    compute_fft_blocks() call between load_and_preprocess() and
    perform_bsmd(). perform_bsmd() now forms the FFT blocks itself.
    """
    analyzer = _make_analyzer_without_fft(tmp_path, triads=[(1, 1, 2)])
    assert analyzer.qhat.size == 0
    with caplog.at_level(logging.INFO, logger="openmodalpy.bsmd"):
        analyzer.perform_bsmd()
    msgs = [r.getMessage() for r in caplog.records if r.name == "openmodalpy.bsmd"]
    assert any("Starting BSMD analysis" in m for m in msgs)
    assert analyzer.qhat.size > 0
    assert analyzer.eigenvalues.size > 0


def test_compute_single_triad_nan_in_qhat_returns_nan(tmp_path):
    """NaN in a used qhat bin yields (nan, None, None) via the LinAlgError path."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 2)], nfft=8, Ns=24)
    analyzer.qhat[1, 0, 0] = np.nan
    eig, mode1, mode2 = analyzer._compute_single_triad(1, 1, 2)
    assert np.isnan(eig)
    assert mode1 is None and mode2 is None


def test_bsmd_save_preserves_fftblocks_on_the_cache_file(tmp_path):
    """save_results onto the open FFT cache must keep FFTBlocks intact.

    BSMD writes the FFT cache under the same auto-name as the result file.
    When save_results targets that path it appends, so FFTBlocks survives.
    Overwriting the file (mode "w") would drop them.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)], nfft=4, Ns=10)
    assert analyzer._qhat_cache_path is not None
    cache_path = analyzer._qhat_cache_path
    with h5py.File(cache_path, "r") as handle:
        assert "FFTBlocks" in handle
        qhat_before = handle["FFTBlocks"][:]

    analyzer._perform_static_bsmd_core()
    # Default filename is the cache path itself — the append edge under test.
    analyzer.save_results()

    with h5py.File(cache_path, "r") as handle:
        assert "FFTBlocks" in handle
        np.testing.assert_array_equal(handle["FFTBlocks"][:], qhat_before)
        assert "triads" in handle
        assert "eigenvalues" in handle
        assert "modes1" in handle


def test_qhat_disk_state_consistent_after_save_results(tmp_path):
    """save_results onto the FFT cache must leave disk-backed qhat readable.

    Closing the cache handle (needed so the append does not open a second
    writer on the same path) used to leave ``_qhat_on_disk`` True with
    ``_qhat_dataset`` None. Readers then hit that None dataset. This test
    saves to the cache filename so it reaches that branch.
    """
    from pathlib import Path

    first_triads = [(1, -1, 0), (0, 0, 0)]
    second_triads = [(1, 1, 2), (2, -2, 0)]

    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    disk = _make_analyzer(disk_dir, triads=first_triads, nfft=8, Ns=24, max_qhat_gb=0)
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"
    assert disk._qhat_dataset is not None
    cache_path = disk._qhat_cache_path
    assert cache_path is not None
    cache_name = Path(cache_path).name
    save_path = Path(disk.results_dir) / cache_name
    assert save_path.resolve() == Path(cache_path).resolve(), (
        "save must target the cache file to hit the using_cache_file branch"
    )

    n_freq_before = disk._n_freq_bins
    n_spatial_before = disk._n_spatial
    disk.perform_bsmd()
    disk.save_results(cache_name)

    # Invariant: flag True implies a live dataset (not a None leftover).
    if disk._qhat_on_disk:
        assert disk._qhat_dataset is not None, "_qhat_on_disk is True but _qhat_dataset is None after save_results"
    assert disk._qhat_on_disk, "disk-backed reuse must survive save_results"
    assert disk._n_freq_bins == n_freq_before
    assert disk._n_spatial == n_spatial_before

    disk.static_triads_list = list(second_triads)
    disk.perform_bsmd()
    eigs_after_save = np.asarray(disk.eigenvalues)

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control = _make_analyzer(control_dir, triads=second_triads, nfft=8, Ns=24, max_qhat_gb=0)
    control.perform_bsmd()
    np.testing.assert_allclose(eigs_after_save, control.eigenvalues, rtol=1e-12, atol=1e-12)

    disk.close()
    assert disk._qhat_file is None
    assert disk._qhat_dataset is None
    assert disk._qhat_on_disk is False
    control.close()


def test_qhat_disk_state_consistent_when_save_results_raises(tmp_path, monkeypatch):
    """A failed save onto the FFT cache must leave disk-backed qhat consistent.

    Closing the cache handle (needed so the append does not open a second
    writer on the same path) used to leave ``_qhat_on_disk`` True with
    ``_qhat_dataset`` None. If the write then failed, ``close()`` could not
    repair it because the handle was already None. The flag must not outlive
    a live dataset, including when rebind itself cannot open the cache.
    """
    from pathlib import Path

    import openmodalpy.bsmd as bsmd
    import openmodalpy.core.results as results_mod

    disk = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24, max_qhat_gb=0)
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"
    assert disk._qhat_dataset is not None
    cache_path = disk._qhat_cache_path
    assert cache_path is not None
    cache_name = Path(cache_path).name
    save_path = Path(disk.results_dir) / cache_name
    assert save_path.resolve() == Path(cache_path).resolve(), (
        "save must target the cache file to hit the using_cache_file branch"
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced write_results failure")

    monkeypatch.setattr(results_mod, "write_results", _boom)

    with pytest.raises(RuntimeError, match="forced write_results failure"):
        disk.save_results(cache_name)

    if disk._qhat_on_disk:
        assert disk._qhat_dataset is not None, (
            "_qhat_on_disk is True but _qhat_dataset is None after failed save_results"
        )
    disk.close()
    assert disk._qhat_file is None
    assert disk._qhat_dataset is None
    assert disk._qhat_on_disk is False

    # Rebind whose open fails must leave the flag False, not True.
    junk = tmp_path / "not-an-hdf5.txt"
    junk.write_text("not hdf5", encoding="utf-8")
    disk._qhat_on_disk = True
    disk._rebind_qhat_dataset(str(junk))
    assert disk._qhat_on_disk is False
    assert disk._qhat_file is None
    assert disk._qhat_dataset is None

    def _bad_open(*_args, **_kwargs):
        raise TypeError("forced non-OSError open failure")

    monkeypatch.setattr(bsmd.h5py, "File", _bad_open)
    disk._qhat_on_disk = True
    disk._rebind_qhat_dataset(cache_path)
    assert disk._qhat_on_disk is False
    assert disk._qhat_file is None
    assert disk._qhat_dataset is None


def test_qhat_disk_state_consistent_when_write_mode_probe_raises(tmp_path, monkeypatch):
    """A raise BEFORE the write must not leave the flag set with no dataset.

    The write mode is probed, and the dataset dict built, after the cache
    handle has been closed but before the write is attempted. A raise in that
    window used to exit with ``_qhat_on_disk`` True and ``_qhat_dataset`` None
    -- the one state ``close()`` cannot repair, because its clear is guarded on
    the handle still being non-None. Distinct from the failed-write test above,
    which exercises the window the ``finally`` already covers.
    """
    from pathlib import Path

    import openmodalpy.bsmd as bsmd

    disk = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24, max_qhat_gb=0)
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"
    cache_path = disk._qhat_cache_path
    assert cache_path is not None
    cache_name = Path(cache_path).name

    def _boom(*_args, **_kwargs):
        raise OSError("forced write-mode probe failure")

    monkeypatch.setattr(bsmd, "_hdf5_write_mode", _boom)

    with pytest.raises(OSError, match="forced write-mode probe failure"):
        disk.save_results(cache_name)

    assert not (disk._qhat_on_disk and disk._qhat_dataset is None), (
        "_qhat_on_disk is True but _qhat_dataset is None after a pre-write failure"
    )

    disk.close()
    assert disk._qhat_file is None
    assert disk._qhat_dataset is None
    assert disk._qhat_on_disk is False


class _FakeTriadHolder:
    """Stand-in exposing only what ``_triad_plot_order`` reads: the triad list."""

    def __init__(self, triads):
        self.static_triads_list = triads


def _triad_order(triads, lambdas):
    """Call the bound method against a minimal object, no analyzer setup needed."""
    holder = _FakeTriadHolder(triads)
    valid_idx = np.arange(len(triads))
    return BSMDAnalyzer._triad_plot_order(holder, np.asarray(lambdas, dtype=float), valid_idx)


def test_triad_plot_order_exact_tie_independent_of_listing_order(tmp_path):
    """Two triads tied at the same |lambda|: plot order does not depend on listing order.

    Whether the triads are computed/listed as A,B or B,A, the tie is broken by
    the triad tuple, not by position, so both listings pick the same triad first.
    """
    triad_a = (0, 0, 0)
    triad_b = (1, 1, 1)

    order_ab = _triad_order([triad_a, triad_b], [1.0, 1.0])
    order_ba = _triad_order([triad_b, triad_a], [1.0, 1.0])

    picked_ab = [[triad_a, triad_b][i] for i in order_ab]
    picked_ba = [[triad_b, triad_a][i] for i in order_ba]
    assert picked_ab == picked_ba == [triad_a, triad_b]


def test_triad_plot_order_end_to_end_through_plot_modes(tmp_path):
    """Same exact-tie invariant, exercised through ``plot_modes`` itself."""
    from matplotlib.figure import Figure

    triad_a = (0, 0, 0)
    triad_b = (1, 1, 2)
    orig_suptitle = Figure.suptitle

    def _titles_for(listing):
        analyzer = _make_analyzer(tmp_path, triads=list(listing), nfft=8, Ns=24, Nspace=4)
        analyzer._perform_static_bsmd_core()
        analyzer.eigenvalues[:] = 1.0 + 0.0j  # force an exact tie regardless of the real computation

        captured = []

        def spy_suptitle(self, text, **kwargs):
            captured.append(text)
            return orig_suptitle(self, text, **kwargs)

        Figure.suptitle = spy_suptitle
        try:
            analyzer.plot_modes()
        finally:
            Figure.suptitle = orig_suptitle
            import matplotlib.pyplot as plt

            plt.close("all")
        return captured

    titles_ab = _titles_for([triad_a, triad_b])
    titles_ba = _titles_for([triad_b, triad_a])
    assert titles_ab == titles_ba


def test_triad_plot_order_near_tie_subset_at_cutoff_matches(tmp_path):
    """A near-tie (relative gap 1e-13, inside the band) at the cutoff: same subset either way.

    Two triads inside the tie band compete for the last plotted slot; the
    triad tuple, not listing order, decides which one wins, so the selected
    subset is identical whichever order the triads were computed in.
    """
    leader = (0, 0, 0)
    near_a = (3, 3, 3)
    near_b = (1, -1, 0)  # smaller tuple than near_a, so it wins the tie

    lam_leader = 1.0
    lam_near = 0.5
    lam_near_tied = lam_near * (1 + 1e-13)  # inside CANONICAL_TIE_RTOL band

    triads_1 = [leader, near_a, near_b]
    lambdas_1 = [lam_leader, lam_near, lam_near_tied]
    triads_2 = [leader, near_b, near_a]
    lambdas_2 = [lam_leader, lam_near_tied, lam_near]

    order_1 = _triad_order(triads_1, lambdas_1)
    order_2 = _triad_order(triads_2, lambdas_2)

    picked_1 = {triads_1[i] for i in order_1[:2]}
    picked_2 = {triads_2[i] for i in order_2[:2]}
    assert picked_1 == picked_2 == {leader, near_b}


def test_triad_plot_order_gap_outside_band_keeps_magnitude_order(tmp_path):
    """A pair with relative gap 1e-11 (outside the tie band) keeps magnitude order both ways."""
    small_tuple_triad = (0, 0, 0)  # would win a tie by tuple, but the gap is too large to tie
    big_lambda_triad = (9, 9, 9)

    lam_big = 1.0
    lam_small = lam_big * (1 - 1e-11)  # outside CANONICAL_TIE_RTOL band

    triads_1 = [small_tuple_triad, big_lambda_triad]
    lambdas_1 = [lam_small, lam_big]
    triads_2 = [big_lambda_triad, small_tuple_triad]
    lambdas_2 = [lam_big, lam_small]

    order_1 = _triad_order(triads_1, lambdas_1)
    order_2 = _triad_order(triads_2, lambdas_2)

    picked_1 = [triads_1[i] for i in order_1]
    picked_2 = [triads_2[i] for i in order_2]
    assert picked_1 == picked_2 == [big_lambda_triad, small_tuple_triad]


@pytest.mark.parametrize("overlap", [10, -0.1])
def test_bsmd_rejects_invalid_overlap(tmp_path, overlap):
    """BSMD rejects an overlap outside [0, 1), matching SPOD's own check."""
    with pytest.raises(ValueError, match="Overlap must be between 0 .inclusive. and 1 .exclusive."):
        BSMDAnalyzer(
            file_path="dummy.h5",
            nfft=8,
            overlap=overlap,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            spatial_weight_type="uniform",
            static_triads=[(0, 0, 0)],
            use_parallel=False,
        )


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_bsmd_accepts_valid_overlap(tmp_path, overlap):
    """BSMD still accepts the boundary-inclusive 0.0 and a normal 0.5 overlap."""
    analyzer = BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=overlap,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: {
            "q": np.random.default_rng(1).standard_normal((10, 4)),
            "x": np.linspace(0, 1, 2),
            "y": np.linspace(0, 1, 2),
            "dt": 1.0,
            "Nx": 2,
            "Ny": 2,
            "Ns": 10,
        },
        spatial_weight_type="uniform",
        static_triads=[(0, 0, 0)],
        use_parallel=False,
    )
    assert analyzer.overlap == overlap
