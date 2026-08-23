import logging

import h5py
import numpy as np
import pytest
from scipy import signal

from openmodalpy import MPODAnalyzer, PODAnalyzer
from openmodalpy.core.base import get_robust_clim, subset_volume_focus_3d


def test_perform_pod_simple():
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    # Centered snapshots are rank-1 (q = a * [1,1]); the relative Gram cutoff
    # returns that single mode rather than padding to n_modes_save=2.
    assert analyzer.modes.shape == (2, 1)
    assert analyzer.time_coefficients.shape == (3, 1)
    assert np.isclose(analyzer.eigenvalues[0], 5.333333333333333, atol=1e-6)


def test_plot_time_coefficients_strouhal(monkeypatch, tmp_path):
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    labels = []
    x_data = []

    def xlabel_mock(text):
        labels.append(text)

    def semilogy_mock(x, y, **kwargs):
        x_data.append(np.array(x))
        return None

    monkeypatch.setattr("matplotlib.pyplot.xlabel", xlabel_mock)
    monkeypatch.setattr("matplotlib.pyplot.semilogy", semilogy_mock)

    analyzer.plot_time_coefficients(n_coeffs_to_plot=1, L=2.0, U=4.0)

    assert "Strouhal Number (St)" in labels
    coeff = analyzer.time_coefficients[:3, 0]
    freqs, _ = signal.periodogram(coeff, analyzer.fs, scaling="density")
    expected = freqs * 2.0 / 4.0
    assert np.allclose(x_data[0], expected)


def test_spatial_kernel_time_coefficients_use_weighted_inner_product():
    """Time coefficients match an independent weighted spatial-kernel solve.

    Fixture is a 5×2 hand-written array with W = [1, 9]. The reference solves
    the spatial eigenproblem with plain numpy (eigh on the sqrt(W)-scaled
    Gram matrix), then recovers modes and coefficients — never from the
    analyzer's own modes. That pins the weighted seam, not just the final
    projection formula.
    """
    from tests.reference_helpers import canonicalize_reference

    q = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 9.0])
    data = {
        "q": q,
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 5,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weights=weights,
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    # A prescribed vector stays in analyzer.W and is used by the eigenproblem.
    np.testing.assert_allclose(np.asarray(analyzer.W).ravel(), weights)
    analyzer.perform_pod()
    np.testing.assert_allclose(np.asarray(analyzer.W).ravel(), weights)

    # Independent spatial-kernel POD: K = (Q_c √W)^T (Q_c √W) / N
    q_centered = q - np.mean(q, axis=0, keepdims=True)
    n_snapshots = q_centered.shape[0]
    sqrt_w = np.sqrt(np.maximum(weights, 1e-12))
    q_weighted = q_centered * sqrt_w
    kernel = (q_weighted.T @ q_weighted) / n_snapshots
    evals, weighted_modes = np.linalg.eigh(kernel)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    weighted_modes = weighted_modes[:, order]
    ref_modes = weighted_modes / sqrt_w[:, np.newaxis]
    ref_coeffs = q_weighted @ weighted_modes
    ref_modes, ref_coeffs = canonicalize_reference(np.real(ref_modes), np.real(ref_coeffs))

    np.testing.assert_allclose(analyzer.eigenvalues, evals, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(analyzer.modes, ref_modes, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        analyzer.time_coefficients,
        ref_coeffs,
        rtol=1e-10,
        atol=1e-10,
    )


def test_pod_spatial_regime_keeps_full_spatial_rank(tmp_path):
    """In the spatial regime, mean-centering must not drop a genuine mode.

    Mean-centering costs one SAMPLE degree of freedom, so the rank bound is
    ``min(n_samples - 1, n_space)`` — not ``min(n_samples, n_space) - 1``.
    When ``n_space < n_samples - 1``, POD must therefore return
    ``min(n_modes_save, n_space)`` modes. Full-column-rank random data makes
    every spatial direction energetic so the solver does not hide the cap.
    """
    rng = np.random.default_rng(0)
    n_samples, n_space = 8, 3
    assert n_space < n_samples - 1
    q = rng.standard_normal((n_samples, n_space))
    # Ensure the centered field still has full spatial rank.
    q_c = q - q.mean(axis=0, keepdims=True)
    assert np.linalg.matrix_rank(q_c, tol=1e-10) == n_space

    data = {
        "q": q,
        "x": np.linspace(0.0, 1.0, n_space),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_space,
        "Ny": 1,
        "Ns": n_samples,
    }
    n_modes_save = n_space  # request every genuine mode
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes_save,
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod(solver="eigh")

    want = min(n_modes_save, n_space)
    assert analyzer.modes.shape == (n_space, want)
    assert analyzer.eigenvalues.shape == (want,)
    assert analyzer.time_coefficients.shape == (n_samples, want)
    assert np.all(analyzer.eigenvalues > 0.0)


def test_pod_save_results_records_second_order_contract(tmp_path):
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.save_results("pod_contract.hdf5")

    with h5py.File(tmp_path / "pod_contract.hdf5", "r") as handle:
        assert handle.attrs["lift_kind"] == "identity_centered_snapshots"
        assert bool(handle.attrs["uses_mean_subtraction"])
        assert bool(handle.attrs["uses_spatial_metric_in_second_order_operator"])
        assert handle.attrs["eigenvalue_normalization"] == "snapshot_average"


def test_run_analysis_uses_3d_slice_plots_for_volumetric_data(monkeypatch, tmp_path):
    data = {
        "q": np.array(
            [
                np.arange(8, dtype=float),
                np.arange(8, dtype=float) + 1.0,
                np.arange(8, dtype=float) + 2.0,
                np.arange(8, dtype=float) + 3.0,
            ]
        ),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
        "z": np.array([0.0, 1.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Nz": 2,
        "Ns": 4,
    }
    analyzer = PODAnalyzer(
        file_path="3d",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )

    slice_calls = []
    iso_calls = []
    monkeypatch.setattr(
        PODAnalyzer, "plot_modes_3d_slices", lambda self, plot_n_modes=4: slice_calls.append(plot_n_modes)
    )
    monkeypatch.setattr(
        PODAnalyzer, "plot_modes_3d_isometric", lambda self, plot_n_modes=4: iso_calls.append(plot_n_modes)
    )
    monkeypatch.setattr(PODAnalyzer, "plot_eigenvalues", lambda self: None)
    monkeypatch.setattr(PODAnalyzer, "plot_time_coefficients", lambda self, **kwargs: None)
    monkeypatch.setattr(PODAnalyzer, "plot_cumulative_energy", lambda self: None)
    monkeypatch.setattr(PODAnalyzer, "plot_reconstruction_error", lambda self: None)
    monkeypatch.setattr(
        PODAnalyzer,
        "plot_reconstruction_comparison",
        lambda self, **kwargs: (_ for _ in ()).throw(
            AssertionError("2D reconstruction comparison should not be used for 3D data")
        ),
    )
    monkeypatch.setattr(
        PODAnalyzer,
        "plot_modes_pair_detailed",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("2D mode plotting should not be used for 3D data")),
    )
    monkeypatch.setattr(
        PODAnalyzer,
        "plot_modes_grid",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("2D grid plotting should not be used for 3D data")),
    )
    monkeypatch.setattr(
        PODAnalyzer,
        "plot_mode_pair_phase",
        lambda self: (_ for _ in ()).throw(AssertionError("2D pair-phase plotting should not be used for 3D data")),
    )

    # The unified run_analysis routes volumetric data to the 3-D hooks via
    # _maybe_plot_volumetric_modes; the cap is min(2, n_modes_save).
    analyzer.run_analysis()

    # Cap is min(2, n_modes_save) AFTER the solver's honest resync: this
    # 4-snapshot fixture supports a single mode, so the cap is 1.
    assert slice_calls == [min(2, analyzer.n_modes_save)]
    assert iso_calls == slice_calls


def test_subset_volume_focus_3d_respects_volume_xlim():
    field = np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)
    data = {
        "metadata": {
            "plot_style": {
                "volume": {
                    "xlim": [0.0, 1.0],
                }
            }
        }
    }
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])

    focused, x_focus, y_focus, z_focus = subset_volume_focus_3d(field, x, y, z, data)

    assert focused.shape == (3, 3, 2)
    np.testing.assert_array_equal(x_focus, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(y_focus, y)
    np.testing.assert_array_equal(z_focus, z)


def test_subset_volume_focus_3d_does_not_copy_when_nothing_is_cropped():
    # Rendering one mode figure used to copy the whole volume here even with no
    # cropping configured, because np.ix_ fancy indexing always copies.
    field = np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])

    focused, x_focus, y_focus, z_focus = subset_volume_focus_3d(field, x, y, z, {})

    assert focused.shape == field.shape
    assert np.shares_memory(focused, field)
    np.testing.assert_array_equal(focused, field)
    np.testing.assert_array_equal(x_focus, x)
    np.testing.assert_array_equal(y_focus, y)
    np.testing.assert_array_equal(z_focus, z)


def test_get_robust_clim_ignores_infinities_as_well_as_nan():
    # np.nanpercentile would drop the NaN but keep the infinities. The limits would
    # then be non-finite and fall through to the (-1, 1) fallback, so the colours
    # would come from nowhere near the data. Assert the limits track the finite
    # values instead: a plain "is it finite" check would pass on that fallback.
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan, np.inf, -np.inf])

    # Limits are symmetrised around zero for diverging colormaps.
    assert get_robust_clim(values, method="minmax") == (-4.0, 4.0)

    for method in ("percentile", "sigma"):
        vmin, vmax = get_robust_clim(values, method=method)
        assert np.isfinite(vmin) and np.isfinite(vmax), method
        assert vmax > 3.0, f"{method} collapsed to the fallback: {vmax}"
        assert vmin == -vmax, method


def _make_pod_analyzer(data, tmp_path, n_modes_save=3, spatial_weights=None):
    return PODAnalyzer(
        file_path="pod_check",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform" if spatial_weights is None else "prescribed",
        spatial_weights=spatial_weights,
        n_modes_save=n_modes_save,
        use_parallel=False,
    )


def test_check_spatial_mode_orthogonality_true_and_false(small_pod_field, tmp_path):
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    assert analyzer.check_spatial_mode_orthogonality()

    # Deliberate corruption: break W-orthonormality of stored modes
    analyzer.modes = analyzer.modes + 1.0
    assert not analyzer.check_spatial_mode_orthogonality()


def test_check_spatial_mode_orthogonality_empty_modes(small_pod_field, tmp_path):
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path)
    # No perform_pod — modes and W remain empty arrays
    assert not analyzer.check_spatial_mode_orthogonality()


def test_check_temporal_coefficient_orthogonality_true_and_false(small_pod_field, tmp_path):
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    assert analyzer.check_temporal_coefficient_orthogonality()

    analyzer.time_coefficients = analyzer.time_coefficients + 5.0
    assert not analyzer.check_temporal_coefficient_orthogonality()


def test_check_temporal_coefficient_orthogonality_empty(small_pod_field, tmp_path):
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path)
    assert not analyzer.check_temporal_coefficient_orthogonality()


def test_pod_temporal_and_spatial_kernel_branches_agree(small_pod_field, tmp_path, caplog):
    """Both POD kernels must agree on the same active field.

    Spatial branch: Ns >= Nspace (small_pod_field is built that way).
    Temporal branch: pad zero spatial columns so Ns < Nspace while leaving the
    active columns identical — zero columns do not change the Gram spectrum.
    Branch taken is pinned by capturing pod.py's own INFO log messages
    ("Using temporal kernel:" / "Using spatial kernel:"), not by re-deriving
    the Ns < Nspace predicate in the test.

    Both runs share a prescribed ones metric: kernel agreement is the property
    under test, and the derived cell-volume metric would otherwise differ
    between the variants (the padded grid makes former boundary columns
    interior, doubling their cell weight), which is a metric difference, not a
    kernel difference. Metric provenance has its own test
    (test_pod_uniform_metric_moves_eigenvalues).
    """
    n_modes = 2
    data_spatial = {
        "q": small_pod_field["q"].copy(),
        "x": small_pod_field["x"].copy(),
        "y": small_pod_field["y"].copy(),
        "dt": small_pod_field["dt"],
        "Nx": small_pod_field["Nx"],
        "Ny": small_pod_field["Ny"],
        "Ns": small_pod_field["Ns"],
    }
    Ns_s = data_spatial["Ns"]
    Nspace_s = data_spatial["Nx"]
    assert Ns_s >= Nspace_s  # fixture setup for the spatial case

    analyzer_spatial = _make_pod_analyzer(
        data_spatial,
        tmp_path / "spatial",
        n_modes_save=n_modes,
        spatial_weights=np.ones(Nspace_s),
    )
    analyzer_spatial.load_and_preprocess()
    with caplog.at_level(logging.INFO, logger="openmodalpy.pod"):
        caplog.clear()
        analyzer_spatial.perform_pod()
        spatial_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Using spatial kernel:" in m for m in spatial_msgs)
    assert not any("Using temporal kernel:" in m for m in spatial_msgs)

    # Temporal-kernel case: same active q, zero-padded in space until Ns < Nspace.
    n_pad = Ns_s - Nspace_s + 1  # guarantees Ns < Nspace_padded
    q_temporal = np.hstack([data_spatial["q"], np.zeros((Ns_s, n_pad))])
    Nspace_t = q_temporal.shape[1]
    data_temporal = {
        "q": q_temporal,
        "x": np.linspace(0.0, 1.0, Nspace_t),
        "y": np.array([0.0]),
        "dt": data_spatial["dt"],
        "Nx": Nspace_t,
        "Ny": 1,
        "Ns": Ns_s,
    }
    assert data_temporal["Ns"] < data_temporal["Nx"]

    analyzer_temporal = _make_pod_analyzer(
        data_temporal,
        tmp_path / "temporal",
        n_modes_save=n_modes,
        spatial_weights=np.ones(Nspace_t),
    )
    analyzer_temporal.load_and_preprocess()
    with caplog.at_level(logging.INFO, logger="openmodalpy.pod"):
        caplog.clear()
        analyzer_temporal.perform_pod()
        temporal_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Using temporal kernel:" in m for m in temporal_msgs)
    assert not any("Using spatial kernel:" in m for m in temporal_msgs)

    np.testing.assert_allclose(
        analyzer_spatial.eigenvalues,
        analyzer_temporal.eigenvalues,
        atol=1e-10,
        rtol=0.0,
    )
    # Same data, both kernels, both sign-canonicalized → coeffs match directly.
    np.testing.assert_allclose(
        analyzer_spatial.time_coefficients,
        analyzer_temporal.time_coefficients,
        atol=1e-10,
        rtol=0.0,
    )


def test_pod_save_load_roundtrip_arrays(small_pod_field, tmp_path):
    """POD save → load restores modes, eigenvalues, and time coefficients exactly."""
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path, n_modes_save=2)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.save_results("pod_roundtrip.hdf5")

    reloaded = _make_pod_analyzer(small_pod_field, tmp_path, n_modes_save=2)
    reloaded.load_results("pod_roundtrip.hdf5")

    np.testing.assert_array_equal(reloaded.modes, analyzer.modes)
    np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
    np.testing.assert_array_equal(reloaded.time_coefficients, analyzer.time_coefficients)


def test_pod_energy_captured_fraction_two_truncation_levels(small_pod_field, tmp_path):
    """Truncated energy over pre-truncation total at two ranks (analytic rank-2).

    Expected ratios are derived from the full spectrum of a full-rank run of the
    same field — never pasted from a prior printout.
    """
    full = _make_pod_analyzer(small_pod_field, tmp_path / "full", n_modes_save=10)
    full.load_and_preprocess()
    full.perform_pod()
    full_eigs = full.eigenvalues.copy()
    total = float(np.sum(full_eigs))
    assert total > 0.0
    # Rank-2 field: only two positive eigenvalues.
    assert np.count_nonzero(full_eigs > 1e-12) == 2

    one = _make_pod_analyzer(small_pod_field, tmp_path / "one", n_modes_save=1)
    one.load_and_preprocess()
    one.perform_pod()
    expected_one = float(full_eigs[0] / total)
    assert 0.0 < expected_one < 1.0
    assert abs(one.energy_captured_fraction - expected_one) < 1e-10

    two = _make_pod_analyzer(small_pod_field, tmp_path / "two", n_modes_save=2)
    two.load_and_preprocess()
    two.perform_pod()
    expected_two = float(np.sum(full_eigs[:2]) / total)
    assert abs(expected_two - 1.0) < 1e-10
    assert abs(two.energy_captured_fraction - 1.0) < 1e-10

    one.save_results("pod_energy.hdf5")
    with h5py.File(tmp_path / "one" / "pod_energy.hdf5", "r") as handle:
        assert "energy_captured_fraction" in handle.attrs
        assert abs(float(handle.attrs["energy_captured_fraction"]) - expected_one) < 1e-10


def test_pod_cumulative_percentage_below_100_when_truncated(small_pod_field, tmp_path):
    """With n_modes_save below full rank, cumulative energy share is < 100%.

    Under the retained-sum denominator this was exactly 100 by construction.
    """
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path, n_modes_save=1)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    denom, suffix = analyzer._energy_denominator()
    assert suffix == ""
    assert denom > 0.0
    cum_pct = 100.0 * float(np.sum(analyzer.eigenvalues)) / denom
    assert cum_pct < 100.0


def test_pod_mode_percentage_uses_pretruncation_total(small_pod_field, tmp_path):
    """A mode percentage is 100 * lambda_i / total_energy, not / sum(retained)."""
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path, n_modes_save=1)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    lambda_i = float(analyzer.eigenvalues[0])
    total = float(analyzer.total_energy)
    retained = float(np.sum(analyzer.eigenvalues))
    assert total > retained > 0.0

    pct_true = 100.0 * lambda_i / total
    pct_retained = 100.0 * lambda_i / retained
    denom, suffix = analyzer._energy_denominator()
    assert suffix == ""
    assert abs(100.0 * lambda_i / denom - pct_true) < 1e-12
    assert pct_true < pct_retained


def test_pod_energy_denominator_falls_back_with_honest_label(small_pod_field, tmp_path):
    """When total_energy is unknown, denominator is retained sum and suffix is set."""
    analyzer = _make_pod_analyzer(small_pod_field, tmp_path, n_modes_save=2)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.total_energy = float("nan")

    denom, suffix = analyzer._energy_denominator()
    assert abs(denom - float(np.sum(analyzer.eigenvalues))) < 1e-12
    assert suffix  # non-empty — a silent fallback would hide the missing total
    assert "retained modes only" in suffix


def test_mpod_energy_label_says_retained_only(tmp_path):
    """mPOD never sets total_energy, so percentages must carry the fallback label.

    Uses multi-band edges so perform_mpod takes its own eigenvalue path
    (mpod.py band POD), not the single-full-band shortcut into perform_pod.
    """
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = np.array([1.0, 0.0, 0.0, 1.0])
    phi_low = phi_low / np.linalg.norm(phi_low)
    phi_high = np.array([0.0, 1.0, 1.0, 0.0])
    phi_high = phi_high / np.linalg.norm(phi_high)
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )
    data = {
        "q": q,
        "x": np.arange(4, dtype=float),
        "y": np.array([0.0]),
        "dt": dt,
        "Nx": 4,
        "Ny": 1,
        "Ns": ns,
    }
    analyzer = MPODAnalyzer(
        file_path="mpod_energy_label",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    assert not np.isfinite(getattr(analyzer, "total_energy", float("nan")))
    denom, suffix = analyzer._energy_denominator()
    assert abs(denom - float(np.sum(analyzer.eigenvalues))) < 1e-12
    assert suffix
    assert "retained modes only" in suffix


def test_unknown_spatial_weight_type_raises_at_construction():
    """A typo is a construction error naming the accepted values, not silent keep."""
    with pytest.raises(ValueError, match=r"uniform.*polar|polar.*prescribed"):
        PODAnalyzer(
            file_path="dummy",
            data_loader=lambda _: {},
            spatial_weight_type="unifrom",
        )


def test_auto_spatial_weight_type_raises_at_construction():
    """'auto' is gone: raise, and name the real choices instead of only echoing."""
    with pytest.raises(ValueError, match=r"Accepted values:\s*uniform,\s*polar,\s*prescribed") as excinfo:
        PODAnalyzer(
            file_path="dummy",
            data_loader=lambda _: {},
            spatial_weight_type="auto",
        )
    # The message may echo 'auto' as the rejected input, but must not offer it
    # back as a remedy. Match only the accepted list, not the whole message.
    remedy = str(excinfo.value).split("Accepted values:", 1)[1]
    assert "auto" not in remedy


def test_array_without_type_still_prescribes():
    """spatial_weights=array with default type None remains a valid prescribe path."""
    data = {
        "q": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    weights = np.array([2.0, 3.0])
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weights=weights,
    )
    assert analyzer.spatial_weight_type == "prescribed"
    analyzer.load_and_preprocess()
    np.testing.assert_allclose(np.asarray(analyzer.W).ravel(), weights)


def test_prescribed_without_array_raises_at_construction():
    with pytest.raises(ValueError, match=r"spatial_weights"):
        PODAnalyzer(
            file_path="dummy",
            data_loader=lambda _: {},
            spatial_weight_type="prescribed",
        )


def test_conflicting_spatial_weight_type_and_array_raises():
    with pytest.raises(ValueError, match=r"conflict"):
        PODAnalyzer(
            file_path="dummy",
            data_loader=lambda _: {},
            spatial_weight_type="uniform",
            spatial_weights=np.array([1.0, 1.0]),
        )


def test_prescribed_wrong_length_raises_in_load_and_preprocess():
    data = {
        "q": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weights=np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match=r"n_space|length"):
        analyzer.load_and_preprocess()


def test_prescribed_negative_weight_raises_in_load_and_preprocess():
    data = {
        "q": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weights=np.array([1.0, -0.5]),
    )
    with pytest.raises(ValueError, match=r"negative"):
        analyzer.load_and_preprocess()


def _uniform_field(ns: int = 24, nx: int = 6, ny: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "q": rng.standard_normal((ns, nx * ny)),
        "x": np.linspace(0.5, 2.0, nx),
        "y": np.linspace(0.5, 1.5, ny),
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }


def _ramp_uniform_weights(x, y, z=None, n_space=None):
    if n_space is None:
        n_space = int(np.asarray(x).shape[0] * np.asarray(y).shape[0])
    return np.linspace(0.2, 3.0, int(n_space)).reshape(-1, 1)


def test_pod_uniform_metric_moves_eigenvalues(monkeypatch, tmp_path):
    """The uniform path must use the metric load_and_preprocess built.

    calculate_uniform_weights returns ones by contract, so a shape-only check
    stays green while perform_pod overwrites W with ones. Patch the builder to
    a ramp and demand the eigenvalues move — that is the provenance, not the
    shape.
    """
    field = _uniform_field()
    common = dict(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
        n_modes_save=4,
        use_parallel=False,
    )
    plain = PODAnalyzer(
        results_dir=str(tmp_path / "plain"),
        figures_dir=str(tmp_path / "plain"),
        **common,
    )
    plain.load_and_preprocess()
    plain.perform_pod()

    monkeypatch.setattr(
        "openmodalpy.core.base.calculate_uniform_weights",
        _ramp_uniform_weights,
    )
    ramped = PODAnalyzer(
        results_dir=str(tmp_path / "ramp"),
        figures_dir=str(tmp_path / "ramp"),
        **common,
    )
    ramped.load_and_preprocess()
    ramped.perform_pod()

    assert not np.allclose(plain.eigenvalues, ramped.eigenvalues), (
        "POD eigenvalues did not move when calculate_uniform_weights returned "
        "a ramp; the uniform path discarded the metric"
    )


def test_pod_total_energy_matches_full_untruncated_weighted_spectrum():
    """total_energy is the pre-truncation total, independent of the internal path.

    Build the sqrt(W)-weighted, mean-removed data by hand and take the full
    eigh of the temporal kernel; the sum of ALL its eigenvalues must match
    ``analyzer.total_energy`` even though the analyzer only keeps a few
    modes. Weights are non-uniform so the comparison actually exercises the
    metric.
    """
    rng = np.random.default_rng(0)
    n_samples, n_space = 5, 4
    q = rng.standard_normal((n_samples, n_space))
    weights = np.array([0.5, 1.0, 2.0, 3.5])
    data = {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_space,
        "Ny": 1,
        "Ns": n_samples,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weights=weights,
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    mean_removed = q - np.mean(q, axis=0)
    weighted = mean_removed * np.sqrt(weights)
    kernel = weighted @ weighted.T / n_samples
    full_eigenvalues = np.linalg.eigvalsh(kernel)
    expected_total_energy = float(np.sum(full_eigenvalues))

    assert analyzer.total_energy == pytest.approx(expected_total_energy, rel=1e-13)
    assert analyzer.energy_captured_fraction <= 1 + 1e-12
