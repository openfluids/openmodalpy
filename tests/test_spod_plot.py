import h5py
import numpy as np

from openmodalpy import SPODAnalyzer


def test_plot_eigenvalues(tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values."""
    rng = np.random.default_rng(0)
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
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_eigenvalues()
    expected = tmp_path / "dummy_SPOD_eigenvalues_nfft4_noverlap0.0.png"
    assert expected.exists()


def test_plot_modes_and_timecoeffs(tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values."""
    rng = np.random.default_rng(1)
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
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_modes(plot_n_modes=1)
    freq_idx = int(np.argmax(analyzer.eigenvalues[:, 0]))
    expected_modes = tmp_path / f"dummy_SPOD_mode1_freq{freq_idx}_q.png"
    assert expected_modes.exists()
    analyzer.plot_time_coefficients()
    expected_time = tmp_path / (f"dummy_SPOD_timecoeffs_freq{freq_idx}_nfft4_noverlap0.0.png")
    assert expected_time.exists()


def test_plot_reconstruction_error(tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values."""
    rng = np.random.default_rng(2)
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
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_reconstruction_error()
    freq_idx = int(np.argmax(analyzer.eigenvalues[:, 0]))
    expected = tmp_path / (f"dummy_SPOD_reconstruction_error_freq{freq_idx}_nfft4_noverlap0.0.png")
    assert expected.exists()


def test_save_results_records_spod_contract(tmp_path):
    """Smoke test: asserts execution and artifact only, not numerical values."""
    rng = np.random.default_rng(3)
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
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.save_results()

    with h5py.File(tmp_path / "dummy_Nfft4_ovlap0.0_8snapshots_spod.hdf5", "r") as handle:
        assert handle.attrs["lift_kind"] == "block_fourier_realizations"
        assert bool(handle.attrs["uses_mean_subtraction"])
        assert bool(handle.attrs["uses_spatial_metric_in_second_order_operator"])
        assert handle.attrs["spectral_estimator"] == "welch_block_average"


def test_spod_save_load_roundtrip_arrays(tmp_path):
    """SPOD save_results() → load_results() restores arrays to machine precision.

    Both take an optional filename; when omitted they share the same auto-name.
    """
    rng = np.random.default_rng(11)
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
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.save_results()

    reloaded = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=4,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    reloaded.load_results()

    np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
    np.testing.assert_array_equal(reloaded.modes, analyzer.modes)
    np.testing.assert_array_equal(reloaded.freq, analyzer.freq)
    np.testing.assert_array_equal(reloaded.St, analyzer.St)
