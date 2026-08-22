"""Library-API tests for PSDPODAnalyzer."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from openmodalpy import PSDPODAnalyzer, read_results, run_from_config
from openmodalpy.example_data import generate_example_dataset


def _write_jsonc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
def test_psdpod_analyzer_runs_from_library_api(tmp_path: Path) -> None:
    """Construct and run PSDPODAnalyzer without going through the CLI."""
    data = generate_example_dataset("double_gyre", {"Nx": 8, "Ny": 4, "Nt": 24})

    analyzer = PSDPODAnalyzer(
        file_path="double_gyre",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        nfft=8,
        overlap=0.5,
        n_modes_save=3,
    )
    analyzer.run_analysis()

    assert analyzer.modes.shape == (8 * 4, 3)
    assert analyzer.eigenvalues.shape == (3,)
    assert analyzer.time_coefficients.shape[1] == 3
    assert analyzer.results_path is not None
    assert Path(analyzer.results_path).is_file()

    res = read_results(analyzer.results_path)
    assert res.attrs.get("analysis_type") == "psd_pod"
    assert res.attrs.get("lift_kind") == "flattened_block_fourier_realizations"
    assert np.asarray(res.modes).shape == analyzer.modes.shape
    np.testing.assert_array_equal(np.asarray(res.eigenvalues), analyzer.eigenvalues)


def test_psdpod_config_and_api_agree(tmp_path: Path) -> None:
    """A config run and a direct API run on the same case produce identical arrays."""
    params = {"Nx": 10, "Ny": 6, "Nt": 40}
    nfft, overlap, n_modes = 8, 0.5, 4

    # Config / command path
    config_path = tmp_path / "psd_pod_case.jsonc"
    results_cfg = tmp_path / "results_cfg"
    figures_cfg = tmp_path / "figures_cfg"
    _write_jsonc(
        config_path,
        {
            "name": "PSD-POD agreement case",
            "description": "Config vs API on the same analytical field",
            "case": {
                "name": "agree",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": params},
                "spatial_weight_type": "uniform",
                "n_modes_save": n_modes,
                "nfft": nfft,
                "overlap": overlap,
                "generate_plots": False,
                "results_root": str(results_cfg),
                "figures_root": str(figures_cfg),
            },
            "runs": [{"id": "psd", "method": "psd-pod"}],
        },
    )
    outcomes = run_from_config(config_path)
    assert len(outcomes) == 1 and outcomes[0].results_path is not None
    cfg_path = Path(outcomes[0].results_path)

    # Direct library API on the same generator output
    data = generate_example_dataset("double_gyre", params)
    api_root = tmp_path / "results_api"
    analyzer = PSDPODAnalyzer(
        file_path="agree",
        results_dir=str(api_root),
        figures_dir=str(tmp_path / "figures_api"),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        nfft=nfft,
        overlap=overlap,
        n_modes_save=n_modes,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_psd_pod()
    analyzer.save_results()
    assert analyzer.results_path is not None

    with h5py.File(cfg_path, "r") as cfg, h5py.File(analyzer.results_path, "r") as api:
        for name in ("modes", "eigenvalues", "time_coefficients", "freq", "st"):
            np.testing.assert_array_equal(cfg[name][()], api[name][()])
        for key in (
            "analysis_type",
            "lift_kind",
            "n_fourier_realizations",
            "uses_mean_subtraction",
            "blockwise_mean",
            "spectral_estimator",
        ):
            assert cfg.attrs[key] == api.attrs[key]


def test_psdpod_save_results_records_prescribed_metric(tmp_path: Path) -> None:
    """A saved PSD-POD file holds the prescribed W that produced it, as a column."""
    data = generate_example_dataset("double_gyre", {"Nx": 8, "Ny": 4, "Nt": 24})
    n_space = 8 * 4
    weights = np.linspace(0.5, 2.0, n_space)

    analyzer = PSDPODAnalyzer(
        file_path="double_gyre",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        data_loader=lambda _: data,
        spatial_weight_type="prescribed",
        spatial_weights=weights,
        nfft=8,
        overlap=0.5,
        n_modes_save=3,
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_psd_pod()
    analyzer.save_results("psd_pod_w.hdf5")

    assert analyzer.results_path is not None
    with h5py.File(analyzer.results_path, "r") as handle:
        assert "W" in handle
        stored = np.asarray(handle["W"])
    used = np.asarray(analyzer.W)
    np.testing.assert_array_equal(stored, used)
    assert stored.shape == (n_space, 1)
    # Pin the file to the weights that were ASKED for, not only to whatever sits
    # on the object at save time. Without this, a save that faithfully records a
    # metric the run had already replaced would still pass.
    np.testing.assert_array_equal(stored, weights.reshape(-1, 1))


def test_psdpod_load_results_reads_legacy_040_layout(tmp_path: Path) -> None:
    """A 0.4.0-layout file (capitalised dataset names) loads through load_results."""
    legacy = tmp_path / "legacy_psd_pod.hdf5"
    n_space, n_modes, n_realizations = 8 * 4, 3, 10
    rng = np.random.default_rng(7)
    with h5py.File(legacy, "w") as handle:
        handle.create_dataset("Modes", data=rng.standard_normal((n_space, n_modes)) + 1j)
        handle.create_dataset("Eigenvalues", data=np.abs(rng.standard_normal(n_modes)))
        handle.create_dataset("TimeCoefficients", data=rng.standard_normal((n_realizations, n_modes)))
        handle.create_dataset("Freq", data=np.linspace(0.0, 1.0, 5))
        handle.create_dataset("St", data=np.linspace(0.0, 2.0, 5))
        handle.create_dataset("Weights", data=np.ones(n_space))
        handle.attrs["analysis_type"] = "psd_pod"

    analyzer = PSDPODAnalyzer(
        file_path="double_gyre",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        data_loader=lambda _: generate_example_dataset("double_gyre", {"Nx": 8, "Ny": 4, "Nt": 24}),
        spatial_weight_type="uniform",
        nfft=8,
        overlap=0.5,
        n_modes_save=n_modes,
    )
    analyzer.load_and_preprocess()

    # Absolute filename bypasses the results_dir join, as in the SPOD loader tests.
    with pytest.warns(DeprecationWarning, match="legacy name"):
        analyzer.load_results(str(legacy))

    assert analyzer.modes.shape == (n_space, n_modes)
    assert analyzer.eigenvalues.shape == (n_modes,)
    assert analyzer.time_coefficients.shape == (n_realizations, n_modes)
    assert analyzer.freq.shape == (5,)
    assert analyzer.St.shape == (5,)
    np.testing.assert_array_equal(analyzer.W, np.ones((n_space, 1)))
