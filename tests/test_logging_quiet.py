"""Silence checks for the logging seam (phase 1: core/ + perform_spod guard).

Under default logging configuration the library must not write to stdout.
CLI installs its own handler; library import paths do not.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer, DMDAnalyzer, MPODAnalyzer, PODAnalyzer, SPODAnalyzer, STPODAnalyzer
from openmodalpy.core.base import BaseAnalyzer, print_summary
from openmodalpy.core.io import MATDataLoader


def _synthetic_data(Ns: int = 32, Nspace: int = 4, dt: float = 1.0) -> dict:
    rng = np.random.default_rng(0)
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    return {
        "q": rng.standard_normal((Ns, Nspace)),
        "x": np.linspace(0.0, 1.0, Nx),
        "y": np.linspace(0.0, 1.0, Ny),
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }


def _make_base(tmp_path, data: dict) -> BaseAnalyzer:
    """BaseAnalyzer only — no analyzer-module prints (those are phase 2)."""
    return BaseAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )


def test_load_and_preprocess_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    BaseAnalyzer.load_and_preprocess logs shapes; stdout stays empty.
    """
    analyzer = _make_base(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_and_preprocess_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    The stdout checks above would also pass if every print AND every logger call
    were simply removed. This pins the other half: the same call still reports
    what it loaded, on the module logger, at INFO.
    """
    analyzer = _make_base(tmp_path, _synthetic_data())
    with caplog.at_level(logging.INFO, logger="openmodalpy.core.base"):
        analyzer.load_and_preprocess()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "load_and_preprocess emitted no INFO records at all"
    assert any("Data loaded" in r.getMessage() for r in info), [r.getMessage() for r in info]


def test_compute_fft_blocks_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    BaseAnalyzer.compute_fft_blocks logs timings; stdout stays empty.
    """
    analyzer = _make_base(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    capsys.readouterr()
    analyzer.compute_fft_blocks()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_summary_is_quiet(capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    print_summary routes through the module logger; stdout stays empty.
    """
    print_summary("POD", "/tmp/results", "/tmp/figures")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_mat_loader_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    MATDataLoader.load logs progress; stdout stays empty under default logging.
    """
    import h5py

    file_path = tmp_path / "quiet.mat"
    q = np.arange(12, dtype=float).reshape(6, 2)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("u", data=q)
        f.create_dataset("x", data=np.array([0.0, 0.5, 1.0]))
        f.create_dataset("y", data=np.array([0.0, 1.0]))
        f.create_dataset("dt", data=np.array([[0.25]]))

    data = MATDataLoader().load(str(file_path))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert data["Ns"] == 2


def test_perform_spod_raises_when_qhat_not_computed(tmp_path):
    """Calling perform_spod before FFT blocks exist raises RuntimeError.

    Previously this path printed one line and returned None, so the caller
    continued as if an analysis had run.
    """
    data = _synthetic_data()
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    assert analyzer.qhat is None or analyzer.qhat.size == 0

    with pytest.raises(RuntimeError, match=r"qhat not computed|compute_fft_blocks|run\(compute_fft"):
        analyzer.perform_spod()


def _make_pod(tmp_path, data: dict) -> PODAnalyzer:
    """PODAnalyzer on synthetic data — same shape as the base helper."""
    return PODAnalyzer(
        file_path="dummy.h5",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=4,
        use_parallel=False,
    )


def test_pod_run_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    End-to-end POD (load + perform_pod) must leave stdout empty.
    """
    analyzer = _make_pod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_pod_run_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    After perform_pod the openmodalpy.pod logger should still report the
    completed decomposition (mode count) at INFO.
    """
    analyzer = _make_pod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    with caplog.at_level(logging.INFO, logger="openmodalpy.pod"):
        analyzer.perform_pod()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "perform_pod emitted no INFO records at all"
    # Pin the actual mode count, not just the word "Computed" — a message like
    # "Computed spatial weights" would satisfy a bare substring check.
    expected = f"Computed {analyzer.modes.shape[1]} POD modes"
    assert any(expected in r.getMessage() for r in info), [r.getMessage() for r in info]


def _make_spod(tmp_path, data: dict) -> SPODAnalyzer:
    """SPODAnalyzer on synthetic data — load + FFT + perform_spod path."""
    return SPODAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )


def test_spod_run_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    End-to-end SPOD (load + FFT + perform_spod) must leave stdout empty.
    """
    analyzer = _make_spod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_spod_run_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    After perform_spod the openmodalpy.spod logger should still report the
    completed eigenvalue decomposition at INFO, including the wall-clock duration.
    """
    analyzer = _make_spod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    with caplog.at_level(logging.INFO, logger="openmodalpy.spod"):
        analyzer.perform_spod()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "perform_spod emitted no INFO records at all"
    # Pin the completion duration (a real reported quantity), not bare existence
    # of any INFO record. Deleting that log line turns this red.
    pattern = re.compile(r"SPOD eigenvalue decomposition completed in \d+\.\d{2} seconds")
    assert any(pattern.search(r.getMessage()) for r in info), [r.getMessage() for r in info]


def _make_stpod(tmp_path, data: dict) -> STPODAnalyzer:
    """STPODAnalyzer on synthetic data — delay-embedded POD path."""
    return STPODAnalyzer(
        file_path="dummy.h5",
        embedding_dim=2,
        n_modes_save=4,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )


def test_stpod_run_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    End-to-end ST-POD (load + perform_stpod) must leave stdout empty.
    """
    analyzer = _make_stpod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.perform_stpod()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stpod_run_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    After perform_stpod the openmodalpy.stpod logger should still report the
    mode count and energy fraction at INFO.
    """
    analyzer = _make_stpod(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    with caplog.at_level(logging.INFO, logger="openmodalpy.stpod"):
        analyzer.perform_stpod()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "perform_stpod emitted no INFO records at all"
    # Pin mode count and energy percentage — a bare "Computed" would not bite.
    expected = (
        f"Computed {analyzer.n_modes_save} ST-POD modes "
        f"({100.0 * analyzer.energy_captured_fraction:.2f}% of total energy)."
    )
    assert any(expected in r.getMessage() for r in info), [r.getMessage() for r in info]


def _make_bsmd(tmp_path, data: dict) -> BSMDAnalyzer:
    """BSMDAnalyzer on synthetic data — load + FFT + perform_bsmd path."""
    return BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=[(0, 0, 0), (1, -1, 0), (1, 1, 2)],
        use_parallel=False,
    )


def test_bsmd_run_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    End-to-end BSMD (load + FFT + perform_bsmd) must leave stdout empty.
    """
    analyzer = _make_bsmd(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_bsmd()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_bsmd_run_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    After perform_bsmd the openmodalpy.bsmd logger should still report the
    completed analysis at INFO, including the wall-clock duration.
    """
    analyzer = _make_bsmd(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    with caplog.at_level(logging.INFO, logger="openmodalpy.bsmd"):
        analyzer.perform_bsmd()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "perform_bsmd emitted no INFO records at all"
    # Pin the completion duration (a real reported quantity), not bare existence
    # of any INFO record. Deleting that log line turns this red.
    pattern = re.compile(r"BSMD analysis completed in \d+\.\d{2} seconds")
    assert any(pattern.search(r.getMessage()) for r in info), [r.getMessage() for r in info]


def _make_dmd(tmp_path, data: dict) -> DMDAnalyzer:
    """DMDAnalyzer on synthetic data — load + perform_dmd + save_results path."""
    return DMDAnalyzer(
        file_path="dummy.h5",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=4,
        rank=4,
    )


def test_dmd_run_is_quiet_on_stdout(tmp_path, capsys):
    """Smoke test: asserts execution and artifact only, not numerical values.

    End-to-end DMD (load + perform_dmd + save) must leave stdout empty.
    """
    analyzer = _make_dmd(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()
    analyzer.save_results()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_dmd_run_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    After save_results the openmodalpy.dmd logger should still report the
    result path at INFO (perform_dmd itself has no progress prints to convert).
    """
    analyzer = _make_dmd(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()
    with caplog.at_level(logging.INFO, logger="openmodalpy.dmd"):
        analyzer.save_results()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "save_results emitted no INFO records at all"
    # Pin the results path (a real reported quantity). Deleting that log line turns this red.
    pattern = re.compile(r"DMD results saved to .+/.*_dmd\.hdf5")
    assert any(pattern.search(r.getMessage()) for r in info), [r.getMessage() for r in info]


def _make_mpod(tmp_path, data: dict) -> MPODAnalyzer:
    """MPODAnalyzer on synthetic data — same shape as the POD helper."""
    return MPODAnalyzer(
        file_path="dummy.h5",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=4,
        band_edges=[0.0, 0.5],
        use_parallel=False,
    )


def test_mpod_banner_does_not_say_pod(tmp_path, caplog):
    """mPOD run_analysis banner uses display_name_for → mPOD, not POD/MPOD."""
    analyzer = _make_mpod(tmp_path, _synthetic_data())
    with caplog.at_level(logging.INFO, logger="openmodalpy.pod"):
        analyzer.run_analysis(plot_n_modes_spatial=1, plot_n_coeffs_time=1)

    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Starting mPOD analysis" in m for m in info_msgs), info_msgs
    assert not any("Starting POD analysis" in m for m in info_msgs), info_msgs
    assert not any("Starting MPOD analysis" in m for m in info_msgs), info_msgs


def test_pod_banner_still_says_pod(tmp_path, caplog):
    """POD run_analysis banner remains Starting POD analysis."""
    analyzer = _make_pod(tmp_path, _synthetic_data())
    with caplog.at_level(logging.INFO, logger="openmodalpy.pod"):
        analyzer.run_analysis(plot_n_modes_spatial=1, plot_n_coeffs_time=1)

    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Starting POD analysis" in m for m in info_msgs), info_msgs
