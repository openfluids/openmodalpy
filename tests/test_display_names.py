"""Display-name map and drawn figure titles for POD / mPOD."""

from __future__ import annotations

import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import logging

import numpy as np
import pytest

from openmodalpy.mpod import MPODAnalyzer
from openmodalpy.pod import PODAnalyzer
from openmodalpy.specs import METHOD_REGISTRY, display_name_for

# Bare "POD" not inside mPOD / PSD-POD / ST-POD.
BARE_POD = re.compile(r"(?<![A-Za-z-])POD")


def test_display_name_for_known_types():
    assert display_name_for("pod") == "POD"
    assert display_name_for("mpod") == "mPOD"
    assert display_name_for("stpod") == "ST-POD"
    assert display_name_for("psd_pod") == "PSD-POD"
    assert display_name_for("dmd") == "DMD"


def test_display_name_for_unknown_falls_back_to_upper():
    assert display_name_for("nope") == "NOPE"
    assert display_name_for("custom_method") == "CUSTOM_METHOD"


def test_method_registry_identity_reexport():
    from openmodalpy.commands import METHOD_REGISTRY as from_commands

    assert from_commands is METHOD_REGISTRY
    assert METHOD_REGISTRY["mpod"].display_name == "mPOD"


def _synthetic_2d(Ns: int = 24, Nx: int = 4, Ny: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return {
        "q": rng.standard_normal((Ns, Nx * Ny)),
        "x": np.linspace(0.0, 1.0, Nx),
        "y": np.linspace(0.0, 1.0, Ny),
        "dt": 1.0,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }


def _make_mpod(tmp_path, data: dict) -> MPODAnalyzer:
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


def _make_pod(tmp_path, data: dict) -> PODAnalyzer:
    return PODAnalyzer(
        file_path="dummy.h5",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=4,
        use_parallel=False,
    )


def _prepare(analyzer) -> None:
    analyzer.load_and_preprocess()
    analyzer.perform_pod()


def _assert_label_titles(titles: list[str], label: str) -> None:
    assert titles, "no titles were drawn"
    if label == "mPOD":
        offenders = [t for t in titles if BARE_POD.search(t)]
        assert not offenders, f"bare POD in mPOD titles: {offenders}"
        assert any("mPOD" in t for t in titles), titles
    elif label == "POD":
        assert any(BARE_POD.search(t) for t in titles), titles
        assert not any("mPOD" in t for t in titles), titles


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_eigenvalue_spectrum_title(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_eigenvalues()
    assert any(t == f"{label} Eigenvalue Spectrum" for t in titles), titles
    _assert_label_titles(titles, label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_plot_modes_titles(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_modes(plot_n_modes=1, modes_per_fig=1)
    mode_titles = [t for t in titles if "Mode" in t and "[" in t]
    assert mode_titles, titles
    assert any(t.startswith(f"{label} Mode ") for t in mode_titles), mode_titles
    _assert_label_titles(mode_titles, label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_modes_grid_suptitle(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_modes_grid(energy_threshold=99.5)
    assert any(t.startswith(f"{label} Modes up to") for t in titles), titles
    _assert_label_titles([t for t in titles if "Modes up to" in t], label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_time_coefficient_titles(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_time_coefficients(n_coeffs_to_plot=1)
    assert any(t == f"Temporal Coefficient for {label} Mode 1" for t in titles), titles
    _assert_label_titles([t for t in titles if "Temporal Coefficient" in t], label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_phase_portrait_titles(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    # Force a pair: threshold 0 accepts any correlation.
    with capture_titles() as titles:
        analyzer.plot_mode_pair_phase(start_mode=1, threshold=0.0)
    assert any(t.startswith(f"{label} Phase Portrait Modes") for t in titles), titles
    _assert_label_titles([t for t in titles if "Phase Portrait" in t], label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_cumulative_energy_title(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_cumulative_energy()
    assert any(t == f"Cumulative Energy of {label} Modes" for t in titles), titles
    _assert_label_titles(titles, label)


@pytest.mark.parametrize("maker,label", [(_make_mpod, "mPOD"), (_make_pod, "POD")])
def test_reconstruction_error_title(tmp_path, maker, label, capture_titles):
    analyzer = maker(tmp_path, _synthetic_2d())
    _prepare(analyzer)
    with capture_titles() as titles:
        analyzer.plot_reconstruction_error()
    want = f"Data Reconstruction Error vs. Number of {label} Modes"
    assert any(t == want for t in titles), titles
    _assert_label_titles(titles, label)


def test_pod_py_3d_titles_have_no_bare_pod_mode_literal():
    """Structural guard for the two 3-D title sites (volumetric path is heavy)."""
    pod_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "openmodalpy" / "pod.py"
    src = pod_path.read_text(encoding="utf-8")
    # The 3-D builder must resolve the label via the helper, not a hardcoded POD.
    assert "display_name_for(self.analysis_type)" in src
    assert 'f"POD Mode' not in src
    assert "f'POD Mode" not in src
    assert '"POD Mode' not in src
    assert "'POD Mode" not in src
    # Both 3-D sites — the "E=%" branch and the plain branch — must build their
    # title from the resolved label. Counting them is what catches one of the two
    # being hardcoded back; `"Mode {" in src` would stay true either way.
    built_from_label = src.count('title = f"{label} Mode {mode_idx + 1}')
    assert built_from_label == 2, f"expected 2 label-built 3-D titles, found {built_from_label}"


def test_run_analysis_completion_log_uses_display_name(tmp_path, caplog):
    """The unified run_analysis epilogue logs the resolved display name.

    The old per-class epilogues called print_summary; the unified seam keeps
    the display-name discipline in its start/complete banners instead.
    """
    analyzer = _make_mpod(tmp_path, _synthetic_2d())
    with caplog.at_level("INFO"):
        analyzer.run_analysis()
    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("Starting mPOD analysis" in m for m in info_msgs), info_msgs
    assert not any("Starting POD analysis" in m for m in info_msgs), info_msgs
    assert any("mPOD analysis and plotting completed successfully" in m for m in info_msgs), info_msgs
