"""Drawn POD energy percentages must use the pre-truncation total.

The helper-level tests stay green under a per-site revert of the plotting
denominator. These tests pin the numbers that actually leave the plotters.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.axes import Axes

from openmodalpy.pod import PODAnalyzer


def _synthetic_data(Ns: int = 32, Nspace: int = 4, dt: float = 1.0) -> dict:
    """Same layout as tests/test_logging_quiet._synthetic_data (gate step 1)."""
    rng = np.random.default_rng(0)
    nx = int(np.sqrt(Nspace))
    ny = Nspace // nx
    return {
        "q": rng.standard_normal((Ns, Nspace)),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.linspace(0.0, 1.0, ny),
        "dt": dt,
        "Nx": nx,
        "Ny": ny,
        "Ns": Ns,
    }


def _make_truncated_pod(tmp_path, data: dict, *, n_modes_save: int = 3) -> PODAnalyzer:
    analyzer = PODAnalyzer(
        file_path="dummy.h5",
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes_save,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    return analyzer


@pytest.fixture
def truncated_pod(tmp_path):
    """Fixture chosen so the denominators differ (retained/total ~ 0.5306)."""
    data = _synthetic_data(Ns=32, Nspace=9)
    analyzer = _make_truncated_pod(tmp_path, data, n_modes_save=3)
    retained = float(np.sum(analyzer.eigenvalues))
    total = float(analyzer.total_energy)
    ratio = retained / total
    assert 0.3 < ratio < 0.8, f"fixture does not separate denominators: {ratio}"
    assert ratio == pytest.approx(0.5306271829005565, rel=1e-9)
    return analyzer


def _pct_of_total(analyzer: PODAnalyzer) -> np.ndarray:
    total = float(analyzer.total_energy)
    assert total > 0.0
    return 100.0 * np.asarray(analyzer.eigenvalues, dtype=float) / total


def _cum_pct_of_total(analyzer: PODAnalyzer) -> np.ndarray:
    return np.cumsum(_pct_of_total(analyzer))


def test_fixture_separates_denominators(truncated_pod):
    """Sanity: retained sum and pre-truncation total are not interchangeable."""
    retained = float(np.sum(truncated_pod.eigenvalues))
    total = float(truncated_pod.total_energy)
    assert retained == pytest.approx(4.949501618601282, rel=1e-9)
    assert total == pytest.approx(9.327644301119143, rel=1e-9)
    assert retained / total == pytest.approx(0.5306271829005565, rel=1e-9)


def test_plot_eigenvalues_ydata_uses_total_energy(truncated_pod, capture_line_ydata):
    """pod.py plot_eigenvalues: drawn line is 100 * lambda / total_energy."""
    expected = _pct_of_total(truncated_pod)
    ylabels: list[str] = []
    orig_ylabel = Axes.set_ylabel

    def set_ylabel(self, label, *a, **k):
        ylabels.append(str(label))
        return orig_ylabel(self, label, *a, **k)

    Axes.set_ylabel = set_ylabel
    try:
        with capture_line_ydata() as ydatas:
            truncated_pod.plot_eigenvalues()
    finally:
        Axes.set_ylabel = orig_ylabel

    assert ydatas, "no line y-data was drawn"
    assert np.asarray(ydatas[0]) == pytest.approx(expected, rel=1e-9)
    # Known total_energy → empty label_suffix on the y-axis.
    assert ylabels
    assert ylabels[0] == "Normalized Eigenvalue (Energy Percentage %)"


def test_plot_modes_title_energy_uses_total_energy(truncated_pod, capture_titles):
    """pod.py plot_modes: title Energy/Cumulative come from total_energy."""
    energy = float(_pct_of_total(truncated_pod)[0])
    cum = float(_cum_pct_of_total(truncated_pod)[0])
    with capture_titles() as titles:
        truncated_pod.plot_modes(plot_n_modes=1, modes_per_fig=1)
    mode_titles = [t for t in titles if "Energy:" in t and "Cumulative:" in t]
    assert mode_titles, titles
    title = mode_titles[0]
    m = re.search(r"Energy:\s*([0-9.]+)%\s*\|\s*Cumulative:\s*([0-9.]+)%", title)
    assert m, title
    assert float(m.group(1)) == pytest.approx(float(f"{energy:.2f}"), rel=1e-9)
    assert float(m.group(2)) == pytest.approx(float(f"{cum:.2f}"), rel=1e-9)


def test_plot_modes_pair_detailed_title_energy_uses_total_energy(truncated_pod, capture_titles):
    """pod.py plot_modes_pair_detailed: E=/Cum= title numbers use total_energy."""
    energy = float(_pct_of_total(truncated_pod)[0])
    cum = float(_cum_pct_of_total(truncated_pod)[0])
    with capture_titles() as titles:
        truncated_pod.plot_modes_pair_detailed(plot_n_modes=2)
    mode_titles = [t for t in titles if "E=" in t and "Cum=" in t]
    assert mode_titles, titles
    m = re.search(r"E=([0-9.]+)%\s+Cum=([0-9.]+)%", mode_titles[0])
    assert m, mode_titles[0]
    assert float(m.group(1)) == pytest.approx(float(f"{energy:.2f}"), rel=1e-9)
    assert float(m.group(2)) == pytest.approx(float(f"{cum:.2f}"), rel=1e-9)


def test_plot_modes_3d_title_energy_uses_total_energy(tmp_path, monkeypatch):
    """pod.py 3-D builder: title_prefix E=X% is 100 * lambda / total_energy."""
    rng = np.random.default_rng(0)
    nx = ny = nz = 2
    data = {
        "q": rng.standard_normal((32, nx * ny * nz)),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.linspace(0.0, 1.0, ny),
        "z": np.linspace(0.0, 1.0, nz),
        "dt": 1.0,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "Ns": 32,
    }
    analyzer = _make_truncated_pod(tmp_path, data, n_modes_save=3)
    retained = float(np.sum(analyzer.eigenvalues))
    total = float(analyzer.total_energy)
    assert 0.3 < retained / total < 0.8

    drawn_titles: list[str] = []

    def _fake_plot_modes_3d(kind, items, *a, **k):
        for item in items:
            drawn_titles.append(str(item["title_prefix"]))

    monkeypatch.setattr("openmodalpy.pod.plot_modes_3d", _fake_plot_modes_3d)
    analyzer.plot_modes_3d_slices(plot_n_modes=1)

    assert drawn_titles, "3-D builder produced no titles"
    energy = 100.0 * float(analyzer.eigenvalues[0]) / total
    m = re.search(r"E=([0-9.]+)%", drawn_titles[0])
    assert m, drawn_titles[0]
    assert float(m.group(1)) == pytest.approx(float(f"{energy:.2f}"), rel=1e-9)


def test_plot_modes_grid_panel_count_uses_total_energy(truncated_pod, capture_titles):
    """pod.py plot_modes_grid: total_energy decides both the panel count and the numbers.

    At energy_threshold=50 the correct cumulative curve needs 3 panels; the
    retained-sum curve only needs 2. Assert the drawn panel count is 3, AND the
    per-panel E=/Cum= numbers: the count alone stays green if someone reverts only
    the title arithmetic and leaves the searchsorted call correct (measured).
    """
    cum = _cum_pct_of_total(truncated_pod)
    thr = 50.0
    expected_panels = int(np.searchsorted(cum, thr, side="right")) + 1
    assert expected_panels == 3
    # Prove the buggy denominator would disagree, so this test has teeth.
    buggy = 100.0 * np.cumsum(truncated_pod.eigenvalues) / float(np.sum(truncated_pod.eigenvalues))
    buggy_panels = int(np.searchsorted(buggy, thr, side="right")) + 1
    assert buggy_panels == 2

    with capture_titles() as titles:
        truncated_pod.plot_modes_grid(energy_threshold=thr)
    panel_titles = [t for t in titles if re.search(r"Mode\s+\d+", t) and "E=" in t]
    assert len(panel_titles) == expected_panels, titles

    # Each panel's own E=/Cum= must come from the pre-truncation total too.
    pct = _pct_of_total(truncated_pod)
    for k, panel in enumerate(panel_titles):
        m = re.search(r"E=([0-9.]+)%\s+Cum=([0-9.]+)%", panel)
        assert m, panel
        assert float(m.group(1)) == pytest.approx(float(f"{pct[k]:.2f}"), rel=1e-9), panel
        assert float(m.group(2)) == pytest.approx(float(f"{cum[k]:.2f}"), rel=1e-9), panel


def test_plot_cumulative_energy_ydata_uses_total_energy(truncated_pod, capture_line_ydata):
    """pod.py plot_cumulative_energy: last drawn point is ~53.063%, not 100%."""
    expected = _cum_pct_of_total(truncated_pod)
    with capture_line_ydata() as ydatas:
        truncated_pod.plot_cumulative_energy()
    assert ydatas, "no cumulative line was drawn"
    y = np.asarray(ydatas[0], dtype=float)
    assert y == pytest.approx(expected, rel=1e-9)
    assert y[-1] == pytest.approx(53.062718290055656, rel=1e-9)
    assert y[-1] != pytest.approx(100.0, abs=1e-6)
