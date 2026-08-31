"""Shared pytest fixtures for openmodalpy tests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@pytest.fixture(autouse=True)
def _reseed_numpy_rng():
    """Reseed NumPy's global RNG before every test from OMPY_TEST_RNG_JITTER.

    Default jitter is 0 (deterministic). Setting the env var to a different int
    changes the global stream so tests that assert on unseeded data fail under
    one jitter and not another — the discriminator for accidental unseeded draws.
    """
    jitter = int(os.environ.get("OMPY_TEST_RNG_JITTER", "0"))
    np.random.seed(jitter)
    yield


@pytest.fixture(autouse=True)
def _run_tests_in_tmp_cwd(tmp_path, monkeypatch):
    """Run every test with CWD under pytest's tmp_path.

    Analyzers default results_dir/figures_dir to relative paths (./results,
    ./figures). BaseAnalyzer makedirs those on construct; most tests build an
    analyzer without overriding them, so a full suite otherwise leaves
    results/ and figures/ in the repo root (gitignored, so invisible to git).
    Monkeypatching the RESULTS_DIR_* constants does not help: they are bound
    as default argument values at function definition time. Changing CWD once
    here makes the defaults resolve under the temp dir instead.
    """
    monkeypatch.chdir(tmp_path)


def _analytic_rank2_field(Ns: int, Nspace: int) -> dict:
    """Deterministic rank-2 travelling-wave field for POD/ST-POD tests.

    Mean is non-zero so mean-subtraction is exercised. Spatial points exceed or
    trail snapshots depending on the (Ns, Nspace) pair the caller chooses.
    """
    t = np.linspace(0.0, 2.0 * np.pi, Ns, endpoint=False)
    x = np.linspace(0.0, 1.0, Nspace)
    q = 1.0 + np.outer(np.sin(t), np.sin(2.0 * np.pi * x)) + 0.4 * np.outer(np.cos(3.0 * t), np.cos(2.0 * np.pi * x))
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": x,
        "y": np.array([0.0]),
        "dt": 0.1,
        "Nx": Nspace,
        "Ny": 1,
        "Ns": Ns,
    }


@pytest.fixture
def small_pod_field():
    """POD-able analytic field with Ns > Nspace (spatial-kernel default)."""
    return _analytic_rank2_field(Ns=16, Nspace=10)


@pytest.fixture
def small_stpod_field():
    """ST-POD-able analytic field long enough for modest delay embedding."""
    return _analytic_rank2_field(Ns=40, Nspace=12)


@contextmanager
def _capture_titles():
    """Record every string drawn via set_title / suptitle / plt.title."""
    seen: list[str] = []
    orig_ax, orig_fig, orig_plt = Axes.set_title, Figure.suptitle, plt.title

    def ax_title(self, label, *a, **k):
        seen.append(str(label))
        return orig_ax(self, label, *a, **k)

    def fig_title(self, t, *a, **k):
        seen.append(str(t))
        return orig_fig(self, t, *a, **k)

    def plt_title(label, *a, **k):
        seen.append(str(label))
        return orig_plt(label, *a, **k)

    Axes.set_title, Figure.suptitle, plt.title = ax_title, fig_title, plt_title
    try:
        yield seen
    finally:
        Axes.set_title, Figure.suptitle, plt.title = orig_ax, orig_fig, orig_plt
        plt.close("all")


@pytest.fixture
def capture_titles():
    """Fixture exposing the shared title-capture context manager."""
    return _capture_titles


@contextmanager
def _capture_line_ydata():
    """Record y-data arrays from every Axes.plot call (before figures close)."""
    seen: list[np.ndarray] = []
    orig_plot = Axes.plot

    def plot(self, *a, **k):
        lines = orig_plot(self, *a, **k)
        for line in lines:
            seen.append(np.asarray(line.get_ydata(), dtype=float).copy())
        return lines

    Axes.plot = plot
    try:
        yield seen
    finally:
        Axes.plot = orig_plot
        plt.close("all")


@pytest.fixture
def capture_line_ydata():
    """Fixture exposing the shared line y-data capture context manager."""
    return _capture_line_ydata


# --- Evidence taxonomy enforcement ---
# A test that replays the library's own formula in the library's own order is a
# refactoring guard, not physics evidence. Every such test must be listed here
# AND carry @pytest.mark.characterization; the hook below fails collection when
# either side drifts (unlabelled twin, or marker used outside the registry).
# Oracle tests carry @pytest.mark.oracle instead; counts:
#   uv run pytest -q -m characterization --collect-only | tail -1
#   uv run pytest -q -m oracle --collect-only | tail -1
CHARACTERISATION_REGISTRY: dict[str, set[str]] = {
    "test_psd_pod_numerics.py": {
        "test_psd_pod_positive_nonuniform_metric",
        "test_psd_pod_isolated_zero_weight_station",
    },
    "test_bsmd_core.py": {"test_static_bsmd_core_small"},
    "test_spod_function.py": {"test_spod_modes_deterministic_and_canonical"},
    "test_dmd.py": {
        "test_dmd_uses_raw_shifted_snapshots_without_weighting",
        "test_default_args_match_original",
    },
    "test_welch_analytical.py": {"test_param_surface_matches_blockwise_definition"},
    "test_spod_n_modes_save.py": {"test_kept_modes_are_the_leading_ones"},
}


def pytest_collection_modifyitems(config, items):
    problems: list[str] = []
    for item in items:
        module_name = Path(item.fspath).name
        # Parametrized items carry bracketed ids; originalname is the bare
        # function name the registry keys on.
        bare_name = getattr(item, "originalname", None) or item.name
        marked = item.get_closest_marker("characterization") is not None
        expected = bare_name in CHARACTERISATION_REGISTRY.get(module_name, set())
        if expected and not marked:
            problems.append(f"{item.nodeid}: replays the library formula; add @pytest.mark.characterization")
        if marked and not expected:
            problems.append(
                f"{item.nodeid}: marked characterization but absent from conftest CHARACTERISATION_REGISTRY"
            )
    if problems:
        raise pytest.UsageError("characterization label registry out of sync:\n  " + "\n  ".join(problems))
