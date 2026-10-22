"""Timestep validation in BaseAnalyzer.load_and_preprocess."""

import os
import tempfile

import h5py
import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.core.base import BaseAnalyzer
from openmodalpy.core.io import MATDataLoader, _infer_dt_from_times
from openmodalpy.dmd import DMDAnalyzer
from openmodalpy.mpod import MPODAnalyzer

_MISSING = object()


def _make_data(dt):
    Ns, Nx, Ny = 12, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    if dt is not _MISSING:
        data["dt"] = dt
    return data


def _load(dt, file_path="case_snapshot.h5"):
    analyzer = PODAnalyzer(
        file_path=file_path,
        data_loader=lambda _: _make_data(dt),
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    return analyzer


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        _MISSING,
        None,
        np.array([0.1, 0.2]),
    ],
    ids=["zero", "negative", "nan", "inf", "neg_inf", "missing", "none", "array"],
)
def test_invalid_timestep_raises_value_error_naming_source(dt):
    """Every way of not having a usable dt fails the same way, with the same guidance.

    The assertions pin the *content* the caller needs -- which dataset, which
    value, and that they are expected to supply one -- not the exact prose, so
    the message can be reworded without a spurious failure.
    """
    source = "my_dataset.npz"
    with pytest.raises(ValueError, match=r"timestep") as exc_info:
        _load(dt, file_path=source)
    msg = str(exc_info.value)
    assert source in msg, "the message must name the data source"
    assert "dt" in msg, "the message must name the offending parameter"
    assert "provide" in msg.lower(), "the message must tell the caller to supply a dt"


@pytest.mark.parametrize("dt", [0.25, 2.0])
def test_valid_timestep_sets_fs(dt):
    """Two distinct dt, so this check cannot be satisfied by always raising."""
    analyzer = _load(dt)
    assert np.isclose(analyzer.fs, 1.0 / dt)
    # Validation must read dt, never write it back -- a coerced or defaulted
    # value in data["dt"] would make downstream consumers (dmd omega, SPOD
    # Strouhal) disagree with what the loader actually returned.
    assert analyzer.data["dt"] == dt


def test_require_dt_does_not_write_to_data():
    """_require_dt validates and returns without mutating self.data['dt']."""
    analyzer = PODAnalyzer(
        file_path="unit_test.npz",
        data_loader=lambda _: _make_data(0.1),
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.data = _make_data(0.1)
    before = analyzer.data["dt"]
    got = analyzer._require_dt()
    assert got == 0.1
    assert analyzer.data["dt"] is before
    assert analyzer.data["dt"] == 0.1


def test_load_and_preprocess_routes_through_require_dt(monkeypatch):
    """load_and_preprocess must call _require_dt, not keep a second copy of the check.

    Asserting the attribute merely exists would pass against a duplicated
    inline check, which is the arrangement this fix removes. Making the
    accessor raise a sentinel is what actually proves the call routes
    through it.
    """
    sentinel = ValueError("sentinel raised by _require_dt")

    def boom(self):
        raise sentinel

    monkeypatch.setattr(BaseAnalyzer, "_require_dt", boom)
    with pytest.raises(ValueError) as exc_info:
        _load(0.25)
    assert exc_info.value is sentinel


def test_mat_loader_does_not_fabricate_dt():
    """MAT file without dt leaves dt absent/None so the guard can fire."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nodt.mat")
    with h5py.File(path, "w") as f:
        f.create_dataset("p", data=np.random.default_rng(0).random((8, 12)).astype(np.float32))
        f.create_dataset("x", data=np.arange(4, dtype=float))
        f.create_dataset("y", data=np.arange(3, dtype=float))
    d = MATDataLoader().load(path)
    assert d.get("dt") != 1.0
    assert d.get("dt") is None


def test_infer_dt_from_times_returns_none_when_uninferable():
    assert _infer_dt_from_times(np.array([0.0])) is None
    assert _infer_dt_from_times(np.array([2.5, 2.5, 2.5])) is None
    assert np.isclose(_infer_dt_from_times(np.arange(5) * 0.1), 0.1)


def test_dmd_reload_without_dt_raises():
    """Reload-shaped state (data present, no dt) must refuse omega, not use 1.0."""
    Ns, Nx, Ny = 24, 4, 3
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    a = DMDAnalyzer(
        file_path="reloaded_results.hdf5",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        rank=2,
    )
    a.data = dict(data)
    a.W = np.ones(Nx * Ny)
    with pytest.raises(ValueError, match=r"timestep"):
        a.perform_dmd()


def test_dmd_omega_scales_inversely_with_dt():
    """omega = log(lambda)/dt, so halving dt must double every omega.

    The positive control for routing every dt read through validation:
    without it, an accessor that always raised would satisfy the negative
    tests.
    It also pins the direction of the division, which a wrong-way fix would
    otherwise pass silently.
    """
    Ns, Nx, Ny = 24, 4, 3
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))

    def make(dt):
        return {
            "q": q,
            "x": np.arange(Nx, dtype=float),
            "y": np.arange(Ny, dtype=float),
            "dt": dt,
            "Nx": Nx,
            "Ny": Ny,
            "Ns": Ns,
        }

    omegas = {}
    for dt in (0.25, 0.5):
        a = DMDAnalyzer(
            file_path="case.npz",
            data_loader=lambda _, dt=dt: make(dt),
            spatial_weight_type="uniform",
            n_modes_save=2,
            rank=2,
        )
        a.load_and_preprocess()
        a.perform_dmd()
        omegas[dt] = np.asarray(a.omega)

    assert np.allclose(omegas[0.25] / omegas[0.5], 2.0)


def test_mpod_reload_without_dt_raises():
    """Reload-shaped mPOD must refuse band-edge resolution without a real dt."""
    Ns, Nx, Ny = 64, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    a = MPODAnalyzer(
        file_path="reloaded.hdf5",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.1, 0.4],
        band_scale="normalized_nyquist",
    )
    a.data = dict(data)
    a.W = np.ones(Nx * Ny)
    with pytest.raises(ValueError, match=r"timestep"):
        a.perform_mpod()


def _analyzer_for_time_axis(dt=_MISSING, t=None):
    """Minimal PODAnalyzer with data already attached (no load_and_preprocess)."""
    data = _make_data(dt)
    if t is not None:
        data["t"] = t
    a = PODAnalyzer(
        file_path="time_axis.npz",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    a.data = data
    return a


def test_time_axis_valid_dt_scales_and_labels_seconds():
    vals, label = _analyzer_for_time_axis(0.25)._time_axis(4)
    assert np.allclose(vals, [0.0, 0.25, 0.5, 0.75])
    assert "s" in label.lower()
    assert "time" in label.lower()


@pytest.mark.parametrize(
    "dt",
    [_MISSING, None, 0.0, float("nan"), float("inf"), -1.0],
    ids=["absent", "none", "zero", "nan", "inf", "negative"],
)
def test_time_axis_unusable_dt_falls_back_to_sample_index(dt):
    """Plot abscissa must never raise and must not claim time units."""
    vals, label = _analyzer_for_time_axis(dt)._time_axis(4)
    assert np.allclose(vals, [0, 1, 2, 3])
    assert "time" not in label.lower()
    assert "[s]" not in label.lower()


def test_time_axis_explicit_t_wins():
    Ns = 12
    t = np.arange(Ns) * 0.1
    vals, label = _analyzer_for_time_axis(0.25, t=t)._time_axis(4)
    assert np.allclose(vals, [0.0, 0.1, 0.2, 0.3])
    assert "s" in label.lower()


def test_dmd_plot_eigenspectra_refuses_missing_dt():
    """Figure frequencies are physical claims — no fabricated unit timestep."""
    Ns, Nx, Ny = 24, 4, 3
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "dt": 0.25,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    a = DMDAnalyzer(
        file_path="case.npz", data_loader=lambda _: dict(data), spatial_weight_type="uniform", n_modes_save=2, rank=2
    )
    a.load_and_preprocess()
    a.perform_dmd()
    del a.data["dt"]
    with pytest.raises(ValueError, match=r"timestep"):
        a.plot_eigenspectra()


def _dmd_for_mode_freq():
    """DMD analyzer with eigenvalues ready; caller mutates dt for annotation tests."""
    Ns, Nx, Ny = 24, 4, 3
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "dt": 0.25,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    a = DMDAnalyzer(
        file_path="case.npz", data_loader=lambda _: dict(data), spatial_weight_type="uniform", n_modes_save=2, rank=2
    )
    a.load_and_preprocess()
    a.perform_dmd()
    return a


def test_mode_freq_usable_dt():
    """_mode_freq returns angle/(2π·dt) when dt is positive finite."""
    a = _dmd_for_mode_freq()
    eig = a.eigenvalues[:2]
    got = a._mode_freq(eig)
    assert got is not None
    assert np.allclose(got, np.angle(eig) / (2 * np.pi * 0.25))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.pop("dt", None), id="absent"),
        pytest.param(lambda d: d.__setitem__("dt", None), id="none"),
        pytest.param(lambda d: d.__setitem__("dt", 0.0), id="zero"),
        pytest.param(lambda d: d.__setitem__("dt", float("nan")), id="nan"),
        pytest.param(lambda d: d.__setitem__("dt", float("inf")), id="inf"),
        pytest.param(lambda d: d.__setitem__("dt", -1.0), id="negative"),
    ],
)
def test_mode_freq_unusable_dt_returns_none(mutate):
    """_mode_freq never raises and returns None when dt cannot define Hz."""
    a = _dmd_for_mode_freq()
    eig = a.eigenvalues[:2]
    mutate(a.data)
    assert a._mode_freq(eig) is None


def test_dmd_plot_modes_detailed_renders_without_dt(tmp_path):
    """Mode-shape figure must draw even when frequency annotation is dropped."""
    a = _dmd_for_mode_freq()
    a.figures_dir = str(tmp_path)
    del a.data["dt"]
    a.plot_modes_detailed(plot_n_modes=1)  # must not raise


def test_pod_time_series_xlabel_not_clobbered_by_strouhal(tmp_path, monkeypatch):
    """Every mode's time-series panel keeps the _time_axis label (not Strouhal)."""
    import matplotlib.pyplot as plt

    Ns, Nx, Ny = 32, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
        "dt": 0.25,
    }
    a = PODAnalyzer(
        file_path="case.npz",
        data_loader=lambda _: dict(data),
        spatial_weight_type="uniform",
        n_modes_save=3,
        figures_dir=str(tmp_path),
        results_dir=str(tmp_path),
    )
    a.load_and_preprocess()
    a.perform_pod()

    calls = []
    real_xlabel = plt.xlabel

    def spy(label, *args, **kwargs):
        calls.append(label)
        return real_xlabel(label, *args, **kwargs)

    monkeypatch.setattr(plt, "xlabel", spy)
    a.plot_time_coefficients(n_coeffs_to_plot=3)

    n_time = sum(1 for c in calls if "Time" in c or "Sample" in c)
    n_st = sum(1 for c in calls if "Strouhal" in c)
    assert n_time == 3, f"expected 3 time-series labels, got {n_time} in {calls}"
    assert n_st == 3, f"expected 3 Strouhal labels, got {n_st} in {calls}"
