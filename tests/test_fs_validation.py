"""Sampling-rate validation via BaseAnalyzer._require_fs."""

import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.core.base import BaseAnalyzer
from openmodalpy.spod import SPODAnalyzer


def _make_data(dt=0.25):
    Ns, Nx, Ny = 16, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    return {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
        "dt": dt,
    }


def _analyzer_with_fs(fs, file_path="my_case.npz"):
    data = _make_data()
    a = PODAnalyzer(
        file_path=file_path,
        data_loader=lambda _: dict(data),
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    a.data = dict(data)
    a.fs = fs
    return a


@pytest.mark.parametrize(
    "fs",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
    ids=["zero", "negative", "nan", "inf", "neg_inf"],
)
def test_require_fs_invalid_raises_naming_source(fs):
    """Invalid fs fails with ValueError naming the data source and parameter."""
    source = "my_dataset.npz"
    with pytest.raises(ValueError, match=r"sampling rate|fs") as exc_info:
        _analyzer_with_fs(fs, file_path=source)._require_fs()
    msg = str(exc_info.value)
    assert source in msg, "the message must name the data source"
    assert "fs" in msg, "the message must name the offending parameter"
    assert "provide" in msg.lower(), "the message must tell the caller to supply an fs"


@pytest.mark.parametrize("fs", [4.0, 12.5])
def test_require_fs_valid_returns_unchanged(fs):
    """Two distinct valid rates so always-raise cannot pass."""
    got = _analyzer_with_fs(fs)._require_fs()
    assert np.isclose(got, fs)


def test_require_fs_does_not_write_to_self():
    """_require_fs validates and returns without mutating self.fs."""
    a = _analyzer_with_fs(4.0)
    before = a.fs
    got = a._require_fs()
    assert got == 4.0
    assert a.fs is before
    assert a.fs == 4.0


def test_frequency_axis_unchanged_for_valid_fs():
    """rfftfreq via _require_fs matches the validated 1/dt sampling rate."""
    Ns, Nx, Ny = 64, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
        "dt": 0.02,
    }
    a = SPODAnalyzer(
        file_path="case.npz",
        data_loader=lambda _: dict(data),
        spatial_weight_type="uniform",
        nfft=16,
        overlap=0.5,
    )
    a.load_and_preprocess()
    expected = np.fft.rfftfreq(16, d=1.0 / (1.0 / 0.02))
    got = np.fft.rfftfreq(a.nfft, d=1.0 / a._require_fs())
    assert np.allclose(got, expected)


def test_pod_periodogram_names_source_when_fs_zero(tmp_path, monkeypatch):
    """plot_time_coefficients must raise our ValueError, not leak fftkit's."""
    data = _make_data(dt=0.25)
    # larger series so perform_pod has enough snapshots
    Ns, Nx, Ny = 32, 4, 2
    data["q"] = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data["Ns"] = Ns
    a = PODAnalyzer(
        file_path="my_case.npz",
        data_loader=lambda _: dict(data),
        spatial_weight_type="uniform",
        n_modes_save=2,
        figures_dir=str(tmp_path),
        results_dir=str(tmp_path),
    )
    a.load_and_preprocess()
    a.perform_pod()
    a.fs = 0.0
    with pytest.raises(ValueError) as exc_info:
        a.plot_time_coefficients(n_coeffs_to_plot=1)
    msg = str(exc_info.value)
    assert "my_case.npz" in msg
    assert "fs" in msg


def test_pod_plot_routes_through_require_fs(tmp_path, monkeypatch):
    """plot_time_coefficients must call _require_fs, not keep a second check."""
    data = _make_data(dt=0.25)
    Ns, Nx, Ny = 32, 4, 2
    data["q"] = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data["Ns"] = Ns
    a = PODAnalyzer(
        file_path="case.npz",
        data_loader=lambda _: dict(data),
        spatial_weight_type="uniform",
        n_modes_save=2,
        figures_dir=str(tmp_path),
        results_dir=str(tmp_path),
    )
    a.load_and_preprocess()
    a.perform_pod()

    sentinel = ValueError("sentinel raised by _require_fs")

    def boom(self):
        raise sentinel

    monkeypatch.setattr(BaseAnalyzer, "_require_fs", boom)
    with pytest.raises(ValueError) as exc_info:
        a.plot_time_coefficients(n_coeffs_to_plot=1)
    assert exc_info.value is sentinel
