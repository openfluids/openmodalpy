import inspect

import numpy as np
import pytest

from openmodalpy.core.io import DataLoader, DNamiDataLoader, MATDataLoader


def test_parallel_loading_identical(tmp_path, monkeypatch):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    dt = 1.0
    for i in range(3):
        arr = np.full((2, 2, 2), i + 1.0)
        times = np.array([0.0, 1.0]) + 2 * i
        np.savez(tmp_path / f"file_{i}.npz", x=x, y=y, dt=dt, times=times, u=arr)

    loader = DNamiDataLoader()

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    data_seq = loader.load(str(tmp_path / "file_0.npz"))

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    data_par = loader.load(str(tmp_path / "file_0.npz"))

    assert np.array_equal(data_seq["q"], data_par["q"])
    assert np.array_equal(data_seq["x"], data_par["x"])
    assert np.array_equal(data_seq["y"], data_par["y"])
    assert data_seq["dt"] == data_par["dt"]
    assert data_seq["metadata"]["loaded_files"] == data_par["metadata"]["loaded_files"]


def test_dt_from_times(tmp_path):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    arr = np.ones((2, 2, 2))
    np.savez(tmp_path / "file.npz", x=x, y=y, dt=0.01, times=np.array([0.0, 1.0]), u=arr)

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path / "file.npz"), load_single=True)

    assert data["dt"] == 1.0


def test_available_fields_ignores_non_flow_arrays(tmp_path):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    arr = np.ones((2, 2, 2))
    np.savez(
        tmp_path / "file.npz",
        x=x,
        y=y,
        times=np.array([0.0, 1.0]),
        dt=np.array([0.1, 0.1]),
        u=arr,
        mask=np.zeros((2, 2)),
        C=np.array([1.0]),
    )

    loader = DNamiDataLoader()

    assert loader.get_available_fields(str(tmp_path / "file.npz"), load_single=True) == ["u"]


def test_consolidated_loader_applies_time_stride_schema(tmp_path):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    for i in range(2):
        arr = np.full((4, 2, 2), i + 1.0)
        times = np.arange(4, dtype=float) + 4 * i
        np.savez(tmp_path / f"file_{i}.npz", x=x, y=y, times=times, u=arr)

    schema = {
        "layout": "consolidated_npz",
        "files": {"pattern": "*.npz"},
        "reduction": {
            "time_start": 1,
            "time_stop": 8,
            "time_stride": 2,
        },
    }

    data = DNamiDataLoader().load(str(tmp_path), schema=schema)

    assert data["Ns"] == 4
    assert np.array_equal(data["t"], np.array([1.0, 3.0, 5.0, 7.0]))


def test_consolidated_loader_applies_spatial_stride_schema(tmp_path):
    x = np.arange(6, dtype=float)
    y = np.arange(4, dtype=float)
    arr = np.arange(3 * 6 * 4, dtype=float).reshape(3, 6, 4)
    np.savez(tmp_path / "file.npz", x=x, y=y, times=np.array([0.0, 1.0, 2.0]), u=arr)

    schema = {
        "layout": "consolidated_npz",
        "reduction": {
            "x_stride": 2,
            "y_stride": 2,
        },
    }

    data = DNamiDataLoader().load(str(tmp_path / "file.npz"), load_single=True, schema=schema)

    assert data["Nx"] == 3
    assert data["Ny"] == 2
    assert np.array_equal(data["x"], np.array([0.0, 2.0, 4.0]))
    assert np.array_equal(data["y"], np.array([0.0, 2.0]))
    expected = arr[:, ::2, ::2].reshape(3, -1)
    np.testing.assert_array_equal(data["q"], expected)
    assert data["metadata"]["reduction"]["x_stride"] == 2
    assert data["metadata"]["reduction"]["y_stride"] == 2


def test_loader_options_are_keyword_only():
    """Loader options must stay keyword-only so positional meaning cannot drift."""
    options = {"preview_ns", "field", "load_single", "schema"}
    for cls in (DataLoader, MATDataLoader, DNamiDataLoader):
        sig = inspect.signature(cls.load)
        for name in options:
            assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        positional = [
            n
            for n, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and n != "self"
        ]
        assert positional == ["file_path"]

    # End to end on BOTH concrete loaders: signature inspection alone would miss a
    # class that kept a positional option through some other mechanism.
    with pytest.raises(TypeError, match="positional"):
        MATDataLoader().load("nonexistent.mat", 5)
    with pytest.raises(TypeError, match="positional"):
        DNamiDataLoader().load("nonexistent.npz", "u")
