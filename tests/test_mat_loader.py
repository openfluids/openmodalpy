import h5py
import numpy as np

from openmodalpy import PODAnalyzer
from openmodalpy.core.io import MATDataLoader


def test_mat_loader_handles_flattened_space_by_time_matrix(tmp_path):
    file_path = tmp_path / "flattened.mat"
    q = np.arange(12, dtype=float).reshape(6, 2)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("u", data=q)
        f.create_dataset("x", data=np.array([0.0, 0.5, 1.0]))
        f.create_dataset("y", data=np.array([0.0, 1.0]))
        f.create_dataset("dt", data=np.array([[0.25]]))

    data = MATDataLoader().load(str(file_path))

    assert data["q"].shape == (2, 6)
    assert np.array_equal(data["q"], q.T)
    assert np.array_equal(data["x"], np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(data["y"], np.array([0.0, 1.0]))
    assert data["Nx"] == 3
    assert data["Ny"] == 2
    assert data["Nz"] == 1
    assert data["Ns"] == 2
    assert np.isclose(data["dt"], 0.25)


def _write_mat(path, coords, q):
    with h5py.File(path, "w") as handle:
        for name, vec in coords.items():
            handle.create_dataset(name, data=vec)
        handle.create_dataset("p", data=q)
        handle.create_dataset("dt", data=np.array([[0.1]]))
    return path


def test_mat_loader_y_only_grid_matches_snapshot_width(tmp_path):
    """An absent x contributes Nx=1, not the whole snapshot width."""
    path = _write_mat(tmp_path / "y_only.mat", {"y": np.linspace(0.0, 1.0, 4)}, np.arange(20.0).reshape(5, 4))
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 1
    assert data["Ny"] == 4
    assert product == width == 4


def test_mat_loader_z_only_absent_axes_are_extent_one(tmp_path):
    """An absent x must not inherit the snapshot width when only z is present."""
    path = _write_mat(tmp_path / "z_only.mat", {"z": np.linspace(0.0, 1.0, 4)}, np.arange(20.0).reshape(5, 4))
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 1
    assert data["Ny"] == 1
    assert data["Nz"] == 4
    assert product == width == 4


def test_mat_loader_x_only_keeps_nx_as_the_present_axis(tmp_path):
    """A present x is still the spatial width; absent y/z stay at 1."""
    path = _write_mat(tmp_path / "x_only.mat", {"x": np.linspace(0.0, 1.0, 4)}, np.arange(20.0).reshape(5, 4))
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 4
    assert data["Ny"] == 1
    assert int(data.get("Nz") or 1) == 1
    assert product == width == 4


def test_mat_loader_x_and_z_grid_matches_snapshot_width(tmp_path):
    """A present x never used the broken fallback; product is Nx*Nz."""
    path = _write_mat(
        tmp_path / "xz.mat",
        {"x": np.linspace(0.0, 1.0, 2), "z": np.linspace(0.0, 1.0, 3)},
        np.arange(42.0).reshape(7, 6),
    )
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 2
    assert data["Ny"] == 1
    assert data["Nz"] == 3
    assert product == width == 6


def test_mat_loader_y_and_z_grid_matches_snapshot_width(tmp_path):
    """An absent x contributes Nx=1 when y and z are both present."""
    path = _write_mat(
        tmp_path / "yz.mat",
        {"y": np.linspace(0.0, 1.0, 3), "z": np.linspace(0.0, 1.0, 2)},
        np.arange(42.0).reshape(7, 6),
    )
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 1
    assert data["Ny"] == 3
    assert data["Nz"] == 2
    assert product == width == 6


def test_mat_loader_no_coords_2d_reshapes_to_grid_product(tmp_path):
    """The other loader site (no coordinates, 2-D q) reshapes to the product."""
    path = tmp_path / "nocoord.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("p", data=np.arange(20.0).reshape(5, 4))
        handle.create_dataset("dt", data=np.array([[0.1]]))
    data = MATDataLoader().load(str(path))
    width = int(np.asarray(data["q"]).shape[1])
    product = int(data["Nx"]) * int(data["Ny"]) * int(data.get("Nz") or 1)
    assert data["Nx"] == 4
    assert data["Ny"] == 1
    assert int(data.get("Nz") or 1) == 1
    assert product == width == 4


def test_y_only_mat_load_and_preprocess_weight_matches_width(tmp_path):
    """A y-only .mat through the analyzer has W.size equal to snapshot width."""
    path = _write_mat(tmp_path / "y_only.mat", {"y": np.linspace(0.0, 1.0, 4)}, np.arange(20.0).reshape(5, 4))
    analyzer = PODAnalyzer(
        file_path=str(path),
        n_modes_save=2,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == int(np.asarray(analyzer.data["q"]).shape[1])
