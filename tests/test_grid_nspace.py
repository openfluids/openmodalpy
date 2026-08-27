"""Grid product and snapshot width must agree at the analyzer boundary."""

import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer, PODAnalyzer, SPODAnalyzer, STPODAnalyzer
from openmodalpy.core.base import _reported_grid, calculate_polar_weights


def _analyzer(loader, tmp_path):
    return PODAnalyzer(
        file_path="custom",
        n_modes_save=2,
        data_loader=loader,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )


def test_custom_loader_grid_mismatch_is_rejected(tmp_path):
    """A data_loader is any (str) -> dict; a wrong grid must not reach the metric."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 4), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 4),
            "Nx": 4,
            "Ny": 4,
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    with pytest.raises(ValueError, match=r"Nx\*Ny\*Nz") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "16" in msg
    assert "4" in msg
    assert "Nx=4" in msg
    assert "Ny=4" in msg
    assert "Nz=1" in msg
    assert "q.shape[1]" in msg


def test_custom_loader_consistent_grid_is_accepted(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 3),
            "Nx": 4,
            "Ny": 3,
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 12
    assert int(np.asarray(analyzer.data["q"]).shape[1]) == 12


def test_dataset_without_grid_metadata_is_not_rejected(tmp_path):
    """No Nx/Ny/Nz means no grid to check — do not invent 1*1*1 and reject."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((8, 6), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 6),
            "y": np.array([0.0]),
            "Ns": 8,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) is None
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 6


def test_reported_grid_treats_a_missing_axis_as_extent_one():
    assert _reported_grid({"Nx": 4, "Ny": 3}) == (4, 3, 1)
    assert _reported_grid({"Ny": 5}) == (1, 5, 1)
    assert _reported_grid({}) is None


def test_reported_grid_all_ones_is_not_a_claim():
    """Extents that are all 1 say nothing about width."""
    assert _reported_grid({"Nz": 1}) is None
    assert _reported_grid({"Nx": 1}) is None
    assert _reported_grid({"Nx": 1, "Ny": 1, "Nz": 1}) is None
    assert _reported_grid({"Nx": 12}) == (12, 1, 1)


def test_lone_degenerate_grid_key_is_not_a_claim(tmp_path):
    """A leftover Nz: 1 is not a 1*1*1 claim against a wider matrix."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 4), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 1),
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) is None
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 4


def test_lone_nx_matching_claim_is_accepted(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 12),
            "y": np.array([0.0]),
            "Nx": 12,
            "Ns": 6,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) == (12, 1, 1)
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 12


def test_lone_nx_mismatching_claim_is_rejected(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 12),
            "y": np.array([0.0]),
            "Nx": 7,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    with pytest.raises(ValueError, match=r"Nx\*Ny\*Nz") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "7" in msg
    assert "12" in msg


def _scattered_field(n_space: int = 12, ns: int = 16) -> dict:
    t = np.linspace(0.0, 2.0 * np.pi, ns, endpoint=False)
    x = np.linspace(0.0, 1.0, n_space)
    y = np.linspace(0.0, 2.0, n_space)
    q = 1.0 + np.outer(np.sin(t), np.sin(2.0 * np.pi * x))
    return {"q": q, "x": x, "y": y, "dt": 1.0, "Ns": ns}


def test_pod_uniform_scattered_metric_is_length_n(tmp_path):
    """Two length-n coordinate vectors over an n-wide matrix must yield W.size == n."""
    n = 12

    def loader(_: str) -> dict:
        return {
            "q": np.ones((8, n), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, n),
            "y": np.linspace(0.0, 2.0, n),
            "Ns": 8,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == n


def test_scattered_pod_stpod_eigenvalues_unchanged(tmp_path):
    """Scattered POD / ST-POD stay on the ones-metric spectrum they already had."""
    data = _scattered_field()
    n = int(data["q"].shape[1])
    ones = np.ones(n)

    def run(cls, method, extra, weights, tag):
        field = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        kwargs = {
            "file_path": "dummy",
            "data_loader": lambda _: field,
            "results_dir": str(tmp_path / tag / "results"),
            "figures_dir": str(tmp_path / tag / "figures"),
            **extra,
        }
        if weights is not None:
            kwargs["spatial_weights"] = weights
        analyzer = cls(**kwargs)
        analyzer.load_and_preprocess()
        # Capture the metric the LOAD built. POD and ST-POD both replace a
        # uniform metric with ones inside the solver, so the final W is ones
        # either way -- comparing only the spectra would stay green even with
        # the scattered branch deleted, and would pin nothing.
        w_after_load = np.asarray(analyzer.W).copy()
        getattr(analyzer, method)()
        return analyzer, w_after_load

    pod_u, pod_u_w = run(PODAnalyzer, "perform_pod", {"n_modes_save": 2}, None, "pod-u")
    pod_w, _ = run(PODAnalyzer, "perform_pod", {"n_modes_save": 2}, ones, "pod-w")
    assert pod_u_w.size == n, f"load built a metric of {pod_u_w.size} for {n} scattered points"
    np.testing.assert_allclose(pod_u.eigenvalues, pod_w.eigenvalues)

    st_extra = {"n_modes_save": 2, "embedding_dim": 2}
    st_u, st_u_w = run(STPODAnalyzer, "perform_stpod", st_extra, None, "st-u")
    st_w, _ = run(STPODAnalyzer, "perform_stpod", st_extra, ones, "st-w")
    assert st_u_w.size == n, f"load built a metric of {st_u_w.size} for {n} scattered points"
    np.testing.assert_allclose(st_u.eigenvalues, st_w.eigenvalues)


def test_spod_bsmd_accept_scattered_points(tmp_path):
    """SPOD and BSMD used to reject a length-n coordinate pair; they must run."""
    data = _scattered_field()
    n = int(data["q"].shape[1])

    spod = SPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in data.items()
        },
        nfft=8,
        overlap=0.5,
        results_dir=str(tmp_path / "spod" / "results"),
        figures_dir=str(tmp_path / "spod" / "figures"),
    )
    spod.load_and_preprocess()
    assert int(np.asarray(spod.W).size) == n
    spod.compute_fft_blocks()
    spod.perform_spod()
    assert spod.eigenvalues.size > 0

    bsmd = BSMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in data.items()
        },
        nfft=8,
        overlap=0.5,
        static_triads=[(0, 0, 0)],
        use_parallel=False,
        results_dir=str(tmp_path / "bsmd" / "results"),
        figures_dir=str(tmp_path / "bsmd" / "figures"),
    )
    bsmd.load_and_preprocess()
    assert int(np.asarray(bsmd.W).size) == n
    bsmd.compute_fft_blocks()
    bsmd.perform_bsmd()
    assert bsmd.eigenvalues.size > 0


def test_z_missing_while_nz_gt_1_raises(tmp_path):
    """Grid matches q, but z is absent while Nz > 1: the metric is short."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 24), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 3),
            "Nx": 4,
            "Ny": 3,
            "Nz": 2,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    with pytest.raises(ValueError, match=r"q\.shape\[1\]") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "length 12" in msg, msg
    assert "q.shape[1]=24" in msg, msg


def test_three_d_polar_builds_sector_metric(tmp_path):
    """A 3-D polar field reads z as azimuth and builds a full-length sector metric.

    z is a full revolution sampled at two azimuths (half-open); the metric
    then covers all Nx*Ny*Nz columns, and summing it over the two azimuths
    reproduces the 2-D (x, r) weight.
    """
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.2, 1.0, 3)
    z = np.linspace(0.0, 2.0 * np.pi, 2, endpoint=False)

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 24), dtype=np.float32),
            "x": x,
            "y": y,
            "z": z,
            "Nx": 4,
            "Ny": 3,
            "Nz": 2,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = PODAnalyzer(
        file_path="custom",
        n_modes_save=2,
        data_loader=loader,
        spatial_weight_type="polar",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    assert len(np.asarray(analyzer.W).ravel()) == 24

    w2d = calculate_polar_weights(x, y, use_parallel=False).reshape(3, 4)
    w3d = np.asarray(analyzer.W).reshape(2, 3, 4)
    summed = w3d.sum(axis=0)  # (Ny, Nx)
    np.testing.assert_allclose(summed, w2d, rtol=1e-15, atol=0.0)


def test_square_cartesian_grid_is_not_read_as_scattered(tmp_path):
    """The shape closest to tripping the scattered branch must stay a tensor product.

    A square plane has len(x) == len(y) == n while q is n*n wide. Reading it as a
    point cloud would hand the run an n-long metric for n*n columns and silently
    change what a grid case computes. n == n*n only when n == 1, so the branch
    cannot misfire -- this pins that through the analyzer, not just the helper.
    """
    n = 5

    def loader(_: str) -> dict:
        return {
            "q": np.ones((8, n * n), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, n),
            "y": np.linspace(0.0, 2.0, n),
            "Ns": 8,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == n * n
