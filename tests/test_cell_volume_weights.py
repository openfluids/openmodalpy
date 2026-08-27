"""Cell-volume weights for (stretched) Cartesian grids.

Covers the ``calculate_cell_volume_weights`` helper, the ``load_and_preprocess``
uniform branch that now derives cell volumes from 1-D monotone coordinates, and
the resolution-invariance that motivates the change: the same smooth field
sampled on a uniform grid and on a tanh-clustered grid must yield the same
weighted energy and the same leading POD eigenvalues, within the composite
trapezoid quadrature error.
"""

import logging

import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.core.base import (
    calculate_cell_volume_weights,
    calculate_polar_weights,
)
from openmodalpy.core.io import load_data

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_1d_hand_computed_widths():
    """x=[0,1,4] -> [0.5, 2.0, 1.5]: half spacing at the boundary points."""
    w = calculate_cell_volume_weights(np.array([0.0, 1.0, 4.0]), np.array([0.0]))
    assert w.shape == (3, 1)
    assert np.allclose(w.ravel(), [0.5, 2.0, 1.5])
    # Widths must telescope to the domain length: 0.5 + 2.0 + 1.5 == 4.0.


def test_2d_outer_product_order_bruteforce():
    """C-order flattening: weight[iy*Nx + ix] == dy[iy] * dx[ix]."""
    x = np.array([0.0, 0.5, 2.0, 2.5])
    y = np.array([0.0, 1.0, 4.0])
    w = calculate_cell_volume_weights(x, y)
    dx = np.concatenate(((x[1] - x[0],), x[2:] - x[:-2], (x[-1] - x[-2],))) / 2.0
    dy = np.concatenate(((y[1] - y[0],), y[2:] - y[:-2], (y[-1] - y[-2],))) / 2.0
    ny, nx = y.size, x.size
    brute = np.empty((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            brute[iy, ix] = dy[iy] * dx[ix]
    assert w.shape == (ny * nx, 1)
    assert np.allclose(w.ravel(), brute.ravel())


def test_3d_outer_product_order_bruteforce():
    """3-D C-order: weight[(iz*Ny + iy)*Nx + ix] == dz*dy*dx."""
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([0.0, 2.0])
    z = np.array([0.0, 0.5, 2.0, 3.0])
    w = calculate_cell_volume_weights(x, y, z)
    dx = np.array([0.5, 1.5, 1.0])
    dy = np.array([1.0, 1.0])
    dz = np.array([0.25, 1.0, 1.25, 0.5])
    nz, ny, nx = z.size, y.size, x.size
    brute = np.empty((nz, ny, nx))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                brute[iz, iy, ix] = dz[iz] * dy[iy] * dx[ix]
    assert w.shape == (nz * ny * nx, 1)
    assert np.allclose(w.ravel(), brute.ravel())


def test_decreasing_axis_refused_with_flip_hint():
    with pytest.raises(ValueError, match="strictly decreasing.*flip"):
        calculate_cell_volume_weights(np.array([4.0, 1.0, 0.0]), np.array([0.0, 1.0]))


def test_non_monotone_axis_refused():
    with pytest.raises(ValueError, match="not strictly increasing"):
        calculate_cell_volume_weights(np.array([0.0, 1.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="not strictly increasing"):
        calculate_cell_volume_weights(np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Quadrature agreement: uniform vs tanh-clustered grid
# ---------------------------------------------------------------------------

BETA = 2.0  # tanh clustering strength


def _stretched_axis(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 0.5 * (1.0 + np.tanh(BETA * (s - 0.5)) / np.tanh(BETA / 2.0))


def _field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (Ny, Nx), C-order flatten
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)


def _trapz_bound_1d(g: np.ndarray, a: np.ndarray) -> float:
    """Composite trapezoid error bound for one axis: (L/12) h_max^2 max|g''|.

    Holds because the integrand is C^2 on the closed interval and the
    composite trapezoid rule error telescopes to that standard bound.
    """
    h_max = float(np.max(np.diff(a)))
    dd = np.max(np.abs(np.gradient(np.gradient(g, a), a)))
    return (a[-1] - a[0]) / 12.0 * h_max**2 * dd


def _energy_bound(x: np.ndarray, y: np.ndarray) -> float:
    """Bound on |sum(w q^2) - integral| for the separable field q^2 = g(x) g(y)."""
    gx = np.sin(np.pi * x) ** 2
    gy = np.sin(np.pi * y) ** 2
    ix = np.trapezoid(gx, x)
    iy = np.trapezoid(gy, y)
    bx = _trapz_bound_1d(gx, x)
    by = _trapz_bound_1d(gy, y)
    return iy * bx + ix * by + bx * by


def test_energy_grid_independent_uniform_vs_stretched():
    n = 40
    x_u = np.linspace(0.0, 1.0, n)
    y_u = np.linspace(0.0, 1.0, n)
    x_s = _stretched_axis(n)
    y_s = _stretched_axis(n)

    w_u = calculate_cell_volume_weights(x_u, y_u)
    w_s = calculate_cell_volume_weights(x_s, y_s)

    e_u = float(np.sum(w_u.ravel() * _field(x_u, y_u).ravel() ** 2))
    e_s = float(np.sum(w_s.ravel() * _field(x_s, y_s).ravel() ** 2))
    e_s_ones = float(np.sum(_field(x_s, y_s).ravel() ** 2))

    bound = max(_energy_bound(x_u, y_u), _energy_bound(x_s, y_s))
    # Recorded for the close trace: uniform reference vs stretched weighted
    # vs stretched ones-weighted energies.
    print(
        f"\n[recorded] uniform energy={e_u:.10f} "
        f"stretched weighted={e_s:.10f} stretched ones={e_s_ones:.10f} "
        f"quadrature bound={bound:.3e}"
    )

    assert abs(e_u - e_s) <= bound
    # The ones-weighted stretched energy misses the integral by MORE than the
    # quadrature bound: the test proves the fix, not the absence of a fix.
    assert abs(e_s_ones - e_s) > bound


def test_pod_eigenvalues_grid_independent():
    """Two-mode analytic field: leading POD eigenvalues agree across grids."""
    ns = 60
    t = np.linspace(0.0, 2.0, ns)
    a1 = np.cos(2.0 * np.pi * t)
    a2 = 0.5 * np.sin(2.0 * np.pi * 1.5 * t)

    def snapshots(x, y):
        xx, yy = np.meshgrid(x, y, indexing="xy")
        phi1 = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        phi2 = np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
        q = np.outer(a1, phi1.ravel()) + np.outer(a2, phi2.ravel())
        return {"q": q, "x": x, "y": y, "dt": 0.01, "Nx": x.size, "Ny": y.size, "Ns": ns}

    n = 40
    grids = {
        "uniform": (np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)),
        "stretched": (_stretched_axis(n), _stretched_axis(n)),
    }

    eigs = {}
    bounds = {}
    for label, (x, y) in grids.items():
        d = snapshots(x, y)
        pod = PODAnalyzer(
            data=d,
            spatial_weight_type="cell_volume",
            n_modes_save=4,
            results_dir=f"_tmp_cw_{label}",
            figures_dir=f"_tmp_cw_{label}",
        )
        pod.load_and_preprocess()
        pod.perform_pod()
        eigs[label] = np.asarray(pod.eigenvalues[:2])
        bounds[label] = _pod_bound(x, y)

    print(
        f"\n[recorded] POD leading eigenvalues uniform={eigs['uniform']} "
        f"stretched={eigs['stretched']} bound={bounds['uniform']:.3e}/{bounds['stretched']:.3e}"
    )
    diff = float(np.max(np.abs(eigs["uniform"] - eigs["stretched"])))
    assert diff <= bounds["uniform"] + bounds["stretched"]


def _pod_bound(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid bound on the phi^2 integrals of the two POD modes."""
    xx, yy = np.meshgrid(x, y, indexing="xy")
    total = 0.0
    for k in (1, 2):
        gx = np.sin(k * np.pi * x) ** 2
        gy = np.sin(np.pi * y) ** 2
        ix = np.trapezoid(gx, x)
        iy = np.trapezoid(gy, y)
        total += iy * _trapz_bound_1d(gx, x) + ix * _trapz_bound_1d(gy, y)
        total += _trapz_bound_1d(gx, x) * _trapz_bound_1d(gy, y)
    return total


# ---------------------------------------------------------------------------
# End-to-end: stretched-grid .npz through the generic reader
# ---------------------------------------------------------------------------


def test_npz_stretched_grid_matches_prescribed_weights(tmp_path):
    """POD on a stretched-grid .npz (generic reader) equals data= + prescribed volumes."""
    ns, n = 40, 30
    t = np.linspace(0.0, 1.0, ns)
    x = _stretched_axis(n)
    y = _stretched_axis(n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    phi = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    q = np.outer(np.cos(2 * np.pi * t), phi.ravel())  # (Ns, Ny*Nx)

    path = tmp_path / "stretched.npz"
    np.savez(path, q=q.reshape(ns, y.size, x.size), x=x, y=y, dt=np.float64(0.01))
    pod_file = PODAnalyzer(
        file_path=str(path),
        spatial_weight_type="cell_volume",
        n_modes_save=4,
        results_dir=tmp_path / "res_file",
        figures_dir=tmp_path / "fig_file",
    )
    pod_file.load_and_preprocess()
    d = load_data(str(path))

    w = calculate_cell_volume_weights(x, y)
    pod_mem = PODAnalyzer(
        data=d,
        spatial_weights=w,
        n_modes_save=4,
        results_dir=tmp_path / "res_mem",
        figures_dir=tmp_path / "fig_mem",
    )
    pod_mem.load_and_preprocess()

    assert np.allclose(pod_file.W, pod_mem.W, rtol=0, atol=0)
    pod_file.perform_pod()
    pod_mem.perform_pod()
    assert np.allclose(pod_file.eigenvalues, pod_mem.eigenvalues, rtol=1e-12, atol=0)


def test_cell_volume_branch_logs_derived_weights(tmp_path, caplog):
    """The INFO log says cell volumes were derived when 1-D coordinates exist."""
    n = 8
    d = {
        "q": np.random.default_rng(0).standard_normal((5, n * n)),
        "x": np.linspace(0.0, 1.0, n),
        "y": np.linspace(0.0, 1.0, n),
        "dt": 0.1,
        "Nx": n,
        "Ny": n,
        "Ns": 5,
    }
    pod = PODAnalyzer(
        data=d,
        spatial_weight_type="cell_volume",
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    with caplog.at_level(logging.INFO, logger="openmodalpy.core.base"):
        pod.load_and_preprocess()
    assert any("cell-volume" in r.getMessage() for r in caplog.records)


def test_cell_volume_branch_refuses_non_monotone_coordinates(tmp_path):
    n = 6
    d = {
        "q": np.random.default_rng(0).standard_normal((4, n * n)),
        "x": np.linspace(1.0, 0.0, n),  # decreasing
        "y": np.linspace(0.0, 1.0, n),
        "dt": 0.1,
        "Nx": n,
        "Ny": n,
        "Ns": 4,
    }
    pod = PODAnalyzer(data=d, spatial_weight_type="cell_volume", results_dir=tmp_path, figures_dir=tmp_path)
    with pytest.raises(ValueError, match="flip"):
        pod.load_and_preprocess()


def test_scattered_coordinates_still_get_ones(tmp_path):
    """Scattered 1-D x/y of length n_space keep the ones fallback."""
    n = 10
    rng = np.random.default_rng(1)
    d = {
        "q": rng.standard_normal((4, n)),
        "x": np.linspace(0.0, 1.0, n),
        "y": np.linspace(0.0, 1.0, n),
        "dt": 0.1,
        "Ns": 4,
    }
    pod = PODAnalyzer(data=d, results_dir=tmp_path, figures_dir=tmp_path)
    pod.load_and_preprocess()
    assert np.array_equal(np.asarray(pod.W), np.ones((n, 1)))


def test_uniform_stretched_grid_returns_ones(tmp_path):
    """Restored v0.5.0 semantics: "uniform" is ones even on a stretched grid."""
    n = 8
    d = {
        "q": np.random.default_rng(2).standard_normal((5, n * n)),
        "x": _stretched_axis(n),
        "y": _stretched_axis(n),
        "dt": 0.1,
        "Nx": n,
        "Ny": n,
        "Ns": 5,
    }
    pod = PODAnalyzer(data=d, results_dir=tmp_path, figures_dir=tmp_path)
    pod.load_and_preprocess()
    assert np.array_equal(np.asarray(pod.W), np.ones((n * n, 1)))


def test_cell_volume_without_grid_coordinates_raises(tmp_path):
    """Opt-in volumes demand Cartesian grid coordinates: a scattered set raises."""
    n = 10
    rng = np.random.default_rng(3)
    d = {
        "q": rng.standard_normal((4, n)),
        "x": np.linspace(0.0, 1.0, n),
        "y": np.linspace(0.0, 1.0, n),
        "dt": 0.1,
        "Ns": 4,
    }
    pod = PODAnalyzer(
        data=d,
        spatial_weight_type="cell_volume",
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="cell_volume"):
        pod.load_and_preprocess()


# ---------------------------------------------------------------------------
# Polar and prescribed weights untouched
# ---------------------------------------------------------------------------


def test_polar_weights_untouched(tmp_path):
    nx, ny = 6, 8
    x = np.linspace(0.0, 1.0, nx)
    r = np.linspace(0.0, 2.0, ny)
    d = {
        "q": np.random.default_rng(2).standard_normal((4, nx * ny)),
        "x": x,
        "y": r,
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": 4,
    }
    pod = PODAnalyzer(
        data=d,
        spatial_weight_type="polar",
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod.load_and_preprocess()
    expected = calculate_polar_weights(x, r, use_parallel=False)
    assert np.allclose(np.asarray(pod.W), np.asarray(expected))


def test_prescribed_weights_untouched(tmp_path):
    n = 9
    d = {
        "q": np.random.default_rng(3).standard_normal((4, n * n)),
        "x": np.linspace(0.0, 1.0, n),
        "y": np.linspace(0.0, 1.0, n),
        "dt": 0.1,
        "Nx": n,
        "Ny": n,
        "Ns": 4,
    }
    w = np.arange(1.0, n * n + 1.0).reshape(-1, 1)
    pod = PODAnalyzer(
        data=d,
        spatial_weights=w,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod.load_and_preprocess()
    assert np.array_equal(np.asarray(pod.W), w)


def test_generator_flattening_matches_contract():
    """Built-in generators emit q in contract order: index = iy*Nx + ix.

    Regression pin for the layout: the double-gyre field is
    u(x, y) = -pi*A*sin(pi*f(x))*cos(pi*y), and the flattened snapshot must
    equal that field at (x[ix], y[iy]) when read as q.reshape(Ny, Nx).
    Nx != Ny so a transposed layout cannot pass by accident.
    """
    from openmodalpy.example_data import generate_double_gyre

    nx, ny, nt = 9, 5, 3
    d = generate_double_gyre(Nx=nx, Ny=ny, Nt=nt)
    q = np.asarray(d["q"])
    assert q.shape == (nt, nx * ny)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])

    epsilon = 0.25
    omega = 2.0 * np.pi / float(d["metadata"]["period"])
    t = np.linspace(0.0, 20.0, nt)  # generator default t_max
    grid = q[0].reshape(ny, nx)
    for iy in (0, 2, ny - 1):
        for ix in (0, 4, nx - 1):
            ti = t[0]
            aa = epsilon * np.sin(omega * ti)
            bb = 1.0 - 2.0 * epsilon * np.sin(omega * ti)
            f = aa * x[ix] ** 2 + bb * x[ix]
            want = -np.pi * 0.25 * np.sin(np.pi * f) * np.cos(np.pi * y[iy])
            assert grid[iy, ix] == pytest.approx(want, rel=1e-12), (iy, ix)

    # And the whole first snapshot, not just samples.
    xx, yy = np.meshgrid(x, y)  # (Ny, Nx), contract layout
    aa = epsilon * np.sin(omega * t[0])
    bb = 1.0 - 2.0 * epsilon * np.sin(omega * t[0])
    f = aa * xx**2 + bb * xx
    want = -np.pi * 0.25 * np.sin(np.pi * f) * np.cos(np.pi * yy)
    assert np.allclose(grid, want, rtol=1e-12, atol=1e-12)
