"""Tests for the plain-contract NPZ/HDF5 reader (GenericDataLoader)."""

import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.core.io import (
    GenericDataLoader,
    load_data,
)

NS, NY, NX, NZ = 12, 6, 8, 3


def _make_field():
    rng = np.random.default_rng(42)
    return rng.standard_normal((NS, NY, NX))


def _coords_2d():
    x = np.linspace(-1.0, 1.0, NX)
    y = np.linspace(0.0, 2.0, NY)
    return x, y


def _write_plain_npz(path, q, **extra):
    x, y = _coords_2d()
    payload = {"q": q, "x": x, "y": y, "dt": np.float64(0.01)}
    payload.update(extra)
    np.savez(path, **payload)


def _write_plain_h5(path, q):
    import h5py

    x, y = _coords_2d()
    with h5py.File(path, "w") as f:
        f.create_dataset("q", data=q)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("dt", data=np.array(0.01))
        f.create_dataset("Nx", data=np.array(NX))
        f.create_dataset("Ny", data=np.array(NY))
        f.create_dataset("Ns", data=np.array(NS))


def _expected_contract_dict(q):
    """In-memory equivalent of what the files above hold."""
    return {
        "q": q.reshape(q.shape[0], -1),
        "x": _coords_2d()[0],
        "y": _coords_2d()[1],
        "z": None,
        "t": None,
        "dt": 0.01,
        "Nx": NX,
        "Ny": NY,
        "Nz": 1,
        "Ns": NS,
    }


def test_npz_roundtrip_full_contract(tmp_path):
    path = tmp_path / "plain.npz"
    q = _make_field()
    _write_plain_npz(path, q, Nx=NX, Ny=NY, Ns=NS)

    d = load_data(str(path))

    for key in ("q", "x", "y", "z", "t", "dt", "Nx", "Ny", "Nz", "Ns", "metadata"):
        assert key in d, key
    # (Ns, Ny, Nx) flattens C-order to (Ns, Ny*Nx); dtypes survive the trip.
    assert d["q"].shape == (NS, NY * NX)
    assert d["q"].dtype == q.dtype
    assert np.array_equal(d["q"], q.reshape(NS, -1))
    x, y = _coords_2d()
    # Coordinate vectors pass through untouched: values, dtype and identity of layout.
    assert np.array_equal(d["x"], x)
    assert d["x"].dtype == x.dtype
    assert np.array_equal(d["y"], y)
    assert d["z"] is None
    assert d["t"] is None
    assert d["dt"] == 0.01
    assert (d["Nx"], d["Ny"], d["Nz"], d["Ns"]) == (NX, NY, 1, NS)
    assert d["metadata"]["format"] == "generic"


def test_h5_roundtrip_full_contract(tmp_path):
    path = tmp_path / "plain.h5"
    q = _make_field()
    _write_plain_h5(path, q)

    d = load_data(str(path))

    assert d["q"].shape == (NS, NY * NX)
    assert d["q"].dtype == q.dtype
    assert np.array_equal(d["q"], q.reshape(NS, -1))
    x, y = _coords_2d()
    assert np.array_equal(d["x"], x)
    assert np.array_equal(d["y"], y)
    assert (d["Nx"], d["Ny"], d["Nz"], d["Ns"]) == (NX, NY, 1, NS)
    assert d["dt"] == 0.01
    assert d["metadata"]["format"] == "generic"


def test_hdf5_extension_alias(tmp_path):
    path = tmp_path / "plain.hdf5"
    q = _make_field()
    _write_plain_h5(path, q)
    assert GenericDataLoader().supports_format(str(path))
    d = load_data(str(path))
    assert d["Ns"] == NS


def test_q_4d_with_z_dataset(tmp_path):
    rng = np.random.default_rng(7)
    q4 = rng.standard_normal((NS, NY, NX, NZ))
    z = np.linspace(0.0, 1.0, NZ)
    path = tmp_path / "vol.npz"
    x, y = _coords_2d()
    np.savez(path, q=q4, x=x, y=y, z=z, dt=np.float64(0.02))

    d = load_data(str(path))

    assert d["q"].shape == (NS, NY * NX * NZ)
    assert np.array_equal(d["q"], q4.reshape(NS, -1))
    assert np.array_equal(d["z"], z)
    assert (d["Nx"], d["Ny"], d["Nz"]) == (NX, NY, NZ)


def test_counts_derived_when_absent(tmp_path):
    path = tmp_path / "nocounts.npz"
    q = _make_field()
    _write_plain_npz(path, q)

    d = load_data(str(path))

    assert (d["Nx"], d["Ny"], d["Nz"], d["Ns"]) == (NX, NY, 1, NS)


def test_stated_count_mismatch_raises(tmp_path):
    path = tmp_path / "badcount.npz"
    q = _make_field()
    _write_plain_npz(path, q, Nx=NX + 1, Ny=NY, Ns=NS)

    with pytest.raises(ValueError, match="Nx"):
        load_data(str(path))


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "noy.npz"
    q = _make_field()
    x, _ = _coords_2d()
    np.savez(path, q=q, x=x, dt=np.float64(0.01))

    with pytest.raises(KeyError, match="'y'"):
        load_data(str(path))


def test_uniform_t_sets_dt_to_median_step(tmp_path):
    t = np.arange(NS) * 0.05
    q = _make_field()
    path = tmp_path / "timed.npz"
    x, y = _coords_2d()
    np.savez(path, q=q, x=x, y=y, t=t, dt=np.float64(999.0))

    d = load_data(str(path))

    # Verified uniform step wins over whatever the file claimed for dt.
    assert d["dt"] == pytest.approx(0.05)
    assert np.array_equal(d["t"], t)


def test_dnami_style_file_routes_to_dnami_loader(tmp_path):
    # dNami signature: coordinates + 'times' vector + field candidate 'u'. No 'q'.
    # Consolidated dNami arrays are (Ns, Nx, Ny); the loader validates each
    # coordinate against its own axis.
    u = np.random.default_rng(3).standard_normal((6, NX, NY))
    times = np.arange(6) * 0.1
    path = tmp_path / "dnami_like.npz"
    x, y = _coords_2d()
    np.savez(path, x=x, y=y, times=times, u=u)

    d = load_data(str(path))

    assert d["metadata"]["format"] != "generic"
    assert d["metadata"]["format"] == "dnami"


def test_schema_kwarg_routes_to_dnami_loader(tmp_path):
    q = _make_field()
    path = tmp_path / "forced.npz"
    _write_plain_npz(path, q, Nx=NX, Ny=NY, Ns=NS)

    # An explicit schema forces the dNami loader even on a plain file; the
    # dNami-style failure (no u/v/p field key) proves which loader ran.
    with pytest.raises(KeyError, match="Could not resolve snapshot field"):
        load_data(str(path), schema={"layout": "consolidated_npz"})


def test_nonuniform_t_refused_and_names_spread(tmp_path):
    t = np.concatenate([np.arange(6) * 0.01, 0.06 + np.arange(6) * 0.03])
    q = _make_field()
    path = tmp_path / "jittery.npz"
    x, y = _coords_2d()
    np.savez(path, q=q, x=x, y=y, t=t)

    with pytest.raises(ValueError, match="not uniform.*dt spans"):
        load_data(str(path))


def test_resample_only_on_explicit_flag(tmp_path):
    t = np.concatenate([np.arange(8) * 0.01, 0.08 + np.arange(4) * 0.02])
    q = _make_field()[:12]
    path = tmp_path / "resampled.npz"
    x, y = _coords_2d()
    np.savez(path, q=q, x=x, y=y, t=t)

    with pytest.raises(ValueError, match="not uniform"):
        load_data(str(path), resample_time=False)

    d = load_data(str(path), resample_time=True)

    assert d["metadata"]["resampled_time"] is True
    steps = np.diff(d["t"])
    assert np.allclose(steps, steps[0], rtol=1e-12)
    assert d["dt"] == pytest.approx(float(np.median(steps)))
    assert d["q"].shape[0] == len(d["t"])
    assert d["q"].shape[1] == NY * NX


def test_list_supported_formats_reports_generic_formats():
    from openmodalpy.core.io import DataInterfaceManager

    formats = DataInterfaceManager().list_supported_formats()
    assert ".h5" in formats and ".hdf5" in formats
    assert "dNami" in formats[".npz"]
    assert set(formats[".h5"].split()) & {"Generic", "HDF5"}


@pytest.mark.parametrize("extension", ["npz", "h5"])
def test_pod_equivalence_against_in_memory_data(tmp_path, extension):
    # savez/h5py store exact float64 bits, the loaders hand back freshly read
    # arrays (no views into caller memory), and BaseAnalyzer stores data= by
    # reference without copying. Both analyzers therefore run on bit-identical
    # inputs through identical deterministic code, so exact equality is the
    # honest assertion here.
    q = _make_field()
    mem = dict(_expected_contract_dict(q))

    if extension == "npz":
        path = tmp_path / f"pod.{extension}"
        _write_plain_npz(path, q, Nx=NX, Ny=NY, Ns=NS)
    else:
        path = tmp_path / f"pod.{extension}"
        _write_plain_h5(path, q)

    loaded = load_data(str(path))

    results = {}
    for label, source in (("mem", mem), ("file", loaded)):
        analyzer = PODAnalyzer(
            data=source,
            results_dir=tmp_path / f"results_{label}",
            figures_dir=tmp_path / f"figures_{label}",
            spatial_weight_type="uniform",
            n_modes_save=5,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()
        results[label] = analyzer

    a, b = results["mem"], results["file"]
    assert np.array_equal(a.eigenvalues, b.eigenvalues)
    assert np.array_equal(a.modes, b.modes)
    assert np.array_equal(a.time_coefficients, b.time_coefficients)


def test_preview_ns_limits_snapshots(tmp_path):
    q = _make_field()
    path = tmp_path / "preview.npz"
    _write_plain_npz(path, q, Nx=NX, Ny=NY, Ns=NS)

    d = load_data(str(path), preview_ns=5)

    assert d["Ns"] == 5
    assert d["q"].shape == (5, NY * NX)


def test_generic_loader_rejects_foreign_extension(tmp_path):
    stray = tmp_path / "stray.txt"
    stray.write_text("not data")
    with pytest.raises(ValueError, match="cannot read"):
        GenericDataLoader().load(str(stray))


def test_scattered_points_inferred_from_1d_coords(tmp_path):
    rng = np.random.default_rng(11)
    ns, n = 5, 7
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.1, 2.0, n)
    q = rng.standard_normal((ns, n))
    path = tmp_path / "scattered.npz"
    np.savez(path, q=q, x=x, y=y, dt=np.float64(0.1))

    d = load_data(str(path))

    assert (d["Nx"], d["Ny"], d["Nz"]) == (n, 1, 1)
    assert np.array_equal(d["x"], x)
    assert np.array_equal(d["y"], y)
    assert d["q"].shape == (ns, n)


def test_scattered_points_through_pod_analyzer_weights(tmp_path):
    rng = np.random.default_rng(12)
    ns, n = 5, 7
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.1, 2.0, n)
    q = rng.standard_normal((ns, n))
    path = tmp_path / "scattered_weights.npz"
    np.savez(path, q=q, x=x, y=y, dt=np.float64(0.1))

    pod_uniform = PODAnalyzer(
        file_path=str(path),
        spatial_weight_type="uniform",
        n_modes_save=2,
        results_dir=str(tmp_path / "results_uniform"),
        figures_dir=str(tmp_path / "figures_uniform"),
    )
    pod_uniform.load_and_preprocess()
    w_uniform = np.asarray(pod_uniform.W).ravel()
    assert w_uniform.shape == (n,)
    np.testing.assert_array_equal(w_uniform, np.ones(n))

    pod_polar = PODAnalyzer(
        file_path=str(path),
        spatial_weight_type="polar",
        n_modes_save=2,
        results_dir=str(tmp_path / "results_polar"),
        figures_dir=str(tmp_path / "figures_polar"),
    )
    pod_polar.load_and_preprocess()
    w_polar = np.asarray(pod_polar.W).ravel()
    assert w_polar.shape == (n,)
    np.testing.assert_array_equal(w_polar, np.abs(y))


def test_square_grid_still_loads_as_grid(tmp_path):
    # x and y both have length n, same as a scattered file would, but q holds
    # n*n points per snapshot: the product test must pick the grid reading.
    n = 4
    ns = 3
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(0.0, 2.0, n)
    q = np.random.default_rng(13).standard_normal((ns, n * n))
    path = tmp_path / "square_grid.npz"
    np.savez(path, q=q, x=x, y=y, dt=np.float64(0.1))

    d = load_data(str(path))

    assert (d["Nx"], d["Ny"], d["Nz"]) == (n, n, 1)
    assert d["q"].shape == (ns, n * n)


def test_nothing_fits_names_both_readings(tmp_path):
    n = 6
    ns = 3
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(0.0, 2.0, n)
    q = np.random.default_rng(14).standard_normal((ns, n + 1))
    path = tmp_path / "nothing_fits.npz"
    np.savez(path, q=q, x=x, y=y, dt=np.float64(0.1))

    with pytest.raises(ValueError, match="state Nx/Ny.*1-D x and y of length Nspace"):
        load_data(str(path))
