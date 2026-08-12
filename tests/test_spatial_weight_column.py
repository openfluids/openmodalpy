"""self.W is a column (n_space, 1) at every analyzer lifecycle stage."""

import h5py
import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer, PODAnalyzer, SPODAnalyzer, STPODAnalyzer
from openmodalpy.core.base import _as_spatial_weight_column

NX, NY, NS = 8, 4, 24
NSPACE = NX * NY


def _make_data():
    rng = np.random.default_rng(0)
    return {
        "q": rng.standard_normal((NS, NSPACE)),
        "x": np.linspace(0.5, 2.0, NX),
        "y": np.linspace(0.5, 1.5, NY),
        "dt": 0.1,
        "Nx": NX,
        "Ny": NY,
        "Ns": NS,
    }


def _is_column(w):
    a = np.asarray(w)
    return a.ndim == 2 and a.shape == (NSPACE, 1)


def test_as_spatial_weight_column_flat_column_square_and_wrong_length():
    diag = np.array([0.5, 1.0, 2.0, 0.25])
    n = diag.size
    flat = _as_spatial_weight_column(diag, n)
    column = _as_spatial_weight_column(diag.reshape(-1, 1), n)
    square = _as_spatial_weight_column(np.diag(diag), n)
    assert flat.shape == (n, 1)
    assert column.shape == (n, 1)
    assert square.shape == (n, 1)
    np.testing.assert_allclose(flat.ravel(), diag)
    np.testing.assert_allclose(column.ravel(), diag)
    np.testing.assert_allclose(square.ravel(), diag)
    with pytest.raises(ValueError, match="n_space"):
        _as_spatial_weight_column(diag, n - 1)


_ANALYZERS = [
    ("POD", PODAnalyzer, "perform_pod", {}),
    ("ST-POD", STPODAnalyzer, "perform_stpod", {"embedding_dim": 3}),
    ("SPOD", SPODAnalyzer, "perform_spod", {"nfft": 8, "overlap": 0.5}),
    ("BSMD", BSMDAnalyzer, "perform_bsmd", {"nfft": 8, "overlap": 0.5, "static_triads": [(0, 0, 0)]}),
]
_WEIGHTS = ["uniform", "polar", "prescribed"]


def _build(cls, extra, wtype, tmp_path, tag):
    kwargs = dict(extra)
    if wtype == "prescribed":
        kwargs["spatial_weight_type"] = "prescribed"
        kwargs["spatial_weights"] = np.linspace(0.5, 2.0, NSPACE)
    else:
        kwargs["spatial_weight_type"] = wtype
    return cls(
        file_path="dummy",
        data_loader=lambda _: _make_data(),
        results_dir=str(tmp_path / tag / "results"),
        figures_dir=str(tmp_path / tag / "figures"),
        use_parallel=False,
        **kwargs,
    )


@pytest.mark.parametrize("name, cls, perform_name, extra", _ANALYZERS, ids=[c[0] for c in _ANALYZERS])
@pytest.mark.parametrize("wtype", _WEIGHTS)
def test_w_is_column_after_load_perform_and_roundtrip(name, cls, perform_name, extra, wtype, tmp_path):
    analyzer = _build(cls, extra, wtype, tmp_path, "roundtrip")
    analyzer.load_and_preprocess()
    assert _is_column(analyzer.W), f"{name} {wtype} after load: {np.asarray(analyzer.W).shape}"
    if hasattr(analyzer, "compute_fft_blocks"):
        analyzer.compute_fft_blocks()
    getattr(analyzer, perform_name)()
    assert _is_column(analyzer.W), f"{name} {wtype} after perform: {np.asarray(analyzer.W).shape}"

    fname = f"{name.replace('-', '')}_{wtype}.hdf5"
    analyzer.save_results(fname)

    loaded = _build(cls, extra, wtype, tmp_path, "roundtrip")
    loaded.load_results(fname)
    assert _is_column(loaded.W), f"{name} {wtype} after save/load: {np.asarray(loaded.W).shape}"
    np.testing.assert_allclose(np.asarray(loaded.W), np.asarray(analyzer.W))


@pytest.mark.parametrize("name, cls, perform_name, extra", _ANALYZERS, ids=[c[0] for c in _ANALYZERS])
def test_legacy_flat_w_dataset_loads_as_column(name, cls, perform_name, extra, tmp_path):
    analyzer = _build(cls, extra, "polar", tmp_path, "legacy")
    analyzer.load_and_preprocess()
    if hasattr(analyzer, "compute_fft_blocks"):
        analyzer.compute_fft_blocks()
    getattr(analyzer, perform_name)()
    fname = f"legacy_{name.replace('-', '')}.hdf5"
    analyzer.save_results(fname)

    path = tmp_path / "legacy" / "results" / fname
    with h5py.File(path, "r+") as h5:
        assert "W" in h5
        flat = np.asarray(h5["W"]).ravel()
        del h5["W"]
        h5.create_dataset("W", data=flat)

    loaded = _build(cls, extra, "polar", tmp_path, "legacy")
    loaded.load_results(fname)
    assert _is_column(loaded.W), f"{name} legacy flat W loaded as {np.asarray(loaded.W).shape}"
    np.testing.assert_allclose(loaded.W.ravel(), flat)
