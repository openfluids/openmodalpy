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


@pytest.mark.parametrize("name, cls, perform_name, extra", _ANALYZERS, ids=[c[0] for c in _ANALYZERS])
def test_load_results_rejects_mismatched_w_and_keeps_matching_w(name, cls, perform_name, extra, tmp_path):
    analyzer = _build(cls, extra, "prescribed", tmp_path, "mismatch")
    analyzer.load_and_preprocess()
    if hasattr(analyzer, "compute_fft_blocks"):
        analyzer.compute_fft_blocks()
    getattr(analyzer, perform_name)()
    fname = f"mismatch_{name.replace('-', '')}.hdf5"
    analyzer.save_results(fname)

    matching = _build(cls, extra, "prescribed", tmp_path, "mismatch")
    matching.load_results(fname)
    assert _is_column(matching.W), f"{name} matching W loaded as {np.asarray(matching.W).shape}"
    np.testing.assert_allclose(np.asarray(matching.W), np.asarray(analyzer.W))

    path = tmp_path / "mismatch" / "results" / fname
    wrong_len = 3
    with h5py.File(path, "r+") as h5:
        del h5["W"]
        h5.create_dataset("W", data=np.linspace(1.0, 2.0, wrong_len))

    loaded = _build(cls, extra, "prescribed", tmp_path, "mismatch")
    with pytest.raises(ValueError, match=rf"length {wrong_len}.*n_space={NSPACE}"):
        loaded.load_results(fname)


_MODE_KEY = {"POD": "modes", "ST-POD": "modes", "SPOD": "modes", "BSMD": "modes1"}


@pytest.mark.parametrize("name, cls, perform_name, extra", _ANALYZERS, ids=[c[0] for c in _ANALYZERS])
def test_load_results_skips_the_length_check_without_a_usable_mode_array(name, cls, perform_name, extra, tmp_path):
    """No usable rank means no size on the file, so W loads unchecked as it did before.

    Reachable without touching a file by hand: saving before the decomposition
    runs writes an empty, rank-1 mode array.
    """
    analyzer = _build(cls, extra, "prescribed", tmp_path, "nosize")
    analyzer.load_and_preprocess()
    if hasattr(analyzer, "compute_fft_blocks"):
        analyzer.compute_fft_blocks()
    getattr(analyzer, perform_name)()
    fname = f"nosize_{name.replace('-', '')}.hdf5"
    analyzer.save_results(fname)

    path = tmp_path / "nosize" / "results" / fname
    wrong = np.linspace(1.0, 2.0, 3)
    with h5py.File(path, "r+") as h5:
        del h5[_MODE_KEY[name]]
        h5.create_dataset(_MODE_KEY[name], data=np.array([]))
        del h5["W"]
        h5.create_dataset("W", data=wrong)

    loaded = _build(cls, extra, "prescribed", tmp_path, "nosize")
    loaded.load_results(fname)
    np.testing.assert_allclose(np.asarray(loaded.W).ravel(), wrong)


def test_stpod_sizes_w_from_the_files_embedding_dim_not_the_constructors(tmp_path):
    """The delay depth must come from the file, not from the analyzer doing the loading.

    ST-POD modes are (d * n_space, n_modes), so the spatial size needs d. The
    loader assigns self.embedding_dim further down the same method, which means
    reading it at the W line gets whatever the constructor was given. Saved with
    d = 3 and loaded by an analyzer built with the class default d = 10, the
    stale read would size the field at 96 // 10 = 9 and reject a good file.
    """
    saved_d, other_d = 3, 10
    analyzer = _build(STPODAnalyzer, {"embedding_dim": saved_d}, "prescribed", tmp_path, "embed")
    analyzer.load_and_preprocess()
    analyzer.perform_stpod()
    fname = "stpod_embed.hdf5"
    analyzer.save_results(fname)
    assert analyzer.modes.shape[0] == saved_d * NSPACE

    loaded = _build(STPODAnalyzer, {"embedding_dim": other_d}, "prescribed", tmp_path, "embed")
    loaded.load_results(fname)
    assert _is_column(loaded.W)
    np.testing.assert_allclose(np.asarray(loaded.W), np.asarray(analyzer.W))


def test_bsmd_load_results_skips_w_length_check_without_modes1(tmp_path):
    extra = {"nfft": 8, "overlap": 0.5, "static_triads": [(0, 0, 0)]}
    analyzer = _build(BSMDAnalyzer, extra, "prescribed", tmp_path, "nosize")
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_bsmd()
    fname = "bsmd_nosize.hdf5"
    analyzer.save_results(fname)

    path = tmp_path / "nosize" / "results" / fname
    wrong = np.linspace(1.0, 2.0, 3)
    with h5py.File(path, "r+") as h5:
        del h5["modes1"]
        del h5["W"]
        h5.create_dataset("W", data=wrong)

    loaded = _build(BSMDAnalyzer, extra, "prescribed", tmp_path, "nosize")
    loaded.load_results(fname)
    np.testing.assert_allclose(np.asarray(loaded.W).ravel(), wrong)
