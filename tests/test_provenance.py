"""Every result file records what produced it (versions, backend, config hash)."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    PODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
    read_results,
)
from openmodalpy.core.results import write_results

REQUIRED = frozenset(
    {
        "prov_openmodalpy_version",
        "prov_python_version",
        "prov_numpy_version",
        "prov_scipy_version",
        "prov_h5py_version",
        "prov_fftkit_version",
        "prov_fft_backend",
        "prov_blas_threads",
        "prov_blas",
        "prov_platform",
        "prov_machine",
        "prov_hdf5_version",
        "prov_config_sha256",
        "prov_created_utc",
        "prov_git_sha",
        "prov_seed",
    }
)


def _toy_field(ns: int = 40, nx: int = 8, ny: int = 6) -> dict:
    """Synthetic field with enough rank for DMD rank=3 (matches the gate harness)."""
    t = np.arange(ns) * 0.1
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 0.5, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    base = np.sin(2.0 * np.pi * xx) * np.cos(np.pi * yy)
    second = np.cos(4.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    q = np.empty((ns, nx * ny), dtype=float)
    for i, ti in enumerate(t):
        field = 1.0 + np.sin(2.0 * np.pi * 0.5 * ti) * base + 0.3 * np.cos(2.0 * np.pi * ti) * second
        q[i] = field.reshape(-1)
    return {
        "q": q,
        "x": x,
        "y": y,
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }


def _assert_provenance_block(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
    missing = sorted(REQUIRED - set(attrs))
    assert not missing, f"{path}: missing provenance fields {missing}"
    for key in REQUIRED:
        if key == "prov_blas_threads":
            int(attrs[key])
            continue
        value = attrs[key]
        text = value.decode() if isinstance(value, bytes) else value
        assert isinstance(text, str) and text, f"{path}: {key} empty or not a string ({value!r})"
    res = read_results(path)
    assert set(res.provenance) == {k[len("prov_") :] for k in REQUIRED}


def test_provenance_all_analyzers(tmp_path: Path) -> None:
    """AC1: the provenance block is present for every analyzer, not one of them.

    Analyzers covered: PODAnalyzer, SPODAnalyzer, DMDAnalyzer, BSMDAnalyzer,
    STPODAnalyzer.
    """
    field = _toy_field()
    common = dict(
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
    )

    pod = PODAnalyzer(file_path="prov_pod", n_modes_save=3, **common)
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results("pod.hdf5")
    _assert_provenance_block(tmp_path / "pod.hdf5")

    spod = SPODAnalyzer(
        file_path="prov_spod",
        nfft=8,
        overlap=0.5,
        **common,
    )
    spod.load_and_preprocess()
    spod.compute_fft_blocks()
    spod.perform_spod()
    spod.save_results("spod.hdf5")
    _assert_provenance_block(tmp_path / "spod.hdf5")

    dmd = DMDAnalyzer(file_path="prov_dmd", n_modes_save=3, rank=3, **common)
    dmd.load_and_preprocess()
    dmd.perform_dmd()
    dmd.save_results("dmd.hdf5")
    _assert_provenance_block(tmp_path / "dmd.hdf5")

    bsmd = BSMDAnalyzer(
        file_path="prov_bsmd",
        nfft=16,
        overlap=0.5,
        use_parallel=False,
        **common,
    )
    bsmd.load_and_preprocess()
    bsmd.compute_fft_blocks()
    bsmd.perform_bsmd()
    bsmd.save_results("bsmd.hdf5")
    _assert_provenance_block(tmp_path / "bsmd.hdf5")

    stpod = STPODAnalyzer(
        file_path="prov_stpod",
        embedding_dim=3,
        n_modes_save=3,
        **common,
    )
    stpod.load_and_preprocess()
    stpod.perform_stpod()
    stpod.save_results("stpod.hdf5")
    _assert_provenance_block(tmp_path / "stpod.hdf5")


def test_provenance_config_hash_stable_and_discriminating(tmp_path: Path) -> None:
    data = {"modes": np.arange(6.0).reshape(3, 2)}
    write_results(tmp_path / "a.h5", data, attrs={"analysis_type": "pod", "nfft": 8})
    write_results(tmp_path / "b.h5", data, attrs={"analysis_type": "pod", "nfft": 8})
    write_results(tmp_path / "c.h5", data, attrs={"analysis_type": "pod", "nfft": 16})

    def cfg_hash(path: Path) -> str:
        with h5py.File(path, "r") as handle:
            raw = handle.attrs["prov_config_sha256"]
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    assert cfg_hash(tmp_path / "a.h5") == cfg_hash(tmp_path / "b.h5")
    assert cfg_hash(tmp_path / "a.h5") != cfg_hash(tmp_path / "c.h5")


def test_provenance_legacy_file_empty_view(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.h5"
    with h5py.File(legacy, "w") as handle:
        handle.attrs["analysis_type"] = "pod"
        handle.create_dataset("modes", data=np.arange(6.0).reshape(3, 2))
    res = read_results(legacy)
    assert res.provenance == {}
    assert res.modes is not None and res.modes.shape == (3, 2)


def test_config_sha256_ignores_prov_keys() -> None:
    """A prov_ key must not change the digest (timestamp leak cannot be shown by same-second writes)."""
    from openmodalpy.core.provenance import config_sha256

    plain = config_sha256({"nfft": 8})
    poisoned = config_sha256({"nfft": 8, "prov_created_utc": "2099-01-01T00:00:00Z"})
    assert plain == poisoned


def test_config_sha256_never_raises_on_pathological_attrs(tmp_path: Path) -> None:
    """A cyclic or unstringable attr value must not abort the write."""
    data = {"modes": np.arange(6.0).reshape(3, 2)}

    class _Hostile:
        def __str__(self):
            raise RuntimeError("no string for you")

    cyclic: dict = {"nfft": 8}
    cyclic["self"] = cyclic
    write_results(tmp_path / "cyclic.h5", data, attrs=cyclic)
    write_results(tmp_path / "hostile.h5", data, attrs={"nfft": 8, "x": _Hostile()})
    for name in ("cyclic.h5", "hostile.h5"):
        with h5py.File(tmp_path / name, "r") as handle:
            raw = handle.attrs["prov_config_sha256"]
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        assert text  # non-empty (hash or "unavailable")


def test_prov_fft_backend_matches_fftkit() -> None:
    import fftkit

    from openmodalpy.core.provenance import collect_provenance

    block = collect_provenance({})
    assert block["prov_fft_backend"] == str(fftkit.DEFAULT_BACKEND)


def test_unknown_blas_threads_report_zero() -> None:
    """With policy=0 (all cores), a failed observation records 0, not a fabricated 1."""
    import threadpoolctl

    import openmodalpy as omp
    from openmodalpy.core import provenance as prov

    def _boom(*_a, **_k):
        raise RuntimeError("threadpoolctl unavailable")

    saved = threadpoolctl.threadpool_info
    previous = omp.get_blas_threads()
    threadpoolctl.threadpool_info = _boom
    try:
        omp.set_blas_threads(0)
        threads = prov.collect_provenance({})["prov_blas_threads"]
    finally:
        threadpoolctl.threadpool_info = saved
        omp.set_blas_threads(previous)
    assert int(threads) == 0


def test_new_prov_keys_present_pod_and_dmd(tmp_path: Path) -> None:
    """AC1 (regression subset): the four new keys land for POD and DMD alike."""
    field = _toy_field()
    common = dict(
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
    )
    new_keys = {"prov_blas", "prov_platform", "prov_machine", "prov_hdf5_version"}

    pod = PODAnalyzer(file_path="new_prov_pod", n_modes_save=3, **common)
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results("new_prov_pod.hdf5")
    with h5py.File(tmp_path / "new_prov_pod.hdf5", "r") as handle:
        pod_attrs = dict(handle.attrs)

    dmd = DMDAnalyzer(file_path="new_prov_dmd", n_modes_save=3, rank=3, **common)
    dmd.load_and_preprocess()
    dmd.perform_dmd()
    dmd.save_results("new_prov_dmd.hdf5")
    with h5py.File(tmp_path / "new_prov_dmd.hdf5", "r") as handle:
        dmd_attrs = dict(handle.attrs)

    for attrs in (pod_attrs, dmd_attrs):
        for key in new_keys:
            assert key in attrs, f"missing {key}"
            value = attrs[key]
            text = value.decode() if isinstance(value, bytes) else value
            assert isinstance(text, str) and text, f"{key} empty or not a string ({value!r})"


def test_prov_blas_matches_expected_format(tmp_path: Path) -> None:
    """Every `prov_blas` entry is a threadpoolctl record, or the text is "unknown".

    `_blas_identity` returns the sentinel "unknown" when threadpoolctl reports
    no bound threadpool. That is a supported result, not a failure: the
    provenance block must never raise, and macOS wheels that link Accelerate
    report no pool. So accept the sentinel, but only as the whole text.

    Every other entry must match the record shape. The check is on ALL
    entries, not on any one of them, so a malformed entry cannot hide behind
    a well-formed sibling.
    """
    data = {"modes": np.arange(6.0).reshape(3, 2)}
    write_results(tmp_path / "blas.h5", data, attrs={"analysis_type": "pod"})
    with h5py.File(tmp_path / "blas.h5", "r") as handle:
        raw = handle.attrs["prov_blas"]
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    if text == "unknown":
        return
    entries = text.split("; ")
    pattern = re.compile(r"^\w+ \S+ \(\w+\)$")
    assert entries, text
    for entry in entries:
        assert pattern.match(entry), f"malformed entry {entry!r} in {text!r}"


# Pinned at HEAD before the prov_blas/prov_platform/prov_machine/prov_hdf5_version
# addition: config_sha256({"analysis_type": "pod", "nfft": 8}). The new prov_ keys
# must never move this digest, since config_sha256 excludes every prov_ key by
# construction.
_PINNED_CONFIG_SHA256 = "bd6a0fe6dd4174dae76ae40cac1fabd227a7e20584747c27f28928dae4361e39"


def test_config_sha256_unchanged_by_new_prov_keys(tmp_path: Path) -> None:
    from openmodalpy.core.provenance import config_sha256

    attrs = {"analysis_type": "pod", "nfft": 8}
    assert config_sha256(attrs) == _PINNED_CONFIG_SHA256

    data = {"modes": np.arange(6.0).reshape(3, 2)}
    write_results(tmp_path / "pinned.h5", data, attrs=attrs)
    with h5py.File(tmp_path / "pinned.h5", "r") as handle:
        raw = handle.attrs["prov_config_sha256"]
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    assert text == _PINNED_CONFIG_SHA256


def test_provenance_legacy_old_key_set_missing_new_keys(tmp_path: Path) -> None:
    """A file written with only the old twelve prov_ keys reads without raising

    and exposes no value for the four keys added since.
    """
    legacy = tmp_path / "legacy_old_prov.h5"
    with h5py.File(legacy, "w") as handle:
        handle.attrs["analysis_type"] = "pod"
        handle.attrs["prov_openmodalpy_version"] = "0.4.0"
        handle.attrs["prov_python_version"] = "3.11.0"
        handle.attrs["prov_numpy_version"] = "1.26.0"
        handle.attrs["prov_scipy_version"] = "1.11.0"
        handle.attrs["prov_h5py_version"] = "3.10.0"
        handle.attrs["prov_fftkit_version"] = "0.1.0"
        handle.attrs["prov_fft_backend"] = "numpy"
        handle.attrs["prov_blas_threads"] = 1
        handle.attrs["prov_config_sha256"] = "0" * 64
        handle.attrs["prov_created_utc"] = "2026-01-01T00:00:00Z"
        handle.attrs["prov_git_sha"] = "unavailable"
        handle.attrs["prov_seed"] = "none"
        handle.create_dataset("modes", data=np.arange(6.0).reshape(3, 2))

    res = read_results(legacy)
    assert res.modes is not None and res.modes.shape == (3, 2)
    assert set(res.provenance) == {
        "openmodalpy_version",
        "python_version",
        "numpy_version",
        "scipy_version",
        "h5py_version",
        "fftkit_version",
        "fft_backend",
        "blas_threads",
        "config_sha256",
        "created_utc",
        "git_sha",
        "seed",
    }
    for key in ("blas", "platform", "machine", "hdf5_version"):
        assert key not in res.provenance
