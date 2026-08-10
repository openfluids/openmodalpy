"""Result contract: every producer writes the same dataset names and values.

One reader (:func:`read_results`) loads any result file — including the old
capitalised layout — into :class:`AnalysisResults`. Names and shapes alone are
not enough: each in-memory array is compared element-wise with what comes back
from the file. The gate for openmodalpy-unify-result-contract-vig greps this
file for every producer name.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    PSDPODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
    read_results,
)
from openmodalpy.commands import analyze_from_config


def _toy_field(ns: int = 16, nspace: int = 8) -> dict:
    rng = np.random.default_rng(0)
    nx = int(np.sqrt(nspace))
    ny = max(1, nspace // nx)
    nspace = nx * ny
    t = np.linspace(0.0, 2.0 * np.pi, ns, endpoint=False)
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    q = 1.0 + np.outer(np.sin(t), np.sin(2.0 * np.pi * np.tile(x, ny)))
    q = q + 0.3 * rng.standard_normal(q.shape)
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": x,
        "y": y,
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }


def _assert_canonical_keys(path: Path, required: set[str]) -> None:
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
    for name in required:
        assert name in keys, f"{path}: missing canonical dataset '{name}' (have {sorted(keys)})"
    capitalised = {
        "Modes",
        "Eigenvalues",
        "TimeCoefficients",
        "Freq",
        "St",
        "Modes1",
        "Modes2",
        "Weights",
    }
    assert not (keys & capitalised), f"{path}: still writing capitalised names {keys & capitalised}"


def _assert_roundtrip(disk: Any, mem: Any, name: str) -> None:
    """Loaded array must match the producer's in-memory array element-wise.

    gzip + h5py preserve values and dtypes, so exact equality is the default.
    """
    assert disk is not None, f"{name}: missing on disk"
    assert mem is not None, f"{name}: missing in memory"
    np.testing.assert_array_equal(disk, mem, err_msg=name)


def test_result_contract_all_producers(tmp_path: Path) -> None:
    """Every producer writes lowercase keys and round-trips array values.

    Producers covered (gate greps these names): PODAnalyzer, MPODAnalyzer,
    DMDAnalyzer, SPODAnalyzer, BSMDAnalyzer, STPODAnalyzer, psd_pod.
    """
    field = _toy_field()
    common = dict(
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
    )

    # PODAnalyzer
    pod = PODAnalyzer(file_path="pod_contract", n_modes_save=3, **common)
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results("pod.hdf5")
    pod_path = tmp_path / "pod.hdf5"
    _assert_canonical_keys(pod_path, {"modes", "eigenvalues", "time_coefficients"})
    pod_res = read_results(pod_path)
    assert pod_res.modes is not None and pod_res.modes.shape[1] == 3
    _assert_roundtrip(pod_res.modes, pod.modes, "POD.modes")
    _assert_roundtrip(pod_res.eigenvalues, pod.eigenvalues, "POD.eigenvalues")
    _assert_roundtrip(pod_res.time_coefficients, pod.time_coefficients, "POD.time_coefficients")
    _assert_roundtrip(pod_res.W, pod.W, "POD.W")
    _assert_roundtrip(pod_res.temporal_mean, pod.temporal_mean, "POD.temporal_mean")
    _assert_roundtrip(pod_res.x, pod.data["x"], "POD.x")
    _assert_roundtrip(pod_res.y, pod.data["y"], "POD.y")

    # MPODAnalyzer (inherits POD save_results)
    mpod = MPODAnalyzer(file_path="mpod_contract", n_modes_save=2, band_edges=[0.0, 2.0, 5.0], **common)
    mpod.load_and_preprocess()
    mpod.perform_mpod()
    mpod.save_results("mpod.hdf5")
    mpod_path = tmp_path / "mpod.hdf5"
    _assert_canonical_keys(mpod_path, {"modes", "eigenvalues", "time_coefficients"})
    mpod_res = read_results(mpod_path)
    assert mpod_res.eigenvalues is not None
    _assert_roundtrip(mpod_res.modes, mpod.modes, "mPOD.modes")
    _assert_roundtrip(mpod_res.eigenvalues, mpod.eigenvalues, "mPOD.eigenvalues")
    _assert_roundtrip(mpod_res.time_coefficients, mpod.time_coefficients, "mPOD.time_coefficients")
    _assert_roundtrip(mpod_res.W, mpod.W, "mPOD.W")
    _assert_roundtrip(mpod_res.temporal_mean, mpod.temporal_mean, "mPOD.temporal_mean")
    _assert_roundtrip(mpod_res.x, mpod.data["x"], "mPOD.x")
    _assert_roundtrip(mpod_res.y, mpod.data["y"], "mPOD.y")

    # DMDAnalyzer
    dmd = DMDAnalyzer(file_path="dmd_contract", n_modes_save=2, **common, rank=2)
    dmd.load_and_preprocess()
    dmd.perform_dmd()
    dmd.save_results("dmd.hdf5")
    dmd_path = tmp_path / "dmd.hdf5"
    _assert_canonical_keys(dmd_path, {"modes", "eigenvalues", "time_coefficients", "amplitudes"})
    dmd_res = read_results(dmd_path)
    assert dmd_res.amplitudes is not None
    _assert_roundtrip(dmd_res.modes, dmd.modes, "DMD.modes")
    _assert_roundtrip(dmd_res.eigenvalues, dmd.eigenvalues, "DMD.eigenvalues")
    _assert_roundtrip(dmd_res.time_coefficients, dmd.time_coefficients, "DMD.time_coefficients")
    _assert_roundtrip(dmd_res.amplitudes, dmd.amplitudes, "DMD.amplitudes")
    if dmd.omega.size > 0:
        _assert_roundtrip(dmd_res.omega, dmd.omega, "DMD.omega")
    _assert_roundtrip(dmd_res.x, dmd.data["x"], "DMD.x")
    _assert_roundtrip(dmd_res.y, dmd.data["y"], "DMD.y")

    # SPODAnalyzer
    spod = SPODAnalyzer(file_path="spod_contract", nfft=8, overlap=0.0, **common)
    spod.load_and_preprocess()
    spod.compute_fft_blocks()
    spod.perform_spod()
    spod.save_results("spod.hdf5")
    spod_path = tmp_path / "spod.hdf5"
    _assert_canonical_keys(spod_path, {"modes", "eigenvalues", "freq", "st"})
    # Inspect the raw datasets BEFORE reading. x_coords/y_coords/z_coords are
    # LEGACY_ALIASES now, so a reintroduced duplicate would make read_results
    # emit a DeprecationWarning — an error under this suite's filters — and the
    # check would never be reached. Done here, it pins the writer on its own.
    with h5py.File(spod_path, "r") as handle:
        raw_keys = set(handle.keys())
    assert not (raw_keys & {"x_coords", "y_coords", "z_coords"}), (
        f"{spod_path}: SPOD still writes duplicate coordinate datasets "
        f"{raw_keys & {'x_coords', 'y_coords', 'z_coords'}}"
    )
    spod_res = read_results(spod_path)
    assert spod_res.freq is not None and spod_res.st is not None
    assert spod_res.W is not None  # SPOD used to write Weights
    _assert_roundtrip(spod_res.modes, spod.modes, "SPOD.modes")
    _assert_roundtrip(spod_res.eigenvalues, spod.eigenvalues, "SPOD.eigenvalues")
    _assert_roundtrip(spod_res.freq, spod.freq, "SPOD.freq")
    _assert_roundtrip(spod_res.st, spod.St, "SPOD.st")
    _assert_roundtrip(spod_res.W, spod.W, "SPOD.W")
    if spod.time_coefficients is not None and spod.time_coefficients.size > 0:
        _assert_roundtrip(spod_res.time_coefficients, spod.time_coefficients, "SPOD.time_coefficients")
    if spod.qhat is not None and spod.qhat.size > 0:
        _assert_roundtrip(spod_res.FFTBlocks, spod.qhat, "SPOD.FFTBlocks")
    _assert_roundtrip(spod_res.x, spod.data["x"], "SPOD.x")
    _assert_roundtrip(spod_res.y, spod.data["y"], "SPOD.y")
    # BSMDAnalyzer — modes1 and modes2 compared separately so a swap fails
    bsmd = BSMDAnalyzer(
        file_path="bsmd_contract",
        nfft=8,
        overlap=0.0,
        use_static_triads=True,
        static_triads=[(0, 0, 0)],
        use_parallel=False,
        **common,
    )
    bsmd.load_and_preprocess()
    bsmd.compute_fft_blocks()
    bsmd._perform_static_bsmd_core()
    bsmd.save_results("bsmd.hdf5")
    bsmd_path = tmp_path / "bsmd.hdf5"
    _assert_canonical_keys(bsmd_path, {"modes1", "modes2", "triads", "eigenvalues"})
    bsmd_res = read_results(bsmd_path)
    assert bsmd_res.modes1 is not None and bsmd_res.modes2 is not None
    _assert_roundtrip(bsmd_res.modes1, bsmd.modes1, "BSMD.modes1")
    _assert_roundtrip(bsmd_res.modes2, bsmd.modes2, "BSMD.modes2")
    _assert_roundtrip(bsmd_res.triads, np.array(bsmd.triads), "BSMD.triads")
    _assert_roundtrip(bsmd_res.eigenvalues, bsmd.eigenvalues, "BSMD.eigenvalues")
    _assert_roundtrip(bsmd_res.W, bsmd.W, "BSMD.W")
    if bsmd.energy_map.size:
        _assert_roundtrip(bsmd_res.energy_map, bsmd.energy_map, "BSMD.energy_map")
    _assert_roundtrip(bsmd_res.x, bsmd.data["x"], "BSMD.x")
    _assert_roundtrip(bsmd_res.y, bsmd.data["y"], "BSMD.y")

    # STPODAnalyzer
    stpod = STPODAnalyzer(file_path="stpod_contract", embedding_dim=2, n_modes_save=2, **common)
    stpod.load_and_preprocess()
    stpod.perform_stpod()
    stpod.save_results("stpod.hdf5")
    stpod_path = tmp_path / "stpod.hdf5"
    _assert_canonical_keys(stpod_path, {"modes", "eigenvalues", "time_coefficients"})
    stpod_res = read_results(stpod_path)
    assert stpod_res.modes is not None
    _assert_roundtrip(stpod_res.modes, stpod.modes, "ST-POD.modes")
    _assert_roundtrip(stpod_res.eigenvalues, stpod.eigenvalues, "ST-POD.eigenvalues")
    _assert_roundtrip(stpod_res.time_coefficients, stpod.time_coefficients, "ST-POD.time_coefficients")
    _assert_roundtrip(stpod_res.W, stpod.W, "ST-POD.W")
    if stpod.temporal_mean.size > 0:
        _assert_roundtrip(stpod_res.temporal_mean, stpod.temporal_mean, "ST-POD.temporal_mean")
    _assert_roundtrip(stpod_res.x, stpod.data["x"], "ST-POD.x")
    _assert_roundtrip(stpod_res.y, stpod.data["y"], "ST-POD.y")

    # psd_pod via PSDPODAnalyzer (in-memory arrays for value comparison)
    psd = PSDPODAnalyzer(
        file_path="psd_pod_contract",
        n_modes_save=2,
        nfft=8,
        overlap=0.0,
        use_parallel=False,
        **common,
    )
    psd.load_and_preprocess()
    psd.compute_fft_blocks()
    psd.perform_psd_pod()
    psd.save_results("psd_pod.hdf5")
    psd_path = tmp_path / "psd_pod.hdf5"
    _assert_canonical_keys(psd_path, {"modes", "eigenvalues", "time_coefficients", "freq", "st"})
    psd_res = read_results(psd_path)
    assert psd_res.attrs.get("analysis_type") == "psd_pod"
    _assert_roundtrip(psd_res.modes, psd.modes, "psd_pod.modes")
    _assert_roundtrip(psd_res.eigenvalues, psd.eigenvalues, "psd_pod.eigenvalues")
    _assert_roundtrip(psd_res.time_coefficients, psd.time_coefficients, "psd_pod.time_coefficients")
    _assert_roundtrip(psd_res.freq, psd.freq, "psd_pod.freq")
    _assert_roundtrip(psd_res.st, psd.St, "psd_pod.st")

    # config path still reaches psd_pod and writes the same canonical keys
    config_path = tmp_path / "psd_pod.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "name": "psd contract",
                "description": "result contract coverage for psd_pod",
                "case": {
                    "name": "toy",
                    "case_type": "analytical",
                    "data": {
                        "kind": "generator",
                        "name": "double_gyre",
                        "params": {"Nx": 6, "Ny": 4, "Nt": 16},
                    },
                    "spatial_weight_type": "uniform",
                    "n_modes_save": 2,
                    "nfft": 8,
                    "overlap": 0.0,
                    "generate_plots": False,
                    "results_root": str(tmp_path / "psd_results"),
                    "figures_root": str(tmp_path / "psd_figures"),
                },
                "runs": [{"id": "psd", "method": "psd-pod"}],
            },
            indent=2,
        )
    )
    outcome = analyze_from_config(config_path, method="psd-pod")
    assert outcome.results_path is not None
    config_psd_path = Path(outcome.results_path)
    _assert_canonical_keys(config_psd_path, {"modes", "eigenvalues", "time_coefficients", "freq", "st"})
    assert read_results(config_psd_path).attrs.get("analysis_type") == "psd_pod"


def test_read_results_handles_a_zero_dimensional_dataset(tmp_path: Path) -> None:
    """A 0-d (scalar) dataset lands in extra; normal datasets stay intact.

    No writer emits a 0-d dataset today; the reader must still survive one
    instead of raising on ``handle[key][:]`` against a scalar dataspace.
    """
    path = tmp_path / "zerod.hdf5"
    modes = np.arange(6.0).reshape(3, 2)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("modes", data=modes)
        handle.create_dataset("scalar_thing", data=np.float64(2.5))

    res = read_results(path)

    assert "scalar_thing" in res.extra
    # Shape, not just the value: a reader that turned the scalar into shape (1,)
    # would satisfy a value-only check while not handling 0-d at all.
    assert np.asarray(res.extra["scalar_thing"]).shape == ()
    assert float(np.asarray(res.extra["scalar_thing"])) == 2.5
    assert res.modes is not None
    np.testing.assert_array_equal(res.modes, modes)


def test_mixed_legacy_bsmd_file_resolves_every_key(tmp_path: Path) -> None:
    """Pre-unification BSMD mix: Modes1/Modes2 + lowercase triads/eigenvalues.

    Capitalised mode keys resolve and warn; already-canonical keys pass
    through without a deprecation notice.
    """
    path = tmp_path / "mixed_legacy_bsmd.hdf5"
    modes1 = np.ones((4, 2))
    modes2 = 2.0 * np.ones((4, 2))
    triads = np.arange(6.0).reshape(2, 3)
    eigenvalues = np.array([1.0, 0.5])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Modes1", data=modes1)
        handle.create_dataset("Modes2", data=modes2)
        handle.create_dataset("triads", data=triads)
        handle.create_dataset("eigenvalues", data=eigenvalues)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_results(path)

    assert res.modes1 is not None
    assert res.modes2 is not None
    np.testing.assert_array_equal(res.modes1, modes1)
    np.testing.assert_array_equal(res.modes2, modes2)
    assert res.triads is not None
    assert res.eigenvalues is not None
    np.testing.assert_array_equal(res.triads, triads)
    np.testing.assert_array_equal(res.eigenvalues, eigenvalues)

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    messages = [str(w.message) for w in dep]
    assert any("Modes1" in m for m in messages)
    assert any("Modes2" in m for m in messages)
    assert not any("'triads'" in m for m in messages)
    assert not any("'eigenvalues'" in m for m in messages)


def test_read_results_legacy_layout_emits_deprecation(tmp_path: Path) -> None:
    """A file written with the old capitalised SPOD keys still loads."""
    legacy = tmp_path / "legacy_spod.hdf5"
    rng = np.random.default_rng(1)
    nfr, nsp, nmd = 5, 6, 2
    with h5py.File(legacy, "w") as handle:
        handle.create_dataset("Modes", data=rng.standard_normal((nfr, nsp, nmd)))
        handle.create_dataset("Eigenvalues", data=rng.standard_normal((nfr, nmd)))
        handle.create_dataset("TimeCoefficients", data=rng.standard_normal((nfr, nmd, 3)))
        handle.create_dataset("Freq", data=np.linspace(0, 1, nfr))
        handle.create_dataset("St", data=np.linspace(0, 2, nfr))
        handle.create_dataset("Weights", data=np.ones(nsp))
        handle.attrs["analysis_type"] = "spod"

    with pytest.warns(DeprecationWarning, match="legacy name"):
        res = read_results(legacy)

    assert res.modes is not None and res.modes.shape == (nfr, nsp, nmd)
    assert res.eigenvalues is not None and res.eigenvalues.shape == (nfr, nmd)
    assert res.time_coefficients is not None
    assert res.freq is not None and res.st is not None
    assert res.W is not None and res.W.shape == (nsp,)


def test_read_results_legacy_coords_spellings_map_to_canonical(tmp_path: Path) -> None:
    """Older SPOD files that carried only x_coords/y_coords still read as x/y."""
    legacy = tmp_path / "legacy_coords.hdf5"
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 2)
    with h5py.File(legacy, "w") as handle:
        handle.create_dataset("modes", data=np.zeros((8, 2)))
        handle.create_dataset("eigenvalues", data=np.ones(2))
        handle.create_dataset("x_coords", data=x)
        handle.create_dataset("y_coords", data=y)

    # Match the text common to every legacy warning, not just x_coords: this
    # file reads with filterwarnings=error, and pytest re-emits warnings that
    # the match misses, so a narrower pattern turns the y_coords warning into
    # an error. The specific key is asserted below instead.
    with pytest.warns(DeprecationWarning, match="legacy name") as caught:
        res = read_results(legacy)

    assert res.x is not None
    assert res.y is not None
    np.testing.assert_array_equal(res.x, x)
    np.testing.assert_array_equal(res.y, y)
    assert "x_coords" not in res.extra
    assert "y_coords" not in res.extra
    assert any("x_coords" in str(w.message) for w in caught)


def test_spod_load_results_rejects_a_file_without_modes(tmp_path: Path) -> None:
    """A file that is not a SPOD result must fail loudly, not load as empty arrays."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    not_spod = results_dir / "not_spod.hdf5"
    with h5py.File(not_spod, "w") as handle:
        handle.create_dataset("x", data=np.linspace(0, 1, 4))
        handle.attrs["analysis_type"] = "spod"

    analyzer = SPODAnalyzer(
        file_path="not_spod",
        nfft=8,
        overlap=0.0,
        **_analyzer_ctor_kwargs(results_dir),
    )

    with pytest.raises(KeyError, match="not a SPOD result file"):
        analyzer.load_results("not_spod.hdf5")


def _legacy_capitalised_file(path: Path, modes: np.ndarray, eigenvalues: np.ndarray, coeffs: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Modes", data=modes)
        handle.create_dataset("Eigenvalues", data=eigenvalues)
        handle.create_dataset("TimeCoefficients", data=coeffs)


def _analyzer_ctor_kwargs(results_dir: Path) -> dict:
    """Shared real-constructor kwargs for load_results tests (no half-built objects)."""
    field = _toy_field()
    return dict(
        results_dir=results_dir,
        figures_dir=results_dir,
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
    )


def test_legacy_capitalised_file_loads_through_pod(tmp_path: Path) -> None:
    """A pre-unification POD file with capitalised keys loads via the reader."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rng = np.random.default_rng(2)
    modes = rng.standard_normal((12, 2))
    eigenvalues = np.array([3.0, 1.5])
    coeffs = rng.standard_normal((8, 2))
    _legacy_capitalised_file(results_dir / "legacy_pod.hdf5", modes, eigenvalues, coeffs)

    analyzer = PODAnalyzer(
        file_path="legacy_pod",
        n_modes_save=2,
        use_parallel=False,
        **_analyzer_ctor_kwargs(results_dir),
    )

    with pytest.warns(DeprecationWarning, match="legacy name"):
        analyzer.load_results("legacy_pod.hdf5")

    np.testing.assert_array_equal(analyzer.modes, modes)
    np.testing.assert_array_equal(analyzer.eigenvalues, eigenvalues)
    np.testing.assert_array_equal(analyzer.time_coefficients, coeffs)


def test_legacy_capitalised_file_loads_through_stpod(tmp_path: Path) -> None:
    """A pre-unification ST-POD file with capitalised keys loads via the reader."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rng = np.random.default_rng(3)
    modes = rng.standard_normal((18, 2))
    eigenvalues = np.array([4.0, 2.0])
    coeffs = rng.standard_normal((6, 2))
    _legacy_capitalised_file(results_dir / "legacy_stpod.hdf5", modes, eigenvalues, coeffs)

    analyzer = STPODAnalyzer(
        file_path="legacy_stpod",
        embedding_dim=3,
        n_modes_save=2,
        **_analyzer_ctor_kwargs(results_dir),
    )

    with pytest.warns(DeprecationWarning, match="legacy name"):
        analyzer.load_results("legacy_stpod.hdf5")

    np.testing.assert_array_equal(analyzer.modes, modes)
    np.testing.assert_array_equal(analyzer.eigenvalues, eigenvalues)
    np.testing.assert_array_equal(analyzer.time_coefficients, coeffs)
    assert np.isnan(analyzer.total_energy)
    assert np.isnan(analyzer.energy_captured_fraction)


def test_legacy_capitalised_file_loads_through_dmd(tmp_path: Path) -> None:
    """A pre-unification DMD file with capitalised keys loads via the reader."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rng = np.random.default_rng(4)
    modes = rng.standard_normal((12, 2)) + 1j * rng.standard_normal((12, 2))
    eigenvalues = np.array([0.9 + 0.1j, 0.8 - 0.1j])
    coeffs = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
    _legacy_capitalised_file(results_dir / "legacy_dmd.hdf5", modes, eigenvalues, coeffs)

    analyzer = DMDAnalyzer(
        file_path="legacy_dmd",
        n_modes_save=2,
        rank=2,
        **_analyzer_ctor_kwargs(results_dir),
    )

    with pytest.warns(DeprecationWarning, match="legacy name"):
        analyzer.load_results("legacy_dmd.hdf5")

    np.testing.assert_array_equal(analyzer.modes, modes)
    np.testing.assert_array_equal(analyzer.eigenvalues, eigenvalues)
    np.testing.assert_array_equal(analyzer.time_coefficients, coeffs)
    np.testing.assert_array_equal(analyzer.amplitudes, np.abs(eigenvalues))
    assert analyzer._dmd_method == "ls"
    assert analyzer._dmd_delays == 1
    assert analyzer._dmd_named_variant == "dmd"
