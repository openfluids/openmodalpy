"""One loader, every method: the package's central round-trip promise.

The promise (Ricardo, 2026-08-15): you write one data loader for your
discretization and every advertised method is reachable with it — you loop over
the methods, then the next, then the next. This module hands a single
file-backed loader to all nine advertised methods (seven analyzer classes plus
the two DMD delay variants) and, for each one in the loop table, runs
load_and_preprocess -> perform -> save_results, reloads into a fresh instance
with the same loader, and compares the reloaded arrays element-wise. If any
method stops accepting the shared loader or stops round-tripping its results,
this module fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
)
from openmodalpy.example_data import generate_taylor_green

NX, NY, NS = 8, 8, 64
NFFT = 16  # Welch rows: Ns=64, overlap 0.5 -> 7 blocks, rfft bins 0..8
DELAYS = 4


def _npz_loader(path: str) -> dict[str, Any]:
    """Read the .npz written by the fixture into the analyzer data dict."""
    with np.load(path) as handle:
        loaded = {key: handle[key] for key in handle.files}
    # None cannot be stored in an .npz; an absent z means "no third axis".
    loaded.setdefault("z", None)
    return loaded


@pytest.fixture
def taylor_green_npz(tmp_path: Path) -> str:
    """Write a small Taylor-Green dataset to disk once; the loader reads it back."""
    data = generate_taylor_green(Nx=NX, Ny=NY, Nt=NS)
    path = tmp_path / "data" / "taylor_green.npz"
    path.parent.mkdir()
    np.savez(path, **{k: v for k, v in data.items() if k not in ("z", "metadata")})
    return str(path)


def _assert_finite(label: str, value: Any) -> None:
    """The array exists and holds no NaN/Inf (complex parts included)."""
    arr = np.asarray(value)
    assert arr.size > 0, f"{label}: no results produced"
    assert np.all(np.isfinite(arr)), f"{label}: non-finite values"


def _check_modal(analyzer: Any, n_space: int) -> None:
    """POD family / DMD: modes are (k * n_space, n_modes), one eigenvalue per mode."""
    modes = np.asarray(analyzer.modes)
    eigenvalues = np.asarray(analyzer.eigenvalues)
    assert modes.shape[0] % n_space == 0, f"modes.shape={modes.shape} vs n_space={n_space}"
    assert modes.shape[1] == eigenvalues.size >= 1
    assert eigenvalues.size <= analyzer.n_modes_save


def _check_spod(analyzer: Any, n_space: int) -> None:
    """SPOD: modes are (n_freq, n_space, nblocks), eigenvalues (n_freq, nblocks)."""
    modes = np.asarray(analyzer.modes)
    eigenvalues = np.asarray(analyzer.eigenvalues)
    assert modes.ndim == 3 and modes.shape[1] == n_space
    assert modes.shape[0] == eigenvalues.shape[0] == np.asarray(analyzer.freq).size
    assert modes.shape[2] == eigenvalues.shape[1] >= 2


def _check_bsmd(analyzer: Any, n_space: int) -> None:
    """BSMD: modes1/modes2 are (n_triads, n_space), one eigenvalue per triad."""
    modes1 = np.asarray(analyzer.modes1)
    assert modes1.shape[1] == n_space
    assert np.asarray(analyzer.modes2).shape == modes1.shape
    assert np.asarray(analyzer.eigenvalues).size == modes1.shape[0]
    assert len(np.atleast_2d(np.asarray(analyzer.triads))) == modes1.shape[0]


@dataclass(frozen=True)
class MethodRow:
    """One advertised method: how to build it, call it, and verify it."""

    name: str
    cls: type
    ctor: dict[str, Any]
    perform: str
    needs_fft_blocks: bool
    compare: tuple[str, ...]
    check: Callable[[Any, int], None]
    perform_kwargs: dict[str, Any] = field(default_factory=dict)


ROWS: tuple[MethodRow, ...] = (
    MethodRow(
        "pod",
        PODAnalyzer,
        {"n_modes_save": 4},
        "perform_pod",
        False,
        ("modes", "eigenvalues", "time_coefficients"),
        _check_modal,
    ),
    # rank=1 matches the Taylor-Green field's exact rank; a larger request only
    # fires the "effective rank below requested" RuntimeWarning on this data.
    MethodRow(
        "mpod",
        MPODAnalyzer,
        {"n_modes_save": 3, "band_edges": [0.0, 0.5, 1.0], "band_scale": "normalized_nyquist"},
        "perform_mpod",
        False,
        ("modes", "eigenvalues", "time_coefficients"),
        _check_modal,
    ),
    MethodRow(
        "dmd",
        DMDAnalyzer,
        {"n_modes_save": 4, "rank": 1},
        "perform_dmd",
        False,
        ("modes", "eigenvalues", "time_coefficients", "amplitudes", "omega"),
        _check_modal,
    ),
    MethodRow(
        "hodmd",
        DMDAnalyzer,
        {"n_modes_save": 4, "rank": 1},
        "perform_dmd",
        False,
        ("modes", "eigenvalues", "time_coefficients", "amplitudes", "omega"),
        _check_modal,
        {"delays": DELAYS, "named_variant": "hodmd"},
    ),
    MethodRow(
        "tls_hodmd",
        DMDAnalyzer,
        {"n_modes_save": 4, "rank": 1},
        "perform_dmd",
        False,
        ("modes", "eigenvalues", "time_coefficients", "amplitudes", "omega"),
        _check_modal,
        {"method": "tls", "delays": DELAYS, "named_variant": "tls_hodmd"},
    ),
    MethodRow(
        "spod",
        SPODAnalyzer,
        {"nfft": NFFT, "overlap": 0.5},
        "perform_spod",
        True,
        ("modes", "eigenvalues", "freq", "St"),
        _check_spod,
    ),
    MethodRow(
        "bsmd",
        BSMDAnalyzer,
        {"nfft": NFFT, "overlap": 0.5, "use_parallel": False, "static_triads": [(1, 1, 2)]},
        "perform_bsmd",
        True,
        ("modes1", "modes2", "eigenvalues", "triads"),
        _check_bsmd,
    ),
    MethodRow(
        "stpod",
        STPODAnalyzer,
        {"embedding_dim": 4, "n_modes_save": 3},
        "perform_stpod",
        False,
        ("modes", "eigenvalues", "time_coefficients"),
        _check_modal,
    ),
    MethodRow(
        "psd_pod",
        PSDPODAnalyzer,
        {"nfft": NFFT, "overlap": 0.5, "n_modes_save": 3},
        "perform_psd_pod",
        True,
        ("modes", "eigenvalues", "time_coefficients", "freq", "St", "W"),
        _check_modal,
    ),
)


@pytest.mark.parametrize("row", ROWS, ids=[row.name for row in ROWS])
def test_one_loader_round_trip(row: MethodRow, taylor_green_npz: str, tmp_path: Path) -> None:
    """Every method runs on the one shared loader and round-trips its results."""
    common: dict[str, Any] = {
        "results_dir": str(tmp_path / row.name / "results"),
        "figures_dir": str(tmp_path / row.name / "figures"),
        "data_loader": _npz_loader,
        "spatial_weight_type": "uniform",
    }
    analyzer = row.cls(taylor_green_npz, **row.ctor, **common)
    analyzer.load_and_preprocess()
    if row.needs_fft_blocks:
        analyzer.compute_fft_blocks()
    getattr(analyzer, row.perform)(**row.perform_kwargs)

    n_space = int(np.asarray(analyzer.data["q"]).shape[1])
    row.check(analyzer, n_space)
    for name in row.compare:
        _assert_finite(f"{row.name}.{name}", getattr(analyzer, name))

    analyzer.save_results()

    fresh = row.cls(taylor_green_npz, **row.ctor, **common)
    fresh.load_and_preprocess()
    fresh.load_results()

    # Exact comparison: results are float64/complex128 in HDF5, so save/load
    # must be bit-preserving; any tolerance would hide a real round-trip bug.
    for name in row.compare:
        saved = np.asarray(getattr(analyzer, name))
        loaded = np.asarray(getattr(fresh, name))
        np.testing.assert_array_equal(loaded, saved, err_msg=f"{row.name}.{name} changed across save/load")


def test_load_once_loop_over_methods(taylor_green_npz: str, tmp_path: Path) -> None:
    """The documented load-once path: one dict, every analyzer, no per-method reload."""
    loaded = _npz_loader(taylor_green_npz)
    seen_classes: set[type] = set()
    for row in ROWS:
        analyzer = row.cls(
            data=dict(loaded),
            results_dir=str(tmp_path / row.name / "results"),
            figures_dir=str(tmp_path / row.name / "figures"),
            spatial_weight_type="uniform",
            **row.ctor,
        )
        assert analyzer.file_path is None
        analyzer.load_and_preprocess()
        if row.needs_fft_blocks:
            analyzer.compute_fft_blocks()
        getattr(analyzer, row.perform)(**row.perform_kwargs)

        n_space = int(np.asarray(analyzer.data["q"]).shape[1])
        row.check(analyzer, n_space)
        for name in row.compare:
            _assert_finite(f"{row.name}.{name}", getattr(analyzer, name))
        seen_classes.add(row.cls)

    # The nine table rows cover exactly the seven advertised classes.
    assert len(seen_classes) == 7


def test_side_channel_data_assignment_still_works(taylor_green_npz: str, tmp_path: Path) -> None:
    """The legacy ``analyzer.data = d`` escape hatch keeps skipping the loader."""

    def _must_not_load(_path: str) -> dict[str, Any]:
        raise AssertionError("side-channel assignment must skip the loader")

    analyzer = PODAnalyzer(
        file_path="ignored",
        data_loader=_must_not_load,
        n_modes_save=4,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        spatial_weight_type="uniform",
    )
    analyzer.data = _npz_loader(taylor_green_npz)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    assert int(np.asarray(analyzer.data["q"]).shape[1]) == NX * NY
    _assert_finite("side_channel.eigenvalues", analyzer.eigenvalues)
