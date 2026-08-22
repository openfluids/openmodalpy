"""Cross-check openmodalpy SPOD against vendored PySPOD eigenvalues.

This module never imports PySPOD. The numbers live in
``tests/fixtures/reference/external_spod.json``, generated once outside the
repo by ``scripts/regen_external_spod.py`` against **pyspod==2.0.0**
(Python 3.12, NumPy 2.5.2, SciPy 1.18.0). The package has no PySPOD
dependency.

Both sides are built from the manufactured field in
``tests/test_spod_oracle.py`` (nfft 16, 8 blocks, zero overlap, dt 0.5,
4 spatial points, tones at bins 3 and 5). A real cosine of amplitude ``A``
on a spatially orthonormal mode has block coefficient ``A/2``, so the
closed-form eigenvalue at an interior bin is ``(A/2)**2 / dst``:
bin 3 -> 18.0 and 4.5, bin 5 -> 8.0.

The convention mapping is ``λ_openmodalpy = λ_pyspod × nfft × dt / 2``.
``nfft*dt`` is our division by the Strouhal step, which PySPOD does not do;
the 2 is PySPOD's interior-bin doubling, which we do not do. Confirmed by
sweep, not one case: PySPOD's eigenvalue is independent of dt (4.503960 at
dt 0.5, 1.0, 2.0). Predicted vs measured factor: 4 vs 3.996, 8 vs 7.993,
16 vs 15.986, 8 vs 7.990.

Why these tolerances
--------------------
Measured on the prescribed stack (Python 3.12, pyspod 2.0.0, NumPy 2.5.2,
SciPy 1.18.0) against this field, after the mapping:

* bin 3 mode 0: openmodalpy 18.0 vs mapped 18.019628, relative 1.09e-3
* bin 3 mode 1: openmodalpy  4.5 vs mapped  4.500004, relative 9.0e-7
* bin 5 mode 0: openmodalpy  8.0 vs mapped  8.015560, relative 1.95e-3

The residual is a window-definition difference, not a bug, and it cannot
be reconciled. PySPOD hard-codes the symmetric Hamming
``0.54-0.46*cos(2*pi*x/(N-1))`` and offers no other window (``n_dft`` must
be an int). openmodalpy uses ``scipy.signal.get_window(..., fftbins=True)``,
the periodic one. Constant-phase modes take coherent leakage from the other
tone; the mode whose block coefficients turn one full revolution is
orthogonal to that leakage and matches to 1e-6. So the residual is bin-
and mode-dependent, bounded here by ~2e-3.

``5e-3`` for every mapped comparison (~2.5x the worst measured residual).
It still discriminates: dropping the Strouhal division moves the answer by
8x, and the power normalisation by 0.734 — both orders above 5e-3.

``(nfft + nblocks) * eps`` (~5.3e-15) against the closed form. Amplitude-
normalised Hamming recovers that closed form exactly (ratio 1.000000);
this is the same FFT-plus-Gram round-off bound the oracle already uses.
Do not tighten the mapped bound past 5e-3: that is what the measured
window residual supports, not a pasted power of ten.

Already ruled out, do not chase: PySPOD's ``fullspectrum`` changes only
the returned bin count (16 vs 9), not the values at bins 3 and 5;
``mean_type`` and ``normalize_weights`` do not matter on this field. DC
and Nyquist are excluded — the closed form is wrong there (coefficient
``A``, not ``A/2``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openmodalpy import SPODAnalyzer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "reference" / "external_spod.json"
SCRIPT_PATH = ROOT / "scripts" / "regen_external_spod.py"

OCCUPIED = ((3, 0), (3, 1), (5, 0))
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load_regen():
    spec = importlib.util.spec_from_file_location("regen_external_spod", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    saved = {key: os.environ.get(key) for key in _THREAD_ENV_KEYS}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


@pytest.fixture(scope="module")
def regen():
    return _load_regen()


@pytest.fixture(scope="module")
def fixture_doc():
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _tol(doc: dict, key: str) -> float:
    return float(doc["tolerances"][key]["value"])


def _mapping_factor(doc: dict) -> float:
    construction = doc["construction"]
    return float(construction["nfft"]) * float(construction["dt"]) / 2.0


def _occupied_item(items: list, bin_idx: int, mode_idx: int) -> dict:
    for item in items:
        if int(item["bin"]) == bin_idx and int(item["mode"]) == mode_idx:
            return item
    raise KeyError(f"no occupied entry for bin={bin_idx} mode={mode_idx}")


def _rel_err(got: float, want: float) -> float:
    return float(abs(got - want) / abs(want))


def _field_snapshots(fixture_doc: dict) -> np.ndarray:
    """Manufactured snapshots taken FROM the fixture (not rebuilt)."""
    return np.asarray(fixture_doc["cases"]["manufactured"]["snapshots"], dtype=np.float64)


def _openmodalpy_eigs(q: np.ndarray, fixture_doc: dict, tmp_path: Path) -> np.ndarray:
    options = fixture_doc["solver_options"]["openmodalpy"]
    construction = fixture_doc["construction"]
    n_space = int(construction["n_space"])
    field = {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": float(construction["dt"]),
        "Nx": n_space,
        "Ny": 1,
        "Ns": int(q.shape[0]),
    }
    analyzer = SPODAnalyzer(
        file_path="external_spod",
        nfft=int(options["nfft"]),
        overlap=float(options["overlap"]),
        window_type=str(options["window_type"]),
        window_norm=str(options["window_norm"]),
        blockwise_mean=bool(options["blockwise_mean"]),
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: field,
        spatial_weight_type="prescribed",
        # The vendored PySPOD run and the closed form both use identity (ones)
        # spatial weights; prescribe ones so the comparison stays about the
        # spectral-energy convention rather than coordinate-derived volumes.
        spatial_weights=np.ones((n_space, 1)),
        use_parallel=bool(options["use_parallel"]),
        characteristic_length=options["characteristic_length"],
        characteristic_velocity=options["characteristic_velocity"],
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    return np.asarray(analyzer.eigenvalues)


def test_rebuilt_field_matches_vendored_numerically(regen, fixture_doc) -> None:
    vendored = _field_snapshots(fixture_doc)
    rebuilt = regen.manufactured_field()
    if not np.allclose(rebuilt, vendored, rtol=1e-12, atol=0.0):
        raise AssertionError(
            "rebuilt field does not match the vendored snapshots; "
            "the field changed and the vendored numbers no longer describe it"
        )


def test_loaded_field_is_bit_identical_to_generators(regen) -> None:
    """Python's float repr round-trips float64, so vendoring the field is lossless."""
    generated = regen.manufactured_field()
    loaded = regen.snapshots_from_json(json.loads(json.dumps(regen.snapshots_to_json(generated))))
    assert regen.float64_bits_equal(loaded, generated)


def test_live_script_construction_matches_fixture(regen, fixture_doc) -> None:
    construction = fixture_doc["construction"]
    assert regen.N_SPACE == construction["n_space"]
    assert regen.NFFT == construction["nfft"]
    assert regen.NBLOCKS == construction["nblocks"]
    assert regen.K_BIN == construction["k_bin"]
    assert regen.K_BIN2 == construction["k_bin2"]
    assert regen.DT == construction["dt"]
    assert regen.A1 == construction["A1"]
    assert regen.A2 == construction["A2"]
    assert regen.A3 == construction["A3"]
    assert list(regen.PHI1) == construction["phi1"]
    assert list(regen.PHI2) == construction["phi2"]
    assert list(regen.PHI3) == construction["phi3"]
    assert regen.dst() == construction["dst"]
    closed = construction["closed_form"]
    assert regen.expected_lambda(regen.A1) == _occupied_item(closed, 3, 0)["value"]
    assert regen.expected_lambda(regen.A2) == _occupied_item(closed, 3, 1)["value"]
    assert regen.expected_lambda(regen.A3) == _occupied_item(closed, 5, 0)["value"]


def test_script_restates_oracle_construction(regen) -> None:
    from tests import test_spod_oracle as oracle

    assert regen.N_SPACE == oracle.N_SPACE
    assert regen.NFFT == oracle.NFFT
    assert regen.NBLOCKS == oracle.NBLOCKS
    assert regen.K_BIN == oracle.K_BIN
    assert regen.K_BIN2 == oracle.K_BIN2
    assert regen.DT == oracle.DT
    assert regen.A1 == oracle.A1
    assert regen.A2 == oracle.A2
    assert regen.A3 == oracle.A3
    np.testing.assert_array_equal(regen.PHI1, oracle.PHI1)
    np.testing.assert_array_equal(regen.PHI2, oracle.PHI2)
    np.testing.assert_array_equal(regen.PHI3, oracle.PHI3)


@pytest.mark.parametrize(("bin_idx", "mode_idx"), OCCUPIED)
def test_openmodalpy_matches_vendored_pyspod(fixture_doc, tmp_path: Path, bin_idx: int, mode_idx: int) -> None:
    q = _field_snapshots(fixture_doc)
    got = float(_openmodalpy_eigs(q, fixture_doc, tmp_path)[bin_idx, mode_idx])
    raw = float(
        _occupied_item(fixture_doc["cases"]["manufactured"]["occupied"], bin_idx, mode_idx)["pyspod_eigenvalue"]
    )
    mapped = raw * _mapping_factor(fixture_doc)
    err = _rel_err(got, mapped)
    tol = _tol(fixture_doc, "mapped_vs_pyspod")
    assert err <= tol, (
        f"bin {bin_idx} mode {mode_idx}: openmodalpy {got:.6g} vs mapped PySPOD "
        f"{mapped:.6g} relative error {err:.3e} exceeds {tol:.3e}"
    )


@pytest.mark.parametrize(("bin_idx", "mode_idx"), OCCUPIED)
def test_openmodalpy_matches_closed_form(fixture_doc, tmp_path: Path, bin_idx: int, mode_idx: int) -> None:
    q = _field_snapshots(fixture_doc)
    got = float(_openmodalpy_eigs(q, fixture_doc, tmp_path)[bin_idx, mode_idx])
    want = float(_occupied_item(fixture_doc["construction"]["closed_form"], bin_idx, mode_idx)["value"])
    err = _rel_err(got, want)
    tol = _tol(fixture_doc, "closed_form")
    assert err <= tol, (
        f"bin {bin_idx} mode {mode_idx}: openmodalpy {got:.6g} vs closed form "
        f"{want:.6g} relative error {err:.3e} exceeds {tol:.3e}"
    )


def test_fixture_provenance_records_both_solvers_and_the_pinned_pyspod(fixture_doc) -> None:
    prov = fixture_doc["provenance"]
    assert prov["pyspod_version"] == "2.0.0"
    assert prov["python_version"].startswith("3.12")
    assert prov["numpy_version"]
    assert prov["scipy_version"]
    assert prov["generation_date"]
    construction = fixture_doc["construction"]
    assert construction["n_space"] == 4
    assert construction["nfft"] == 16
    assert construction["nblocks"] == 8
    assert construction["dt"] == 0.5
    assert construction["k_bin"] == 3
    assert construction["k_bin2"] == 5
    snapshots = _field_snapshots(fixture_doc)
    n_snapshots = int(construction["nblocks"]) * int(construction["nfft"])
    assert snapshots.shape == (n_snapshots, construction["n_space"])
    assert snapshots.dtype == np.float64
    occupied = {(int(item["bin"]), int(item["mode"])) for item in fixture_doc["cases"]["manufactured"]["occupied"]}
    assert occupied == set(OCCUPIED)
    options = fixture_doc["solver_options"]
    assert options["openmodalpy"]["window_type"] == "hamming"
    assert options["openmodalpy"]["window_norm"] == "amplitude"
    assert options["openmodalpy"]["nfft"] == 16
    assert options["openmodalpy"]["overlap"] == 0.0
    assert options["pyspod"]["n_dft"] == 16
    assert options["pyspod"]["overlap"] == 0
    assert options["pyspod"]["fullspectrum"] is False
    assert fixture_doc["mapping"]["formula"] == "lambda_openmodalpy = lambda_pyspod * nfft * dt / 2"


def test_mapped_tolerance_discriminates_convention_errors(fixture_doc) -> None:
    """The mapped bound must stay tighter than the two convention mistakes §4 measured.

    Dropping the Strouhal division moves the answer by 8x; power instead of
    amplitude scales it by 0.7337695. Either one must sit above 5e-3, or the
    bound stops earning its keep.
    """
    tol = _tol(fixture_doc, "mapped_vs_pyspod")
    nfft = float(fixture_doc["construction"]["nfft"])
    dt = float(fixture_doc["construction"]["dt"])
    correct = nfft * dt / 2.0
    no_strouhal = 2.0
    dropped_strouhal = abs(correct - no_strouhal) / correct
    power_factor = 0.54**2 / (0.54**2 + 0.5 * 0.46**2)
    power_shift = abs(1.0 - power_factor)
    assert dropped_strouhal > tol, (
        f"dropping Strouhal division moves the answer by {dropped_strouhal:.3e}, "
        f"which is not above mapped tol {tol:.3e}"
    )
    assert power_shift > tol, (
        f"power vs amplitude moves the answer by {power_shift:.3e}, which is not above mapped tol {tol:.3e}"
    )


def test_regen_script_names_pinned_pyspod_when_absent() -> None:
    hide = (
        f"import runpy, sys\nsys.modules['pyspod'] = None\nrunpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", hide],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    text = result.stdout + result.stderr
    assert "pyspod==2.0.0" in text
