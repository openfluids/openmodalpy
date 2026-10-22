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

``5e-3`` for every mapped comparison on the clean field (~2.5x the worst
measured residual).
It still discriminates: dropping the Strouhal division moves the answer by
8x, and the power normalisation by 0.734 — both orders above 5e-3.

``(nfft + nblocks) * eps`` (~5.3e-15) against the closed form. Amplitude-
normalised Hamming recovers that closed form exactly (ratio 1.000000);
this is the same FFT-plus-Gram round-off bound the oracle already uses.
Do not tighten the mapped bound past 5e-3: that is what the measured
window residual supports, not a pasted power of ten.

The noisy case, and why it exists
---------------------------------
On the clean field the closed form is known to (nfft + nblocks)*eps while the
PySPOD comparison is held at 5e-3 by the window difference. A 1e-3 error in the
eigenvalues reds the closed-form assertions and leaves the PySPOD one green, so
on that field the external number confirms the convention mapping and supplies
no evidence that nothing else has.

``noise_2e-1`` adds Gaussian noise at 0.2 of the field RMS, seeded in the
generator. There is no closed form for the SPOD estimate of a noisy field, so
the vendored PySPOD number is the only available truth and the comparison
carries the check rather than corroborating it. Both packages see the same
vendored snapshots, so they should still agree to about the window residual.

0.2 is not a round number picked for looking large. Measured shifts away from
the noiseless closed form, by occupied entry: 1.25e-2, 2.21e-2 and 4.43e-3.
It is the smallest level at which all three clear the ~2e-3 window residual,
which is what stops the closed form from being a substitute. Lower levels move
one entry or another by less than the two packages already disagree: at 0.1 the
bin 5 entry shifts only 6.8e-4, and at 0.001 every entry shifts about 5e-5.
The generator refuses to write a case that fails this.

The bound is ``1.6e-2``, 2.5x the worst measured residual of 6.36e-3 at
noise 0.2, the same margin the clean bound uses. It is looser than 5e-3
because the window difference and the noise interact; the same three residuals
are 1.09e-3, 9.0e-7 and 1.95e-3 with no noise. Do not paste a tighter number:
re-measure if you want one.

What this case does and does not buy
------------------------------------
It does not catch a different bug. Both cases run the same code, and the noisy
bound is looser than the clean one, so any error that moves both fields reds
the clean assertions first. Measured: a 0.5 percent scale error in the
spectral normalisation reds every clean assertion and passes the noisy one.

What it buys is that three numbers now have a check at all. The noisy
eigenvalues sit 1.25e-2, 2.21e-2 and 4.43e-3 away from the clean closed form,
so no analytic expression describes them, and this comparison is the only
thing asserting them. A change that altered SPOD only on broadband input would
pass every analytic test in the suite and would have to get past this one.

Already ruled out, do not chase: the noise-only sub-leading modes. On the
clean field they are machine zero (1e-15 at bin 3 modes 2 and 3), and under
noise they carry real energy (3.79e-2 and 1.48e-2), which makes them look like
the ideal load-bearing target. They are not. Measured against mapped PySPOD
they disagree by 2.4e-2, 3.9e-2, 1.7e-2, 2.4e-2 and 1.6e-1: the window
difference is coherent leakage, and on a low-energy mode the leakage is the
signal. A comparison there would need a 40 percent bound and would
discriminate nothing.

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


def _field_snapshots(fixture_doc: dict, case: str = "manufactured") -> np.ndarray:
    """Snapshots of one case taken FROM the fixture (not rebuilt)."""
    return np.asarray(fixture_doc["cases"][case]["snapshots"], dtype=np.float64)


def _openmodalpy_eigs(
    q: np.ndarray,
    fixture_doc: dict,
    tmp_path: Path,
    *,
    window_norm: str | None = None,
    characteristic_length: float | None = None,
) -> np.ndarray:
    """Build and run the analyzer from the fixture options.

    ``window_norm`` and ``characteristic_length`` default to the fixture
    values. Pass either to drive a wrong convention through the library.
    """
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
    if window_norm is None:
        window_norm = str(options["window_norm"])
    if characteristic_length is None:
        characteristic_length = options["characteristic_length"]
    analyzer = SPODAnalyzer(
        file_path="external_spod",
        nfft=int(options["nfft"]),
        overlap=float(options["overlap"]),
        window_type=str(options["window_type"]),
        window_norm=window_norm,
        blockwise_mean=bool(options["blockwise_mean"]),
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: field,
        spatial_weight_type="prescribed",
        # The vendored PySPOD run and the closed form both use identity (ones)
        # spatial weights; prescribe ones so the comparison stays about the
        # spectral-energy convention rather than coordinate-derived volumes.
        spatial_weights=np.ones((n_space, 1)),
        characteristic_length=characteristic_length,
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


def test_mapped_tolerance_discriminates_convention_errors(fixture_doc, tmp_path: Path) -> None:
    """Drive both wrong conventions through the library, not just through arithmetic.

    One mistake skips the Strouhal division; the other normalises the window
    by power instead of amplitude. Each must move the eigenvalue past the
    5e-3 mapped tolerance, or that bound stops catching them.
    """
    q = _field_snapshots(fixture_doc)
    construction = fixture_doc["construction"]
    bin_idx = int(construction["k_bin"])
    tol = _tol(fixture_doc, "mapped_vs_pyspod")

    correct = float(_openmodalpy_eigs(q, fixture_doc, tmp_path)[bin_idx, 0])

    # Setting L/U to nfft*dt makes the Strouhal step dst = 1.0, which is
    # exactly what skipping the division does.
    no_strouhal_length = float(construction["nfft"]) * float(construction["dt"])
    no_strouhal = float(
        _openmodalpy_eigs(q, fixture_doc, tmp_path, characteristic_length=no_strouhal_length)[bin_idx, 0]
    )
    err_no_strouhal = _rel_err(no_strouhal, correct)
    assert err_no_strouhal > tol, (
        f"dropping the Strouhal division moves the eigenvalue by {err_no_strouhal:.3e}, not above mapped tol {tol:.3e}"
    )

    power_norm = float(_openmodalpy_eigs(q, fixture_doc, tmp_path, window_norm="power")[bin_idx, 0])
    err_power = _rel_err(power_norm, correct)
    assert err_power > tol, (
        f"power instead of amplitude normalisation moves the eigenvalue by {err_power:.3e}, "
        f"not above mapped tol {tol:.3e}"
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


NOISY_CASE = "noise_2e-1"


def test_rebuilt_noisy_field_matches_vendored_numerically(regen, fixture_doc) -> None:
    """The seeded noise must rebuild exactly, or the vendored numbers are orphaned.

    The test never regenerates the field it measures; it reads the snapshots
    from the fixture. This check is what ties those snapshots to the generator,
    so a change to the noise seed or level cannot pass unnoticed.
    """
    vendored = _field_snapshots(fixture_doc, NOISY_CASE)
    level = float(fixture_doc["cases"][NOISY_CASE]["noise_relative_rms"])
    rebuilt = regen.manufactured_field(noise_relative_rms=level)
    assert np.allclose(rebuilt, vendored, rtol=1e-12, atol=0.0), (
        "rebuilt noisy field does not match the vendored snapshots; the seed or the noise level changed"
    )


def test_the_noisy_case_carries_more_noise_than_the_clean_one(fixture_doc) -> None:
    """The two cases must be different fields, at the recorded level."""
    clean = _field_snapshots(fixture_doc)
    noisy = _field_snapshots(fixture_doc, NOISY_CASE)
    level = float(fixture_doc["cases"][NOISY_CASE]["noise_relative_rms"])

    assert level > 0.0
    assert clean.shape == noisy.shape
    measured = float(np.sqrt(np.mean((noisy - clean) ** 2)) / np.sqrt(np.mean(clean**2)))
    assert measured == pytest.approx(level, rel=0.1)


@pytest.mark.parametrize(("bin_idx", "mode_idx"), OCCUPIED)
def test_openmodalpy_matches_vendored_pyspod_under_noise(
    fixture_doc,
    tmp_path: Path,
    bin_idx: int,
    mode_idx: int,
) -> None:
    """The load-bearing comparison: nothing else knows these numbers.

    The noisy SPOD estimate has no closed form, so this assertion is the only
    check on the values. A change to the block FFT, the window handling or the
    spectral normalisation that survives every analytic test would have to get
    past PySPOD here.
    """
    q = _field_snapshots(fixture_doc, NOISY_CASE)
    got = float(_openmodalpy_eigs(q, fixture_doc, tmp_path)[bin_idx, mode_idx])
    raw = float(_occupied_item(fixture_doc["cases"][NOISY_CASE]["occupied"], bin_idx, mode_idx)["pyspod_eigenvalue"])
    mapped = raw * _mapping_factor(fixture_doc)
    err = _rel_err(got, mapped)
    tol = _tol(fixture_doc, "mapped_vs_pyspod_noisy")
    assert err <= tol, (
        f"noisy bin {bin_idx} mode {mode_idx}: openmodalpy {got:.6g} vs mapped "
        f"PySPOD {mapped:.6g} relative error {err:.3e} exceeds {tol:.3e}"
    )


@pytest.mark.parametrize(("bin_idx", "mode_idx"), OCCUPIED)
def test_the_closed_form_does_not_describe_the_noisy_answer(
    fixture_doc,
    tmp_path: Path,
    bin_idx: int,
    mode_idx: int,
) -> None:
    """The noisy case must not be answerable from the clean closed form.

    This is what makes the vendored number load-bearing rather than
    corroborating. Every occupied entry has to sit further from the noiseless
    closed form than the two packages disagree with each other; otherwise the
    closed form would still describe the answer and PySPOD would add nothing.
    If someone lowers the noise, this fails and says so.
    """
    closed = float(_occupied_item(fixture_doc["construction"]["closed_form"], bin_idx, mode_idx)["value"])
    q = _field_snapshots(fixture_doc, NOISY_CASE)
    got = float(_openmodalpy_eigs(q, fixture_doc, tmp_path)[bin_idx, mode_idx])

    shift = _rel_err(got, closed)
    window_residual = _tol(fixture_doc, "mapped_vs_pyspod")
    assert shift > window_residual / 2.5, (
        f"noisy bin {bin_idx} mode {mode_idx} sits {shift:.3e} from the noiseless "
        f"closed form {closed:.6g}. The window residual already covers that, so "
        "this case corroborates instead of carrying the check"
    )
