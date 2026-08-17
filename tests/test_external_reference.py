"""Cross-check openmodalpy DMD against vendored PyDMD eigenvalues.

This module never imports PyDMD. The numbers live in
``tests/fixtures/reference/external_dmd.json``, generated once outside the
repo by ``scripts/regen_external_reference.py`` against **pydmd==2025.8.1**
(Python 3.12, NumPy 2.5.2, SciPy 1.18.0). The package has no PyDMD
dependency.

Both sides are built from five chosen eigenvalues (two conjugate pairs plus
one real, spectral radius 0.95) observed in 12-space over 40 snapshots, so
the noiseless spectrum is known by construction. Two fields ship: noiseless,
and additive Gaussian noise at 1e-3 of the field rms. Each is compared at
matched truncation rank 5 for ``method="ls"`` and ``method="tls"``, as
sorted sets (``tests.test_dmd._eig_set_err``).

Why these tolerances
--------------------
Measured on the prescribed stack (Python 3.12, pydmd 2025.8.1, NumPy 2.5.2,
SciPy 1.18.0), then confirmed on this seeded field:

* noiseless, openmodalpy LS vs ``DMD(svd_rank=5)``: 1.4e-15 (here 7e-16)
* noiseless, openmodalpy TLS vs ``DMD(svd_rank=5, tlsq_rank=5)``: 1.7e-15
  (here 6e-16)
* noiseless, either side vs the chosen eigenvalues: ~1e-15
* noise 1e-3 rms, LS vs PyDMD: 1.6e-15 (here 1.9e-15)
* noise 1e-3 rms, TLS vs PyDMD: 3.1e-10 (here 1.5e-10)
* noise 1e-3 rms, TLS vs LS (either package): 6.2e-6 (here 4.5e-6)
* noise 1e-3 rms, either estimate vs the chosen eigenvalues: 8.3e-5

``1e-12`` for every comparison that is algebraically the same operator
(noiseless LS/TLS vs PyDMD and vs the constructed spectrum; noisy LS vs
PyDMD). The generation script refuses to write if noiseless PyDMD misses
the constructed spectrum by more than this, so the bound is the write-gate
itself: three orders above the measured 1e-15 residuals (BLAS headroom)
and six orders below the noisy TLS–LS split. A wrong operator cannot hide
on the noisy rows — that is the load-bearing check. On the noiseless rows
a TLS→LS swap does hide: noiseless LS vs vendored TLS measures 1.024e-15
and stays under 1e-12.

``1e-8`` for noisy TLS vs PyDMD. The two TLS routes are the same estimator
written differently — openmodalpy splits the *left* singular vectors of
stacked ``[X1; X2]``; PyDMD projects both snapshot matrices onto the
leading *right* singular vectors — and that algebraic gap is the 3.1e-10,
not a bug. ``1e-8`` is the first power of ten that sits inside the
measured gap ``(3.1e-10, 6.2e-6)``: ~30× above the TLS-route residual,
~600× below the TLS–LS split. If TLS ever silently degrades to LS the
noisy TLS comparison goes red.

``3e-4`` for noisy estimates vs the chosen eigenvalues. There is no closed
form for the noisy estimator; the 8.3e-5 is the noise-induced bias on this
field. Four times that residual is still three times smaller than the
injected noise amplitude (1e-3). This bound is a loose sanity floor, not a
discriminating check: a method that only recovers the spectrum to the
noise floor can still sit under it. The vendored PyDMD comparison is the
check that catches a wrong operator.
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

from tests.test_dmd import _eig_set_err, _make_analyzer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "reference" / "external_dmd.json"
SCRIPT_PATH = ROOT / "scripts" / "regen_external_reference.py"

CASES = ("noiseless", "noise_1e-3")
METHODS = ("ls", "tls")
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load_regen():
    spec = importlib.util.spec_from_file_location("regen_external_reference", SCRIPT_PATH)
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


def _as_eigs(pairs) -> np.ndarray:
    return np.asarray([complex(re, im) for re, im in pairs], dtype=np.complex128)


@pytest.fixture(scope="module")
def regen():
    return _load_regen()


@pytest.fixture(scope="module")
def fixture_doc():
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _openmodalpy_eigs(q: np.ndarray, method: str, fixture_doc: dict) -> np.ndarray:
    options = fixture_doc["solver_options"]["openmodalpy"]
    analyzer = _make_analyzer(
        q,
        n_modes_save=int(options["n_modes_save"]),
        rank=int(options["rank"]),
    )
    analyzer.perform_dmd(method=str(options["methods"][method]["method"]))
    return np.asarray(analyzer.eigenvalues)


def _case_noise(doc: dict, case: str) -> float:
    return float(doc["cases"][case]["noise_relative_rms"])


def _case_pydmd(doc: dict, case: str, method: str) -> np.ndarray:
    return _as_eigs(doc["cases"][case]["methods"][method]["pydmd_eigenvalues"])


def _tol(doc: dict, key: str) -> float:
    return float(doc["tolerances"][key]["value"])


def _case_snapshots(regen, fixture_doc: dict, case: str) -> np.ndarray:
    """Time-space snapshots, after asserting the vendored field fingerprint."""
    snapshots = regen.snapshots_space_time(noise_relative_rms=_case_noise(fixture_doc, case))
    got = regen.snapshot_sha256(snapshots)
    want = fixture_doc["construction"]["field_sha256"][case]
    assert got == want, (
        f"{case}: snapshot field sha256 {got} != fixture {want}; "
        "the field changed and the vendored numbers no longer describe it"
    )
    return snapshots.T


@pytest.mark.parametrize("case", CASES)
def test_rebuilt_field_matches_fixture_fingerprint(regen, fixture_doc, case: str) -> None:
    _case_snapshots(regen, fixture_doc, case)


def test_live_script_construction_matches_fixture(regen, fixture_doc) -> None:
    construction = fixture_doc["construction"]
    assert regen.FIELD_SEED == construction["field_seed"]
    assert regen.NOISE_SEED == construction["noise_seed"]
    assert regen.N_SPACE == construction["n_space"]
    assert regen.N_SNAPSHOTS == construction["n_snapshots"]
    assert regen.RANK == construction["rank"]
    assert list(regen.CHOSEN_POLAR) == [(item["modulus"], item["angle"]) for item in construction["chosen_polar"]]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_openmodalpy_matches_vendored_pydmd(regen, fixture_doc, case: str, method: str) -> None:
    q = _case_snapshots(regen, fixture_doc, case)
    got = _openmodalpy_eigs(q, method, fixture_doc)
    want = _case_pydmd(fixture_doc, case, method)
    key = "noisy_tls_vs_pydmd" if case != "noiseless" and method == "tls" else "roundoff_vs_pydmd"
    err = _eig_set_err(got, want)
    assert err <= _tol(fixture_doc, key), (
        f"{case}/{method}: openmodalpy vs vendored PyDMD set-error {err:.3e} exceeds {_tol(fixture_doc, key):.3e}"
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_openmodalpy_matches_chosen_eigenvalues(regen, fixture_doc, case: str, method: str) -> None:
    q = _case_snapshots(regen, fixture_doc, case)
    got = _openmodalpy_eigs(q, method, fixture_doc)
    want = _as_eigs(fixture_doc["construction"]["chosen_eigenvalues"])
    key = "roundoff_vs_chosen" if case == "noiseless" else "noisy_vs_chosen"
    err = _eig_set_err(got, want)
    assert err <= _tol(fixture_doc, key), (
        f"{case}/{method}: openmodalpy vs chosen eigenvalues set-error {err:.3e} exceeds {_tol(fixture_doc, key):.3e}"
    )


def test_fixture_provenance_records_both_solvers_and_the_pinned_pydmd(fixture_doc) -> None:
    prov = fixture_doc["provenance"]
    assert prov["pydmd_version"] == "2025.8.1"
    assert prov["python_version"].startswith("3.12")
    assert prov["numpy_version"]
    assert prov["scipy_version"]
    assert prov["generation_date"]
    construction = fixture_doc["construction"]
    assert construction["field_seed"] is not None
    assert construction["noise_seed"] is not None
    assert construction["n_space"] == 12
    assert construction["n_snapshots"] == 40
    assert construction["rank"] == 5
    assert set(construction["field_sha256"]) == {"noiseless", "noise_1e-3"}
    assert set(fixture_doc["cases"]) == {"noiseless", "noise_1e-3"}
    assert fixture_doc["cases"]["noise_1e-3"]["noise_relative_rms"] == 0.001
    options = fixture_doc["solver_options"]
    assert options["openmodalpy"]["methods"]["ls"]["method"] == "ls"
    assert options["openmodalpy"]["methods"]["tls"]["method"] == "tls"
    assert options["pydmd"]["methods"]["ls"]["svd_rank"] == 5
    assert options["pydmd"]["methods"]["tls"]["tlsq_rank"] == 5


def test_noisy_tls_tolerance_sits_inside_the_ls_tls_gap(regen, fixture_doc) -> None:
    """The TLS-vs-PyDMD bound must stay tighter than the noisy LS/TLS split.

    Otherwise a silent TLS→LS fallback would still match the vendored TLS
    numbers at the stated tolerance, and the test would stop earning its keep.
    """
    q = _case_snapshots(regen, fixture_doc, "noise_1e-3")
    ls = _openmodalpy_eigs(q, "ls", fixture_doc)
    tls = _openmodalpy_eigs(q, "tls", fixture_doc)
    split = _eig_set_err(tls, ls)
    tls_tol = _tol(fixture_doc, "noisy_tls_vs_pydmd")
    assert split > tls_tol, (
        f"noisy TLS vs LS split {split:.3e} is not above TLS-vs-PyDMD tol {tls_tol:.3e}; "
        "the bound can no longer catch a TLS→LS fallback"
    )


def test_regen_script_names_pinned_pydmd_when_absent() -> None:
    hide = (
        f"import runpy, sys\nsys.modules['pydmd'] = None\nrunpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')\n"
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
    assert "pydmd==2025.8.1" in text
