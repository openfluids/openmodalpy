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

A third case, ``cylinder_wake``, is a field the package ships and documents:
500 snapshots over 5000 spatial points, against 40 over 12 for the
constructed system. One dataset can be tuned against; two of this different a
shape cannot both be. The generator states the shedding Strouhal number in
closed form, so this case carries a physical anchor as well as the external
one. The field is rebuilt on both sides rather than vendored, because 2.5
million float64 numbers do not belong in a JSON fixture; what pins it is the
generator parameters plus five reduced statistics. A checksum would be the
obvious choice and is the wrong one: ``np.sin`` and ``np.exp`` can differ by
one unit in the last place between platforms, so the bits are not portable
while the sums are, to far better than the bound they are compared at.

On the shipped case the comparison is over the PHYSICAL modes, the three
carrying the most amplitude: the mean and the shedding pair. The full sorted
set is not well posed here. The spectrum spans 1.0 down to 1.6e-3, and where
the truncation rank cuts into the noise floor both packages place spurious
modes, differently. Measured at rank 4 TLS, openmodalpy puts one at
``|lambda| = 3.61`` where PyDMD puts one at 1.0, which makes the sorted-set
error 3.6 while every physical mode still agrees to 1.6e-7. Those spurious
modes carry amplitude 1.1e-2 to 4.5e-2 against 57.8 for the mean and 3.35 for
the shedding pair, three to four orders below the physical content, and a
growth rate no bounded field can support. They are an artifact of rank
truncation, not a defect in either package.

Why these tolerances, shipped case
----------------------------------
Measured on the pinned stack, physical modes, matched to the nearest vendored
eigenvalue:

* rank 6 LS vs PyDMD: 2.8e-15 (1.4e-15 and 3.3e-15 at ranks 4 and 8)
* rank 6 TLS vs PyDMD: 1.9e-6 (1.6e-7 and 5.4e-6 at ranks 4 and 8)
* rank 6 TLS vs LS, same modes: 3.4e-5
* shedding frequency vs the generator's St: 6.3e-5 to 7.4e-5 over ranks
  4, 6 and 8 and both methods

``1e-12`` for LS, the same bound and reasoning as the constructed system.
``1e-5`` for TLS, inside the measured interval (1.9e-6, 3.4e-5): 5.2x above
the route residual and 3.4x below the TLS-LS split. That is narrower headroom
than the 30x and 600x on the constructed system, because the interval itself
is narrower here; the field is ill-conditioned and the two TLS formulations
diverge further as the rank grows. The residual is set by the algebra, not by
rounding, so a different BLAS cannot move it across the bound.
``3e-4`` for the Strouhal anchor, four times the worst measured residual. The
residual is the time discretisation of the field, not a solver error.

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


def _vendored_space_time(fixture_doc: dict, case: str) -> np.ndarray:
    """Space-time snapshots taken FROM the fixture (not rebuilt)."""
    return np.asarray(fixture_doc["cases"][case]["snapshots"], dtype=np.float64)


def _case_snapshots(fixture_doc: dict, case: str) -> np.ndarray:
    """Time-space snapshots taken FROM the fixture (not rebuilt)."""
    return _vendored_space_time(fixture_doc, case).T


@pytest.mark.parametrize("case", CASES)
def test_rebuilt_field_matches_vendored_numerically(regen, fixture_doc, case: str) -> None:
    vendored = _vendored_space_time(fixture_doc, case)
    rebuilt = regen.snapshots_space_time(noise_relative_rms=_case_noise(fixture_doc, case))
    if not np.allclose(rebuilt, vendored, rtol=1e-12, atol=0.0):
        raise AssertionError(
            f"{case}: rebuilt field does not match the vendored snapshots; "
            "the field changed and the vendored numbers no longer describe it"
        )


@pytest.mark.parametrize("case", CASES)
def test_loaded_field_is_bit_identical_to_generators(regen, fixture_doc, case: str) -> None:
    """Python's float repr round-trips float64, so vendoring the field is lossless."""
    generated = regen.snapshots_space_time(noise_relative_rms=_case_noise(fixture_doc, case))
    loaded = regen.snapshots_from_json(json.loads(json.dumps(regen.snapshots_to_json(generated))))
    assert regen.float64_bits_equal(loaded, generated)


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
def test_openmodalpy_matches_vendored_pydmd(fixture_doc, case: str, method: str) -> None:
    q = _case_snapshots(fixture_doc, case)
    got = _openmodalpy_eigs(q, method, fixture_doc)
    want = _case_pydmd(fixture_doc, case, method)
    key = "noisy_tls_vs_pydmd" if case != "noiseless" and method == "tls" else "roundoff_vs_pydmd"
    err = _eig_set_err(got, want)
    assert err <= _tol(fixture_doc, key), (
        f"{case}/{method}: openmodalpy vs vendored PyDMD set-error {err:.3e} exceeds {_tol(fixture_doc, key):.3e}"
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_openmodalpy_matches_chosen_eigenvalues(fixture_doc, case: str, method: str) -> None:
    q = _case_snapshots(fixture_doc, case)
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
    assert set(fixture_doc["cases"]) == {"noiseless", "noise_1e-3", "cylinder_wake"}
    for case in CASES:
        snapshots = _vendored_space_time(fixture_doc, case)
        assert snapshots.shape == (construction["n_space"], construction["n_snapshots"])
        assert snapshots.dtype == np.float64
    assert fixture_doc["cases"]["noise_1e-3"]["noise_relative_rms"] == 0.001
    options = fixture_doc["solver_options"]
    assert options["openmodalpy"]["methods"]["ls"]["method"] == "ls"
    assert options["openmodalpy"]["methods"]["tls"]["method"] == "tls"
    assert options["pydmd"]["methods"]["ls"]["svd_rank"] == 5
    assert options["pydmd"]["methods"]["tls"]["tlsq_rank"] == 5


def test_noisy_tls_tolerance_sits_inside_the_ls_tls_gap(fixture_doc) -> None:
    """The TLS-vs-PyDMD bound must stay tighter than the noisy LS/TLS split.

    Otherwise a silent TLS→LS fallback would still match the vendored TLS
    numbers at the stated tolerance, and the test would stop earning its keep.
    """
    q = _case_snapshots(fixture_doc, "noise_1e-3")
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


SHIPPED_CASE = "cylinder_wake"


def _shipped(fixture_doc: dict) -> dict:
    return fixture_doc["cases"][SHIPPED_CASE]


# The shipped field is 500 x 5000 and every test below needs it, so build it
# and each DMD run once per session rather than once per test.
_SHIPPED_CACHE: dict[str, object] = {}


def _shipped_data(fixture_doc: dict) -> dict:
    """The shipped dataset, built once."""
    if "data" not in _SHIPPED_CACHE:
        from openmodalpy import generate_cylinder_wake

        _SHIPPED_CACHE["data"] = generate_cylinder_wake(**_shipped(fixture_doc)["generator_params"])
    return _SHIPPED_CACHE["data"]


def _shipped_field(fixture_doc: dict) -> tuple[np.ndarray, float]:
    """Rebuild the shipped field from the recorded generator parameters."""
    data = _shipped_data(fixture_doc)
    return np.asarray(data["q"], dtype=np.float64), float(data["dt"])


def _physical_modes(eigenvalues: np.ndarray, amplitudes: np.ndarray, count: int = 3) -> np.ndarray:
    """The ``count`` eigenvalues carrying the most amplitude.

    On this field those are the mean and the shedding pair. The rest sit at
    the noise floor, where rank truncation places them differently in the two
    packages, so they are not a comparable quantity.
    """
    return eigenvalues[np.argsort(-amplitudes)[:count]]


def _shipped_openmodalpy(fixture_doc: dict, method: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Run openmodalpy DMD on the shipped field at the recorded rank, once."""
    key = f"run:{method}"
    if key not in _SHIPPED_CACHE:
        from openmodalpy import DMDAnalyzer

        case = _shipped(fixture_doc)
        data = _shipped_data(fixture_doc)
        rank = int(case["rank"])
        analyzer = DMDAnalyzer(data=data, n_modes_save=rank, rank=rank)
        analyzer.load_and_preprocess()
        analyzer.perform_dmd(method=method)
        _SHIPPED_CACHE[key] = (
            np.asarray(analyzer.eigenvalues, dtype=np.complex128),
            np.asarray(analyzer.amplitudes, dtype=np.float64),
            float(data["dt"]),
        )
    return _SHIPPED_CACHE[key]


def test_the_shipped_field_still_matches_the_vendored_statistics(fixture_doc) -> None:
    """The generator must still build the field the vendored numbers describe.

    Five reduced statistics stand in for the 2.5 million numbers, which do not
    belong in a fixture. The index weight makes them sensitive to a reordering
    as well as to a change of value.
    """
    case = _shipped(fixture_doc)
    q, dt = _shipped_field(fixture_doc)

    assert list(q.shape) == case["shape"]
    assert dt == pytest.approx(float(case["dt"]), rel=1e-12)

    index = np.arange(q.size, dtype=np.float64).reshape(q.shape)
    measured = {
        "sum": float(q.sum()),
        "sum_of_squares": float((q**2).sum()),
        "min": float(q.min()),
        "max": float(q.max()),
        "index_weighted_sum": float((q * index).sum()),
    }
    rtol = float(case["field_rtol"])
    for name, want in case["field_statistics"].items():
        assert measured[name] == pytest.approx(float(want), rel=rtol), (
            f"shipped field statistic {name} is {measured[name]!r} against the vendored "
            f"{want!r}; the generator changed and the vendored PyDMD numbers no longer "
            "describe this field"
        )


@pytest.mark.parametrize("method", ("ls", "tls"))
def test_openmodalpy_matches_vendored_pydmd_on_the_shipped_field(fixture_doc, method: str) -> None:
    """The second dataset. A field the package ships, checked against PyDMD.

    The comparison is over the physical modes, matched to the nearest vendored
    eigenvalue. See the module docstring for why the full sorted set is not a
    well-posed quantity on this spectrum.
    """
    case = _shipped(fixture_doc)
    got, amplitudes, _ = _shipped_openmodalpy(fixture_doc, method)
    vendored = _as_eigs(case["methods"][method]["pydmd_eigenvalues"])

    worst = max(float(np.min(np.abs(vendored - lam))) for lam in _physical_modes(got, amplitudes))
    tol = _tol(fixture_doc, f"shipped_{method}_vs_pydmd")
    assert worst <= tol, (
        f"shipped case {method}: worst physical-mode distance to a vendored PyDMD "
        f"eigenvalue is {worst:.3e}, above {tol:.3e}"
    )


@pytest.mark.parametrize("method", ("ls", "tls"))
def test_the_shipped_case_recovers_the_documented_shedding_frequency(fixture_doc, method: str) -> None:
    """The physical anchor: the generator states St, and DMD must find it.

    This is what the second dataset adds beyond another external number. The
    constructed system has an invented spectrum; this one has a quantity the
    package documents and a user would check.
    """
    case = _shipped(fixture_doc)
    got, amplitudes, dt = _shipped_openmodalpy(fixture_doc, method)
    physical = _physical_modes(got, amplitudes)

    oscillating = physical[np.abs(np.angle(physical)) > 1e-9]
    assert oscillating.size > 0, "no oscillating mode among the physical modes"
    leading = oscillating[np.argmax(np.abs(oscillating))]
    measured = float(abs(np.angle(leading)) / (2.0 * np.pi * dt))

    strouhal = float(case["strouhal"])
    err = abs(measured - strouhal) / strouhal
    tol = _tol(fixture_doc, "shipped_vs_strouhal")
    assert err <= tol, (
        f"shipped case {method}: shedding frequency {measured:.8f} against the "
        f"generator's St {strouhal:.8f}, relative {err:.3e}, above {tol:.3e}"
    )


def test_the_shipped_tls_bound_sits_inside_the_measured_gap(fixture_doc) -> None:
    """The TLS bound must discriminate, not just pass.

    It has to sit above the gap between the two TLS routes and below the gap
    between TLS and LS. If it drifted outside that interval it would either
    fail on a correct run or stop noticing a TLS that had degraded to LS.
    """
    ls, ls_amp, _ = _shipped_openmodalpy(fixture_doc, "ls")
    tls, tls_amp, _ = _shipped_openmodalpy(fixture_doc, "tls")
    physical_ls = _physical_modes(ls, ls_amp)
    physical_tls = _physical_modes(tls, tls_amp)

    split = max(float(np.min(np.abs(physical_ls - lam))) for lam in physical_tls)
    tol = _tol(fixture_doc, "shipped_tls_vs_pydmd")
    assert tol < split, (
        f"the TLS bound {tol:.3e} is not below the TLS-LS split {split:.3e}; a TLS "
        "that silently degraded to LS would pass"
    )


def test_the_shipped_case_is_a_different_dataset(fixture_doc) -> None:
    """Two datasets, or the check can be satisfied by tuning against one.

    The point of the second case is that it is not the first one rescaled: a
    different generator, a spatial dimension three orders larger, and more
    snapshots.
    """
    shipped = _shipped(fixture_doc)
    constructed = np.asarray(fixture_doc["cases"]["noiseless"]["snapshots"], dtype=np.float64)

    n_snapshots, n_space = shipped["shape"]
    assert n_space > 100 * constructed.shape[0]
    assert n_snapshots > constructed.shape[1]
    assert shipped["generator"] == SHIPPED_CASE
