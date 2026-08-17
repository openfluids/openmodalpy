"""Recompute analytic generator spectra and compare to committed fixtures.

Each fixture under ``tests/fixtures/reference/`` records POD energy fractions
and DMD |λ|/phase on the packaged example grids, with ``rtol``/``atol`` stored
in the file. Tolerances are never hard-coded in this test.

Shared spectrum definition: ``tests/reference_helpers.py`` (also used by regen).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openmodalpy.config_io import load_jsonc
from openmodalpy.example_data import GENERATORS
from tests.reference_helpers import (
    canonicalize_dmd_eigenvalues,
    compute_reference_spectra,
)

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "reference"
# Packaged configs ship in the wheel; they are the source of truth for grids.
PACKAGED_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "src" / "openmodalpy" / "examples"
# This directory also holds vendored external-reference files that are not
# analytic generator spectra (external_dmd.json). Those stems are not generators.
_NON_GENERATOR_FIXTURE_STEMS = frozenset({"external_dmd"})

# Non-config extras that fixtures record. Grids and seed come from the
# packaged config (cylinder_wake states seed 42 there).
_FIXTURE_EXTRAS: dict[str, dict] = {
    "cylinder_wake": {},
}


def _fixture_paths() -> list[Path]:
    paths = sorted(p for p in FIX_DIR.glob("*.json") if p.stem not in _NON_GENERATOR_FIXTURE_STEMS)
    if not paths:
        raise FileNotFoundError(f"No reference fixtures in {FIX_DIR}")
    return paths


def _phase_close(got: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> None:
    """Compare phases on the circle (wrap difference into [-π, π])."""
    assert got.shape == expected.shape, f"phase shape {got.shape} != {expected.shape}"
    delta = np.angle(np.exp(1j * (got - expected)))
    # Bound |δ| like allclose on the wrapped residual against zero.
    limit = atol + rtol * np.maximum(np.abs(expected), 1.0)
    bad = np.abs(delta) > limit
    if np.any(bad):
        idx = int(np.argmax(np.abs(delta)))
        raise AssertionError(
            f"phase mismatch at index {idx}: got={got[idx]!r}, expected={expected[idx]!r}, "
            f"|wrap(δ)|={abs(delta[idx]):.3e}, limit={limit[idx]:.3e} (rtol={rtol}, atol={atol})"
        )


def test_canonical_order_invariant_under_tied_pair_permutation():
    """Hand-built tied |λ| pair: the independent oracle must not depend on emission order.

    Bare ``argsort(|λ|)[::-1]`` preserves LAPACK's conjugate order, so the two
    emissions below differ by a phase swap. The test-side oracle (not an import
    of the library helper) must make them agree.
    """
    rtol = 1e-6
    # Same physics, opposite conjugate emission order (goal demo).
    a = np.array([1.0 + 0.0j, 0.7 + 0.7j, 0.7 - 0.7j])
    b = np.array([1.0 + 0.0j, 0.7 - 0.7j, 0.7 + 0.7j])

    bare_a = a[np.argsort(np.abs(a))[::-1]]
    bare_b = b[np.argsort(np.abs(b))[::-1]]
    # Precondition: bare magnitude sort is NOT order-invariant (would be RED
    # if the fixture recorded that order without canonicalization).
    assert not np.allclose(np.angle(bare_a), np.angle(bare_b), atol=1e-12), (
        "precondition failed: bare argsort[::-1] unexpectedly matched for tied |λ|"
    )

    ca = canonicalize_dmd_eigenvalues(a, rtol=rtol)
    cb = canonicalize_dmd_eigenvalues(b, rtol=rtol)
    np.testing.assert_allclose(np.abs(ca), np.abs(cb), rtol=rtol, atol=1e-12)
    np.testing.assert_allclose(np.angle(ca), np.angle(cb), rtol=rtol, atol=1e-12)
    # Magnitudes descending; within each tied group, (Re, Im) ascending.
    mags = np.abs(ca)
    assert np.all(mags[:-1] >= mags[1:] - 1e-15)
    # A conjugate pair shares its real part, so (Re, Im) puts the negative
    # imaginary part first — which is also what the old phase-ascending rule
    # gave. That coincidence is why the committed fixtures did not move.
    assert ca[1].imag <= ca[2].imag


def test_canonical_order_band_pinned_at_one_e_minus_12():
    """Oracle band is the literal 1e-12: just inside reorders, just outside does not."""
    rtol = 1e-12
    inside = np.array([1.0 + 0.0j, -(1.0 - 0.5e-12) + 0.0j])
    outside = np.array([1.0 + 0.0j, -(1.0 - 2.0e-12) + 0.0j])

    got_in = canonicalize_dmd_eigenvalues(inside, rtol=rtol)
    assert got_in[0].real < got_in[1].real

    got_out = canonicalize_dmd_eigenvalues(outside, rtol=rtol)
    assert got_out[0].real > 0.0
    assert got_out[0].real > got_out[1].real

    # Hand-written expected vector against the oracle itself (scrambled input).
    raw = np.array([0.0 + 0.8j, 0.95 + 0.0j, 0.0 - 0.8j])
    expected = np.array([0.95 + 0.0j, 0.0 - 0.8j, 0.0 + 0.8j])
    np.testing.assert_array_equal(canonicalize_dmd_eigenvalues(raw, rtol=rtol), expected)


def test_canonical_order_no_magnitude_chain_merge():
    """Near-neighbour |λ| chain must not form one group and reverse magnitude order.

    Five magnitudes each 0.6*rtol below the previous: consecutive pairs agree
    within rtol, but the ends differ by 2.4*rtol. Phases are laid so that
    phase-ascending order is magnitude-ascending — the worst case for a
    consecutive-neighbour merge. After canonicalization, magnitudes must stay
    descending (within the rtol band) and the order must not depend on input
    permutation.
    """
    rtol = 1e-6
    mags = np.array([1.0 - k * 0.6e-6 for k in range(5)])
    phases = np.array([(2 - k) * 0.5 for k in range(5)])
    eig = mags * np.exp(1j * phases)

    # Precondition: consecutive-neighbour grouping + phase sort would reverse
    # magnitude order (would be RED under the old helper rules).
    sorted_by_mag = eig[np.argsort(np.abs(eig))[::-1]]
    phase_sorted = sorted_by_mag[np.argsort(np.angle(sorted_by_mag))]
    old_mags = np.abs(phase_sorted)
    assert old_mags[-1] - old_mags[0] > rtol * max(old_mags[0], old_mags[-1]), (
        "precondition failed: phase-sorting the full chain did not reverse magnitude order"
    )

    ref = canonicalize_dmd_eigenvalues(eig, rtol=rtol)
    m = np.abs(ref)
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            assert m[j] - m[i] <= rtol * max(m[i], m[j]), (
                f"index {j} (|λ|={m[j]:.12f}) larger than earlier index {i} (|λ|={m[i]:.12f}) by more than rtol"
            )

    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(len(eig))
        got = canonicalize_dmd_eigenvalues(eig[perm], rtol=rtol)
        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-15)


def test_canonical_order_continuous_across_negative_real_axis():
    """Eigenvalues a ULP apart across the negative real axis stay adjacent.

    ``np.angle`` returns +π for a real-negative eigenvalue and −π when the
    imaginary part is 1 ULP negative, so those two sort to opposite ends of a
    tied group. The within-group key must keep them next to each other, and
    the order must not depend on input permutation.
    """
    rtol = 1e-6
    z_a = complex(-1.0, 1e-16)
    z_b = complex(-1.0, -1e-16)
    z_c = complex(np.cos(0.5), np.sin(0.5))
    z_d = np.conj(z_c)
    eig = np.array([z_a, z_c, z_d, z_b])

    # Precondition: phase-ascending on the full |λ|-tie group splits z_a/z_b.
    phase_order = eig[np.argsort(np.angle(eig))]
    ia_old = int(np.argmin(np.abs(phase_order - z_a)))
    ib_old = int(np.argmin(np.abs(phase_order - z_b)))
    assert abs(ia_old - ib_old) != 1, "precondition failed: np.angle order unexpectedly kept the pair adjacent"

    ref = canonicalize_dmd_eigenvalues(eig, rtol=rtol)

    def locate(z: complex) -> int:
        return int(np.argmin(np.abs(ref - z)))

    ia, ib = locate(z_a), locate(z_b)
    assert abs(ia - ib) == 1, f"the two eigenvalues 2e-16 apart landed at indices {ia} and {ib}; they must be adjacent"

    rng = np.random.default_rng(1)
    for _ in range(20):
        perm = rng.permutation(len(eig))
        got = canonicalize_dmd_eigenvalues(eig[perm], rtol=rtol)
        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_reference_fixture_matches_recompute(path: Path):
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    rtol = float(doc["rtol"])
    atol = float(doc["atol"])
    n_modes = int(doc["n_modes"])
    name = doc["generator"]
    assert name == path.stem, f"{path.name}: generator field {name!r} != stem"

    spectra = compute_reference_spectra(name, doc["generator_params"], n_modes=n_modes, rtol=rtol)

    pod_exp = np.asarray(doc["pod_energy_fractions"], dtype=np.float64)
    abs_exp = np.asarray(doc["dmd_abs_lambda"], dtype=np.float64)
    phase_exp = np.asarray(doc["dmd_phase"], dtype=np.float64)
    ecf_exp = float(doc["energy_captured_fraction"])

    np.testing.assert_allclose(
        spectra["energy_captured_fraction"],
        ecf_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: energy_captured_fraction",
    )
    np.testing.assert_allclose(
        spectra["pod_energy_fractions"],
        pod_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: POD energy fractions",
    )
    # Kept fractions sum to the captured fraction, not to 1.
    np.testing.assert_allclose(
        float(np.sum(spectra["pod_energy_fractions"])),
        spectra["energy_captured_fraction"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: sum(pod fractions) vs energy_captured_fraction",
    )
    np.testing.assert_allclose(
        spectra["dmd_abs_lambda"],
        abs_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: DMD |λ|",
    )
    _phase_close(spectra["dmd_phase"], phase_exp, rtol=rtol, atol=atol)


def test_taylor_green_dmd_abs_lambda_matches_closed_form_dmd_eigenvalue():
    """Taylor–Green |λ| is anchored to the generator's closed-form metadata.

    Independent of the golden file: metadata["dmd_eigenvalue"] = exp(-2 ν Δt).
    """
    path = FIX_DIR / "taylor_green.json"
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    rtol = float(doc["rtol"])
    atol = float(doc["atol"])
    spectra = compute_reference_spectra(
        "taylor_green",
        doc["generator_params"],
        n_modes=int(doc["n_modes"]),
        rtol=rtol,
    )
    closed = float(spectra["payload"]["metadata"]["dmd_eigenvalue"])
    got = float(spectra["dmd_abs_lambda"][0])
    fixture_val = float(doc["dmd_abs_lambda"][0])

    np.testing.assert_allclose(
        got,
        closed,
        rtol=rtol,
        atol=atol,
        err_msg="recomputed dmd_abs_lambda[0] vs metadata dmd_eigenvalue",
    )
    np.testing.assert_allclose(
        fixture_val,
        closed,
        rtol=rtol,
        atol=atol,
        err_msg="fixture dmd_abs_lambda[0] vs metadata dmd_eigenvalue",
    )


def _expected_generator_params(generator: str) -> dict:
    """Full generation contract: packaged data.params plus fixture extras (seed)."""
    path = PACKAGED_EXAMPLES_DIR / f"{generator}.jsonc"
    if not path.is_file():
        raise FileNotFoundError(f"no packaged config at {path}")
    doc = load_jsonc(path)
    case = doc.get("case", doc)
    params = dict(case.get("data", {}).get("params", {}))
    params.update(_FIXTURE_EXTRAS.get(generator, {}))
    return params


def test_fixture_set_equals_generator_set():
    """Every built-in generator must have a fixture; deleting one must fail."""
    fixtures = {p.stem for p in FIX_DIR.glob("*.json")}
    extras = fixtures - set(GENERATORS) - _NON_GENERATOR_FIXTURE_STEMS
    assert not extras, f"unexpected reference files: {sorted(extras)}"
    generator_fixtures = fixtures - _NON_GENERATOR_FIXTURE_STEMS
    assert generator_fixtures == set(GENERATORS), (
        f"fixture set {sorted(generator_fixtures)} != generator set {sorted(GENERATORS)}"
    )


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_fixture_generator_params_match_packaged_example_config(path: Path):
    """Fixture generation contract must match packaged configs (+ seed extras).

    Pins how the DATA is generated — Nx/Ny/Nt and seed — so the fixture and the
    shipped example cannot describe different fields. If either side moves
    alone, this test fails. Rank agreement (config rank vs fixture ``n_modes``)
    is pinned separately in ``tests/test_example_configs.py``.
    """
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    name = doc["generator"]
    assert name == path.stem, f"{path.name}: generator field {name!r} != stem"

    got = doc["generator_params"]
    want = _expected_generator_params(name)
    assert got == want, (
        f"{path.name}: fixture generator_params {got} does not match "
        f"packaged generation contract {want} under "
        f"src/openmodalpy/examples/{name}.jsonc"
    )
