"""Shared helpers for analytic reference fixtures (regen + comparison tests).

Canonicalization of the DMD spectrum and POD energy normalisation live here
so the committed JSON and the comparison test cannot drift apart.
"""

from __future__ import annotations

import tempfile
import warnings
from typing import Any

import numpy as np

from openmodalpy import DMDAnalyzer, PODAnalyzer
from openmodalpy.example_data import generate_example_dataset

# Taylor–Green rank-1 fields emit this; ignore only that message.
_DMD_EFFECTIVE_RANK_WARNING = r"DMD effective rank .* is below the requested"


def analyzer_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Hand a generator payload to an analyzer under the loader contract.

    Shared by reference comparison tests and the shipped ground-truth suite.
    """
    return {
        "q": payload["q"],
        "x": payload["x"],
        "y": payload["y"],
        "z": payload["z"],
        "dt": payload["dt"],
        "Nx": payload["Nx"],
        "Ny": payload["Ny"],
        "Nz": payload["Nz"],
        "Ns": payload["Ns"],
        "metadata": payload.get("metadata", {}),
    }


def make_loader(payload: dict[str, Any]):
    data = analyzer_data(payload)
    return lambda _path: data


def _mags_agree(a: float, b: float, rtol: float) -> bool:
    """Relative magnitude agreement (no fixed-digit rounding)."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return True
    return abs(a - b) <= rtol * scale


def reference_pivot_index(col) -> int:
    """Lowest index whose magnitude is within ``1e-12`` of max |col|.

    Test-side copy of the library pivot rule. Empty or all-zero columns return 0.
    Kept here (not imported from the library) for the same reason as
    ``canonicalize_reference``: oracle and SPOD checks must pin WHICH pivot the
    suite intends, so a library change turns them red instead of tracking along.
    Enforced by ``test_oracle_tests_do_not_import_the_library_sign_rule``.
    """
    mag = np.abs(np.asarray(col))
    if mag.size == 0:
        return 0
    m = float(mag.max())
    if m == 0.0:
        return 0
    return int(np.argmax(mag >= (1.0 - 1e-12) * m))


def canonicalize_reference(modes, coeffs=None):
    """Sign/phase-canonicalize mode columns for independent oracle comparison.

    Deliberately duplicates the library sign/phase rule in openmodalpy.core.base
    rather than importing that helper. Oracle tests must pin WHICH convention is
    intended: if the reference side called the library helper, both sides would
    track the same factor under any rule change and the comparison would stay
    green. This copy is the fixed expression of that rule on the reference side.
    Re-coupling is guarded by ``test_oracle_tests_do_not_import_the_library_sign_rule``.

    For each column the pivot is the lowest index whose magnitude is within a
    relative band ``1e-12`` of the column maximum; the scale is
    ``conj(pivot_value)/|pivot_value|`` so that entry becomes real and positive.
    Modes and coeffs both receive the factor. All-zero columns are left alone.
    """
    modes = np.asarray(modes)
    if coeffs is not None:
        coeffs = np.asarray(coeffs)

    if modes.size == 0 or modes.ndim < 2 or modes.shape[1] == 0:
        return modes, coeffs

    modes = modes.copy()
    if coeffs is not None:
        coeffs = coeffs.copy()

    for k in range(modes.shape[1]):
        col = modes[:, k]
        i = reference_pivot_index(col)
        v = col[i]
        # All-zero columns: pivot helper returns 0 and |v| is 0 — leave alone.
        if np.abs(v) == 0.0:
            continue
        s = np.conj(v) / np.abs(v)
        modes[:, k] *= s
        if coeffs is not None:
            coeffs[:, k] *= s

    return modes, coeffs


def canonicalize_dmd_eigenvalues(eigvals: np.ndarray, rtol: float) -> np.ndarray:
    """Stable DMD spectrum order for reference fixtures.

    Magnitudes stay descending (same primary key as the analyzer). After that
    sort, a group is every run of eigenvalues whose ``|λ|`` agrees with the
    group's first (largest) member within ``rtol`` — not merely with the
    previous neighbour. That bound keeps any two members of a group within
    ``rtol`` of each other relative to the larger of their magnitudes, and
    stops a chain of near-neighbours from merging ends that differ by many
    times ``rtol``.

    Within each group the order is lexicographic ``(Re, Im)`` ascending. That
    key is continuous across the negative real axis, so eigenvalues a ULP
    apart do not jump to opposite ends of the group the way ``np.angle``
    does at ``±π``. LAPACK conjugate-pair emission order is removed from the
    recorded spectrum without changing analyzer code.
    """
    eigvals = np.asarray(eigvals, dtype=np.complex128).reshape(-1)
    if eigvals.size == 0:
        return eigvals.copy()

    mag = np.abs(eigvals)
    # Primary order: |λ| descending (stable for unequal magnitudes).
    order = np.argsort(mag)[::-1]
    sorted_eigs = eigvals[order]
    sorted_mag = mag[order]

    out = np.empty_like(sorted_eigs)
    i = 0
    n = sorted_eigs.size
    while i < n:
        j = i + 1
        # Anchor: compare each candidate to the group's first member, not
        # to its immediate predecessor (avoids single-linkage chaining).
        while j < n and _mags_agree(sorted_mag[i], sorted_mag[j], rtol):
            j += 1
        group = sorted_eigs[i:j]
        # (Re, Im) ascending — continuous across the negative real axis.
        group = group[np.lexsort((group.imag, group.real))]
        out[i:j] = group
        i = j
    return out


def pod_fractions_over_pretruncation_total(
    eigenvalues: np.ndarray,
    energy_captured_fraction: float,
) -> np.ndarray:
    """Normalise kept POD eigenvalues by the pre-truncation energy total.

    ``pod.eigenvalues`` is already truncated to ``n_modes_save``. Recover the
    full sum via ``total = sum(kept) / energy_captured_fraction`` (set before
    truncation on the analyzer). Fractions then sum to the captured fraction,
    not to 1.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    kept = float(np.sum(lam))
    ecf = float(energy_captured_fraction)
    if ecf > 0.0:
        total = kept / ecf
    else:
        # Guard: no captured energy — fall back to the kept sum only.
        total = kept
    if total <= 0.0:
        raise RuntimeError(f"POD energy total non-positive (kept={kept}, energy_captured_fraction={ecf})")
    return lam / total


def compute_reference_spectra(
    generator: str,
    params: dict[str, Any],
    n_modes: int,
    rtol: float,
) -> dict[str, Any]:
    """Run POD + DMD; return arrays ready for fixture write or comparison."""
    payload = generate_example_dataset(generator, params)
    loader = make_loader(payload)

    with tempfile.TemporaryDirectory(prefix=f"ref_{generator}_") as tmp:
        pod = PODAnalyzer(
            file_path=f"ref_{generator}",
            data_loader=loader,
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
            results_dir=tmp,
            figures_dir=tmp,
            use_parallel=False,
        )
        pod.load_and_preprocess()
        pod.perform_pod()
        lam = np.asarray(pod.eigenvalues, dtype=np.float64)
        ecf = float(pod.energy_captured_fraction)
        pod_frac = pod_fractions_over_pretruncation_total(lam, ecf)

        dmd = DMDAnalyzer(
            file_path=f"ref_{generator}_dmd",
            data_loader=loader,
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
            results_dir=tmp,
            figures_dir=tmp,
            rank=n_modes,
            use_parallel=False,
        )
        dmd.load_and_preprocess()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_DMD_EFFECTIVE_RANK_WARNING,
                category=RuntimeWarning,
            )
            dmd.perform_dmd()
        eig = canonicalize_dmd_eigenvalues(np.asarray(dmd.eigenvalues), rtol=rtol)
        dmd_abs = np.abs(eig).astype(np.float64)
        dmd_phase = np.angle(eig).astype(np.float64)

    return {
        "pod_energy_fractions": pod_frac,
        "dmd_abs_lambda": dmd_abs,
        "dmd_phase": dmd_phase,
        "energy_captured_fraction": ecf,
        "payload": payload,
    }
