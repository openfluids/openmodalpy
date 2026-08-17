#!/usr/bin/env python3
"""Generate tests/fixtures/reference/external_dmd.json from PyDMD.

Developer-only. Never collected or run by pytest. The package does not
depend on PyDMD: run this in a throwaway venv *outside* the repo tree.

    uv venv --python 3.12 <tmp>/.venv
    uv pip install pydmd==2025.8.1 numpy==2.5.2 scipy==1.18.0
    <tmp>/.venv/bin/python scripts/regen_external_reference.py

Builds a 12-space, 40-snapshot linear system FROM five chosen eigenvalues
and refuses to write if noiseless PyDMD disagrees with that constructed
spectrum by more than 1e-12. Re-running in the same pinned environment
overwrites the fixture byte-for-byte identically (fixed date, single-thread
BLAS, stable JSON).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from pathlib import Path

# Pin BLAS before NumPy import so a re-run cannot change SVD rounding.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "tests" / "fixtures" / "reference" / "external_dmd.json"

PINNED_PYDMD = "2025.8.1"
GENERATION_DATE = "2026-08-15"
WRITE_GATE = 1e-12

N_SPACE = 12
N_SNAPSHOTS = 40
RANK = 5
FIELD_SEED = 20260815
NOISE_SEED = 20260816
NOISE_RELATIVE_RMS = 1e-3

# Two conjugate pairs plus one real; spectral radius 0.95.
CHOSEN_POLAR: tuple[tuple[float, float], ...] = (
    (0.95, 0.6),
    (0.95, -0.6),
    (0.80, 1.2),
    (0.80, -1.2),
    (0.70, 0.0),
)

# Derived bounds: see tests/test_external_reference.py for the prose.
TOLERANCES: dict[str, dict[str, object]] = {
    "roundoff_vs_pydmd": {
        "value": 1e-12,
        "applies": "noiseless LS/TLS vs PyDMD; noisy LS vs PyDMD",
        "reason": (
            "Same-algebra comparisons measure 1.4e-15 to 1.9e-15 on the pinned "
            "stack. 1e-12 is the write-gate (three orders of BLAS headroom) and "
            "six orders below the noisy TLS-LS split. A wrong operator cannot hide "
            "on the noisy rows (the load-bearing check); a noiseless TLS->LS swap "
            "measures 1.024e-15 vs vendored TLS and does hide."
        ),
    },
    "roundoff_vs_chosen": {
        "value": 1e-12,
        "applies": "noiseless LS/TLS vs the constructed eigenvalues",
        "reason": (
            "Noiseless PyDMD and openmodalpy both recover the constructed "
            "spectrum to ~1e-15. The same 1e-12 write-gate is the bound."
        ),
    },
    "noisy_tls_vs_pydmd": {
        "value": 1e-8,
        "applies": "noise 1e-3 rms, TLS vs PyDMD",
        "reason": (
            "The TLS routes differ algebraically (left-singular split vs right-"
            "singular projection). Measured residual 3.1e-10; TLS vs LS is 6.2e-6. "
            "1e-8 is the first power of ten inside that gap: ~30x above the "
            "route residual, ~600x below a TLS->LS fallback."
        ),
    },
    "noisy_vs_chosen": {
        "value": 3e-4,
        "applies": "noise 1e-3 rms, LS/TLS vs the constructed eigenvalues",
        "reason": (
            "No closed form for the noisy estimator. Measured bias on this field "
            "is 8.3e-5. 3e-4 is ~4x that residual and 3x below the injected "
            "noise amplitude. This is a loose sanity floor; the discriminating "
            "check is the vendored PyDMD comparison, not this bound."
        ),
    },
}


def _rot_scale(theta: float, radius: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return radius * np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def reduced_operator() -> np.ndarray:
    """Real 5×5 block-diagonal operator whose spectrum is CHOSEN_POLAR."""
    operator = np.zeros((RANK, RANK), dtype=np.float64)
    operator[0:2, 0:2] = _rot_scale(0.6, 0.95)
    operator[2:4, 2:4] = _rot_scale(1.2, 0.80)
    operator[4, 4] = 0.70
    return operator


def chosen_eigenvalues() -> np.ndarray:
    return np.asarray([modulus * np.exp(1j * angle) for modulus, angle in CHOSEN_POLAR], dtype=np.complex128)


def eig_set_err(got, want) -> float:
    """Max distance after matching spectra as sorted sets (order-independent)."""
    got_arr = np.asarray(got)
    want_arr = np.asarray(want)
    if got_arr.size != want_arr.size:
        return np.inf
    return float(np.max(np.abs(np.sort_complex(got_arr) - np.sort_complex(want_arr))))


def snapshot_sha256(snapshots: np.ndarray) -> str:
    """Hex sha256 of the snapshot array as float64 bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(snapshots, dtype=np.float64).tobytes()).hexdigest()


def snapshots_space_time(*, noise_relative_rms: float) -> np.ndarray:
    """Return X with shape (n_space, n_snapshots)."""
    rng = np.random.default_rng(FIELD_SEED)
    observe = rng.standard_normal((N_SPACE, RANK))
    operator = reduced_operator()
    state = np.ones(RANK, dtype=np.float64)
    columns = np.empty((N_SPACE, N_SNAPSHOTS), dtype=np.float64)
    for step in range(N_SNAPSHOTS):
        columns[:, step] = observe @ state
        state = operator @ state
    if noise_relative_rms > 0.0:
        rms = float(np.sqrt(np.mean(columns * columns)))
        noise_rng = np.random.default_rng(NOISE_SEED)
        columns = columns + (noise_relative_rms * rms) * noise_rng.standard_normal(columns.shape)
    return columns


def snapshots_time_space(*, noise_relative_rms: float) -> np.ndarray:
    """Return q with shape (n_snapshots, n_space) for DMDAnalyzer."""
    return snapshots_space_time(noise_relative_rms=noise_relative_rms).T


def _clean_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _cplx_pairs(eigs) -> list[list[float]]:
    ordered = np.sort_complex(np.asarray(eigs, dtype=np.complex128).reshape(-1))
    return [[_clean_float(z.real), _clean_float(z.imag)] for z in ordered]


def _require_pydmd():
    try:
        from pydmd import DMD
    except ImportError:
        sys.stderr.write(
            "PyDMD is not installed. This generator is not part of the test "
            "suite and must be run in a throwaway venv with the pinned reference:\n"
            "\n"
            "    uv venv --python 3.12 <tmp>/.venv\n"
            f"    uv pip install pydmd=={PINNED_PYDMD} numpy==2.5.2 scipy==1.18.0\n"
            "    <tmp>/.venv/bin/python scripts/regen_external_reference.py\n"
            "\n"
            "openmodalpy itself must not depend on PyDMD.\n"
        )
        raise SystemExit(1) from None

    version = importlib.metadata.version("pydmd")
    if version != PINNED_PYDMD:
        raise SystemExit(
            f"REFUSING TO WRITE: pydmd version is {version!r}; this script is pinned to pydmd=={PINNED_PYDMD}."
        )
    return DMD, version


def _pydmd_eigs(dmd_cls, snapshots: np.ndarray, *, tls: bool) -> np.ndarray:
    kwargs: dict[str, int] = {"svd_rank": RANK}
    if tls:
        kwargs["tlsq_rank"] = RANK
    dmd = dmd_cls(**kwargs)
    # Rank-5 data in 12-space is exactly singular; PyDMD warns about cond(X).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dmd.fit(snapshots)
    return np.asarray(dmd.eigs, dtype=np.complex128)


def _build_document(dmd_cls, pydmd_version: str) -> dict:
    import scipy

    chosen = chosen_eigenvalues()
    cases: dict[str, dict] = {}
    field_sha256: dict[str, str] = {}
    for case_name, noise in (("noiseless", 0.0), ("noise_1e-3", NOISE_RELATIVE_RMS)):
        snapshots = snapshots_space_time(noise_relative_rms=noise)
        field_sha256[case_name] = snapshot_sha256(snapshots)
        methods: dict[str, dict] = {}
        for method, tls in (("ls", False), ("tls", True)):
            eigs = _pydmd_eigs(dmd_cls, snapshots, tls=tls)
            if noise == 0.0:
                err = eig_set_err(eigs, chosen)
                if err > WRITE_GATE:
                    raise SystemExit(
                        f"REFUSING TO WRITE: noiseless PyDMD method={method} disagrees "
                        f"with the constructed eigenvalues by {err:.3e} "
                        f"(gate {WRITE_GATE:.0e}). The construction is wrong or PyDMD "
                        "changed; a bad number must not reach the fixture."
                    )
            methods[method] = {"pydmd_eigenvalues": _cplx_pairs(eigs)}
        cases[case_name] = {
            "noise_relative_rms": noise,
            "methods": methods,
        }

    return {
        "description": (
            "Vendored PyDMD eigenvalues for the openmodalpy external DMD cross-check. "
            "Generated once outside the repo; the package does not depend on PyDMD."
        ),
        "provenance": {
            "pydmd_version": pydmd_version,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "generation_date": GENERATION_DATE,
            "pinned_pydmd": f"pydmd=={PINNED_PYDMD}",
        },
        "construction": {
            "kind": "linear_observation_of_chosen_spectrum",
            "n_space": N_SPACE,
            "n_snapshots": N_SNAPSHOTS,
            "rank": RANK,
            "spectral_radius": 0.95,
            "field_seed": FIELD_SEED,
            "noise_seed": NOISE_SEED,
            "noise_relative_rms": NOISE_RELATIVE_RMS,
            "z0": "ones(5)",
            "chosen_polar": [{"modulus": modulus, "angle": angle} for modulus, angle in CHOSEN_POLAR],
            "chosen_eigenvalues": _cplx_pairs(chosen),
            "field_sha256": field_sha256,
            "operator": (
                "Real 5x5 block-diagonal: 2x2 rotation-scaling blocks for the "
                "conjugate pairs (radius 0.95 at angle +/-0.6, radius 0.80 at "
                "angle +/-1.2) and a 1x1 block 0.70. Observation C is a seeded "
                "real 12x5 standard_normal matrix from numpy.random.default_rng("
                f"{FIELD_SEED}); snapshots are x_k = C z_k with z_0 = ones(5) "
                "and z_{k+1} = A z_k. Additive noise (noise_1e-3 only) is "
                f"independent N(0, (1e-3 * rms)^2) from default_rng({NOISE_SEED}), "
                "where rms is the noiseless field RMS."
            ),
        },
        "solver_options": {
            "openmodalpy": {
                "class": "openmodalpy.DMDAnalyzer",
                "rank": RANK,
                "n_modes_save": RANK,
                "delays": 1,
                "spatial_weight_type": "uniform",
                "mean_subtraction": False,
                "methods": {
                    "ls": {"method": "ls"},
                    "tls": {"method": "tls"},
                },
            },
            "pydmd": {
                "class": "pydmd.DMD",
                "exact": False,
                "opt": False,
                "methods": {
                    "ls": {"svd_rank": RANK, "tlsq_rank": 0},
                    "tls": {"svd_rank": RANK, "tlsq_rank": RANK},
                },
            },
        },
        "tolerances": TOLERANCES,
        "cases": cases,
    }


def write_fixture(doc: dict, path: Path) -> None:
    """Write JSON with stable formatting (trailing newline, LF, no sort_keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    dmd_cls, version = _require_pydmd()
    document = _build_document(dmd_cls, version)
    write_fixture(document, OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
