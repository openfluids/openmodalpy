#!/usr/bin/env python3
"""Generate tests/fixtures/reference/external_spod.json from PySPOD.

Developer-only. Never collected or run by pytest. The package does not
depend on PySPOD: run this in a throwaway venv *outside* the repo tree.

    uv venv --python 3.12 <tmp>/.venv
    uv pip install pyspod==2.0.0 numpy==2.5.2 scipy==1.18.0
    <tmp>/.venv/bin/python scripts/regen_external_spod.py

Builds the SAME manufactured field as tests/test_spod_oracle.py (nfft 16,
8 blocks, zero overlap, dt 0.5, 4 points, tones at bins 3 and 5) and
refuses to write if mapped PySPOD disagrees with the closed form
(bin 3 -> 18.0 and 4.5, bin 5 -> 8.0) by more than 5e-3 relative.
The mapping is λ_openmodalpy = λ_pyspod × nfft × dt / 2. Re-running in
the same pinned environment overwrites the fixture byte-for-byte
identically (fixed date, single-thread BLAS, stable JSON).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import warnings
from pathlib import Path

# Pin BLAS before NumPy import so a re-run cannot change eig rounding.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "tests" / "fixtures" / "reference" / "external_spod.json"

PINNED_PYSPOD = "2.0.0"
GENERATION_DATE = "2026-08-17"
WRITE_GATE = 5e-3

# Restated from tests/test_spod_oracle.py. Do not invent a new field.
N_SPACE = 4
NFFT = 16
NBLOCKS = 8
K_BIN = 3
K_BIN2 = 5
DT = 0.5
A1 = 3.0
A2 = 1.5
A3 = 2.0
PHI1 = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0
PHI2 = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
PHI3 = np.array([1.0, 1.0, -1.0, -1.0]) / 2.0

OCCUPIED: tuple[tuple[int, int, float], ...] = (
    (K_BIN, 0, A1),
    (K_BIN, 1, A2),
    (K_BIN2, 0, A3),
)

# Derived bounds: see tests/test_external_spod.py for the prose.
TOLERANCES: dict[str, dict[str, object]] = {
    "mapped_vs_pyspod": {
        "value": 5e-3,
        "kind": "relative",
        "applies": "openmodalpy vs mapped PySPOD at occupied interior bins",
        "reason": (
            "After λ_openmodalpy = λ_pyspod * nfft * dt / 2, residuals on this "
            "field are 1.09e-3 (bin 3 mode 0), 9.0e-7 (bin 3 mode 1), 1.95e-3 "
            "(bin 5 mode 0). The split is the irreconcilable Hamming window "
            "(PySPOD symmetric vs openmodalpy periodic). 5e-3 is ~2.5x the "
            "worst measured residual. Dropping the Strouhal division moves the "
            "answer by 8x; power vs amplitude by 0.734 — both orders above "
            "5e-3. This bound is also the write-gate."
        ),
    },
    "closed_form": {
        "value": (NFFT + NBLOCKS) * float(np.finfo(float).eps),
        "kind": "relative",
        "applies": "openmodalpy vs (A/2)**2/dst at occupied interior bins",
        "reason": (
            "Amplitude-normalised Hamming recovers the closed form exactly "
            "(ratio 1.000000). The bound is (nfft + nblocks) * eps, the same "
            "FFT-plus-Gram round-off the oracle uses: about 13x the observed "
            "~2 eps residual. Not a mapped-PySPOD bound; do not loosen it "
            "toward 5e-3."
        ),
    },
}


def dst() -> float:
    """Strouhal step from the construction: ``df * L / U`` with L = U = 1."""
    return 1.0 / (NFFT * DT)


def expected_lambda(amplitude: float) -> float:
    """Closed-form SPOD energy of a real cosine of amplitude ``A``: ``(A/2)**2 / dst``."""
    return (amplitude / 2.0) ** 2 / dst()


def mapping_factor() -> float:
    """``λ_openmodalpy = λ_pyspod * nfft * dt / 2``."""
    return NFFT * DT / 2.0


def snapshot_sha256(snapshots: np.ndarray) -> str:
    """Hex sha256 of the snapshot array as float64 bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(snapshots, dtype=np.float64).tobytes()).hexdigest()


def manufactured_field() -> np.ndarray:
    """Rank-2 tone at ``K_BIN`` plus a third tone at ``K_BIN2``.

    Identical construction to ``tests/test_spod_oracle.py``: mode 1 has
    constant phase across blocks; mode 2 advances one full turn.
    """
    n_snapshots = NBLOCKS * NFFT
    field = np.zeros((n_snapshots, N_SPACE))
    for block in range(NBLOCKS):
        time = np.arange(NFFT)
        sl = slice(block * NFFT, (block + 1) * NFFT)
        phase2 = 2.0 * np.pi * block / NBLOCKS
        field[sl] = (
            A1 * np.outer(np.cos(2.0 * np.pi * K_BIN * time / NFFT), PHI1)
            + A2 * np.outer(np.cos(2.0 * np.pi * K_BIN * time / NFFT + phase2), PHI2)
            + A3 * np.outer(np.cos(2.0 * np.pi * K_BIN2 * time / NFFT), PHI3)
        )
    return field


def _clean_float(value: float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _require_pyspod():
    try:
        from pyspod.spod.standard import Standard
    except ImportError:
        sys.stderr.write(
            "PySPOD is not installed. This generator is not part of the test "
            "suite and must be run in a throwaway venv with the pinned reference:\n"
            "\n"
            "    uv venv --python 3.12 <tmp>/.venv\n"
            f"    uv pip install pyspod=={PINNED_PYSPOD} numpy==2.5.2 scipy==1.18.0\n"
            "    <tmp>/.venv/bin/python scripts/regen_external_spod.py\n"
            "\n"
            "openmodalpy itself must not depend on PySPOD.\n"
        )
        raise SystemExit(1) from None

    version = importlib.metadata.version("pyspod")
    if version != PINNED_PYSPOD:
        raise SystemExit(
            f"REFUSING TO WRITE: pyspod version is {version!r}; this script is pinned to pyspod=={PINNED_PYSPOD}."
        )
    return Standard, version


def _pyspod_eigs(standard_cls, snapshots: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="pyspod_ref_") as tmpdir:
        params = {
            "time_step": DT,
            "n_space_dims": 1,
            "n_variables": 1,
            "n_dft": NFFT,
            "overlap": 0,
            "mean_type": "longtime",
            "normalize_weights": False,
            "normalize_data": False,
            "n_modes_save": 4,
            "fullspectrum": False,
            "savefft": False,
            "savefreq_disk": False,
            "reuse_blocks": False,
            "savedir": tmpdir,
        }
        weights = {"weights": np.ones(N_SPACE, dtype=np.float64), "weights_name": "uniform"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spod = standard_cls(params=params, weights=weights, comm=None)
            spod.fit(snapshots)
        return np.asarray(spod.eigs, dtype=np.float64)


def _build_document(standard_cls, pyspod_version: str) -> dict:
    import scipy

    snapshots = manufactured_field()
    eigs = _pyspod_eigs(standard_cls, snapshots)
    factor = mapping_factor()
    occupied: list[dict] = []
    closed_form: list[dict] = []
    for bin_idx, mode_idx, amplitude in OCCUPIED:
        raw = _clean_float(float(eigs[bin_idx, mode_idx]))
        closed = _clean_float(expected_lambda(amplitude))
        mapped = raw * factor
        err = abs(mapped - closed) / abs(closed)
        if err > WRITE_GATE:
            raise SystemExit(
                f"REFUSING TO WRITE: mapped PySPOD bin={bin_idx} mode={mode_idx} "
                f"is {mapped:.6g} vs closed form {closed:.6g} "
                f"(relative {err:.3e}, gate {WRITE_GATE:.0e}). The mapping is "
                "wrong or PySPOD changed; a bad number must not reach the fixture."
            )
        occupied.append(
            {
                "bin": bin_idx,
                "mode": mode_idx,
                "pyspod_eigenvalue": raw,
            }
        )
        closed_form.append({"bin": bin_idx, "mode": mode_idx, "value": closed})

    return {
        "description": (
            "Vendored PySPOD eigenvalues for the openmodalpy external SPOD cross-check. "
            "Generated once outside the repo; the package does not depend on PySPOD."
        ),
        "provenance": {
            "pyspod_version": pyspod_version,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "generation_date": GENERATION_DATE,
            "pinned_pyspod": f"pyspod=={PINNED_PYSPOD}",
        },
        "construction": {
            "kind": "manufactured_cosine_tones",
            "source": "tests/test_spod_oracle.py",
            "n_space": N_SPACE,
            "nfft": NFFT,
            "nblocks": NBLOCKS,
            "overlap": 0.0,
            "dt": DT,
            "k_bin": K_BIN,
            "k_bin2": K_BIN2,
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "phi1": [float(v) for v in PHI1],
            "phi2": [float(v) for v in PHI2],
            "phi3": [float(v) for v in PHI3],
            "dst": dst(),
            "closed_form": closed_form,
            "field_sha256": snapshot_sha256(snapshots),
            "field": (
                "Real cosine tones on spatially orthonormal Walsh modes, "
                f"{NBLOCKS} contiguous blocks of length {NFFT}, zero overlap. "
                f"Bin {K_BIN}: amplitude {A1} on phi1 (constant phase) and "
                f"amplitude {A2} on phi2 (one full turn across blocks). "
                f"Bin {K_BIN2}: amplitude {A3} on phi3 (constant phase). "
                "Interior-bin closed form is (A/2)**2 / dst with "
                f"dst = 1/(nfft*dt) = {dst()}."
            ),
        },
        "mapping": {
            "formula": "lambda_openmodalpy = lambda_pyspod * nfft * dt / 2",
            "factor": factor,
            "nfft_dt": (
                "openmodalpy divides by the Strouhal step dst = 1/(nfft*dt); PySPOD does not. "
                "Verified by sweep: PySPOD's eigenvalue is independent of dt."
            ),
            "interior_doubling": ("PySPOD doubles interior bins (L[1:-1] *= 2 for real data); openmodalpy does not."),
            "window": (
                "PySPOD hard-codes symmetric Hamming 0.54-0.46*cos(2*pi*x/(N-1)) "
                "and offers no other window. openmodalpy uses "
                "scipy.signal.get_window(..., fftbins=True), the periodic one. "
                "The residual after mapping is that shape difference."
            ),
        },
        "solver_options": {
            "openmodalpy": {
                "class": "openmodalpy.SPODAnalyzer",
                "nfft": NFFT,
                "overlap": 0.0,
                "window_type": "hamming",
                "window_norm": "amplitude",
                "blockwise_mean": False,
                "spatial_weight_type": "uniform",
                "use_parallel": False,
                "characteristic_length": 1.0,
                "characteristic_velocity": 1.0,
            },
            "pyspod": {
                "class": "pyspod.spod.standard.Standard",
                "n_dft": NFFT,
                "overlap": 0,
                "overlap_unit": "percent",
                "mean_type": "longtime",
                "fullspectrum": False,
                "normalize_weights": False,
                "normalize_data": False,
                "n_modes_save": 4,
                "savefreq_disk": False,
                "weights": "uniform",
            },
        },
        "tolerances": TOLERANCES,
        "cases": {"manufactured": {"occupied": occupied}},
    }


def write_fixture(doc: dict, path: Path) -> None:
    """Write JSON with stable formatting (trailing newline, LF, no sort_keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    standard_cls, version = _require_pyspod()
    document = _build_document(standard_cls, version)
    write_fixture(document, OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
