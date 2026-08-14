"""Closed-form oracle for SPOD eigenvalue magnitudes.

The existing twin hands the kernel a hand-made ``qhat`` and rebuilds the
same formula the library uses, so it cannot tell a right energy convention
from a wrong one. This module drives ``SPODAnalyzer`` end to end on a
manufactured field whose modal energies are known from the construction: a
real cosine of amplitude ``A`` on a spatially orthonormal mode has block
coefficient ``|c| = A/2``, and the predicted eigenvalue is ``(A/2)**2 / dst``.

``|c| = A/2`` is the plain (UNDOUBLED) transform identity, not a one-sided
fold. SPOD does not double its interior bins, so ``lambda * dst = A**2/4`` --
HALF the tone variance ``A**2/2``. Do not read this as the one-sided energy
that ``tests/test_welch_analytical.py`` pins, where inner bins are doubled and
Parseval closes. It also holds only at INTERIOR bins: at DC and Nyquist the
coefficient is ``A``, not ``A/2``, and this closed form would be wrong there.

Two modes at the same frequency separate exactly only when BOTH their
per-block coefficient vectors AND their spatial shapes are orthogonal. The
phase ramp (constant phase for one, one full turn across blocks for the other)
gives the first; the Walsh vectors below give the second. Either one alone is
not enough -- with spatial overlap 0.3 the same construction returns 18.52 and
3.98 instead of 18 and 4.5. Their SUM is still 22.5, so a test that checked
only the total energy at this bin would stay green; that is why each
eigenvalue is asserted on its own. Every tone sits on an integer bin so energy
does not leak, and the two occupied bins are two apart so window side lobes
cannot mix them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openmodalpy import SPODAnalyzer

N_SPACE = 4
NFFT = 16
NBLOCKS = 8
K_BIN = 3
K_BIN2 = 5
# Deliberately not 1.0: with dt = 1 the Strouhal step 1/(nfft*dt) collapses to
# 1/nfft, and a library that dropped dt from the frequency axis would still
# produce these numbers.
DT = 0.5
A1 = 3.0
A2 = 1.5
A3 = 2.0

# Spatially orthonormal under uniform unit weights (phi · phi = 1, pairwise 0).
PHI1 = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0
PHI2 = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
PHI3 = np.array([1.0, 1.0, -1.0, -1.0]) / 2.0

_WINDOWS = ("boxcar", "hamming", "hann")

# (mean(w)/rms(w))**2 for the PERIODIC window of each shape, written out from
# the window definitions rather than fetched from scipy. `core/welch.py` builds
# its window with `scipy.signal.get_window(name, nfft, fftbins=True)`, so
# computing this factor with that same call would make the prediction a twin of
# the library's window handling: a shared-source mistake (both sides on the
# wrong shape, or both on `fftbins=False`) would stay green. Written out, a
# switch to the symmetric window turns these red -- the factors differ by 4.5%
# for hamming and 6.25% for hann.
# Periodic hann is (1 - cos(2*pi*n/N))/2: mean 1/2, mean(w**2) 3/8 -> 2/3.
# Periodic hamming is 0.54 - 0.46*cos(2*pi*n/N): mean 0.54, mean(w**2)
# 0.54**2 + 0.46**2/2.
_POWER_NORM_FACTOR = {
    "boxcar": 1.0,
    "hamming": 0.54**2 / (0.54**2 + 0.5 * 0.46**2),
    "hann": 2.0 / 3.0,
}


def _relative_tolerance() -> float:
    """Relative round-off bound for the block FFT plus the block Gram.

    The block FFT sums ``nfft`` terms and the Gram sums ``nblocks``, so
    relative round-off goes like ``(nfft + nblocks) * eps``. Observed error
    is about 2 eps, so this bound carries roughly 13x margin.
    """
    return (NFFT + NBLOCKS) * float(np.finfo(float).eps)


def _dst(*, length: float = 1.0, velocity: float = 1.0) -> float:
    """Strouhal step from the construction: ``df * L / U``, ``df = 1/(nfft*dt)``."""
    df = 1.0 / (NFFT * DT)
    return df * length / velocity


def _expected_lambda(amplitude: float, dst: float) -> float:
    """Closed-form SPOD energy of a real cosine of amplitude ``A``: ``(A/2)**2 / dst``."""
    return (amplitude / 2.0) ** 2 / dst


def _manufactured_field() -> dict:
    """Rank-2 tone at ``K_BIN`` plus a third tone at ``K_BIN2``.

    Mode 1 has constant phase across blocks; mode 2 advances one full turn
    (``2*pi*b/nblocks``). Those two block-coefficient vectors are orthogonal,
    so the two energies at ``K_BIN`` separate exactly.
    """
    ns = NBLOCKS * NFFT
    q = np.zeros((ns, N_SPACE))
    for block in range(NBLOCKS):
        t = np.arange(NFFT)
        sl = slice(block * NFFT, (block + 1) * NFFT)
        phase2 = 2.0 * np.pi * block / NBLOCKS
        q[sl] = (
            A1 * np.outer(np.cos(2.0 * np.pi * K_BIN * t / NFFT), PHI1)
            + A2 * np.outer(np.cos(2.0 * np.pi * K_BIN * t / NFFT + phase2), PHI2)
            + A3 * np.outer(np.cos(2.0 * np.pi * K_BIN2 * t / NFFT), PHI3)
        )
    return {
        "q": q,
        "x": np.arange(N_SPACE, dtype=float),
        "y": np.array([0.0]),
        "dt": DT,
        "Nx": N_SPACE,
        "Ny": 1,
        "Ns": ns,
    }


def _run_spod(
    tmp_path: Path,
    *,
    window_type: str,
    window_norm: str,
    characteristic_length: float | None = None,
    characteristic_velocity: float | None = None,
    file_path: str = "spod_oracle",
) -> np.ndarray:
    """Drive ``SPODAnalyzer`` end to end; cache lands under ``tmp_path``."""
    field = _manufactured_field()
    analyzer = SPODAnalyzer(
        file_path=file_path,
        nfft=NFFT,
        overlap=0.0,
        window_type=window_type,
        window_norm=window_norm,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
        use_parallel=False,
        characteristic_length=characteristic_length,
        characteristic_velocity=characteristic_velocity,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    return np.asarray(analyzer.eigenvalues)


@pytest.mark.parametrize("window_type", _WINDOWS)
def test_amplitude_norm_eigenvalues_match_closed_form(tmp_path: Path, window_type: str) -> None:
    """Amplitude-normalised boxcar/hamming/hann recover ``(A/2)**2 / dst`` exactly.

    Three windows so a single window's coherent gain cannot masquerade as the
    CSD convention. Two amplitudes at one bin pin order, ratio, and rank;
    a second bin with a different amplitude stops one lucky frequency carrying
    the test.
    """
    lam = _run_spod(tmp_path, window_type=window_type, window_norm="amplitude")
    dst = _dst()
    rtol = _relative_tolerance()
    want1 = _expected_lambda(A1, dst)
    want2 = _expected_lambda(A2, dst)
    want3 = _expected_lambda(A3, dst)

    np.testing.assert_allclose(lam[K_BIN, 0], want1, rtol=rtol, atol=0.0)
    np.testing.assert_allclose(lam[K_BIN, 1], want2, rtol=rtol, atol=0.0)
    assert lam[K_BIN, 0] > lam[K_BIN, 1]
    np.testing.assert_allclose(lam[K_BIN, 0] / lam[K_BIN, 1], (A1 / A2) ** 2, rtol=rtol, atol=0.0)
    assert lam[K_BIN, 2] <= rtol * lam[K_BIN, 0]

    np.testing.assert_allclose(lam[K_BIN2, 0], want3, rtol=rtol, atol=0.0)
    assert lam[K_BIN2, 1] <= rtol * lam[K_BIN2, 0]


@pytest.mark.parametrize("window_type", _WINDOWS)
def test_power_norm_energy_matches_window_factor(tmp_path: Path, window_type: str) -> None:
    """Power normalisation scales the energy by a factor the window itself fixes.

    ``window_norm="power"`` is the DEFAULT (``core/config.py``), so this is the
    path most callers take. It divides by ``rms(w)`` where amplitude
    normalisation divides by ``mean(w)``, so a bin-centred tone comes back
    scaled by ``(mean(w)/rms(w))**2``. That factor follows from the window
    definition, not from anything the library computes, so the closed form
    still applies here.

    The within-bin ratio is asserted too, but it cannot carry this test on its
    own: a ratio is invariant to ANY global gain, so it stays green under every
    normalisation error. Measured that directly -- the ratio-only version of
    this test passed under all three library mutations the gate applies.
    """
    lam = _run_spod(tmp_path, window_type=window_type, window_norm="power")
    rtol = _relative_tolerance()

    factor = _POWER_NORM_FACTOR[window_type]
    np.testing.assert_allclose(lam[K_BIN, 0], _expected_lambda(A1, _dst()) * factor, rtol=rtol, atol=0.0)
    np.testing.assert_allclose(lam[K_BIN, 1], _expected_lambda(A2, _dst()) * factor, rtol=rtol, atol=0.0)
    np.testing.assert_allclose(lam[K_BIN, 0] / lam[K_BIN, 1], (A1 / A2) ** 2, rtol=rtol, atol=0.0)
    assert lam[K_BIN, 0] > lam[K_BIN, 1]


def test_eigenvalues_scale_with_u_over_l(tmp_path: Path) -> None:
    """Same field, two characteristic scales: eigenvalues scale by the predicted ``U/L``.

    ``dst`` is a Strouhal step (``df * L / U``), so the reported energies scale
    with ``U/L``. This is a metamorphic relation, not a restatement of the
    closed form at a second pair of scales.
    """
    lam_ref = _run_spod(
        tmp_path,
        window_type="boxcar",
        window_norm="amplitude",
        characteristic_length=1.0,
        characteristic_velocity=1.0,
        file_path="spod_oracle_L1U1",
    )
    cases = (
        (2.0, 1.0),
        (1.0, 2.0),
        (4.0, 2.0),
    )
    rtol = _relative_tolerance()
    for length, velocity in cases:
        lam = _run_spod(
            tmp_path,
            window_type="boxcar",
            window_norm="amplitude",
            characteristic_length=length,
            characteristic_velocity=velocity,
            file_path=f"spod_oracle_L{length:g}U{velocity:g}",
        )
        scale = velocity / length
        # Only the occupied energies are a physical scale; empty-bin residues
        # are round-off and some are already clamped to zero.
        np.testing.assert_allclose(lam[K_BIN, :2], lam_ref[K_BIN, :2] * scale, rtol=rtol, atol=0.0)
        np.testing.assert_allclose(lam[K_BIN2, :1], lam_ref[K_BIN2, :1] * scale, rtol=rtol, atol=0.0)
