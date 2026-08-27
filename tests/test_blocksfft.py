import inspect

import numpy as np
import pytest

from openmodalpy.core.base import blocksfft, spod_function
from openmodalpy.core.parallel import blocksfft_optimized
from openmodalpy.core.welch import windowed_block_fft

WINDOWS = ("hamming", "hann", "blackman", "bartlett", "sine")


# Floor partitioning matching scipy.signal.welch / the fixed production formula.
def _nblocks_floor(Ns, nfft, overlap):
    novlap = int(overlap * nfft)
    return (Ns - novlap) // (nfft - novlap), novlap


def _block_starts(nblocks, nfft, novlap):
    hop = nfft - novlap
    return [iblk * hop for iblk in range(nblocks)]


def test_blocksfft_constant_signal():
    q = np.ones((4, 2))
    result = blocksfft(q, nfft=4, nblocks=1, novlap=0)
    assert result.shape == (4 // 2 + 1, 2, 1)
    assert np.allclose(result, 0)


def test_public_blocksfft_names_delegate_to_windowed_block_fft():
    """Both public names are thin wrappers around the single shared implementation.

    Since cee9a89, blocksfft and blocksfft_optimized are not two algorithms —
    they are two names for windowed_block_fft. The only contract still worth
    pinning between them is that delegation. Numerical correctness is pinned
    by the definition oracles in test_welch_analytical.py, not by comparing
    the two names to each other.
    """
    assert "return windowed_block_fft(" in inspect.getsource(blocksfft)
    assert "return windowed_block_fft(" in inspect.getsource(blocksfft_optimized)
    # Same callable object on both wrapper modules (not a re-export copy).
    from openmodalpy.core import base as base_mod
    from openmodalpy.core import parallel as parallel_mod

    assert base_mod.windowed_block_fft is windowed_block_fft
    assert parallel_mod.windowed_block_fft is windowed_block_fft


def test_blocksfft_hann_blackman_differ_from_hamming():
    """hann/blackman must not silently collapse onto the hamming path."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    hamming = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="hamming")
    for w in ("hann", "blackman", "sine"):
        other = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=w)
        assert not np.allclose(other, hamming), f"{w} collapsed onto hamming"


def test_blocksfft_unsupported_window_raises():
    """Unsupported window names must raise, not silently substitute."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    with pytest.raises(ValueError) as exc_info:
        blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="not_a_window")
    # The offending name must appear, so an unrelated ValueError cannot satisfy this.
    assert "not_a_window" in str(exc_info.value)


def test_normvar_divides_by_variance_two_scales():
    """normvar divides by variance: normalised block has var 1/v, at two scales.

    Provenance: spod_matlab opts.normvar / PySPOD normalize_data. Dividing by
    the standard deviation would yield unit variance and scale invariance;
    both scales must assert var -> 1/v so a "fix" to std would break this.
    Absolute ddof=1 scaling is pinned in test_welch_analytical (definition oracle).
    """
    rng = np.random.default_rng(0)
    nfft, npts = 32, 3
    base = rng.standard_normal((nfft, npts))
    base = base - base.mean(axis=0)

    for scale in (1.0, 7.0):
        q = base * scale
        v = np.var(q, axis=0, ddof=1)
        # What the arithmetic does under the documented definition.
        normalised = q / v
        got = np.var(normalised, axis=0, ddof=1)
        expected = 1.0 / v
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

        # Full path: scaling input by c scales the FFT by 1/c under normvar.
        out = blocksfft(
            q,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            window_type="boxcar",
        )
        out_base = blocksfft(
            base,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            window_type="boxcar",
        )
        np.testing.assert_allclose(out, out_base / scale, rtol=0, atol=1e-12)


def test_normvar_zero_variance_channel_isfinite():
    """Constant (zero-variance) channel is clamped, not Inf/NaN, under normvar.

    After mean subtraction a constant channel has variance 0; the 4*eps clamp
    sets the divisor to 1 so the channel matches the no-normvar path.
    """
    rng = np.random.default_rng(1)
    nfft = 32
    varying = rng.standard_normal(nfft)
    constant = np.full(nfft, 3.0)
    q = np.column_stack([varying, constant])

    with_nv = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=True)
    without = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=False)

    assert np.all(np.isfinite(with_nv)), "normvar produced non-finite values on a constant channel"
    # Constant channel: clamp divisor to 1 → identical to no-normvar path.
    np.testing.assert_allclose(with_nv[:, 1, :], without[:, 1, :], rtol=0, atol=1e-12)
    # Varying channel must actually change under normvar (else the flag is a no-op).
    assert not np.allclose(with_nv[:, 0, :], without[:, 0, :])


# ---------------------------------------------------------------------------
# Welch block partitioning (floor, drop remainder — no end-clamp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Ns,nfft,overlap,label",
    [
        (100, 128, 0.5, "Ns < nfft"),
        (64, 128, 0.5, "Ns <= novlap"),
    ],
)
def test_short_record_raises_naming_Ns_and_nfft(Ns, nfft, overlap, label):
    """Short records must raise ValueError naming both Ns and nfft.

    Pre-fix: ceil/floor of a short record could yield nblocks=0 (silent empty)
    or a negative start (numpy wrap → garbage). Assert on the exception message,
    not on an output shape. One public entry point is enough: both wrappers
    share windowed_block_fft (see delegation test).
    """
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    q = np.zeros((Ns, 4))
    with pytest.raises(ValueError) as ei:
        blocksfft(q, nfft, max(nb, 1), novlap)
    msg = str(ei.value)
    assert str(Ns) in msg and str(nfft) in msg, f"{label}: message {msg!r} must name Ns and nfft"


@pytest.mark.parametrize("Ns,nfft,overlap", [(500, 128, 0.5), (300, 128, 0.5)])
def test_oversize_nblocks_raises_not_clamp(Ns, nfft, overlap):
    """Old ceil nblocks that no longer fit must raise, not silently clamp."""
    nb_floor, novlap = _nblocks_floor(Ns, nfft, overlap)
    nb_ceil = int(np.ceil((Ns - novlap) / (nfft - novlap)))
    if nb_ceil == nb_floor:
        pytest.skip("ceil and floor agree; clamp path not exercised")
    q = np.zeros((Ns, 4))
    with pytest.raises(ValueError) as ei:
        blocksfft(q, nfft, nb_ceil, novlap)
    msg = str(ei.value)
    assert str(Ns) in msg or str(nb_ceil) in msg, msg


@pytest.mark.parametrize(
    "Ns,nfft,overlap",
    [
        (500, 128, 0.5),  # shipped cylinder_wake shape: 6 blocks, remainder dropped
        (300, 128, 0.5),  # does not divide evenly
        (500, 100, 0.5),
        (256, 64, 0.25),
    ],
)
def test_block_starts_strict_hop_and_fit(Ns, nfft, overlap):
    """Block starts must be strictly increasing with constant hop, last fits.

    Asserted on the indices themselves (not merely on nblocks), via an impulse
    basis: column j is an impulse at time j; energy in block k recovers coverage.
    boxcar is required so edge impulses are not window-attenuated below threshold.

    hop = nfft - novlap by definition (Welch / scipy.signal.welch). One entry
    point is enough — both public names delegate to the same implementation.
    """
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    hop = nfft - novlap
    expected_starts = _block_starts(nb, nfft, novlap)
    assert expected_starts[-1] + nfft <= Ns

    qhat = blocksfft(np.eye(Ns), nfft, nb, novlap, window_type="boxcar")
    energy = np.sum(np.abs(qhat) ** 2, axis=0)  # [time, block]
    starts = []
    for k in range(nb):
        idx = np.flatnonzero(energy[:, k] > 0.5 * energy[:, k].max())
        assert idx.size > 0
        starts.append(int(idx[0]))

    assert starts == expected_starts, f"starts {starts} != {expected_starts}"
    assert all(s2 - s1 == hop for s1, s2 in zip(starts, starts[1:])), "non-constant hop"
    assert starts[-1] + nfft <= Ns


@pytest.mark.parametrize(
    "novlap",
    [64, 65],  # == nfft and > nfft; both yield hop <= 0
    ids=["novlap_eq_nfft", "novlap_gt_nfft"],
)
def test_novlap_ge_nfft_raises_not_repeat_block0(novlap):
    """hop <= 0 must raise: otherwise every block starts at 0 (identical members)."""
    q = np.zeros((300, 4))
    with pytest.raises(ValueError, match=r"hop|novlap|nfft"):
        blocksfft(q, 64, 3, novlap)


def test_apply_snapshot_limit_uses_floor_nblocks():
    """BaseAnalyzer._apply_snapshot_limit must set floor nblocks and stay runnable.

    Slice-2 regression: ceil after max_snapshots truncation requested more
    blocks than fit (Ns=400, nfft=128, overlap=0.5 → floor 5, ceil 6 needs 448).
    Pinning nblocks alone is not enough — blocksfft must accept the result.
    The limiter moved from commands.py onto BaseAnalyzer with the unified seam.
    """
    import types

    from openmodalpy.core.base import BaseAnalyzer

    nfft, novlap, limit = 128, 64, 400
    expect = (limit - novlap) // (nfft - novlap)
    assert expect == 5
    # Old ceil would have been 6 and needed (6-1)*64+128 = 448 > 400.
    assert int(np.ceil((limit - novlap) / (nfft - novlap))) == 6

    an = types.SimpleNamespace(
        data={"q": np.zeros((500, 4)), "Ns": 500},
        novlap=novlap,
        nfft=nfft,
        nblocks=6,
    )
    BaseAnalyzer._apply_snapshot_limit(an, limit)

    assert an.data["Ns"] == limit
    assert an.data["q"].shape[0] == limit
    assert an.nblocks == expect

    # Must actually be usable: the original break was a ValueError here.
    out = blocksfft(an.data["q"], nfft, an.nblocks, novlap)
    assert out.shape[2] == expect


def test_compute_fft_blocks_nblocks_uses_floor(tmp_path):
    """BaseAnalyzer.compute_fft_blocks must use floor nblocks, not ceil.

    Slice-1 tests call blocksfft with a precomputed nblocks, so the analyzer's
    own formula was untested. Ns=400, nfft=128, overlap=0.5 → floor 5, ceil 6.
    """
    from openmodalpy import SPODAnalyzer

    Ns, nfft, overlap = 400, 128, 0.5
    novlap = int(overlap * nfft)
    expect_floor = (Ns - novlap) // (nfft - novlap)
    expect_ceil = int(np.ceil((Ns - novlap) / (nfft - novlap)))
    assert expect_floor == 5 and expect_ceil == 6

    q = np.zeros((Ns, 4))
    data = {
        "q": q,
        "x": np.linspace(0, 1, 4),
        "y": np.linspace(0, 1, 1),
        "dt": 1.0,
        "Nx": 4,
        "Ny": 1,
        "Ns": Ns,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=nfft,
        overlap=overlap,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    assert analyzer.nblocks == expect_floor


def test_compute_fft_blocks_short_record_raises(tmp_path):
    """Truncated-to-shorter-than-nfft record must raise a clear ValueError."""
    from openmodalpy import SPODAnalyzer

    Ns, nfft, overlap = 100, 128, 0.5
    q = np.zeros((Ns, 4))
    data = {
        "q": q,
        "x": np.linspace(0, 1, 4),
        "y": np.linspace(0, 1, 1),
        "dt": 1.0,
        "Nx": 4,
        "Ny": 1,
        "Ns": Ns,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=nfft,
        overlap=overlap,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    with pytest.raises(ValueError) as ei:
        analyzer.compute_fft_blocks()
    msg = str(ei.value)
    assert str(Ns) in msg and str(nfft) in msg


def test_white_noise_spod_eigenvalue_matches_analytic():
    """Unit-variance white noise: mean mid-band SPOD eigenvalue ≈ 1.

    Derivation (single spatial point, W=1, boxcar, dt=1 so fs=1, dst=df=1/nfft):
    blocksfft stores q_hat = FFT(x)/nfft. For unit-variance white noise,
    E[|X[k]/nfft|^2] = 1/nfft at each one-sided bin (numpy unnormalized FFT).
    spod_function forms λ = mean_b |q_hat_b|^2 / dst = mean |q_hat|^2 * nfft,
    so E[λ(f)] = 1 at every interior frequency.

    This pins the FFT → SPOD normalization chain (window, 1/nfft, dst), not
    block placement. On white noise a clamped/reused block does not bias
    E[λ]; it only changes the variance of the mean. Partitioning is covered
    by the impulse-probe and oversize-nblocks tests above.

    Tolerance: an ensemble of nblocks approximately independent periodogram
    members has relative standard error ~ 1/sqrt(nblocks) for the mean over
    frequencies of comparable width. We average over ~nfft/2 mid-band bins and
    require |mean(λ) - 1| < 4/sqrt(nblocks). Tighter than ~1/sqrt(nblocks) is
    flaky under finite-sample noise.
    """
    rng = np.random.default_rng(0)
    Ns, nfft, overlap = 4096, 128, 0.5
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    assert nb >= 16
    q = rng.standard_normal((Ns, 1))
    qhat = blocksfft(q, nfft, nb, novlap, window_type="boxcar")
    dst = 1.0 / nfft
    w = np.ones((1, 1))
    lams = []
    for ifreq in range(1, qhat.shape[0] - 1):  # skip DC and Nyquist
        _, lam = spod_function(qhat[ifreq], nblocks=nb, dst=dst, w=w, use_parallel=False)
        lams.append(lam[0])
    mean_lam = float(np.mean(lams))
    tol = 4.0 / np.sqrt(nb)
    assert abs(mean_lam - 1.0) < tol, (
        f"mean mid-band SPOD λ={mean_lam:.4f} vs analytic 1.0 (tol={tol:.4f} = 4/sqrt(nblocks={nb}))"
    )
