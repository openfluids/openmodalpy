"""Analytical checks for the single windowed-block FFT convention.

These tests pin the scaling and window against independent references
(Parseval on a bin-centred tone, amplitude recovery, scipy.signal.welch).
They do not compare the two public entry points to each other.
"""

import numpy as np
import pytest
from scipy.signal import get_window, welch

from openmodalpy.core.base import blocksfft
from openmodalpy.core.welch import windowed_block_fft


def _onesided_mean_square(q_hat):
    """Mean over blocks of |q_hat|^2, inner bins doubled (one-sided energy)."""
    ms = np.mean(np.abs(q_hat) ** 2, axis=-1)
    # ms shape: (nfreq, nmesh) — double positive-frequency bins, not DC/Nyquist.
    if ms.shape[0] > 2:
        ms = ms.copy()
        ms[1:-1] *= 2.0
    return ms


def _manual_windowed_rfft(block, window, window_norm):
    """One-sided windowed rFFT from the documented scaling rule.

    Definition (welch.windowed_block_fft docstring):
        q_hat = FFT(w * x) / (nfft * scale)
    with scale = sqrt(mean(w^2)) for power and mean(w) for amplitude.
    Uses numpy.fft directly — not the production FFT backend helper.
    """
    nfft = block.shape[0]
    if window_norm == "amplitude":
        scale = float(np.mean(window))
    else:
        scale = float(np.sqrt(np.mean(window**2)))
    centered = block - np.mean(block, axis=0)
    windowed = centered * window[:, np.newaxis]
    full = np.fft.fft(windowed, axis=0)
    return (1.0 / (nfft * scale)) * full[: nfft // 2 + 1]


@pytest.mark.oracle
def test_parseval_bin_centred_tone_power_norm():
    """Power-norm one-sided mean-square spectrum sums to signal variance.

    fs=1000 Hz, nfft=256, f0=125 Hz (bin 32), hann, amplitude A=2 so var=A^2/2=2.
    """
    fs = 1000.0
    nfft = 256
    f0 = 125.0
    amp = 2.0
    # Integer number of periods in nfft so the tone sits on a bin with no leakage.
    t = np.arange(nfft) / fs
    x = amp * np.cos(2.0 * np.pi * f0 * t)
    q = x[:, np.newaxis]

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="hann",
        window_norm="power",
        blockwise_mean=False,
    )

    peak_bin = int(round(f0 * nfft / fs))
    assert peak_bin == 32
    assert np.argmax(np.abs(q_hat[:, 0, 0])) == peak_bin

    ms = _onesided_mean_square(q_hat)
    total = float(np.sum(ms[:, 0]))
    variance = float(np.var(x))  # population var of pure tone = A^2/2
    # Bin-centred pure tone under power window: energy conserved to ~1e-15.
    np.testing.assert_allclose(total, variance, rtol=0, atol=1e-14)


@pytest.mark.oracle
def test_amplitude_norm_recovers_half_tone_amplitude():
    """With window_norm='amplitude', peak |q_hat| is half the cosine amplitude."""
    fs = 1000.0
    nfft = 256
    f0 = 125.0
    amp = 2.0
    t = np.arange(nfft) / fs
    x = amp * np.cos(2.0 * np.pi * f0 * t)
    q = x[:, np.newaxis]

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="hann",
        window_norm="amplitude",
        blockwise_mean=False,
    )

    peak_bin = 32
    peak_mag = float(np.abs(q_hat[peak_bin, 0, 0]))
    np.testing.assert_allclose(peak_mag, amp / 2.0, rtol=0, atol=1e-12)


@pytest.mark.oracle
def test_scipy_welch_broadband_mean_square_matches_psd_times_df():
    """One-sided mean-square spectrum equals scipy.signal.welch density * df.

    Independent of our own FFT path: scipy builds the density, and we check that
    mean |q_hat|^2 (inner bins doubled) equals PSD * (fs / nfft).

    The agreement is exact, not approximate. Both sides reduce to
    2 |FFT(w*x)|^2 / (nfft * sum(w^2)) once scipy's 1/(fs * sum(w^2)) density
    scaling is multiplied by df = fs/nfft, so any real difference in window,
    hop, or scaling shows up immediately. Measured worst deviation on this
    case, DC included, is 1.1e-15; the assertion leaves five orders of slack
    for platform FFT noise and no more.
    """
    rng = np.random.default_rng(0)
    fs = 1000.0
    nfft = 256
    novlap = nfft // 2
    # Long enough for several Welch blocks under 50% overlap.
    n_samples = nfft + 7 * (nfft - novlap)
    # Zero-mean so our global-mean subtraction matches scipy detrend=False.
    x = rng.standard_normal(n_samples)
    x = x - x.mean()
    q = x[:, np.newaxis]
    nblocks = 1 + (n_samples - nfft) // (nfft - novlap)

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=nblocks,
        novlap=novlap,
        window_type="hann",
        window_norm="power",
        blockwise_mean=False,
    )
    ms = _onesided_mean_square(q_hat)[:, 0]

    # scipy welch: density (power / Hz); floor partition, same hop and nfft.
    f_scipy, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nfft,
        noverlap=novlap,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    df = fs / nfft
    expected = psd * df

    # Same frequency grid.
    assert f_scipy.shape == ms.shape
    # Every bin, DC and Nyquist included. A loose bound here would let a
    # percent-level scaling error through, which is the error this test exists
    # to catch.
    np.testing.assert_allclose(ms, expected, rtol=1e-10, atol=0.0)


def test_power_norm_matches_manual_formula_on_boxcar():
    """q_hat = FFT(w*x) / (nfft * sqrt(mean(w^2))) with w=boxcar is FFT/nfft."""
    rng = np.random.default_rng(1)
    nfft = 64
    x = rng.standard_normal(nfft)
    x = x - x.mean()  # path always subtracts the long-time mean
    q = x[:, np.newaxis]
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="boxcar",
        window_norm="power",
        blockwise_mean=False,
    )
    # boxcar periodic window is all ones; power scale is 1.
    w = get_window("boxcar", nfft, fftbins=True)
    assert np.allclose(w, 1.0)
    manual = np.fft.fft(x)[: nfft // 2 + 1] / nfft
    np.testing.assert_allclose(q_hat[:, 0, 0], manual, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "window_type",
    ["hamming", "hann", "blackman", "bartlett", "sine"],
)
def test_named_window_matches_periodic_scipy_or_sine(window_type):
    """Each named window matches its definition (periodic scipy, or sine_window).

    Replaces the old serial/parallel window-identity cases: agreement between two
    names of the same function cannot fail, so pin the window against scipy
    (or the documented mid-bin sine formula) instead.
    """
    rng = np.random.default_rng(5)
    nfft = 32
    x = rng.standard_normal(nfft)
    x = x - x.mean()
    q = x[:, np.newaxis]
    got = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type=window_type,
        window_norm="power",
    )
    if window_type == "sine":
        # Documented mid-bin sine window: sin(pi * (i+0.5) / n)
        window = np.sin(np.pi * (np.arange(nfft) + 0.5) / nfft)
    else:
        window = get_window(window_type, nfft, fftbins=True)
    expected = _manual_windowed_rfft(q, window, "power")[:, 0]
    np.testing.assert_allclose(got[:, 0, 0], expected, rtol=0, atol=1e-12)


def test_default_window_is_periodic_hamming():
    """Omitting window_type must apply scipy's periodic Hamming (fftbins=True).

    The default lives on windowed_block_fft. The public wrappers re-declare
    window_type='hamming' and pass it through, so calling blocksfft() without
    the kwarg never reads the shared default — the oracle must call the shared
    function directly with window_type omitted.

    An independent numpy rFFT with get_window('hamming', ..., fftbins=True) is
    the definition; hann (the nearby look-alike default) must NOT match.
    """
    rng = np.random.default_rng(2)
    nfft = 64
    # Long-time zero mean so global-mean subtraction is a no-op on the formula.
    x = rng.standard_normal(nfft)
    x = x - x.mean()
    q = x[:, np.newaxis]

    # No window_type kwarg — pins the default argument on the shared function.
    got = windowed_block_fft(q, nfft=nfft, nblocks=1, novlap=0, window_norm="power")

    w_hamming = get_window("hamming", nfft, fftbins=True)
    expected = _manual_windowed_rfft(q, w_hamming, "power")[:, 0]
    np.testing.assert_allclose(
        got[:, 0, 0],
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="default window is not periodic hamming",
    )

    w_hann = get_window("hann", nfft, fftbins=True)
    hann_expected = _manual_windowed_rfft(q, w_hann, "power")[:, 0]
    assert not np.allclose(expected, hann_expected, atol=1e-12), (
        "hamming and hann oracles collapsed — test cannot distinguish the default"
    )
    assert not np.allclose(got[:, 0, 0], hann_expected, atol=1e-12), "default window matched hann; expected hamming"


def test_normvar_divides_by_unbiased_sample_variance():
    """normvar applies the documented divide-by-sample-variance (ddof=1) step.

    Product definition (blocksfft docstring, spod_matlab opts.normvar, PySPOD
    normalize_data): pointwise divide each centred block by its unbiased
    variance. That is intentionally NOT unit-variance normalisation (divide by
    std): scaling the input by c scales the FFT by 1/c.

    Independent oracle: numpy rFFT of (block - mean) / var(ddof=1) under a
    boxcar. A population-variance (ddof=0) oracle must disagree so the ddof is
    pinned rather than only the 1/c scale relation.
    """
    rng = np.random.default_rng(3)
    nfft = 32
    nmesh = 2
    block = rng.standard_normal((nfft, nmesh))
    # Nonzero mean so centering is real work; single block, global mean = block mean.
    block = block + np.array([0.4, -0.7])
    q = block.copy()

    got = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        normvar=True,
        window_type="boxcar",
        window_norm="power",
        blockwise_mean=False,
    )

    centered = block - block.mean(axis=0)
    var_sample = np.var(centered, axis=0, ddof=1)
    var_population = np.var(centered, axis=0, ddof=0)
    assert np.all(var_sample > var_population), "ddof must move the variance"

    scaled_sample = centered / var_sample
    scaled_population = centered / var_population
    # boxcar + power → scale factor 1, q_hat = rFFT / nfft
    expected_sample = np.fft.fft(scaled_sample, axis=0)[: nfft // 2 + 1] / nfft
    expected_population = np.fft.fft(scaled_population, axis=0)[: nfft // 2 + 1] / nfft

    # Sanity: population oracle really is a different number, not a tol artefact.
    rel = np.max(np.abs(expected_sample - expected_population)) / np.max(np.abs(expected_sample))
    assert rel > 1e-3, f"sample vs population oracles too close (rel={rel:.3e})"

    if not np.allclose(got[:, :, 0], expected_sample, rtol=0, atol=1e-12):
        if np.allclose(got[:, :, 0], expected_population, rtol=0, atol=1e-12):
            raise AssertionError(
                "normvar used population variance (ddof=0); documented definition is unbiased sample variance (ddof=1)"
            )
        np.testing.assert_allclose(got[:, :, 0], expected_sample, rtol=0, atol=1e-12)
    assert not np.allclose(got[:, :, 0], expected_population, atol=1e-12), (
        "normvar matched population variance (ddof=0); documented definition is ddof=1"
    )


def test_overlapped_blocks_share_the_overlap_region():
    """Welch hop is nfft - novlap: a sample in the overlap belongs to two blocks.

    Definition: block k covers samples [k*hop, k*hop + nfft) with
    hop = nfft - novlap. For novlap > 0, index ``hop`` lies inside block 0 and
    is the first sample of block 1. An impulse there must therefore light up
    both blocks. If hop were silently nfft (overlap ignored), block 1 would
    start at nfft and miss the impulse.
    """
    nfft = 16
    novlap = 4
    hop = nfft - novlap  # definition under test
    assert hop == 12
    nblocks = 3
    # Long enough that even a defective hop=nfft still indexes in-bounds, so the
    # failure mode is a wrong energy pattern (missed overlap), not an IndexError.
    Ns = (nblocks - 1) * nfft + nfft  # 48
    assert Ns == 48

    impulse_at = hop  # first sample of block 1 under true hop; still inside block 0
    q = np.zeros((Ns, 1))
    q[impulse_at, 0] = 1.0

    # boxcar so the impulse is not window-attenuated at the block edge.
    # blockwise_mean=True so a lone impulse does not leak into empty blocks
    # through global-mean subtraction (the long-time mean is 1/Ns everywhere).
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=nblocks,
        novlap=novlap,
        window_type="boxcar",
        window_norm="power",
        blockwise_mean=True,
    )
    # Energy per block (sum over frequency of |coeff|^2).
    energy = np.sum(np.abs(q_hat[:, 0, :]) ** 2, axis=0)

    assert energy[0] > 0.0, "impulse at hop must fall inside block 0"
    assert energy[1] > 0.0, (
        "impulse at hop must fall inside block 1 (overlap); "
        "energy[1]==0 means hop ignored novlap (block 1 started at nfft)"
    )
    assert energy[2] == 0.0, "impulse at hop must not reach block 2"

    # Negative control of the definition: under hop=nfft the second block
    # would start at 16 and miss index 12 — so requiring energy[1] > 0 is
    # exactly the overlap constraint.
    starts = [iblk * hop for iblk in range(nblocks)]
    assert starts == [0, 12, 24]
    assert starts[0] <= impulse_at < starts[0] + nfft
    assert starts[1] <= impulse_at < starts[1] + nfft
    assert not (starts[2] <= impulse_at < starts[2] + nfft)


@pytest.mark.parametrize("window_type", ["hamming", "hann", "boxcar"])
@pytest.mark.parametrize("window_norm", ["power", "amplitude"], ids=["power", "amplitude"])
@pytest.mark.parametrize("blockwise_mean", [False, True], ids=["bwmean0", "bwmean1"])
@pytest.mark.parametrize("normvar", [False, True], ids=["nv0", "nv1"])
@pytest.mark.parametrize(
    "novlap,nfft,Ns,nblocks",
    [
        (0, 16, 64, 3),
        (4, 16, 70, 5),  # hop=12; Ns not a multiple of hop
    ],
    ids=["ovl0", "ovl4-uneven"],
)
@pytest.mark.characterization
def test_param_surface_matches_blockwise_definition(
    window_type, window_norm, blockwise_mean, normvar, novlap, nfft, Ns, nblocks
):
    """Parameter surface vs the Welch block-FFT definition (not wrapper equality).

    Covers the same cross product the old serial/parallel sweep claimed:
    window in {hamming, hann} × window_norm in {power, amplitude} ×
    blockwise_mean × normvar × two placements (no overlap; 25% overlap on an
    uneven-length record). Expected values come from the documented formula
    applied block-by-block with numpy.fft — hop = nfft - novlap, periodic
    scipy windows, power/amplitude scale, optional per-block mean and
    ddof=1 variance divide.
    """
    rng = np.random.default_rng(4)
    q = rng.standard_normal((Ns, 3))
    got = blocksfft(
        q,
        nfft=nfft,
        nblocks=nblocks,
        novlap=novlap,
        window_type=window_type,
        window_norm=window_norm,
        blockwise_mean=blockwise_mean,
        normvar=normvar,
    )
    window = get_window(window_type, nfft, fftbins=True)
    if window_norm == "amplitude":
        scale = float(np.mean(window))
    else:
        scale = float(np.sqrt(np.mean(window**2)))
    hop = nfft - novlap
    q_mean = np.mean(q, axis=0)
    expected = np.zeros_like(got)
    for iblk in range(nblocks):
        ts = iblk * hop
        block = q[ts : ts + nfft]
        mean = np.mean(block, axis=0) if blockwise_mean else q_mean
        centered = block - mean
        if normvar:
            block_var = np.var(centered, axis=0, ddof=1)
            block_var = np.where(
                block_var < 4 * np.finfo(float).eps,
                1.0,
                block_var,
            )
            centered = centered / block_var
        windowed = centered * window[:, np.newaxis]
        full = np.fft.fft(windowed, axis=0)
        expected[:, :, iblk] = (1.0 / (nfft * scale)) * full[: nfft // 2 + 1]

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Definition-level oracles extended across the flag/window surface, closing
# earlier gaps: blackman / bartlett / sine were oracled only at single-block
# power norm, and bartlett multi-block was never exercised.
# ---------------------------------------------------------------------------

_ALL_WINDOWS = ["hamming", "hann", "boxcar", "blackman", "bartlett", "sine"]


def _bin_centred_tone(nfft, fs=1000.0, f0=125.0, amp=2.0):
    """Pure cosine with an integer number of periods in nfft (no leakage)."""
    t = np.arange(nfft) / fs
    return amp * np.cos(2.0 * np.pi * f0 * t)


@pytest.mark.oracle
@pytest.mark.parametrize("normvar", [False, True], ids=["nv0", "nv1"])
@pytest.mark.parametrize("blockwise_mean", [False, True], ids=["bwmean0", "bwmean1"])
@pytest.mark.parametrize("window_type", _ALL_WINDOWS)
def test_parseval_power_norm_across_windows_and_flags(window_type, blockwise_mean, normvar):
    """Power-norm mean-square spectrum sums to the signal variance.

    For a bin-centred tone the window's coherent gain cancels against the
    power scaling for every window in the set, with any blockwise-mean flag
    (the tone's block mean is zero over full periods), and across blocks.
    With normvar the divide is by the unbiased variance A^2/2, so the
    recovered mean-square is var / var^2 = 1/var.
    """
    nfft = 256
    amp = 2.0
    x = _bin_centred_tone(nfft, amp=amp)
    q = np.tile(x, 4)[:, np.newaxis]  # 4 identical blocks stacked in time, novlap=0
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=4,
        novlap=0,
        window_type=window_type,
        window_norm="power",
        blockwise_mean=blockwise_mean,
        normvar=normvar,
    )
    total = float(np.sum(_onesided_mean_square(q_hat)[:, 0]))
    # The normvar divide is by the block's ddof=1 sample variance, so the
    # normalized signal's population variance is var_pop / sample_var. Both
    # quantities are measured here from the input itself.
    var_pop = amp**2 / 2.0
    sample_var = float(np.var(x, ddof=1))
    # normvar divides amplitudes by the VARIANCE, so mean-square scales as
    # 1/sample_var^2. rtol is 1e-3, not 1e-12: periodic bartlett is asymmetric
    # (zero first sample), leaking O(1e-4) of the tone into the +-1 bins; every
    # symmetric window here matches to ~1e-12.
    expected = var_pop / sample_var**2 if normvar else var_pop
    np.testing.assert_allclose(total, expected, rtol=1e-3, atol=1e-14)


@pytest.mark.oracle
@pytest.mark.parametrize("normvar", [False, True], ids=["nv0", "nv1"])
@pytest.mark.parametrize("window_type", _ALL_WINDOWS)
def test_amplitude_norm_peak_recovers_half_amplitude_across_windows(window_type, normvar):
    """Amplitude-norm peak |q_hat| is A/2 (scaled by 1/var when normvar)."""
    nfft = 256
    amp = 2.0
    x = _bin_centred_tone(nfft, amp=amp)
    q = x[:, np.newaxis]
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type=window_type,
        window_norm="amplitude",
        blockwise_mean=False,
        normvar=normvar,
    )
    peak = float(np.abs(q_hat[32, 0, 0]))
    sample_var = float(np.var(x, ddof=1))
    # rtol 1e-3: the module's sine window uses mid-bin placement, so its
    # coherent gain at an exact bin differs from scipy's periodic sine by
    # O(1e-5); every scipy window matches to ~1e-12.
    expected = amp / 2.0 / sample_var if normvar else amp / 2.0
    np.testing.assert_allclose(peak, expected, rtol=1e-3, atol=1e-12)


@pytest.mark.oracle
def test_bartlett_multiblock_average_matches_single_block():
    """Bartlett multi-block is the single
    block's periodogram, averaged.

    With novlap=0 and an integer number of periods per block every Bartlett
    segment is identical, so per-block spectra must agree exactly and their
    mean must reproduce the one-block Parseval value.
    """
    nfft, nblocks = 128, 5
    amp = 2.0
    x = _bin_centred_tone(nfft, amp=amp)
    q = np.tile(x, nblocks)[:, np.newaxis]
    q_hat = blocksfft(q, nfft=nfft, nblocks=nblocks, novlap=0, window_type="bartlett", window_norm="power")
    per_block = np.abs(q_hat[:, 0, :]) ** 2
    np.testing.assert_allclose(per_block, np.broadcast_to(per_block[:, [0]], per_block.shape), rtol=1e-12, atol=0)
    total = float(np.sum(_onesided_mean_square(q_hat)[:, 0]))
    # Same asymmetric-bartlett leakage tolerance as the sweep above.
    np.testing.assert_allclose(total, amp**2 / 2.0, rtol=1e-3, atol=1e-14)


@pytest.mark.oracle
def test_normvar_clamp_sits_between_roundoff_and_meaningful_variances():
    """The zero-variance clamp must not fire on real (tiny) variance.

    Choose a tone whose unbiased sample variance (~1e-15) sits just above the
    clamp region but far below anything a larger clamp constant would spare.
    The expected peak amplitude is derived from the amplitude-recovery
    identity divided by the tone's variance — no clamp constant is copied
    from the source. If the clamp threshold were two orders of magnitude
    larger, this variance would be clamped to 1.0 and the peak would come out
    A/2 instead of A/(2 var), failing the comparison.
    """
    nfft = 256
    var = 1.0e-15
    amp = float(np.sqrt(2.0 * var))  # pure cosine: var = A^2 / 2
    x = _bin_centred_tone(nfft, amp=amp)
    q = x[:, np.newaxis]
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="hann",
        window_norm="amplitude",
        blockwise_mean=False,
        normvar=True,
    )
    peak = float(np.abs(q_hat[32, 0, 0]))
    np.testing.assert_allclose(peak, amp / 2.0 / np.var(x, ddof=1), rtol=1e-6)


@pytest.mark.oracle
@pytest.mark.parametrize("offset", [0.0, 25.0])
def test_blockwise_mean_output_invariant_to_constant_offset(offset):
    """blockwise_mean must remove the offset BEFORE windowing.

    Metamorphic pin on operation order: adding a constant to the signal must
    not change the block-mean-removed output at all (the per-block mean
    absorbs it exactly). If the implementation windowed first and centred
    second, the residual offset * (window - mean(window)) would inject
    offset-scaled energy into every non-DC bin and the comparison would fail.
    """
    nfft = 64
    base = _bin_centred_tone(nfft, f0=250.0, amp=1.0) + 0.3  # non-integer periods
    q0 = (base)[:, np.newaxis]
    qC = (base + offset)[:, np.newaxis]
    kwargs = dict(nfft=nfft, nblocks=1, novlap=0, window_type="hamming", window_norm="power", blockwise_mean=True)
    np.testing.assert_allclose(blocksfft(qC, **kwargs), blocksfft(q0, **kwargs), rtol=0, atol=1e-12)
