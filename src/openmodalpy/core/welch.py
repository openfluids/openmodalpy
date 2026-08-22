"""Welch block helpers and the single windowed-block FFT implementation.

Shared by the serial path in ``base.blocksfft`` and the parallel-module entry
point ``parallel.blocksfft_optimized``. Both names remain public; both
delegate here so the window, hop, and scaling live in one place.

Dependencies are intentional: ``numpy``, ``scipy.signal.get_window``, and
``fftkit``. That is the ceiling for this module — not the optional parallel
stack (``threadpoolctl`` and friends). ``base.py`` must stay importable when
that stack fails to load; this module does not import the parallel package.
"""

import numpy as np
from fftkit import get_fft_func
from scipy.signal import get_window


def welch_nblocks(Ns: int, nfft: int, novlap: int) -> int:
    """Number of full Welch blocks under floor partitioning.

    Matches ``scipy.signal.welch``: hop = nfft - novlap, drop the remainder.
    Returns 0 when fewer than one full block fits (including hop <= 0).
    """
    hop = nfft - novlap
    if hop <= 0 or Ns < nfft:
        return 0
    return (int(Ns) - int(novlap)) // hop


def _validate_welch_blocks(Ns: int, nfft: int, nblocks: int, novlap: int) -> None:
    """Reject short records and over-subscribed Welch partitions.

    Matches ``scipy.signal.welch``: floor partitioning, drop the remainder.
    Never clamps the final block start — a clamped block is not an independent
    ensemble member and biases SPOD/BSMD eigenvalues.

    ``novlap >= nfft`` (hop <= 0) is rejected: every block would start at the
    same index and the ensemble would be three copies of one periodogram.
    """
    hop = nfft - novlap
    if hop <= 0:
        raise ValueError(f"Invalid Welch hop: nfft={nfft}, novlap={novlap} (hop={hop} <= 0); novlap must be < nfft")
    if Ns < nfft or nblocks < 1:
        raise ValueError(
            f"Cannot form Welch blocks: Ns={Ns}, nfft={nfft} "
            f"(novlap={novlap}) yield fewer than one full block "
            f"(requested nblocks={nblocks})"
        )
    needed = (nblocks - 1) * hop + nfft
    if needed > Ns:
        raise ValueError(
            f"Requested nblocks={nblocks} does not fit in Ns={Ns} with "
            f"nfft={nfft}, novlap={novlap} (need {needed} samples)"
        )


def sine_window(n: int) -> np.ndarray:
    """Return a sine window of length n (periodic-style mid-bin placement)."""
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


def windowed_block_fft(
    q: np.ndarray,
    nfft: int,
    nblocks: int,
    novlap: int,
    blockwise_mean: bool = False,
    normvar: bool = False,
    window_norm: str = "power",
    window_type: str = "hamming",
) -> np.ndarray:
    """Windowed block FFT for Welch CSD estimation.

    Returns complex coefficients shaped ``[freq, space, block]`` (one-sided).
    Scaling is ``q_hat = FFT(w * x) / (nfft * scale)`` where ``scale`` is
    ``sqrt(mean(w^2))`` for ``window_norm='power'`` and ``mean(w)`` for
    ``'amplitude'``. Window is periodic (scipy ``get_window`` with
    ``fftbins=True``), except ``window_type='sine'`` which uses
    :func:`sine_window`.

    Block starts are ``iblk * (nfft - novlap)`` with no end-of-record clamp.
    Callers must pass an ``nblocks`` that fits; oversize requests raise
    ``ValueError``.
    """
    _validate_welch_blocks(q.shape[0], nfft, nblocks, novlap)

    if window_type == "sine":
        window = sine_window(nfft)
    else:
        window = get_window(window_type, nfft, fftbins=True)

    if window_norm == "amplitude":
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))

    nmesh = q.shape[1]
    n_freq_out = nfft // 2 + 1
    q_hat = np.zeros((n_freq_out, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0)
    window_broadcast = window[:, np.newaxis]

    fft_func = get_fft_func()
    hop = nfft - novlap
    for iblk in range(nblocks):
        ts = iblk * hop
        tf = np.arange(ts, ts + nfft)
        block = q[tf, :]

        if blockwise_mean:
            block_mean = np.mean(block, axis=0)
        else:
            block_mean = q_mean
        block_centered = block - block_mean

        if normvar:
            # Two-pass variance (mean first, then squared deviations) of an
            # exactly constant block is exactly 0, and a division by it is
            # impossible. For a NEARLY constant block every deviation is a
            # handful of rounding steps on O(1)-scale samples, so the computed
            # variance sits within a few eps of zero; dividing by such a value
            # amplifies pure round-off to O(1) amplitudes. 4 * eps is two
            # roundings of headroom above a single-precision artifact — the
            # smallest round multiple that cannot be reached by honest
            # round-off of one subtraction — while staying 10+ orders below
            # any variance worth normalizing. Caveat: this is an absolute
            # threshold for O(1)-scale data; data at scale S carries
            # round-off-scale variances near eps*S^2, which for S >> 1 will
            # not clamp (and does not need to: their relative variation is
            # already resolved).
            block_var = np.var(block_centered, axis=0, ddof=1)
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0
            block_centered = block_centered / block_var

        full_fft_result = fft_func(block_centered * window_broadcast, axis=0)
        q_hat[:, :, iblk] = (cw / nfft) * full_fft_result[:n_freq_out, :]

    return q_hat
