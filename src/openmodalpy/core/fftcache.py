"""The stamp that decides whether a saved FFT block cache can be reused.

SPOD and BSMD spend most of a run inside the block FFT, so the blocks are
written next to the results and read again on a later run. They may only be
read again when the run that wrote them asked the same question of the same
data. These functions write the parameters and a digest of the snapshots into
the file, and check them before anything is reused.

The digest covers the content, not the file name: two files with the same name
and different data must not be confused, and the same data under a new name
must not force a recompute.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from openmodalpy.core.base import BaseAnalyzer

logger = logging.getLogger(__name__)


_QHAT_STAMP_ATTR_PREFIX = "_fftcache_"


def _qhat_content_digest(q: np.ndarray) -> str:
    """Return a blake2b digest of ``q``'s raw bytes (plus shape/dtype).

    A full hash is O(n), i.e. cheaper than the O(n log n) FFT it lets us
    avoid recomputing, so hashing the exact content is affordable here and a
    sampled/strided checksum is not worth the false-negative risk of two
    different arrays colliding.
    """
    arr = np.ascontiguousarray(q)
    # ``arr.data`` is a memoryview onto the array's own buffer, so hashing it copies
    # nothing. ``tobytes()`` would duplicate the whole snapshot matrix just to hash it.
    h = hashlib.blake2b(arr.data.cast("B"), digest_size=16)
    h.update(str(arr.shape).encode())
    h.update(arr.dtype.str.encode())
    return h.hexdigest()


def _qhat_cache_stamp(analyzer: BaseAnalyzer, q: np.ndarray) -> dict[str, str | float | int | bool]:
    """Return the parameters that determine the FFT blocks produced for ``q``.

    Note: ``spatial_weight_type`` is deliberately excluded. ``blocksfft`` (see
    below) only ever receives ``q``, ``nfft``, ``nblocks``, ``novlap``,
    ``blockwise_mean``, ``normvar``, ``window_norm`` and ``window_type`` — the
    spatial weights are applied later, in the SPOD/BSMD eigenproblem, never in
    the FFT block computation. So it cannot affect ``qhat`` and does not need
    to be stamped.
    """
    return {
        "window_type": str(getattr(analyzer, "window_type", "hamming")),
        "window_norm": str(getattr(analyzer, "window_norm", "power")),
        "overlap": float(analyzer.overlap),
        "nfft": int(analyzer.nfft),
        "blockwise_mean": bool(getattr(analyzer, "blockwise_mean", False)),
        "normvar": bool(getattr(analyzer, "normvar", False)),
        "q_digest": _qhat_content_digest(q),
    }


def _write_qhat_stamp(h5file: h5py.File, analyzer: BaseAnalyzer, q: np.ndarray) -> None:
    """Stamp the parameters that produced ``qhat`` into ``h5file``'s attrs."""
    for key, value in _qhat_cache_stamp(analyzer, q).items():
        h5file.attrs[f"{_QHAT_STAMP_ATTR_PREFIX}{key}"] = value


def _verify_qhat_stamp(h5file: h5py.File, analyzer: BaseAnalyzer, q: np.ndarray) -> bool:
    """Return whether ``h5file``'s stamped FFT parameters match ``analyzer``/``q``.

    On any mismatch — or an absent stamp (e.g. a cache file from an older
    build) — this returns False rather than raising. That is the opposite
    policy from result files (which must raise on staleness): FFT blocks are
    cheaply re-derivable from the raw data, so silently recomputing them is
    correct and non-destructive. Do not "harmonise" this with the stricter
    policy used for saved results/modes.
    """
    expected = _qhat_cache_stamp(analyzer, q)
    for key, exp_value in expected.items():
        attr_name = f"{_QHAT_STAMP_ATTR_PREFIX}{key}"
        if attr_name not in h5file.attrs:
            logger.warning("FFT cache stamp missing '%s' (older cache file) — recomputing FFT blocks.", key)
            return False
        actual = h5file.attrs[attr_name]
        if isinstance(exp_value, bool):
            actual = bool(actual)
        elif isinstance(exp_value, int):
            actual = int(actual)
        elif isinstance(exp_value, float):
            actual = float(actual)
        else:
            actual = str(actual)
        if actual != exp_value:
            logger.warning(
                "FFT cache stamp mismatch on '%s': cached=%r != current=%r — recomputing FFT blocks.",
                key,
                actual,
                exp_value,
            )
            return False
    return True
