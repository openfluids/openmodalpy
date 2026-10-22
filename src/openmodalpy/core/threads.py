"""Process-wide BLAS/OpenMP thread policy for OpenModalPy.

One place decides how many threads the underlying BLAS/OpenMP pools use inside
``svd`` / ``eigh`` / ``eig``. The default is 1 (reproducible for a fixed
environment). Speed is opt-in via :func:`set_blas_threads` or the env var
``OPENMODALPY_BLAS_THREADS``. ``0`` means this package applies no limit —
an existing ``OMP_NUM_THREADS`` or an outer ``threadpoolctl`` limiter still
applies.

Cross-vendor bit-identity is not promised: OpenBLAS vs MKL vs Accelerate can
still differ under the same thread count.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from threadpoolctl import threadpool_limits

_ENV = "OPENMODALPY_BLAS_THREADS"
_DEFAULT = 1

# ``None`` until first read — then holds the active limit (int >= 0).
_active: int | None = None


def _parse_env() -> int:
    raw = os.environ.get(_ENV)
    if raw is None or raw == "":
        return _DEFAULT
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT
    if n < 0:
        return _DEFAULT
    return n


def get_blas_threads() -> int:
    """Return the active BLAS thread limit (``0`` = no limit from this package).

    The env var is parsed lazily on the first call (or after a reset), not at
    import time.
    """
    global _active
    if _active is None:
        _active = _parse_env()
    return _active


def set_blas_threads(n: int) -> None:
    """Set the process-wide BLAS thread limit.

    Parameters
    ----------
    n
        Positive integer for a hard limit, or ``0`` so this package applies no
        limit (outer env / limiters still apply).
    """
    global _active
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"BLAS thread limit must be an int >= 0, got {n!r}")
    _active = n


@contextmanager
def blas_threads(n: int) -> Iterator[None]:
    """Temporarily set the BLAS thread limit; restore the previous value on exit."""
    previous = get_blas_threads()
    set_blas_threads(n)
    try:
        yield
    finally:
        set_blas_threads(previous)


@contextmanager
def apply_blas_limit() -> Iterator[None]:
    """Apply the active policy to BLAS/OpenMP pools for the duration.

    When the policy is ``0``, the pools are left alone (no limiter entered).
    Nested use is safe on one thread: each entry re-applies the current
    policy. Concurrent workers must not enter/exit this context themselves —
    pin once with an enclosing limiter around the whole pool instead
    (``threadpool_limits`` is process-global with no per-thread isolation).
    """
    n = get_blas_threads()
    if n == 0:
        yield
    else:
        with threadpool_limits(limits=n):
            yield


def get_num_threads() -> int:
    """Return thread count from ``OMP_NUM_THREADS`` or ``os.cpu_count()``."""
    env = os.environ.get("OMP_NUM_THREADS")
    try:
        val = int(env) if env is not None else None
    except (TypeError, ValueError):
        val = None
    if val is not None and val > 0:
        return val
    cpu = os.cpu_count() or 1
    return cpu
