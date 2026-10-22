"""Unit tests for the process-wide BLAS thread policy (core.threads)."""

from __future__ import annotations

import pytest

from openmodalpy.core import threads as thr


@pytest.fixture(autouse=True)
def _isolate_policy(monkeypatch):
    """Each test starts with a clean policy module state (no env, unset active)."""
    monkeypatch.delenv(thr._ENV, raising=False)
    monkeypatch.setattr(thr, "_active", None)
    yield
    # Leave nothing sticky for later tests that import openmodalpy at module level.
    thr._active = None


def test_default_is_one_with_no_env() -> None:
    """With no OPENMODALPY_BLAS_THREADS, the default limit is 1."""
    assert thr.get_blas_threads() == 1


def test_invalid_env_falls_back_to_one(monkeypatch) -> None:
    """Non-integer or empty env values fall back to 1."""
    for raw in ("not-a-number", "", "1.5", "abc"):
        monkeypatch.setenv(thr._ENV, raw)
        monkeypatch.setattr(thr, "_active", None)
        assert thr.get_blas_threads() == 1, f"env={raw!r} should fall back to 1"


def test_negative_env_falls_back_to_one(monkeypatch) -> None:
    """A negative env value is rejected and falls back to 1."""
    monkeypatch.setenv(thr._ENV, "-3")
    monkeypatch.setattr(thr, "_active", None)
    assert thr.get_blas_threads() == 1


def test_blas_threads_restores_previous() -> None:
    """blas_threads(n) restores the previous value on normal exit."""
    thr.set_blas_threads(1)
    with thr.blas_threads(4):
        assert thr.get_blas_threads() == 4
    assert thr.get_blas_threads() == 1


def test_blas_threads_restores_when_body_raises() -> None:
    """blas_threads(n) restores the previous value even when the body raises."""
    thr.set_blas_threads(2)
    with pytest.raises(RuntimeError, match="boom"):
        with thr.blas_threads(8):
            assert thr.get_blas_threads() == 8
            raise RuntimeError("boom")
    assert thr.get_blas_threads() == 2


def test_set_blas_threads_rejects_negative() -> None:
    """set_blas_threads rejects a negative limit."""
    with pytest.raises(ValueError, match="int >= 0"):
        thr.set_blas_threads(-1)


def test_set_blas_threads_rejects_non_int() -> None:
    """set_blas_threads rejects non-int types (including bool)."""
    with pytest.raises(ValueError, match="int >= 0"):
        thr.set_blas_threads(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int >= 0"):
        thr.set_blas_threads(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int >= 0"):
        thr.set_blas_threads("4")  # type: ignore[arg-type]


def test_zero_applies_no_limiter(monkeypatch) -> None:
    """Policy 0 must not enter threadpool_limits (no limit from this package)."""
    entered = []

    class _Sentinel:
        def __enter__(self):
            entered.append(True)
            return self

        def __exit__(self, *args):
            return False

    def _boom_limits(*args, **kwargs):
        entered.append(True)
        return _Sentinel()

    monkeypatch.setattr(thr, "threadpool_limits", _boom_limits)
    thr.set_blas_threads(0)
    with thr.apply_blas_limit():
        pass
    assert entered == [], "apply_blas_limit under policy 0 must not call threadpool_limits"


def test_nonzero_enters_limiter(monkeypatch) -> None:
    """A positive policy must pass that limit into threadpool_limits."""
    seen: list[int] = []

    class _Sentinel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _capture(limits=None, **kwargs):
        seen.append(limits)
        return _Sentinel()

    monkeypatch.setattr(thr, "threadpool_limits", _capture)
    thr.set_blas_threads(3)
    with thr.apply_blas_limit():
        pass
    assert seen == [3]
