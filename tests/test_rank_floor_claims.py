"""What the numerical-rank floor claims, stated near the threshold.

Both solver routes drop a mode whose eigenvalue or singular value sits below
``n_kernel * eps * peak``. Three numbers set that floor: the size of the matrix
that was factored, the machine epsilon of the working precision, and the
largest value in the spectrum. These tests put values on each side of the
resulting cutoff and state what must happen there.

The float32 cases are the reason this file exists. On float64 both branches of
``_working_eps`` return the same number, so no float64 test can tell a correct
implementation from one that ignores the dtype. On float32 the two answers are
5.4e8 apart, and taking the wrong one would keep rounding noise as a mode.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from openmodalpy.core.decomposition import (
    _relative_floor,
    _significant_eigenvalue_mask,
    _significant_singular_value_mask,
    _working_eps,
)

FLOAT32_EPS = float(np.finfo(np.float32).eps)  # 1.1920929e-07
FLOAT64_EPS = float(np.finfo(np.float64).eps)  # 2.220446049250313e-16


def test_the_epsilon_follows_the_working_precision() -> None:
    """A float32 spectrum must get the float32 epsilon, not the float64 one.

    They differ by a factor of 5.4e8. Using the smaller one on float32 data
    puts the floor far below the rounding level of that data, so noise is kept
    as a mode.
    """
    assert _working_eps(np.float32) == FLOAT32_EPS
    assert _working_eps(np.float64) == FLOAT64_EPS
    assert _working_eps(np.dtype("float32")) == FLOAT32_EPS
    assert FLOAT32_EPS / FLOAT64_EPS > 1e8


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.int64, np.bool_])
def test_a_dtype_with_no_epsilon_falls_back_to_float64(dtype: type) -> None:
    """Only a real float dtype has its own epsilon here; the rest use float64.

    The masks take the real part first, so the dtype reaching this function is
    normally real. The fallback covers the rest and must be the float64 value.
    """
    assert _working_eps(dtype) == FLOAT64_EPS


def test_the_floor_is_the_product_of_its_three_terms() -> None:
    """``n_kernel * eps * peak``, with every term doing its part."""
    assert _relative_floor(1.0, 1, np.float64) == pytest.approx(FLOAT64_EPS)
    assert _relative_floor(1.0, 100, np.float64) == pytest.approx(100 * FLOAT64_EPS)
    assert _relative_floor(7.0, 100, np.float64) == pytest.approx(700 * FLOAT64_EPS)
    # The dtype is not decoration: the same peak and size give a far larger
    # floor in single precision.
    assert _relative_floor(1.0, 100, np.float32) == pytest.approx(100 * FLOAT32_EPS)


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
def test_the_cutoff_keeps_above_and_drops_below(mask: Callable[[np.ndarray, int], np.ndarray]) -> None:
    """A value above the cutoff is kept and one below it is dropped.

    Both are placed a factor of two from the cutoff, so this measures the
    cutoff itself and not a value far away from it.
    """
    n_kernel = 64
    peak = 3.0
    cutoff = _relative_floor(peak, n_kernel, np.dtype(np.float64))

    values = np.array([peak, 2.0 * cutoff, 0.5 * cutoff])
    assert list(mask(values, n_kernel)) == [True, True, False]


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
def test_a_value_exactly_at_the_cutoff_is_dropped(mask: Callable[[np.ndarray, int], np.ndarray]) -> None:
    """The comparison is strict, so a value equal to the cutoff goes."""
    n_kernel = 32
    peak = 1.0
    cutoff = _relative_floor(peak, n_kernel, np.dtype(np.float64))

    values = np.array([peak, cutoff])
    assert list(mask(values, n_kernel)) == [True, False]


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
def test_the_kernel_size_moves_the_cutoff(mask: Callable[[np.ndarray, int], np.ndarray]) -> None:
    """A larger factored matrix raises the floor and drops more modes.

    ``n_kernel`` is the dimension of the Gram matrix. A bigger matrix
    accumulates more rounding, so the floor rises with it.
    """
    peak = 1.0
    value = 50.0 * FLOAT64_EPS
    values = np.array([peak, value])

    assert list(mask(values, 10)) == [True, True]
    assert list(mask(values, 1000)) == [True, False]


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
def test_the_cutoff_follows_the_working_precision(mask: Callable[[np.ndarray, int], np.ndarray]) -> None:
    """The same spectrum in float32 loses the mode that float64 keeps.

    A value at 1e-10 of the peak is far above the float64 floor and far below
    the float32 one. This is the claim that no float64 test can make.
    """
    n_kernel = 100
    values64 = np.array([1.0, 1e-10], dtype=np.float64)
    values32 = values64.astype(np.float32)

    assert list(mask(values64, n_kernel)) == [True, True]
    assert list(mask(values32, n_kernel)) == [True, False]


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
def test_an_empty_spectrum_gives_an_empty_mask(mask: Callable[[np.ndarray, int], np.ndarray]) -> None:
    """No values in, no values out, and the result stays boolean."""
    result = mask(np.array([]), 10)
    assert result.shape == (0,)
    assert result.dtype == bool


@pytest.mark.parametrize(
    "mask",
    [_significant_eigenvalue_mask, _significant_singular_value_mask],
    ids=["eigenvalue", "singular_value"],
)
@pytest.mark.parametrize("peak", [0.0, -1.0, np.nan, np.inf])
def test_a_spectrum_with_no_usable_peak_keeps_nothing(
    mask: Callable[[np.ndarray, int], np.ndarray], peak: float
) -> None:
    """A peak that is zero, negative or not finite makes the floor meaningless.

    There is no scale to measure against, so nothing is significant. Keeping a
    mode here would rank it by a number the floor cannot judge.
    """
    values = np.array([peak, peak / 2.0 if np.isfinite(peak) else peak])
    assert not mask(values, 10).any()


def test_the_two_routes_differ_by_the_squaring_of_the_gram_matrix() -> None:
    """The SVD route keeps a mode the eigh route cannot see.

    The eigh route factors the Gram matrix, whose conditioning is squared, so
    its floor lives in the eigenvalue domain. A mode at singular-value ratio
    1e-10 sits at eigenvalue ratio 1e-20. That is below the eigh floor of
    n_kernel * eps and far above the squared floor the SVD route applies. The
    docstrings state this; here is the number.
    """
    n_kernel = 100
    sigma = np.array([1.0, 1e-10])
    lam = sigma**2  # the eigenvalues the eigh route would see

    assert list(_significant_singular_value_mask(sigma, n_kernel)) == [True, True]
    assert list(_significant_eigenvalue_mask(lam, n_kernel)) == [True, False]
