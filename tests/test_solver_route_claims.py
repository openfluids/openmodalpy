"""What the two second-order solver routes promise each other.

``weighted_second_order`` factors either the temporal kernel or the spatial
one, whichever is smaller, and it can go through eigh or through an SVD. The
answer is a property of the data, not of the route. These tests state that,
and they state what comes back when the data carries no significant mode at
all.

Measured route agreement, square problem (12 x 12, rank 4, non-uniform
weights, float64): the temporal and spatial eigh branches agree to 4.0e-15
relative on the modes, 6.6e-16 on the eigenvalues and 7.4e-16 on the
coefficients. The tolerances below are set from those numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.decomposition import weighted_second_order

ROUTES = ["eigh", "svd"]


def _rank_deficient(n_samples: int, n_space: int, rank: int, seed: int = 0) -> np.ndarray:
    """Return data of exactly ``rank``, so the mode count is known."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, rank)) @ rng.standard_normal((rank, n_space))


@pytest.mark.parametrize("method", ROUTES)
@pytest.mark.parametrize(
    ("n_samples", "n_space"),
    [(6, 10), (10, 6), (8, 8)],
    ids=["more_space", "more_samples", "square"],
)
def test_no_significant_mode_returns_correctly_shaped_empty_arrays(
    method: str,
    n_samples: int,
    n_space: int,
) -> None:
    """Data that is all zeros has no mode, and the shapes must still be right.

    A caller stacks these arrays or asks for their column count. An empty
    result with the wrong second dimension, or with one column of rubbish,
    breaks that caller far from here. The square case also pins the branch
    choice, where n_samples equals n_space.
    """
    data = np.zeros((n_samples, n_space))
    weights = np.ones(n_space)

    modes, eigenvalues, coefficients = weighted_second_order(data, weights, method=method)

    assert modes.shape == (n_space, 0)
    assert eigenvalues.shape == (0,)
    assert coefficients.shape == (n_samples, 0)


@pytest.mark.parametrize(
    ("n_samples", "n_space"),
    [(6, 10), (10, 6), (8, 8)],
    ids=["more_space", "more_samples", "square"],
)
def test_the_two_routes_give_the_same_decomposition(n_samples: int, n_space: int) -> None:
    """The answer must not depend on which kernel the code chose to factor.

    The routes run different algorithms on differently sized matrices, so they
    agree to rounding, not to the bit. The square case is the one where the
    branch rule reads n_samples against n_space with no margin.
    """
    rank = 4
    data = _rank_deficient(n_samples, n_space, rank)
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.5, 2.0, n_space)

    modes_e, eig_e, coeff_e = weighted_second_order(data, weights, method="eigh")
    modes_s, eig_s, coeff_s = weighted_second_order(data, weights, method="svd")

    assert eig_e.shape == eig_s.shape == (rank,)
    np.testing.assert_allclose(eig_e, eig_s, rtol=1e-10)
    np.testing.assert_allclose(np.abs(modes_e), np.abs(modes_s), atol=1e-9)
    np.testing.assert_allclose(np.abs(coeff_e), np.abs(coeff_s), atol=1e-9)


@pytest.mark.parametrize("method", ROUTES)
@pytest.mark.parametrize(
    ("n_samples", "n_space"),
    [(6, 10), (10, 6), (8, 8)],
    ids=["more_space", "more_samples", "square"],
)
def test_the_modes_and_coefficients_rebuild_the_data(
    method: str,
    n_samples: int,
    n_space: int,
) -> None:
    """Coefficients times modes must return the field that went in.

    This is the claim that makes a decomposition a decomposition. It holds on
    both routes and in all three shape cases, including the square one where
    the branch rule has no margin.
    """
    data = _rank_deficient(n_samples, n_space, rank=4)
    rng = np.random.default_rng(2)
    weights = rng.uniform(0.5, 2.0, n_space)

    modes, _, coefficients = weighted_second_order(data, weights, method=method)

    np.testing.assert_allclose(coefficients @ modes.T, data, atol=1e-9)


@pytest.mark.parametrize("method", ROUTES)
def test_the_mode_count_matches_the_rank_of_the_data(method: str) -> None:
    """Rank-4 data gives four modes, not the full count of either dimension.

    The relative floor is what removes the rest. A route that skipped it would
    return ten modes here, six of them rounding noise.
    """
    data = _rank_deficient(9, 11, rank=4)
    weights = np.ones(11)

    _, eigenvalues, _ = weighted_second_order(data, weights, method=method)

    assert eigenvalues.size == 4
    assert np.all(eigenvalues > 0.0)


@pytest.mark.parametrize(
    ("n_samples", "n_space"),
    [(6, 10), (10, 6), (8, 8)],
    ids=["more_space", "more_samples", "square"],
)
def test_a_complex_ensemble_with_no_mode_keeps_its_shape_and_its_dtype(
    n_samples: int,
    n_space: int,
) -> None:
    """The complex path must return empty arrays that are still complex.

    PSD-POD hands this path a Fourier ensemble. A caller that stacks the
    result, or writes it to a file with a declared type, needs the second
    dimension and the dtype to survive a case with no significant mode. The
    real path is covered above; this is the same claim for the complex one.
    """
    data = np.zeros((n_samples, n_space), dtype=np.complex128)
    weights = np.ones(n_space)

    modes, eigenvalues, coefficients = weighted_second_order(data, weights, method="eigh")

    assert modes.shape == (n_space, 0)
    assert eigenvalues.shape == (0,)
    assert coefficients.shape == (n_samples, 0)
    assert np.iscomplexobj(modes)
    assert np.iscomplexobj(coefficients)
