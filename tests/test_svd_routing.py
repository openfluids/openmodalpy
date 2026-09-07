"""SVD routing: iterative only for small rank fractions on large matrices.

``compute_reduced_svd`` must not send near-full-rank requests to ARPACK.
The decision lives in ``use_iterative_svd``; these tests pin the predicate and
check that the two routes agree on a fast-decaying planted spectrum.
"""

from __future__ import annotations

import numpy as np

from openmodalpy.core.operators import (
    ARPACK_MAX_RANK_FRACTION,
    ARPACK_MIN_DIM,
    compute_reduced_svd,
    svd_route,
    use_iterative_svd,
)


def test_near_full_rank_does_not_use_iterative():
    """k = min_dim - 1 at min_dim = 2000 must stay dense (the near-full-rank case)."""
    min_dim = 2000
    assert use_iterative_svd(min_dim, min_dim - 1) is False


def test_small_rank_large_matrix_uses_iterative():
    """Small k on a large matrix still takes the iterative path."""
    assert use_iterative_svd(2000, 10) is True


def test_small_matrix_never_uses_iterative():
    """Below ARPACK_MIN_DIM, any rank stays dense."""
    min_dim = ARPACK_MIN_DIM - 1
    assert min_dim == 255
    for rank in (1, 10, 100, min_dim - 1):
        assert use_iterative_svd(min_dim, rank) is False


def test_routing_boundaries_are_exact():
    """Pin both edges of the fraction and the minimum-dimension cutoff."""
    min_dim = 2000
    # 0.05 * 2000 = 100 → k < 100 iterative, k == 100 dense
    assert ARPACK_MAX_RANK_FRACTION * min_dim == 100.0
    assert use_iterative_svd(min_dim, 99) is True
    assert use_iterative_svd(min_dim, 100) is False

    assert use_iterative_svd(ARPACK_MIN_DIM - 1, 1) is False
    assert use_iterative_svd(ARPACK_MIN_DIM, 1) is True


def test_iterative_and_dense_agree_on_leading_triplets():
    """On a fast-decaying planted spectrum, both routes match leading singular values."""
    rng = np.random.default_rng(0)
    m, n = 400, 300
    rank = 5
    min_dim = min(m, n)
    assert use_iterative_svd(min_dim, rank) is True

    u_fac, _ = np.linalg.qr(rng.standard_normal((m, n)))
    v_fac, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Exponential decay so the leading modes dominate and ARPACK is well posed.
    sigma = np.exp(-np.arange(n, dtype=float))
    X = (u_fac * sigma) @ v_fac.T

    _, s_iter, _ = compute_reduced_svd(X, rank)
    s_dense = np.linalg.svd(X, full_matrices=False, compute_uv=False)[:rank]

    np.testing.assert_allclose(s_iter, s_dense, rtol=1e-8)


def test_svd_route_names_the_route_the_rule_picks():
    """``svd_route`` must report exactly what ``use_iterative_svd`` decides.

    The two are separate functions, and a result file records what the first
    one says. If they disagree, a saved run names a route it did not take.
    """
    for min_dim in (10, ARPACK_MIN_DIM - 1, ARPACK_MIN_DIM, 2000, 5000):
        for rank in (1, 10, 100, 300, min_dim - 1):
            if rank >= min_dim:
                continue
            expected = "iterative" if use_iterative_svd(min_dim, rank) else "dense"
            assert svd_route(min_dim, rank) == expected, (min_dim, rank)


def test_svd_route_reports_both_routes():
    """The report must not be constant. Both routes must be reachable."""
    assert svd_route(2000, 10) == "iterative"
    assert svd_route(2000, 1999) == "dense"
    assert svd_route(10, 2) == "dense"


def test_a_forced_route_is_reported_as_forced():
    """A caller that names a route gets that route in the report."""
    assert svd_route(2000, 10, method="dense") == "dense"
    assert svd_route(10, 2, method="iterative") == "iterative"
    assert svd_route(2000, 10, method="randomized") == "randomized"


def test_the_reported_route_is_the_route_that_runs():
    """Pin the report to the code path, not to a second copy of the rule.

    A dense solve returns every singular value it can. ARPACK returns exactly
    the rank asked for. That difference tells the two apart from the outside.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 300))

    rank = 5
    min_dim = min(x.shape)
    assert svd_route(min_dim, rank) == "iterative"
    _, s_iterative, _ = compute_reduced_svd(x, rank)
    assert s_iterative.size == rank

    rank_dense = min_dim - 1
    assert svd_route(min_dim, rank_dense) == "dense"
    _, s_dense, _ = compute_reduced_svd(x, rank_dense)
    assert s_dense.size == min_dim
