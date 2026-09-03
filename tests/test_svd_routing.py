"""SVD routing: iterative only for small rank fractions on large matrices.

``compute_reduced_svd`` must not send near-full-rank requests to ARPACK.
The decision lives in ``use_iterative_svd``; these tests pin the predicate and
check that the two routes agree on a fast-decaying planted spectrum.
"""

from __future__ import annotations

import numpy as np

from openmodalpy.core.operators import ARPACK_MAX_RANK_FRACTION, ARPACK_MIN_DIM, compute_reduced_svd, use_iterative_svd


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
