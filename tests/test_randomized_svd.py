"""Randomized SVD: opt-in Halko route and method= dispatch on compute_reduced_svd.

Accuracy of the randomized route depends on spectral decay, so method="auto"
must never select it. These tests pin the shapes/ordering contract, determinism
for a fixed seed, improvement with power iterations, and the documented
limitation on a slowly decaying spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.base import (
    compute_reduced_svd,
    randomized_svd,
    use_iterative_svd,
)


def _planted_matrix(m: int, n: int, spectrum: np.ndarray, seed: int = 0):
    """Orthonormal factors times a planted singular spectrum."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((m, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (u * spectrum) @ v.T


def test_randomized_matches_dense_on_fast_decay():
    """Planted 0.70^j spectrum: randomized agrees with dense to rtol=1e-10."""
    m, n, k = 400, 200, 15
    spectrum = 0.70 ** np.arange(n)
    X = _planted_matrix(m, n, spectrum)
    _, s_rand, _ = randomized_svd(X, k, n_power_iterations=2, seed=0)
    s_dense = np.linalg.svd(X, full_matrices=False, compute_uv=False)[:k]
    np.testing.assert_allclose(s_rand, s_dense, rtol=1e-10)


def test_randomized_improves_with_power_iterations():
    """On 0.95^j, error at 8 power iterations is at least 100x smaller than at 0."""
    m, n, k = 400, 200, 15
    spectrum = 0.95 ** np.arange(n)
    X = _planted_matrix(m, n, spectrum)
    s_true = spectrum[:k]

    _, s0, _ = randomized_svd(X, k, n_power_iterations=0, seed=0)
    _, s8, _ = randomized_svd(X, k, n_power_iterations=8, seed=0)
    err0 = float(np.max(np.abs(s0 - s_true) / s_true))
    err8 = float(np.max(np.abs(s8 - s_true) / s_true))
    assert err8 * 100 <= err0, (
        f"power iterations did not improve enough: err0={err0:.3e}, err8={err8:.3e}"
    )


def test_randomized_is_inaccurate_on_slow_decay():
    """Planted 0.999^j: error worse than 1e-2 at 2 power iterations.

    Keeps the docstring honest. If a future change makes this pass, update the
    docstring numbers in the same change rather than deleting the test.
    """
    m, n, k = 400, 200, 15
    spectrum = 0.999 ** np.arange(n)
    X = _planted_matrix(m, n, spectrum)
    s_true = spectrum[:k]
    _, s, _ = randomized_svd(X, k, n_power_iterations=2, seed=0)
    err = float(np.max(np.abs(s - s_true) / s_true))
    assert err > 1e-2, (
        f"slow-decay error {err:.3e} is better than 1e-2 — update docstring numbers"
    )


def test_auto_never_selects_randomized():
    """method='auto' matches dense-or-iterative across shapes and ranks; never randomized."""
    rng = np.random.default_rng(1)
    cases = []
    for min_dim in (100, 300, 2000):
        # Tall matrix so min(X.shape) == min_dim (columns).
        m = min_dim + 50
        n = min_dim
        for rank in (1, 10, 99, 100, min_dim - 1):
            if rank < 1 or rank >= min_dim:
                continue
            cases.append((m, n, rank))

    for m, n, rank in cases:
        X = rng.standard_normal((m, n))
        u_auto, s_auto, vh_auto = compute_reduced_svd(X, rank, method="auto")
        want_iter = use_iterative_svd(min(X.shape), rank)
        forced = "iterative" if want_iter else "dense"
        u_f, s_f, vh_f = compute_reduced_svd(X, rank, method=forced)

        # auto must match the forced dense/iterative choice bit-for-bit on s.
        np.testing.assert_array_equal(s_auto, s_f)
        np.testing.assert_array_equal(u_auto, u_f)
        np.testing.assert_array_equal(vh_auto, vh_f)

        # and must not be identical to a fresh randomized draw (atol/rtol 0).
        _, s_r, _ = randomized_svd(X, rank, seed=0)
        # Compare leading rank singular values; dense may return a longer s.
        if np.array_equal(s_auto[:rank], s_r[:rank]):
            pytest.fail(
                f"auto returned bit-identical singular values to randomized "
                f"(m={m}, n={n}, rank={rank})"
            )


def test_randomized_shapes_and_ordering():
    """u, s, vh shapes match the dense route for the same rank; s non-increasing."""
    rng = np.random.default_rng(2)
    m, n, k = 80, 40, 7
    X = rng.standard_normal((m, n))
    u_r, s_r, vh_r = randomized_svd(X, k, seed=0)
    u_d, s_d, vh_d = compute_reduced_svd(X, k, method="dense")

    assert u_r.shape == (m, k)
    assert s_r.shape == (k,)
    assert vh_r.shape == (k, n)
    # Dense returns the full economy SVD; leading-k slices match randomized shapes.
    assert u_d[:, :k].shape == u_r.shape
    assert s_d[:k].shape == s_r.shape
    assert vh_d[:k, :].shape == vh_r.shape
    assert np.all(np.diff(s_r) <= 0)


def test_unknown_method_raises():
    """method='nope' raises ValueError naming the four accepted options."""
    X = np.random.default_rng(0).standard_normal((50, 20))
    with pytest.raises(ValueError, match=r"auto.*dense.*iterative.*randomized") as ei:
        compute_reduced_svd(X, 3, method="nope")
    msg = str(ei.value)
    for name in ("auto", "dense", "iterative", "randomized"):
        assert name in msg


def test_randomized_is_deterministic_for_a_seed():
    """Two calls with the same seed are bit-identical."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 30))
    a = randomized_svd(X, 5, seed=42)
    b = randomized_svd(X, 5, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
