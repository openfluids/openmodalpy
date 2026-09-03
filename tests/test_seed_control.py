"""Reproducibility guarantees that must live in the committed test suite.

Ported from the gate's AC1/AC2 so a future change cannot silently reintroduce
non-deterministic ARPACK or an ignored generator seed without pytest failing.
"""

from __future__ import annotations

import h5py
import numpy as np

from openmodalpy.core.base import generate_dummy_data_like_jetles
from openmodalpy.core.operators import compute_reduced_svd, use_iterative_svd


def test_arpack_path_bit_identical():
    """Two ARPACK-branch SVDs on the same input must be bit-identical.

    Size is chosen so ``use_iterative_svd`` is True (small rank fraction on a
    large enough matrix). If this precondition ever stops holding, the
    assertion fails instead of silently testing the dense fallback.
    """
    rank = 10
    rng = np.random.default_rng(12345)
    # min_dim=300, rank=10 → 10 < 0.05*300 and min_dim >= 256 → iterative
    X = rng.standard_normal((300, 300))
    min_dim = min(X.shape)
    assert use_iterative_svd(min_dim, rank), (
        f"test no longer exercises the ARPACK branch (rank={rank}, min_dim={min_dim})"
    )

    u1, s1, vh1 = compute_reduced_svd(X, rank)
    u2, s2, vh2 = compute_reduced_svd(X, rank)

    assert np.array_equal(u1, u2)
    assert np.array_equal(s1, s2)
    assert np.array_equal(vh1, vh2)
    # Sanity: determinism is not bought by a degenerate return.
    assert s1.shape == (rank,)
    assert np.all(np.diff(s1) <= 0)
    assert np.all(s1 > 0)


def test_generate_dummy_data_like_jetles_honours_seed(tmp_path):
    """Same seed → identical array; different seed → different array."""

    def make(seed: int) -> np.ndarray:
        path = tmp_path / f"dummy_seed_{seed}.h5"
        generate_dummy_data_like_jetles(str(path), Ns=16, Nx=8, Ny=6, seed=seed, save_mat=False)
        with h5py.File(path, "r") as f:
            return np.array(f["p"])

    same_a = make(3)
    same_b = make(3)
    other = make(4)

    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)
