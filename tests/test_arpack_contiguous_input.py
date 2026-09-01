"""ARPACK must receive a contiguous matrix, whatever view the caller holds.

``svds`` does one matrix-vector product per iteration, so it reads the whole
matrix tens of times. A non-contiguous view makes every read stride through
memory. Callers usually hand over a view: DMD passes ``X[:, :-1]``, which drops
the last column and leaves the row stride of the full array behind.

Measured on a delay-embedded double gyre, ``X[:, :-1]`` at (80000, 396),
rank 10, one BLAS thread: 2.792 s for the view against 0.045 s to copy plus
0.338 s to solve. End to end, ``perform_dmd`` with ``embedding_dim=4`` fell
from 2.559 s to 0.557 s.

A timing assertion would flake on a busy machine, so this pins the cause
instead: the array that reaches ``svds`` is contiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core import base as base_mod
from openmodalpy.core.base import compute_reduced_svd, use_iterative_svd


def test_arpack_receives_a_contiguous_matrix(monkeypatch):
    """A sliced, non-contiguous caller argument reaches ``svds`` contiguous."""
    rng = np.random.default_rng(0)
    # Wide enough that use_iterative_svd picks ARPACK: min_dim >= 256 and
    # rank < 0.05 * min_dim.
    full = rng.standard_normal((2000, 300))
    sliced = full[:, :-1]
    assert not sliced.flags["C_CONTIGUOUS"], "the fixture must hand over a view"

    rank = 5
    assert use_iterative_svd(min(sliced.shape), rank), "this shape must route to ARPACK"

    seen: list[bool] = []
    real_svds = base_mod.svds

    def recording_svds(matrix, **kwargs):
        seen.append(bool(np.asarray(matrix).flags["C_CONTIGUOUS"]))
        return real_svds(matrix, **kwargs)

    monkeypatch.setattr(base_mod, "svds", recording_svds)
    compute_reduced_svd(sliced, rank)

    assert seen == [True], f"svds saw contiguous={seen}, expected [True]"


def test_the_copy_does_not_change_the_answer():
    """The contiguous copy holds the same leading singular values as the view."""
    rng = np.random.default_rng(1)
    full = rng.standard_normal((2000, 40)) @ rng.standard_normal((40, 300))
    sliced = full[:, :-1]
    rank = 5

    _, s_from_view, _ = compute_reduced_svd(sliced, rank)
    exact = np.linalg.svd(np.ascontiguousarray(sliced), full_matrices=False, compute_uv=False)
    assert s_from_view == pytest.approx(exact[:rank], rel=1e-10)
