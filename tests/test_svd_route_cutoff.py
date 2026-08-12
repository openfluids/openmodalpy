"""SVD route relative cutoff: honest rank on null space, keeps weak modes.

The eigh path already drops eigenvalues below ``n_kernel * eps * lambda_max``.
The SVD path applies the same relative scale in the singular-value domain
(``sigma > n_kernel * eps * sigma_max``), so rank-deficient data returns the
honest mode count without deleting modes the SVD route exists to recover.
"""

from __future__ import annotations

import numpy as np

from openmodalpy.core.decomposition import weighted_second_order


def test_both_routes_return_exact_rank_on_rank3_data():
    """Exactly rank-3 snapshots: both routes keep exactly 3 modes."""
    rng = np.random.default_rng(0)
    n_s, n_x = 40, 400
    base = rng.standard_normal((n_s, 3)) @ rng.standard_normal((3, n_x))
    assert np.linalg.matrix_rank(base) == 3
    w = np.ones(n_x)
    for method in ("eigh", "svd"):
        modes, eigs, _ = weighted_second_order(base, w, method=method)
        assert modes.shape[1] == 3, f"{method} kept {modes.shape[1]}, expected 3"
        assert eigs.shape == (3,)


def test_svd_route_keeps_planted_mode_at_singular_ratio_1e_10():
    """A genuine mode at singular-value ratio 1e-10 survives the SVD floor.

    That ratio sits at eigenvalue ratio 1e-20 — below the eigh-style floor
    ``n_kernel * eps`` (~1e-14) but above the SVD floor
    ``(n_kernel * eps)**2`` (~1e-28). Recovering it is why the floor must
    live in the singular-value domain.
    """
    rng = np.random.default_rng(0)
    n_s, n_x = 40, 400
    base = rng.standard_normal((n_s, 3)) @ rng.standard_normal((3, n_x))
    weak_dir = rng.standard_normal(n_x)
    weak_time = rng.standard_normal(n_s)
    outer = np.outer(weak_time, weak_dir)
    ratio = 1e-10
    q = base + ratio * np.linalg.norm(base) / np.linalg.norm(outer) * outer
    target = weak_dir / np.linalg.norm(weak_dir)
    modes, _eigs, _ = weighted_second_order(q, np.ones(n_x), method="svd")
    assert modes.shape[1] >= 4, f"expected at least 4 modes, got {modes.shape[1]}"
    corr = max(
        (abs(float(modes[:, k] @ target)) / float(np.linalg.norm(modes[:, k])) for k in range(modes.shape[1])),
        default=0.0,
    )
    assert corr > 0.9, f"planted mode lost (best corr {corr:.6f})"


def test_svd_n_keep_none_drops_null_on_centered_keeps_delay_lift_rank():
    """Centered large-offset input caps at m-1; delay lifts keep matrix rank.

    With n_keep=None the SVD route measures row-centeredness and tightens the
    cap only when that measurement fires. A matrix centered after a 1e3 offset
    must return m-1 modes (the residual null is junk). A genuine delay lift
    must still return its matrix_rank — never drop a real mode.
    """
    from openmodalpy.core.decomposition import (
        CENTERED_ROW_MEAN_RATIO,
        _row_mean_to_std_ratio,
    )

    rng = np.random.default_rng(5)
    m, n = 30, 100
    offset = 1e3
    x = rng.standard_normal((m, n)) + offset
    xc = x - x.mean(axis=0)
    centered_stat = _row_mean_to_std_ratio(xc)
    assert centered_stat < CENTERED_ROW_MEAN_RATIO, centered_stat
    # Independent oracle: numpy agrees the centered matrix lost exactly one
    # direction. Without this the next assert only restates the formula the
    # implementation already uses.
    assert int(np.linalg.matrix_rank(xc, tol=1e-10)) == m - 1
    kept = weighted_second_order(xc, np.ones(n), method="svd", n_keep=None)[1].size
    assert kept == min(m - 1, n), f"centered kept {kept}, want {min(m - 1, n)}"

    # Explicit n_keep is the caller's bound — measurement must not rewrite it.
    kept_explicit = weighted_second_order(xc, np.ones(n), method="svd", n_keep=m)[1].size
    assert kept_explicit == m  # may still include the junk mode; path unchanged

    series = rng.standard_normal(256)
    d = 8
    n_cols = series.shape[0] - d + 1
    lifted = np.stack([series[i : i + n_cols] for i in range(d)], axis=1)
    lift_stat = _row_mean_to_std_ratio(lifted)
    assert lift_stat >= CENTERED_ROW_MEAN_RATIO, lift_stat
    kept_lift = weighted_second_order(lifted, np.ones(lifted.shape[1]), method="svd", n_keep=None)[1].size
    rank = int(np.linalg.matrix_rank(lifted, tol=1e-10))
    assert kept_lift == rank, f"delay lift kept {kept_lift}, rank {rank}"


def test_svd_n_keep_none_keeps_the_mode_of_a_constant_matrix():
    """A constant array has no spread, and must not be read as centered.

    Every column mean equals the constant, so the row-mean / std ratio is
    infinite, not zero. Calling it centered tightens the cap by one, and for a
    single-sample array that removes its only mode. The check is against
    numpy's rank rather than against the cap formula.
    """
    for data in (np.ones((1, 5)), np.ones((4, 5)), np.zeros((4, 5))):
        kept = weighted_second_order(data, np.ones(data.shape[1]), method="svd", n_keep=None)[1].size
        rank = int(np.linalg.matrix_rank(data, tol=1e-10))
        assert kept == rank, f"shape {data.shape}: kept {kept}, rank {rank}"
