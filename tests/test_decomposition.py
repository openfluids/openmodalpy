"""Tests for the lift / metric / weighted-second-order seam."""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.base import (
    CANONICAL_TIE_RTOL,
    canonical_pivot_index,
    canonicalize_modes,
)
from openmodalpy.core.decomposition import (
    BandFilteredLift,
    DelayEmbeddingLift,
    IdentityLift,
    SpatialMetric,
    _significant_eigenvalue_mask,
    weighted_second_order,
)


def test_identity_lift_kind_and_apply():
    lift = IdentityLift()
    assert lift.kind == "identity_centered_snapshots"
    q = np.arange(12.0).reshape(4, 3)
    out = lift.apply(q)
    np.testing.assert_array_equal(out, q)


def test_delay_embedding_lift_shape_and_kind():
    lift = DelayEmbeddingLift(embedding_dim=3)
    assert lift.kind == "delay_embedding"
    # snapshots: t=0..4, space=2
    data = np.arange(10.0).reshape(5, 2)
    lifted = lift.apply(data)
    # m = 5 - 3 + 1 = 3 rows, d*Nspace = 6 cols
    assert lifted.shape == (3, 6)
    # first sample: stack of rows 0,1,2
    np.testing.assert_array_equal(lifted[0], np.concatenate([data[0], data[1], data[2]]))
    np.testing.assert_array_equal(lifted[2], np.concatenate([data[2], data[3], data[4]]))


def test_delay_embedding_rejects_small_series():
    lift = DelayEmbeddingLift(embedding_dim=5)
    with pytest.raises(ValueError, match="embedding_dim"):
        lift.apply(np.zeros((4, 2)))


def test_band_filtered_lift_kind_and_passband():
    lift = BandFilteredLift()
    assert lift.kind == "multiscale_filtered_snapshots"
    # A pure tone at 1 Hz, dt=0.1 → Nyquist 5 Hz
    t = np.arange(64) * 0.1
    signal = np.sin(2 * np.pi * 1.0 * t)[:, None]
    band = BandFilteredLift(f_low=0.5, f_high=1.5, dt=0.1, is_last=True)
    filtered = band.apply(signal)
    # Energy should remain concentrated near the tone
    assert filtered.shape == signal.shape
    assert np.linalg.norm(filtered) > 0.5 * np.linalg.norm(signal)

    out_of_band = BandFilteredLift(f_low=3.0, f_high=4.0, dt=0.1, is_last=True)
    quiet = out_of_band.apply(signal)
    assert np.linalg.norm(quiet) < 0.1 * np.linalg.norm(signal)


def test_spatial_metric_tile():
    m = SpatialMetric(np.array([1.0, 2.0, 3.0]))
    tiled = m.tile(2)
    np.testing.assert_array_equal(tiled, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_weighted_second_order_eigh_matches_manual_unweighted():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((20, 5))
    q = q - q.mean(axis=0)
    w = np.ones(5)
    modes, eigs, coeffs = weighted_second_order(q, w, method="eigh")
    # Spatial kernel path (Ns > Nspace): eigenvalues of (X^T X)/Ns
    gram = (q.T @ q) / q.shape[0]
    ref_eigs = np.sort(np.linalg.eigvalsh(gram))[::-1]
    np.testing.assert_allclose(eigs, ref_eigs, rtol=1e-10, atol=1e-12)
    assert modes.shape == (5, 5)
    assert coeffs.shape == (20, 5)


def test_weighted_second_order_relative_cutoff_and_n_keep():
    """Relative cutoff keeps only significant modes; n_keep truncates further."""
    rng = np.random.default_rng(1)
    # Rank-2 signal in 6D space
    a = rng.standard_normal((30, 2))
    b = rng.standard_normal((2, 6))
    q = a @ b
    q = q - q.mean(axis=0)
    modes, eigs, coeffs = weighted_second_order(
        q,
        np.ones(6),
        method="eigh",
        n_keep=2,
    )
    assert eigs.size == 2
    assert modes.shape == (6, 2)
    assert coeffs.shape == (30, 2)
    assert np.all(eigs > 0.0)


def test_weighted_second_order_svd_route():
    rng = np.random.default_rng(2)
    # samples × features (as DelayEmbeddingLift returns)
    data = rng.standard_normal((12, 8))
    modes, eigs, coeffs = weighted_second_order(
        data,
        np.ones(8),
        method="svd",
        n_keep=3,
    )
    assert modes.shape == (8, 3)
    assert eigs.shape == (3,)
    assert coeffs.shape == (12, 3)
    # eigenvalues = sigma^2 / n_samples
    sigma = np.linalg.svd(data, full_matrices=False, compute_uv=False)[:3]
    np.testing.assert_allclose(eigs, sigma**2 / 12, rtol=1e-10, atol=1e-12)


def _assert_modes_canonical(modes: np.ndarray) -> None:
    """Each mode's band-pivot entry must be real and positive."""
    for k in range(modes.shape[1]):
        col = modes[:, k]
        if not np.any(np.abs(col) > 0):
            continue
        i = canonical_pivot_index(col)
        v = col[i]
        assert float(np.real(v)) > 0
        if np.iscomplexobj(modes):
            assert abs(float(np.imag(v))) <= 1e-9 * float(np.abs(v))


def test_canonicalize_modes_real_sign_and_reconstruction():
    """Real columns flip so the dominant entry is positive; coeffs keep recon."""
    modes = np.array(
        [
            [0.1, -0.9],
            [-2.0, 0.3],
            [0.5, 0.4],
        ],
        dtype=float,
    )
    coeffs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    before = coeffs @ modes.T
    out_m, out_c = canonicalize_modes(modes, coeffs)
    _assert_modes_canonical(out_m)
    np.testing.assert_allclose(out_c @ out_m.T, before, rtol=0, atol=1e-15)
    # mode 0 had dominant -2 at index 1 → flipped; mode 1 already +0.9 at index 0
    assert out_m[1, 0] == pytest.approx(2.0)
    assert out_m[0, 1] == pytest.approx(0.9)


def test_canonicalize_modes_complex_phase_and_reconstruction():
    """Complex columns get a unit phase so the dominant entry is real positive."""
    modes = np.array(
        [
            [1 + 0j, 0.1 + 0.1j],
            [0.5 - 0.5j, 3 - 4j],
            [0.2j, 0.0j],
        ],
        dtype=complex,
    )
    coeffs = np.array([[1 + 1j, 2j], [0.5, 1 - 1j]], dtype=complex)
    # Complex reconstruction is coeffs @ modes^H (Hermitian transpose).
    before = coeffs @ modes.conj().T
    out_m, out_c = canonicalize_modes(modes, coeffs)
    _assert_modes_canonical(out_m)
    np.testing.assert_allclose(out_c @ out_m.conj().T, before, rtol=0, atol=1e-14)


def test_complex_route_coeffs_remain_weighted_projection():
    """Complex eigh coeffs stay conj(data) @ (W * modes) after canonicalize.

    Reference is the projection formula only — no library helper on that side.
    """
    rng = np.random.default_rng(7)
    ens = rng.standard_normal((12, 8)) + 1j * rng.standard_normal((12, 8))
    w = np.linspace(0.5, 2.0, 8)
    modes, _eig, coeffs = weighted_second_order(ens, SpatialMetric(w), method="eigh", n_keep=3)
    expected = ens.conj() @ (w[:, np.newaxis] * modes)
    scale = float(np.linalg.norm(ens))
    rel = float(np.linalg.norm(coeffs - expected) / scale)
    assert rel <= 1e-14, f"projection residual {rel:.3e} exceeds 1e-14"
    recon = float(np.linalg.norm(coeffs @ modes.conj().T))
    assert recon == pytest.approx(8.423251717664, abs=1e-10)


def test_canonicalize_modes_near_tie_stable_under_ulp_noise():
    """Two opposite-sign near-equal peaks keep the same sign under small eps."""
    for eps in (1e-16, 1e-15, 1e-14, 1e-13):
        a, _ = canonicalize_modes(np.array([[1.0 + eps], [-1.0], [0.3]]))
        b, _ = canonicalize_modes(np.array([[1.0 - eps], [-1.0], [0.3]]))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
        _assert_modes_canonical(a)
        _assert_modes_canonical(b)
        # Non-adjacent peaks, same rule.
        a2, _ = canonicalize_modes(np.array([[0.2], [-1.0], [0.5], [1.0 + eps], [-0.1]]))
        b2, _ = canonicalize_modes(np.array([[0.2], [-1.0], [0.5], [1.0 - eps], [-0.1]]))
        np.testing.assert_allclose(a2, b2, rtol=0, atol=1e-9)
        _assert_modes_canonical(a2)
        _assert_modes_canonical(b2)


def test_canonicalize_modes_exact_tie_and_zero_column():
    """Exact ties are deterministic; an all-zero column is left alone."""
    modes = np.array([[-2.0], [2.0], [-2.0]])
    t1, _ = canonicalize_modes(modes)
    t2, _ = canonicalize_modes(modes.copy())
    np.testing.assert_array_equal(t1, t2)
    _assert_modes_canonical(t1)
    z, _ = canonicalize_modes(np.zeros((3, 1)))
    np.testing.assert_array_equal(z, 0.0)


def test_canonicalize_modes_band_prefers_lowest_in_band_index():
    """Sign follows the lowest in-band index, not the strict magnitude max."""
    # |col[0]| is inside the band of the peak but not the strict maximum.
    # Pivot is index 0 (negative) → whole column flips; strict argmax would
    # leave the already-positive peak at index 1 and skip the flip.
    peak = 1.0
    near = peak * (1.0 - 0.5 * CANONICAL_TIE_RTOL)
    modes = np.array([[-near], [peak], [0.1]], dtype=float)
    assert canonical_pivot_index(modes[:, 0]) == 0
    assert int(np.argmax(np.abs(modes[:, 0]))) == 1
    out, _ = canonicalize_modes(modes)
    _assert_modes_canonical(out)
    assert float(out[0, 0]) == pytest.approx(near)
    assert float(out[1, 0]) == pytest.approx(-peak)


def test_canonicalize_modes_nan_and_coeffs_shape():
    """Non-finite modes and a coeffs column count mismatch raise ValueError."""
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_modes(np.array([[1.0], [np.nan], [0.5]]))
    with pytest.raises(ValueError, match="coeffs shape"):
        canonicalize_modes(np.ones((4, 3)), np.ones((5, 2)))
    # Mismatched coeffs must raise even when modes are empty (no columns).
    with pytest.raises(ValueError, match="coeffs shape"):
        canonicalize_modes(np.ones((4, 0)), np.ones((5, 2)))


def test_canonicalize_modes_integer_promotes_float32_preserved():
    """Integer modes promote to float64; float32 stays float32."""
    modes_i = np.array([[1, -2], [-3, 4], [0, 1]], dtype=np.int64)
    modes_f = modes_i.astype(np.float64)
    out_i, _ = canonicalize_modes(modes_i)
    out_f, _ = canonicalize_modes(modes_f)
    assert out_i.dtype == np.float64
    np.testing.assert_array_equal(out_i, out_f)
    _assert_modes_canonical(out_i)

    modes32 = np.array([[0.1, -0.9], [-2.0, 0.3], [0.5, -0.4]], dtype=np.float32)
    out32, _ = canonicalize_modes(modes32)
    assert out32.dtype == np.float32
    _assert_modes_canonical(out32)


def test_weighted_second_order_modes_are_canonical():
    """eigh (real + complex) and svd routes leave modes canonical."""
    rng = np.random.default_rng(11)
    q = rng.standard_normal((20, 6))
    q = q - q.mean(axis=0)
    modes, _e, coeffs = weighted_second_order(q, np.ones(6), method="eigh")
    _assert_modes_canonical(modes)
    # Full-rank spatial path: reconstruction of centered data is exact.
    np.testing.assert_allclose(coeffs @ modes.T, q, rtol=1e-10, atol=1e-10)

    data = rng.standard_normal((12, 8))
    modes_s, _e_s, _c_s = weighted_second_order(data, np.ones(8), method="svd", n_keep=3)
    _assert_modes_canonical(modes_s)

    ens = rng.standard_normal((10, 5)) + 1j * rng.standard_normal((10, 5))
    modes_c, _e_c, _coeffs_c = weighted_second_order(ens, np.ones(5), method="eigh", n_keep=3)
    _assert_modes_canonical(modes_c)


def test_eigh_spatial_nonuniform_metric_matches_weighted_gram():
    """Spatial branch (n_samples >= n_space) honours a non-uniform metric.

    Expected eigenvalues are those of ``(Xw.T @ Xw) / n_samples`` with
    ``Xw = X * sqrt(W)`` — the weighted spatial Gram, formed in-test. Modes
    must be W-orthonormal: ``modes.T @ (W[:, None] * modes) == I``.
    """
    rng = np.random.default_rng(21)
    n_samples, n_space = 15, 5
    x = rng.standard_normal((n_samples, n_space))
    x = x - x.mean(axis=0)
    w = np.array([0.5, 1.0, 2.0, 0.25, 3.0])
    xw = x * np.sqrt(w)
    ref_eigs = np.sort(np.linalg.eigvalsh((xw.T @ xw) / n_samples))[::-1]

    modes, eigs, coeffs = weighted_second_order(x, w, method="eigh")
    np.testing.assert_allclose(eigs, ref_eigs, rtol=1e-10, atol=1e-12)
    gram_w = modes.T @ (w[:, np.newaxis] * modes)
    np.testing.assert_allclose(gram_w, np.eye(n_space), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(coeffs @ modes.T, x, rtol=1e-10, atol=1e-10)


def test_eigh_temporal_nonuniform_metric_matches_temporal_kernel():
    """Temporal branch (n_samples < n_space) uses kernel ``Xw @ Xw.T / n_samples``.

    Shape 6 × 20 forces the temporal path. Expected spectrum is eigvalsh of
    that kernel (formed in-test). Division by ``n_samples - 1`` would shift
    every eigenvalue by the factor ``n / (n - 1)``.
    """
    rng = np.random.default_rng(22)
    n_samples, n_space = 6, 20
    x = rng.standard_normal((n_samples, n_space))
    x = x - x.mean(axis=0)
    w = np.linspace(0.5, 2.5, n_space)
    xw = x * np.sqrt(w)
    ref_eigs = np.sort(np.linalg.eigvalsh((xw @ xw.T) / n_samples))[::-1]
    # Centered data has rank ≤ n_samples-1; drop the numerical null of the Gram.
    ref_pos = ref_eigs[_significant_eigenvalue_mask(ref_eigs, n_samples)]

    modes, eigs, coeffs = weighted_second_order(x, w, method="eigh")
    assert modes.shape == (n_space, ref_pos.size)
    assert coeffs.shape == (n_samples, ref_pos.size)
    np.testing.assert_allclose(eigs, ref_pos, rtol=1e-10, atol=1e-12)
    gram_w = modes.T @ (w[:, np.newaxis] * modes)
    np.testing.assert_allclose(gram_w, np.eye(ref_pos.size), rtol=1e-10, atol=1e-12)


def test_svd_nonuniform_metric_matches_weighted_singular_values():
    """SVD route eigenvalues are ``sigma**2 / n_samples`` of the weighted data.

    With ``Xw = X * sqrt(W)``, ``eigs`` must match the squared singular values
    of ``Xw.T`` divided by ``n_samples``. Weighted modes ``modes * sqrt(W)``
    recover the left singular vectors up to sign.
    """
    rng = np.random.default_rng(23)
    n_samples, n_space = 12, 8
    data = rng.standard_normal((n_samples, n_space))
    w = np.linspace(0.4, 2.0, n_space)
    xw = data * np.sqrt(w)
    n_keep = 3
    u, sigma, _vt = np.linalg.svd(xw.T, full_matrices=False)
    ref_eigs = (sigma[:n_keep] ** 2) / n_samples

    modes, eigs, coeffs = weighted_second_order(data, w, method="svd", n_keep=n_keep)
    assert modes.shape == (n_space, n_keep)
    assert coeffs.shape == (n_samples, n_keep)
    np.testing.assert_allclose(eigs, ref_eigs, rtol=1e-10, atol=1e-12)
    # Sign-invariant: |<modes * sqrt(W), u_k>| == 1 for each kept mode.
    weighted_modes = modes * np.sqrt(w)[:, np.newaxis]
    overlap = np.abs(np.sum(weighted_modes * u[:, :n_keep], axis=0))
    np.testing.assert_allclose(overlap, np.ones(n_keep), rtol=1e-10, atol=1e-12)


def test_relative_cutoff_keeps_leading_reference_spectrum():
    """Relative cutoff keeps only significant eigenvalues, and those values.

    Rank-2 product ``a @ b`` in 6-D space: the weighted spatial Gram has two
    eigenvalues above ``n_space * eps * lambda_max`` (eigvalsh of
    ``Xw.T @ Xw / n``, formed in-test) and four near-zero. With
    ``n_keep=None`` the kept spectrum must equal that leading pair — not
    merely have size 2.
    """
    rng = np.random.default_rng(24)
    n_samples, n_space = 30, 6
    a = rng.standard_normal((n_samples, 2))
    b = rng.standard_normal((2, n_space))
    x = a @ b
    x = x - x.mean(axis=0)
    w = np.ones(n_space)
    xw = x * np.sqrt(w)
    ref_all = np.sort(np.linalg.eigvalsh((xw.T @ xw) / n_samples))[::-1]
    cutoff = n_space * float(np.finfo(float).eps) * float(ref_all[0])
    ref_pos = ref_all[ref_all > cutoff]
    assert ref_pos.size == 2

    modes, eigs, coeffs = weighted_second_order(x, w, method="eigh", n_keep=None)
    assert modes.shape == (n_space, 2)
    assert coeffs.shape == (n_samples, 2)
    np.testing.assert_allclose(eigs, ref_pos, rtol=1e-10, atol=1e-12)
    # n_keep larger than the significant count still returns only the leading pair.
    _m2, eigs2, _c2 = weighted_second_order(x, w, method="eigh", n_keep=10)
    np.testing.assert_allclose(eigs2, ref_pos, rtol=1e-10, atol=1e-12)


def test_relative_cutoff_empty_result_shapes():
    """All-zero data: no eigenvalue exceeds the relative cutoff, so empty.

    Expected shapes from the seam contract: modes ``(n_space, 0)``, eigs
    ``(0,)``, coeffs ``(n_samples, 0)``.
    """
    n_samples, n_space = 10, 5
    modes, eigs, coeffs = weighted_second_order(
        np.zeros((n_samples, n_space)),
        np.ones(n_space),
        method="eigh",
    )
    assert modes.shape == (n_space, 0)
    assert eigs.shape == (0,)
    assert coeffs.shape == (n_samples, 0)


def test_n_keep_exceeds_available_rank_returns_significant_count():
    """n_keep above the significant rank returns only significant modes.

    Rank-2 product in 6-D space with ``n_keep=100``: the seam keeps the two
    eigenvalues above the relative cutoff, not the full spatial spectrum.
    """
    rng = np.random.default_rng(25)
    n_samples, n_space = 30, 6
    a = rng.standard_normal((n_samples, 2))
    b = rng.standard_normal((2, n_space))
    x = a @ b
    x = x - x.mean(axis=0)
    w = np.ones(n_space)
    xw = x * np.sqrt(w)
    ref_all = np.sort(np.linalg.eigvalsh((xw.T @ xw) / n_samples))[::-1]
    cutoff = n_space * float(np.finfo(float).eps) * float(ref_all[0])
    ref_pos = ref_all[ref_all > cutoff]
    assert ref_pos.size == 2

    modes, eigs, coeffs = weighted_second_order(x, w, method="eigh", n_keep=100)
    assert modes.shape == (n_space, 2)
    assert eigs.shape == (2,)
    assert coeffs.shape == (n_samples, 2)
    np.testing.assert_allclose(eigs, ref_pos, rtol=1e-10, atol=1e-12)


def _rank_deficient_field(nt=20, nsp=64, r=3, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nsp)
    p = np.stack([np.sin((k + 1) * np.pi * x) for k in range(r)])
    a = rng.standard_normal((nt, r))
    q = (a @ p) * scale
    return q - q.mean(axis=0, keepdims=True), np.full(nsp, 1.0 / nsp)


def test_eigh_scale_invariant_mode_count_and_spectrum():
    """Same flow in different units keeps mode count and normalised spectrum.

    Absolute eigenvalue floors fail here: eigenvalues carry units, so a
    millimetre-scale copy of a metre-scale field would drop every mode.
    """
    scales = [1.0, 1e-3, 1e-6, 1e-9]
    counts = []
    spectra = []
    for s in scales:
        q, w = _rank_deficient_field(scale=s)
        _m, lam, _c = weighted_second_order(q, w, method="eigh")
        counts.append(lam.size)
        spectra.append(np.sort(lam / lam.max())[::-1] if lam.size else np.zeros(0))
    assert len(set(counts)) == 1, f"mode count drifted across units: {counts}"
    assert counts[0] == 3
    ref = spectra[0]
    for sp in spectra[1:]:
        np.testing.assert_allclose(sp, ref, rtol=0, atol=1e-10)


def test_eigh_rank_deficient_returns_three_unit_norm_modes():
    """Rank-3 snapshot field returns three modes, each with unit weighted norm."""
    q, w = _rank_deficient_field()
    modes, lam, _c = weighted_second_order(q, w, method="eigh")
    assert lam.size == 3
    assert np.all(lam >= 0.0)
    norms = np.sqrt(np.sum(w[:, None] * modes**2, axis=0))
    np.testing.assert_allclose(norms, np.ones(3), rtol=0, atol=1e-8)


def test_eigh_spatial_branch_rank_deficient_uses_n_space():
    """Spatial branch (n_samples >= n_space): rank-2 field returns two modes.

    n_kernel must be n_space here — using n_samples would mis-place the cutoff.
    """
    # nt=40, nsp=8 → spatial kernel (8 × 8)
    q, w = _rank_deficient_field(nt=40, nsp=8, r=2, scale=1.0, seed=3)
    modes, lam, _c = weighted_second_order(q, w, method="eigh")
    assert lam.size == 2
    assert modes.shape == (8, 2)
    assert np.all(lam >= 0.0)
    norms = np.sqrt(np.sum(w[:, None] * modes**2, axis=0))
    np.testing.assert_allclose(norms, np.ones(2), rtol=0, atol=1e-8)


def test_eigh_svd_agree_mode_count_in_spatial_regime():
    """eigh and svd keep the same mode count on centered spatial-regime data.

    Mean-centering costs one SAMPLE degree of freedom, so the rank bound is
    ``min(n_samples - 1, n_space)``. The old SVD cap ``min(n_samples, n_space) - 1``
    drops one genuine mode whenever ``n_space < n_samples``. Full-column-rank
    random data makes every spatial direction energetic so the floor cannot
    hide the off-by-one. Goes RED if ``_solve_svd`` reverts that bound.
    """
    rng = np.random.default_rng(0)
    cases = [(8, 3), (40, 25), (12, 4)]
    for n_samples, n_space in cases:
        assert n_space < n_samples, (n_samples, n_space)
        x = rng.standard_normal((n_samples, n_space))
        xc = x - x.mean(axis=0)
        assert np.linalg.matrix_rank(xc, tol=1e-10) == n_space
        w = np.ones(n_space)
        n_eigh = weighted_second_order(xc, w, method="eigh")[1].size
        n_svd = weighted_second_order(xc, w, method="svd")[1].size
        want = min(n_samples - 1, n_space)
        assert n_eigh == want, f"{n_samples}x{n_space}: eigh={n_eigh} want={want}"
        assert n_svd == want, f"{n_samples}x{n_space}: svd={n_svd} want={want}"
        assert n_eigh == n_svd
