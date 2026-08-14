from __future__ import annotations

import h5py
import numpy as np
import pytest

from openmodalpy import MPODAnalyzer, PODAnalyzer
from tests.reference_helpers import canonicalize_reference

# Measured max |oracle − analyzer| on eigenvalues and |modes| was 0.0 for the
# two-tone multi-band case below (same rfft/irfft + eigh construction). Bound
# is a few ulps of O(1) so a 5% eigenvalue shift still fails hard.
_BAND_ORACLE_RTOL = 1e-12
_BAND_ORACLE_ATOL = 1e-12


def _make_uniform_data(q: np.ndarray, dt: float = 1.0) -> dict:
    n_space = q.shape[1]
    return {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": dt,
        "Nx": n_space,
        "Ny": 1,
        "Ns": q.shape[0],
    }


def _normalized(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _independent_band_weighted_pod(
    data_centered: np.ndarray,
    weight_vector: np.ndarray,
    n_modes_save: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted snapshot POD with plain numpy (no openmodalpy helpers).

    Stated construction: weight the centered snapshots by exact sqrt(W), form
    the Gram matrix / n_snapshots, eigh, drop eigenvalues at or below
    ``n_kernel * eps * peak`` (peak = largest eigenvalue, n_kernel = Gram
    matrix size, eps = machine epsilon of the working dtype), recover spatial
    modes. Snapshot-space path when n_t < n_x, otherwise spatial-space path.
    """
    n_snapshots, n_space = data_centered.shape
    sqrt_w = np.sqrt(weight_vector)
    data_w = data_centered * sqrt_w

    if n_snapshots < n_space:
        n_kernel = n_snapshots
        kernel = np.dot(data_w, data_w.T) / n_snapshots
        eigenvalues, temporal = np.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[order])
        temporal = temporal[:, order]
        peak = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            return np.empty((n_space, 0)), np.empty((0,))
        cutoff = float(n_kernel) * float(np.finfo(eigenvalues.dtype).eps) * peak
        significant = eigenvalues > cutoff
        eigenvalues = eigenvalues[significant]
        temporal = temporal[:, significant]
        if eigenvalues.size == 0:
            return np.empty((n_space, 0)), np.empty((0,))
        keep = min(n_modes_save, eigenvalues.size)
        eigenvalues = eigenvalues[:keep]
        temporal = temporal[:, :keep]
        scale = 1.0 / np.sqrt(eigenvalues * n_snapshots)
        modes_w = np.dot(data_w.T, temporal) * scale
        modes = modes_w / sqrt_w[:, np.newaxis]
    else:
        n_kernel = n_space
        kernel = np.dot(data_w.T, data_w) / n_snapshots
        eigenvalues, modes_w = np.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[order])
        modes_w = modes_w[:, order]
        peak = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            return np.empty((n_space, 0)), np.empty((0,))
        cutoff = float(n_kernel) * float(np.finfo(eigenvalues.dtype).eps) * peak
        significant = eigenvalues > cutoff
        eigenvalues = eigenvalues[significant]
        modes_w = modes_w[:, significant]
        if eigenvalues.size == 0:
            return np.empty((n_space, 0)), np.empty((0,))
        keep = min(n_modes_save, eigenvalues.size)
        eigenvalues = eigenvalues[:keep]
        modes_w = modes_w[:, :keep]
        modes = modes_w / sqrt_w[:, np.newaxis]

    return np.real(modes), np.real(eigenvalues)


def _oracle_pooled_order(eigenvalues: np.ndarray, band_ids: np.ndarray) -> np.ndarray:
    """Pooled-mode order: energy descending, ties by band then position.

    Eigenvalues that agree to within 1e-12 relative are treated as tied
    (matches the library's stated convention; literal so this oracle stays
    independent of the implementation under test). Inside a tied group the
    order is band index ascending, then position within that band.
    """
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    bands = np.asarray(band_ids).reshape(-1)
    n = int(values.size)
    if n == 0:
        return np.zeros(0, dtype=int)

    # 1e-12 matches the library's stated relative-tie convention.
    tie_rtol = 1e-12

    by_energy = np.argsort(-values, kind="stable")
    group_rank = np.empty(n, dtype=int)
    cursor = 0
    rank = 0
    while cursor < n:
        peak = float(values[int(by_energy[cursor])])
        floor = peak - tie_rtol * abs(peak)
        stop = cursor + 1
        while stop < n and float(values[int(by_energy[stop])]) >= floor:
            stop += 1
        group_rank[by_energy[cursor:stop]] = rank
        rank += 1
        cursor = stop

    # lexsort uses the last key as primary: group (energy), then band, then index.
    positions = np.arange(n)
    return np.lexsort((positions, bands, group_rank))


def test_band_oracle_answer_is_linear_in_the_measure():
    """The oracle's own cutoff must carry no absolute scale.

    Spatial weights are a quadrature measure, so multiplying them by any
    positive factor must multiply every eigenvalue by that same factor and keep
    the same modes. An absolute floor (the ``1e-12`` this oracle used to carry)
    breaks that: at a measure of order 1e-14 the floor dominates and the oracle
    silently keeps fewer modes than it should — while still agreeing with the
    library on the fixtures below, which sit far from the cutoff. That is why
    this property needs its own test rather than riding on the band tests.
    """
    rng = np.random.default_rng(23)
    data = rng.standard_normal((14, 6))
    data -= data.mean(axis=0)
    weights = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    _modes, reference = _independent_band_weighted_pod(data, weights, 4)
    assert reference.size == 4, f"fixture kept {reference.size} modes, expected 4"

    for scale in (1e-14, 1e-6, 1e6, 1e14):
        _m, scaled = _independent_band_weighted_pod(data, weights * scale, 4)
        assert scaled.size == reference.size, (
            f"measure x {scale:.0e}: kept {scaled.size} modes, reference kept {reference.size}"
        )
        np.testing.assert_allclose(
            scaled / scale,
            reference,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"measure x {scale:.0e}: the oracle's spectrum is not linear in the measure",
        )


def _independent_multiband_mpod(
    q: np.ndarray,
    dt: float,
    band_edges_hz: list[float],
    n_modes_save: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent multi-band mPOD reference: center → rfft → band mask → irfft → POD.

    Bands are half-open on interior edges ([f_low, f_high) for non-last bands;
    the last band is closed on the right: [f_low, f_high]). Uniform spatial
    weights. Modes from all bands are concatenated and re-sorted by eigenvalue
    descending, then truncated to n_modes_save — matching the stated perform_mpod
    assembly, built here without calling library band/POD helpers.
    """
    n_snapshots, n_space = q.shape
    weight_vector = np.ones(n_space, dtype=float)
    centered = q - np.mean(q, axis=0, dtype=np.float64)
    freq = np.fft.rfftfreq(n_snapshots, d=dt)
    qhat = np.fft.rfft(centered, axis=0)

    band_modes: list[np.ndarray] = []
    band_eigs: list[np.ndarray] = []
    band_ids: list[np.ndarray] = []
    band_mode_counts: list[int] = []
    n_bands = len(band_edges_hz) - 1

    for band_index in range(n_bands):
        f_low = band_edges_hz[band_index]
        f_high = band_edges_hz[band_index + 1]
        is_last = band_index == n_bands - 1
        mask = (freq >= f_low) & ((freq <= f_high) if is_last else (freq < f_high))
        if not np.any(mask):
            band_mode_counts.append(0)
            continue
        qhat_band = np.zeros_like(qhat)
        qhat_band[mask, :] = qhat[mask, :]
        data_band = np.real(np.fft.irfft(qhat_band, n=n_snapshots, axis=0))
        modes, eigenvalues = _independent_band_weighted_pod(data_band, weight_vector, n_modes_save)
        if eigenvalues.size == 0:
            band_mode_counts.append(0)
            continue
        band_modes.append(modes)
        band_eigs.append(eigenvalues)
        band_ids.append(np.full(eigenvalues.size, band_index, dtype=int))
        band_mode_counts.append(eigenvalues.size)

    eigenvalues = np.concatenate(band_eigs)
    modes = np.concatenate(band_modes, axis=1)
    ids = np.concatenate(band_ids)
    order = _oracle_pooled_order(eigenvalues, ids)
    keep = min(n_modes_save, eigenvalues.size)
    return (
        np.real(eigenvalues[order][:keep]),
        np.real(modes[:, order][:, :keep]),
        ids[order][:keep],
        np.asarray(band_mode_counts, dtype=int),
    )


def test_single_band_mpod_matches_pod():
    data = _make_uniform_data(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [4.0, 3.0],
            ],
            dtype=float,
        )
    )

    pod = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    pod.load_and_preprocess()
    pod.perform_pod()

    mpod = MPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 0.5],
    )
    mpod.load_and_preprocess()
    mpod.perform_mpod()

    np.testing.assert_allclose(mpod.eigenvalues, pod.eigenvalues, rtol=1e-10, atol=1e-10)
    # Both routes go through the same seam, so signs match without |.|.
    np.testing.assert_allclose(mpod.modes, pod.modes, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mpod.time_coefficients, pod.time_coefficients, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(mpod.mode_band_indices, np.zeros(2, dtype=int))


def test_mpod_separates_modes_by_frequency_band():
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )
    data = _make_uniform_data(q, dt=dt)

    analyzer = MPODAnalyzer(
        file_path="toy_signal",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    assert set(analyzer.mode_band_indices.tolist()) == {0, 1}

    low_mode = analyzer.modes[:, np.where(analyzer.mode_band_indices == 0)[0][0]]
    high_mode = analyzer.modes[:, np.where(analyzer.mode_band_indices == 1)[0][0]]
    assert abs(np.dot(_normalized(low_mode), phi_low)) > 0.95
    assert abs(np.dot(_normalized(high_mode), phi_high)) > 0.95


def test_mpod_modes_not_orthonormal_across_bands():
    """Pooled mPOD modes are not jointly W-orthonormal across bands.

    Implementation is POD-per-band then concatenate/re-sort — no joint
    orthonormalization. Cross-band entries of Φᵀ W Φ must be materially
    nonzero so a future fix cannot land silently.
    """
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    # Overlapping spatial support so independent band PODs are not automatically
    # orthogonal under the uniform metric.
    phi_mid = _normalized(np.array([1.0, 1.0, 0.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.9 * np.sin(2 * np.pi * 3.0 * t)[:, None] * phi_mid[None, :]
        + 0.7 * np.sin(2 * np.pi * 6.0 * t)[:, None] * phi_high[None, :]
    )
    data = _make_uniform_data(q, dt=dt)

    analyzer = MPODAnalyzer(
        file_path="toy_signal",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=6,
        band_edges=[0.0, 2.0, 4.5, 8.0],
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    bands = analyzer.mode_band_indices
    assert len(set(bands.tolist())) >= 2, "need modes from at least two bands"

    W = np.asarray(analyzer.W).reshape(-1)
    phi = analyzer.modes  # (Nspace, n_modes)
    gram = phi.T @ (W[:, None] * phi)

    # Collect off-diagonal entries between modes from different bands.
    cross = []
    for i in range(phi.shape[1]):
        for j in range(i + 1, phi.shape[1]):
            if bands[i] != bands[j]:
                cross.append(abs(gram[i, j]))
    assert cross, "no cross-band mode pairs to check"
    max_cross = max(cross)
    # Materially nonzero: not a floating-point residual of orthonormality.
    assert max_cross > 1e-2, f"expected material cross-band W-inner product, got max |ΦᵀWΦ|_cross = {max_cross}"


def test_mpod_save_results_records_band_metadata(tmp_path):
    data = _make_uniform_data(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        dt=0.1,
    )

    analyzer = MPODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
        filter_kind="rectangular",
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()
    analyzer.save_results("mpod_contract.hdf5")

    with h5py.File(tmp_path / "mpod_contract.hdf5", "r") as handle:
        assert handle.attrs["lift_kind"] == "multiscale_filtered_snapshots"
        assert handle.attrs["filter_kind"] == "rectangular"
        np.testing.assert_allclose(handle.attrs["band_edges_hz"], np.array([0.0, 2.0, 5.0]))
        np.testing.assert_array_equal(handle.attrs["mode_band_indices"], analyzer.mode_band_indices)


def test_mpod_save_load_roundtrip_arrays(tmp_path):
    """mPOD inherits PODAnalyzer.save_results; load_results restores arrays + band attrs.

    Note: mPOD does not define its own save_results — it uses POD's. That path
    still writes modes/eigenvalues/time_coefficients and algorithm metadata, so
    a full round-trip is possible (contrary to the goal-file expectation that
    only the base-class stub was available).
    """
    data = _make_uniform_data(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        dt=0.1,
    )
    analyzer = MPODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
        filter_kind="rectangular",
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()
    analyzer.save_results("mpod_roundtrip.hdf5")

    reloaded = MPODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
        filter_kind="rectangular",
    )
    reloaded.load_results("mpod_roundtrip.hdf5")

    np.testing.assert_array_equal(reloaded.modes, analyzer.modes)
    np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
    np.testing.assert_array_equal(reloaded.time_coefficients, analyzer.time_coefficients)
    np.testing.assert_array_equal(reloaded.mode_band_indices, analyzer.mode_band_indices)
    np.testing.assert_array_equal(reloaded.band_mode_counts, analyzer.band_mode_counts)


def test_mpod_band_oracle_matches_two_tone():
    """Multi-band mPOD eigenvalues and |modes| match an independent numpy oracle.

    WHY: the band loop (rfft → rectangular mask → irfft → weighted POD per band,
    then concatenate and re-sort) is the path a future solver unification will
    touch. Building that construction here with plain numpy — not the library's
    band/POD helpers — pins the numbers so a 5% eigenvalue shift fails this test
    while the shape-correlation check alone would still pass.

    Measured max |oracle − analyzer| on this case: 0.0 (eigs and |modes|).
    """
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )
    band_edges = [0.0, 2.0, 5.0]
    n_modes = 2
    data = _make_uniform_data(q, dt=dt)

    analyzer = MPODAnalyzer(
        file_path="band_oracle_two_tone",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes,
        band_edges=band_edges,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    ref_eigs, ref_modes, ref_bands, ref_counts = _independent_multiband_mpod(q, dt, band_edges, n_modes)

    # Both bands carry energy (1 Hz and 4 Hz tones).
    assert set(analyzer.mode_band_indices.tolist()) == {0, 1}
    assert set(ref_bands.tolist()) == {0, 1}

    np.testing.assert_allclose(analyzer.eigenvalues, ref_eigs, rtol=_BAND_ORACLE_RTOL, atol=_BAND_ORACLE_ATOL)
    # Independent oracle uses plain eigh; apply the same sign rule as the seam.
    ref_modes, _ = canonicalize_reference(ref_modes)
    np.testing.assert_allclose(analyzer.modes, ref_modes, rtol=_BAND_ORACLE_RTOL, atol=_BAND_ORACLE_ATOL)
    np.testing.assert_array_equal(analyzer.mode_band_indices, ref_bands)
    np.testing.assert_array_equal(analyzer.band_mode_counts, ref_counts)


def test_mpod_band_oracle_interior_edge_half_open():
    """A pure tone exactly on an interior band edge lands in one band only.

    WHY: perform_mpod uses half-open interior intervals — non-last bands take
    freq < f_high; only the last band is closed on the right (freq <= f_high).
    A component at the interior edge must not be double-counted or dropped.
    Here f = 2.0 Hz sits on the shared edge of [0, 2) U [2, 5]; it belongs to
    band 1 alone. Oracle and analyzer must agree on that assignment.

    Measured max |oracle − analyzer| on this case: 0.0 (eigs and |modes|).
    """
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    # df = 1/(ns*dt) = 0.1 Hz so f_edge = 2.0 is an exact rfft bin.
    f_edge = 2.0
    band_edges = [0.0, f_edge, 5.0]
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_edge = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    # 1 Hz lives strictly inside band 0; f_edge sits on the interior boundary.
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + np.cos(2 * np.pi * f_edge * t)[:, None] * phi_edge[None, :]
    )
    n_modes = 4
    data = _make_uniform_data(q, dt=dt)

    analyzer = MPODAnalyzer(
        file_path="band_oracle_interior_edge",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes,
        band_edges=band_edges,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    ref_eigs, ref_modes, ref_bands, ref_counts = _independent_multiband_mpod(q, dt, band_edges, n_modes)

    # Content in both bands; edge tone must not vanish or double-count.
    assert set(analyzer.mode_band_indices.tolist()) == {0, 1}
    assert list(analyzer.band_mode_counts) == list(ref_counts)
    # Exactly one energetic mode per band for pure sinusoids after centering.
    assert analyzer.band_mode_counts[0] == 1
    assert analyzer.band_mode_counts[1] == 1

    # Spatial shapes: low tone → band 0, edge tone → band 1 (not band 0).
    low_idx = int(np.where(analyzer.mode_band_indices == 0)[0][0])
    edge_idx = int(np.where(analyzer.mode_band_indices == 1)[0][0])
    low_mode = _normalized(analyzer.modes[:, low_idx])
    edge_mode = _normalized(analyzer.modes[:, edge_idx])
    assert abs(np.dot(low_mode, phi_low)) > 0.95
    assert abs(np.dot(edge_mode, phi_edge)) > 0.95
    assert abs(np.dot(low_mode, phi_edge)) < 0.1
    assert abs(np.dot(edge_mode, phi_low)) < 0.1

    np.testing.assert_allclose(analyzer.eigenvalues, ref_eigs, rtol=_BAND_ORACLE_RTOL, atol=_BAND_ORACLE_ATOL)
    ref_modes, _ = canonicalize_reference(ref_modes)
    np.testing.assert_allclose(analyzer.modes, ref_modes, rtol=_BAND_ORACLE_RTOL, atol=_BAND_ORACLE_ATOL)
    np.testing.assert_array_equal(analyzer.mode_band_indices, ref_bands)


def test_mpod_accepts_a_non_square_weight_matrix_row_major():
    """Non-square analyzer.W is flattened row-major at the weight seam.

    Same modes/eigenvalues as the explicitly hand-written row-major vector.
    The reference is written out by hand (not W.reshape(-1) from the library
    path) so a column-major flip of the flatten order fails this test.

    Multi-band edges are required: a single full band collapses to
    perform_pod, which rejects this (3, 2) W as an unexpected shape rather
    than flattening it, so that route never reaches the seam pinned here.
    (perform_pod used to overwrite W under spatial_weight_type="uniform";
    it no longer does, so the reason changed but the requirement did not.)
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal((40, 6))
    data = _make_uniform_data(q, dt=0.05)  # nyquist = 10 Hz
    # (3, 2) non-square; explicit row-major reference: [1, 2, 3, 4, 5, 6]
    W_non_square = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    W_row_major = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    W_column_major = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    band_edges = [0.0, 2.0, 10.0]

    def run(weights):
        analyzer = MPODAnalyzer(
            file_path="dummy",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=2,
            band_edges=band_edges,
        )
        analyzer.load_and_preprocess()
        analyzer.W = weights
        analyzer.perform_mpod()
        return analyzer

    a_matrix = run(W_non_square)
    a_row = run(W_row_major.reshape(6, 1))
    a_column = run(W_column_major.reshape(6, 1))

    np.testing.assert_allclose(a_matrix.eigenvalues, a_row.eigenvalues, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a_matrix.modes, a_row.modes, rtol=1e-12, atol=1e-12)

    # Agreement alone would also hold if the weights stopped reaching the
    # eigenproblem at all -- every ordering agrees when none of them is used.
    # The column-major control is what makes this a statement about the ORDER:
    # measured separation is 19% in the eigenvalues, against 1e-12 agreement
    # above, so 1% is far outside numerical noise and far inside the signal.
    relative_shift = np.max(np.abs(a_matrix.eigenvalues - a_column.eigenvalues)) / np.max(np.abs(a_matrix.eigenvalues))
    assert relative_shift > 0.01, (
        f"column-major weights gave the same answer as row-major (relative eigenvalue shift "
        f"{relative_shift:.3g}), so this test is not sensitive to the flatten order"
    )


def test_mpod_rejects_a_zero_measure_weight():
    """Assigning a zero-measure W after load must fail at perform_mpod.

    Direct assignment is the route under test; the prescribed-at-load path
    validates earlier and is a different gate.
    """
    # The weight check runs before any decomposition, so only the shape of q
    # matters here: enough snapshots to load, and 6 spatial points.
    q = np.random.default_rng(0).standard_normal((4, 6))
    data = _make_uniform_data(q, dt=0.1)
    analyzer = MPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 5.0],
    )
    analyzer.load_and_preprocess()
    analyzer.W = np.zeros((6, 1))
    with pytest.raises(ValueError, match="zero total measure"):
        analyzer.perform_mpod()


def test_mpod_figures_are_named_mpod(tmp_path):
    """mPOD eigenvalue figures use _mpod_ and leave no _pod_ figure behind."""
    q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    data = _make_uniform_data(q, dt=0.1)
    analyzer = MPODAnalyzer(
        file_path="case_mpod",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 5.0],
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()
    analyzer.plot_eigenvalues()

    expected = tmp_path / f"{analyzer.data_root}_mpod_eigenvalues.png"
    assert expected.is_file(), f"missing {expected.name}; dir={list(tmp_path.iterdir())}"
    leftovers = sorted(tmp_path.glob(f"{analyzer.data_root}_pod_*.png"))
    assert leftovers == [], f"mPOD run left POD-named figures: {[p.name for p in leftovers]}"


def test_pod_figure_names_are_unchanged(tmp_path):
    """POD still writes <case>_pod_*.png — regression guard for the label substitution."""
    q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    data = _make_uniform_data(q, dt=0.1)
    analyzer = PODAnalyzer(
        file_path="case_pod",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.plot_eigenvalues()

    expected = tmp_path / f"{analyzer.data_root}_pod_eigenvalues.png"
    assert expected.is_file(), f"missing {expected.name}; dir={list(tmp_path.iterdir())}"
    mpod_named = sorted(tmp_path.glob(f"{analyzer.data_root}_mpod_*.png"))
    assert mpod_named == [], f"POD run wrote mPOD-named figures: {[p.name for p in mpod_named]}"


def test_mpod_tied_band_order_is_platform_independent(monkeypatch):
    """Near-tied band energies keep the same column order on every machine.

    Feed the same two-band pool twice: once with the 4-ulp-class perturbation
    making band 0 infinitesimally larger, once the other way. Both must return
    the band-ascending order. A raw argsort cannot see the tie, so this reds
    on the pre-fix sort and greens after.
    """
    import openmodalpy.mpod as mpod_mod

    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )
    data = _make_uniform_data(q, dt=dt)

    base = 0.5
    # Inside the 1e-12 relative tie band, well above a few ulps so equality
    # cannot hide a swap, and well below any physically distinct energy.
    delta = 5e-14

    def run(sign: float) -> MPODAnalyzer:
        calls = {"n": 0}

        def fake_wso(data_band, metric, method="eigh", n_keep=10):
            i = calls["n"]
            calls["n"] += 1
            n_space = data_band.shape[1]
            n_time = data_band.shape[0]
            modes = np.zeros((n_space, 1))
            modes[i, 0] = 1.0
            eigenvalues = np.array([base + sign * (1.0 if i == 0 else -1.0) * delta])
            coeffs = np.zeros((n_time, 1))
            coeffs[i, 0] = 1.0
            return modes, eigenvalues, coeffs

        monkeypatch.setattr(mpod_mod.decomposition, "weighted_second_order", fake_wso)
        analyzer = MPODAnalyzer(
            file_path="tied_band_order",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=2,
            band_edges=[0.0, 2.0, 5.0],
        )
        analyzer.load_and_preprocess()
        analyzer.perform_mpod()
        assert calls["n"] == 2, f"expected one solver call per band, got {calls['n']}"
        return analyzer

    a_plus = run(+1.0)
    a_minus = run(-1.0)

    expected_bands = np.array([0, 1])
    np.testing.assert_array_equal(a_plus.mode_band_indices, expected_bands)
    np.testing.assert_array_equal(a_minus.mode_band_indices, expected_bands)
    np.testing.assert_array_equal(a_plus.modes, a_minus.modes)
    np.testing.assert_array_equal(a_plus.time_coefficients, a_minus.time_coefficients)


def test_oracle_pooled_order_is_platform_independent(monkeypatch):
    """Near-tied pooled energies keep the same oracle column order.

    Feed the same two-band pool twice: once with a perturbation inside the
    1e-12 relative tie band making band 0 infinitesimally larger, once the
    other way. Both must return the band-ascending order. A raw argsort
    cannot see the tie, so this reds on HEAD's oracle and greens after.
    """
    base = 0.5
    # Inside the 1e-12 relative tie band, well above a few ulps so equality
    # cannot hide a swap, and well below any physically distinct energy.
    delta = 5e-14

    def helper_order(sign: float) -> np.ndarray:
        eigenvalues = np.array([base + sign * delta, base - sign * delta], dtype=float)
        band_ids = np.array([0, 1], dtype=int)
        return _oracle_pooled_order(eigenvalues, band_ids)

    expected = np.array([0, 1])
    np.testing.assert_array_equal(helper_order(+1.0), expected)
    np.testing.assert_array_equal(helper_order(-1.0), expected)

    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )

    def run(sign: float):
        calls = {"n": 0}

        def fake_band_pod(data_centered, weight_vector, n_modes_save):
            i = calls["n"]
            calls["n"] += 1
            n_space = data_centered.shape[1]
            modes = np.zeros((n_space, 1))
            modes[i, 0] = 1.0
            eigenvalues = np.array([base + sign * (1.0 if i == 0 else -1.0) * delta])
            return modes, eigenvalues

        monkeypatch.setattr(
            "tests.test_mpod._independent_band_weighted_pod",
            fake_band_pod,
        )
        result = _independent_multiband_mpod(q, dt, [0.0, 2.0, 5.0], 2)
        assert calls["n"] == 2, f"expected one oracle POD call per band, got {calls['n']}"
        return result

    _plus_eigs, plus_modes, plus_bands, _plus_counts = run(+1.0)
    _minus_eigs, minus_modes, minus_bands, _minus_counts = run(-1.0)

    expected_bands = np.array([0, 1])
    np.testing.assert_array_equal(plus_bands, expected_bands)
    np.testing.assert_array_equal(minus_bands, expected_bands)
    np.testing.assert_array_equal(plus_modes, minus_modes)


def test_dmd_log_pattern_accepts_a_windows_path():
    """The DMD save-path pin must match a Windows results path, not only POSIX."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent / "test_logging_quiet.py").read_text(encoding="utf-8")
    match = re.search(
        r"re\.compile\(r([\"'])((?:(?!\1).)*_dmd(?:(?!\1).)*)\1\)",
        text,
    )
    assert match, "test_logging_quiet.py has no re.compile(r'...') pinning a *_dmd path"
    pattern_text = match.group(2)
    windows_msg = r"DMD results saved to C:\Users\runner\work\out\case_dmd.hdf5"
    assert re.search(pattern_text, windows_msg), (
        f"DMD log pattern does not accept a Windows path; compiled {pattern_text!r} against {windows_msg!r}"
    )
