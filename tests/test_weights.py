import warnings

import numpy as np
import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    PSDPODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
)
from openmodalpy.core.base import (
    BaseAnalyzer,
    _coerce_spatial_weights,
    _polar_theta_sector_fractions,
    calculate_polar_weights,
    calculate_uniform_weights,
    require_spatial_metric,
)
from openmodalpy.core.decomposition import (
    SpatialMetric,
    _as_weight_vector,
    apply_sqrt_metric,
    weighted_total_energy,
)
from openmodalpy.core.parallel import calculate_polar_weights_optimized


def test_square_weight_matrix_yields_diagonal():
    """A diagonal spatial metric stored as a full matrix keeps its diagonal."""
    diag = np.array([0.5, 1.0, 2.0, 0.25])
    W = np.diag(diag)
    col = _coerce_spatial_weights(W, 4).reshape(-1, 1)
    vec = _as_weight_vector(W, 4)
    assert col.shape == (4, 1)
    assert vec.shape == (4,)
    np.testing.assert_allclose(col.ravel(), diag)
    np.testing.assert_allclose(vec, diag)


def test_complex_weights_are_rejected_not_truncated():
    """A complex metric must fail loudly rather than lose its imaginary part."""
    W = np.array([1.0 + 0j, 2.0 + 1j, 3.0 + 0j])
    for entry in (
        lambda: require_spatial_metric(W),
        lambda: _coerce_spatial_weights(W, 3),
        lambda: _as_weight_vector(W, 3),
        lambda: _as_weight_vector(SpatialMetric(W), 3),
        lambda: SpatialMetric(W),
    ):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="complex"):
                entry()
        assert not any(w.category is np.exceptions.ComplexWarning for w in rec)


@pytest.mark.parametrize(
    "W",
    [
        np.diag([0.5, 1.0, 2.0, 0.25]),
        np.ones((2, 2)),
        np.arange(1.0, 10.0).reshape(3, 3),
        np.eye(5),
    ],
    ids=["diagonal-4x4", "smallest-square-2x2", "non-diagonal-3x3", "identity-5x5"],
)
def test_spatial_metric_rejects_a_square_matrix_by_name(W):
    """Any square matrix is ambiguous for a diagonal-only container — refuse it.

    Several shapes, and a non-diagonal one, so a guard that only recognised the
    one fixture below would not pass this.
    """
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        SpatialMetric(W)
    msg = str(info.value)
    # Names the shape the caller actually passed, and points at the fix.
    assert str(W.shape) in msg
    # The old error reported the flattened length, which told the caller nothing.
    assert f"length {W.size}" not in msg


@pytest.mark.parametrize(
    "w3",
    [
        np.stack([np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])], axis=2),
        np.ones((1, 1, 1)),
        np.ones((2, 2, 1)),
        np.ones((2, 3, 4)),
    ],
    ids=["per-component-3x3x2", "unit-1x1x1", "single-component-2x2x1", "unequal-first-dims-2x3x4"],
)
def test_spatial_metric_rejects_3d_weights_by_name(w3):
    """Any 3-D weight array is refused, whatever its dimensions."""
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        SpatialMetric(w3)
    msg = str(info.value)
    # Requiring the shape string, not just the words "3-D", keeps this honest:
    # the message must name what the caller actually passed.
    assert str(w3.shape) in msg
    assert f"length {w3.size}" not in msg


def test_spatial_metric_still_accepts_the_shapes_that_already_agreed():
    """1-D, column, row, 1x1 and non-square flatten the same way they always have.

    These are the shapes on which the wrapped and raw paths already agreed, so
    the rejection above must not have widened into them. 1x1 is deliberately
    included: it is square, but there is nothing ambiguous about a single
    weight, and the raw path treats it as a plain scalar too.
    """
    cases = (
        (np.array([1.0, 2.0, 3.0]), 3, [1.0, 2.0, 3.0]),
        (np.array([[1.0], [2.0], [3.0]]), 3, [1.0, 2.0, 3.0]),
        (np.array([[1.0, 2.0, 3.0]]), 3, [1.0, 2.0, 3.0]),
        (np.array([[7.0]]), 1, [7.0]),
        (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    for w, n, expect in cases:
        got = _as_weight_vector(SpatialMetric(w), n)
        np.testing.assert_array_equal(got, np.asarray(expect, dtype=float))


def test_three_d_weights_with_equal_space_and_components():
    """Per-component 3-D weights must not fall through into a second diagonal.

    When the stacked (n, k) matrix is square (n == k), the 3-D branch alone
    is the correct route: flatten the stacked diagonals, do not re-diag.
    """
    w3 = np.stack([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])], axis=2)
    got = _coerce_spatial_weights(w3, 4)
    np.testing.assert_allclose(got, [1.0, 3.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="n_space=2"):
        _coerce_spatial_weights(w3, 2)


def test_three_d_per_component_route_through_the_seam():
    """3-D per-component weights reach the seam as stacked diagonals.

    Pins ``_as_weight_vector`` (not only ``_coerce_spatial_weights``) so a
    future narrowing of the seam cannot drop the 3-D route silently.
    """
    # shape (3, 3, 2): two component diagonals -> n_space=6
    w3 = np.stack(
        [np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])],
        axis=2,
    )
    got = _as_weight_vector(w3, 6)
    np.testing.assert_allclose(got, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])


def test_nondiagonal_square_raises_instead_of_truncating():
    """A square W with real off-diagonals must not be silently reduced to its diagonal."""
    W = np.diag([1.0, 2.0, 3.0, 4.0])
    W[0, 1] = W[1, 0] = 0.5
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        _coerce_spatial_weights(W, 4)
    msg = str(info.value)
    assert str(W.shape) in msg
    assert "0.5" in msg
    assert "off-diag" in msg.lower()


def test_roundoff_offdiagonals_are_accepted_but_1e8_is_not():
    """Diagonality is relative to the diagonal and tighter than np.allclose's default.

    1e-16 is round-off against O(1) entries and must still pass. 1e-8 is what
    np.allclose would wave through and must raise. A zero-diagonal matrix treats
    any non-zero off-diagonal as meaningful.
    """
    diag = np.array([1.0, 2.0, 3.0, 4.0])
    near = np.diag(diag)
    near[0, 1] = near[1, 0] = 1e-16
    np.testing.assert_allclose(_coerce_spatial_weights(near, 4), diag)

    too_big = np.diag(diag)
    too_big[0, 1] = too_big[1, 0] = 1e-8
    with pytest.raises(ValueError, match=r"np\.diag"):
        _coerce_spatial_weights(too_big, 4)

    zero_diag = np.zeros((3, 3))
    zero_diag[0, 1] = zero_diag[1, 0] = 1e-12
    with pytest.raises(ValueError, match=r"np\.diag"):
        _coerce_spatial_weights(zero_diag, 3)


def test_nondiagonal_3d_planes_raise_instead_of_truncating():
    """Each component plane is held to the same diagonality rule as a 2-D square."""
    w3 = np.zeros((3, 3, 2))
    for i in range(2):
        w3[:, :, i] = np.diag([1.0, 2.0, 3.0])
        w3[0, 1, i] = w3[1, 0, i] = 0.7
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        _coerce_spatial_weights(w3, 6)
    msg = str(info.value)
    assert str(w3.shape) in msg
    assert "0.7" in msg
    assert "off-diag" in msg.lower()
    assert "plane" in msg.lower()
    assert "square spatial metric" not in msg.lower()


def test_analyzer_spatial_weights_nondiagonal_square_raises(tmp_path):
    """load_and_preprocess must raise, not truncate, when W is a non-diagonal square.

    The constructor only stores the array; the raise is at load time.
    """
    n = 4
    bad = np.diag([1.0, 2.0, 3.0, 4.0])
    bad[0, 1] = bad[1, 0] = 0.5
    field = {
        "q": np.arange(float(12 * n)).reshape(12, n),
        "x": np.arange(n, dtype=float),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n,
        "Ny": 1,
        "Ns": 12,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weights=bad,
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "0.5" in msg
    assert "off-diag" in msg.lower()


def _coerce_accepts(W, n: int) -> bool:
    try:
        _coerce_spatial_weights(W, n)
        return True
    except ValueError:
        return False


def test_nan_or_inf_offdiagonal_raises():
    """A NaN or inf off-diagonal is discarded coupling, not round-off. Both must raise."""
    for bad in (np.nan, np.inf, -np.inf):
        W = np.diag([1.0, 2.0, 3.0, 4.0])
        W[0, 1] = W[1, 0] = bad
        with pytest.raises(ValueError, match=r"np\.diag"):
            _coerce_spatial_weights(W, 4)


def test_object_dtype_diagonal_is_not_rejected_for_coupling():
    """A diagonal object-dtype square is not off-diagonal coupling.

    ``np.isfinite`` raises TypeError on object arrays. The old guard turned
    that into ``return np.inf``, so a purely diagonal object matrix was
    rejected as coupled with magnitude 0.0. After the guard is gone the
    ratio is computed (numeric objects convert) and a later check may still
    refuse the dtype — that is a different, honest reason. Assert the
    reason, not merely that something raised. A genuinely coupled object
    square must still raise for coupling.
    """
    W = np.diag([1.0, 2.0, 3.0]).astype(object)
    try:
        got = _coerce_spatial_weights(W, 3)
    except Exception as exc:
        msg = str(exc)
        assert "off-diagonal coupling" not in msg, "diagonal object-dtype square was rejected as coupled: " + msg
        assert "largest off-diagonal magnitude 0.0" not in msg, msg
        assert isinstance(exc, (TypeError, ValueError)), type(exc)
    else:
        np.testing.assert_allclose(got.astype(float), [1.0, 2.0, 3.0])

    coupled = np.diag([1.0, 2.0, 3.0]).astype(object)
    coupled[0, 1] = coupled[1, 0] = 0.5
    with pytest.raises(ValueError, match=r"np\.diag") as info:
        _coerce_spatial_weights(coupled, 3)
    msg = str(info.value)
    assert "off-diag" in msg.lower()
    assert "0.5" in msg


def test_spread_diagonal_does_not_hide_local_coupling():
    """A coupling 1e4x the weights it couples must raise even next to a 1e16 entry."""
    W = np.diag([1e16, 1e-4, 1e-4, 1e-4])
    W[1, 2] = W[2, 1] = 1.0
    with pytest.raises(ValueError, match=r"np\.diag"):
        _coerce_spatial_weights(W, 4)


def test_positive_diagonal_rescaling_preserves_verdict():
    """Rescaling W -> S W S by a positive diagonal must not change accept/reject."""
    S = np.diag([1e-6, 1e3, 1.0, 1e6])

    reject = np.diag([1.0, 2.0, 3.0, 4.0])
    reject[0, 1] = reject[1, 0] = 0.5
    assert _coerce_accepts(reject, 4) is False
    assert _coerce_accepts(S @ reject @ S, 4) is False

    accept = np.diag([1.0, 2.0, 3.0, 4.0])
    accept[0, 1] = accept[1, 0] = 1e-16
    assert _coerce_accepts(accept, 4) is True
    assert _coerce_accepts(S @ accept @ S, 4) is True


def test_float32_after_arithmetic_is_accepted():
    """A float32 diagonal metric after arithmetic is judged with float32 eps, not float64.

    E.T @ D @ E with E = I + 1e-7 in float32 has measured r ≈ 2.10 float32-eps.
    Cycle 1 compared that fill-in to a float64 ulp and rejected it.
    """
    e32 = (np.eye(4) + 1e-7).astype(np.float32)
    d32 = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    m32 = (e32.T @ d32 @ e32).astype(np.float32)
    assert m32.dtype == np.float32
    got = _coerce_spatial_weights(m32, 4)
    np.testing.assert_allclose(got, np.diag(m32).astype(float))
    # A coupling far above C*n*eps64 and far below C*n*eps32 must follow float32.
    tiny = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    tiny[0, 1] = tiny[1, 0] = np.float32(1e-10)
    np.testing.assert_allclose(_coerce_spatial_weights(tiny, 4), [1.0, 2.0, 3.0, 4.0])


def test_measured_diagonality_table_lands_on_the_stated_side():
    """Every measured case in the cycle-2 table accepts or rejects as stated."""
    assert _coerce_accepts(np.diag([1.0, 2.0, 3.0, 4.0]), 4)  # 0.00 eps

    near = np.diag([1.0, 2.0, 3.0, 4.0])
    near[0, 1] = near[1, 0] = 1e-16
    assert _coerce_accepts(near, 4)  # 0.32 eps

    e32 = (np.eye(4) + 1e-7).astype(np.float32)
    d32 = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    m32 = (e32.T @ d32 @ e32).astype(np.float32)
    assert _coerce_accepts(m32, 4)  # 2.10 float32-eps

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(np.eye(4) + 1e-14 * rng.standard_normal((4, 4)))
    cob = q.T @ np.diag([1.0, 2.0, 3.0, 4.0]) @ q
    assert not _coerce_accepts(cob, 4)  # 157 float64-eps, real coupling

    too_big = np.diag([1.0, 2.0, 3.0, 4.0])
    too_big[0, 1] = too_big[1, 0] = 1e-8
    assert not _coerce_accepts(too_big, 4)  # 3.2e7 eps

    spread = np.diag([1e16, 1e-4, 1e-4, 1e-4])
    spread[1, 2] = spread[2, 1] = 1.0
    assert not _coerce_accepts(spread, 4)  # 4.5e19 eps


def test_uniform_weights_1d_vs_2d():
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 2.0, 3)
    x2d = np.repeat(x[:, None], len(y), axis=1)
    y2d = np.repeat(y[None, :], len(x), axis=0)
    w_1d = calculate_uniform_weights(x, y)
    w_2d = calculate_uniform_weights(x2d, y2d)
    assert w_1d.shape == (len(x) * len(y), 1)
    assert np.array_equal(w_1d, w_2d)


def test_polar_weights_1d_vs_2d():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x2d = np.repeat(x[:, None], len(y), axis=1)
    y2d = np.repeat(y[None, :], len(x), axis=0)
    w_1d = calculate_polar_weights(x, y, use_parallel=False)
    w_2d = calculate_polar_weights(x2d, y2d, use_parallel=False)
    assert w_1d.shape == (len(x) * len(y), 1)
    assert np.allclose(w_1d, w_2d)


def test_weights_with_npz_grid_fixture(tmp_path):
    x1d = np.array([0.0, 0.5, 1.5, 3.0])
    y1d = np.array([0.0, 0.25, 0.75, 1.5])
    x2d = np.repeat(x1d[:, None], len(y1d), axis=1)
    y2d = np.repeat(y1d[None, :], len(x1d), axis=0)

    fixture_path = tmp_path / "grid_fixture.npz"
    np.savez(fixture_path, x=x2d, y=y2d)

    npz = np.load(fixture_path)
    x2d = npz["x"]
    y2d = npz["y"]
    x1d = x2d[:, 0]
    y1d = y2d[0, :]
    w_uniform_1d = calculate_uniform_weights(x1d, y1d)
    w_uniform_2d = calculate_uniform_weights(x2d, y2d)
    assert np.allclose(w_uniform_1d, w_uniform_2d)
    w_polar_1d = calculate_polar_weights(x1d, y1d, use_parallel=False)
    w_polar_2d = calculate_polar_weights(x2d, y2d, use_parallel=False)
    assert np.allclose(w_polar_1d, w_polar_2d)


def test_uniform_weights_3d_length():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])
    w = calculate_uniform_weights(x, y, z)
    assert w.shape == (len(x) * len(y) * len(z), 1)


def test_uniform_weights_scattered_points_are_length_n():
    """1-D x, y of length n with n_space=n is a point cloud, not a tensor product."""
    n = 12
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 2.0, n)
    w = calculate_uniform_weights(x, y, n_space=n)
    assert w.shape == (n, 1)
    # n_space omitted keeps today's tensor product, so positional callers are safe.
    w_grid = calculate_uniform_weights(x, y)
    assert w_grid.shape == (n * n, 1)


def test_uniform_weights_n_equals_1_follows_n_space():
    """n == 1 is the only collision: scattered wins only when the width says so."""
    x = np.array([0.0])
    y = np.array([1.0])
    z = np.array([0.0, 1.0, 2.0])
    w_scattered = calculate_uniform_weights(x, y, z, n_space=1)
    assert w_scattered.shape == (1, 1)
    w_tensor = calculate_uniform_weights(x, y, z, n_space=3)
    assert w_tensor.shape == (1 * 1 * len(z), 1)
    w_default = calculate_uniform_weights(x, y, z)
    assert w_default.shape == (1 * 1 * len(z), 1)


_SENTINEL_GRID = {
    "q": np.arange(24.0).reshape(4, 6),
    "x": np.array([0.0, 1.0, 2.0]),
    "y": np.array([0.0, 1.0]),
    "dt": 1.0,
    "Nx": 3,
    "Ny": 2,
    "Ns": 4,
}


@pytest.mark.parametrize(
    "analyzer_cls, extra_kwargs",
    [
        (BaseAnalyzer, {}),
        (PODAnalyzer, {}),
        (MPODAnalyzer, {}),
        (SPODAnalyzer, {}),
        (STPODAnalyzer, {}),
        (DMDAnalyzer, {"rank": 2}),
        (BSMDAnalyzer, {}),
        (PSDPODAnalyzer, {}),
    ],
)
def test_omitted_weight_type_resolves_to_uniform(analyzer_cls, extra_kwargs):
    """Omitting the weight type means uniform, on every analyzer.

    The default lives as a separate ``spatial_weight_type=None`` in eight
    signatures, and only POD's is covered elsewhere. The second assert is the
    one that bites: ``None`` is a sentinel for "not specified", so replacing it
    with a plain ``"uniform"`` string would keep this first assert passing while
    making an array with no type raise instead of prescribing a metric.
    """
    kwargs = {"file_path": "dummy", "data_loader": lambda _: _SENTINEL_GRID, **extra_kwargs}
    assert analyzer_cls(**kwargs).spatial_weight_type == "uniform"

    weights = np.arange(1.0, 7.0)  # Nx * Ny = 6
    prescribed = analyzer_cls(spatial_weights=weights, **kwargs)
    assert prescribed.spatial_weight_type == "prescribed"


_SURVIVAL_NS = 16
_SURVIVAL_NX = 4
_SURVIVAL_NY = 2
_SURVIVAL_NSPACE = _SURVIVAL_NX * _SURVIVAL_NY
_SURVIVAL_T = np.linspace(0.0, 2.0 * np.pi, _SURVIVAL_NS, endpoint=False)
_SURVIVAL_X = np.linspace(0.0, 1.0, _SURVIVAL_NSPACE)
_SURVIVAL_GRID = {
    "q": 1.0 + np.outer(np.sin(_SURVIVAL_T), np.sin(2.0 * np.pi * _SURVIVAL_X)),
    "x": np.linspace(0.0, 1.0, _SURVIVAL_NX),
    "y": np.linspace(0.0, 1.0, _SURVIVAL_NY),
    "dt": 1.0,
    "Nx": _SURVIVAL_NX,
    "Ny": _SURVIVAL_NY,
    "Ns": _SURVIVAL_NS,
}
_SURVIVAL_WEIGHTS = np.linspace(0.5, 1.5, _SURVIVAL_NSPACE)


@pytest.mark.parametrize(
    "analyzer_cls, extra_kwargs, method, needs_fft",
    [
        (PODAnalyzer, {"n_modes_save": 2}, "perform_pod", False),
        (STPODAnalyzer, {"n_modes_save": 2, "embedding_dim": 2}, "perform_stpod", False),
        (MPODAnalyzer, {"n_modes_save": 2}, "perform_mpod", False),
        (DMDAnalyzer, {"rank": 2}, "perform_dmd", False),
        (SPODAnalyzer, {"nfft": 8, "overlap": 0.5}, "perform_spod", True),
        (BSMDAnalyzer, {"nfft": 8, "overlap": 0.5, "static_triads": [(0, 0, 0)]}, "perform_bsmd", True),
        (PSDPODAnalyzer, {"nfft": 8, "overlap": 0.5, "n_modes_save": 2}, "perform_psd_pod", True),
    ],
    ids=["POD", "ST-POD", "mPOD", "DMD", "SPOD", "BSMD", "PSD-POD"],
)
def test_prescribed_weights_survive_decomposition(analyzer_cls, extra_kwargs, method, needs_fft, tmp_path):
    """A prescribed metric must still be analyzer.W after the eigenproblem runs.

    Coverage used to exist only for POD. This checks presence, not use: the
    vector is still on the object afterwards. Whether the solver consulted it
    is a separate question, answered by
    ``test_prescribed_weights_change_the_eigenvalues`` below. What a run alone
    proves is that the decomposition completes at all with a prescribed metric,
    which BSMD did not — it raised on the flat prescribed shape.
    """
    analyzer = analyzer_cls(
        file_path="dummy",
        data_loader=lambda _: _SURVIVAL_GRID,
        spatial_weights=_SURVIVAL_WEIGHTS,
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        **extra_kwargs,
    )
    analyzer.load_and_preprocess()
    np.testing.assert_array_equal(np.asarray(analyzer.W).ravel(), _SURVIVAL_WEIGHTS)
    if needs_fft:
        analyzer.compute_fft_blocks()
    getattr(analyzer, method)()
    np.testing.assert_array_equal(np.asarray(analyzer.W).ravel(), _SURVIVAL_WEIGHTS)


# Equal-mean pair on purpose. ones vs linspace(0.2, 3.0) is blind here: the
# field is q = 1 + outer(sin(t), g) with g odd, so g² is a palindrome and every
# palindromic pair of that ramp sums to 3.2. The ramp then acts exactly like
# mean(ramp)*ones, and a solver that used only a scalar derived from W still
# passed. Both of these have mean 1.0; any movement is spatial structure.
_FLAT_WEIGHTS = np.ones(_SURVIVAL_NSPACE)
_BUMP_WEIGHTS = np.exp(-(((np.arange(_SURVIVAL_NSPACE) - 1.5) / 1.2) ** 2))
_BUMP_WEIGHTS = _BUMP_WEIGHTS / _BUMP_WEIGHTS.mean()


def _copy_survival_grid() -> dict:
    return {
        "q": np.array(_SURVIVAL_GRID["q"], copy=True),
        "x": np.array(_SURVIVAL_GRID["x"], copy=True),
        "y": np.array(_SURVIVAL_GRID["y"], copy=True),
        "dt": _SURVIVAL_GRID["dt"],
        "Nx": _SURVIVAL_GRID["Nx"],
        "Ny": _SURVIVAL_GRID["Ny"],
        "Ns": _SURVIVAL_GRID["Ns"],
    }


def _analyzer_for(analyzer_cls, extra_kwargs, method, needs_fft, weights, tmp_path, tag):
    analyzer = analyzer_cls(
        file_path="dummy",
        data_loader=lambda _: _copy_survival_grid(),
        spatial_weights=weights,
        use_parallel=False,
        results_dir=str(tmp_path / tag / "results"),
        figures_dir=str(tmp_path / tag / "figures"),
        **extra_kwargs,
    )
    analyzer.load_and_preprocess()
    if needs_fft:
        analyzer.compute_fft_blocks()
    getattr(analyzer, method)()
    return analyzer


@pytest.mark.parametrize(
    "analyzer_cls, extra_kwargs, method, needs_fft, uses_metric",
    [
        (PODAnalyzer, {"n_modes_save": 2}, "perform_pod", False, True),
        (STPODAnalyzer, {"n_modes_save": 2, "embedding_dim": 2}, "perform_stpod", False, True),
        (MPODAnalyzer, {"n_modes_save": 2}, "perform_mpod", False, True),
        (SPODAnalyzer, {"nfft": 8, "overlap": 0.5}, "perform_spod", True, True),
        (BSMDAnalyzer, {"nfft": 8, "overlap": 0.5, "static_triads": [(0, 0, 0)]}, "perform_bsmd", True, True),
        (PSDPODAnalyzer, {"nfft": 8, "overlap": 0.5, "n_modes_save": 2}, "perform_psd_pod", True, True),
        (DMDAnalyzer, {"rank": 2}, "perform_dmd", False, False),
    ],
    ids=["POD", "ST-POD", "mPOD", "SPOD", "BSMD", "PSD-POD", "DMD"],
)
def test_prescribed_weights_change_the_eigenvalues(
    analyzer_cls, extra_kwargs, method, needs_fft, uses_metric, tmp_path
):
    """Two equal-mean metrics must change the answer iff the solver uses W.

    Survival of analyzer.W after the run is not enough: a refactor could stop
    consulting the metric and leave the vector on the object. The pair is
    ones(n) against an off-centre Gaussian bump renormalised to mean 1. The
    means are equal on purpose: a ones-vs-ramp pair is isospectral on this
    field (g odd ⇒ g² palindromic, so the ramp equals mean(ramp)*ones). A
    solver that used only a scalar derived from W would still pass that pair.

    DMD documents at dmd.py:350 that the regression does not use the spatial
    metric. Eigenvalues are the wrong tripwire there — rank-2 DMD is
    isospectral under weighting on this fixture — so the check is on modes,
    which stay put today only because W is unused and would move if the
    regression started consulting it.
    """
    assert np.isclose(_FLAT_WEIGHTS.mean(), _BUMP_WEIGHTS.mean()), (
        "the pair must share a mean; otherwise a mean(W)*ones solver still passes"
    )
    flat = _analyzer_for(analyzer_cls, extra_kwargs, method, needs_fft, _FLAT_WEIGHTS, tmp_path, "flat")
    bump = _analyzer_for(analyzer_cls, extra_kwargs, method, needs_fft, _BUMP_WEIGHTS, tmp_path, "bump")
    evals_changed = flat.eigenvalues.shape != bump.eigenvalues.shape or not np.allclose(
        flat.eigenvalues, bump.eigenvalues
    )
    if uses_metric:
        assert evals_changed, (
            f"{analyzer_cls.__name__} eigenvalues were identical under ones() "
            "and an equal-mean off-centre bump; the metric never reached the eigenproblem"
        )
    else:
        modes_changed = flat.modes.shape != bump.modes.shape or not np.allclose(flat.modes, bump.modes)
        assert not modes_changed, (
            "DMD modes changed under a different spatial metric, but "
            "dmd.py documents that the regression does not use self.W"
        )


_SQUARE_DIAG_W = np.diag([0.5, 1.0, 2.0, 4.0])
_SQUARE_NONDIAG_W = _SQUARE_DIAG_W.copy()
_SQUARE_NONDIAG_W[0, 1] = _SQUARE_NONDIAG_W[1, 0] = 1e-8


def _square_metric_field(n_space: int = 4, n_snapshots: int = 12) -> dict:
    rng = np.random.default_rng(0)
    return {
        "q": rng.standard_normal((n_snapshots, n_space)),
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_space,
        "Ny": 1,
        "Ns": n_snapshots,
    }


def test_pod_perform_pod_square_diagonal_matches_column_form(tmp_path):
    """perform_pod reads a diagonal square W the same way it reads its column form."""
    field = _square_metric_field()
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weights=np.diag(_SQUARE_DIAG_W),
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    column_eigenvalues = analyzer.eigenvalues.copy()

    analyzer.W = _SQUARE_DIAG_W.copy()
    analyzer.perform_pod()

    np.testing.assert_allclose(analyzer.eigenvalues, column_eigenvalues)


def test_pod_perform_pod_nondiagonal_square_raises(tmp_path):
    """perform_pod rejects a square W with a real off-diagonal, not just a loose one."""
    field = _square_metric_field()
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weights=np.diag(_SQUARE_DIAG_W),
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    analyzer.W = _SQUARE_NONDIAG_W.copy()
    with pytest.raises(ValueError, match=r"np\.diag"):
        analyzer.perform_pod()


def test_pod_orthogonality_check_square_diagonal_matches_column_form(tmp_path):
    """check_spatial_mode_orthogonality reads a diagonal square W the same way it reads its column form."""
    field = _square_metric_field()
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weights=np.diag(_SQUARE_DIAG_W),
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    column_result = analyzer.check_spatial_mode_orthogonality()

    analyzer.W = _SQUARE_DIAG_W.copy()
    square_result = analyzer.check_spatial_mode_orthogonality()

    assert square_result == column_result


def test_pod_orthogonality_check_nondiagonal_square_raises(tmp_path):
    """check_spatial_mode_orthogonality rejects a square W with a real off-diagonal instead of the allclose it used to accept."""
    field = _square_metric_field()
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: field,
        spatial_weights=np.diag(_SQUARE_DIAG_W),
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.W = _SQUARE_NONDIAG_W.copy()
    with pytest.raises(ValueError, match=r"np\.diag"):
        analyzer.check_spatial_mode_orthogonality()


def test_stpod_get_weight_vector_square_diagonal_matches_column_form(tmp_path):
    """_get_weight_vector reads a diagonal square W the same way it reads its column form."""
    analyzer = STPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: {},
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.W = _SQUARE_DIAG_W.copy()
    vector = analyzer._get_weight_vector(4)
    np.testing.assert_allclose(vector, np.diag(_SQUARE_DIAG_W))


def test_stpod_get_weight_vector_nondiagonal_square_raises(tmp_path):
    """_get_weight_vector rejects a square W with a real off-diagonal, matching the shared coercion."""
    analyzer = STPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: {},
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    analyzer.W = _SQUARE_NONDIAG_W.copy()
    with pytest.raises(ValueError, match=r"np\.diag"):
        analyzer._get_weight_vector(4)


# ---------------------------------------------------------------------------
# Polar weights on scattered points (1-D x, y, no grid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_parallel", [False, True])
def test_polar_weights_scattered_points_equal_radius(use_parallel):
    """Scattered polar points: the weight per point is just its radius |y_i|."""
    rng = np.random.default_rng(7)
    n = 7
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(-2.0, 2.0, n)
    w = calculate_polar_weights(x, y, use_parallel=use_parallel, n_space=n)
    assert w.shape == (n, 1)
    np.testing.assert_array_equal(w.ravel(), np.abs(y))


def test_polar_weights_grid_unchanged_by_n_space():
    """Grid input keeps today's tensor-product result, bit-for-bit, once n_space is passed."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    n_space = x.size * y.size
    w_no_n_space = calculate_polar_weights(x, y, use_parallel=False)
    w_with_n_space = calculate_polar_weights(x, y, use_parallel=False, n_space=n_space)
    assert np.array_equal(w_no_n_space, w_with_n_space)

    x2d = np.repeat(x[:, None], len(y), axis=1)
    y2d = np.repeat(y[None, :], len(x), axis=0)
    w_2d = calculate_polar_weights(x2d, y2d, use_parallel=False, n_space=n_space)
    assert np.array_equal(w_no_n_space, w_2d)


def test_polar_weights_scattered_optimized_matches_plain():
    """The optimized route and the plain route agree on scattered points."""
    rng = np.random.default_rng(11)
    n = 7
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(-2.0, 2.0, n)
    w_plain = calculate_polar_weights(x, y, use_parallel=False, n_space=n)
    w_optimized = calculate_polar_weights(x, y, use_parallel=True, n_space=n)
    np.testing.assert_array_equal(w_plain, w_optimized)


def test_polar_weights_scattered_end_to_end_npz(tmp_path):
    """POD with spatial_weight_type='polar' on a scattered .npz builds a length-n W and runs."""
    rng = np.random.default_rng(3)
    ns, n = 6, 9
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.1, 2.0, n)
    q = rng.standard_normal((ns, n))

    path = tmp_path / "scattered_polar.npz"
    np.savez(path, q=q, x=x, y=y, dt=np.float64(0.1))
    pod = PODAnalyzer(
        file_path=str(path),
        spatial_weight_type="polar",
        use_parallel=False,
        n_modes_save=2,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    pod.load_and_preprocess()
    assert np.asarray(pod.W).shape == (n, 1)
    np.testing.assert_array_equal(np.asarray(pod.W).ravel(), np.abs(y))
    pod.perform_pod()
    assert pod.eigenvalues.shape[0] > 0


def test_polar_weights_3d_grid_metric_length(tmp_path):
    """A 3-D (x, r, theta) polar field: the metric length equals q.shape[1].

    The third grid axis is azimuth theta in radians; once it is passed as
    ``z``, the 3-D polar weight covers the whole (x, r, theta) grid and its
    length matches q.shape[1] exactly.
    """
    nx, ny, nz = 3, 4, 2
    n_space = nx * ny * nz
    d = {
        "q": np.random.default_rng(5).standard_normal((5, n_space)),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.linspace(0.1, 2.0, ny),
        "z": np.linspace(0.0, 2.0 * np.pi, nz, endpoint=False),
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "Ns": 5,
    }
    pod = PODAnalyzer(
        data=d,
        spatial_weight_type="polar",
        use_parallel=False,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    pod.load_and_preprocess()
    assert len(np.asarray(pod.W).ravel()) == n_space


def _xr_grid_for_theta_tests():
    """A small (x, r) grid, distinct sizes, shared by the theta-axis tests below."""
    x = np.array([0.0, 0.4, 1.0])
    y = np.array([0.0, 0.3, 0.9, 2.0])
    return x, y


@pytest.mark.parametrize(
    "theta",
    [
        np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False),
        np.array([0.0, 0.3, 1.1, 2.0, 3.9, 5.8]),
    ],
    ids=["uniform", "stretched"],
)
def test_polar_weights_3d_theta_sum_matches_2d(theta):
    """Summing the 3-D weight over theta at each (x, r) reproduces the 2-D weight."""
    x, y = _xr_grid_for_theta_tests()
    Nx, Ny, Ntheta = len(x), len(y), len(theta)
    w2d = calculate_polar_weights(x, y, use_parallel=False).reshape(Nx, Ny)
    w3d = calculate_polar_weights(x, y, z=theta, use_parallel=False).reshape(Ntheta, Ny, Nx)
    summed = w3d.sum(axis=0)  # (Ny, Nx)
    np.testing.assert_allclose(summed, w2d.T, rtol=1e-15, atol=0.0)


def test_polar_weights_3d_uniform_theta_equal_sectors():
    """A uniform theta axis splits each 2-D weight into Ntheta equal sectors."""
    x, y = _xr_grid_for_theta_tests()
    Ntheta = 6
    theta = np.linspace(0.0, 2.0 * np.pi, Ntheta, endpoint=False)
    Nx, Ny = len(x), len(y)
    w2d = calculate_polar_weights(x, y, use_parallel=False).reshape(Nx, Ny)
    w3d = calculate_polar_weights(x, y, z=theta, use_parallel=False).reshape(Ntheta, Ny, Nx)
    expected_sector = w2d.T / Ntheta  # (Ny, Nx)
    for a in range(Ntheta):
        np.testing.assert_allclose(w3d[a], expected_sector, rtol=1e-14)


def test_polar_weights_theta_range_over_2pi_raises():
    """A theta axis spanning more than one revolution is refused, both routes."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    theta_bad = np.linspace(0.0, 2.0 * np.pi * 1.1, 5)
    with pytest.raises(ValueError, match="azimuth"):
        calculate_polar_weights(x, y, z=theta_bad, use_parallel=False)
    with pytest.raises(ValueError, match="azimuth"):
        calculate_polar_weights_optimized(x, y, z=theta_bad)


def test_polar_weights_z_none_matches_no_z_argument():
    """Passing z=None reproduces today's (Nx*Ny, 1) result bit-for-bit, both routes."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    w_default = calculate_polar_weights(x, y, use_parallel=False)
    w_explicit_none = calculate_polar_weights(x, y, z=None, use_parallel=False)
    np.testing.assert_array_equal(w_default, w_explicit_none)

    w_opt_default = calculate_polar_weights_optimized(x, y)
    w_opt_none = calculate_polar_weights_optimized(x, y, z=None)
    np.testing.assert_array_equal(w_opt_default, w_opt_none)
    np.testing.assert_array_equal(w_default, w_opt_default)


@pytest.mark.parametrize(
    "theta",
    [
        np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False),
        np.array([0.0, 0.3, 1.1, 2.0, 3.9, 5.8]),
    ],
    ids=["uniform", "stretched"],
)
def test_polar_weights_3d_plain_and_optimized_agree(theta):
    """calculate_polar_weights and calculate_polar_weights_optimized agree to round-off."""
    x, y = _xr_grid_for_theta_tests()
    w_plain = calculate_polar_weights(x, y, z=theta, use_parallel=False)
    w_optimized = calculate_polar_weights(x, y, z=theta, use_parallel=True)
    np.testing.assert_allclose(w_plain, w_optimized, rtol=1e-14, atol=0.0)


def test_polar_weights_3d_c_order_matches_analytic_integral():
    """The (theta, r, x) flatten order integrates a separable field correctly.

    x, r and theta each carry a distinct, non-constant factor and the grid
    sizes are all different, so a swapped flatten order would not reproduce
    the analytic total (a plain product of three independent 1-D sums).
    """
    x = np.array([0.0, 0.4, 1.0])
    y = np.array([0.0, 0.3, 0.9, 2.0])
    theta = np.array([0.0, 0.5, 1.3, 3.0, 5.5])
    Nx, Ny, Ntheta = len(x), len(y), len(theta)

    def f(v):
        return 2.0 * v + 1.0

    def g(v):
        return v**2 + 0.5

    def h(v):
        return np.sin(v) + 2.0

    theta_fraction = _polar_theta_sector_fractions(theta)
    expected_theta_factor = float(np.dot(theta_fraction, h(theta)))

    w2d = calculate_polar_weights(x, y, use_parallel=False).reshape(Nx, Ny)
    expected_xr_factor = float(np.sum(w2d * np.outer(f(x), g(y))))
    expected_total = expected_xr_factor * expected_theta_factor

    w3d = calculate_polar_weights(x, y, z=theta, use_parallel=False).reshape(Ntheta, Ny, Nx)
    field = h(theta)[:, None, None] * g(y)[None, :, None] * f(x)[None, None, :]
    actual_total = float(np.dot(field.reshape(-1), w3d.reshape(-1)))

    assert actual_total == pytest.approx(expected_total, rel=1e-12)


def test_polar_weights_partial_sector_raises():
    """A theta axis covering only part of the circle is refused, both routes.

    linspace(0, pi/2, 5) is a quarter revolution: the wrap gap back to
    theta[0] + 2*pi is far larger than the regular interior spacing, so
    sector weights (which assume a full revolution) must not be built
    silently.
    """
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    theta_wedge = np.linspace(0.0, np.pi / 2.0, 5)
    with pytest.raises(ValueError, match="part of one revolution"):
        calculate_polar_weights(x, y, z=theta_wedge, use_parallel=False)
    with pytest.raises(ValueError, match="part of one revolution"):
        calculate_polar_weights_optimized(x, y, z=theta_wedge)


def test_polar_weights_endpoint_false_full_circle_passes():
    """A half-open full-circle theta axis (wrap gap == regular spacing) is accepted."""
    x, y = _xr_grid_for_theta_tests()
    Nx, Ny = len(x), len(y)
    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    w3d = calculate_polar_weights(x, y, z=theta, use_parallel=False)
    assert w3d.shape == (len(theta) * Ny * Nx, 1)


def test_polar_weights_duplicated_endpoint_full_circle_passes():
    """A closed theta axis (both 0 and 2*pi present, wrap gap 0) is accepted.

    Its sectors still sum to the 2-D weight: the duplicated endpoint carries
    two half-weight columns that together make up the true sector at that
    angle.
    """
    x, y = _xr_grid_for_theta_tests()
    Nx, Ny = len(x), len(y)
    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=True)
    Ntheta = len(theta)
    w2d = calculate_polar_weights(x, y, use_parallel=False).reshape(Nx, Ny)
    w3d = calculate_polar_weights(x, y, z=theta, use_parallel=False).reshape(Ntheta, Ny, Nx)
    summed = w3d.sum(axis=0)  # (Ny, Nx)
    np.testing.assert_allclose(summed, w2d.T, rtol=1e-15, atol=0.0)


def test_apply_sqrt_metric_matches_elementwise_scaling():
    """apply_sqrt_metric on a 3x4 matrix equals data * sqrt(w) elementwise."""
    data = np.arange(12.0).reshape(3, 4)
    w = np.array([1.0, 4.0, 9.0, 16.0])
    scaled = apply_sqrt_metric(data, w)
    np.testing.assert_allclose(scaled, data * np.sqrt(w))


def test_weighted_total_energy_with_unit_weights_is_plain_frobenius():
    """With w = ones, weighted_total_energy is the plain Frobenius norm^2 / n."""
    data = np.arange(12.0).reshape(3, 4)
    w = np.ones(4)
    energy = weighted_total_energy(data, w)
    expected = float(np.linalg.norm(data, "fro") ** 2 / data.shape[0])
    assert energy == pytest.approx(expected, rel=1e-14)
