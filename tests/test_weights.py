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
    calculate_polar_weights,
    calculate_uniform_weights,
    require_spatial_metric,
)
from openmodalpy.core.decomposition import SpatialMetric, _as_weight_vector


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
