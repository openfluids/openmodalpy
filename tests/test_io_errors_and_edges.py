"""Tests for error handling and edge cases in core.io module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openmodalpy.core.io import (
    DNamiDataLoader,
    GenericDataLoader,
    _apply_snapshot_axis_reversals,
    _coerce_coordinate,
    _normalize_coordinate,
    _resolve_reduction,
    _slice_block_in_time,
    _validated_stride,
)


def test_normalize_coordinate_1d() -> None:
    """_normalize_coordinate returns 1D array unchanged."""
    x_mesh = np.arange(5.0)
    result = _normalize_coordinate(x_mesh, "x")
    np.testing.assert_array_equal(result, x_mesh)


def test_normalize_coordinate_2d_x() -> None:
    """_normalize_coordinate extracts first column for x from 2D array."""
    x_mesh = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])  # (5, 1)
    result = _normalize_coordinate(x_mesh, "x")
    np.testing.assert_array_equal(result, np.arange(5.0))


def test_normalize_coordinate_2d_y() -> None:
    """_normalize_coordinate extracts first row for y from 2D array."""
    y_mesh = np.array([[0.0, 1.0, 2.0]])  # (1, 3)
    result = _normalize_coordinate(y_mesh, "y")
    np.testing.assert_array_equal(result, np.arange(3.0))


def test_normalize_coordinate_3d_z() -> None:
    """_normalize_coordinate extracts first z-layer from 3D array."""
    z_mesh_reshaped = np.arange(4.0).reshape(1, 1, 4)
    result = _normalize_coordinate(z_mesh_reshaped, "z")
    np.testing.assert_array_equal(result, np.arange(4.0))


def test_coerce_coordinate_exact_match() -> None:
    """_coerce_coordinate accepts exact-size coordinate."""
    coord = np.array([0.0, 1.0, 2.0])
    result = _coerce_coordinate(coord, 3, "x", "test.npz")
    np.testing.assert_array_equal(result, coord)


def test_coerce_coordinate_one_extra() -> None:
    """_coerce_coordinate trims one extra element."""
    coord = np.array([0.0, 1.0, 2.0, 3.0])
    result = _coerce_coordinate(coord, 3, "x", "test.npz")
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))


def test_coerce_coordinate_extra() -> None:
    """_coerce_coordinate trims excess elements."""
    coord = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = _coerce_coordinate(coord, 3, "x", "test.npz")
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))


def test_coerce_coordinate_too_small_raises() -> None:
    """_coerce_coordinate raises when coordinate is too small."""
    coord = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="length 2.*expected at least 3"):
        _coerce_coordinate(coord, 3, "x", "test.npz")


def test_coerce_coordinate_2d_reshaped() -> None:
    """_coerce_coordinate reshapes 2D arrays to 1D."""
    coord = np.array([[0.0, 1.0, 2.0]])
    result = _coerce_coordinate(coord, 3, "x", "test.npz")
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))


def test_validated_stride_positive() -> None:
    """_validated_stride accepts positive integers."""
    assert _validated_stride(1, "x") == 1
    assert _validated_stride(5, "y") == 5
    assert _validated_stride(100, "z") == 100


def test_validated_stride_zero_raises() -> None:
    """_validated_stride rejects zero stride."""
    with pytest.raises(ValueError, match="x_stride must be >= 1"):
        _validated_stride(0, "x")


def test_validated_stride_negative_raises() -> None:
    """_validated_stride rejects negative stride."""
    with pytest.raises(ValueError, match="y_stride must be >= 1"):
        _validated_stride(-1, "y")


def test_resolve_reduction_min() -> None:
    """_resolve_reduction computes min of array."""
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = _resolve_reduction(arr, "min")
    assert result == 1.0


def test_resolve_reduction_max() -> None:
    """_resolve_reduction computes max of array."""
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = _resolve_reduction(arr, "max")
    assert result == 5.0


def test_resolve_reduction_mean() -> None:
    """_resolve_reduction computes mean of array."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = _resolve_reduction(arr, "mean")
    assert result == 2.5


def test_resolve_reduction_unsupported_raises() -> None:
    """_resolve_reduction rejects unknown mode."""
    arr = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported crop reduction"):
        _resolve_reduction(arr, "median")


def test_resolve_reduction_empty_raises() -> None:
    """_resolve_reduction raises on empty array."""
    arr = np.array([])
    with pytest.raises(ValueError, match="Cannot reduce an empty array"):
        _resolve_reduction(arr, "min")


def test_apply_snapshot_axis_reversals_x() -> None:
    """_apply_snapshot_axis_reversals flips x-axis."""
    arr = np.arange(2 * 2 * 2).reshape(2, 2, 2)  # Ns=2, Nx=2, Ny=2
    original_first = arr[0, 0, 0]
    result = _apply_snapshot_axis_reversals(arr, ["x"])
    # After flip along axis 1, original [0, 0] becomes [0, 1]
    assert result[0, 1, 0] == original_first


def test_apply_snapshot_axis_reversals_y() -> None:
    """_apply_snapshot_axis_reversals flips y-axis."""
    arr = np.arange(2 * 2 * 2).reshape(2, 2, 2)  # Ns=2, Nx=2, Ny=2
    original_first = arr[0, 0, 0]
    result = _apply_snapshot_axis_reversals(arr, ["y"])
    # After flip along axis 2, original [0, 0] becomes [0, 1]
    assert result[0, 0, 1] == original_first


def test_apply_snapshot_axis_reversals_z() -> None:
    """_apply_snapshot_axis_reversals flips z-axis."""
    arr = np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2)  # Ns=2, Nx=2, Ny=2, Nz=2
    original_first = arr[0, 0, 0, 0]
    result = _apply_snapshot_axis_reversals(arr, ["z"])
    # After flip along axis 3, original [..., 0] becomes [..., 1]
    assert result[0, 0, 0, 1] == original_first


def test_apply_snapshot_axis_reversals_invalid_raises() -> None:
    """_apply_snapshot_axis_reversals rejects invalid axis name."""
    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match="Unsupported snapshot axis name"):
        _apply_snapshot_axis_reversals(arr, ["invalid"])


def test_apply_snapshot_axis_reversals_too_many_dims_raises() -> None:
    """_apply_snapshot_axis_reversals raises when axis exceeds ndim."""
    arr = np.arange(8).reshape(2, 2, 2)  # 3D array
    with pytest.raises(ValueError, match="Cannot reverse axis 'z'"):
        _apply_snapshot_axis_reversals(arr, ["z"])


def test_consolidated_loader_applies_field_key_schema(tmp_path: Path) -> None:
    """Consolidated loading respects explicit field_key in schema."""
    npz_file = tmp_path / "data.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    u = np.random.randn(3, 2, 2)
    v = np.random.randn(3, 2, 2)
    times = np.array([0.0, 1.0, 2.0])
    np.savez(npz_file, x=x, y=y, u=u, v=v, times=times)

    schema = {
        "layout": "consolidated_npz",
        "reduction": {},
        "snapshot": {
            "field_key": "v",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(npz_file), load_single=True, schema=schema)

    assert data["metadata"]["var_name"] == "v"


def test_consolidated_loader_missing_time_key_raises(tmp_path: Path) -> None:
    """Consolidated loading raises when time_key not found."""
    npz_file = tmp_path / "data.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    u = np.random.randn(3, 2, 2)
    np.savez(npz_file, x=x, y=y, u=u)  # No times

    schema = {
        "layout": "consolidated_npz",
        "snapshot": {
            "field_key": "u",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    with pytest.raises(KeyError, match="Time key 'times' not found"):
        loader.load(str(npz_file), load_single=True, schema=schema)


def test_split_loader_missing_time_vector_raises(tmp_path: Path) -> None:
    """Split loading raises when both snapshot and track times are absent."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(2, 2, 2)
    # No times key
    np.savez(snap_npz, u=u)

    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
            "snapshots": "snapshot_*.npz",
        },
        "coordinates": {
            "group": "mesh",
            "x_key": "x",
            "y_key": "y",
        },
        "snapshot": {
            "group": "snapshots",
            "field_key": "u",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    with pytest.raises(KeyError, match="No time vector was found"):
        loader.load(str(tmp_path), schema=schema)


def test_split_loader_y_max_zero_raises(tmp_path: Path) -> None:
    """Split loading raises when y_max crops to zero points."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    np.savez(mesh_npz, x=x, y=y)

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(2, 2, 3)
    times = np.array([0.0, 1.0])
    np.savez(snap_npz, u=u, times=times)

    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
            "snapshots": "snapshot_*.npz",
        },
        "coordinates": {
            "group": "mesh",
            "x_key": "x",
            "y_key": "y",
        },
        "snapshot": {
            "group": "snapshots",
            "field_key": "u",
            "time_key": "times",
        },
        "crop": {
            "y_max": -5.0,  # Far from all y values
        },
    }

    loader = DNamiDataLoader()
    with pytest.raises(ValueError, match="selects no points from mesh"):
        loader.load(str(tmp_path), schema=schema)


def test_generic_loader_missing_q_raises(tmp_path: Path) -> None:
    """Generic loader raises when q dataset missing."""
    npz_file = tmp_path / "data.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    np.savez(npz_file, x=x, y=y)  # No q

    loader = GenericDataLoader()
    with pytest.raises(KeyError, match="Missing required dataset"):
        loader.load(str(npz_file))


def test_generic_loader_missing_x_raises(tmp_path: Path) -> None:
    """Generic loader raises when x dataset missing."""
    npz_file = tmp_path / "data.npz"
    y = np.array([0.0, 1.0])
    q = np.random.randn(5, 2)
    np.savez(npz_file, q=q, y=y)  # No x

    loader = GenericDataLoader()
    with pytest.raises(KeyError, match="Missing required dataset"):
        loader.load(str(npz_file))


def test_generic_loader_missing_y_raises(tmp_path: Path) -> None:
    """Generic loader raises when y dataset missing."""
    npz_file = tmp_path / "data.npz"
    x = np.array([0.0, 1.0])
    q = np.random.randn(5, 2)
    np.savez(npz_file, q=q, x=x)  # No y

    loader = GenericDataLoader()
    with pytest.raises(KeyError, match="Missing required dataset"):
        loader.load(str(npz_file))


def test_time_stride_stays_aligned_across_block_boundaries() -> None:
    """The time stride counts from time_start, not from the start of each block.

    A split dataset arrives as separate blocks that are joined end to end. The
    stride must step through the joined series. If each block restarted the
    count, the user would get a time series with an irregular step and no
    error. Two blocks of five, time_start=1 and time_stride=3, must keep the
    global samples 1, 4 and 7.
    """
    kept: list[int] = []
    for offset in (0, 5):
        block = np.arange(offset, offset + 5)
        sliced, _, _ = _slice_block_in_time(
            block,
            offset=offset,
            time_start=1,
            time_stop=None,
            time_stride=3,
        )
        assert sliced is not None
        kept.extend(int(value) for value in sliced)

    assert kept == [1, 4, 7]
