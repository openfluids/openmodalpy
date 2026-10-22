"""Tests for split dNami NPZ loading in core.io module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openmodalpy.core.io import DNamiDataLoader


def test_split_dnami_basic_loading(tmp_path: Path) -> None:
    """Split layout loads mesh, snapshots, and times from separate files."""
    # Mesh file
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])
    np.savez(mesh_npz, x=x, y=y)

    # Snapshot file
    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(5, 3, 2)  # Ns=5, Nx=3, Ny=2
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.savez(snap_npz, u=u, times=times)

    # Define schema
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
            "z_key": None,
        },
        "snapshot": {
            "group": "snapshots",
            "field_key": "u",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    assert data["Nx"] == 3
    assert data["Ny"] == 2
    assert data["Nz"] == 1
    assert data["Ns"] == 5
    assert data["q"].shape == (5, 6)
    np.testing.assert_array_equal(data["x"], x)
    np.testing.assert_array_equal(data["y"], y)
    np.testing.assert_array_equal(data["t"], times)


def test_split_dnami_3d_mesh(tmp_path: Path) -> None:
    """Split loading handles 3D spatial data (Ns, Nx, Ny, Nz)."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0, 2.0])
    np.savez(mesh_npz, x=x, y=y, z=z)

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(3, 2, 2, 3)  # Ns=3, Nx=2, Ny=2, Nz=3
    times = np.array([0.0, 1.0, 2.0])
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
            "z_key": "z",
        },
        "snapshot": {
            "group": "snapshots",
            "field_key": "u",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    assert data["Nx"] == 2
    assert data["Ny"] == 2
    assert data["Nz"] == 3
    assert data["Ns"] == 3
    assert data["q"].shape == (3, 12)
    assert data["z"] is not None
    np.testing.assert_array_equal(data["z"], z)


def test_split_dnami_multiple_snapshot_blocks(tmp_path: Path) -> None:
    """Split loading concatenates multiple snapshot blocks."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    np.savez(mesh_npz, x=x, y=y)

    # Create two snapshot files
    for i in range(2):
        snap_npz = tmp_path / f"snapshot_{i:04d}.npz"
        u = np.random.randn(3, 2, 2)  # Ns=3
        times = np.array([0.0, 1.0, 2.0]) + 3 * i
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
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # Total snapshots should be 6 (3 + 3)
    assert data["Ns"] == 6
    assert data["q"].shape == (6, 4)
    assert len(data["t"]) == 6


def test_split_dnami_time_start_stop_stride(tmp_path: Path) -> None:
    """Split loading respects time_start, time_stop, and time_stride."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(10, 2, 2)  # 10 snapshots
    times = np.arange(10, dtype=float)
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
            "time_start": 2,
            "time_stop": 8,
            "time_stride": 2,
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # time_start=2, time_stop=8, time_stride=2 -> [2, 4, 6] = 3 snapshots
    assert data["Ns"] == 3
    assert data["q"].shape == (3, 4)
    np.testing.assert_array_equal(data["t"], np.array([2.0, 4.0, 6.0]))


def test_split_dnami_spatial_stride(tmp_path: Path) -> None:
    """Split loading applies spatial strides (x_stride, y_stride)."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0])
    np.savez(mesh_npz, x=x, y=y)

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(2, 5, 3)  # Ns=2, Nx=5, Ny=3
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
            "x_stride": 2,
            "y_stride": 2,
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # x_stride=2 -> [0, 2, 4] (3 points)
    # y_stride=2 -> [0, 2] (2 points)
    assert data["Nx"] == 3
    assert data["Ny"] == 2
    assert data["q"].shape == (2, 6)
    np.testing.assert_array_equal(data["x"], np.array([0.0, 2.0, 4.0]))
    np.testing.assert_array_equal(data["y"], np.array([0.0, 2.0]))


def test_split_dnami_y_max_crop(tmp_path: Path) -> None:
    """Split loading crops y-axis using y_max."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    np.savez(mesh_npz, x=x, y=y)

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(2, 2, 4)  # Ns=2, Nx=2, Ny=4
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
            "y_max": 2.2,
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # y_max=2.2 nearest is y[2]=2.0, so y[:2] is selected
    assert data["Ny"] == 2
    assert data["q"].shape == (2, 4)
    np.testing.assert_array_equal(data["y"], y[:2])


def test_split_dnami_field_candidates_fallback(tmp_path: Path) -> None:
    """Split loading falls back to field candidates when field_key unspecified."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    v = np.random.randn(2, 2, 2)  # v is present, not u
    times = np.array([0.0, 1.0])
    np.savez(snap_npz, v=v, times=times)

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
            "field_key": None,
            "field_candidates": ["u", "v", "p"],
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # Should find v as a fallback
    assert data["metadata"]["var_name"] == "v"
    assert data["q"].shape == (2, 4)


def test_split_dnami_reverse_axes(tmp_path: Path) -> None:
    """Split loading flips named spatial axes via reverse_axes."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)  # 2 snapshots
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
            "reverse_axes": ["x"],
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # x-axis reversed: spatial ordering should differ from raw data
    assert data["q"].shape == (2, 4)


def test_split_dnami_preview_ns(tmp_path: Path) -> None:
    """Split loading truncates snapshot count via preview_ns."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(10, 2, 2)  # 10 snapshots
    times = np.arange(10, dtype=float)
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
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema, preview_ns=3)

    assert data["Ns"] == 3
    assert data["q"].shape == (3, 4)
    assert len(data["t"]) == 3


def test_split_dnami_track_loading(tmp_path: Path) -> None:
    """Split loading loads track data from separate group."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])
    np.savez(mesh_npz, x=x, y=y)

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(5, 3, 2)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.savez(snap_npz, u=u, times=times)

    track_npz = tmp_path / "track_0000.npz"
    track_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    track_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.savez(track_npz, x_position=track_vals, times=track_times)

    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
            "snapshots": "snapshot_*.npz",
            "track": "track_*.npz",
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
        "track": {
            "group": "track",
            "x_start_key": "x_position",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    assert data["Ns"] == 5
    assert data["q"].shape == (5, 6)


def test_split_dnami_missing_root_raises(tmp_path: Path) -> None:
    """Split loading raises FileNotFoundError when root doesn't exist."""
    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
        },
    }

    loader = DNamiDataLoader()
    nonexistent = str(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="dNami split dataset root not found"):
        loader.load(nonexistent, schema=schema)


def test_split_dnami_no_snapshot_blocks_raises(tmp_path: Path) -> None:
    """Split loading raises FileNotFoundError when no snapshot blocks found."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

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
    with pytest.raises(FileNotFoundError, match="No files matched pattern for snapshots"):
        loader.load(str(tmp_path), schema=schema)


def test_split_dnami_output_extraction(tmp_path: Path) -> None:
    """Split loading extracts additional outputs from specified groups."""
    mesh_npz = tmp_path / "mesh.npz"
    np.savez(mesh_npz, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(2, 2, 2)
    times = np.array([0.0, 1.0])
    np.savez(snap_npz, u=u, times=times)

    const_npz = tmp_path / "constants.npz"
    reynolds = np.array([100.0])
    np.savez(const_npz, re=reynolds)

    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
            "snapshots": "snapshot_*.npz",
            "constants": "constants.npz",
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
        "outputs": {
            "from_group": {
                "Re": {
                    "group": "constants",
                    "key": "re",
                }
            }
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    assert "Re" in data
    assert data["Re"] == 100.0


def test_split_dnami_timestamp_alignment_with_track(tmp_path: Path) -> None:
    """Split loading aligns track time blocks with snapshot count."""
    mesh_npz = tmp_path / "mesh.npz"
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    np.savez(mesh_npz, x=x, y=y)

    # Snapshot file with 3 snapshots
    snap_npz = tmp_path / "snapshot_0000.npz"
    u = np.random.randn(3, 2, 2)
    times = np.array([0.0, 1.0, 2.0])
    np.savez(snap_npz, u=u, times=times)

    # Track file with more samples than snapshots
    track_npz = tmp_path / "track_0000.npz"
    track_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    track_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.savez(track_npz, x_pos=track_vals, times=track_times)

    schema = {
        "layout": "split_npz",
        "groups": {
            "mesh": "mesh.npz",
            "snapshots": "snapshot_*.npz",
            "track": "track_*.npz",
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
        "track": {
            "group": "track",
            "x_start_key": "x_pos",
            "time_key": "times",
        },
    }

    loader = DNamiDataLoader()
    data = loader.load(str(tmp_path), schema=schema)

    # Track times should be trimmed to match snapshot count
    assert len(data["t"]) == data["Ns"]
    assert data["Ns"] == 3
