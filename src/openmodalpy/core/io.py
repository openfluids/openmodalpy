#!/usr/bin/env python3
"""
Modular data interfaces for modal decomposition.

This module centralizes data loading so analyzers can consume a uniform data
contract regardless of archive layout.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import h5py
import numpy as np
from fftkit import SamplingReport, describe_sampling, resample_uniform

logger = logging.getLogger(__name__)

DEFAULT_DNAMI_SCHEMA: dict[str, Any] = {
    "layout": "consolidated_npz",
    "files": {
        "pattern": "*.npz",
    },
    "coordinates": {
        "x_key": "x",
        "y_key": "y",
        "z_key": "z",
    },
    "snapshot": {
        "container_key": None,
        "field_key": None,
        "field_candidates": ["u", "v", "p"],
        "time_key": "times",
    },
    "reduction": {
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
        "x_stride": 1,
        "y_stride": 1,
        "z_stride": 1,
    },
    "outputs": {
        "constants": {},
        "from_group": {},
    },
}


SPLIT_SCHEMA_DEFAULTS: dict[str, Any] = {
    "layout": "split_npz",
    "groups": {},
    "coordinates": {
        "group": "mesh",
        "x_key": "x",
        "y_key": "y",
        "z_key": None,
    },
    "snapshot": {
        "group": "snapshots",
        "container_key": None,
        "field_key": None,
        "field_candidates": ["u", "v", "p"],
        "time_key": "times",
        "expected_ndim": None,
        "reverse_axes": [],
    },
    "track": None,
    "crop": {
        "x_start_group": None,
        "x_start_key": None,
        "reduction": "min",
        "x_offset": 0.0,
        "y_max": None,
        "x_stride": 1,
        "y_stride": 1,
        "z_stride": 1,
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
    },
    "outputs": {
        "constants": {},
        "from_group": {},
    },
}


def natural_sort_key(string: str) -> list[int | str]:
    """Return a key for natural sorting of strings with numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", string)]


def _ensure_list(value: Any) -> list[str]:
    """Normalize a single value or sequence into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return a new dict."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_path(root_dir: str, value: str) -> Path:
    """Expand env vars and resolve relative paths against ``root_dir``."""
    candidate = Path(os.path.expandvars(value)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(root_dir) / candidate
    return candidate


def _resolve_paths(root_dir: str, spec: Any, label: str, *, allow_many: bool) -> list[str]:
    """Resolve one or more file paths or glob patterns relative to ``root_dir``."""
    items = _ensure_list(spec)
    if not items:
        raise ValueError(f"No file specification provided for {label}.")

    matches: list[str] = []
    for item in items:
        candidate = _expand_path(root_dir, item)
        candidate_str = str(candidate)
        if any(char in candidate_str for char in "*?[]"):
            found = [str(Path(path)) for path in glob.glob(candidate_str)]
            if not found:
                raise FileNotFoundError(f"No files matched pattern for {label}: {candidate_str}")
            matches.extend(found)
            continue

        if not candidate.is_file():
            raise FileNotFoundError(f"Missing {label} file: {candidate}")
        matches.append(str(candidate))

    resolved = sorted(set(matches), key=natural_sort_key)
    if not allow_many and len(resolved) != 1:
        raise ValueError(f"Expected exactly one file for {label}, found {len(resolved)}: {resolved}")
    return resolved


def _resolve_group_paths(root_dir: str, groups: dict[str, Any], group_name: str, *, allow_many: bool) -> list[str]:
    """Resolve paths for a named schema group."""
    if group_name not in groups:
        raise ValueError(f"Schema group '{group_name}' is not defined.")
    spec = groups[group_name]
    if isinstance(spec, dict) and "files" in spec:
        allow_many = bool(spec.get("allow_many", allow_many))
        spec = spec["files"]
    return _resolve_paths(root_dir, spec, group_name, allow_many=allow_many)


def _normalize_schema(schema: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return a normalized loader schema."""
    if schema is None:
        return deepcopy(DEFAULT_DNAMI_SCHEMA)

    layout = schema.get("layout", DEFAULT_DNAMI_SCHEMA["layout"])
    if layout == "consolidated_npz":
        return _deep_update(DEFAULT_DNAMI_SCHEMA, schema)
    if layout == "split_npz":
        return _deep_update(SPLIT_SCHEMA_DEFAULTS, schema)
    raise ValueError(f"Unsupported dNami loader layout '{layout}'.")


def _extract_npz_value(
    npz_file: np.lib.npyio.NpzFile,
    *,
    file_path: str,
    key: Optional[str] = None,
    candidates: Optional[list[str]] = None,
    container_key: Optional[str] = None,
    label: str,
) -> tuple[str, np.ndarray]:
    """Extract an array from an NPZ file, optionally from an object container."""
    source: Any = npz_file
    available_keys: list[str]

    if container_key:
        if container_key not in npz_file:
            raise KeyError(f"Container key '{container_key}' not found in {file_path} for {label}.")
        source = npz_file[container_key].item()
        if not isinstance(source, dict):
            raise TypeError(f"Container key '{container_key}' in {file_path} did not contain a dict.")
        available_keys = list(source.keys())
    else:
        available_keys = list(npz_file.files)

    if key is not None:
        if key not in source:
            raise KeyError(f"Key '{key}' not found in {file_path} for {label}. Available: {available_keys}")
        return key, np.asarray(source[key])

    for candidate in candidates or []:
        if candidate in source:
            return candidate, np.asarray(source[candidate])

    raise KeyError(f"Could not resolve {label} in {file_path}. Available keys: {available_keys}")


def _normalize_coordinate(coord: np.ndarray, axis_name: str) -> np.ndarray:
    """Reduce mesh-style coordinate arrays to 1D vectors."""
    coord_arr = np.asarray(coord)
    if coord_arr.ndim == 2:
        if axis_name == "x":
            coord_arr = coord_arr[:, 0]
        elif axis_name == "y":
            coord_arr = coord_arr[0, :]
    elif coord_arr.ndim == 3 and axis_name == "z":
        coord_arr = coord_arr[0, 0, :]
    return np.asarray(coord_arr).squeeze()


def _coerce_coordinate(coord: np.ndarray, expected_size: int, label: str, file_path: str) -> np.ndarray:
    """Trim or validate a coordinate vector against an expected axis length."""
    coord_arr = np.asarray(coord).squeeze()
    if coord_arr.ndim != 1:
        coord_arr = coord_arr.reshape(-1)

    if coord_arr.size == expected_size:
        return coord_arr
    if coord_arr.size == expected_size + 1:
        return coord_arr[:-1]
    if coord_arr.size > expected_size:
        return coord_arr[:expected_size]

    raise ValueError(
        f"Coordinate '{label}' from {file_path} has length {coord_arr.size}, but expected at least {expected_size}."
    )


def _infer_dt_from_times(times: np.ndarray) -> float | None:
    """Infer ``dt`` from a time vector, ignoring repeated entries.

    Returns ``None`` when a timestep cannot be inferred (fewer than two
    samples, or an all-constant time vector) so callers do not fabricate
    a unit timestep that silently rescales physical quantities.
    """
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if times_arr.size < 2:
        return None
    diffs = np.diff(times_arr)
    diffs = diffs[np.nonzero(diffs)]
    if diffs.size == 0:
        return None
    return float(np.mean(np.abs(diffs)))


def _reshape_snapshot_block(arr: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Return a flattened snapshot matrix and spatial dimensions."""
    if arr.ndim < 3:
        raise ValueError(f"Expected snapshots with at least 3 dimensions [Ns, ...], got shape {arr.shape}")

    spatial_shape = arr.shape[1:]
    if len(spatial_shape) == 2:
        nx, ny = spatial_shape
        nz = 1
    elif len(spatial_shape) == 3:
        nx, ny, nz = spatial_shape
    else:
        raise ValueError(f"Unsupported snapshot spatial rank {len(spatial_shape)} for shape {arr.shape}")

    return arr.reshape(arr.shape[0], -1), nx, ny, nz


def _apply_snapshot_axis_reversals(arr: np.ndarray, reverse_axes: list[str]) -> np.ndarray:
    """Reverse named spatial axes of a snapshot block."""
    axis_map = {
        "x": 1,
        "y": 2,
        "z": 3,
    }
    transformed = np.asarray(arr)
    for axis_name in reverse_axes:
        if axis_name not in axis_map:
            raise ValueError(f"Unsupported snapshot axis name '{axis_name}' in reverse_axes.")
        axis = axis_map[axis_name]
        if axis >= transformed.ndim:
            raise ValueError(f"Cannot reverse axis '{axis_name}' for snapshot array with shape {transformed.shape}.")
        transformed = np.flip(transformed, axis=axis)
    return transformed


def _resolve_reduction(values: np.ndarray, mode: str) -> float:
    """Reduce an array to a scalar using the configured mode."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot reduce an empty array.")
    if mode == "min":
        return float(np.min(arr))
    if mode == "max":
        return float(np.max(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unsupported crop reduction '{mode}'.")


def _validated_stride(value: Any, axis_name: str) -> int:
    """Return a positive integer stride."""
    stride = int(value)
    if stride <= 0:
        raise ValueError(f"{axis_name}_stride must be >= 1, got {value}.")
    return stride


def _slice_block_in_time(
    arr: Optional[np.ndarray],
    *,
    offset: int,
    block_len: Optional[int] = None,
    time_start: int,
    time_stop: Optional[int],
    time_stride: int,
    times: Optional[np.ndarray] = None,
    track_values: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply a global time slice to one block of snapshots and aligned arrays.

    If *arr* is ``None``, *block_len* must be provided to define the block size;
    the returned arr slot will be ``None``.
    """
    if arr is not None:
        blen = arr.shape[0]
    elif block_len is not None:
        blen = block_len
    else:
        raise ValueError("Either arr or block_len must be provided.")

    def _empty() -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        empty_arr = arr[:0] if arr is not None else None
        empty_times = None if times is None else times[:0]
        empty_track = None if track_values is None else track_values[:0]
        return empty_arr, empty_times, empty_track

    block_start = offset
    block_end = offset + blen
    stop = block_end if time_stop is None else min(int(time_stop), block_end)
    start = max(int(time_start), block_start)
    if start >= stop:
        return _empty()

    anchor = int(time_start)
    remainder = (start - anchor) % time_stride
    first = start if remainder == 0 else start + (time_stride - remainder)
    if first >= stop:
        return _empty()

    local = slice(first - block_start, stop - block_start, time_stride)
    arr_sliced = np.asarray(arr[local]) if arr is not None else None
    times_sliced = None if times is None else np.asarray(times[local])
    track_sliced = None if track_values is None else np.asarray(track_values[local])
    return arr_sliced, times_sliced, track_sliced


def _resolve_mat_time_axis(q_shape: tuple[int, ...], expected_nspace: int) -> int:
    """Infer the time axis for MAT data from coordinate lengths when possible."""
    if expected_nspace > 1 and len(q_shape) >= 2:
        matching_axes = [
            axis for axis, _ in enumerate(q_shape) if int(np.prod(q_shape) / q_shape[axis]) == expected_nspace
        ]
        if len(matching_axes) == 1:
            return matching_axes[0]
    return int(np.argmax(q_shape))


def _resolve_available_fields(available_keys: list[str], snapshot_cfg: dict[str, Any]) -> list[str]:
    """Return only configured flow-field keys from a data container."""
    field_key = snapshot_cfg.get("field_key")
    if field_key:
        return [field_key] if field_key in available_keys else []

    field_candidates = snapshot_cfg.get("field_candidates") or []
    if field_candidates:
        return [candidate for candidate in field_candidates if candidate in available_keys]

    return sorted(available_keys)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self,
        file_path: str,
        *,
        preview_ns: int | None = None,
        field: str | None = None,
        load_single: bool = False,
        schema: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> Dict[str, Any]:
        """Load data from ``file_path`` and return the standardized data contract.

        Loader options after ``file_path`` are keyword-only so subclasses can share
        one override-safe signature without positional meaning shifting between them.
        """

    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """Check if this loader supports the given input."""


class MATDataLoader(DataLoader):
    """Loader for MATLAB .mat files."""

    def supports_format(self, file_path: str) -> bool:
        return file_path.lower().endswith(".mat")

    def load(
        self,
        file_path: str,
        *,
        preview_ns: int | None = None,
        field: str | None = None,
        load_single: bool = False,
        schema: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> Dict[str, Any]:
        """Load data from .mat file with flexible variable detection.

        ``field``, ``load_single``, and ``schema`` are accepted for interface parity
        with other loaders and are ignored — the .mat path has no use for them.
        Extra keyword arguments are discarded.
        """
        del field, load_single, schema, kwargs

        file_size = format_file_size(file_path)
        logger.info("Loading .mat data from %s (%s)", file_path, file_size)

        with h5py.File(file_path, "r") as fread:
            var_name = None
            x_dataset = fread["x"][:] if "x" in fread else None
            y_dataset = fread["y"][:] if "y" in fread else None
            z_dataset = fread["z"][:] if "z" in fread else None

            x_vec = _normalize_coordinate(x_dataset, "x") if x_dataset is not None else None
            y_vec = _normalize_coordinate(y_dataset, "y") if y_dataset is not None else None
            z_vec = _normalize_coordinate(z_dataset, "z") if z_dataset is not None else None
            expected_nspace = 1
            for coord in (x_vec, y_vec, z_vec):
                if coord is not None:
                    expected_nspace *= len(np.asarray(coord).reshape(-1))

            for var in ["p", "u", "v", "data", "field"]:
                if var in fread:
                    var_name = var
                    dataset = fread[var]
                    if preview_ns is not None:
                        shape = dataset.shape
                        if len(shape) == 2 and expected_nspace > 1:
                            if shape[1] == expected_nspace:
                                q = dataset[:preview_ns, :]
                            elif shape[0] == expected_nspace:
                                q = dataset[:, :preview_ns]
                            else:
                                time_axis = _resolve_mat_time_axis(shape, expected_nspace)
                                index = [slice(None)] * len(shape)
                                index[time_axis] = slice(0, preview_ns)
                                q = dataset[tuple(index)]
                        else:
                            time_axis = _resolve_mat_time_axis(shape, expected_nspace)
                            index = [slice(None)] * len(shape)
                            index[time_axis] = slice(0, preview_ns)
                            q = dataset[tuple(index)]
                    else:
                        q = dataset[:]
                    logger.info("Found data variable: '%s'", var)
                    break
            else:
                q = None

            if q is None:
                raise KeyError(f"No recognized data variable in file. Available: {list(fread.keys())}")

            if q.dtype == np.float64:
                q = q.astype(np.float32, copy=False)

            if "dt" in fread:
                dt_data = np.array(fread["dt"])
                dt = dt_data[0][0] if dt_data.ndim > 1 else float(dt_data)
            else:
                # Leave absent so BaseAnalyzer._require_dt() raises rather than
                # silently rescaling every growth rate / band edge.
                dt = None
                logger.warning("No 'dt' found in %r; dt left unset", file_path)

        if q.ndim == 2 and expected_nspace > 1:
            nx = len(x_vec) if x_vec is not None else 1
            ny = len(y_vec) if y_vec is not None else 1
            nz = len(z_vec) if z_vec is not None else 1
            if q.shape[1] == expected_nspace:
                q_reshaped = q
            elif q.shape[0] == expected_nspace:
                q_reshaped = q.T
            else:
                time_axis = _resolve_mat_time_axis(q.shape, expected_nspace)
                if time_axis != 0:
                    q = np.moveaxis(q, time_axis, 0)
                q_reshaped = q.reshape(q.shape[0], expected_nspace)
        else:
            q_shape = q.shape
            time_axis = _resolve_mat_time_axis(q_shape, expected_nspace)
            if time_axis != 0:
                q = np.moveaxis(q, time_axis, 0)

            spatial_shapes = q.shape[1:]
            coords = []
            if x_vec is not None:
                coords.append(x_vec)
            if y_vec is not None:
                coords.append(y_vec)
            if z_vec is not None:
                coords.append(z_vec)
            for i, size in enumerate(spatial_shapes):
                if i < len(coords):
                    if len(coords[i]) != size:
                        coords[i] = np.arange(size)
                else:
                    coords.append(np.arange(size))

            x_vec = coords[0] if coords else np.arange(q.shape[1] if q.ndim > 1 else 1)
            y_vec = coords[1] if len(coords) > 1 else np.arange(q.shape[2] if q.ndim > 2 else 1)
            z_vec = coords[2] if len(coords) > 2 else None

            nx = len(x_vec)
            ny = len(y_vec)
            nz = len(z_vec) if z_vec is not None else 1
            q_reshaped = q.reshape(q.shape[0], nx * ny * nz)

        if x_vec is None:
            x_vec = np.arange(nx)
        if y_vec is None:
            y_vec = np.arange(ny)
        if nz > 1 and z_vec is None:
            z_vec = np.arange(nz)

        ns = q_reshaped.shape[0]

        logger.info(
            "Processed shape: q=%s, Nx=%s, Ny=%s, Nz=%s, Ns=%s",
            q_reshaped.shape,
            nx,
            ny,
            nz,
            ns,
        )
        return {
            "q": q_reshaped,
            "x": x_vec,
            "y": y_vec,
            "z": z_vec,
            "dt": dt,
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "Ns": ns,
            "metadata": {
                "format": "mat",
                "original_shape": q.shape,
                "file_path": file_path,
                "var_name": var_name,
            },
        }


class DNamiDataLoader(DataLoader):
    """General loader for dNami-family NPZ datasets."""

    def supports_format(self, file_path: str) -> bool:
        if file_path.lower().endswith(".npz"):
            return True
        if os.path.isdir(file_path):
            return any(name.lower().endswith(".npz") for name in os.listdir(file_path))
        return False

    def get_available_fields(
        self,
        file_path: str,
        schema: Optional[dict[str, Any]] = None,
        load_single: bool = False,
    ) -> list[str]:
        """Return available flow-field keys for the configured layout."""
        normalized = _normalize_schema(schema)
        if normalized["layout"] == "consolidated_npz":
            files = self._resolve_consolidated_files(file_path, normalized, load_single=load_single)
            snapshot_cfg = normalized["snapshot"]
            with np.load(files[0], allow_pickle=True) as npz:
                if snapshot_cfg.get("container_key"):
                    container = npz[snapshot_cfg["container_key"]].item()
                    return _resolve_available_fields(sorted(container.keys()), snapshot_cfg)
                return _resolve_available_fields(sorted(npz.files), snapshot_cfg)

        snapshot_cfg = normalized["snapshot"]
        group_name = snapshot_cfg["group"]
        files = _resolve_group_paths(file_path, normalized["groups"], group_name, allow_many=True)
        with np.load(files[0], allow_pickle=True) as npz:
            if snapshot_cfg.get("container_key"):
                container = npz[snapshot_cfg["container_key"]].item()
                return _resolve_available_fields(sorted(container.keys()), snapshot_cfg)
            return _resolve_available_fields(sorted(npz.files), snapshot_cfg)

    def load(
        self,
        file_path: str,
        *,
        preview_ns: Optional[int] = None,
        field: Optional[str] = None,
        load_single: bool = False,
        schema: Optional[dict[str, Any]] = None,
        **kwargs: object,
    ) -> Dict[str, Any]:
        """Load data from a consolidated or split NPZ layout."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected dNami loader options: {unexpected}. Pass schema-driven options via 'schema'.")

        normalized = _normalize_schema(schema)
        if normalized["layout"] == "consolidated_npz":
            return self._load_consolidated(
                file_path, normalized, field=field, load_single=load_single, preview_ns=preview_ns
            )
        return self._load_split(file_path, normalized, field=field, preview_ns=preview_ns)

    def _resolve_consolidated_files(self, file_path: str, schema: dict[str, Any], *, load_single: bool) -> list[str]:
        """Resolve the list of consolidated NPZ blocks to load."""
        files_cfg = schema.get("files", {})
        root = Path(file_path).expanduser()

        explicit = files_cfg.get("paths") or files_cfg.get("files")
        if explicit:
            base_dir = str(root if root.is_dir() else root.parent)
            return _resolve_paths(base_dir, explicit, "consolidated data", allow_many=True)

        pattern = files_cfg.get("pattern", "*.npz")
        if root.is_dir():
            return _resolve_paths(str(root), pattern, "consolidated data", allow_many=True)
        if load_single:
            return [str(root)]
        return _resolve_paths(str(root.parent), pattern, "consolidated data", allow_many=True)

    def _load_consolidated(
        self,
        file_path: str,
        schema: dict[str, Any],
        *,
        field: Optional[str],
        load_single: bool,
        preview_ns: Optional[int],
    ) -> Dict[str, Any]:
        """Load consolidated dNami NPZ blocks."""
        files = self._resolve_consolidated_files(file_path, schema, load_single=load_single)
        coords_cfg = schema["coordinates"]
        snapshot_cfg = schema["snapshot"]

        logger.info("Loading dNami consolidated npz files:")
        for idx, path in enumerate(files, 1):
            logger.info("%s. %s (%s)", idx, os.path.basename(path), format_file_size(path))

        q_blocks: list[np.ndarray] = []
        time_blocks: list[np.ndarray] = []
        x = y = z = None
        available_fields = None
        preview_remaining = preview_ns
        reduction_cfg = schema.get("reduction", {})
        time_start = int(reduction_cfg.get("time_start", 0))
        time_stop = reduction_cfg.get("time_stop")
        time_stride = _validated_stride(reduction_cfg.get("time_stride", 1), "time")
        x_stride = _validated_stride(reduction_cfg.get("x_stride", 1), "x")
        y_stride = _validated_stride(reduction_cfg.get("y_stride", 1), "y")
        z_stride = _validated_stride(reduction_cfg.get("z_stride", 1), "z")
        time_offset = 0
        nx = ny = nz = 0
        actual_field: str | None = None

        for path in files:
            with np.load(path, allow_pickle=True) as npz:
                if x is None:
                    x = _normalize_coordinate(npz[coords_cfg["x_key"]], "x")
                    y = _normalize_coordinate(npz[coords_cfg["y_key"]], "y")
                    z_key = coords_cfg.get("z_key")
                    z = _normalize_coordinate(npz[z_key], "z") if z_key and z_key in npz else None

                actual_field, arr = _extract_npz_value(
                    npz,
                    file_path=path,
                    key=field or snapshot_cfg.get("field_key"),
                    candidates=snapshot_cfg.get("field_candidates"),
                    container_key=snapshot_cfg.get("container_key"),
                    label="snapshot field",
                )
                if available_fields is None:
                    available_fields = self.get_available_fields(path, schema=schema, load_single=True)

                time_key = snapshot_cfg.get("time_key")
                if time_key not in npz:
                    raise KeyError(f"Time key '{time_key}' not found in {path}.")
                times_raw = np.asarray(npz[time_key]).reshape(-1)

                arr_block = np.asarray(arr)
                arr_sliced, times_sliced, _ = _slice_block_in_time(
                    arr_block,
                    offset=time_offset,
                    time_start=time_start,
                    time_stop=time_stop,
                    time_stride=time_stride,
                    times=times_raw,
                )
                time_offset += times_raw.shape[0]
                if arr_sliced is None or times_sliced is None or arr_sliced.shape[0] == 0:
                    continue
                arr = arr_sliced
                times = times_sliced

                if preview_remaining is not None:
                    if preview_remaining <= 0:
                        break
                    arr = arr[:preview_remaining]
                    times = times[:preview_remaining]
                    preview_remaining -= arr.shape[0]

                if arr.ndim == 3:
                    arr = np.asarray(arr[:, ::x_stride, ::y_stride])
                elif arr.ndim == 4:
                    arr = np.asarray(arr[:, ::x_stride, ::y_stride, ::z_stride])

                q_flat, nx, ny, nz = _reshape_snapshot_block(arr)
                q_blocks.append(q_flat)
                time_blocks.append(times)

                if preview_remaining == 0:
                    break

        if x is None or y is None:
            raise ValueError(f"No coordinate arrays resolved under {file_path}.")
        if not q_blocks:
            raise ValueError(f"No snapshot blocks resolved under {file_path}.")

        q = np.concatenate(q_blocks, axis=0)
        times = np.concatenate(time_blocks)
        ns = q.shape[0]
        dt = _infer_dt_from_times(times)

        logger.info(
            "Processed shape: q=%s, Nx=%s, Ny=%s, Nz=%s, Ns=%s, dt=%s, field=%s",
            q.shape,
            nx,
            ny,
            nz,
            ns,
            dt,
            actual_field,
        )
        return {
            "q": q,
            "x": _coerce_coordinate(x[::x_stride], nx, "x", files[0]),
            "y": _coerce_coordinate(y[::y_stride], ny, "y", files[0]),
            "z": _coerce_coordinate(z[::z_stride], nz, "z", files[0]) if z is not None else None,
            "t": times,
            "dt": dt,
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "Ns": ns,
            "metadata": {
                "format": "dnami",
                "layout": "consolidated_npz",
                "file_path": file_path,
                "var_name": actual_field,
                "available_fields": available_fields,
                "loaded_files": files,
                "reduction": {
                    "time_start": time_start,
                    "time_stop": time_stop,
                    "time_stride": time_stride,
                    "x_stride": x_stride,
                    "y_stride": y_stride,
                    "z_stride": z_stride,
                },
                "schema": schema,
            },
        }

    def _load_split(
        self,
        root_dir: str,
        schema: dict[str, Any],
        *,
        field: Optional[str],
        preview_ns: Optional[int],
    ) -> Dict[str, Any]:
        """Load split dNami NPZ datasets from a schema-defined directory layout."""
        dataset_root = str(_expand_path(".", root_dir))
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"dNami split dataset root not found: {dataset_root}")

        groups = schema["groups"]
        coords_cfg = schema["coordinates"]
        snapshot_cfg = schema["snapshot"]
        track_cfg = schema.get("track")
        crop_cfg = schema.get("crop", {})
        outputs_cfg = schema.get("outputs", {})

        mesh_group = coords_cfg["group"]
        mesh_path = _resolve_group_paths(dataset_root, groups, mesh_group, allow_many=False)[0]
        snapshot_group = snapshot_cfg["group"]
        snapshot_paths = _resolve_group_paths(dataset_root, groups, snapshot_group, allow_many=True)

        track_paths: list[str] = []
        if track_cfg and track_cfg.get("group"):
            track_paths = _resolve_group_paths(dataset_root, groups, track_cfg["group"], allow_many=True)

        with np.load(mesh_path, allow_pickle=True) as mesh_npz:
            x_mesh = _normalize_coordinate(mesh_npz[coords_cfg["x_key"]], "x")
            y_mesh = _normalize_coordinate(mesh_npz[coords_cfg["y_key"]], "y")
            z_key = coords_cfg.get("z_key")
            z_mesh = _normalize_coordinate(mesh_npz[z_key], "z") if z_key and z_key in mesh_npz else None

        snapshot_blocks = []
        time_blocks = []
        preview_remaining = preview_ns
        actual_field = None
        available_fields = None
        time_start = int(crop_cfg.get("time_start", 0))
        time_stop = crop_cfg.get("time_stop")
        time_stride = _validated_stride(crop_cfg.get("time_stride", 1), "time")
        time_offset = 0

        for path in snapshot_paths:
            with np.load(path, allow_pickle=True) as snapshot_npz:
                actual_field, arr = _extract_npz_value(
                    snapshot_npz,
                    file_path=path,
                    key=field or snapshot_cfg.get("field_key"),
                    candidates=snapshot_cfg.get("field_candidates"),
                    container_key=snapshot_cfg.get("container_key"),
                    label="snapshot field",
                )
                arr = _apply_snapshot_axis_reversals(arr, snapshot_cfg.get("reverse_axes", []))
                original_block_len = np.asarray(arr).shape[0]
                if available_fields is None:
                    available_fields = self.get_available_fields(dataset_root, schema=schema)

                time_key = snapshot_cfg.get("time_key")
                if time_key in snapshot_npz:
                    times = np.asarray(snapshot_npz[time_key]).reshape(-1)
                else:
                    times = None

                arr_sliced, times_sliced, _ = _slice_block_in_time(
                    np.asarray(arr),
                    offset=time_offset,
                    time_start=time_start,
                    time_stop=time_stop,
                    time_stride=time_stride,
                    times=times,
                )
                time_offset += original_block_len
                if arr_sliced is None or arr_sliced.shape[0] == 0:
                    continue
                arr = arr_sliced
                times = times_sliced

                if preview_remaining is not None:
                    if preview_remaining <= 0:
                        break
                    arr = arr[:preview_remaining]
                    if times is not None:
                        times = times[:preview_remaining]
                    preview_remaining -= arr.shape[0]

                snapshot_blocks.append(np.asarray(arr))
                if times is not None:
                    time_blocks.append(times)

                if preview_remaining == 0:
                    break

        if not snapshot_blocks:
            raise ValueError(f"No snapshot blocks resolved under {dataset_root}.")

        snapshots = np.concatenate(snapshot_blocks, axis=0) if len(snapshot_blocks) > 1 else snapshot_blocks[0]

        expected_ndim = snapshot_cfg.get("expected_ndim")
        if expected_ndim is None:
            expected_ndim = 4 if z_mesh is not None else 3
        if snapshots.ndim != expected_ndim:
            raise ValueError(
                f"Split dNami dataset expected {expected_ndim} snapshot dimensions, got shape {snapshots.shape}."
            )

        track_values = None
        track_time_blocks: list[np.ndarray] = []
        if track_paths:
            track_offset = 0
            track_key = crop_cfg.get("x_start_key") or (track_cfg.get("x_start_key") if track_cfg else None)
            for path in track_paths:
                with np.load(path, allow_pickle=True) as track_npz:
                    track_times = None
                    time_key = track_cfg.get("time_key") if track_cfg else None
                    if time_key and time_key in track_npz:
                        track_times = np.asarray(track_npz[time_key]).reshape(-1)
                    if track_key:
                        _, track_arr = _extract_npz_value(
                            track_npz,
                            file_path=path,
                            key=track_key,
                            candidates=None,
                            container_key=track_cfg.get("container_key") if track_cfg else None,
                            label="track field",
                        )
                        track_arr = np.asarray(track_arr).reshape(-1)
                    else:
                        track_arr = None

                    if track_times is not None:
                        track_block_len = len(track_times)
                    elif track_arr is not None:
                        track_block_len = len(track_arr)
                    else:
                        raise ValueError(f"Track file {path} has neither time key nor track field.")
                    _, track_times_sliced, track_arr_sliced = _slice_block_in_time(
                        None,
                        offset=track_offset,
                        block_len=track_block_len,
                        time_start=time_start,
                        time_stop=time_stop,
                        time_stride=time_stride,
                        times=track_times,
                        track_values=track_arr,
                    )
                    track_offset += track_block_len

                    if track_arr_sliced is not None:
                        if track_values is None:
                            track_values = track_arr_sliced
                        else:
                            track_values = np.concatenate([track_values, track_arr_sliced])
                    if track_times_sliced is not None:
                        if (
                            preview_ns is not None
                            and len(track_time_blocks) == 0
                            and snapshots.shape[0] < track_times_sliced.shape[0]
                        ):
                            track_times_sliced = track_times_sliced[: snapshots.shape[0]]
                        track_time_blocks.append(track_times_sliced)

        if time_blocks:
            times = np.concatenate(time_blocks) if len(time_blocks) > 1 else time_blocks[0]
        elif track_time_blocks:
            times = np.concatenate(track_time_blocks) if len(track_time_blocks) > 1 else track_time_blocks[0]
            times = times[: snapshots.shape[0]]
        else:
            raise KeyError(f"No time vector was found for split dNami dataset under {dataset_root}.")

        imin = 0
        if crop_cfg.get("x_start_group") and track_values is not None:
            x_base = _resolve_reduction(track_values, crop_cfg.get("reduction", "min"))
            x_start = x_base + float(crop_cfg.get("x_offset", 0.0))
            imin = int(np.abs(x_mesh - x_start).argmin())

        y_max = crop_cfg.get("y_max")
        jmax = len(y_mesh)
        if y_max is not None:
            jmax = int(np.abs(y_mesh - float(y_max)).argmin())
            if jmax <= 0:
                raise ValueError(
                    f"Configured y_max={y_max} selects no points from mesh {mesh_path}. Check the schema crop block."
                )

        x_stride = _validated_stride(crop_cfg.get("x_stride", 1), "x")
        y_stride = _validated_stride(crop_cfg.get("y_stride", 1), "y")
        z_stride = _validated_stride(crop_cfg.get("z_stride", 1), "z")

        if snapshots.ndim == 3:
            cropped = np.asarray(snapshots[:, imin::x_stride, :jmax:y_stride])
            ns, nx, ny = cropped.shape
            nz = 1
            z = None
        else:
            cropped = np.asarray(snapshots[:, imin::x_stride, :jmax:y_stride, ::z_stride])
            ns, nx, ny, nz = cropped.shape
            z_source = z_mesh[::z_stride] if z_mesh is not None else None
            z = _coerce_coordinate(z_source, nz, "z", mesh_path) if z_source is not None else None

        q = cropped.reshape(ns, -1)
        dt = _infer_dt_from_times(times[:ns])

        data = {
            "q": q,
            "x": _coerce_coordinate(x_mesh[imin::x_stride], nx, "x", mesh_path),
            "y": _coerce_coordinate(y_mesh[:jmax:y_stride], ny, "y", mesh_path),
            "z": z,
            "t": np.asarray(times[:ns]),
            "dt": dt,
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "Ns": ns,
            "metadata": {
                "format": "dnami",
                "layout": "split_npz",
                "dataset_root": dataset_root,
                "var_name": actual_field,
                "field_key": actual_field,
                "mesh_path": mesh_path,
                "snapshot_paths": snapshot_paths,
                "track_paths": track_paths,
                "available_fields": available_fields,
                "crop": {
                    "imin": imin,
                    "jmax": jmax,
                    "x_offset": float(crop_cfg.get("x_offset", 0.0)),
                    "y_max": y_max,
                    "x_stride": x_stride,
                    "y_stride": y_stride,
                    "z_stride": z_stride,
                    "time_start": time_start,
                    "time_stop": time_stop,
                    "time_stride": time_stride,
                },
                "schema": schema,
            },
        }

        constants = outputs_cfg.get("constants", {})
        for name, value in constants.items():
            data[name] = value

        extracted = outputs_cfg.get("from_group", {})
        for name, spec in extracted.items():
            group_name = spec["group"]
            source_path = _resolve_group_paths(dataset_root, groups, group_name, allow_many=False)[0]
            container_key = spec.get("container_key")
            with np.load(source_path, allow_pickle=True) as npz:
                _, value = _extract_npz_value(
                    npz,
                    file_path=source_path,
                    key=spec["key"],
                    candidates=None,
                    container_key=container_key,
                    label=f"output '{name}'",
                )
            data[name] = np.asarray(value).squeeze().item() if np.asarray(value).size == 1 else np.asarray(value)

        logger.info(
            "Loading dNami split dataset from %s: q=%s, Nx=%s, Ny=%s, Nz=%s, Ns=%s, dt=%s, field=%s",
            dataset_root,
            q.shape,
            nx,
            ny,
            nz,
            ns,
            dt,
            actual_field,
        )
        return data


def _read_generic_npz(file_path: str) -> dict[str, np.ndarray]:
    """Read every array of a plain NPZ file into memory."""
    with np.load(file_path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _read_generic_h5(file_path: str) -> dict[str, np.ndarray]:
    """Read every root-level dataset of an HDF5 file into memory."""
    datasets: dict[str, np.ndarray] = {}
    with h5py.File(file_path, "r") as handle:
        for name in handle:
            obj = handle[name]
            if isinstance(obj, h5py.Dataset):
                datasets[name] = np.asarray(obj[()])
    return datasets


def _looks_like_generic_npz(file_path: str) -> bool:
    """Decide whether a single .npz follows the plain contract instead of dNami.

    A plain contract file carries ``q`` and never carries the dNami signature
    (the ``times`` vector or one of the snapshot field candidates ``u``/``v``/
    ``p`` from ``DEFAULT_DNAMI_SCHEMA``). Key inspection is lazy — no array
    payloads are read.
    """
    with np.load(file_path, allow_pickle=False) as npz:
        keys = set(npz.files)
    if "q" not in keys:
        return False
    if "times" in keys:
        return False
    return not any(key in keys for key in DEFAULT_DNAMI_SCHEMA["snapshot"]["field_candidates"])


def derive_grid_and_snapshot_counts(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    stated: Mapping[str, Any],
    *,
    source: str,
) -> tuple[int, int, int, int]:
    """Derive Nx, Ny, Nz and Ns from array shapes.

    This is the one rule for turning ``q``, ``x``, ``y`` and ``z`` into grid
    counts. The generic file reader and the ``data=`` constructor path both
    call it, so a dict built by hand and a dict read from a file follow the
    same logic.

    Parameters
    ----------
    q : np.ndarray
        Snapshot array. Shape is ``(Ns, Nspace)`` or ``(Ns, Ny, Nx[, Nz])``.
    x : np.ndarray
        X-coordinate array.
    y : np.ndarray
        Y-coordinate array.
    z : np.ndarray | None
        Z-coordinate array, or ``None`` for a 2-D field.
    stated : Mapping[str, Any]
        Any ``Nx``/``Ny``/``Nz``/``Ns`` values the caller already gave. A
        value present here is checked against the derived one, not
        overridden by it.
    source : str
        Short text naming where ``q`` came from, used in error messages.

    Returns
    -------
    tuple[int, int, int, int]
        ``(Nx, Ny, Nz, Ns)``.

    Raises
    ------
    ValueError
        If ``q`` has fewer than two axes, if no grid count is consistent
        with ``q``'s spatial width, or if a stated count disagrees with the
        derived one.
    """
    if q.ndim < 2:
        raise ValueError(f"'q' in {source} must be (Ns, Nspace) or (Ns, Ny, Nx[, Nz]); got shape {q.shape}.")

    ns = int(q.shape[0])
    nspace = int(np.prod(q.shape[1:]))

    if q.ndim == 2:
        nx = len(x)
        ny = len(y)
        nz = len(z) if z is not None else 1
        if nx * ny * nz != nspace and all(name in stated for name in ("Nx", "Ny")):
            stated_nx = int(np.asarray(stated["Nx"]).reshape(-1)[0])
            stated_ny = int(np.asarray(stated["Ny"]).reshape(-1)[0])
            stated_nz = int(np.asarray(stated["Nz"]).reshape(-1)[0]) if "Nz" in stated else 1
            if stated_nx * stated_ny * stated_nz == nspace:
                nx, ny, nz = stated_nx, stated_ny, stated_nz
        if (
            nx * ny * nz != nspace
            and x.ndim == 1
            and y.ndim == 1
            and len(x) == len(y) == nspace
            and (z is None or (z.ndim == 1 and len(z) == nspace))
        ):
            # Two equal-length axes are ambiguous between an n x n grid and n
            # scattered points; the product test above already ruled out the
            # grid reading, so treat x/y(/z) as per-point coordinates.
            nx, ny, nz = nspace, 1, 1
    else:
        ny, nx = int(q.shape[1]), int(q.shape[2])
        nz = int(q.shape[3]) if q.ndim == 4 else 1

    if nx * ny * nz != nspace:
        raise ValueError(
            f"Grid counts x={nx}, y={ny}, z={nz} give {nx * ny * nz} points but 'q' in "
            f"{source} holds {nspace} per snapshot. Supply consistent coordinates: "
            f"state Nx/Ny[/Nz], or give 1-D x and y of length Nspace for scattered points."
        )

    for name, derived in (("Nx", nx), ("Ny", ny), ("Nz", nz), ("Ns", ns)):
        if name in stated:
            stated_val = int(np.asarray(stated[name]).reshape(-1)[0])
            if stated_val != derived:
                raise ValueError(f"{name}={stated_val} in {source} disagrees with derived value {derived}.")

    return nx, ny, nz, ns


def _assemble_contract_data(
    datasets: dict[str, np.ndarray],
    *,
    file_path: str,
    preview_ns: int | None,
    resample_time: bool,
) -> Dict[str, Any]:
    """Build the data contract dict from named generic-reader datasets."""
    missing = [key for key in ("q", "x", "y") if key not in datasets]
    if missing:
        raise KeyError(f"Missing required dataset(s) {missing} in {file_path}. Available keys: {sorted(datasets)}")

    q = np.asarray(datasets["q"])
    x = np.asarray(datasets["x"])
    y = np.asarray(datasets["y"])
    z = np.asarray(datasets["z"]) if "z" in datasets else None
    t = np.asarray(datasets["t"]).reshape(-1) if "t" in datasets else None

    nx, ny, nz, ns = derive_grid_and_snapshot_counts(q, x, y, z, datasets, source=file_path)
    nspace = int(np.prod(q.shape[1:]))

    if preview_ns is not None:
        q = q[:preview_ns]
        if t is not None:
            t = t[:preview_ns]
        ns = int(q.shape[0])

    q_flat = np.ascontiguousarray(q).reshape(ns, nspace)
    resampled_time = False
    dt: float | None = None
    if t is not None and t.size >= 2:
        report: SamplingReport = describe_sampling(t)
        if report.is_uniform:
            dt = report.dt_median
        elif resample_time:
            columns = []
            grid = None
            for j in range(nspace):
                result = resample_uniform(t, q_flat[:, j])
                if grid is None:
                    grid = result.t
                columns.append(result.x)
            q_flat = np.stack(columns, axis=1)
            t = np.asarray(grid)
            ns = int(q_flat.shape[0])
            dt = float(np.median(np.diff(t)))
            resampled_time = True
            logger.warning(
                "Non-uniform time base in %s (dt spans [%.6g, %.6g]) resampled onto %d uniform "
                "samples at fs=%.6g Hz by explicit request.",
                file_path,
                report.dt_min,
                report.dt_max,
                ns,
                1.0 / dt,
            )
        else:
            raise ValueError(
                f"Time base in {file_path} is not uniform: dt spans [{report.dt_min:.6g}, "
                f"{report.dt_max:.6g}] (relative jitter {report.jitter:.3e}). Pass "
                f"resample_time=True to resample onto a uniform grid via fftkit.resample_uniform."
            )

    if dt is None and t is None:
        if "dt" in datasets:
            dt = float(np.asarray(datasets["dt"]).reshape(-1)[0])
        else:
            # Leave absent so BaseAnalyzer._require_dt() raises rather than
            # silently rescaling every growth rate / band edge.
            logger.warning("No 't' or 'dt' found in %r; dt left unset", file_path)

    return {
        "q": q_flat,
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "dt": dt,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "Ns": ns,
        "metadata": {
            "format": "generic",
            "original_shape": tuple(int(size) for size in q.shape),
            "file_path": file_path,
            "var_name": "q",
            "available_fields": sorted(datasets),
            "resampled_time": resampled_time,
        },
    }


class GenericDataLoader(DataLoader):
    """Loader for plain Cartesian NPZ/HDF5 files holding contract-named datasets.

    Datasets are addressed by name: ``q``, ``x`` and ``y`` are required;
    ``z``, ``t`` and ``dt`` are optional; integer ``Nx``/``Ny``/``Nz``/``Ns``
    may be supplied and are otherwise derived from the array shapes. ``q`` may
    be (Ns, Nspace) or (Ns, Ny, Nx[, Nz]) and is flattened C-order. Coordinate
    vectors pass through unchanged. A supplied ``t`` must sample uniformly;
    non-uniform records are refused unless ``resample_time=True`` forwards them
    through ``fftkit.resample_uniform``.
    """

    extensions = (".h5", ".hdf5")

    def supports_format(self, file_path: str) -> bool:
        return str(file_path).lower().endswith(self.extensions)

    def load(
        self,
        file_path: str,
        *,
        preview_ns: int | None = None,
        field: str | None = None,
        load_single: bool = False,
        schema: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> Dict[str, Any]:
        """Load a plain NPZ or HDF5 file into the standardized contract."""
        del field, load_single
        if schema is not None:
            raise TypeError("GenericDataLoader takes no schema; it reads named contract datasets.")
        resample_time = bool(kwargs.pop("resample_time", False))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected generic loader options: {unexpected}.")

        suffix = Path(file_path).suffix.lower()
        if suffix == ".npz":
            datasets = _read_generic_npz(str(file_path))
        elif suffix in self.extensions:
            datasets = _read_generic_h5(str(file_path))
        else:
            raise ValueError(f"GenericDataLoader cannot read '{suffix}' files: {file_path}")

        return _assemble_contract_data(
            datasets,
            file_path=str(file_path),
            preview_ns=preview_ns,
            resample_time=resample_time,
        )


class DataInterfaceManager:
    """Select and run the appropriate loader."""

    def __init__(self) -> None:
        self.loaders = [MATDataLoader(), DNamiDataLoader(), GenericDataLoader()]

    def _select_file_loader(self, file_path: str, kwargs: dict[str, object]) -> DataLoader:
        """Pick the loader for a single file, sniffing .npz layout when needed."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".h5", ".hdf5"):
            return GenericDataLoader()
        if ext == ".npz":
            if "schema" in kwargs:
                return DNamiDataLoader()
            return GenericDataLoader() if _looks_like_generic_npz(file_path) else DNamiDataLoader()
        for loader in self.loaders:
            if loader.supports_format(file_path):
                return loader
        raise ValueError(
            f"No loader found for file extension '{ext}'. "
            f"Supported formats: ['.mat', '.npz', '.h5', '.hdf5', directory]"
        )

    def load_data(self, file_path: str, loader_type: Optional[str] = None, **kwargs: object) -> Dict[str, Any]:
        """Load ``file_path`` with the matching loader."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        if loader_type:
            loader_map: dict[str, type[MATDataLoader] | type[DNamiDataLoader] | type[GenericDataLoader]] = {
                "mat": MATDataLoader,
                "dnami": DNamiDataLoader,
                "dnami_npz": DNamiDataLoader,
                "dnamiX_npz": DNamiDataLoader,
                "generic": GenericDataLoader,
            }
            if loader_type not in loader_map:
                raise ValueError(f"Unknown loader type: {loader_type}")
            # cast: typed keyword-only options on load cannot accept **dict[str, object]
            load_fn = cast(Callable[..., Dict[str, Any]], loader_map[loader_type]().load)
            return load_fn(file_path, **kwargs)

        if os.path.isdir(file_path):
            loader = next(loader for loader in self.loaders if loader.supports_format(file_path))
        else:
            loader = self._select_file_loader(file_path, kwargs)
        # cast: typed keyword-only options on load cannot accept **dict[str, object]
        load_fn = cast(Callable[..., Dict[str, Any]], loader.load)
        return load_fn(file_path, **kwargs)

    def get_weight_type(self, data: Dict[str, Any], file_path: str) -> str:
        """Return spatial weight type for ``file_path``."""
        del data, file_path
        return "uniform"

    def list_supported_formats(self) -> Dict[str, str]:
        return {
            ".mat": "MATLAB data files",
            ".npz": "NumPy NPZ files — plain contract layout (GenericDataLoader) or dNami-family layouts (auto-detected)",
            ".h5": "Generic HDF5 files with named q/x/y[/z/t/dt] datasets",
            ".hdf5": "Generic HDF5 files with named q/x/y[/z/t/dt] datasets",
            "directory": "dNami-family split NPZ dataset directories",
        }


def format_file_size(file_path: str) -> str:
    """Format file size in human-readable format."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.1f} MB"


data_manager = DataInterfaceManager()


def load_data(file_path: str, loader_type: Optional[str] = None, **kwargs: object) -> Dict[str, Any]:
    """Convenience entry point for loading data."""
    return data_manager.load_data(file_path, loader_type, **kwargs)


def get_weight_type(data: Dict[str, Any], file_path: str) -> str:
    """Convenience function for determining weight type."""
    return data_manager.get_weight_type(data, file_path)


def load_jetles_data(file_path: str, **kwargs: object) -> Dict[str, Any]:
    """Legacy alias for ``load_mat_data``."""
    return load_mat_data(file_path, **kwargs)


def load_mat_data(file_path: str, **kwargs: object) -> Dict[str, Any]:
    """Legacy compatibility function for .mat data."""
    return load_data(file_path, loader_type="mat", **kwargs)


DNamiXNPZLoader = DNamiDataLoader
