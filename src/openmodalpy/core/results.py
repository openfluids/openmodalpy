"""Unified HDF5 result contract: one name table, one writer, one reader.

Every analyzer writes the same dataset names for the same concepts. Older files
that used capitalised names still load through :func:`read_results`, which maps
legacy keys onto the canonical fields and emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from openmodalpy.core.provenance import collect_provenance, safe_attrs_for_hdf5

# Result concepts written by analyzers (lowercase). FFTBlocks is deliberately
# outside this table: it is an FFT *cache* key, not a downstream result field.
CANONICAL_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "modes",
        "eigenvalues",
        "time_coefficients",
        "freq",
        "st",
        "modes1",
        "modes2",
        "triads",
        "amplitudes",
        "omega",
    }
)

# Already uniform across writers; never renamed.
SHARED_KEYS: frozenset[str] = frozenset(
    {
        "x",
        "y",
        "z",
        "W",
        "temporal_mean",
        "energy_map",
        "FFTBlocks",
    }
)

# Pre-unification on-disk names → canonical names.
LEGACY_ALIASES: dict[str, str] = {
    "Modes": "modes",
    "Eigenvalues": "eigenvalues",
    "TimeCoefficients": "time_coefficients",
    "Freq": "freq",
    "St": "st",
    "Modes1": "modes1",
    "Modes2": "modes2",
    "Weights": "W",
    "Triads": "triads",
    # Older SPOD files wrote the grid twice (x/y/z and x_coords/y_coords/z_coords).
    "x_coords": "x",
    "y_coords": "y",
    "z_coords": "z",
}

_KNOWN_FIELD_NAMES: frozenset[str] = CANONICAL_RESULT_KEYS | SHARED_KEYS


@dataclass
class AnalysisResults:
    """Typed view of one result file under the canonical dataset names."""

    path: str
    attrs: dict[str, Any] = field(default_factory=dict)
    modes: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    time_coefficients: np.ndarray | None = None
    freq: np.ndarray | None = None
    st: np.ndarray | None = None
    modes1: np.ndarray | None = None
    modes2: np.ndarray | None = None
    triads: np.ndarray | None = None
    amplitudes: np.ndarray | None = None
    omega: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    W: np.ndarray | None = None
    temporal_mean: np.ndarray | None = None
    energy_map: np.ndarray | None = None
    FFTBlocks: np.ndarray | None = None
    # Datasets that are not part of the named contract (e.g. x_coords).
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def provenance(self) -> dict[str, Any]:
        """``prov_*`` attributes with the prefix stripped.

        Files written before provenance existed report an empty mapping;
        missing keys never raise.
        """
        out: dict[str, Any] = {}
        for key, value in self.attrs.items():
            if isinstance(key, str) and key.startswith("prov_"):
                out[key[len("prov_") :]] = value
        return out


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _read_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Load one HDF5 dataset, including 0-d (scalar) ones.

    Slicing ``dset[:]`` raises on a scalar dataspace; those are read with
    ``dset[()]`` and returned as a 0-d array so callers see a NumPy value.
    """
    if dset.ndim == 0:
        return np.asarray(dset[()])
    return dset[:]


def write_results(
    path: str | Path,
    datasets: Mapping[str, Any],
    *,
    attrs: Mapping[str, Any] | None = None,
    mode: str = "w",
    compression: str | None = "gzip",
) -> None:
    """Write result datasets under their given (canonical) names.

    Parameters
    ----------
    path
        Destination HDF5 path.
    datasets
        Mapping of dataset name → array. ``None`` values are skipped. Names
        should be canonical; this function does not rename.
    attrs
        Optional HDF5 attribute mapping applied with ``update``.
    mode
        ``"w"`` overwrite (default) or ``"a"`` append. Append is only for the
        BSMD path that reuses the FFT-cache file and must keep ``FFTBlocks``.
    compression
        h5py compression filter, or ``None`` to disable.
    """
    path_str = str(path)
    # A results file without decomposition output looks complete (it carries
    # W, coordinates, metadata) but holds an empty mode array. Refuse before
    # opening the file: mode="w" would truncate a previous good result.
    # Carve-out: SPOD/BSMD persist FFTBlocks into the same file before the
    # decomposition runs - such a save is a cache write, not a fake result.
    for mode_name in ("modes", "modes1"):
        value = datasets.get(mode_name)
        if value is not None and np.asarray(value).size == 0 and "FFTBlocks" not in datasets:
            raise ValueError(
                f"refusing to write {path_str}: dataset '{mode_name}' is empty "
                "- the decomposition has not run. Call the perform step before "
                "save_results."
            )
    merged_attrs = dict(attrs or {})
    # Provenance is attached here only so every write path inherits it.
    # Caller keys keep their names; prov_* is reserved for this block.
    merged_attrs.update(collect_provenance(merged_attrs))
    # Hash used the raw attrs; only the on-disk copy must be h5py-safe.
    merged_attrs = safe_attrs_for_hdf5(merged_attrs)
    with h5py.File(path_str, mode) as handle:
        handle.attrs.update(merged_attrs)
        for name, value in datasets.items():
            if value is None:
                continue
            if name in handle:
                del handle[name]
            kwargs: dict[str, Any] = {}
            if compression is not None:
                kwargs["compression"] = compression
            handle.create_dataset(name, data=value, **kwargs)


def read_results(path: str | Path) -> AnalysisResults:
    """Load one result file into :class:`AnalysisResults`.

    Accepts both the current lowercase layout and the pre-unification
    capitalised names. Legacy keys emit a :class:`DeprecationWarning` that
    names the file and the old key.
    """
    path_str = str(Path(path).expanduser())
    fields: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}

    with h5py.File(path_str, "r") as handle:
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
        # Prefer canonical keys when both spellings are present.
        for key in handle.keys():
            if key in LEGACY_ALIASES:
                continue
            fields[key] = _read_dataset(handle[key])
        for key in handle.keys():
            if key not in LEGACY_ALIASES:
                continue
            canon = LEGACY_ALIASES[key]
            warnings.warn(
                f"{path_str}: dataset '{key}' is a legacy name; use '{canon}'",
                DeprecationWarning,
                stacklevel=2,
            )
            if canon not in fields:
                fields[canon] = _read_dataset(handle[key])

    result = AnalysisResults(path=path_str, attrs=attrs)
    for key, value in fields.items():
        if key in _KNOWN_FIELD_NAMES:
            setattr(result, key, value)
        else:
            result.extra[key] = value
    return result


def find_latest_result(results_dir: str | Path, pattern: str) -> str | None:
    """Return the newest path under ``results_dir`` matching ``pattern``, or None.

    Deduplicates the *search* only. Each caller keeps its own not-found
    policy (print-and-return, silent return, …).
    """
    matches = sorted(
        glob.glob(os.path.join(str(results_dir), pattern)),
        key=os.path.getmtime,
        reverse=True,
    )
    return matches[0] if matches else None
