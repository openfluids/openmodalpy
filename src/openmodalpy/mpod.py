#!/usr/bin/env python3
"""Multiscale Proper Orthogonal Decomposition (mPOD)."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from typing import Any, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import CANONICAL_TIE_RTOL
from openmodalpy.core.config import FIGURES_DIR_POD, RESULTS_DIR_POD
from openmodalpy.pod import PODAnalyzer
from openmodalpy.specs import display_name_for

logger = logging.getLogger(__name__)


def _resolve_band_edges(band_edges: Iterable[float] | None, nyquist: float) -> np.ndarray:
    """Validate and resolve mPOD band edges in Hz."""
    if band_edges is None:
        resolved = np.array([0.0, nyquist], dtype=float)
    else:
        resolved = np.asarray(list(band_edges), dtype=float)
    if resolved.ndim != 1 or resolved.size < 2:
        raise ValueError("band_edges must define at least one interval via two or more edges.")
    if np.any(np.diff(resolved) <= 0):
        raise ValueError("band_edges must be strictly increasing.")
    if resolved[0] < 0.0:
        raise ValueError("band_edges must be non-negative.")
    if resolved[-1] > nyquist + 1e-12:
        raise ValueError(f"band_edges upper bound {resolved[-1]} exceeds Nyquist frequency {nyquist}.")
    return resolved


def _pooled_mode_order(eigenvalues: np.ndarray, band_ids: np.ndarray) -> np.ndarray:
    """Order pooled mPOD modes by energy, breaking ties by band then position.

    Eigenvalues that agree to within ``CANONICAL_TIE_RTOL`` relative are one
    group. A few ulps of platform noise is a real difference to ``argsort``,
    so a stable sort alone cannot keep the column order reproducible.
    Inside a tied group: band index ascending, then position within that band.
    """
    n = int(eigenvalues.size)
    if n == 0:
        return np.array([], dtype=int)

    # Stable: with several exactly-equal peaks the group boundaries below must
    # not depend on the platform's sort either.
    energy_order = np.argsort(-np.asarray(eigenvalues, dtype=float), kind="stable")
    ordered = np.empty(n, dtype=int)
    filled = 0
    i = 0
    while i < n:
        peak = float(eigenvalues[int(energy_order[i])])
        floor = peak - CANONICAL_TIE_RTOL * abs(peak)
        j = i + 1
        while j < n and float(eigenvalues[int(energy_order[j])]) >= floor:
            j += 1
        group = energy_order[i:j]
        # Bands are concatenated whole, so the original index is the within-band position.
        tie_key = np.lexsort((group, band_ids[group]))
        size = j - i
        ordered[filled : filled + size] = group[tie_key]
        filled += size
        i = j
    return ordered


class MPODAnalyzer(PODAnalyzer):
    """Multiscale POD using non-overlapping temporal frequency bands."""

    _METHOD_NAME = "mpod"

    def __init__(
        self,
        file_path: str | None = None,
        results_dir: str = RESULTS_DIR_POD,
        figures_dir: str = FIGURES_DIR_POD,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        n_modes_save: int = 10,
        band_edges: Iterable[float] | None = None,
        band_scale: str = "hz",
        filter_kind: str = "rectangular",
        use_parallel: bool = True,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the MPODAnalyzer.

        Args:
            file_path (str | None): Path to the data file. Optional when
                ``data`` carries the loaded dataset instead.
            data (dict | None): Already-loaded dataset following the data
                contract (see DOC.md). Given instead of ``file_path``.
        """
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            n_modes_save=n_modes_save,
            use_parallel=use_parallel,
            spatial_weights=spatial_weights,
            data=data,
        )
        self.band_edges = list(band_edges) if band_edges is not None else None
        self.band_scale = band_scale
        self.filter_kind = filter_kind
        self.mode_band_indices = np.array([], dtype=int)
        self.band_mode_counts = np.array([], dtype=int)
        self.analysis_type = "mpod"
        self._resolved_band_edges_hz = np.array([])

    def _get_algorithm_metadata(self) -> dict:
        lift = getattr(self, "_lift", None) or decomposition.BandFilteredLift()
        return {
            "lift_kind": lift.kind,
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_second_order_operator": True,
            "eigenvalue_normalization": "snapshot_average",
            "filter_kind": self.filter_kind,
            "band_scale": self.band_scale,
            "band_edges_hz": np.asarray(self._resolved_band_edges_hz, dtype=float),
            "mode_band_indices": np.asarray(self.mode_band_indices, dtype=int),
            "band_mode_counts": np.asarray(self.band_mode_counts, dtype=int),
        }

    def load_results(self, filename: str | None = None) -> None:
        super().load_results(filename=filename)
        if not filename:
            filename = f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"
        load_path = os.path.join(self.results_dir, filename)
        if not os.path.isfile(load_path):
            from openmodalpy.core.results import find_latest_result

            latest = find_latest_result(self.results_dir, f"*_{self.analysis_type}.hdf5")
            if not latest:
                return
            load_path = latest
        with h5py.File(load_path, "r") as handle:  # type: ignore[name-defined]
            if "band_edges_hz" in handle.attrs:
                self._resolved_band_edges_hz = np.asarray(handle.attrs["band_edges_hz"], dtype=float)
            if "mode_band_indices" in handle.attrs:
                self.mode_band_indices = np.asarray(handle.attrs["mode_band_indices"], dtype=int)
            if "band_mode_counts" in handle.attrs:
                self.band_mode_counts = np.asarray(handle.attrs["band_mode_counts"], dtype=int)
            if "filter_kind" in handle.attrs:
                self.filter_kind = str(handle.attrs["filter_kind"])
            if "band_scale" in handle.attrs:
                self.band_scale = str(handle.attrs["band_scale"])

    def perform_mpod(self) -> None:
        """Perform mPOD by POD-decomposing non-overlapping band-limited data."""
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        if self.filter_kind != "rectangular":
            raise ValueError(f"Unsupported filter_kind '{self.filter_kind}'.")

        data_matrix = np.asarray(self.data["q"], dtype=float)
        n_snapshots, n_space = data_matrix.shape
        if n_snapshots < 2:
            raise ValueError("Need at least 2 snapshots for mPOD.")

        self.temporal_mean = cast(np.ndarray, np.mean(data_matrix, axis=0, dtype=np.float64))
        data_centered = data_matrix - self.temporal_mean

        weight_vector = decomposition._as_weight_vector(np.asarray(self.W), n_space)
        dt = self._require_dt()
        nyquist = 0.5 / dt
        candidate_edges = self.band_edges
        if self.band_edges is not None and self.band_scale == "normalized_nyquist":
            candidate_edges = (np.asarray(self.band_edges, dtype=float) * nyquist).tolist()
        elif self.band_edges is not None and self.band_scale != "hz":
            raise ValueError(f"Unsupported band_scale '{self.band_scale}'.")
        self._resolved_band_edges_hz = _resolve_band_edges(candidate_edges, nyquist)

        # A single full band should reduce exactly to POD.
        if (
            self._resolved_band_edges_hz.size == 2
            and np.isclose(self._resolved_band_edges_hz[0], 0.0)
            and np.isclose(self._resolved_band_edges_hz[-1], nyquist)
        ):
            super().perform_pod()
            self.mode_band_indices = np.zeros(self.eigenvalues.size, dtype=int)
            self.band_mode_counts = np.array([self.eigenvalues.size], dtype=int)
            return

        name = display_name_for(self.analysis_type)
        logger.info("Performing %s analysis...", name)
        start_time = time.time()

        metric = decomposition.SpatialMetric(weight_vector)
        # Kind is shared across bands; each band builds its own BandFilteredLift.
        # BandFilteredLift satisfies the Lift protocol structurally; no cast needed.
        self._lift = decomposition.BandFilteredLift()

        band_modes: list[np.ndarray] = []
        band_eigenvalues: list[np.ndarray] = []
        band_coefficients: list[np.ndarray] = []
        mode_band_indices: list[np.ndarray] = []
        band_mode_counts: list[int] = []

        for band_index, (f_low, f_high) in enumerate(
            zip(self._resolved_band_edges_hz[:-1], self._resolved_band_edges_hz[1:])
        ):
            is_last = band_index == self._resolved_band_edges_hz.size - 2
            lift = decomposition.BandFilteredLift(
                f_low=float(f_low),
                f_high=float(f_high),
                dt=dt,
                is_last=is_last,
            )
            # Ask the lift for its own mask; the edge convention lives in one place.
            if not np.any(lift.mask(n_snapshots)):
                band_mode_counts.append(0)
                continue

            data_band = lift.apply(data_centered)
            # mPOD drops non-positive eigenvalues and truncates inside the solver.
            modes, eigenvalues, coeffs = decomposition.weighted_second_order(
                data_band,
                metric,
                method="eigh",
                n_keep=self.n_modes_save,
            )
            if eigenvalues.size == 0:
                band_mode_counts.append(0)
                continue

            band_modes.append(modes)
            band_eigenvalues.append(eigenvalues)
            band_coefficients.append(coeffs)
            mode_band_indices.append(np.full(eigenvalues.size, band_index, dtype=int))
            band_mode_counts.append(eigenvalues.size)

        if not band_eigenvalues:
            raise ValueError("mPOD band decomposition produced no energetic modes.")

        eigenvalues = np.concatenate(band_eigenvalues)
        modes = np.concatenate(band_modes, axis=1)
        coeffs = np.concatenate(band_coefficients, axis=1)
        band_ids = np.concatenate(mode_band_indices)

        order = _pooled_mode_order(eigenvalues, band_ids)
        keep = min(self.n_modes_save, eigenvalues.size)

        self.eigenvalues = np.real(eigenvalues[order][:keep])
        self.modes = np.real(modes[:, order][:, :keep])
        self.time_coefficients = np.real(coeffs[:, order][:, :keep])
        self.mode_band_indices = band_ids[order][:keep]
        self.band_mode_counts = np.asarray(band_mode_counts, dtype=int)
        # keep may be < n_modes_save when bands yield fewer modes than the cap.
        self._resync_mode_count()

        logger.info("%s analysis completed in %.2f seconds.", name, time.time() - start_time)
        logger.info("Computed %d %s modes.", self.modes.shape[1], name)
        logger.info("%s per-band mode counts: %s", name, [int(n) for n in self.band_mode_counts])

    _perform_name = "perform_mpod"

    # Reuse POD save/load/plot methods.
