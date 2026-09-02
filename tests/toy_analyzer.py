#!/usr/bin/env python3
"""Toy analyzer to measure the cost of adding an eighth method.

Deliberately trivial decomposition: modes are the first k snapshots,
eigenvalues are their norms. This keeps every line of code a measurement,
not method content.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from openmodalpy.core.base import BaseAnalyzer
from openmodalpy.core.config import FIGURES_DIR_POD, RESULTS_DIR_POD
from openmodalpy.specs import display_name_for

logger = logging.getLogger(__name__)


class ToyAnalyzer(BaseAnalyzer):
    """Toy analyzer for measurement: modes are first k snapshots, eigenvalues are norms."""

    _METHOD_NAME = "toy"

    def __init__(
        self,
        file_path: str | None = None,
        *,
        results_dir: str = RESULTS_DIR_POD,
        figures_dir: str = FIGURES_DIR_POD,
        data_loader: Any | None = None,
        spatial_weight_type: str | None = None,
        n_modes_save: int = 10,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ToyAnalyzer."""
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            spatial_weights=spatial_weights,
            data=data,
        )
        self.n_modes_save = n_modes_save
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.analysis_type = "toy"

    def _get_algorithm_metadata(self) -> dict[str, Any]:
        """Return algorithm metadata for provenance."""
        return {
            "method": "toy",
            "description": "Trivial modes (first k snapshots), norms as eigenvalues",
        }

    def perform_toy(self) -> None:
        """Perform trivial decomposition: modes are first k snapshots."""
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        logger.info("Performing %s analysis...", display_name_for(self.analysis_type))

        data_matrix = np.asarray(self.data["q"], dtype=float)
        n_snapshots, n_space = data_matrix.shape

        if n_snapshots < 2:
            raise ValueError(f"Need at least 2 snapshots, got {n_snapshots}")
        if n_space < 1:
            raise ValueError(f"Need at least 1 spatial point, got {n_space}")

        # Modes are the first k snapshots (normalized).
        k = min(self.n_modes_save, n_snapshots)
        modes_raw = data_matrix[:k, :].T  # (n_space, k)

        # Eigenvalues are L2 norms of each mode.
        eigenvalues = np.linalg.norm(modes_raw, axis=0)

        # Normalize modes to unit norm.
        modes_normalized = modes_raw / np.maximum(eigenvalues[np.newaxis, :], 1e-14)

        self.modes = np.real(modes_normalized[:, :k])
        self.eigenvalues = np.real(eigenvalues[:k])
        # Time coefficients: data projected onto normalized modes.
        self.time_coefficients = np.real(data_matrix @ self.modes)
        self._resync_mode_count()

        logger.info("Computed %d %s modes.", self.modes.shape[1], display_name_for(self.analysis_type))

    _perform_name = "perform_toy"

    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return toy analyzer datasets and metadata to save."""
        datasets: dict[str, Any] = {
            "modes": self.modes,
            "eigenvalues": self.eigenvalues,
            "time_coefficients": self.time_coefficients,
            "x": self.data["x"],
            "y": self.data["y"],
            "W": self.W,
        }
        return datasets, self._get_metadata()

    def _required_result_fields(self) -> tuple[str, ...]:
        """Name the datasets that make a file a toy analyzer result."""
        return ("modes", "eigenvalues", "time_coefficients")

    def _result_filename(self) -> str:
        """The toy method has no Welch blocks, so its name holds no block size."""
        return f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"

    def _plot_run(self, run_id: str | None = None) -> None:
        """Minimal plot for the toy analyzer."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(self.eigenvalues, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Mode Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"{display_name_for(self.analysis_type)} Eigenvalues")
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_eigenvalues.png")
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved plot to %s", plot_path)
