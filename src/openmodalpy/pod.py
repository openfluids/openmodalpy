#!/usr/bin/env python3
"""
Proper orthogonal decomposition (POD)

Following the implementation of https://github.com/MathEXLab/PySPOD/blob/main/pyspod/pod/standard.py

we want a pure python version using the same style and language as spod.py

Author: R. Frantz

Reference codes:
    - https://github.com/MathEXLab/PySPOD/blob/main/pyspod/pod/standard.py
"""

from __future__ import annotations

# Standard library imports
import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal, Optional, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from fftkit import find_peaks, periodogram_rfft
from numpy.typing import ArrayLike

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import (
    BaseAnalyzer,
    add_inset_colorbar,
    format_mode_title,
    get_fig_aspect_ratio,
    plot_modes_3d,
    print_summary,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.config import (
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_POD,
    RESULTS_DIR_POD,
)
from openmodalpy.specs import display_name_for

logger = logging.getLogger(__name__)


class PODAnalyzer(BaseAnalyzer):
    """Proper Orthogonal Decomposition (POD) analyzer.

    This class implements POD, a technique to decompose a data ensemble into
    a set of optimal orthogonal modes (spatial structures) and corresponding
    time coefficients. The modes are ranked by their energy content, captured
    by the eigenvalues.

    The POD method typically involves:
    1. Forming a data matrix from snapshots of the flow field (or other data).
    2. Subtracting the temporal mean from the data.
    3. Performing a Singular Value Decomposition (SVD) of the (weighted) mean-subtracted data matrix.
       Alternatively, for snapshot POD, an eigenvalue decomposition of the covariance matrix.

    Key Attributes:
        modes (np.ndarray): Spatial POD modes (Phi). Shape: (n_spatial_points, n_modes_save).
        eigenvalues (np.ndarray): POD eigenvalues (lambda), representing energy of modes.
                                  Shape: (n_modes_save,).
        time_coefficients (np.ndarray): Temporal coefficients (A) corresponding to POD modes.
                                        Shape: (n_snapshots, n_modes_save).
        temporal_mean (np.ndarray): Mean snapshot subtracted from the data before POD.
                                    Shape: (n_spatial_points,).
        n_modes_save (int): Number of POD modes to compute, save, and use for plotting.
        data_matrix (np.ndarray): Preprocessed data matrix [time, space].
        W (np.ndarray): Spatial weighting matrix (diagonal).
        fs (float): Sampling frequency of the data (if available, mainly for context).

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading and preprocessing.
                      Note: `nfft` and `overlap` from `BaseAnalyzer` are not directly used by POD
                      but are initialized with dummy values.
    """

    def __init__(
        self,
        file_path: str,
        results_dir: str = RESULTS_DIR_POD,
        figures_dir: str = FIGURES_DIR_POD,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        n_modes_save: int = 10,
        use_parallel: bool = True,
        spatial_weights: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the PODAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_POD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_POD` from `configs.py`.
            data_loader (callable, optional): Custom function to load data from `file_path`.
                                              If None, `BaseAnalyzer` attempts to auto-detect.
                                              Defaults to None.
            spatial_weight_type (str | None, optional): Type of spatial weights to apply
                (None → 'uniform', or 'uniform', 'polar', 'prescribed'). Defaults to None.
            n_modes_save (int, optional): Number of dominant POD modes to compute, save,
                                          and consider for plotting/reconstruction.
                                          Defaults to 10.
            spatial_weights: Optional array of spatial integration weights. When given,
                the type becomes 'prescribed' and the vector is checked against the grid
                in load_and_preprocess.
        """
        # Call BaseAnalyzer's __init__.
        # nfft and overlap are not directly used by POD but are part of BaseAnalyzer.
        super().__init__(
            file_path=file_path,
            nfft=1,  # Not used by POD, can be a dummy value
            overlap=0,  # Not used by POD, can be a dummy value
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
            spatial_weights=spatial_weights,
        )

        self.n_modes_save = n_modes_save
        self.modes = np.array([])  # Spatial modes (Phi)
        self.eigenvalues = np.array([])  # Eigenvalues (lambda)
        self.time_coefficients = np.array([])  # Temporal coefficients (Psi)
        self.temporal_mean = np.array([])  # Temporal mean of the data
        # Pre-truncation total energy (sum of all eigenvalues); nan until perform/load.
        self.total_energy = float("nan")
        # Truncated energy / pre-truncation total; set in perform_pod.
        self.energy_captured_fraction = float("nan")

        # Update the analysis type for filenames
        self.analysis_type = "pod"

    def perform_pod(self, *, solver: Literal["eigh", "svd"] = "eigh") -> None:
        """Perform POD analysis on the loaded and preprocessed data.

        Parameters
        ----------
        solver
            Second-order route: ``"eigh"`` (default) factors the correlation /
            Gram kernel; ``"svd"`` factors the weighted snapshot matrix. See
            DOC.md for when the SVD route is worth selecting. Unknown names
            raise ``ValueError``.

        This method computes the POD modes, eigenvalues, and time coefficients.
        The steps involved are:
        1. Ensure data is loaded (expects `self.data['q']` to be [time, space]).
        2. Subtract the temporal mean (`self.temporal_mean`) from the data matrix.
        3. Apply spatial weights (`self.W`) to the mean-subtracted data.
        4. On the ``"eigh"`` route, form the correlation matrix (snapshot POD
           approach: C = X^T * W * X) and solve its eigenvalue problem. On the
           ``"svd"`` route, factor the weighted snapshot matrix directly, which
           never squares the data and so keeps far more dynamic range.
        5. Recover eigenvalues and the vectors that relate to time coefficients.
        6. Reconstruct spatial modes by projecting the data onto those vectors.
        7. Sort modes and eigenvalues by energy (descending eigenvalues).
        8. Truncate to `self.n_modes_save`.

        Attributes set:
            eigenvalues (np.ndarray): Sorted POD eigenvalues.
            modes (np.ndarray): Sorted spatial POD modes.
            time_coefficients (np.ndarray): Sorted temporal coefficients.
            temporal_mean (np.ndarray): Calculated temporal mean of the data.
        """
        if solver not in ("eigh", "svd"):
            raise ValueError(f"solver must be 'eigh' or 'svd', got {solver!r}")
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        logger.info("Performing %s analysis...", display_name_for(self.analysis_type))
        start_time = time.time()

        # Data is expected as [time, space]
        data_matrix = self.data["q"]  # Shape (Ns, Nspace)
        num_snapshots, num_space_points = data_matrix.shape

        # Input validation
        if num_snapshots < 2:
            raise ValueError(f"Need at least 2 snapshots for POD, got {num_snapshots}")
        if num_space_points < 1:
            raise ValueError(f"Need at least 1 spatial point, got {num_space_points}")

        # 1. Subtract temporal mean (more efficient with axis parameter)
        self.temporal_mean = cast(np.ndarray, np.mean(data_matrix, axis=0, dtype=np.float64))
        data_mean_removed = data_matrix - self.temporal_mean

        # 2. Spatial metric (inner-product weights)
        if self.spatial_weight_type == "uniform":
            self.W = np.ones(num_space_points, dtype=np.float64)

        # Ensure W is a 1D array for efficient broadcasting
        if self.W.ndim == 2:
            if self.W.shape[0] == self.W.shape[1]:
                weight_vector = np.diag(self.W)
            elif self.W.shape[1] == 1:
                weight_vector = self.W.ravel()
            else:
                raise ValueError(f"Unexpected shape for spatial weights W: {self.W.shape}")
        else:
            weight_vector = self.W

        # Validate weight vector
        if len(weight_vector) != num_space_points:
            raise ValueError(
                f"Weight vector length {len(weight_vector)} doesn't match spatial points {num_space_points}"
            )

        use_temporal_kernel = num_snapshots < num_space_points
        if use_temporal_kernel:
            logger.info(
                "Using temporal kernel: %d snapshots < %d spatial points",
                num_snapshots,
                num_space_points,
            )
        else:
            logger.info(
                "Using spatial kernel: %d spatial points <= %d snapshots",
                num_space_points,
                num_snapshots,
            )

        # 3. Identity lift + weighted second-order eigenproblem.
        # Mean-centering costs one SAMPLE degree of freedom, not one of the
        # smaller matrix dimension. Rank bound is therefore
        # min(n_samples - 1, n_space), floored at 1 so a single-DOF field still
        # returns a mode. The solver may still drop values below its relative
        # cutoff, so fewer than k may come back.
        self._lift: decomposition.Lift = decomposition.IdentityLift()
        metric = decomposition.SpatialMetric(weight_vector)
        lifted = self._lift.apply(data_mean_removed)
        n_samples_lift, n_space_lift = lifted.shape
        max_rank = max(min(n_samples_lift - 1, n_space_lift), 1)
        k = min(self.n_modes_save, max_rank)
        if k < self.n_modes_save:
            logger.warning("Only %d modes available, requested %d", k, self.n_modes_save)
            self.n_modes_save = k

        self.modes, self.eigenvalues, self.time_coefficients = decomposition.weighted_second_order(
            lifted,
            metric,
            method=solver,
            n_keep=k,
        )

        # True pre-truncation total: sum of all sigma²/m = ‖data_weighted‖_F² / m.
        # Same exact sqrt(W) as the solver so the identity holds on both routes,
        # independent of k.
        sqrt_weights = np.sqrt(metric.weights)
        data_weighted = lifted * sqrt_weights
        n_samples = lifted.shape[0]
        self.total_energy = float(np.linalg.norm(data_weighted, "fro") ** 2 / n_samples)

        # Truncate to requested number of modes (solver may still return fewer
        # after its relative cutoff).
        self._resync_mode_count()

        end_time = time.time()
        logger.info(
            "%s analysis completed in %.2f seconds.",
            display_name_for(self.analysis_type),
            end_time - start_time,
        )
        logger.info(
            "Computed %d %s modes.",
            self.modes.shape[1],
            display_name_for(self.analysis_type),
        )

        # Fraction of pre-truncation energy retained by the saved modes.
        if self.total_energy > 0:
            self.energy_captured_fraction = float(np.sum(self.eigenvalues) / self.total_energy)
            logger.info(
                "Energy captured by %d modes: %.2f%%",
                self.n_modes_save,
                100.0 * self.energy_captured_fraction,
            )
        else:
            self.energy_captured_fraction = 0.0

    def _energy_denominator(self) -> tuple[float, str]:
        """Denominator for energy percentages and an optional label suffix.

        Prefer the true pre-truncation total when known. Result files written
        before that attribute was stored fall back to the sum of retained
        eigenvalues; the label suffix says so so titles stay honest.
        """
        total = getattr(self, "total_energy", float("nan"))
        if np.isfinite(total) and total > 0:
            return float(total), ""
        retained = float(self.eigenvalues.sum()) if self.eigenvalues.size else 0.0
        return retained, " (retained modes only)"

    def _get_algorithm_metadata(self) -> dict:
        """Describe the current POD contract."""
        lift = getattr(self, "_lift", None) or decomposition.IdentityLift()
        meta = {
            "lift_kind": lift.kind,
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_second_order_operator": True,
            "eigenvalue_normalization": "snapshot_average",
        }
        if np.isfinite(self.total_energy):
            meta["total_energy"] = float(self.total_energy)
        if np.isfinite(self.energy_captured_fraction):
            meta["energy_captured_fraction"] = float(self.energy_captured_fraction)
        return meta

    def load_results(self, filename: str | None = None) -> None:
        """Load POD modes, eigenvalues, and time coefficients from an HDF5 file."""
        from openmodalpy.core.results import read_results

        if not filename:
            filename = f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"
        load_path = os.path.join(self.results_dir, filename)
        logger.info("Loading %s results from %s", display_name_for(self.analysis_type), load_path)
        if not os.path.isfile(load_path):
            # Try to auto-detect a results file for this variable and analysis type
            from openmodalpy.core.results import find_latest_result

            latest = find_latest_result(self.results_dir, f"*_{self.analysis_type}.hdf5")
            if latest:
                load_path = latest
                logger.info("[Auto-detect] Using available results file: %s", load_path)
            else:
                logger.error(
                    "No results file found for plotting in %s matching '*_%s.hdf5'. Run with --compute first.",
                    self.results_dir,
                    self.analysis_type,
                )
                return  # Or: raise FileNotFoundError("No POD results file found for plotting.")

        res = read_results(load_path)
        # Before the unified reader this indexed modes/eigenvalues directly, so a
        # file that was not a POD result raised. Assigning only when present would
        # turn that into empty arrays and a "results loaded" print, so keep it loud.
        if res.modes is None or res.eigenvalues is None or res.time_coefficients is None:
            missing = [
                n
                for n, v in (
                    ("modes", res.modes),
                    ("eigenvalues", res.eigenvalues),
                    ("time_coefficients", res.time_coefficients),
                )
                if v is None
            ]
            raise KeyError(f"{load_path} is not a POD result file: missing {', '.join(missing)}")
        self.modes = res.modes
        self.eigenvalues = res.eigenvalues
        self.time_coefficients = res.time_coefficients
        if res.W is not None:
            self.W = res.W
        if res.temporal_mean is not None:
            self.temporal_mean = res.temporal_mean
        for coord_key in ("x", "y", "z"):
            value = getattr(res, coord_key, None)
            if value is not None:
                self.data[coord_key] = value
            elif coord_key in res.extra:
                self.data[coord_key] = res.extra[coord_key]
        if "dt" in res.attrs:
            self.data["dt"] = res.attrs["dt"]
        if "n_snapshots" in res.attrs:
            self.data["Ns"] = res.attrs["n_snapshots"]
        if "Nspace" in res.attrs:
            self.data["Nspace"] = res.attrs["Nspace"]
        if "Nx" in res.attrs:
            self.data["Nx"] = int(res.attrs["Nx"])
        if "Ny" in res.attrs:
            self.data["Ny"] = int(res.attrs["Ny"])
        if "Nz" in res.attrs:
            self.data["Nz"] = int(res.attrs["Nz"])
        # Reset first: a file without these attrs means "unknown", and a
        # stale total from an earlier run on other data would otherwise be
        # used as the denominator with no "retained modes only" label.
        self.total_energy = float("nan")
        self.energy_captured_fraction = float("nan")
        if "total_energy" in res.attrs:
            self.total_energy = float(res.attrs["total_energy"])
        if "energy_captured_fraction" in res.attrs:
            self.energy_captured_fraction = float(res.attrs["energy_captured_fraction"])
        logger.info("%s results loaded.", display_name_for(self.analysis_type))

    def save_results(self, filename: str | None = None) -> None:
        """Save POD modes, eigenvalues, and time coefficients to an HDF5 file.

        The results are saved in `self.results_dir`. If `filename` is None,
        a simplified name is used (POD does not key on nfft/overlap).

        Datasets (canonical names): modes, eigenvalues, time_coefficients,
        coordinates, W, temporal_mean.
        """
        from openmodalpy.core.results import write_results

        if not filename:
            # Use a simplified name for POD as nfft/overlap are not primary params
            filename = f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"

        save_path = os.path.join(self.results_dir, filename)
        logger.info("Saving %s results to %s", display_name_for(self.analysis_type), save_path)

        datasets: dict = {
            "modes": self.modes,
            "eigenvalues": self.eigenvalues,
            "time_coefficients": self.time_coefficients,
        }
        if "x" in self.data:
            datasets["x"] = self.data["x"]
        if "y" in self.data:
            datasets["y"] = self.data["y"]
        if "z" in self.data and self.data["z"] is not None:
            datasets["z"] = self.data["z"]
        if self.W.size > 0:
            datasets["W"] = self.W
        if self.temporal_mean.size > 0:
            datasets["temporal_mean"] = self.temporal_mean

        attrs = self._get_metadata()
        attrs["n_modes_saved"] = self.n_modes_save
        attrs["n_snapshots"] = self.data.get("Ns", 0)
        attrs["Nspace"] = self.modes.shape[0]
        write_results(save_path, datasets, attrs=attrs)
        logger.info("%s results saved.", display_name_for(self.analysis_type))

    def plot_eigenvalues(self) -> None:
        """Plot the POD eigenvalue spectrum (energy vs. mode number).

        Shows the decay of energy (eigenvalues) with increasing mode number.
        The plot is saved to `self.figures_dir`.
        """
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_pod() first.")
            return

        # Create figure with better memory management
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            mode_indices = np.arange(1, len(self.eigenvalues) + 1)
            denom, label_suffix = self._energy_denominator()
            normalized_eigenvals = (self.eigenvalues / denom * 100) if denom > 0 else np.zeros_like(self.eigenvalues)

            ax.plot(mode_indices, normalized_eigenvals, "o-", linewidth=2, markersize=6)

            # Annotate only first few and last few points to avoid clutter
            n_annotate = min(5, len(mode_indices))
            for idx in range(n_annotate):
                ax.text(mode_indices[idx], normalized_eigenvals[idx], f" {idx + 1}", fontsize=7, va="bottom")
            if len(mode_indices) > n_annotate:
                for idx in range(max(n_annotate, len(mode_indices) - 3), len(mode_indices)):
                    ax.text(mode_indices[idx], normalized_eigenvals[idx], f" {idx + 1}", fontsize=7, va="bottom")

            ax.set_yscale("log")
            ax.set_xlabel("Mode Number")
            ax.set_ylabel(f"Normalized Eigenvalue (Energy Percentage %){label_suffix}")
            method_label = display_name_for(self.analysis_type)
            ax.set_title(f"{method_label} Eigenvalue Spectrum")
            ax.grid(True, which="both", ls="--")

            plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_eigenvalues.png")
            plt.savefig(plot_filename, dpi=FIG_DPI, bbox_inches="tight")
            logger.info("Saving figure %s", plot_filename)

        finally:
            plt.close(fig)  # Ensure figure is closed even if error occurs

    def plot_modes(self, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1, show_cylinder: bool = False) -> None:
        """Plot the spatial POD modes.

        Visualizes the first `n_modes_to_plot` dominant spatial modes.
        Requires spatial coordinates (e.g., `self.data['x']`, `self.data['y']`) to be loaded.
        Assumes 2D spatial data that can be reshaped using `Nx` and `Ny` from `self.data`.
        Plots are saved to `self.figures_dir`.

        Args:
            plot_n_modes (int | None, optional): Number of leading spatial modes to
                plot. If ``None`` all available modes are plotted. Defaults to 10.
            modes_per_fig (int): Number of modes per figure. Defaults to 1.
            show_cylinder (bool): If True, add cylinder mask at origin with radius 0.5. Defaults to False.
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_pod() first.")
            return

        n_modes = self.modes.shape[1]
        if plot_n_modes is not None:
            n_modes = min(plot_n_modes, n_modes, self.n_modes_save)
        if n_modes == 0:
            logger.warning("No modes available to plot.")
            return

        if resolve_volume_layout(self.data, self.modes.shape[0]) is not None:
            self.plot_modes_3d_slices(plot_n_modes=n_modes)
            return

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        for start in range(0, n_modes, modes_per_fig):
            end = min(start + modes_per_fig, n_modes)
            ncols = end - start
            if not is_2d_plot:
                logger.warning("plot_modes currently supports 2-D fields only.")
                return

            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(4 * ncols * fig_aspect, 4),
                squeeze=False,
            )
            for idx, mode_idx in enumerate(range(start, end)):
                ax = axes[0, idx]
                mode = self.modes[:, mode_idx]
                # Reshape mode to 2D
                mode_2d = mode.reshape((Nx, Ny))
                # Get meshgrid for plotting
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords
                # Optionally apply cylinder mask (always mask NaNs)
                nan_mask = np.isnan(mode_2d)
                if show_cylinder:
                    distance = np.sqrt(x_mesh**2 + y_mesh**2)
                    cylinder_mask = distance <= 0.5
                    combined_mask = nan_mask | cylinder_mask
                else:
                    combined_mask = nan_mask
                mode_plot = np.ma.array(mode_2d, mask=combined_mask)
                mode_flat = mode_2d[~combined_mask]
                # Guard against empty array (e.g., all points masked)
                if mode_flat.size == 0:
                    logger.warning("Mode %d has no valid data points, skipping plot.", mode_idx)
                    continue
                # Compute levels with robust limits

                # Use percentile method but don't force symmetry for sequential colormap
                mode_clean = mode_flat[np.isfinite(mode_flat)]
                vmin, vmax = np.percentile(mode_clean, [2, 98]) if len(mode_clean) > 0 else (0, 1)
                levels = np.linspace(vmin, vmax, 21)
                # Plot filled contour
                cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_SEQ, extend="both")
                # Contour lines
                _ = ax.contour(x_mesh, y_mesh, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                # Optionally add cylinder overlay
                if show_cylinder:
                    cylinder = plt.Circle(
                        (0, 0), 0.5, fill=True, linewidth=0.5, zorder=3, facecolor="lightgray", edgecolor="black"
                    )
                    ax.add_patch(cylinder)
                # Labels and aspect
                ax.set_xlabel(r"$x/D$")
                ax.set_ylabel(r"$y/D$")
                ax.set_aspect("equal", "box")
                ax.set_xlim(np.min(x_coords), np.max(x_coords))
                ax.set_ylim(np.min(y_coords), np.max(y_coords))
                ax.grid(True, linestyle="--", alpha=0.3)
                # Calculate energy and cumulative energy
                if self.eigenvalues is not None and len(self.eigenvalues) > mode_idx:
                    denom, label_suffix = self._energy_denominator()
                    energy_pct = 100.0 * self.eigenvalues[mode_idx] / denom if denom > 0 else 0.0
                    cum_energy_pct = 100.0 * np.sum(self.eigenvalues[: mode_idx + 1]) / denom if denom > 0 else 0.0
                    label = display_name_for(self.analysis_type)
                    title_str = (
                        f"{label} Mode {mode_idx + 1} [{var_name}] | Energy: {energy_pct:.2f}% | "
                        f"Cumulative: {cum_energy_pct:.2f}%{label_suffix}"
                    )
                else:
                    title_str = f"{display_name_for(self.analysis_type)} Mode {mode_idx + 1} [{var_name}]"
                ax.set_title(title_str)
                # Colorbar
                fig.colorbar(cf, ax=ax, format="%.2f")

            fig.tight_layout()
            # Save figure as PNG with dpi=FIG_DPI
            plot_filename = os.path.join(
                self.figures_dir, f"{self.data_root}_{self.analysis_type}_mode_{start + 1}_to_{end}.png"
            )
            plt.savefig(plot_filename, dpi=FIG_DPI)
            plt.close(fig)
            logger.info("Saving figure %s", plot_filename)

    def plot_modes_pair_detailed(
        self, plot_n_modes: int = 4, cmap: str = CMAP_SEQ, show_cylinder: bool = False
    ) -> None:
        """Plot modes in pairs with an additional magnitude row (2×2 per figure).

        Produces figures where the top row contains the raw spatial fields for a
        pair of modes (e.g. mode 1 and 2) and the bottom row contains their
        magnitudes.  Designed to replicate the 4-panel style the user wants for
        `pod_mode_1_to_2.png`, `pod_mode_3_to_4.png`, etc.
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_pod() first.")
            return

        n_modes = min(plot_n_modes, self.modes.shape[1], self.n_modes_save)
        if n_modes == 0:
            logger.warning("No modes available to plot.")
            return

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)

        # --- Colormap setup for raw and magnitude plots ---

        # Hold contour handles for shared colorbars (set in first loop iteration)

        for start in range(0, n_modes, 2):
            end = min(start + 2, n_modes)
            ncols = end - start
            if not is_2d_plot:
                logger.warning("plot_modes_pair_detailed currently supports 2-D fields only.")
                return

            fig, axes = plt.subplots(
                1, ncols, figsize=(4 * ncols * fig_aspect, 4), squeeze=False, constrained_layout=True
            )

            denom, label_suffix = self._energy_denominator()
            for idx, mode_idx in enumerate(range(start, end)):
                # ------------------ Only plot top row: raw mode ------------------
                ax = axes[0, idx]
                mode_vec = self.modes[:, mode_idx]
                mode_2d = mode_vec.reshape((Nx, Ny))
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords
                # Optionally apply cylinder mask
                field: np.ndarray | np.ma.MaskedArray
                if show_cylinder:
                    dist = np.sqrt(x_mesh**2 + y_mesh**2)
                    mask = dist <= 0.5
                    field = np.ma.array(mode_2d, mask=mask)
                else:
                    field = mode_2d
                from openmodalpy.core.base import get_robust_clim

                vmin, vmax = get_robust_clim(field, method="percentile")
                levels = np.linspace(vmin, vmax, 21)

                cf = ax.contourf(
                    x_mesh,
                    y_mesh,
                    field,
                    levels=levels,
                    cmap=CMAP_DIV,  # diverging colormap for signed mode field
                    extend="both",
                )
                if show_cylinder:
                    ax.add_patch(
                        plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                    )
                style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                add_inset_colorbar(
                    fig,
                    ax,
                    cf,
                    self.data,
                    ticks=[vmin, 0, vmax],
                    ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
                )

                energy_pct = 100.0 * self.eigenvalues[mode_idx] / denom if denom > 0 else 0.0
                cum_pct = 100.0 * np.sum(self.eigenvalues[: mode_idx + 1]) / denom if denom > 0 else 0.0
                ax.set_title(
                    format_mode_title(
                        self.data,
                        mode_idx,
                        default=f"Mode {mode_idx + 1}\nE={energy_pct:.2f}%  Cum={cum_pct:.2f}%{label_suffix}",
                    ),
                    fontsize=8,
                    pad=20,
                )
            plot_filename = os.path.join(
                self.figures_dir, f"{self.data_root}_{self.analysis_type}_mode_{start + 1}_to_{end}.png"
            )
            plt.savefig(plot_filename, dpi=FIG_DPI)
            plt.close(fig)
            logger.info("Saving figure %s", plot_filename)

    def plot_modes_3d_slices(self, plot_n_modes: Optional[int] = 4) -> None:
        """Plot orthogonal 3D slices for the leading POD modes."""
        self._plot_modes_3d("slices", plot_n_modes=plot_n_modes)

    def plot_modes_3d_isometric(self, plot_n_modes: Optional[int] = 4) -> None:
        """Plot isometric 3D views for the leading POD modes."""
        self._plot_modes_3d("isometric", plot_n_modes=plot_n_modes)

    def _plot_modes_3d(self, kind: str, plot_n_modes: Optional[int] = 4) -> None:
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_pod() first.")
            return

        nx = int(self.data.get("Nx", 1))
        ny = int(self.data.get("Ny", 1))
        nz = int(self.data.get("Nz", 1))
        if nz <= 1 or self.modes.shape[0] != nx * ny * nz:
            logger.warning("plot_modes_3d_%s requires volumetric data.", kind)
            return

        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        z_coords = self.data.get("z", np.arange(nz))
        n_modes = self.modes.shape[1]
        if plot_n_modes is not None:
            n_modes = min(plot_n_modes, n_modes, self.n_modes_save)

        denom, label_suffix = self._energy_denominator()
        items = []
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.modes[:, mode_idx], self.data)
            label = display_name_for(self.analysis_type)
            if denom > 0:
                energy_pct = 100.0 * self.eigenvalues[mode_idx] / denom
                title = f"{label} Mode {mode_idx + 1} | E={energy_pct:.2f}%{label_suffix}"
            else:
                title = f"{label} Mode {mode_idx + 1}"
            output_path = os.path.join(
                self.figures_dir, f"{self.data_root}_{self.analysis_type}_mode_{mode_idx + 1}_{kind}.png"
            )
            items.append({"mode_3d": mode_3d, "output_path": output_path, "title_prefix": title})
        plot_modes_3d(kind, items, x_coords, y_coords, z_coords, data=self.data)

    def plot_modes_grid(
        self, energy_threshold: float = 99.5, cmap: str = CMAP_DIV, show_cylinder: bool = False
    ) -> None:
        """Plot spatial POD modes side-by-side up to a cumulative energy threshold.

        This produces a single figure containing all modes required to reach the
        specified cumulative energy percentage (default 99.5%).  Each mode is
        displayed with a diverging colormap so positive and negative regions
        are easily distinguished.  The subplot title indicates the mode number,
        its individual energy contribution, and the cumulative energy captured
        up to that mode.  Axes limits, cylinder overlay, and other style
        choices mirror those used in the DMD detailed mode plots so that the
        two decompositions can be compared directly.
        """
        # Preconditions – ensure POD has been performed
        if self.modes.size == 0 or self.eigenvalues.size == 0:
            logger.warning("No POD modes/eigenvalues to plot. Run perform_pod() first.")
            return

        denom, label_suffix = self._energy_denominator()
        cumulative_pct = np.cumsum(self.eigenvalues) / denom * 100.0 if denom > 0 else np.zeros_like(self.eigenvalues)
        # Number of modes needed to reach threshold (inclusive)
        n_modes_plot = int(np.searchsorted(cumulative_pct, energy_threshold, side="right")) + 1
        n_modes_plot = min(n_modes_plot, self.n_modes_save, self.modes.shape[1])
        if n_modes_plot <= 0:
            logger.warning("Energy threshold too low – nothing to plot.")
            return

        # Spatial grid information
        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        # Always use two columns so rows list modes sequentially (1-2, 3-4, …)
        ncols = 2 if n_modes_plot > 1 else 1
        nrows = int(np.ceil(n_modes_plot / ncols))

        # Create figure with constrained_layout for better spacing
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols * fig_aspect, 4 * nrows), squeeze=False, constrained_layout=True
        )

        # Import make_axes_locatable for better colorbar placement

        # Track the first contourf for colorbar
        # Plot each mode
        for k in range(n_modes_plot):
            row, col = divmod(k, ncols)
            ax = axes[row][col]
            mode_vec = self.modes[:, k]

            if is_2d_plot:
                # Reshape mode to 2D grid
                mode_2d = mode_vec.reshape((Nx, Ny))

                # Create meshgrid for plotting
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords

                # Optionally mask interior of cylinder (radius 0.5)
                mode_plot: np.ndarray | np.ma.MaskedArray
                if show_cylinder:
                    distance = np.sqrt(x_mesh**2 + y_mesh**2)
                    cylinder_mask = distance <= 0.5
                    mode_plot = np.ma.array(mode_2d, mask=cylinder_mask)
                else:
                    mode_plot = mode_2d

                # Calculate contour levels with robust symmetric diverging scale
                from openmodalpy.core.base import get_robust_clim

                vmin, vmax = get_robust_clim(mode_plot, method="percentile")
                levels = np.linspace(vmin, vmax, 21)

                # Plot contours
                cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=cmap, extend="both")

                # Optionally add cylinder overlay
                if show_cylinder:
                    cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                    ax.add_patch(cyl)

                # Set axis properties
                style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                add_inset_colorbar(
                    fig,
                    ax,
                    cf,
                    self.data,
                    ticks=[vmin, 0, vmax],
                    ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
                )

                # Add title with energy information
                energy_pct = 100.0 * self.eigenvalues[k] / denom if denom > 0 else 0.0
                cum_pct = cumulative_pct[k]
                ax.set_title(
                    format_mode_title(
                        self.data,
                        k,
                        default=f"Mode {k + 1}\nE={energy_pct:.2f}%  Cum={cum_pct:.2f}%{label_suffix}",
                    ),
                    fontsize=8,
                    pad=20,
                )
            else:
                # 1D mode plotting (fallback)
                ax.plot(mode_vec)
                ax.set_xlabel("Spatial Index")
                ax.set_ylabel(f"{var_name} amplitude")
                ax.set_title(f"Mode {k + 1}")

        # Hide any extra subplots
        for idx in range(n_modes_plot, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")

        # Add title and save
        fig.suptitle(
            f"{display_name_for(self.analysis_type)} Modes up to {energy_threshold:.1f}% "
            f"cumulative energy ({n_modes_plot} modes)",
            fontsize=12,
        )

        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_{self.analysis_type}_modes_grid_{energy_threshold:.1f}perc.png",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def plot_time_coefficients(
        self,
        n_coeffs_to_plot: int = 2,
        n_snapshots_plot: int | None = None,
        L: float | None = 1.0,
        U: float | None = 1.0,
    ) -> None:
        """Plot the temporal coefficients for selected modes.

        Displays the time evolution of the coefficients for the first `n_coeffs_to_plot` modes.
        If `self.data['t']` (time vector) is available, it's used for the x-axis.
        Otherwise, snapshot index is used.
        Plots are saved to `self.figures_dir`.

        Args:
            n_coeffs_to_plot (int, optional): Number of leading temporal coefficients to plot.
                Defaults to 2.
            n_snapshots_plot (int, optional): Number of time snapshots to include in the plot.
                                              If None, all snapshots are used. Defaults to None.
            L (float, optional): Characteristic length for Strouhal number conversion. Defaults to ``1.0``.
            U (float, optional): Characteristic velocity for Strouhal number conversion. Defaults to ``1.0``.

        The periodogram axis is always shown in terms of Strouhal number.  The
        frequencies are converted according to ``freqs_st = freqs * L / U``.  With
        the default ``L = U = 1`` the plot displays the unscaled frequencies but
        labeled as Strouhal numbers.
        """
        if self.time_coefficients.size == 0:
            logger.warning("No time coefficients to plot. Run perform_pod() first.")
            return

        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1], self.n_modes_save)
        if n_coeffs_to_plot == 0:
            logger.warning("No coefficients available to plot.")
            return

        Ns_total = self.time_coefficients.shape[0]
        if n_snapshots_plot is None or n_snapshots_plot > Ns_total:
            n_snapshots_plot = Ns_total

        time_vector, time_xlabel = self._time_axis(n_snapshots_plot)
        st_xlabel = "Strouhal Number (St)"
        fs = self._require_fs()

        plt.figure(figsize=(12, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 2, 2 * i + 1)
            coeff = self.time_coefficients[:n_snapshots_plot, i]
            plt.plot(time_vector, coeff, ls="-", lw=0.8, marker="o", markersize=1)
            plt.xlabel(time_xlabel)
            plt.ylabel(f"Amplitude Mode {i + 1}")
            plt.title(f"Temporal Coefficient for {display_name_for(self.analysis_type)} Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(time_vector.min(), time_vector.max())

            plt.subplot(n_coeffs_to_plot, 2, 2 * i + 2)
            freqs, psd = periodogram_rfft(coeff, fs)
            peak_freqs, peak_psd = find_peaks(freqs, psd)

            if L is not None and U is not None:
                freqs = freqs * L / U
            plt.semilogy(freqs, psd)
            if peak_freqs.size > 0:
                plt.plot(
                    peak_freqs,
                    peak_psd,
                    ls="-",
                    lw=0.8,
                    marker="o",
                    markersize=2,
                )
                for pf, pv in zip(peak_freqs, peak_psd):
                    plt.text(pf, pv, f"{pf:.2f}", fontsize=8, ha="left", va="bottom")
            plt.xscale("log")
            if peak_freqs.size > 0:
                xlim_min = 0.7 * peak_freqs[0]
                plt.xlim(xlim_min, fs / 2)
            else:
                plt.xlim(1e-3, fs / 2)
            plt.ylim(1e-6, None)
            plt.xlabel(st_xlabel)
            plt.ylabel("PSD")
            plt.title(f"Periodogram Mode {i + 1}")
            plt.grid(True, linestyle=":")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_time_coeffs.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)

    def check_mode_pair_phase(self, start_mode: int = 1, threshold: float = 0.9) -> Iterator[tuple[int, int]]:
        """Identify mode pairs with strongly correlated time coefficients.

        Starting from ``start_mode`` (1-indexed), iterate through the saved POD
        modes and compute the Pearson correlation coefficient between the time
        coefficients of mode ``i`` and candidate mode ``j``.  If the absolute
        value of the correlation exceeds ``threshold`` the pair ``(i, j)`` is
        considered a valid phase pair and is yielded.  If the correlation is
        below the threshold the search continues with ``j`` incremented until a
        suitable partner for mode ``i`` is found or the available modes are
        exhausted.

        Parameters
        ----------
        start_mode : int, optional
            First mode index to test (1-indexed).  Defaults to ``1``.
        threshold : float, optional
            Minimum absolute correlation required to accept a pair.  A value of
            ``1.0`` would require perfectly correlated time coefficients while a
            value of ``0`` would accept any pair.  Defaults to ``0.9``.

        Yields
        ------
        tuple[int, int]
            Mode index pairs that satisfy the correlation criterion.
        """

        if self.time_coefficients.size == 0:
            logger.warning("No time coefficients available. Run perform_pod() first.")
            return

        n_modes = self.time_coefficients.shape[1]
        i = start_mode
        while i < n_modes:
            found = False
            for j in range(i + 1, n_modes + 1):
                coeff_i = self.time_coefficients[:, i - 1]
                coeff_j = self.time_coefficients[:, j - 1]
                corr = np.corrcoef(coeff_i, coeff_j)[0, 1]
                if np.abs(corr) >= threshold:
                    logger.info("Found correlated pair (%d, %d) with r=%.3f", i, j, corr)
                    yield (i, j)
                    i = j + 1
                    found = True
                    break
            if not found:
                logger.info("No correlated partner found for mode %d", i)
                i += 1

    def plot_mode_pair_phase(self, start_mode: int = 1, threshold: float = 0.9) -> None:
        """Plot phase portraits of automatically detected mode pairs.

        Mode pairs are identified using :meth:`check_mode_pair_phase`.  For each
        accepted pair the temporal coefficients of the two modes are plotted
        against each other to visualize their phase relationship.

        Parameters
        ----------
        start_mode : int, optional
            Initial mode index to search from.  Defaults to ``1``.
        threshold : float, optional
            Correlation threshold passed to :meth:`check_mode_pair_phase`.
            Defaults to ``0.9``.
        """

        pairs = list(self.check_mode_pair_phase(start_mode=start_mode, threshold=threshold))
        if not pairs:
            logger.warning("No mode pairs met the correlation threshold.")
            return

        for i, j in pairs:
            coeff_i = self.time_coefficients[:, i - 1]
            coeff_j = self.time_coefficients[:, j - 1]
            plt.figure(figsize=(5, 5))
            plt.plot(coeff_i, coeff_j, "o-", markersize=3, linewidth=0.8)
            plt.xlabel(f"Coefficient {i}")
            plt.ylabel(f"Coefficient {j}")
            plt.title(f"{display_name_for(self.analysis_type)} Phase Portrait Modes {i} & {j}")
            plt.grid(True)
            fname = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_phase_pair_{i}_{j}.png")
            plt.savefig(fname, dpi=FIG_DPI)
            plt.close()
            logger.info("Saving figure %s", fname)

    def plot_cumulative_energy(self) -> None:
        """Plot the cumulative energy captured by POD modes.

        Shows the percentage of total energy captured as more modes are included.
        The plot is saved to `self.figures_dir`.
        """
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_pod() first.")
            return

        denom, label_suffix = self._energy_denominator()
        cumulative_energy = np.cumsum(self.eigenvalues) / denom * 100 if denom > 0 else np.zeros_like(self.eigenvalues)
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        # Annotate cumulative curve with mode numbers
        for idx, (x, y) in enumerate(zip(mode_indices, cumulative_energy)):
            plt.text(float(x), float(y), f" {idx + 1}", fontsize=7, va="bottom")
        plt.xlabel("Number of Modes")
        plt.ylabel(f"Cumulative Energy (%){label_suffix}")
        plt.title(f"Cumulative Energy of {display_name_for(self.analysis_type)} Modes")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)  # Show up to 100% or slightly more
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_cumulative_energy.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)

    def plot_reconstruction_error(self) -> None:
        """Plot the data reconstruction error using an increasing number of POD modes.

        Calculates and plots the normalized mean squared error (NMSE) of reconstructing
        the original data using a subset of POD modes. The error is shown as a function
        of the number of modes used for reconstruction.
        The plot is saved to `self.figures_dir`.
        """
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            logger.warning("Data, modes, or time coefficients not available. Run perform_pod() first.")
            return

        data_matrix = self.data["q"]
        data_mean_removed = data_matrix - self.temporal_mean
        norm_data_mean_removed = np.linalg.norm(data_mean_removed, "fro")

        reconstruction_errors = []
        n_modes_check = self.modes.shape[1]  # Number of available/saved modes

        for k in range(1, n_modes_check + 1):
            # Reconstruct data using k modes: Psi_k @ Phi_k.T
            # self.time_coefficients is (Ns, n_modes_save)
            # self.modes is (Nspace, n_modes_save)
            reconstructed_data_k_modes = self.time_coefficients[:, :k] @ self.modes[:, :k].T
            error = np.linalg.norm(data_mean_removed - reconstructed_data_k_modes, "fro") / norm_data_mean_removed
            reconstruction_errors.append(error * 100)  # As percentage

        mode_indices = np.arange(1, n_modes_check + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, reconstruction_errors, "s-", linewidth=2, markersize=6)
        # Annotate each reconstruction error point with mode number
        for idx, (x, y) in enumerate(zip(mode_indices, reconstruction_errors)):
            plt.text(float(x), float(y), f" {idx + 1}", fontsize=7, va="bottom")
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title(f"Data Reconstruction Error vs. Number of {display_name_for(self.analysis_type)} Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")  # Error often drops off exponentially
        plot_filename = os.path.join(
            self.figures_dir, f"{self.data_root}_{self.analysis_type}_reconstruction_error.png"
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)

    def plot_reconstruction_comparison(
        self,
        snapshot_indices_to_plot: list[int] | None = None,
        modes_for_reconstruction: list[int] | None = None,
    ) -> None:
        """Compare original snapshots with their POD reconstructions.

        Visualizes selected original data snapshots alongside their reconstructions
        using a specified number of POD modes. Requires spatial coordinates for plotting.
        Plots are saved to `self.figures_dir`.

        Args:
            snapshot_indices_to_plot (list of int, optional): Indices of snapshots to plot.
                                                              Defaults to [0, Ns//2, Ns-1] if None.
            modes_for_reconstruction (list of int, optional): Number of modes to use for reconstruction
                                                              in each comparison plot.
                                                              Defaults to [1, n_modes_save//2, n_modes_save] if None.
        """
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            logger.warning("Data, modes, or time coefficients not available. Run perform_pod() first.")
            return

        data_matrix = self.data["q"]
        data_mean_removed = data_matrix - self.temporal_mean
        num_snapshots, num_space_points = data_mean_removed.shape

        if snapshot_indices_to_plot is None:
            snapshot_indices_to_plot = [0, num_snapshots // 2, num_snapshots - 1]
            # Ensure indices are unique and within bounds, especially for small datasets
            snapshot_indices_to_plot = sorted(
                list(set(idx for idx in snapshot_indices_to_plot if 0 <= idx < num_snapshots))
            )
            if not snapshot_indices_to_plot:  # if Ns is too small, pick at least the first one
                snapshot_indices_to_plot = [0]

        if modes_for_reconstruction is None:
            k_max = self.modes.shape[1]
            modes_for_reconstruction = [1, k_max // 2, k_max]
            # Ensure values are unique, positive, and within bounds
            modes_for_reconstruction = sorted(list(set(k for k in modes_for_reconstruction if 0 < k <= k_max)))
            if not modes_for_reconstruction and k_max > 0:  # if k_max is small
                modes_for_reconstruction = [k_max]
            elif k_max == 0:
                logger.warning("No modes available for reconstruction comparison.")
                return

        if not snapshot_indices_to_plot or not modes_for_reconstruction:
            logger.warning("Not enough snapshots or modes to plot reconstruction comparison.")
            return

        # Determine plot layout details (similar to plot_modes)
        Nx = self.data.get("Nx", int(np.sqrt(num_space_points)))
        Ny = self.data.get("Ny", int(np.sqrt(num_space_points)))
        is_2d_plot = (num_space_points == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        num_snapshots_to_show = len(snapshot_indices_to_plot)
        num_recons_per_snapshot = len(modes_for_reconstruction)

        # Each row: Original + Reconstructions
        # num_cols = 1 (original) + num_recons_per_snapshot
        fig, axes = plt.subplots(
            num_snapshots_to_show,
            1 + num_recons_per_snapshot,
            figsize=(5 * (1 + num_recons_per_snapshot), 4 * num_snapshots_to_show),
            squeeze=False,
        )  # ensure axes is always 2D array

        # Setup mesh coordinates for 2D plotting
        if is_2d_plot:
            if x_coords.ndim == 1 and y_coords.ndim == 1:
                x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
            else:
                x_mesh, y_mesh = x_coords, y_coords

        for i, snap_idx in enumerate(snapshot_indices_to_plot):
            original_snapshot = data_mean_removed[snap_idx, :]

            # Plot original snapshot
            ax = axes[i, 0]
            if is_2d_plot:
                field_2d = original_snapshot.reshape((Nx, Ny))
                vmin, vmax = np.nanmin(field_2d), np.nanmax(field_2d)
                levels = np.linspace(vmin, vmax, 21)
                cf = ax.contourf(x_mesh, y_mesh, field_2d, levels=levels, cmap=CMAP_DIV, extend="both")
                ax.set_xlabel(r"$x/D$")
                ax.set_ylabel(r"$y/D$")
                ax.set_aspect("equal", "box")
                ax.set_title(f"Original (t={snap_idx})")
                plt.colorbar(cf, ax=ax, shrink=0.8)
            else:
                ax.plot(original_snapshot)
                ax.set_xlabel("Spatial index")
                ax.set_ylabel("Value")
                ax.set_title(f"Original (t={snap_idx})")

            # Plot reconstructions with different numbers of modes
            for j, n_modes_recon in enumerate(modes_for_reconstruction):
                ax_recon = axes[i, j + 1]
                reconstructed = self.modes[:, :n_modes_recon] @ self.time_coefficients[snap_idx, :n_modes_recon]

                if is_2d_plot:
                    recon_2d = reconstructed.reshape((Nx, Ny))
                    cf = ax_recon.contourf(x_mesh, y_mesh, recon_2d, levels=levels, cmap=CMAP_DIV, extend="both")
                    ax_recon.set_xlabel(r"$x/D$")
                    ax_recon.set_ylabel(r"$y/D$")
                    ax_recon.set_aspect("equal", "box")
                    ax_recon.set_title(f"Recon. k={n_modes_recon}")
                    plt.colorbar(cf, ax=ax_recon, shrink=0.8)
                else:
                    ax_recon.plot(reconstructed)
                    ax_recon.set_xlabel("Spatial index")
                    ax_recon.set_ylabel("Value")
                    ax_recon.set_title(f"Recon. k={n_modes_recon}")

        plt.tight_layout()
        plot_filename = os.path.join(
            self.figures_dir, f"{self.data_root}_{self.analysis_type}_reconstruction_comparison.png"
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)

    def check_spatial_mode_orthogonality(self, tolerance: float = 1e-9) -> bool:
        """Check the orthogonality of spatial modes with respect to weights W.

        Verifies that `Modes.T @ W_diag @ Modes` is close to the identity matrix,
        where W_diag is the diagonal weight matrix. This ensures that POD modes
        form an orthonormal basis with respect to the inner product defined by W.

        Args:
            tolerance: Maximum allowed deviation from identity matrix.
                Defaults to 1e-9.

        Returns:
            bool: True if modes are orthogonal within the specified tolerance.
        """
        if self.modes.size == 0 or self.W.size == 0:
            logger.warning("Modes or weights not available. Run perform_pod() first.")
            return False

        logger.info("Checking spatial mode orthogonality (Modes.T @ W @ Modes)...")
        Nspace, n_saved_modes = self.modes.shape

        # Ensure W is a diagonal matrix for the check
        if self.W.ndim == 1:
            W_diag_matrix = np.diag(self.W)
        elif self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1] and np.allclose(self.W, np.diag(np.diag(self.W))):
            W_diag_matrix = self.W
        elif self.W.ndim == 2 and self.W.shape[1] == 1:  # (Nspace, 1) column vector
            W_diag_matrix = np.diag(self.W.flatten())
        else:
            logger.warning(
                "Unexpected shape or type for spatial weights W: %s. Cannot perform accurate orthogonality check.",
                self.W.shape,
            )
            return False

        ortho_check_matrix = self.modes.T @ W_diag_matrix @ self.modes
        # Check diagonals are close to 1
        diag_diff = np.abs(np.diag(ortho_check_matrix) - 1.0)
        max_diag_deviation = np.max(diag_diff)

        # Check off-diagonals are close to 0
        off_diag_mask = ~np.eye(n_saved_modes, dtype=bool)
        max_off_diag_val = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_saved_modes > 1 else 0.0

        is_orthogonal = (max_diag_deviation < tolerance) and (max_off_diag_val < tolerance)

        logger.info("Max deviation of diagonal elements from 1: %.2e", max_diag_deviation)
        logger.info("Max absolute value of off-diagonal elements: %.2e", max_off_diag_val)
        if is_orthogonal:
            logger.info("Spatial modes appear to be W-orthogonal.")
        else:
            logger.warning("Spatial modes may not be perfectly W-orthogonal within tolerance.")

        # Optional: Plot the orthogonality matrix
        plt.figure(figsize=(7, 6))
        plt.imshow(
            ortho_check_matrix,
            cmap=CMAP_DIV,
            vmin=-np.max(np.abs(ortho_check_matrix)),
            vmax=np.max(np.abs(ortho_check_matrix)),
        )
        plt.colorbar(label="Value")
        plt.title("Spatial Mode Orthogonality Check (Modes.T @ W @ Modes)")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_{self.analysis_type}_spatial_ortho_check.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)
        return is_orthogonal

    def check_temporal_coefficient_orthogonality(self, tolerance: float = 1e-9) -> bool:
        """Check the orthogonality of temporal coefficients.

        Verifies that `(1/Ns) * TimeCoeffs.T @ TimeCoeffs` is close to `diag(Eigenvalues)`,
        where `Ns` is the number of snapshots.
        Prints a message indicating whether the coefficients are orthogonal (scaled by eigenvalues)
        within the given tolerance.

        Args:
            tolerance (float, optional): Tolerance for checking orthogonality.
                                       Defaults to 1e-9.
        """
        if self.time_coefficients.size == 0 or self.eigenvalues.size == 0 or "Ns" not in self.data:
            logger.warning("Time coefficients, eigenvalues, or Ns not available. Run perform_pod() first.")
            return False

        logger.info("Checking temporal coefficient pseudo-orthogonality ((1/Ns) * Psi.T @ Psi)...")
        num_snapshots = self.data["Ns"]
        n_saved_coeffs = self.time_coefficients.shape[1]

        # Expected matrix based on POD theory for snapshot POD temporal eigenvectors
        # (Psi_temp.T @ Psi_temp) should be Identity if Psi_temp are normalized eigenvectors of K_t.
        # self.time_coefficients = Psi_temp * sqrt(eigenvalues_temp * Ns)
        # So (1/Ns) * self.time_coefficients.T @ self.time_coefficients =
        # (1/Ns) * sqrt(L*Ns) * Psi_temp.T @ Psi_temp * sqrt(L*Ns) = L (diag(eigenvalues))
        ortho_check_matrix = (1.0 / num_snapshots) * (self.time_coefficients.T @ self.time_coefficients)
        expected_diag_matrix = np.diag(self.eigenvalues[:n_saved_coeffs])

        diff_matrix = ortho_check_matrix - expected_diag_matrix

        # Check diagonals
        diag_abs_error = np.abs(np.diag(diff_matrix))
        max_diag_abs_error = np.max(diag_abs_error)

        # Check off-diagonals (should be close to 0 in both ortho_check_matrix and expected_diag_matrix)
        off_diag_mask = ~np.eye(n_saved_coeffs, dtype=bool)
        max_off_diag_val_computed = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_saved_coeffs > 1 else 0.0

        # is_orthogonal means diag(ortho_check_matrix) approx diag(expected_matrix) AND off-diag(ortho_check_matrix) approx 0
        is_pseudo_orthogonal = (max_diag_abs_error < tolerance) and (max_off_diag_val_computed < tolerance)

        logger.info(
            "Max absolute error of diagonal elements from eigenvalues: %.2e",
            max_diag_abs_error,
        )
        logger.info(
            "Max absolute value of off-diagonal elements in computed matrix: %.2e",
            max_off_diag_val_computed,
        )
        if is_pseudo_orthogonal:
            logger.info("Temporal coefficients appear to satisfy (1/Ns) * Psi.T @ Psi = diag(Lambda).")
        else:
            logger.warning("Temporal coefficients may not perfectly satisfy (1/Ns) * Psi.T @ Psi = diag(Lambda).")

        # Optional: Plot the computed matrix and the expected diagonal matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot 1: (1/Ns) * Psi.T @ Psi
        im1 = axes[0].imshow(
            ortho_check_matrix,
            cmap=CMAP_DIV,
            vmin=-np.max(np.abs(ortho_check_matrix)),
            vmax=np.max(np.abs(ortho_check_matrix)),
        )
        fig.colorbar(im1, ax=axes[0], label="Value")
        axes[0].set_title("(1/Ns) * Psi.T @ Psi (Computed)")
        axes[0].set_xlabel("Mode Index")
        axes[0].set_ylabel("Mode Index")
        # Plot 2: diag(Eigenvalues)
        im2 = axes[1].imshow(
            expected_diag_matrix,
            cmap=CMAP_DIV,
            vmin=-np.max(np.abs(expected_diag_matrix)),
            vmax=np.max(np.abs(expected_diag_matrix)),
        )
        fig.colorbar(im2, ax=axes[1], label="Value")
        axes[1].set_title("diag(Eigenvalues) (Expected)")
        axes[1].set_xlabel("Mode Index")
        axes[1].set_ylabel("Mode Index")

        plt.tight_layout()
        plot_filename = os.path.join(
            self.figures_dir, f"{self.data_root}_{self.analysis_type}_temporal_ortho_check.png"
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        logger.info("Saving figure %s", plot_filename)
        return is_pseudo_orthogonal

    def _perform_decomposition(self) -> None:
        """Run this analyzer's decomposition. Subclasses override (e.g. mPOD)."""
        self.perform_pod()

    def run_analysis(
        self,
        plot_n_modes_spatial: int = 4,
        plot_n_coeffs_time: int = 5,
        plot_snapshot_indices: list[int] | None = None,
        modes_for_reconstruction: list[int] | None = None,
        check_orthogonality: bool = False,
    ) -> None:
        """
        Main entry point for running POD analysis and plotting.
            check_orthogonality (bool, optional): If True, perform and print orthogonality checks.
                                                Defaults to True.
        """
        logger.info(
            "Starting %s analysis for %s",
            display_name_for(self.analysis_type),
            os.path.basename(self.file_path),
        )
        start_total_time = time.time()

        # Load data and calculate weights via BaseAnalyzer's run method.
        # compute_fft=False because POD is time-domain.
        super().run(compute_fft=False)

        # Subclasses override _perform_decomposition (mPOD → perform_mpod).
        self._perform_decomposition()

        # Identify correlated mode pairs only when plotting

        # Save results
        self.save_results()  # This already calls super().save_results()

        # Plotting
        self.plot_eigenvalues()
        if int(self.data.get("Nz", 1)) > 1:
            for stale_name in (
                f"{self.data_root}_{self.analysis_type}_modes_grid_99.5perc.png",
                f"{self.data_root}_{self.analysis_type}_reconstruction_comparison.png",
            ):
                stale_path = os.path.join(self.figures_dir, stale_name)
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            self.plot_modes_3d_slices(plot_n_modes=plot_n_modes_spatial)
            self.plot_modes_3d_isometric(plot_n_modes=plot_n_modes_spatial)
        else:
            # Detailed 4-panel mode plots (pairs with magnitude)
            self.plot_modes_pair_detailed(plot_n_modes=plot_n_modes_spatial)
            # Phase portraits for correlated pairs
            self.plot_mode_pair_phase()
            # New: comprehensive grid of modes up to cumulative energy threshold for easy DMD comparison
            self.plot_modes_grid(energy_threshold=99.5)
        self.plot_time_coefficients(n_coeffs_to_plot=plot_n_coeffs_time)
        self.plot_cumulative_energy()
        self.plot_reconstruction_error()
        if int(self.data.get("Nz", 1)) <= 1:
            self.plot_reconstruction_comparison(
                snapshot_indices_to_plot=plot_snapshot_indices, modes_for_reconstruction=modes_for_reconstruction
            )

        if check_orthogonality:
            self.check_spatial_mode_orthogonality()
            self.check_temporal_coefficient_orthogonality()

        end_total_time = time.time()
        logger.info(
            "%s analysis and plotting completed successfully in %.2f seconds.",
            display_name_for(self.analysis_type),
            end_total_time - start_total_time,
        )
        print_summary(display_name_for(self.analysis_type), self.results_dir, self.figures_dir)
