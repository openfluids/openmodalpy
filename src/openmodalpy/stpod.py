#!/usr/bin/env python3
"""
Spatio-Temporal Proper Orthogonal Decomposition (ST-POD)

The current analyzer is delay-embedded POD: it constructs a block Hankel matrix
from time-delayed snapshots and performs a weighted SVD in that lifted space.
It should not be read as a direct discretization of the full two-time
space-time covariance operator.

Mathematical basis:
    H = [q₁    q₂    ...  qₘ   ]     d = embedding dimension
        [q₂    q₃    ...  qₘ₊₁ ]     m = Ns - d + 1 (columns)
        [⋮     ⋮     ⋱    ⋮    ]
        [qₐ   qₐ₊₁   ...  qₘ₊ₐ₋₁]

    SVD(H) = U Σ Vᴴ  →  U columns = space-time modes (d stacked spatial fields)

Author: R. Frantz

Reference:
    - Sieber, M., Paschereit, C. O., & Oberleithner, K. (2016).
      "Spectral proper orthogonal decomposition." JFM, 792, 798-828.
"""

import logging
import os
import time
from collections.abc import Callable
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fftkit import find_peaks, periodogram_rfft

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import (
    BaseAnalyzer,
    get_fig_aspect_ratio,
    plot_modes_3d,
    reshape_mode_to_volume,
    resolve_volume_layout,
)
from openmodalpy.core.config import (
    CMAP_DIV,
    FIG_DPI,
    FIGURES_DIR_STPOD,
    RESULTS_DIR_STPOD,
)
from openmodalpy.core.results import AnalysisResults

logger = logging.getLogger(__name__)


class STPODAnalyzer(BaseAnalyzer):
    """Spatio-Temporal POD analyzer using time-delay embedding.

    ST-POD constructs a block Hankel matrix from the data snapshots and performs
    SVD to extract space-time modes that capture both spatial structure and
    temporal evolution.

    Key Attributes:
        embedding_dim (int): Number of time delays (d). The Hankel matrix has
            d*Nspace rows.
        modes (np.ndarray): Space-time modes. Shape: (d * Nspace, n_modes_save).
            Each mode consists of d stacked spatial fields.
        eigenvalues (np.ndarray): Squared singular values from the weighted
            Hankel matrix divided by the number of Hankel columns. Shape:
            (n_modes_save,).
        time_coefficients (np.ndarray): Temporal coefficients from Vᵀ.
            Shape: (m, n_modes_save) where m = Ns - d + 1.
        temporal_mean (np.ndarray): Mean snapshot. Shape: (Nspace,).

    Example:
        >>> analyzer = STPODAnalyzer("data.npz", embedding_dim=20, n_modes_save=10)
        >>> analyzer.run_analysis()
    """

    _METHOD_NAME = "stpod"

    def __init__(
        self,
        file_path: str | None = None,
        *,
        embedding_dim: int = 10,
        n_modes_save: int = 10,
        results_dir: str = RESULTS_DIR_STPOD,
        figures_dir: str = FIGURES_DIR_STPOD,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        spatial_weights: np.ndarray | None = None,
        data: dict[str, Any] | None = None,
    ):
        """Initialize the STPODAnalyzer.

        Args:
            file_path: Path to the data file. Optional when ``data`` carries
                the loaded dataset instead.
            n_modes_save: Number of modes to compute and save.
            results_dir: Directory to save results.
            figures_dir: Directory to save figures.
            data_loader: Custom function to load data.
            spatial_weight_type: Type of spatial weights
                (None → 'uniform', or 'uniform', 'polar', 'prescribed').
            spatial_weights: Optional array of spatial integration weights. When given,
                the type becomes 'prescribed'.
            data: Already-loaded dataset following the data contract (see
                DOC.md). Given instead of ``file_path``.
        """
        # ST-POD forms no FFT blocks, so it takes no nfft/overlap;
        # BaseAnalyzer sets its own dummy stamp.
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            spatial_weights=spatial_weights,
            data=data,
        )

        self.embedding_dim = embedding_dim
        self.n_modes_save = n_modes_save
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        # Any: np.mean(..., axis=0, dtype=float64) is typed as scalar | ndarray.
        self.temporal_mean: Any = np.array([])
        # Pre-truncation total energy (‖data_weighted‖_F² / m); nan until perform/load.
        self.total_energy = float("nan")
        self.energy_captured_fraction = float("nan")

        self.analysis_type = "stpod"

    def _build_hankel_matrix(self, data_centered: np.ndarray) -> np.ndarray:
        """Build the block Hankel matrix from centered data.

        Returns shape ``(d * Nspace, m)`` with ``m = Ns - d + 1`` — the
        historical layout used by reconstruction checks. The delay lift
        itself produces samples × features; this method transposes for the
        column-oriented Hankel contract.
        """
        lift = decomposition.DelayEmbeddingLift(self.embedding_dim)
        return np.ascontiguousarray(lift.apply(data_centered).T)

    def _get_weight_vector(self, num_space_points: int) -> np.ndarray:
        """Extract weight vector from self.W, handling various shapes."""
        if self.W.ndim == 2:
            if self.W.shape[0] == self.W.shape[1]:
                return decomposition._as_weight_vector(np.asarray(self.W), num_space_points)
            elif self.W.shape[1] == 1:
                return self.W.ravel()
            else:
                raise ValueError(f"Unexpected weight shape: {self.W.shape}")
        return self.W

    def perform_stpod(self) -> None:
        """Perform ST-POD analysis on the loaded data.

        The algorithm:
        1. Validate embedding dimension.
        2. Subtract temporal mean.
        3. Apply the delay lift → (m, d*Nspace), m = Ns - d + 1.
        4. Tile the spatial metric over the d delay blocks (I_d ⊗ W).
        5. Solve via ``weighted_second_order(..., method="svd")``, which
           weights, takes the SVD, unweights the modes, and returns
           eigenvalues = sigma²[:k] / m with coefficients Vt scaled by sigma.

        The SVD route is deliberate: forming the Gram matrix would square the
        condition number of an already ill-conditioned Hankel matrix.
        """
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        data_matrix = self.data["q"]  # Shape (Ns, Nspace)
        Ns, Nspace = data_matrix.shape

        # Validate parameters
        if self.embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {self.embedding_dim}")
        if self.embedding_dim >= Ns:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be < number of snapshots ({Ns})")

        m = Ns - self.embedding_dim + 1  # Number of Hankel columns
        logger.info(
            "Performing ST-POD: d=%d, m=%d columns, Hankel shape=(%d, %d)",
            self.embedding_dim,
            m,
            self.embedding_dim * Nspace,
            m,
        )
        start_time = time.time()

        # 1. Subtract temporal mean
        self.temporal_mean = np.mean(data_matrix, axis=0, dtype=np.float64)
        data_centered = data_matrix - self.temporal_mean

        # 2. Delay lift → samples × lifted features (m, d*Nspace)
        self._lift = decomposition.DelayEmbeddingLift(self.embedding_dim)
        lifted = self._lift.apply(data_centered)

        # 3. Metric in the lifted space: I_d ⊗ W
        weight_vector = self._get_weight_vector(Nspace)
        base_metric = decomposition.SpatialMetric(weight_vector)
        lifted_metric = decomposition.SpatialMetric(base_metric.tile(self.embedding_dim))

        # 4. Weighted SVD in the lifted space (do not square the Hankel matrix).
        # Use the same cap _solve_svd applies internally, so the caller and the
        # solver agree instead of clamping twice to different values.
        # Snapshots are mean-centered before the lift, but a delay window of a
        # zero-mean series is not itself zero-mean: the lifted matrix has full
        # row rank in the temporal-lift regime. Cap at the honest matrix bound.
        n_samples_lift, n_space_lift = lifted.shape
        max_rank = max(min(n_samples_lift, n_space_lift), 0)
        k = min(self.n_modes_save, max_rank)
        if k < self.n_modes_save:
            logger.warning("Only %d modes available, requested %d", k, self.n_modes_save)
            self.n_modes_save = k

        self.modes, self.eigenvalues, self.time_coefficients = decomposition.weighted_second_order(
            lifted,
            lifted_metric,
            method="svd",
            n_keep=k,
        )

        # True pre-truncation total: sum of all sigma²/m = ‖data_weighted‖_F² / m.
        self.total_energy = decomposition.weighted_total_energy(lifted, lifted_metric)
        # Solver may return fewer modes than the caller's cap; keep the counter
        # honest before energy logging / save / plot paths read it.
        self._resync_mode_count()

        if self.total_energy > 0:
            self.energy_captured_fraction = float(np.sum(self.eigenvalues) / self.total_energy)
        else:
            self.energy_captured_fraction = 0.0

        end_time = time.time()
        logger.info("ST-POD completed in %.2f seconds.", end_time - start_time)
        logger.info(
            "Computed %d ST-POD modes (%.2f%% of total energy).",
            self.n_modes_save,
            100.0 * self.energy_captured_fraction,
        )

    def _energy_denominator(self) -> tuple[float, str]:
        """Denominator for energy percentages and an optional label suffix.

        Prefer the true pre-truncation total when known. Result files written
        before that attribute was stored fall back to the sum of retained
        eigenvalues; the label suffix says so so titles stay honest.
        """
        total = getattr(self, "total_energy", float("nan"))
        if np.isfinite(total) and total > 0:
            return float(total), ""
        retained = float(np.sum(self.eigenvalues)) if self.eigenvalues.size else 0.0
        return retained, " (retained modes only)"

    def _get_algorithm_metadata(self) -> dict:
        """Describe the current delay-embedded POD contract."""
        lift = getattr(self, "_lift", None) or decomposition.DelayEmbeddingLift(max(self.embedding_dim, 2))
        meta = {
            "lift_kind": lift.kind,
            "stpod_variant": "delay_embedded_pod",
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_lifted_space": True,
            "eigenvalue_normalization": "sigma_squared_over_n_hankel_cols",
            "is_full_spacetime_pod": False,
        }
        if np.isfinite(self.total_energy):
            meta["total_energy"] = float(self.total_energy)
        if np.isfinite(self.energy_captured_fraction):
            meta["energy_captured_fraction"] = float(self.energy_captured_fraction)
        return meta

    def extract_spatial_mode(self, mode_idx: int, delay_idx: int = 0) -> np.ndarray:
        """Extract a single spatial field from a space-time mode.

        Args:
            mode_idx: Mode index (0-based).
            delay_idx: Which delay to extract (0 to d-1).

        Returns:
            Spatial mode field, shape (Nspace,).
        """
        if self.modes.size == 0:
            raise ValueError("No modes available. Run perform_stpod() first.")
        if delay_idx < 0 or delay_idx >= self.embedding_dim:
            raise ValueError(f"delay_idx must be in [0, {self.embedding_dim - 1}]")

        Nspace = self.modes.shape[0] // self.embedding_dim
        start = delay_idx * Nspace
        end = (delay_idx + 1) * Nspace
        return self.modes[start:end, mode_idx]

    def get_mode_as_movie(self, mode_idx: int) -> np.ndarray:
        """Get a mode as a sequence of spatial fields (for animation).

        Args:
            mode_idx: Mode index (0-based).

        Returns:
            Array of shape (d, Nspace) representing temporal evolution.
        """
        if self.modes.size == 0:
            raise ValueError("No modes available. Run perform_stpod() first.")

        Nspace = self.modes.shape[0] // self.embedding_dim
        mode_frames = np.zeros((self.embedding_dim, Nspace))
        for k in range(self.embedding_dim):
            mode_frames[k, :] = self.extract_spatial_mode(mode_idx, k)
        return mode_frames

    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ST-POD datasets and metadata to save."""
        datasets: dict[str, Any] = {
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
        attrs["embedding_dim"] = self.embedding_dim
        attrs["n_modes_saved"] = self.modes.shape[1] if self.modes.ndim == 2 else 0
        attrs["n_snapshots"] = self.data.get("Ns", 0)
        attrs["Nspace"] = self.modes.shape[0] // self.embedding_dim
        return datasets, attrs

    def load_results(self, filename: str | None = None) -> None:
        """Load ST-POD results and restore state."""
        super().load_results(filename=filename)

        from openmodalpy.core.results import read_results

        if not filename:
            filename = (
                f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_d{self.embedding_dim}_{self.analysis_type}.hdf5"
            )
        load_path = os.path.join(self.results_dir, filename)

        res = read_results(load_path)
        # Validate required fields.
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
            raise KeyError(f"{load_path} is not an ST-POD result file: missing {', '.join(missing)}")

        # Restore embedding_dim.
        if "embedding_dim" in res.attrs:
            self.embedding_dim = res.attrs["embedding_dim"]

        # Restore energy tracking metadata.
        self.total_energy = float("nan")
        self.energy_captured_fraction = float("nan")
        if "total_energy" in res.attrs:
            self.total_energy = float(res.attrs["total_energy"])
        if "energy_captured_fraction" in res.attrs:
            self.energy_captured_fraction = float(res.attrs["energy_captured_fraction"])

    def _assign_loaded_results(self, res: AnalysisResults) -> None:
        """Assign loaded results, restore embedding_dim, and reshape W."""
        # Restore embedding_dim before using it.
        if "embedding_dim" in res.attrs:
            self.embedding_dim = res.attrs["embedding_dim"]

        super()._assign_loaded_results(res)

        if res.W is not None:
            n_space = None
            embedding_dim = self.embedding_dim
            if res.modes is not None and res.modes.ndim == 2 and embedding_dim is not None and int(embedding_dim) > 0:
                n_space = int(res.modes.shape[0]) // int(embedding_dim)
            from openmodalpy.core.base import _as_spatial_weight_column

            self.W = _as_spatial_weight_column(res.W, n_space)

        # Cap n_modes_save to actual modes available.
        if self.modes.ndim == 2:
            self.n_modes_save = min(self.n_modes_save, self.modes.shape[1])

    def save_results(self, filename: str | None = None) -> None:
        """Save ST-POD results using embedding-aware filename."""
        if not filename:
            filename = (
                f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_d{self.embedding_dim}_{self.analysis_type}.hdf5"
            )
        super().save_results(filename=filename)

    def plot_eigenvalues(self) -> None:
        os.makedirs(self.figures_dir, exist_ok=True)
        """Plot the ST-POD eigenvalue spectrum."""
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_stpod() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            mode_indices = np.arange(1, len(self.eigenvalues) + 1)
            denom, label_suffix = self._energy_denominator()
            normalized = (self.eigenvalues / denom * 100) if denom > 0 else np.zeros_like(self.eigenvalues)

            ax.plot(mode_indices, normalized, "o-", linewidth=2, markersize=6)

            n_annotate = min(5, len(mode_indices))
            for idx in range(n_annotate):
                ax.text(mode_indices[idx], normalized[idx], f" {idx + 1}", fontsize=7, va="bottom")

            ax.set_yscale("log")
            ax.set_xlabel("Mode Number")
            ax.set_ylabel(f"Normalized Eigenvalue (%){label_suffix}")
            ax.set_title(f"ST-POD Eigenvalue Spectrum (d={self.embedding_dim})")
            ax.grid(True, which="both", ls="--")

            plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_eigenvalues.png")
            plt.savefig(plot_filename, dpi=FIG_DPI, bbox_inches="tight")
            logger.info("Saving figure %s", plot_filename)
        finally:
            plt.close(fig)

    def plot_modes(
        self,
        plot_n_modes: int = 4,
        delay_idx: int = 0,
        show_cylinder: bool = False,
    ) -> None:
        """Plot spatial modes at a specific delay index.

        Args:
            plot_n_modes: Number of modes to plot.
            delay_idx: Which time delay to show (0 to d-1).
            show_cylinder: If True, mask cylinder at origin.
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_stpod() first.")
            return

        Nspace = self.modes.shape[0] // self.embedding_dim
        if resolve_volume_layout(self.data, Nspace) is not None:
            self.plot_modes_3d_slices(plot_n_modes=plot_n_modes, delay_idx=delay_idx)
            return
        Nx = self.data.get("Nx", int(np.sqrt(Nspace)))
        Ny = self.data.get("Ny", int(np.sqrt(Nspace)))
        is_2d = (Nspace == Nx * Ny) and (Nx > 1 and Ny > 1)

        if not is_2d:
            logger.warning("plot_modes currently supports 2-D fields only.")
            return

        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)

        n_modes = min(plot_n_modes, self.modes.shape[1], self.n_modes_save)
        ncols = min(n_modes, 2)
        nrows = int(np.ceil(n_modes / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols * fig_aspect, 4 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
        else:
            x_mesh, y_mesh = x_coords, y_coords

        denom, label_suffix = self._energy_denominator()

        for k in range(n_modes):
            row, col = divmod(k, ncols)
            ax = axes[row, col]

            mode_spatial = self.extract_spatial_mode(k, delay_idx)
            mode_2d = mode_spatial.reshape((Ny, Nx))

            mode_plot: Any
            if show_cylinder:
                dist = np.sqrt(x_mesh**2 + y_mesh**2)
                mask = dist <= 0.5
                mode_plot = np.ma.array(mode_2d, mask=mask)
            else:
                mode_plot = mode_2d

            from openmodalpy.core.base import get_robust_clim

            vmin, vmax = get_robust_clim(mode_plot, method="percentile")
            levels = np.linspace(vmin, vmax, 21)

            cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")

            if show_cylinder:
                cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cyl)

            ax.set_aspect("equal", "box")
            ax.set_xlim(np.min(x_coords), np.max(x_coords))
            ax.set_ylim(np.min(y_coords), np.max(y_coords))
            ax.set_xlabel(r"$x/D$")
            ax.set_ylabel(r"$y/D$")
            ax.grid(True, linestyle="--", alpha=0.3)

            if denom > 0:
                energy_pct = 100.0 * self.eigenvalues[k] / denom
                cum_pct = 100.0 * np.sum(self.eigenvalues[: k + 1]) / denom
            else:
                energy_pct = 0.0
                cum_pct = 0.0
            ax.set_title(
                f"Mode {k + 1} (τ={delay_idx})\nE={energy_pct:.2f}% Cum={cum_pct:.2f}%{label_suffix}",
                fontsize=9,
            )

            fig.colorbar(cf, ax=ax, shrink=0.8)

        # Hide empty subplots
        for idx in range(n_modes, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        fig.suptitle(f"ST-POD Modes (d={self.embedding_dim}, delay={delay_idx})", fontsize=12)
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_modes_delay{delay_idx}.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def plot_modes_3d_slices(self, plot_n_modes: int = 4, delay_idx: int = 0) -> None:
        """Plot orthogonal 3D slices for ST-POD modes at one delay index."""
        self._plot_modes_3d("slices", plot_n_modes=plot_n_modes, delay_idx=delay_idx)

    def plot_modes_3d_isometric(self, plot_n_modes: int = 4, delay_idx: int = 0) -> None:
        """Plot 3D isosurfaces for ST-POD modes at one delay index."""
        self._plot_modes_3d("isometric", plot_n_modes=plot_n_modes, delay_idx=delay_idx)

    def _plot_modes_3d(self, kind: str, plot_n_modes: int = 4, delay_idx: int = 0) -> None:
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_stpod() first.")
            return
        nspace = self.modes.shape[0] // self.embedding_dim
        layout = resolve_volume_layout(self.data, nspace)
        if layout is None:
            logger.warning("plot_modes_3d_%s requires volumetric data.", kind)
            return
        if not 0 <= delay_idx < self.embedding_dim:
            raise ValueError(f"delay_idx={delay_idx} outside [0, {self.embedding_dim - 1}]")
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        n_modes = min(plot_n_modes, self.modes.shape[1], self.n_modes_save)
        denom, label_suffix = self._energy_denominator()
        items = []
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.extract_spatial_mode(mode_idx, delay_idx), self.data)
            if denom > 0:
                energy_pct = 100.0 * self.eigenvalues[mode_idx] / denom
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx} | E={energy_pct:.2f}%{label_suffix}"
            else:
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx}"
            output_path = os.path.join(
                self.figures_dir, f"{self.data_root}_stpod_mode_{mode_idx + 1}_delay{delay_idx}_{kind}.png"
            )
            items.append(
                {
                    "mode_3d": mode_3d,
                    "output_path": output_path,
                    "title_prefix": title,
                    "scalar_name": "stpod_mode",
                }
            )
        plot_modes_3d(kind, items, x_coords, y_coords, z_coords, data=self.data)

    def plot_spacetime_mode(
        self,
        mode_idx: int = 0,
        n_delays_show: Optional[int] = None,
        show_cylinder: bool = False,
    ) -> None:
        """Plot a space-time mode showing its temporal evolution.

        Args:
            mode_idx: Which mode to visualize.
            n_delays_show: Number of delay frames to show. Default: min(d, 6).
            show_cylinder: If True, mask cylinder at origin.
        """
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_stpod() first.")
            return

        if n_delays_show is None:
            n_delays_show = min(self.embedding_dim, 6)

        Nspace = self.modes.shape[0] // self.embedding_dim
        Nx = self.data.get("Nx", int(np.sqrt(Nspace)))
        Ny = self.data.get("Ny", int(np.sqrt(Nspace)))
        is_2d = (Nspace == Nx * Ny) and (Nx > 1 and Ny > 1)

        if not is_2d:
            logger.warning("plot_spacetime_mode currently supports 2-D fields only.")
            return

        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)

        # Select delays to show (evenly spaced)
        if n_delays_show < self.embedding_dim:
            delay_indices = np.linspace(0, self.embedding_dim - 1, n_delays_show, dtype=int)
        else:
            delay_indices = np.arange(self.embedding_dim)

        ncols = min(len(delay_indices), 3)
        nrows = int(np.ceil(len(delay_indices) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols * fig_aspect, 4 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
        else:
            x_mesh, y_mesh = x_coords, y_coords

        # Get global vmax for consistent colorscale
        mode_frames = self.get_mode_as_movie(mode_idx)
        global_vmax = np.max(np.abs(mode_frames))
        levels = np.linspace(-global_vmax, global_vmax, 21)

        for i, delay_idx in enumerate(delay_indices):
            row, col = divmod(i, ncols)
            ax = axes[row, col]

            mode_spatial = self.extract_spatial_mode(mode_idx, delay_idx)
            mode_2d = mode_spatial.reshape((Ny, Nx))

            mode_plot: Any
            if show_cylinder:
                dist = np.sqrt(x_mesh**2 + y_mesh**2)
                mask = dist <= 0.5
                mode_plot = np.ma.array(mode_2d, mask=mask)
            else:
                mode_plot = mode_2d

            _ = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")

            if show_cylinder:
                cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cyl)

            ax.set_aspect("equal", "box")
            ax.set_xlim(np.min(x_coords), np.max(x_coords))
            ax.set_ylim(np.min(y_coords), np.max(y_coords))
            ax.set_xlabel(r"$x/D$")
            ax.set_ylabel(r"$y/D$")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(f"τ = {delay_idx}", fontsize=9)

        # Hide empty subplots
        for idx in range(len(delay_indices), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        denom, label_suffix = self._energy_denominator()
        energy_pct = 100.0 * self.eigenvalues[mode_idx] / denom if denom > 0 else 0.0
        fig.suptitle(
            f"ST-POD Mode {mode_idx + 1} Evolution (E={energy_pct:.2f}%{label_suffix})",
            fontsize=12,
        )

        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_spacetime_mode{mode_idx + 1}.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def plot_time_coefficients(
        self,
        n_coeffs_to_plot: int = 2,
        n_snapshots_plot: Optional[int] = None,
        L: float = 1.0,
        U: float = 1.0,
    ) -> None:
        """Plot temporal coefficients and their spectra.

        Args:
            n_coeffs_to_plot: Number of coefficients to plot.
            n_snapshots_plot: Number of time points to show.
            L: Characteristic length for Strouhal number.
            U: Characteristic velocity for Strouhal number.
        """
        if self.time_coefficients.size == 0:
            logger.warning("No time coefficients to plot. Run perform_stpod() first.")
            return

        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1])
        m = self.time_coefficients.shape[0]  # Number of Hankel columns

        if n_snapshots_plot is None or n_snapshots_plot > m:
            n_snapshots_plot = m

        time_vector, xlabel = self._time_axis(n_snapshots_plot)
        fs = self._require_fs()

        fig, axes = plt.subplots(n_coeffs_to_plot, 2, figsize=(12, 3 * n_coeffs_to_plot))
        if n_coeffs_to_plot == 1:
            axes = axes.reshape(1, 2)

        for i in range(n_coeffs_to_plot):
            coeff = self.time_coefficients[:n_snapshots_plot, i]

            # Time series
            axes[i, 0].plot(time_vector, coeff, ls="-", lw=0.8, marker="o", markersize=1)
            axes[i, 0].set_xlabel(xlabel)
            axes[i, 0].set_ylabel(f"a_{i + 1}(t)")
            axes[i, 0].set_title(f"ST-POD Coefficient {i + 1}")
            axes[i, 0].grid(True, linestyle=":")
            axes[i, 0].set_xlim(time_vector.min(), time_vector.max())

            # Periodogram
            freqs, psd = periodogram_rfft(coeff, fs)
            peak_freqs, peak_psd = find_peaks(freqs, psd)

            if L is not None and U is not None:
                freqs_st = freqs * L / U
                peak_freqs_st = peak_freqs * L / U if peak_freqs.size > 0 else peak_freqs
            else:
                freqs_st = freqs
                peak_freqs_st = peak_freqs

            axes[i, 1].semilogy(freqs_st, psd)
            if peak_freqs_st.size > 0:
                axes[i, 1].plot(peak_freqs_st, peak_psd, "o", markersize=4)
                for pf, pv in zip(peak_freqs_st[:3], peak_psd[:3]):
                    axes[i, 1].text(pf, pv, f" {pf:.2f}", fontsize=8, ha="left", va="bottom")

            axes[i, 1].set_xscale("log")
            # Fit y-limits: show 6 decades below peak
            psd_pos = psd[psd > 0]
            if len(psd_pos) > 0:
                ymax = psd_pos.max() * 3
                axes[i, 1].set_ylim(ymax * 1e-6, ymax)
            axes[i, 1].set_xlabel("Strouhal Number (St)")
            axes[i, 1].set_ylabel("PSD")
            axes[i, 1].set_title(f"Periodogram Mode {i + 1}")
            axes[i, 1].grid(True, linestyle=":")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_time_coeffs.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def plot_cumulative_energy(self) -> None:
        """Plot cumulative energy captured by ST-POD modes."""
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_stpod() first.")
            return

        denom, label_suffix = self._energy_denominator()
        if denom > 0:
            cumulative = np.cumsum(self.eigenvalues) / denom * 100
        else:
            cumulative = np.zeros_like(self.eigenvalues)
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(mode_indices, cumulative, "o-", linewidth=2, markersize=6)

        for idx, (x, y) in enumerate(zip(mode_indices, cumulative)):
            ax.text(float(x), float(y), f" {idx + 1}", fontsize=7, va="bottom")

        ax.set_xlabel("Number of Modes")
        ax.set_ylabel(f"Cumulative Energy (%){label_suffix}")
        ax.set_title(f"Cumulative Energy of ST-POD Modes (d={self.embedding_dim})")
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(0, 105)

        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_cumulative_energy.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Saving figure %s", plot_filename)

    def check_mode_orthogonality(self, tolerance: float = 1e-9) -> bool:
        """Check orthonormality of modes with respect to extended weights.

        The weighted inner product uses W extended to d copies for the
        d*Nspace-dimensional mode vectors.

        Args:
            tolerance: Maximum allowed deviation from identity.

        Returns:
            True if modes are orthonormal within tolerance.
        """
        if self.modes.size == 0 or self.W.size == 0:
            logger.warning("Modes or weights not available.")
            return False

        logger.info("Checking ST-POD mode orthonormality...")
        Nspace = self.modes.shape[0] // self.embedding_dim
        n_modes = self.modes.shape[1]

        weight_vector = self._get_weight_vector(Nspace)
        W_extended = np.tile(weight_vector, self.embedding_dim)

        # Element-wise multiply instead of forming dense (d*Nspace)² diagonal matrix
        gram = self.modes.T @ (W_extended[:, np.newaxis] * self.modes)

        diag_dev = np.max(np.abs(np.diag(gram) - 1.0))
        off_diag_mask = ~np.eye(n_modes, dtype=bool)
        off_diag_max = np.max(np.abs(gram[off_diag_mask])) if n_modes > 1 else 0.0

        is_orthonormal = (diag_dev < tolerance) and (off_diag_max < tolerance)

        logger.info("Max diagonal deviation from 1: %.2e", diag_dev)
        logger.info("Max off-diagonal value: %.2e", off_diag_max)
        logger.info("Orthonormal: %s", "Yes" if is_orthonormal else "No")

        return is_orthonormal

    _perform_name = "perform_stpod"

    def _plot_run(self, run_id: str | None = None) -> None:
        """Default figures after run_analysis — the CLI st-pod set.

        The CLI passed delay_idx=0 to the volumetric hooks and plotted the two
        leading spacetime modes for 2-D data; both preserved here.
        """
        self.plot_eigenvalues()
        if not self._maybe_plot_volumetric_modes(
            plot_n_modes=min(2, self.n_modes_save),
            slices_kwargs={"delay_idx": 0},
            iso_kwargs={"delay_idx": 0},
        ):
            self.plot_modes(plot_n_modes=min(2, self.n_modes_save))
        self.plot_time_coefficients(n_coeffs_to_plot=min(2, self.n_modes_save))
        self.plot_cumulative_energy()
