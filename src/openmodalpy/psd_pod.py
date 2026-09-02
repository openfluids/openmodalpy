#!/usr/bin/env python3
"""Power-Spectral-Density Proper Orthogonal Decomposition (PSD-POD).

PSD-POD pools all Welch-block Fourier realizations across frequencies into one
ensemble and solves a single weighted second-order eigenproblem on that
ensemble. It shares the FFT-block preprocessing with SPOD but solves one global
problem rather than one per frequency bin.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fftkit import rfftfreq

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import (
    BaseAnalyzer,
    add_inset_colorbar,
    get_fig_aspect_ratio,
    get_robust_clim,
    plot_isometric_slices_3d,
    plot_orthogonal_slices_3d,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    WINDOW_NORM,
    WINDOW_TYPE,
)
from openmodalpy.core.results import AnalysisResults

logger = logging.getLogger(__name__)


class PSDPODAnalyzer(BaseAnalyzer):
    """POD on the ensemble of blockwise Fourier realizations.

    Lifecycle: ``load_and_preprocess`` → ``perform_psd_pod`` →
    ``save_results``. ``perform_psd_pod`` forms the FFT blocks itself, via
    ``compute_fft_blocks()``, on first use. ``run_analysis`` runs the full
    sequence.

    Key attributes after a successful run:
        modes: spatial modes, shape (Nspace, n_modes_save), complex
        eigenvalues: energy of each mode, shape (n_modes_save,)
        time_coefficients: projections of the Fourier ensemble, shape
            (n_fourier_realizations, n_modes_save)
        freq, St: frequency and Strouhal axes from the Welch blocks
    """

    _METHOD_NAME = "psd_pod"

    def __init__(
        self,
        file_path: str | None = None,
        *,
        results_dir: str = RESULTS_DIR,
        figures_dir: str = FIGURES_DIR,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        nfft: int = 128,
        overlap: float = 0.5,
        n_modes_save: int = 10,
        blockwise_mean: bool = False,
        window_norm: str = WINDOW_NORM,
        window_type: str = WINDOW_TYPE,
        characteristic_length: float | None = None,
        characteristic_velocity: float | None = None,
        spatial_weights: np.ndarray | None = None,
        data: dict[str, Any] | None = None,
    ):
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            spatial_weights=spatial_weights,
            data=data,
        )
        # PSD-POD is one of the three methods that form FFT blocks, so it
        # keeps its own nfft/overlap rather than the base's dummy stamp.
        self.nfft = nfft
        self.overlap = overlap
        self.novlap = int(overlap * nfft)
        if not (0 <= self.overlap < 1):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")
        if self.nfft <= 0:
            raise ValueError("NFFT must be positive.")

        self.n_modes_save = int(n_modes_save)
        self.blockwise_mean = bool(blockwise_mean)
        self.normvar = False
        self.window_norm = window_norm
        self.window_type = window_type
        self.L = characteristic_length
        self.U = characteristic_velocity

        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.freq = np.array([])
        self.St = np.array([])
        self.n_fourier_realizations = 0
        self.results_path: Optional[str] = None
        self.analysis_type = "psd_pod"

    def load_and_preprocess(self) -> None:
        """Load data, spatial weights, and the Welch frequency / Strouhal axes."""
        super().load_and_preprocess()
        self.L = self.L if self.L is not None else self.data.get("D", 1.0)
        self.U = self.U if self.U is not None else self.data.get("U0", 1.0)
        # Same axis construction as the previous command-path SPOD loader so
        # freq / st stay bit-identical with the pre-move baseline.
        self.freq = rfftfreq(self.nfft, d=self.data["dt"])
        self.St = self.freq * self.L / self.U

    def _weight_vector(self, n_space: int) -> np.ndarray:
        """Flatten self.W to a length-n_space weight vector."""
        weights = np.asarray(self.W).reshape(-1)
        if weights.size == n_space:
            return weights
        if self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1]:
            return np.diag(self.W)
        raise ValueError("PSD-POD could not flatten the spatial weights to match the Fourier ensemble.")

    def perform_psd_pod(self) -> None:
        """Solve the pooled Fourier-ensemble eigenproblem via the shared seam.

        Mode recovery in ``weighted_second_order`` (complex path) uses the
        unweighted ensemble for the spatial modes:

            Phi = E^H V / sqrt(lambda * N)

        That is algebraically correct: the ``sqrt(w)`` factors cancel between
        the weighted ensemble that formed the kernel (E_w) and the unweighting
        division that recovers physical modes. It reads like a missing weight;
        do not "fix" it by inserting an extra ``/ sqrt(w)`` or by swapping
        ``E`` for ``E_w`` in the recovery formula.
        """
        # Form the FFT blocks on first use, so callers never need a separate
        # compute_fft_blocks() step between load_and_preprocess() and this call.
        if self.qhat is None or self.qhat.size == 0:
            self.compute_fft_blocks()
        qhat = self.qhat
        if qhat is None or qhat.size == 0:
            raise ValueError("FFT blocks were not computed; PSD-POD cannot proceed.")

        # qhat: (n_freq, Nspace, n_blocks) → ensemble: (n_freq * n_blocks, Nspace)
        ensemble = np.transpose(qhat, (0, 2, 1)).reshape(-1, qhat.shape[1])
        n_realizations = ensemble.shape[0]
        if n_realizations < 2:
            raise ValueError("PSD-POD needs at least two Fourier realizations.")

        weights = self._weight_vector(ensemble.shape[1])
        modes, eigenvalues, time_coefficients = decomposition.weighted_second_order(
            ensemble,
            decomposition.SpatialMetric(weights),
            method="eigh",
            n_keep=self.n_modes_save,
        )

        self.modes = modes
        self.eigenvalues = np.real(eigenvalues)
        self.time_coefficients = time_coefficients
        self.n_fourier_realizations = n_realizations
        self._resync_mode_count()

    def _get_algorithm_metadata(self) -> dict:
        """Describe the PSD-POD contract recorded in result-file attributes."""
        return {
            "lift_kind": "flattened_block_fourier_realizations",
            # blocksfft removes a mean on every path, so this is unconditionally True.
            # blockwise_mean chooses which mean (global or per-block), never whether.
            # It is recorded separately so the two facts stay distinguishable.
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_second_order_operator": True,
            "spectral_estimator": "welch_block_average",
            "n_fourier_realizations": int(self.n_fourier_realizations),
        }

    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return PSD-POD datasets and metadata to save."""
        datasets: dict[str, Any] = {
            "eigenvalues": np.asarray(self.eigenvalues),
            "modes": self.modes,
            "time_coefficients": self.time_coefficients,
            "freq": self.freq,
            "st": self.St,
        }
        if self.W.size > 0:
            datasets["W"] = self.W
        attrs = self._get_metadata()
        attrs["analysis_type"] = "psd_pod"
        return datasets, attrs

    def save_results(self, filename: str | None = None) -> None:
        """Save PSD-POD results and record the path the CLI reads back."""
        super().save_results(filename=filename)
        self.results_path = os.path.join(self.results_dir, filename or self._result_filename())

    def _required_result_fields(self) -> tuple[str, ...]:
        """PSD-POD requires modes and eigenvalues."""
        return ("modes", "eigenvalues")

    def _assign_loaded_results(self, res: AnalysisResults) -> None:
        """Assign loaded results and reshape W to column form."""
        super()._assign_loaded_results(res)

        if res.W is not None:
            from openmodalpy.core.base import _as_spatial_weight_column

            n_space = int(res.modes.shape[0]) if res.modes is not None and res.modes.ndim == 2 else None
            self.W = _as_spatial_weight_column(res.W, n_space)

        if res.freq is not None:
            self.freq = res.freq
        if res.st is not None:
            self.St = res.st

    def _figure_prefix(self, run_id: str | None) -> str:
        """Figure-name stem: the CLI run id when given, the dataset root otherwise."""
        return run_id if run_id is not None else self.data_root

    def plot_eigenvalues(self, run_id: str | None = None) -> list[Path]:
        os.makedirs(self.figures_dir, exist_ok=True)
        """Plot the pooled-ensemble eigenvalue spectrum on a log axis."""
        saved: list[Path] = []
        eigenvalues = np.asarray(self.eigenvalues)
        if eigenvalues.size == 0:
            return saved

        fig, ax = plt.subplots()
        ax.semilogy(np.arange(1, len(eigenvalues) + 1), np.maximum(eigenvalues.real, 1e-16), marker="o")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("PSD-POD eigenvalue")
        ax.set_title("PSD-POD eigenvalues")
        path = Path(self.figures_dir) / f"{self._figure_prefix(run_id)}_eigenvalues.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        return saved

    def plot_cumulative_energy(self, run_id: str | None = None) -> list[Path]:
        """Plot cumulative energy versus mode index."""
        saved: list[Path] = []
        eigenvalues = np.asarray(self.eigenvalues)
        if eigenvalues.size == 0:
            return saved

        cumulative = np.cumsum(eigenvalues.real)
        total = cumulative[-1] if cumulative.size else 0.0
        if total <= 0:
            return saved
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(eigenvalues) + 1), cumulative / total * 100.0, marker="o")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Cumulative energy [%]")
        ax.set_title("PSD-POD cumulative energy")
        path = Path(self.figures_dir) / f"{self._figure_prefix(run_id)}_cumulative_energy.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        return saved

    def plot_modes(self, *, plot_n_modes: int = 2, run_id: str | None = None) -> list[Path]:
        """Plot the leading PSD-POD spatial modes with the same 2D styling as other analyzers."""
        saved: list[Path] = []
        modes = np.asarray(self.modes)
        eigenvalues = np.asarray(self.eigenvalues)
        data = self.data
        figures_dir = Path(self.figures_dir)
        run_id = self._figure_prefix(run_id)
        if modes.size == 0:
            return saved

        nx = int(data.get("Nx", 0))
        ny = int(data.get("Ny", 0))
        if nx <= 1 or ny <= 1 or modes.shape[0] != nx * ny:
            return saved

        x_coords = data.get("x", np.arange(nx))
        y_coords = data.get("y", np.arange(ny))
        if np.ndim(x_coords) == 1 and np.ndim(y_coords) == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
        else:
            x_mesh, y_mesh = x_coords, y_coords

        total_energy = float(np.sum(np.real(eigenvalues)))
        fig_aspect = get_fig_aspect_ratio(data)
        n_modes = min(plot_n_modes, modes.shape[1])
        ncols = n_modes
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols * fig_aspect, 4), squeeze=False)
        axes = axes.ravel()
        var_name = data.get("metadata", {}).get("var_name", "q")

        for idx in range(n_modes):
            ax = axes[idx]
            mode = np.asarray(modes[:, idx].real).reshape(ny, nx)
            vmin, vmax = get_robust_clim(mode, method="percentile")
            levels = np.linspace(vmin, vmax, 21)
            cf = ax.contourf(x_mesh, y_mesh, mode, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(x_mesh, y_mesh, mode, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
            style_spatial_axes(ax, data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
            energy_pct = 100.0 * float(np.real(eigenvalues[idx])) / total_energy if total_energy > 0 else 0.0
            ax.set_title(f"PSD-POD Mode {idx + 1} [{var_name}] | E={energy_pct:.2f}%")
            add_inset_colorbar(
                fig,
                ax,
                cf,
                data,
                ticks=[vmin, 0, vmax],
                ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
            )

        with plt.rc_context():
            fig.tight_layout()
        path = figures_dir / f"{run_id}_modes_1_to_{n_modes}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        return saved

    def plot_modes_3d(self, *, plot_n_modes: int = 2, run_id: str | None = None) -> list[Path]:
        """Plot the leading PSD-POD modes with the shared 3D helpers."""
        saved: list[Path] = []
        modes = np.asarray(self.modes)
        eigenvalues = np.asarray(self.eigenvalues)
        data = self.data
        figures_dir = Path(self.figures_dir)
        run_id = self._figure_prefix(run_id)
        if modes.size == 0 or resolve_volume_layout(data, modes.shape[0]) is None:
            return saved

        x_coords = data.get("x")
        y_coords = data.get("y")
        z_coords = data.get("z")
        total_energy = float(np.sum(np.real(eigenvalues))) if eigenvalues.size else 0.0
        n_modes = min(plot_n_modes, modes.shape[1])
        for idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(np.asarray(modes[:, idx]).real, data)
            energy_pct = 100.0 * float(np.real(eigenvalues[idx])) / total_energy if total_energy > 0 else 0.0
            title = f"PSD-POD Mode {idx + 1} | E={energy_pct:.2f}%"
            slice_path = figures_dir / f"{run_id}_mode_{idx + 1}_slices.png"
            iso_path = figures_dir / f"{run_id}_mode_{idx + 1}_isometric.png"
            plot_orthogonal_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=str(slice_path),
                title_prefix=title,
                data=data,
                scalar_name="psd_pod_mode",
            )
            plot_isometric_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=str(iso_path),
                title_prefix=title,
                data=data,
                scalar_name="psd_pod_mode",
            )
            saved.extend([slice_path, iso_path])
        return saved

    _perform_name = "perform_psd_pod"
    _needs_fft_blocks = True

    def _plot_run(self, run_id: str | None = None) -> None:
        """Default figures after run_analysis — the CLI psd-pod set."""
        modes = np.asarray(self.modes)
        self.plot_eigenvalues(run_id=run_id)
        self.plot_cumulative_energy(run_id=run_id)
        if resolve_volume_layout(self.data, modes.shape[0]) is not None:
            self.plot_modes_3d(plot_n_modes=min(2, self.n_modes_save), run_id=run_id)
        else:
            self.plot_modes(plot_n_modes=min(2, self.n_modes_save), run_id=run_id)
