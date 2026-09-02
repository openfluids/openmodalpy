"""Spectral Proper Orthogonal Decomposition.

SPOD splits a record into overlapping Welch blocks, Fourier transforms each
block, and then solves one energy eigenproblem per frequency over the block
ensemble. At frequency f the cross-spectral density is estimated from the
blocks and its eigenvectors are the SPOD modes:

    (1/n_blocks) X_f^H W X_f  psi = lambda psi,   phi = X_f psi / sqrt(lambda)

where ``X_f`` holds one Fourier coefficient column per block and ``W`` is the
spatial integration metric. Towne, Schmidt & Colonius (2018), JFM 847.

That eigenproblem lives in ``openmodalpy.core.decomposition``, in
``spod_single_frequency``. This module builds the blocks, runs the frequency
loop, and stores, saves and plots the result.
"""

from __future__ import annotations

# Standard library imports
import logging
import os
import time
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Optional

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from fftkit import rfftfreq
from numpy.typing import ArrayLike
from tqdm import tqdm

from openmodalpy.core.base import (
    BaseAnalyzer,
    _write_qhat_stamp,
    add_inset_colorbar,
    format_mode_title,
    get_fig_aspect_ratio,
    plot_modes_3d,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
    validate_nfft_overlap,
)
from openmodalpy.core.config import (
    CMAP_DIV,
    FIG_DPI,
    FIG_FORMAT,
    FIGURES_DIR_SPOD,
    RESULTS_DIR_SPOD,
    WINDOW_NORM,
    WINDOW_TYPE,
)
from openmodalpy.core.decomposition import spod_single_frequency
from openmodalpy.core.results import AnalysisResults

logger = logging.getLogger(__name__)


class SPODAnalyzer(BaseAnalyzer):
    """
    Spectral Proper Orthogonal Decomposition (SPOD) analyzer.

    This class implements the SPOD method to decompose a data sequence into
    modes that are optimal in terms of energy for each frequency. It is
    particularly useful for analyzing spatio-temporal data from fluid dynamics
    simulations or experiments.

    ## Note: we need to test different inner products SPOD-u (using TKE) and for SPOD-p (known as extended SPOD which includes the acoustic radiation pressure)

    The SPOD algorithm involves:
    1. Computing the cross-spectral density (CSD) matrix from blocked FFTs of the data.
    2. Performing an eigenvalue decomposition of the CSD matrix for each frequency.
    The eigenvalues represent the energy of each mode at a given frequency, and
    the eigenvectors are the SPOD modes.

    Key Attributes:
        eigenvalues (np.ndarray): SPOD eigenvalues (energy) for each mode and frequency.
                                  Shape: (n_freq_bins, n_blocks). Always the full
                                  block count, even when ``n_modes_save`` cuts the modes.
        modes (np.ndarray): SPOD spatial modes. Shape: (n_freq_bins, n_spatial_points, n_modes_kept),
                            where n_modes_kept is ``n_modes_save`` or n_blocks when it is not set.
        time_coefficients (np.ndarray): SPOD temporal coefficients (reconstructed from modes and qhat).
                                        Shape: (n_freq_bins, n_blocks, n_modes_kept). The block axis
                                        comes before the mode axis, which the eigenproblem sets.
        freq (np.ndarray): Array of frequencies corresponding to FFT bins.
        St (np.ndarray): Array of Strouhal numbers corresponding to `freq`.
        dst (float): Strouhal number step, used for integral weights in the eigenproblem.
        qhat_cached (bool): Flag indicating if FFT blocks (q_hat) were loaded from cache.
        W (np.ndarray): Spatial weighting matrix (diagonal).
        fs (float): Sampling frequency of the data.
        L (float): Characteristic length for Strouhal number calculation.
        U (float): Characteristic velocity for Strouhal number calculation.

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading, preprocessing, and FFT computation.
                      SPOD makes one mode per Welch block at each frequency, so
                      the block count is the ceiling on ``n_modes_save``. Leave
                      ``n_modes_save`` unset to keep every block.
    """

    ############################################################
    # Initialization and Core Parameters                       #
    ############################################################
    _METHOD_NAME = "spod"

    def __init__(
        self,
        file_path: str | None = None,
        *,
        nfft: int = 128,
        overlap: float = 0.5,
        n_modes_save: int | None = None,
        results_dir: str = RESULTS_DIR_SPOD,
        figures_dir: str = FIGURES_DIR_SPOD,
        blockwise_mean: bool = False,
        normvar: bool = False,
        window_norm: str = WINDOW_NORM,
        window_type: str = WINDOW_TYPE,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        characteristic_length: float | None = None,
        characteristic_velocity: float | None = None,
        spatial_weights: ArrayLike | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the SPODAnalyzer instance.

        Args:
            file_path (str | None): Path to the data file (e.g., .mat, .h5).
                Optional when ``data`` carries the loaded dataset instead.
            nfft (int, optional): Number of points per FFT block. Defaults to 128.
            overlap (float, optional): Overlap fraction between FFT blocks (0 to <1).
                                     Defaults to 0.5 (50% overlap).
            n_modes_save (int | None, optional): Number of leading modes to keep at
                each frequency. SPOD computes one mode per Welch block, so the
                default None keeps every block and ``modes`` is
                ``(n_freq, n_space, n_blocks)``. Set this to cut the size of
                ``modes`` and ``time_coefficients`` on a long record. A value
                above the block count keeps every block and reports a
                ``RuntimeWarning``. Eigenvalues always keep every block, because
                the spectrum plot draws one line per block.
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_SPOD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_SPOD` from `configs.py`.
            blockwise_mean (bool, optional): If True, subtracts the mean of each block before FFT.
                                           If False, subtracts the global mean. Defaults to False.
            normvar (bool, optional): If True, divide each FFT block pointwise
                in space by its variance (unbiased, ``ddof=1``), matching
                ``spod_matlab`` (``opts.normvar``) and PySPOD
                (``normalize_data``). This does **not** produce unit variance
                and is therefore scale-dependent: scaling the input by ``c``
                scales the normalized block by ``1/c``. Values below
                ``4*eps`` are clamped to 1. Implementation option, not a step
                in Towne, Schmidt & Colonius (2018). Defaults to False.
            window_norm (str, optional): Normalization type for the window function ('amplitude' or 'power').
                                         Defaults to `WINDOW_NORM` from `configs.py`.
            window_type (str, optional): Type of window function to use (e.g., 'hamming', 'hanning', 'sine').
                                         Defaults to `WINDOW_TYPE` from `configs.py`.
            data_loader (callable, optional): Custom function to load data from `file_path`.
                                              If None, `BaseAnalyzer` attempts to auto-detect.
                                              Defaults to None.
            spatial_weight_type (str | None, optional): Type of spatial weights to apply
                (None → 'uniform', or 'uniform', 'polar', 'prescribed'). Defaults to None.
            spatial_weights: Optional array of spatial integration weights. When given,
                the type becomes 'prescribed'.
            data (dict | None): Already-loaded dataset following the data contract
                (see DOC.md). Given instead of ``file_path``.
        """
        super().__init__(
            file_path=file_path,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            spatial_weights=spatial_weights,
            data=data,
        )
        # SPOD is one of the three methods that form FFT blocks, so it keeps
        # its own nfft/overlap rather than the base's dummy stamp.
        self.nfft = nfft
        self.overlap = overlap
        self.novlap = int(overlap * nfft)
        # The request may be None ("keep every block"), which the shared
        # `n_modes_save` int cannot hold. perform_spod resolves it against the
        # block count and publishes the effective number, as ST-POD does.
        self._n_modes_save_request = n_modes_save
        if n_modes_save is not None:
            self.n_modes_save = n_modes_save

        self._validate_inputs()
        # SPOD specific attributes
        self.blockwise_mean = blockwise_mean
        self.normvar = normvar
        self.window_norm = window_norm
        self.window_type = window_type

        self.L = characteristic_length  # resolved in load_and_preprocess
        self.U = characteristic_velocity  # resolved in load_and_preprocess
        self.St_normalization_factor = 1.0  # For Strouhal number calculation
        self.analysis_type = "spod"

        self.eigenvalues = np.array([])  # SPOD eigenvalues (L_d)
        self.modes = np.array([])  # SPOD spatial modes (Phi)
        self.time_coefficients = np.array([])  # SPOD temporal coefficients (Psi)
        self.freq = np.array([])  # Frequencies (from rfft)
        self.St = np.array([])  # Strouhal numbers
        self.dst = 0.0  # Strouhal step (integral weight in the eigenproblem)
        self.qhat_cached = False  # Flag whether FFT blocks were loaded from cache

    def _validate_inputs(self) -> None:
        """
        Validates SPOD-specific input parameters.

        Ensures that `overlap` is within the range [0, 1) and `nfft` is positive.

        Raises:
            ValueError: If `overlap` is not in [0, 1) or `nfft` is not positive,
                or if `n_modes_save` is below one.
        """
        validate_nfft_overlap(self.nfft, self.overlap)
        # The block count needs the record length, which arrives later, so the
        # upper bound is checked in perform_spod. Only the sign is known here.
        request = self._n_modes_save_request
        if request is not None and request < 1:
            raise ValueError(f"n_modes_save must be >= 1, got {request}")

    def _modes_to_keep(self) -> int:
        """Number of leading modes to keep per frequency, against the block count.

        SPOD makes one mode per Welch block, so the block count is the ceiling.
        A request above it keeps every block and reports both numbers. POD caps
        the same request without a message; that silence is what this avoids.
        """
        request = self._n_modes_save_request
        if request is None:
            self.n_modes_save = int(self.nblocks)
            return self.n_modes_save
        if request > self.nblocks:
            # RuntimeWarning, not UserWarning: the ceiling comes from the record
            # length and the block settings, so asking for more is a property of
            # the data, not misuse of the API. Same category as the DMD rank
            # notice.
            warnings.warn(
                f"n_modes_save={request} exceeds the {self.nblocks} "
                f"Welch blocks; keeping all {self.nblocks} modes per frequency.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.n_modes_save = int(self.nblocks)
            return self.n_modes_save
        self.n_modes_save = int(request)
        return self.n_modes_save

    def _get_algorithm_metadata(self) -> dict:
        """Describe the current SPOD contract."""
        return {
            "lift_kind": "block_fourier_realizations",
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_second_order_operator": True,
            "spectral_estimator": "welch_block_average",
        }

    ############################################################
    # Data Loading and Preprocessing                           #
    ############################################################
    def load_and_preprocess(self) -> None:
        """
        Loads data, computes spatial weights, and sets SPOD-specific parameters.

        This method extends `BaseAnalyzer.load_and_preprocess()` by:
        1. Calling the parent method to load data, compute spatial weights,
           and set `nblocks` and `fs` (stored on `self.data`, `self.W`, `self.fs`).
           Mean subtraction is not done here; `blocksfft` removes the mean when
           FFT blocks are computed.
        2. Setting characteristic length (`self.L`) and velocity (`self.U`)
           based on filename conventions (e.g., 'cavity' or 'jet') for
           Strouhal number calculation.
        3. Calculating the frequency array (`self.freq`) from `rfftfreq`,
           the Strouhal numbers (`self.St`), and the Strouhal step (`self.dst`).
        """
        super().load_and_preprocess()  # Loads data, weights, nblocks, fs.

        # Resolve characteristic length and velocity for Strouhal normalization.
        # Explicit constructor values take precedence; fall back to data dict
        # entries (D, U0) if present, then to unity.
        self.L = self.L if self.L is not None else self.data.get("D", 1.0)
        self.U = self.U if self.U is not None else self.data.get("U0", 1.0)
        logger.info("Strouhal normalization: L=%s, U=%s.", self.L, self.U)

        # Calculate Strouhal vector and frequency axis (self.freq is set by BaseAnalyzer)
        # Here, we ensure self.freq and self.St are set before perform_spod
        # BaseAnalyzer.compute_fft_blocks sets self.freq based on rfftfreq
        # If super().run() calls compute_fft_blocks, self.freq should be populated.
        # For safety, we can calculate it here if not already done or to ensure consistency.
        if not hasattr(self, "freq") or self.freq.size == 0:
            self.freq = rfftfreq(self.nfft, d=self.data["dt"])

        self.St = self.freq * self.L / self.U

        if len(self.St) > 1:
            self.dst = self.St[1] - self.St[0]
        elif len(self.St) == 1:
            self.dst = self.St[0]  # Or some other appropriate non-zero value if St[0] can be 0
        else:
            self.dst = 0

    ############################################################
    # Core SPOD Computation                                    #
    ############################################################
    def perform_spod(self, weights: ArrayLike | None = None) -> None:
        """
        Performs the core SPOD analysis (eigenvalue decomposition for each frequency).

        This method computes the SPOD modes and eigenvalues by performing an
        eigenvalue decomposition of the cross-spectral density (CSD) matrix
        at each frequency bin. The CSD matrix is constructed from the
        Fourier-transformed data blocks (`self.qhat`).

        The eigenproblem itself is `spod_single_frequency`, in
        `openmodalpy.core.decomposition`.

        Forms the FFT blocks itself, via ``compute_fft_blocks()``, on first
        use — no separate call is needed between ``load_and_preprocess()``
        and this method.

        Attributes set:
            eigenvalues (np.ndarray): SPOD eigenvalues.
            modes (np.ndarray): SPOD spatial modes.
            time_coefficients (np.ndarray): SPOD time coefficients.
        """
        # Form the FFT blocks on first use, so callers never need a separate
        # compute_fft_blocks() step between load_and_preprocess() and this call.
        if self.qhat is None or self.qhat.size == 0:
            self.compute_fft_blocks()

        start_time = time.time()

        num_space_points = self.qhat.shape[1]
        num_freq_bins = self.qhat.shape[0]

        # Check if self.freq and self.St are consistent with qhat's frequency bins
        if len(self.freq) != num_freq_bins:
            logger.warning(
                "self.freq length (%d) mismatch with qhat bins (%d). Recalculating.",
                len(self.freq),
                num_freq_bins,
            )
            # Recalculate freq and St based on nfft and fs (from BaseAnalyzer)
            self.freq = rfftfreq(self.nfft, d=1.0 / self._require_fs())[:num_freq_bins]
            L = self.L
            U = self.U
            if L is None or U is None:
                raise RuntimeError("Characteristic L/U unset; call load_and_preprocess() first.")
            self.St = self.freq * L / U
            logger.info("Realigned self.freq to %d elements and self.St.", len(self.freq))

        n_keep = self._modes_to_keep()

        # Initialize result arrays using num_freq_bins. Eigenvalues keep every
        # block; modes and time coefficients keep the leading n_keep of them.
        self.eigenvalues = np.zeros((num_freq_bins, self.nblocks))
        self.modes = np.zeros((num_freq_bins, num_space_points, n_keep), dtype=complex)  # Spatial modes
        self.time_coefficients = np.zeros(
            (num_freq_bins, self.nblocks, n_keep), dtype=complex
        )  # Temporal coefficients, (block, mode) at each frequency

        logger.info("Performing SPOD for each frequency...")

        weights = np.asarray(weights if weights is not None else self.W)
        for i in tqdm(range(num_freq_bins), desc="SPOD Computation", unit="freq"):
            qhat_freq = self.qhat[i, :, :]
            # return_psi=True always yields the 3-tuple; unpack via star so the
            # 2-tuple overload of the eigenproblem does not block annotation.
            phi_freq, lambda_freq, *psi_rest = spod_single_frequency(
                qhat_freq,
                self.nblocks,
                self.dst,
                weights,
                return_psi=True,
            )
            if not psi_rest:
                raise RuntimeError("spod_single_frequency(return_psi=True) did not return psi")
            psi_freq = psi_rest[0]
            # phi is (space, mode) and psi is (block, mode), so both truncate on
            # their last axis. Eigenvalues are kept whole.
            self.modes[i, :, :] = phi_freq[:, :n_keep]
            self.eigenvalues[i, :] = lambda_freq
            self.time_coefficients[i, :, :] = psi_freq[:, :n_keep]
        logger.info("SPOD eigenvalue decomposition completed in %.2f seconds", time.time() - start_time)

    ############################################################
    # Results Handling                                         #
    ############################################################
    def _result_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return SPOD datasets and metadata to save."""
        datasets: dict[str, Any] = {
            "eigenvalues": self.eigenvalues,
            "modes": self.modes,
            "freq": self.freq,
            "st": self.St,
        }
        if self.time_coefficients is not None and self.time_coefficients.size > 0:
            datasets["time_coefficients"] = self.time_coefficients
        if self.qhat is not None and self.qhat.size > 0:
            datasets["FFTBlocks"] = self.qhat
        if self.W is not None and getattr(self.W, "size", 0) > 0:
            datasets["W"] = self.W
        if self.data.get("x") is not None:
            datasets["x"] = self.data["x"]
        if self.data.get("y") is not None:
            datasets["y"] = self.data["y"]
        if self.data.get("z") is not None:
            datasets["z"] = self.data["z"]
        return datasets, self._get_metadata()

    def _after_write(self, save_path: str) -> None:
        """Write FFT cache stamp after saving SPOD results.

        Reopens the result file in append mode to restore the FFT cache stamp
        for reuse in subsequent runs.
        """
        if self.qhat is not None and self.qhat.size > 0 and self.data.get("q") is not None:
            with h5py.File(save_path, "a") as handle:
                _write_qhat_stamp(handle, self, self.data["q"])
        elif self.qhat is not None and self.qhat.size > 0 and self.data.get("q") is None:
            logger.warning(
                "FFT blocks saved to %s without a cache stamp because source "
                "snapshots are not in memory; the next run will recompute them",
                save_path,
            )

    def _required_result_fields(self) -> tuple[str, ...]:
        """SPOD requires modes and eigenvalues."""
        return ("modes", "eigenvalues")

    ############################################################
    # Main Analysis Pipeline Orchestration                     #
    ############################################################
    def _assign_loaded_results(self, res: AnalysisResults) -> None:
        """Assign loaded results and reshape W to column form."""
        super()._assign_loaded_results(res)

        if res.W is not None:
            from openmodalpy.core.base import _as_spatial_weight_column

            n_space = int(res.modes.shape[1]) if res.modes is not None and res.modes.ndim == 3 else None
            self.W = _as_spatial_weight_column(res.W, n_space)

        if res.freq is not None:
            self.freq = res.freq
        if res.st is not None:
            self.St = res.st
        if res.FFTBlocks is not None:
            self.qhat = res.FFTBlocks

        if "dt" in self.data:
            self.fs = 1.0 / self._require_dt()

    @staticmethod
    def _run_plot(method: Callable[..., None], options: dict | None) -> None:
        """Call *method* with *options* unless explicitly disabled."""
        if not options:
            method()
            return
        if options.get("enabled") is False:
            return
        kwargs = {k: v for k, v in options.items() if k != "enabled"}
        method(**kwargs)

    _perform_name = "perform_spod"
    _needs_fft_blocks = True

    def _plot_run(self, run_id: str | None = None) -> None:
        os.makedirs(self.figures_dir, exist_ok=True)
        """Default figures after run_analysis — the CLI spod set.

        Finer per-plot enable/disable control stays available through
        ``_run_plot``; this default mirrors what the CLI always produced.
        """
        self.plot_eigenvalues()
        dominant_idx = int(np.argmax(self.eigenvalues[:, 0]))
        if not self._maybe_plot_volumetric_modes(
            plot_n_modes=min(2, self.modes.shape[2]),
            slices_kwargs={"freqs_to_plot": [dominant_idx]},
            iso_kwargs={"freqs_to_plot": [dominant_idx]},
        ):
            self.plot_modes(
                freqs_to_plot=[dominant_idx],
                plot_n_modes=min(2, self.modes.shape[2]),
                modes_per_fig=2,
            )
        self.plot_cumulative_energy()

    def plot_eigenvalues(self, n_modes_line_plot: int = 20, shading_cmap: str = "inferno_r") -> None:
        """Plot the SPOD eigenvalue spectrum (energy vs. Strouhal number).

        Shaded background for the eigenvalue bundle and grayscale lines for modes.

        Args:
            n_modes_line_plot (int, optional): Number of dominant mode eigenvalue lines to plot.
                                             Defaults to 20.
            shading_cmap (str, optional): Colormap for the background shading of the eigenvalue bundle.
                                        Defaults to 'inferno_r' (yellow at bottom, dark at top).
        """
        if self.eigenvalues.size == 0 or self.St.size == 0:
            logger.warning(
                "Eigenvalues (self.eigenvalues) or Strouhal numbers (self.St) not computed. Run perform_spod() first."
            )
            return

        L_plot = self.eigenvalues  # Use self.eigenvalues for the eigenvalue matrix
        St_plot = self.St

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Determine y-range for pcolormesh shading (covering the bundle)
        # Ensure l_min is positive for log scale
        l_min_positive = L_plot[L_plot > 0].min() if np.any(L_plot > 0) else 1e-6
        y_fill_min = l_min_positive * 0.1
        y_fill_max = L_plot[:, 0].max()  # Shade up to the first mode's max
        if y_fill_min >= y_fill_max:
            y_fill_min = y_fill_max * 0.01  # Ensure min < max

        # Create a grid for pcolormesh
        # y_coords_mesh should be log-spaced for even coloring in log y-scale
        num_y_points_mesh = 100
        y_coords_mesh = np.logspace(np.log10(y_fill_min), np.log10(y_fill_max), num_y_points_mesh)

        # Fill the background shading without explicit Python loops
        first_mode = L_plot[:, 0]
        mask = y_coords_mesh[:, None] <= first_mode[None, :]
        C_fill_data = np.where(
            mask,
            np.broadcast_to(y_coords_mesh[:, None], (num_y_points_mesh, len(St_plot))),
            np.nan,
        )
        # Mask NaNs to avoid matplotlib warnings when using LogNorm
        C_fill_data = np.ma.masked_invalid(C_fill_data)

        norm_vmin = np.min(y_coords_mesh[y_coords_mesh > 0])
        norm_vmax = np.max(y_coords_mesh)
        if norm_vmin >= norm_vmax:
            norm_vmin = norm_vmax * 0.1  # ensure vmin < vmax and positive

        if norm_vmin > 0 and norm_vmax > 0 and norm_vmin < norm_vmax:
            _ = ax.pcolormesh(
                St_plot,
                y_coords_mesh,
                C_fill_data,
                shading="gouraud",
                cmap=shading_cmap,
                norm=colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax),
                rasterized=True,
            )  # Rasterize for potentially large mesh
        else:
            logger.warning("Could not generate pcolormesh shading due to invalid vmin/vmax for LogNorm.")

        # Plot individual modal energies (eigenvalues) - grayscale lines
        num_modes_actual = min(n_modes_line_plot, L_plot.shape[1])
        mode_colors = plt.get_cmap("gray")(np.linspace(0.0, 0.7, num_modes_actual))
        for i in range(num_modes_actual):
            ax.plot(St_plot, L_plot[:, i], color=mode_colors[i], linewidth=0.8, alpha=0.9)

        # Plot total energy (sum of eigenvalues) - red line
        ax.plot(St_plot, np.sum(L_plot, axis=1), color="red", linewidth=1.5, label="Total Energy")

        ax.set_xlabel("Strouhal number")
        ax.set_ylabel(r"$\lambda$")  # Use raw string for LaTeX
        ax.set_title("SPOD Eigenvalue Spectrum")

        # Set plot limits - adjust as necessary
        if np.any(St_plot > 0):
            st_min = St_plot[St_plot > 0].min()
        else:
            st_min = St_plot.min()
        ax.set_xlim(st_min, St_plot.max())
        plot_y_min = y_fill_min * 0.5
        plot_y_max = np.max(np.sum(L_plot, axis=1)) * 2.0
        if plot_y_min > 0 and plot_y_max > 0 and plot_y_min < plot_y_max:
            ax.set_ylim(plot_y_min, plot_y_max)
        else:  # Fallback if limits are problematic
            current_st_min, current_st_max = ax.get_xlim()
            if np.any(L_plot[L_plot > 0]):
                ax.set_ylim(L_plot[L_plot > 0].min() * 0.1, np.sum(L_plot, axis=1).max() * 2.0)

        # Use settings from configs.py for saving
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_eigenvalues_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )  # Corrected self.novlap

        # Save the figure
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("SPOD eigenvalue plot saved to %s", plot_filename)

    def plot_modes(
        self,
        freqs_to_plot: Iterable[int | float] | None = None,
        plot_n_modes: Optional[int] = 10,
        modes_per_fig: int = 1,
        show_cylinder: bool = False,
    ) -> None:
        """Plot spatial modes for selected frequencies as individual figures.

        Parameters
        ----------
        freqs_to_plot : list, tuple or numpy array, optional
            Frequencies to plot modes for (bin indices or Strouhal numbers)
        plot_n_modes : int, optional
            Number of modes to plot per frequency (default 10)
        modes_per_fig : int, optional
            Number of modes per figure (default 1)
        show_cylinder : bool, optional
            If True, add cylinder mask at origin with radius 0.5 (default False)
        """

        if self.modes.size == 0 or self.St.size == 0:
            logger.warning("No modes to plot. Run perform_spod() first.")
            return

        n_modes_total = self.modes.shape[2]
        if plot_n_modes is None:
            n_modes = n_modes_total
        else:
            n_modes = min(plot_n_modes, n_modes_total)

        if resolve_volume_layout(self.data, self.modes.shape[1]) is not None:
            self.plot_modes_3d_slices(freqs_to_plot=freqs_to_plot, plot_n_modes=n_modes)
            return

        if freqs_to_plot is None:
            dominant_idx = int(np.argmax(self.eigenvalues[:, 0]))
            freqs_to_plot = [dominant_idx]

        freq_indices = []
        for f in freqs_to_plot:
            if isinstance(f, (int, np.integer)) and 0 <= f < len(self.St):
                freq_indices.append(int(f))
            else:
                freq_indices.append(int(np.argmin(np.abs(self.St - float(f)))))

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[1])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[1])))
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        for f_idx in freq_indices:
            st_val = self.St[f_idx]
            for start in range(0, n_modes, modes_per_fig):
                end = min(start + modes_per_fig, n_modes)
                ncols = end - start
                if Nx * Ny == self.modes.shape[1] and Nx > 1 and Ny > 1:
                    fig, axes = plt.subplots(
                        1,
                        ncols,
                        figsize=(4 * ncols * fig_aspect, 4),
                        squeeze=False,
                    )
                else:
                    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3), squeeze=False)
                axes = axes.ravel()
                for j, m_idx in enumerate(range(start, end)):
                    if m_idx >= self.modes.shape[2]:
                        continue
                    ax = axes[j]
                    mode_real = self.modes[f_idx, :, m_idx].real
                    if Nx * Ny == mode_real.size and Nx > 1 and Ny > 1:
                        mode_2d = mode_real.reshape(Ny, Nx)
                        if x_coords.ndim == 1 and y_coords.ndim == 1:
                            X, Y = np.meshgrid(x_coords, y_coords)  # contract layout: arrays are (Ny, Nx)
                        else:
                            X, Y = x_coords, y_coords
                        # Optionally apply cylinder mask
                        if show_cylinder:
                            dist = np.sqrt(X**2 + Y**2)
                            cyl_mask = dist <= 0.5
                            mode_plot = np.ma.array(mode_2d, mask=np.isnan(mode_2d) | cyl_mask)
                        else:
                            mode_plot = np.ma.array(mode_2d, mask=np.isnan(mode_2d))
                        from openmodalpy.core.base import get_robust_clim

                        vmin, vmax = get_robust_clim(mode_plot, method="percentile")
                        levels = np.linspace(vmin, vmax, 21)
                        im = ax.contourf(X, Y, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")
                        ax.contour(X, Y, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                        if show_cylinder:
                            ax.add_patch(
                                plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                            )
                        style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                        add_inset_colorbar(
                            fig,
                            ax,
                            im,
                            self.data,
                            ticks=[vmin, 0, vmax],
                            ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
                        )
                    else:
                        ax.plot(mode_real)
                        ax.set_xlabel("Spatial index")
                        ax.set_ylabel("Amplitude")
                    ax.set_title(
                        format_mode_title(
                            self.data,
                            m_idx,
                            default=f"SPOD Mode {m_idx + 1} at St={st_val:.4f} [{var_name}]",
                        )
                    )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    fig.tight_layout()
                if modes_per_fig == 1 and ncols == 1:
                    fname = os.path.join(
                        self.figures_dir,
                        f"{self.data_root}_SPOD_mode{start + 1}_freq{f_idx}_{var_name}.png",
                    )
                else:
                    fname = os.path.join(
                        self.figures_dir,
                        f"{self.data_root}_SPOD_modes_{start + 1}_to_{end}_freq{f_idx}_{var_name}.png",
                    )
                fig.savefig(fname, dpi=FIG_DPI)
                plt.close(fig)
                logger.info(
                    "SPOD modes %d-%d at St=%.4f (freq index %d) saved to %s",
                    start + 1,
                    end,
                    st_val,
                    f_idx,
                    fname,
                )

    def plot_modes_3d_slices(
        self, freqs_to_plot: Iterable[int | float] | None = None, plot_n_modes: Optional[int] = 2
    ) -> None:
        """Plot orthogonal 3D slices for selected SPOD frequencies and modes.

        ``freqs_to_plot`` is a list, tuple, or numpy array of bin indices or
        Strouhal numbers.
        """
        self._plot_modes_3d("slices", freqs_to_plot=freqs_to_plot, plot_n_modes=plot_n_modes)

    def plot_modes_3d_isometric(
        self, freqs_to_plot: Iterable[int | float] | None = None, plot_n_modes: Optional[int] = 2
    ) -> None:
        """Plot 3D isosurfaces for selected SPOD frequencies and modes.

        ``freqs_to_plot`` is a list, tuple, or numpy array of bin indices or
        Strouhal numbers.
        """
        self._plot_modes_3d("isometric", freqs_to_plot=freqs_to_plot, plot_n_modes=plot_n_modes)

    def _plot_modes_3d(
        self,
        kind: str,
        freqs_to_plot: Iterable[int | float] | None = None,
        plot_n_modes: Optional[int] = 2,
    ) -> None:
        if self.modes.size == 0 or self.St.size == 0:
            logger.warning("No modes to plot. Run perform_spod() first.")
            return
        if resolve_volume_layout(self.data, self.modes.shape[1]) is None:
            logger.warning("plot_modes_3d_%s requires volumetric data.", kind)
            return
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        n_modes = self.modes.shape[2] if plot_n_modes is None else min(plot_n_modes, self.modes.shape[2])
        if freqs_to_plot is None:
            dominant_idx = int(np.argmax(self.eigenvalues[:, 0]))
            freq_indices = [dominant_idx]
        else:
            freq_indices = []
            for f in freqs_to_plot:
                if isinstance(f, (int, np.integer)) and 0 <= f < len(self.St):
                    freq_indices.append(int(f))
                else:
                    freq_indices.append(int(np.argmin(np.abs(self.St - float(f)))))
        items = []
        for f_idx in freq_indices:
            st_val = self.St[f_idx]
            for mode_idx in range(n_modes):
                mode_3d = reshape_mode_to_volume(self.modes[f_idx, :, mode_idx].real, self.data)
                output_path = os.path.join(
                    self.figures_dir, f"{self.data_root}_SPOD_mode{mode_idx + 1}_freq{f_idx}_{kind}.png"
                )
                items.append(
                    {
                        "mode_3d": mode_3d,
                        "output_path": output_path,
                        "title_prefix": f"SPOD Mode {mode_idx + 1} | St={st_val:.4f}",
                        "scalar_name": "spod_mode",
                    }
                )
        plot_modes_3d(kind, items, x_coords, y_coords, z_coords, data=self.data)

    def plot_cumulative_energy(self, freq_idx: int | None = None) -> None:
        """Plot cumulative energy captured by modes at a given frequency."""
        if self.eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot. Run perform_spod() first.")
            return
        if freq_idx is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        lambdas = self.eigenvalues[freq_idx, :]
        cumulative = np.cumsum(lambdas)
        cumulative /= cumulative[-1]
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, "o-")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Cumulative energy")
        ax.set_title(f"Cumulative energy at St={self.St[freq_idx]:.3f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_cumulative_freq{freq_idx}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Cumulative energy plot saved to %s", plot_filename)

    def plot_time_coefficients(
        self,
        modes_to_plot: Iterable[int] | None = None,
        freq: float | None = None,
        n_blocks: int | None = None,
    ) -> None:
        """Plot temporal coefficients for selected modes.

        ``modes_to_plot`` is a list, tuple, or numpy array of mode indices.
        """
        if self.time_coefficients.size == 0:
            logger.warning("No time coefficients to plot. Run perform_spod() first.")
            return
        if freq is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(freq))))
        coeffs = self.time_coefficients[freq_idx, :, :]
        if n_blocks is not None:
            coeffs = coeffs[:n_blocks, :]
        if modes_to_plot is None:
            modes_to_plot = list(range(min(4, coeffs.shape[1])))
        fig, ax = plt.subplots()
        for m in modes_to_plot:
            if m < coeffs.shape[1]:
                ax.plot(coeffs[:, m].real, label=f"Mode {m + 1} (real)")
                ax.plot(coeffs[:, m].imag, "--", label=f"Mode {m + 1} (imag)")
        ax.set_xlabel("Block index")
        ax.set_ylabel("Coefficient")
        ax.legend()
        ax.set_title(f"SPOD Time Coefficients at St={self.St[freq_idx]:.4f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_timecoeffs_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Time coefficients plot saved to %s", plot_filename)

    def plot_reconstruction_error(self, st_target: float | None = None, n_modes_max: int | None = None) -> None:
        """Plot reconstruction error of qhat as a function of modes used."""
        if self.qhat.size == 0 or self.modes.size == 0:
            logger.warning("No data to plot reconstruction error. Run perform_spod() first.")
            return
        if st_target is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(st_target))))
        qhat_f = self.qhat[freq_idx, :, :]
        modes_f = self.modes[freq_idx, :, :]
        n_modes_avail = modes_f.shape[1]
        if n_modes_max is None:
            n_modes_max = n_modes_avail
        n_modes_max = min(n_modes_max, n_modes_avail)
        W = np.diagflat(self.W) if self.W.ndim == 1 or self.W.shape[1] == 1 else self.W
        norm_orig = np.linalg.norm(qhat_f, "fro")
        errors = []
        mode_counts = range(1, n_modes_max + 1)
        for k in mode_counts:
            phi_k = modes_f[:, :k]
            coeffs = phi_k.conj().T @ W @ qhat_f
            if coeffs.shape != (k, qhat_f.shape[1]):
                raise ValueError(
                    f"Coefficient matrix has unexpected shape: {coeffs.shape}, expected ({k}, {qhat_f.shape[1]})"
                )
            qrec = phi_k @ coeffs
            errors.append(np.linalg.norm(qhat_f - qrec, "fro") / norm_orig)
        fig, ax = plt.subplots()
        ax.plot(list(mode_counts), errors, "o-")
        ax.set_yscale("log")
        ax.set_xlabel("Number of SPOD Modes")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"Reconstruction Error at St={self.St[freq_idx]:.4f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_reconstruction_error_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Reconstruction error plot saved to %s", plot_filename)

    def plot_eig_complex_plane(self, freq: float | None = None, n_modes: int = 4) -> None:
        """Plot modes in the complex plane for a given frequency."""
        if self.modes.size == 0:
            logger.warning("No modes to plot. Run perform_spod() first.")
            return
        if freq is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(freq))))
        modes = self.modes[freq_idx, :, :n_modes]
        fig, ax = plt.subplots()
        for i in range(modes.shape[1]):
            ax.scatter(modes[:, i].real, modes[:, i].imag, s=10, alpha=0.7, label=f"Mode {i + 1}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title(f"SPOD Modes Complex Plane St={self.St[freq_idx]:.4f}")
        ax.legend()
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_complex_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        logger.info("Complex plane plot saved to %s", plot_filename)
