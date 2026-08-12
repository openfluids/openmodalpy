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
from typing import Optional

import numpy as np

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import BaseAnalyzer, make_result_filename
from openmodalpy.core.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    WINDOW_NORM,
    WINDOW_TYPE,
)

logger = logging.getLogger(__name__)


class PSDPODAnalyzer(BaseAnalyzer):
    """POD on the ensemble of blockwise Fourier realizations.

    Lifecycle: ``load_and_preprocess`` → ``compute_fft_blocks`` →
    ``perform_psd_pod`` → ``save_results``. ``run_analysis`` runs that sequence.

    Key attributes after a successful run:
        modes: spatial modes, shape (Nspace, n_modes_save), complex
        eigenvalues: energy of each mode, shape (n_modes_save,)
        time_coefficients: projections of the Fourier ensemble, shape
            (n_fourier_realizations, n_modes_save)
        freq, St: frequency and Strouhal axes from the Welch blocks
    """

    def __init__(
        self,
        file_path: str,
        results_dir: str = RESULTS_DIR,
        figures_dir: str = FIGURES_DIR,
        data_loader: Callable[[str], dict] | None = None,
        spatial_weight_type: str | None = None,
        nfft: int = 128,
        overlap: float = 0.5,
        n_modes_save: int = 10,
        blockwise_mean: bool = False,
        window_norm: str = WINDOW_NORM,
        window_type: str = WINDOW_TYPE,
        use_parallel: bool = True,
        characteristic_length: float | None = None,
        characteristic_velocity: float | None = None,
        spatial_weights: np.ndarray | None = None,
    ):
        super().__init__(
            file_path=file_path,
            nfft=nfft,
            overlap=overlap,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
            spatial_weights=spatial_weights,
        )
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
        self.freq = np.fft.rfftfreq(self.nfft, d=self.data["dt"])
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

    def save_results(self, filename: str | None = None) -> None:
        """Write modes, eigenvalues, time coefficients, freq and st to HDF5."""
        from openmodalpy.core.results import write_results

        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        save_path = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        attrs = self._get_metadata()
        attrs["analysis_type"] = "psd_pod"
        write_results(
            save_path,
            {
                "eigenvalues": np.asarray(self.eigenvalues),
                "modes": self.modes,
                "time_coefficients": self.time_coefficients,
                "freq": self.freq,
                "st": self.St,
            },
            attrs=attrs,
        )
        self.results_path = save_path
        logger.info("Saved PSD-POD results to %s", save_path)

    def run_analysis(self) -> None:
        """One-call entry: load, FFT blocks, eigenproblem, save."""
        logger.info("Starting PSD-POD analysis for %s", os.path.basename(self.file_path))
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_psd_pod()
        self.save_results()
