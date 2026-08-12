#!/usr/bin/env python3
"""
Extract coherent bispectral modes with BiSpectral Mode Decomposition (BSMD)

Reference: "Bispectral mode decomposition of nonlinear flows."  Schmidt, O. T. (2020).

Definitions:
  bispectrum B(f1,f2) = ⟨ X(f1) X(f2) X*(f1+f2) ⟩,
  triad (f1,f2,f3) satisfying f1 + f2 = f3.

Current implementation note:
  the ideal BSMD objective is a numerical-radius problem for a generally
  non-normal cross-bispectral operator. The current analyzer uses the dominant
  eigenpair of the assembled matrix ``C`` as a practical approximation.

Method:
  1. Compute FFT blocks via Welch’s method: qhat[f, j, b].
  2. For each triad, form:
       A_jb = conj[ qhat[p1, j, b] · qhat[p2, j, b] ],
       B_jb =     qhat[p3, j, b].
  3. Build bispectral correlation:
       C = A^H W B,  C_{bb'} = Σ_j A_jb^* W_j B_jb'.
  4. Solve: C a = λ a, obtain eigenmodes a.
  5. Spatial modes:
       Φ1_j = Σ_b a_b^* B_jb,  Φ2_j = Σ_b a_b^* A_jb.
"""

from __future__ import annotations

# Standard library imports
import logging
import os
import re
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from openmodalpy.core.base import (
    BaseAnalyzer,
    _as_spatial_weight_column,
    _hdf5_write_mode,
    add_inset_colorbar,
    canonicalize_modes,
    get_fig_aspect_ratio,
    make_result_filename,
    plot_modes_3d,
    print_summary,
    require_spatial_metric,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.config import (
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_BSMD,
    RESULTS_DIR_BSMD,
)
from openmodalpy.core.threads import apply_blas_limit

logger = logging.getLogger(__name__)

# Standard static triad list
ALL_TRIADS = [
    (8, -8, 0),
    (7, -7, 0),
    (8, -7, 1),
    (6, -6, 0),
    (7, -6, 1),
    (8, -6, 2),
    (5, -5, 0),
    (6, -5, 1),
    (7, -5, 2),
    (8, -5, 3),
    (4, -4, 0),
    (5, -4, 1),
    (6, -4, 2),
    (7, -4, 3),
    (8, -4, 4),
    (3, -3, 0),
    (4, -3, 1),
    (5, -3, 2),
    (6, -3, 3),
    (7, -3, 4),
    (8, -3, 5),
    (2, -2, 0),
    (3, -2, 1),
    (4, -2, 2),
    (5, -2, 3),
    (6, -2, 4),
    (7, -2, 5),
    (8, -2, 6),
    (1, -1, 0),
    (2, -1, 1),
    (3, -1, 2),
    (4, -1, 3),
    (5, -1, 4),
    (6, -1, 5),
    (7, -1, 6),
    (8, -1, 7),
    (0, 0, 0),
    (1, 0, 1),
    (2, 0, 2),
    (3, 0, 3),
    (4, 0, 4),
    (5, 0, 5),
    (6, 0, 6),
    (7, 0, 7),
    (8, 0, 8),
    (1, 1, 2),
    (2, 1, 3),
    (3, 1, 4),
    (4, 1, 5),
    (5, 1, 6),
    (6, 1, 7),
    (7, 1, 8),
    (2, 2, 4),
    (3, 2, 5),
    (4, 2, 6),
    (5, 2, 7),
    (6, 2, 8),
    (3, 3, 6),
    (4, 3, 7),
    (5, 3, 8),
    (4, 4, 8),
]


class BSMDAnalyzer(BaseAnalyzer):
    """
    Bispectral Mode Decomposition (BSMD) Analyzer.

    This class implements BSMD to extract coherent structures involved in triadic interactions,
    typically indicative of nonlinear processes in fluid flows or other dynamical systems.
    The method is based on the paper: Schmidt, O. T. (2020). "Bispectral mode decomposition
    of nonlinear flows."

    Key concepts:
    - Bispectrum: B(f1, f2) = < X(f1) X(f2) X*(f1+f2) >, measures the statistical
      dependence between three frequency components satisfying the triadic relation f1 + f2 = f3.
    - Triad: A set of three frequencies (f1, f2, f3) such that f1 + f2 = f3.
    - BSMD Eigenvalue Problem: Solved for each triad to find modes (modes1, modes2) and
      eigenvalues that characterize the strength and spatial structure of the interaction.

    The typical BSMD process involves:
    1. Computing FFT blocks of the data (e.g., using Welch's method) to get q_hat[f, j, b]
       (frequency, spatial_point, block_index).
    2. For each selected triad (p1, p2, p3) where p_k are frequency indices:
       a. Form auxiliary matrices A_jb = conj(q_hat[p1,j,b] * q_hat[p2,j,b]) and B_jb = q_hat[p3,j,b].
       b. Construct the bispectral correlation matrix C_bb' = sum_j (A_jb^* W_j B_jb').
       c. Solve the dominant-eigenpair approximation to the BSMD operator.
    3. Reconstruct spatial modes: modes1_j = sum_b (a_b^* B_jb) and modes2_j = sum_b (a_b^* A_jb).

    Key Attributes:
        modes1 (np.ndarray): BSMD spatial modes (related to f1, f2 interaction product).
                           Shape: (n_triads, n_spatial_points).
        modes2 (np.ndarray): BSMD spatial modes (related to f3).
                           Shape: (n_triads, n_spatial_points).
        eigenvalues (np.ndarray): BSMD eigenvalues (lambda), complex values indicating interaction strength and phase.
                                  Shape: (n_triads,).
        triads (list of tuples): List of frequency index triads (p1, p2, p3) analyzed.
        qhat (np.ndarray): STFT of the data, q_hat[frequency_bin, spatial_point, block].
        fs (float): Sampling frequency of the data.
        nfft (int): Number of points per FFT block.
        W (np.ndarray): Spatial weighting matrix (diagonal).

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading, STFT computation,
                      and preprocessing.
    """

    def __init__(
        self,
        file_path: str,
        nfft: int = 128,
        overlap: float = 0.5,
        results_dir: str = RESULTS_DIR_BSMD,
        figures_dir: str = FIGURES_DIR_BSMD,
        data_loader: Callable[..., dict[str, Any]] | None = None,
        spatial_weight_type: str | None = None,
        use_static_triads: bool = True,
        static_triads: Sequence[tuple[int, int, int]] | None = None,
        use_parallel: bool = True,
        max_qhat_gb: float = 4.0,
        spatial_weights: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the BSMDAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            nfft (int, optional): Number of points per FFT segment for STFT.
                                  Defaults to 128.
            overlap (float, optional): Overlap ratio between FFT segments (0 to 1).
                                     Defaults to 0.5.
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_BSMD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_BSMD` from `configs.py`.
            data_loader (callable, optional): Custom function to load data from `file_path`.
                                              If None, `BaseAnalyzer` attempts to auto-detect.
                                              Defaults to None.
            spatial_weight_type (str | None, optional): Type of spatial weights to apply
                (None → 'uniform', or 'uniform', 'polar', 'prescribed'). Defaults to None.
            spatial_weights: Optional array of spatial integration weights. When given,
                the type becomes 'prescribed'.
            use_static_triads (bool, optional): If True, use the `static_triads` list.
                                                If False, dynamic triad selection (not yet fully implemented)
                                                would be attempted. Defaults to True.
            static_triads (list of tuples, optional): List of predefined frequency index triads
                                                     (p_k, p_l, p_k+p_l) to analyze. ``None``
                                                     (the default) resolves to a private copy of
                                                     ``ALL_TRIADS``. Provenance is recorded so a
                                                     small ``nfft`` filters the default list with a
                                                     warning, while a user-supplied list still raises.
            max_qhat_gb (float, optional): Maximum qhat size (GB) to keep in RAM.
                                           Larger arrays are offloaded to HDF5 and served
                                           slice-by-slice during BSMD.  Defaults to 4.0.
        """
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
        self.use_static_triads = use_static_triads
        # Provenance is fixed at construction: comparing the list to ALL_TRIADS
        # later would misclassify a user who legitimately passes that same list.
        # Keep the resolved default object so a post-construction replacement of
        # static_triads_list is detected as user-supplied (identity check).
        self._static_triads_from_default = use_static_triads and static_triads is None
        if not use_static_triads:
            self.static_triads_list = []
            self._resolved_default_triads = None
        elif static_triads is None:
            self.static_triads_list = list(ALL_TRIADS)
            self._resolved_default_triads = self.static_triads_list
        else:
            self.static_triads_list = list(static_triads)
            self._resolved_default_triads = None
        self.analysis_type = "bsmd"

        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Derive base name for outputs
        base = os.path.basename(file_path)
        self.data_root = re.sub(r"\.[^.]*$", "", base)

        # Placeholders
        self.data: dict[str, Any] = {}
        self.W = np.array([])
        self.novlap = int(overlap * nfft)
        self.nblocks = 0
        self.qhat = np.array([])
        self.qhat_cached = False
        self.triads: list[tuple[int, int, int]] | np.ndarray = []
        self.eigenvalues = np.array([])
        self.modes1 = np.array([])
        self.modes2 = np.array([])
        self.freq: np.ndarray | None = None
        self.St: np.ndarray | None = None
        self.energy_map = np.array([])

        # Disk-backed qhat for large datasets
        self._max_qhat_bytes = int(max_qhat_gb * 1024**3)
        self._qhat_file: h5py.File | None = None  # kept open in disk mode
        self._qhat_dataset: h5py.Dataset | None = None
        self._qhat_on_disk = False
        self._qhat_bin_cache: dict[int, np.ndarray] = {}  # {abs_freq_bin: np.ndarray}

    # -- Disk-backed qhat management -----------------------------------------

    def _maybe_offload_qhat(self) -> None:
        """If qhat exceeds the memory threshold, offload to HDF5 and free RAM.

        After this call, ``self.qhat`` is an empty array and all frequency-bin
        access goes through ``self._qhat_dataset`` (an open h5py Dataset).
        """
        if self._qhat_on_disk or self.qhat.size == 0:
            return
        if self.qhat.nbytes <= self._max_qhat_bytes:
            return

        cache_path = self._qhat_cache_path
        if cache_path is None or not os.path.exists(cache_path):
            return  # No cache file to back onto

        qhat_gb = self.qhat.nbytes / 1024**3
        logger.info(
            "qhat is %.1f GB (threshold %.1f GB) — switching to disk-backed mode.",
            qhat_gb,
            self._max_qhat_bytes / 1024**3,
        )
        self._qhat_file = h5py.File(cache_path, "r")
        self._qhat_dataset = self._qhat_file["FFTBlocks"]
        self._qhat_on_disk = True
        self.qhat = np.array([])  # release RAM

    def _prefetch_bins(self) -> None:
        """Pre-load all frequency bins needed by the triad list into the cache.

        In disk-backed mode this reads each unique bin from HDF5 exactly once,
        *before* threads are spawned, so the parallel loop never touches h5py.
        In RAM mode this is a no-op.
        """
        if not self._qhat_on_disk:
            return
        dataset = self._qhat_dataset
        assert dataset is not None
        needed = set()
        for p1, p2, p3 in self.static_triads_list:
            needed.update([abs(p1), abs(p2), abs(p3)])
        to_read = sorted(needed - set(self._qhat_bin_cache))
        if not to_read:
            return
        for bin_idx in to_read:
            if bin_idx < dataset.shape[0]:
                self._qhat_bin_cache[bin_idx] = dataset[bin_idx, :, :]
        total_mb = sum(v.nbytes for v in self._qhat_bin_cache.values()) / 1024**2
        logger.info(
            "Pre-fetched %d frequency bins (%.0f MB) for %d triads.",
            len(to_read),
            total_mb,
            len(self.static_triads_list),
        )

    def close(self) -> None:
        """Release disk-backed resources (HDF5 file handle, bin cache)."""
        self._qhat_bin_cache.clear()
        if self._qhat_file is not None:
            self._qhat_file.close()
            self._qhat_file = None
            self._qhat_dataset = None
            self._qhat_on_disk = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # best-effort cleanup during GC

    # -- Data loading --------------------------------------------------------

    def load_and_preprocess(self) -> None:
        """
        Loads data, computes spatial weights, and STFT using BaseAnalyzer methods.

        This method orchestrates:
        1. Loading data via `_load_data()`.
        2. Determining and applying spatial weights via `_calculate_spatial_weights()`.
        3. Computing the STFT of the data via `compute_fft_blocks()`.

        Sets attributes like `self.data`, `self.W`, `self.qhat`, `self.fs`, `self.freq`.
        """
        super().load_and_preprocess()  # Leverages BaseAnalyzer's core logic

    def _on_fft_blocks_ready(self) -> None:
        """Set frequency axes and optionally offload large ``qhat`` to disk.

        The shared cache lives in ``BaseAnalyzer.compute_fft_blocks``; BSMD
        only needs this post-step. ``_qhat_cache_path`` is set by the base
        path so ``save_results`` can still append onto the same file.
        """
        freq = np.fft.rfftfreq(self.nfft, d=1.0 / self._require_fs())
        self.freq = freq
        self.St = freq.copy()  # Default: Strouhal equals frequency if no scaling
        self._maybe_offload_qhat()

    # Main method to perform BSMD analysis based on configuration.
    def perform_bsmd(self) -> None:
        """
        Perform Bispectral Mode Decomposition (BSMD) analysis.

        This method acts as a dispatcher based on the `self.use_static_triads` attribute.
        - If True, it calls `_perform_static_bsmd_core` to analyze predefined triads.
        - If False (or for future dynamic triad selection), it would call `perform_dynamic_bsmd`.

        Ensures data is loaded and preprocessed (STFT computed) before proceeding.
        """
        if self.qhat.size == 0 and not self._qhat_on_disk:
            raise ValueError("STFT data (qhat) not found. Call load_and_preprocess() first.")
        start_time = time.time()
        logger.info("Starting BSMD analysis...")

        if self.use_static_triads:
            self._perform_static_bsmd_core()
        else:
            raise NotImplementedError("Dynamic BSMD is not yet implemented.")

        logger.info("BSMD analysis completed in %.2f seconds.", time.time() - start_time)

    @property
    def _n_freq_bins(self) -> int:
        """Number of frequency bins, whether qhat is in RAM or on disk."""
        if self._qhat_on_disk:
            dataset = self._qhat_dataset
            assert dataset is not None
            return dataset.shape[0]
        return self.qhat.shape[0]

    @property
    def _n_spatial(self) -> int:
        """Number of spatial points, whether qhat is in RAM or on disk."""
        if self._qhat_on_disk:
            dataset = self._qhat_dataset
            assert dataset is not None
            return dataset.shape[1]
        return self.qhat.shape[1]

    def _get_qhat_for_index(self, idx: int) -> np.ndarray:
        """Return qhat slice for a frequency bin index, handling negatives via conjugate symmetry.

        For real-valued signals the DFT satisfies X(-k) = conj(X(k)).
        Since ``self.qhat`` stores only non-negative frequency bins (rfftfreq),
        negative indices are served by conjugating the corresponding positive bin.

        In disk-backed mode, slices are read from HDF5 and cached in
        ``self._qhat_bin_cache`` so each physical bin is read at most once.

        Args:
            idx: Integer frequency bin index (can be negative).

        Returns:
            Array of shape ``(Nspace, Nblocks)``.

        Raises:
            IndexError: If ``|idx|`` exceeds the number of available frequency bins.
        """
        n_freq_bins = self._n_freq_bins
        abs_idx = abs(idx)
        if abs_idx >= n_freq_bins:
            raise IndexError(f"Frequency bin index {idx} out of range [{-(n_freq_bins - 1)}, {n_freq_bins - 1}]")

        # Fetch the positive-frequency slice (from cache, disk, or RAM)
        if abs_idx in self._qhat_bin_cache:
            data = self._qhat_bin_cache[abs_idx]
        elif self._qhat_on_disk:
            dataset = self._qhat_dataset
            assert dataset is not None
            data = dataset[abs_idx, :, :]
            self._qhat_bin_cache[abs_idx] = data
        else:
            data = self.qhat[abs_idx, :, :]

        return np.conj(data) if idx < 0 else data

    def _compute_single_triad(
        self, p1: int, p2: int, p3: int
    ) -> tuple[complex | float, np.ndarray | None, np.ndarray | None]:
        """Compute BSMD eigenvalue and spatial modes for one triad.

        Thread-safe: reads from shared ``self.qhat`` and ``self.W`` (read-only),
        all outputs are returned as local values.

        Returns:
            (eigenvalue, mode1, mode2) on success, or (np.nan, None, None) on
            failure (constraint violation ``p1+p2 != p3``, empty blocks, or
            NaN/Inf in the assembled matrix that makes ``np.linalg.eig`` raise
            ``LinAlgError``). Out-of-range bin indices are not caught here;
            they propagate from ``_get_qhat_for_index``.
        """
        if p1 + p2 != p3:
            return np.nan, None, None

        Q1 = self._get_qhat_for_index(p1)  # (Nspace, Nblocks)
        Q2 = self._get_qhat_for_index(p2)
        Q3 = self._get_qhat_for_index(p3)

        nblocks = Q1.shape[1]
        if nblocks == 0:
            return np.nan, None, None

        # Assemble the finite-dimensional cross-bispectral matrix and use the
        # dominant eigenpair as the current approximation to the ideal BSMD
        # numerical-radius problem.
        prod = Q1 * Q2  # (Nspace, Nblocks)
        # Schmidt (2020), Bispectral mode decomposition of nonlinear flows:
        #   B = Q_{k+l}^H W (Q_k o Q_l) / N_blk
        # The sum-frequency term carries the conjugate; E[X(f1)X(f2)X*(f1+f2)].
        C = (np.conj(Q3).T @ (self.W * prod)) / nblocks  # (Nblocks, Nblocks)

        try:
            # BLAS limit is applied once around the triad loop (serial or
            # ThreadPoolExecutor), never here: threadpool_limits is process-
            # global, so a worker exit would un-pin a sibling still in eig.
            eigvals, eigvecs = np.linalg.eig(C)
            dom = np.argmax(np.abs(eigvals))
            a = eigvecs[:, dom]
            # Fix LAPACK phase on the shared coefficients only. mode1 and mode2
            # then inherit the same unit factor, so their relative phase is kept.
            a_col, _ = canonicalize_modes(a.reshape(-1, 1))
            a = a_col[:, 0]

            mode1 = Q3 @ a
            mode2 = prod @ a
            return eigvals[dom], mode1, mode2
        except np.linalg.LinAlgError:
            return np.nan, None, None

    def _get_algorithm_metadata(self) -> dict:
        """Describe the current BSMD approximation contract."""
        return {
            "lift_kind": "triadic_spectral_product",
            "bsmd_solver": "dominant_eigenpair_approximation",
            "bsmd_target_objective": "numerical_radius",
            "uses_spatial_metric_in_cross_operator": True,
            "uses_shared_triadic_coefficients": True,
            "bispectrum_conjugation": "sum_frequency_conjugated",
        }

    # Core logic for BSMD with statically defined triads.
    def _perform_static_bsmd_core(self) -> None:
        """
        Perform BSMD for a statically defined list of frequency triads.

        When ``self.use_parallel`` is True, triads are processed concurrently
        using a thread pool.  NumPy releases the GIL during BLAS calls, and
        Python 3.14+ free-threading removes it entirely, so threads give
        near-linear speedup for the matmul-dominated inner loop.
        """
        logger.info("Performing static BSMD core analysis...")
        start_time = time.time()
        if not self.static_triads_list or len(self.static_triads_list) == 0:
            logger.error("Static triads list is empty. Cannot perform static BSMD.")
            self.modes1 = np.array([])
            self.modes2 = np.array([])
            self.eigenvalues = np.array([])
            self.triads = np.array([])
            return

        # Once per analysis (not per triad): refuse a metric that is not an
        # inner product before any eigenproblem is formed.
        require_spatial_metric(self.W)

        # Reject triads outside the analysable range before any analysis.
        # Bound by both the physical rfft limit (nfft//2) and the bins actually
        # loaded in qhat (stale/truncated cache can be shorter than rfft).
        # Default list: warn and drop out-of-range triads (a default must not fail
        # a default configuration). User-supplied list: collect every offender
        # and raise once naming them all.
        nfft_limit = self.nfft // 2
        n_loaded = self._n_freq_bins
        loaded_limit = n_loaded - 1
        max_bin = min(nfft_limit, loaded_limit)

        def _out_of_range_kind(p_int: int) -> str | None:
            if abs(p_int) <= max_bin:
                return None
            if abs(p_int) > nfft_limit:
                return "rfft"
            return "loaded"

        # Still the default only while the list object is the one construction
        # (or a prior filter step) resolved. A post-construction assignment is
        # a new object and is treated as user-supplied — even if its values
        # happen to equal a subset of ALL_TRIADS.
        from_default = (
            self._static_triads_from_default
            and self._resolved_default_triads is not None
            and self.static_triads_list is self._resolved_default_triads
        )

        if from_default:
            kept = []
            dropped = []
            for triad in self.static_triads_list:
                bad = [int(p) for p in triad if _out_of_range_kind(int(p)) is not None]
                if bad:
                    dropped.append(tuple(int(p) for p in triad))
                else:
                    kept.append(triad)
            if dropped:
                # Prefer the tighter bound in the warning so the user sees why.
                if n_loaded == 0:
                    bound_note = "no frequency bins are loaded"
                else:
                    bound_note = (
                        f"|p| must be <= {max_bin} (nfft//2 = {nfft_limit}, loaded bins allow |p| <= {loaded_limit})"
                    )
                dropped_str = ", ".join(repr(t) for t in dropped)
                warnings.warn(
                    f"Default static triads filtered for nfft={self.nfft} "
                    f"({bound_note}): dropped {len(dropped)} triad(s) "
                    f"outside range: {dropped_str}",
                    UserWarning,
                    stacklevel=2,
                )
                # Post-filter list is still the default's: update the resolved
                # object so a later re-run does not treat `kept` as user input.
                self.static_triads_list = kept
                self._resolved_default_triads = kept
            if not self.static_triads_list:
                # Filtering emptied the list — refuse before any executor is built.
                named = ", ".join(repr(t) for t in dropped) if dropped else "(none)"
                if n_loaded == 0:
                    bound_phrase = "no frequency bins are loaded"
                else:
                    bound_phrase = f"|p| must be <= {max_bin}"
                raise ValueError(
                    f"No statically defined triads remain after filtering for "
                    f"nfft={self.nfft} ({bound_phrase}): dropped all "
                    f"triad(s) outside range: {named}. "
                    f"This configuration cannot be analysed."
                )
        else:
            rfft_offenders = []
            loaded_offenders = []
            seen_rfft = set()
            seen_loaded = set()
            for triad in self.static_triads_list:
                for p in triad:
                    p_int = int(p)
                    kind = _out_of_range_kind(p_int)
                    if kind == "rfft" and p_int not in seen_rfft:
                        seen_rfft.add(p_int)
                        rfft_offenders.append(p_int)
                    elif kind == "loaded" and p_int not in seen_loaded:
                        seen_loaded.add(p_int)
                        loaded_offenders.append(p_int)
            if rfft_offenders or loaded_offenders:
                parts = []
                if rfft_offenders:
                    named = ", ".join(f"p={p}" for p in rfft_offenders)
                    parts.append(
                        f"Triad component(s) {named} outside the rfft bin range "
                        f"(|p| must be <= nfft//2 = {nfft_limit}; "
                        f"{nfft_limit + 1} bins for nfft={self.nfft})"
                    )
                if loaded_offenders:
                    named = ", ".join(f"p={p}" for p in loaded_offenders)
                    if n_loaded == 0:
                        parts.append(
                            f"Triad component(s) {named} outside the loaded frequency range "
                            f"(no frequency bins are loaded; "
                            f"rfft would allow up to nfft//2 = {nfft_limit})"
                        )
                    else:
                        parts.append(
                            f"Triad component(s) {named} outside the loaded frequency range "
                            f"(only {n_loaded} bins are loaded, so |p| must be <= {loaded_limit}; "
                            f"rfft would allow up to nfft//2 = {nfft_limit})"
                        )
                raise ValueError("; ".join(parts))

        num_triads = len(self.static_triads_list)
        Nspace = self._n_spatial
        logger.info("Using %d statically defined triads (%d spatial points).", num_triads, Nspace)

        self.modes1 = np.zeros((num_triads, Nspace), dtype=complex)
        self.modes2 = np.zeros((num_triads, Nspace), dtype=complex)
        self.eigenvalues = np.zeros(num_triads, dtype=complex)
        self.triads = np.array(self.static_triads_list)

        # Ensure freq/St arrays are set (needed for post-analysis plotting, not for the core loop)
        if self.freq is None or self.St is None:
            n_freq = self._n_freq_bins
            if n_freq > 0:
                freq = np.fft.rfftfreq(n_freq * 2 - 2, d=1.0 / self._require_fs())[:n_freq]
                self.freq = freq
                self.St = freq.copy()

        # Pre-fetch frequency bins from HDF5 into RAM cache before threading.
        # In disk-backed mode this avoids h5py reads inside threads (not thread-safe).
        # In RAM mode this is a no-op.
        self._prefetch_bins()

        def _store_result(i: int, lam: complex | float, m1: np.ndarray | None, m2: np.ndarray | None) -> None:
            """Write one triad's results into the pre-allocated arrays."""
            self.eigenvalues[i] = lam
            if m1 is not None:
                self.modes1[i, :] = m1
                self.modes2[i, :] = m2
            else:
                self.modes1[i, :] = np.nan
                self.modes2[i, :] = np.nan

        # One limiter spans the whole triad loop (parallel or serial). Do not
        # enter/exit a process-global limiter from workers.
        with apply_blas_limit():
            if self.use_parallel:
                n_workers = min(num_triads, os.cpu_count() or 1)
                logger.info("Thread-parallel BSMD with %d workers.", n_workers)
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(self._compute_single_triad, p1, p2, p3): i
                        for i, (p1, p2, p3) in enumerate(self.static_triads_list)
                    }
                    for future in tqdm(as_completed(futures), total=num_triads, desc="BSMD Triads"):
                        i = futures[future]
                        lam, m1, m2 = future.result()
                        _store_result(i, lam, m1, m2)
            else:
                for i, (p1, p2, p3) in enumerate(tqdm(self.static_triads_list, desc="BSMD Triads")):
                    lam, m1, m2 = self._compute_single_triad(p1, p2, p3)
                    _store_result(i, lam, m1, m2)

        logger.info("Static BSMD core analysis completed in %.2f seconds.", time.time() - start_time)

        # Build energy map for quick visualisation
        self.energy_map = self._compute_energy_map()

    def perform_dynamic_bsmd(self) -> None:
        """
        Perform BSMD with dynamically identified triads (Placeholder).

        This method is intended for future implementation where significant triads
        are identified from the data (e.g., based on bispectrum peaks) rather than
        being predefined.

        Currently, this method will raise a NotImplementedError.
        """
        raise NotImplementedError("Dynamic BSMD is not yet implemented.")

    def _compute_energy_map(self) -> np.ndarray:
        """Return a 2D map of eigenvalue magnitudes indexed by (p1,p2).

        Grid half-width is derived from the triads actually analysed
        (``p_max = max |p1|, |p2|``), so every analysed triad lands on the map.
        Default ``ALL_TRIADS`` (|p| <= 8) still yields a 17x17 grid.
        """
        if self.eigenvalues.size == 0:
            return np.array([])

        p_max = 0
        for p1, p2, _p3 in self.triads:
            p_max = max(p_max, abs(int(p1)), abs(int(p2)))
        offset = p_max
        size = 2 * offset + 1
        grid = np.full((size, size), np.nan)
        for val, (p1, p2, _p3) in zip(np.abs(self.eigenvalues), self.triads):
            i = int(p1) + offset
            j = int(p2) + offset
            if 0 <= i < size and 0 <= j < size:
                grid[i, j] = val
        return grid

    # Save triads, eigenvalues, modes, and weights to HDF5.
    def save_results(self, filename: str | None = None) -> None:
        """Save BSMD results (triads, eigenvalues, modes) to an HDF5 file.

        When the destination is the same path as the open FFT cache, the write
        opens in append mode so ``FFTBlocks`` is preserved. Otherwise it
        overwrites with mode ``"w"``.
        """
        from openmodalpy.core.results import write_results

        if filename is None:
            results_path = os.path.join(
                self.results_dir,
                make_result_filename(self.data_root, self.nfft, self.overlap, self.data["Ns"], "bsmd"),
            )
        else:
            results_path = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        qhat_cache_path = self._qhat_cache_path
        using_cache_file = qhat_cache_path is not None and os.path.abspath(results_path) == os.path.abspath(
            qhat_cache_path
        )
        if using_cache_file and self._qhat_file is not None:
            # The FFT cache may already hold an open handle to this same path.
            # Close it before updating the file in append mode.
            self._qhat_file.close()
            self._qhat_file = None
            self._qhat_dataset = None

        file_mode = _hdf5_write_mode(results_path) if using_cache_file else "w"
        datasets: dict = {
            "triads": np.array(self.triads),
            "eigenvalues": self.eigenvalues,
            "modes1": self.modes1,
            "modes2": self.modes2,
            "x": self.data["x"],
            "y": self.data["y"],
            "W": self.W,
        }
        if "z" in self.data and self.data["z"] is not None:
            datasets["z"] = self.data["z"]
        if self.energy_map.size:
            datasets["energy_map"] = self.energy_map
        write_results(results_path, datasets, attrs=self._get_metadata(), mode=file_mode, compression=None)
        logger.info("Results saved to %s", results_path)

    def load_results(self, filename: str | None = None) -> None:
        """Load BSMD results from an HDF5 file."""
        from openmodalpy.core.results import read_results

        if filename is None:
            load_path = os.path.join(
                self.results_dir,
                make_result_filename(self.data_root, self.nfft, self.overlap, self.data.get("Ns", 0), "bsmd"),
            )
        else:
            load_path = os.path.join(self.results_dir, filename)
        logger.info("Loading BSMD results from %s", load_path)
        if not os.path.isfile(load_path):
            from openmodalpy.core.results import find_latest_result

            latest = find_latest_result(self.results_dir, "*_bsmd.hdf5")
            if latest:
                load_path = latest
                logger.info("[Auto-detect] Using: %s", load_path)
            else:
                logger.error("No BSMD results file found in %s", self.results_dir)
                return
        res = read_results(load_path)
        stamp = res.attrs.get("bispectrum_conjugation")
        if stamp != "sum_frequency_conjugated":
            raise ValueError(
                f"{load_path} was written by a pre-fix BSMD build in which the "
                "sum-frequency term was not conjugated; its eigenvalues and modes "
                "are invalid. Recompute from the raw data."
            )
        self.triads = res.triads if res.triads is not None else np.array([])
        self.eigenvalues = res.eigenvalues if res.eigenvalues is not None else np.array([])
        self.modes1 = res.modes1 if res.modes1 is not None else np.array([])
        self.modes2 = res.modes2 if res.modes2 is not None else np.array([])
        if res.energy_map is not None:
            self.energy_map = res.energy_map
        if res.W is not None:
            self.W = _as_spatial_weight_column(res.W)
        for coord_key in ("x", "y", "z"):
            value = getattr(res, coord_key, None)
            if value is not None:
                self.data[coord_key] = value
        for attr_key in ("dt", "Ns", "Nx", "Ny", "Nz", "nfft", "overlap"):
            if attr_key in res.attrs:
                self.data[attr_key] = res.attrs[attr_key]
        logger.info("BSMD results loaded.")

    def plot_modes(self, triad_indices: Sequence[int] | None = None, plot_n_modes: Optional[int] = 10) -> None:
        """Plot spatial BSMD modes for selected triads."""
        if self.modes1.size == 0 or self.modes2.size == 0:
            logger.warning("No BSMD modes to plot. Run perform_bsmd() first.")
            return
        if resolve_volume_layout(self.data, self.modes1.shape[1]) is not None:
            self.plot_modes_3d_slices(triad_indices=triad_indices, plot_n_modes=plot_n_modes)
            return

        if triad_indices is None:
            lambdas = np.abs(self.eigenvalues)
            valid = ~np.isnan(lambdas)
            triad_indices = list(np.argsort(lambdas[valid])[::-1])
            # Map back to original indices (skip NaN triads)
            valid_idx = np.where(valid)[0]
            triad_indices = [int(valid_idx[k]) for k in triad_indices]
        if plot_n_modes is not None:
            triad_indices = triad_indices[:plot_n_modes]

        nx = self.data.get("Nx", int(np.sqrt(self.modes1.shape[1])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes1.shape[1])))
        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        # Pre-compute mesh once (outside the loop)
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
        else:
            x_mesh, y_mesh = x_coords, y_coords

        # Figure sizing: wide domains → side-by-side, tall/square → stacked
        # Each panel's plot area targets ~5" on its long side; the colorbar
        # and labels add ~1.5" per panel.
        if fig_aspect >= 2.0:
            # Wide domain (e.g. cavity): 1×2 layout, height from aspect
            nrows, ncols = 1, 2
            plot_w = 6.0
            plot_h = max(plot_w / fig_aspect, 2.5)
            fig_w = 2 * (plot_w + 1.5)
            fig_h = plot_h + 1.5
        else:
            # Square-ish or tall domain (e.g. jet): 2×1 layout for bigger panels
            nrows, ncols = 2, 1
            plot_w = 7.0
            plot_h = plot_w / fig_aspect
            fig_w = plot_w + 2.0
            fig_h = 2 * plot_h + 2.0

        for idx in triad_indices:
            mode1 = self.modes1[idx, :].real.reshape(nx, ny)
            mode2 = self.modes2[idx, :].real.reshape(nx, ny)
            triad = tuple(int(v) for v in self.triads[idx])
            lam = self.eigenvalues[idx]

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_w, fig_h),
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            fig.suptitle(
                f"Triad ({triad[0]}, {triad[1]}, {triad[2]})   "
                rf"$|\lambda|$ = {np.abs(lam):.3e}",
                fontsize=12,
            )

            for ax, mode, label in [(axes[0], mode1, r"$\Phi_1$"), (axes[1], mode2, r"$\Phi_2$")]:
                vmax = np.max(np.abs(mode))
                if vmax == 0:
                    vmax = 1.0
                levels = np.linspace(-vmax, vmax, 21)
                cf = ax.contourf(
                    x_mesh,
                    y_mesh,
                    mode,
                    levels=levels,
                    cmap=CMAP_DIV,
                    extend="both",
                )
                ax.contour(
                    x_mesh,
                    y_mesh,
                    mode,
                    levels=levels[::4],
                    colors="k",
                    linewidths=0.5,
                    alpha=0.5,
                )
                ax.set_title(f"{label} [{var_name}]")
                style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                add_inset_colorbar(
                    fig,
                    ax,
                    cf,
                    self.data,
                    ticks=[-vmax, 0, vmax],
                    ticklabels=[f"{-vmax:.2f}", "0", f"{vmax:.2f}"],
                )

            fname = os.path.join(self.figures_dir, f"{self.data_root}_BSMD_triad{idx}_{var_name}.png")
            plt.savefig(fname, dpi=FIG_DPI)
            plt.close(fig)
            logger.info("BSMD mode plot saved to %s", fname)

    def plot_modes_3d_slices(
        self, triad_indices: Sequence[int] | None = None, plot_n_modes: Optional[int] = 10
    ) -> None:
        """Plot orthogonal 3D slices for selected BSMD triads."""
        self._plot_modes_3d("slices", triad_indices=triad_indices, plot_n_modes=plot_n_modes)

    def plot_modes_3d_isometric(
        self, triad_indices: Sequence[int] | None = None, plot_n_modes: Optional[int] = 10
    ) -> None:
        """Plot 3D isosurfaces for selected BSMD triads."""
        self._plot_modes_3d("isometric", triad_indices=triad_indices, plot_n_modes=plot_n_modes)

    def _plot_modes_3d(
        self,
        kind: str,
        triad_indices: Sequence[int] | None = None,
        plot_n_modes: Optional[int] = 10,
    ) -> None:
        if self.modes1.size == 0 or self.modes2.size == 0:
            logger.warning("No BSMD modes to plot. Run perform_bsmd() first.")
            return
        if resolve_volume_layout(self.data, self.modes1.shape[1]) is None:
            logger.warning("plot_modes_3d_%s requires volumetric data.", kind)
            return
        if triad_indices is None:
            lambdas = np.abs(self.eigenvalues)
            valid = ~np.isnan(lambdas)
            triad_indices = list(np.argsort(lambdas[valid])[::-1])
            valid_idx = np.where(valid)[0]
            triad_indices = [int(valid_idx[k]) for k in triad_indices]
        if plot_n_modes is not None:
            triad_indices = triad_indices[:plot_n_modes]
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        items = []
        for idx in triad_indices:
            triad = tuple(int(v) for v in self.triads[idx])
            for label, mode_arr in (("phi1", self.modes1[idx, :].real), ("phi2", self.modes2[idx, :].real)):
                mode_3d = reshape_mode_to_volume(mode_arr, self.data)
                output_path = os.path.join(self.figures_dir, f"{self.data_root}_BSMD_triad{idx}_{label}_{kind}.png")
                items.append(
                    {
                        "mode_3d": mode_3d,
                        "output_path": output_path,
                        "title_prefix": f"BSMD {label} | triad={triad}",
                        "scalar_name": f"bsmd_{label}",
                    }
                )
        plot_modes_3d(kind, items, x_coords, y_coords, z_coords, data=self.data)

    def plot_energy_map(self) -> None:
        """Plot a 2D heatmap of eigenvalue magnitudes indexed by triad frequencies."""
        if self.energy_map.size == 0:
            logger.warning("No energy map available. Run perform_bsmd() first.")
            return

        offset = (self.energy_map.shape[0] - 1) // 2
        extent = (-offset - 0.5, offset + 0.5, -offset - 0.5, offset + 0.5)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            self.energy_map,
            origin="lower",
            extent=extent,
            cmap=CMAP_SEQ,
            aspect="equal",
        )
        ax.set_xlabel("p2 index")
        ax.set_ylabel("p1 index")
        ax.set_title("BSMD energy map |lambda|")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fname = os.path.join(self.figures_dir, f"{self.data_root}_BSMD_energy_map.png")
        plt.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Energy map saved to %s", fname)

    # Execute the full BSMD pipeline.
    def run_analysis(self) -> None:
        """
        Execute the full BSMD analysis pipeline.

        This method orchestrates the entire BSMD process:
        1. Loads and preprocesses data, including STFT computation (calls `load_and_preprocess`).
           This step sets `self.qhat`, `self.W`, `self.freq`, `self.fs`, etc.
        2. Performs BSMD computation (calls `perform_bsmd`), which internally chooses
           between static or dynamic triad analysis (currently static is implemented).
           This step sets `self.modes1`, `self.modes2`, `self.eigenvalues`, `self.triads`.
        3. Saves the results to an HDF5 file (calls `save_results`).

        This is the primary method to call to run a complete BSMD study on a dataset.
        """
        logger.info("Starting BSMD analysis for %s", os.path.basename(self.file_path))
        start_total_time = time.time()
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_bsmd()
        self.save_results()
        self.close()  # Release disk-backed resources if any
        logger.info("Total BSMD runtime: %.2f s", time.time() - start_total_time)
        print_summary("BSMD", self.results_dir, self.figures_dir)
