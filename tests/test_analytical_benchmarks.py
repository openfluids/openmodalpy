"""
Analytical benchmark tests for OpenModalPy.

These tests use synthetic data with KNOWN mathematical properties to verify
that the decomposition methods produce correct results - not just "runs without
crashing" but "the math is right".

Key principle: If we construct data with known structure, we can verify the
decomposition recovers that exact structure.
"""

import numpy as np
import pytest

from openmodalpy import DMDAnalyzer, PODAnalyzer, STPODAnalyzer


def _weighted_centered_trace(q: np.ndarray, weights: np.ndarray) -> float:
    """Closed-form POD eigenvalue sum under snapshot averaging.

    For mean-centred snapshots ``Q_c`` and diagonal metric ``W``:

        sum_i λ_i  ==  (1/N) * sum(Q_c**2 * W)

    Independent of the eigensolver; a kernel rescaled by ``N/(N-1)`` breaks it.
    """
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    q_c = q - np.mean(q, axis=0, dtype=np.float64)
    return float(np.sum(q_c * q_c * w) / q.shape[0])


def _assert_pod_spectrum_matches_trace(analyzer: PODAnalyzer, *, rtol: float = 1e-10) -> None:
    """Retained sum(λ) must match the closed-form trace times energy_captured_fraction."""
    expected_total = _weighted_centered_trace(analyzer.data["q"], analyzer.W)
    retained = float(np.sum(analyzer.eigenvalues))
    frac = float(analyzer.energy_captured_fraction)
    expected_retained = expected_total * frac
    rel = abs(retained - expected_retained) / max(abs(expected_total), 1e-30)
    assert rel <= rtol, (
        f"POD eigenvalue sum vs weighted centered trace: "
        f"sum(λ)={retained!r}, expected_retained={expected_retained!r} "
        f"(trace={expected_total!r} × energy_captured_fraction={frac!r}), "
        f"rel_err={rel:.3e} (rtol={rtol})"
    )


# =============================================================================
# POD Analytical Benchmarks
# =============================================================================


class TestPODAnalytical:
    """POD tests with known mathematical solutions."""

    def test_pod_rank2_captures_100_percent_energy(self):
        """Exact rank-2 data should have 100% energy in exactly 2 modes.

        Mathematical basis:
        - Data = σ₁(u₁ ⊗ v₁) + σ₂(u₂ ⊗ v₂) has rank 2
        - POD (SVD) should recover exactly 2 nonzero singular values
        - Energy in modes 1-2 should equal total energy
        - Absolute scale: sum_i λ_i = (1/N) sum(Q_c**2 * W) (snapshot-averaged Gram)
        """
        Ns, Nx = 100, 50

        # Construct rank-2 data: two orthogonal spatial patterns with time variation
        t = np.linspace(0, 4 * np.pi, Ns)
        x = np.linspace(0, 1, Nx)

        # Mode 1: sin(t) * sin(πx) with amplitude 1.0
        mode1_time = np.sin(t)
        mode1_space = np.sin(np.pi * x)

        # Mode 2: cos(2t) * sin(2πx) with amplitude 0.5
        mode2_time = np.cos(2 * t)
        mode2_space = np.sin(2 * np.pi * x)

        # Data matrix: (Ns, Nspace)
        data_matrix = np.outer(mode1_time, mode1_space) + 0.5 * np.outer(mode2_time, mode2_space)

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": t[1] - t[0],
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = PODAnalyzer(
            file_path="analytical_rank2",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

        # Verify: first 2 modes capture essentially all energy
        total_energy = np.sum(analyzer.eigenvalues)
        energy_in_2_modes = np.sum(analyzer.eigenvalues[:2])
        energy_fraction = energy_in_2_modes / total_energy

        assert energy_fraction > 0.9999, (
            f"Rank-2 data should have >99.99% energy in 2 modes, got {energy_fraction * 100:.4f}%"
        )

        # Verify: mode 3+ eigenvalues are essentially zero (numerical noise)
        if len(analyzer.eigenvalues) > 2:
            assert analyzer.eigenvalues[2] / analyzer.eigenvalues[0] < 1e-10, (
                "Mode 3 should have negligible energy for rank-2 data"
            )

        # Absolute magnitude (scale-sensitive): full retained spectrum matches the
        # independent weighted centered trace — not just a relative energy share.
        _assert_pod_spectrum_matches_trace(analyzer)

    def test_pod_reconstruction_error_decreases(self):
        """Reconstruction error must monotonically decrease with more modes.

        Mathematical basis:
        - POD provides optimal (in L2 sense) low-rank approximation
        - Adding more modes can only reduce or maintain error, never increase
        - Absolute scale: retained sum(λ) = energy_captured_fraction × (1/N) sum(Q_c² W)
        """
        Ns, Nx = 50, 30

        # Create data with decaying mode amplitudes
        t = np.linspace(0, 2 * np.pi, Ns)
        x = np.linspace(0, 1, Nx)

        data_matrix = np.zeros((Ns, Nx))
        for k in range(1, 6):  # 5 modes with decaying amplitude
            amplitude = 1.0 / k
            data_matrix += amplitude * np.outer(np.sin(k * t), np.sin(k * np.pi * x))

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": t[1] - t[0],
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = PODAnalyzer(
            file_path="analytical_recon",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

        # Compute reconstruction errors for increasing number of modes
        data_centered = data_matrix - analyzer.temporal_mean
        errors = []
        for n_modes in range(1, min(6, analyzer.modes.shape[1] + 1)):
            # modes: (Nspace, n_modes), time_coefficients: (Ns, n_modes)
            # reconstruction: modes @ time_coefficients.T = (Nspace, Ns)
            reconstructed = analyzer.modes[:, :n_modes] @ analyzer.time_coefficients[:, :n_modes].T
            error = np.linalg.norm(data_centered - reconstructed.T) / np.linalg.norm(data_centered)
            errors.append(error)

        # Verify monotonic decrease
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-10, f"Error should decrease: {errors[i]:.6f} -> {errors[i + 1]:.6f}"

        _assert_pod_spectrum_matches_trace(analyzer)

    def test_pod_modes_orthonormal(self):
        """POD spatial modes must be orthonormal with respect to weights.

        Mathematical basis:
        - POD modes Φ satisfy: Φᵀ W Φ = I (identity matrix)
        - This is a fundamental property of the decomposition
        - Absolute scale: sum_i λ_i = (1/N) sum(Q_c**2 * W) for the retained share
        """
        rng = np.random.default_rng(456)
        Ns, Nx = 40, 25

        # Random data (to test general case, not just special structure)
        data_matrix = rng.standard_normal((Ns, Nx))

        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = PODAnalyzer(
            file_path="analytical_ortho",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

        # Check orthonormality: Φᵀ W Φ ≈ I
        n_modes = analyzer.modes.shape[1]
        W = analyzer.W.flatten() if analyzer.W.ndim > 1 else analyzer.W
        W_diag = np.diag(W)

        gram_matrix = analyzer.modes.T @ W_diag @ analyzer.modes
        identity = np.eye(n_modes)

        assert np.allclose(gram_matrix, identity, atol=1e-10), (
            f"Modes not orthonormal. Max deviation: {np.max(np.abs(gram_matrix - identity)):.2e}"
        )

        _assert_pod_spectrum_matches_trace(analyzer)

    def test_pod_eigenvalues_nonnegative_ordered(self):
        """POD eigenvalues must be non-negative and sorted in descending order.

        Mathematical basis:
        - Eigenvalues represent energy (variance) in each mode
        - Energy cannot be negative
        - POD convention: modes ordered by decreasing energy
        - Absolute scale: sum_i λ_i = energy_captured_fraction × (1/N) sum(Q_c² W)
        """
        rng = np.random.default_rng(789)
        Ns, Nx = 60, 40

        data_matrix = rng.standard_normal((Ns, Nx))

        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = PODAnalyzer(
            file_path="analytical_eig",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=20,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

        # All eigenvalues non-negative
        assert np.all(analyzer.eigenvalues >= -1e-14), (
            f"Eigenvalues must be non-negative, got min: {np.min(analyzer.eigenvalues)}"
        )

        # Eigenvalues in descending order
        for i in range(len(analyzer.eigenvalues) - 1):
            assert analyzer.eigenvalues[i] >= analyzer.eigenvalues[i + 1] - 1e-14, (
                f"Eigenvalues not sorted: λ[{i}]={analyzer.eigenvalues[i]} < λ[{i + 1}]={analyzer.eigenvalues[i + 1]}"
            )

        _assert_pod_spectrum_matches_trace(analyzer)

    def test_pod_eigenvalue_sum_matches_weighted_centered_trace(self):
        """Full POD spectrum sum equals (1/N) sum(Q_c² W) on random data.

        Closed form for the snapshot-averaged weighted Gram kernel (temporal or
        spatial path — same trace):

            sum_i λ_i  =  (1/N) * sum_n ||q_c[n]||_W^2
                       =  (1/N) * sum(Q_c**2 * W)

        with Q_c = Q − mean_t(Q) and W the spatial metric. ``n_modes_save`` is
        the full spatial rank so the retained sum is the whole trace. Relative
        energy shares and mode orthonormality are scale-invariant; this pins
        magnitude.
        """
        rng = np.random.default_rng(101)
        Ns, Nx = 45, 28
        data_matrix = rng.standard_normal((Ns, Nx))
        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = PODAnalyzer(
            file_path="analytical_trace",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=Nx,  # full spatial rank
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()
        assert analyzer.energy_captured_fraction > 1.0 - 1e-12
        _assert_pod_spectrum_matches_trace(analyzer)


# =============================================================================
# DMD Analytical Benchmarks
# =============================================================================


class TestDMDAnalytical:
    """DMD tests with known mathematical solutions."""

    def test_dmd_recovers_oscillation_frequency(self):
        """DMD should recover the exact frequency of a traveling wave.

        Mathematical basis:
        - Traveling wave cos(kx - ωt) has DMD eigenvalue λ = e^(iω·dt)
        - Frequency f = arg(λ) / (2π·dt)

        Note: We use a traveling wave (not standing wave) because DMD needs
        spatially-varying dynamics, not just temporal scaling of a fixed pattern.
        """
        dt = 0.01
        f_true = 5.0  # Hz
        omega = 2 * np.pi * f_true
        k = 2  # wavenumber
        Ns = 200
        Nx = 50

        t = np.arange(Ns) * dt
        x = np.linspace(0, 2 * np.pi, Nx)

        # Traveling wave: cos(kx - ωt)
        data_matrix = np.cos(k * x[None, :] - omega * t[:, None])

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": dt,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = DMDAnalyzer(
            file_path="analytical_dmd_freq",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
            rank=10,
        )
        analyzer.load_and_preprocess()
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()

        # Extract frequencies from eigenvalues
        # DMD eigenvalue λ relates to frequency via: f = arg(λ) / (2π·dt)
        frequencies = np.abs(np.angle(analyzer.eigenvalues)) / (2 * np.pi * dt)

        # Should find the true frequency
        freq_errors = np.abs(frequencies - f_true)
        min_error = np.min(freq_errors)

        assert min_error < 0.1, (
            f"DMD should recover f={f_true} Hz, closest found: {frequencies[np.argmin(freq_errors)]:.2f} Hz"
        )

    def test_dmd_exponential_growth_rate(self):
        """DMD should recover the growth rate of exponentially growing data.

        Mathematical basis:
        - Data x(t) = e^(σt) has DMD eigenvalue λ = e^(σ·dt)
        - Growth rate σ = log|λ| / dt
        """
        dt = 0.1
        sigma_true = 0.5  # growth rate
        Ns = 50
        Nx = 15

        t = np.arange(Ns) * dt
        x = np.linspace(0, 1, Nx)

        # Exponentially growing signal
        spatial_pattern = np.sin(np.pi * x)
        data_matrix = np.outer(np.exp(sigma_true * t), spatial_pattern)

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": dt,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = DMDAnalyzer(
            file_path="analytical_dmd_growth",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=5,
            rank=5,
        )
        analyzer.load_and_preprocess()
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()

        # Extract growth rates: σ = log|λ| / dt
        growth_rates = np.log(np.abs(analyzer.eigenvalues)) / dt

        # Find the dominant mode (largest eigenvalue magnitude)
        dominant_idx = np.argmax(np.abs(analyzer.eigenvalues))
        recovered_sigma = growth_rates[dominant_idx]

        assert np.abs(recovered_sigma - sigma_true) < 0.05, (
            f"DMD should recover σ={sigma_true}, got σ={recovered_sigma:.3f}"
        )

    def test_dmd_unit_circle_for_periodic(self):
        """Purely periodic (non-decaying) signals should have eigenvalues on unit circle.

        Mathematical basis:
        - No growth/decay means |λ| = 1
        - λ = e^(iω·dt) lies exactly on the unit circle

        Note: We use traveling waves (spatially-varying dynamics) so DMD can
        properly identify the oscillatory modes.
        """
        dt = 0.02
        Ns = 100
        Nx = 50

        t = np.arange(Ns) * dt
        x = np.linspace(0, 2 * np.pi, Nx)

        # Two traveling waves with different frequencies
        f1, f2 = 3.0, 7.0
        k1, k2 = 1, 2
        data_matrix = np.cos(k1 * x[None, :] - 2 * np.pi * f1 * t[:, None]) + 0.5 * np.cos(
            k2 * x[None, :] - 2 * np.pi * f2 * t[:, None]
        )

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": dt,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = DMDAnalyzer(
            file_path="analytical_dmd_periodic",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
            rank=10,
        )
        analyzer.load_and_preprocess()
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()

        # All eigenvalues should have magnitude ≈ 1 for periodic data
        magnitudes = np.abs(analyzer.eigenvalues)

        # At least the dominant modes should be on/near unit circle
        dominant_mags = np.sort(magnitudes)[-4:]  # top 4 eigenvalues
        for mag in dominant_mags:
            assert np.abs(mag - 1.0) < 0.1, f"Periodic data should have |λ|≈1, got |λ|={mag:.3f}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases that might break the algorithms."""

    def test_pod_single_snapshot_raises(self):
        """POD should raise ValueError for single snapshot (needs at least 2)."""
        data = {
            "q": np.array([[1.0, 2.0, 3.0]]),  # 1 snapshot, 3 spatial points
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0]),
            "dt": 1.0,
            "Nx": 3,
            "Ny": 1,
            "Ns": 1,
        }

        analyzer = PODAnalyzer(
            file_path="edge_single",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=1,
        )
        analyzer.load_and_preprocess()

        # POD requires at least 2 snapshots
        with pytest.raises(ValueError, match="at least 2 snapshots"):
            analyzer.perform_pod()

    def test_pod_constant_data(self):
        """POD on constant data should have zero variance modes."""
        Ns, Nx = 20, 10
        constant_value = 5.0
        data_matrix = np.full((Ns, Nx), constant_value)

        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = PODAnalyzer(
            file_path="edge_constant",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=5,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

        # After mean subtraction, data is all zeros -> zero eigenvalues
        assert np.allclose(analyzer.eigenvalues, 0, atol=1e-10), (
            "Constant data should have zero eigenvalues after mean subtraction"
        )

    def test_dmd_two_snapshots(self):
        """DMD with minimum viable data (2 snapshots)."""
        data = {
            "q": np.array([[1.0, 2.0], [2.0, 4.0]]),  # 2 snapshots
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0]),
            "dt": 1.0,
            "Nx": 2,
            "Ny": 1,
            "Ns": 2,
        }

        analyzer = DMDAnalyzer(
            file_path="edge_two_snaps",
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=2,
            rank=2,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_dmd()

        # Should produce at least 1 mode
        assert analyzer.modes.shape[1] >= 1


# =============================================================================
# ST-POD Analytical Benchmarks
# =============================================================================


class TestSTPODAnalytical:
    """ST-POD tests with known mathematical solutions."""

    def test_stpod_traveling_wave_rank(self):
        """Traveling wave should have bounded rank in Hankel matrix.

        Mathematical basis:
        - A traveling wave cos(kx - ωt) can be written as:
          cos(kx)cos(ωt) + sin(kx)sin(ωt)
        - This is rank-2 in standard POD
        - In ST-POD with embedding dimension d, the Hankel matrix captures
          d consecutive time steps, but the underlying dynamics still have
          bounded rank related to the number of independent oscillations.
        """
        dt = 0.02
        f = 5.0
        omega = 2 * np.pi * f
        k = 2
        Ns = 100
        Nx = 40

        t = np.arange(Ns) * dt
        x = np.linspace(0, 2 * np.pi, Nx)

        # Traveling wave
        data_matrix = np.cos(k * x[None, :] - omega * t[:, None])

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": dt,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        embedding_dim = 10
        analyzer = STPODAnalyzer(
            file_path="analytical_stpod_wave",
            embedding_dim=embedding_dim,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=20,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # The first few modes should capture essentially all energy
        total_energy = np.sum(analyzer.eigenvalues)
        energy_first_4 = np.sum(analyzer.eigenvalues[:4])
        energy_fraction = energy_first_4 / total_energy

        assert energy_fraction > 0.99, (
            f"Traveling wave should have >99% energy in first 4 modes, got {energy_fraction * 100:.2f}%"
        )

    def test_stpod_reconstruction_error_decreases(self):
        """Reconstruction error must decrease with more modes.

        Mathematical basis:
        - Adding more SVD modes can only reduce or maintain approximation error
        - This is a fundamental property of SVD truncation
        """
        Ns, Nx = 50, 30

        # Multi-mode signal
        t = np.linspace(0, 4 * np.pi, Ns)
        x = np.linspace(0, 1, Nx)

        data_matrix = np.zeros((Ns, Nx))
        for k in range(1, 5):
            data_matrix += (1.0 / k) * np.outer(np.sin(k * t), np.sin(k * np.pi * x))

        data = {
            "q": data_matrix,
            "x": x,
            "y": np.array([0.0]),
            "dt": t[1] - t[0],
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        embedding_dim = 8
        analyzer = STPODAnalyzer(
            file_path="analytical_stpod_recon",
            embedding_dim=embedding_dim,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=15,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Reconstruct Hankel matrix with increasing modes
        # The SVD gives us: H_weighted = U @ diag(sigma) @ Vt
        # After unweighting modes: modes = U / sqrt_w_ext
        # And time_coefficients = Vt.T * sigma (so Vt = time_coeffs.T / sigma)
        # Reconstruction of H: modes @ time_coefficients.T = (U/sqrt_w) @ sigma @ Vt
        # To get back H, we need: H = (modes * sqrt_w) @ time_coefficients.T / sqrt_w
        # But simpler: just use modes @ time_coefficients.T which gives H (unweighted)
        data_centered = data_matrix - analyzer.temporal_mean
        H = analyzer._build_hankel_matrix(data_centered)
        H_norm = np.linalg.norm(H, "fro")

        errors = []
        for n_modes in range(1, min(8, analyzer.n_modes_save + 1)):
            # Reconstruct: modes @ time_coefficients.T = U @ sigma @ Vt (unweighted form)
            H_recon = analyzer.modes[:, :n_modes] @ analyzer.time_coefficients[:, :n_modes].T
            error = np.linalg.norm(H - H_recon, "fro") / H_norm
            errors.append(error)

        # Verify monotonic decrease (with small tolerance for numerical noise)
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-8, f"Error should decrease: {errors[i]:.6f} -> {errors[i + 1]:.6f}"

    def test_stpod_modes_orthonormal(self):
        """ST-POD modes must be orthonormal with respect to extended weights.

        Mathematical basis:
        - ST-POD modes U satisfy: Uᵀ W_ext U = I
        - where W_ext is the spatial weight matrix extended to d copies
        """
        rng = np.random.default_rng(789)
        Ns, Nx = 40, 20

        data_matrix = rng.standard_normal((Ns, Nx))

        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        embedding_dim = 6
        analyzer = STPODAnalyzer(
            file_path="analytical_stpod_ortho",
            embedding_dim=embedding_dim,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=10,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Check orthonormality
        n_modes = analyzer.modes.shape[1]
        # The metric the load built (cell volumes for grid-shaped 1-D
        # coordinates), extended to one copy per embedding delay.
        W_ext = np.tile(np.asarray(analyzer.W).ravel(), embedding_dim)
        W_diag = np.diag(W_ext)

        gram = analyzer.modes.T @ W_diag @ analyzer.modes
        identity = np.eye(n_modes)

        assert np.allclose(gram, identity, atol=1e-10), (
            f"Modes not orthonormal. Max deviation: {np.max(np.abs(gram - identity)):.2e}"
        )

    def test_stpod_eigenvalues_nonnegative_sorted(self):
        """ST-POD eigenvalues must be non-negative and descending.

        Mathematical basis:
        - Eigenvalues are squared singular values, hence non-negative
        - Convention: sorted in descending order by energy
        """
        rng = np.random.default_rng(101)
        Ns, Nx = 50, 25

        data_matrix = rng.standard_normal((Ns, Nx))

        data = {
            "q": data_matrix,
            "x": np.linspace(0, 1, Nx),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }

        embedding_dim = 7
        analyzer = STPODAnalyzer(
            file_path="analytical_stpod_eig",
            embedding_dim=embedding_dim,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
            n_modes_save=15,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Non-negative
        assert np.all(analyzer.eigenvalues >= -1e-14), (
            f"Eigenvalues must be non-negative, min: {np.min(analyzer.eigenvalues)}"
        )

        # Descending order
        for i in range(len(analyzer.eigenvalues) - 1):
            assert analyzer.eigenvalues[i] >= analyzer.eigenvalues[i + 1] - 1e-14, (
                f"Not sorted: λ[{i}]={analyzer.eigenvalues[i]} < λ[{i + 1}]={analyzer.eigenvalues[i + 1]}"
            )
