"""
Unit tests for STPODAnalyzer.
"""

import logging

import h5py
import numpy as np
import pytest

from openmodalpy import STPODAnalyzer


class TestSTPODBasic:
    """Basic functionality tests for ST-POD."""

    def test_perform_stpod_simple(self):
        """ST-POD on synthetic data: shapes plus POD spectral invariants.

        Eigenvalues must be non-negative and non-increasing; modes must be
        orthonormal in the (uniform) lifted metric. The energy-fraction check
        is attribute consistency (stored fraction == sum(lambda)/total_energy),
        not an independent energy oracle — that lives in
        test_total_energy_matches_full_svd_frobenius. Arbitrary arrays of the
        right shape would fail the spectral and orthonormality checks.
        """
        rng = np.random.default_rng(42)
        Ns, Nx, Ny = 50, 10, 10
        Nspace = Nx * Ny

        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.linspace(0, 1, Nx),
            "y": np.linspace(0, 1, Ny),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": Ny,
            "Ns": Ns,
        }

        embedding_dim = 5
        n_modes = 10
        analyzer = STPODAnalyzer(
            file_path="test_stpod",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Check output shapes
        m = Ns - embedding_dim + 1
        assert analyzer.modes.shape == (embedding_dim * Nspace, n_modes)
        assert analyzer.time_coefficients.shape == (m, n_modes)
        assert analyzer.eigenvalues.shape == (n_modes,)

        # Spectral ordering and positivity (POD eigenvalues = sigma^2 / m).
        assert np.all(analyzer.eigenvalues >= -1e-14)
        assert np.all(np.diff(analyzer.eigenvalues) <= 1e-12)

        # Weighted orthonormality under the uniform lifted metric (W = 1).
        gram = analyzer.modes.T @ analyzer.modes
        np.testing.assert_allclose(gram, np.eye(n_modes), rtol=0.0, atol=1e-10)

        # Attribute consistency: stored fraction equals sum(lambda) / total_energy.
        retained = float(np.sum(analyzer.eigenvalues))
        assert analyzer.total_energy > retained
        np.testing.assert_allclose(
            analyzer.energy_captured_fraction,
            retained / analyzer.total_energy,
            rtol=1e-12,
            atol=0.0,
        )

    def test_hankel_matrix_shape(self):
        """Verify Hankel matrix construction."""
        Ns, Nspace = 20, 15
        embedding_dim = 5

        data = {
            "q": np.arange(Ns * Nspace).reshape(Ns, Nspace).astype(float),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_hankel",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        # Build Hankel manually to test
        data_centered = data["q"] - np.mean(data["q"], axis=0)
        H = analyzer._build_hankel_matrix(data_centered)

        m = Ns - embedding_dim + 1
        assert H.shape == (embedding_dim * Nspace, m)

    def test_extract_spatial_mode(self):
        """Spatial slices of a space-time mode reassemble the full mode vector.

        Stacking extract_spatial_mode over every delay must recover modes[:, k]
        exactly, and the extracted blocks must be pairwise disjoint partitions
        of that column. Shape alone would not catch a wrong delay stride.
        """
        rng = np.random.default_rng(123)
        Ns, Nspace = 30, 20
        embedding_dim = 4

        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_extract",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Extract at different delays
        for delay in range(embedding_dim):
            spatial_mode = analyzer.extract_spatial_mode(0, delay)
            assert spatial_mode.shape == (Nspace,)

        # Partition invariant: stack of delay slices == full space-time mode.
        for mode_idx in range(analyzer.modes.shape[1]):
            stacked = np.concatenate([analyzer.extract_spatial_mode(mode_idx, d) for d in range(embedding_dim)])
            np.testing.assert_array_equal(stacked, analyzer.modes[:, mode_idx])
            # Direct stride check against the raw mode column.
            for delay in range(embedding_dim):
                start = delay * Nspace
                end = start + Nspace
                np.testing.assert_array_equal(
                    analyzer.extract_spatial_mode(mode_idx, delay),
                    analyzer.modes[start:end, mode_idx],
                )

    def test_get_mode_as_movie(self):
        """Movie frames are the delay blocks of the space-time mode column.

        Flattening the movie in delay-major order must equal modes[:, idx], and
        each frame must match extract_spatial_mode. A wrong reshape or stride
        would pass a pure shape check.
        """
        rng = np.random.default_rng(456)
        Ns, Nspace = 40, 25
        embedding_dim = 6

        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_movie",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        movie = analyzer.get_mode_as_movie(0)
        assert movie.shape == (embedding_dim, Nspace)

        # Flattened movie recovers the full mode; each frame matches extract.
        np.testing.assert_array_equal(movie.reshape(-1), analyzer.modes[:, 0])
        for delay in range(embedding_dim):
            np.testing.assert_array_equal(
                movie[delay],
                analyzer.extract_spatial_mode(0, delay),
            )
        # Movie L2 energy equals the space-time mode energy (Parseval of the reshape).
        np.testing.assert_allclose(
            np.linalg.norm(movie),
            np.linalg.norm(analyzer.modes[:, 0]),
            rtol=0.0,
            atol=1e-14,
        )

    def test_eigenvalues_match_sigma_squared_over_hankel_columns(self):
        """ST-POD spectrum equals SVD of an independently built block-Hankel.

        Centered snapshots are stacked into a Hankel matrix with plain numpy
        (same construction as tests/test_metamorphic.py::_independent_hankel),
        not via the analyzer's lift helper. Eigenvalues must be sigma^2 / m
        with m = number of Hankel columns; modes must match the left singular
        vectors under the library's sign convention. Agreement pins both the
        delay embedding and the per-column normalisation.
        """
        from tests.reference_helpers import canonicalize_reference

        rng = np.random.default_rng(7)
        Ns, Nspace = 12, 3
        embedding_dim = 4
        n_modes = 3

        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_stpod_norm",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Independent block-Hankel: column j stacks delays [Q[j], ..., Q[j+d-1]].
        # Shape (d * Nspace, m); library lift is the transpose layout, same SVD.
        data_centered = data["q"] - np.mean(data["q"], axis=0)
        m_cols = Ns - embedding_dim + 1
        hankel = np.empty((embedding_dim * Nspace, m_cols), dtype=data_centered.dtype)
        for lag in range(embedding_dim):
            hankel[lag * Nspace : (lag + 1) * Nspace, :] = data_centered[lag : lag + m_cols, :].T
        u, sigma, _vt = np.linalg.svd(hankel, full_matrices=False)
        ref_eigs = (sigma[:n_modes] ** 2) / m_cols
        ref_modes, _ = canonicalize_reference(u[:, :n_modes])

        np.testing.assert_allclose(
            analyzer.eigenvalues,
            ref_eigs,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            analyzer.modes,
            ref_modes,
            rtol=1e-10,
            atol=1e-10,
        )
        assert not np.allclose(
            analyzer.eigenvalues,
            sigma[:n_modes] ** 2,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_save_results_records_delay_embedded_contract(self, tmp_path):
        """Smoke test: asserts execution and artifact only, not numerical values."""
        rng = np.random.default_rng(9)
        Ns, Nspace = 10, 3
        analyzer = STPODAnalyzer(
            file_path="test_stpod_contract",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: {
                "q": rng.standard_normal((Ns, Nspace)),
                "x": np.arange(Nspace),
                "y": np.array([0.0]),
                "dt": 0.1,
                "Nx": Nspace,
                "Ny": 1,
                "Ns": Ns,
            },
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.save_results("stpod_contract.hdf5")

        with h5py.File(tmp_path / "stpod_contract.hdf5", "r") as handle:
            assert handle.attrs["stpod_variant"] == "delay_embedded_pod"
            assert handle.attrs["lift_kind"] == "delay_embedding"
            assert handle.attrs["eigenvalue_normalization"] == "sigma_squared_over_n_hankel_cols"
            assert not bool(handle.attrs["is_full_spacetime_pod"])

    def test_stpod_save_load_roundtrip_arrays(self, tmp_path):
        """ST-POD save → load restores modes, eigenvalues, coefficients exactly."""
        rng = np.random.default_rng(41)
        Ns, Nx, Ny = 20, 4, 3
        Nspace = Nx * Ny
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.linspace(0, 1, Nx),
            "y": np.linspace(0, 1, Ny),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": Ny,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_roundtrip",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.save_results("stpod_roundtrip.hdf5")

        reloaded = STPODAnalyzer(
            file_path="test_stpod_roundtrip",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        reloaded.load_results("stpod_roundtrip.hdf5")

        np.testing.assert_array_equal(reloaded.modes, analyzer.modes)
        np.testing.assert_array_equal(reloaded.eigenvalues, analyzer.eigenvalues)
        np.testing.assert_array_equal(reloaded.time_coefficients, analyzer.time_coefficients)

    def test_plot_spacetime_mode_writes_file(self, tmp_path):
        """Smoke test: asserts execution and artifact only, not numerical values."""
        rng = np.random.default_rng(42)
        Ns, Nx, Ny = 24, 5, 4
        Nspace = Nx * Ny
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.linspace(0, 1, Nx),
            "y": np.linspace(0, 1, Ny),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": Ny,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_spacetime",
            embedding_dim=4,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.plot_spacetime_mode(mode_idx=0, n_delays_show=2)

        expected = tmp_path / "test_stpod_spacetime_stpod_spacetime_mode1.png"
        assert expected.is_file()
        assert expected.stat().st_size > 0


class TestSTPODValidation:
    """Validation tests for ST-POD parameters."""

    def test_embedding_dim_too_small_raises(self):
        """embedding_dim < 2 should raise ValueError."""
        rng = np.random.default_rng(30)
        data = {
            "q": rng.standard_normal((20, 10)),
            "x": np.arange(10),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": 10,
            "Ny": 1,
            "Ns": 20,
        }

        analyzer = STPODAnalyzer(
            file_path="test_small_d",
            embedding_dim=1,  # Invalid
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        with pytest.raises(ValueError, match="embedding_dim must be >= 2"):
            analyzer.perform_stpod()

    def test_embedding_dim_too_large_raises(self):
        """embedding_dim >= Ns should raise ValueError."""
        rng = np.random.default_rng(31)
        Ns = 10
        data = {
            "q": rng.standard_normal((Ns, 5)),
            "x": np.arange(5),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": 5,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_large_d",
            embedding_dim=Ns,  # Invalid: equal to Ns
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        with pytest.raises(ValueError, match="must be < number of snapshots"):
            analyzer.perform_stpod()

    def test_no_data_raises(self):
        """Calling perform_stpod without data should raise."""
        analyzer = STPODAnalyzer(
            file_path="nonexistent",
            embedding_dim=5,
        )
        # Don't call load_and_preprocess

        with pytest.raises(ValueError, match="Data not loaded"):
            analyzer.perform_stpod()


def test_check_mode_orthogonality_true_and_false(small_stpod_field, tmp_path):
    analyzer = STPODAnalyzer(
        file_path="stpod_ortho",
        embedding_dim=5,
        n_modes_save=4,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: small_stpod_field,
        spatial_weight_type="uniform",
        use_parallel=False,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_stpod()

    assert analyzer.check_mode_orthogonality()

    analyzer.modes = analyzer.modes + 0.5
    assert not analyzer.check_mode_orthogonality()


def test_check_mode_orthogonality_empty(small_stpod_field, tmp_path):
    analyzer = STPODAnalyzer(
        file_path="stpod_ortho_empty",
        embedding_dim=5,
        n_modes_save=4,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: small_stpod_field,
        spatial_weight_type="uniform",
        use_parallel=False,
    )
    assert not analyzer.check_mode_orthogonality()


class TestSTPODTotalEnergy:
    """True pre-truncation energy denominator for ST-POD percentages."""

    def test_total_energy_matches_full_svd_frobenius(self):
        """Cheap Frobenius total equals sum(all sigma²)/m from a full SVD.

        Builds the same weighted lifted matrix that _solve_svd uses and checks
        ‖data_weighted‖_F² / m against the full singular-value sum.
        """
        from openmodalpy.core import decomposition

        rng = np.random.default_rng(11)
        Ns, Nspace = 16, 5
        embedding_dim = 4
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_frob",
            embedding_dim=embedding_dim,
            n_modes_save=3,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        data_centered = data["q"] - np.mean(data["q"], axis=0)
        lifted = decomposition.DelayEmbeddingLift(embedding_dim).apply(data_centered)
        weights = np.tile(np.ones(Nspace), embedding_dim)
        sqrt_w = np.sqrt(weights)
        data_weighted = lifted * sqrt_w
        m = lifted.shape[0]
        _u, sigma, _vt = np.linalg.svd(data_weighted.T, full_matrices=False)
        full_svd_total = float(np.sum(sigma**2) / m)
        frobenius_total = float(np.linalg.norm(data_weighted, "fro") ** 2 / m)

        np.testing.assert_allclose(frobenius_total, full_svd_total, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(analyzer.total_energy, full_svd_total, rtol=1e-12, atol=0.0)
        assert analyzer.total_energy > float(np.sum(analyzer.eigenvalues))

    def test_total_energy_uses_the_weights(self):
        """The identity must hold with NON-UNIFORM weights, where W is visible.

        Under uniform weights the weighted and unweighted Frobenius norms are
        equal, so dropping the sqrt-weight factor would go unnoticed. These
        weights span four decades and include one near-zero entry (1e-14)
        that must enter as its true measure, not a floored substitute.
        """
        from openmodalpy.core import decomposition

        rng = np.random.default_rng(21)
        Ns, Nspace = 20, 6
        embedding_dim = 4
        weights = np.array([1.0e-14, 1.0e-2, 0.5, 1.0, 3.0, 100.0])
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_weighted",
            embedding_dim=embedding_dim,
            n_modes_save=3,
            data_loader=lambda _: data,
            spatial_weights=weights,
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        data_centered = data["q"] - np.mean(data["q"], axis=0)
        lifted = decomposition.DelayEmbeddingLift(embedding_dim).apply(data_centered)
        tiled = np.tile(weights, embedding_dim)
        data_weighted = lifted * np.sqrt(tiled)
        _u, sigma, _vt = np.linalg.svd(data_weighted.T, full_matrices=False)
        expected = float(np.sum(sigma**2) / lifted.shape[0])

        np.testing.assert_allclose(analyzer.total_energy, expected, rtol=1e-10, atol=0.0)

        # The weighting must actually matter: the unweighted total is far off.
        unweighted = float(np.linalg.norm(lifted, "fro") ** 2 / lifted.shape[0])
        assert abs(analyzer.total_energy - unweighted) > 0.5 * unweighted

    def test_truncated_percentages_sum_to_captured_fraction(self):
        """With truncation below full rank, percentages sum to less than 100%.

        Their sum equals 100 * energy_captured_fraction against the true total,
        computed independently from the weighted lifted matrix (not from the
        analyzer's own total_energy attribute).
        """
        from openmodalpy.core import decomposition

        rng = np.random.default_rng(12)
        Ns, Nspace = 30, 8
        embedding_dim = 5
        n_modes = 3  # well below full rank
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_pct",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Independent pre-truncation total: ‖data_weighted‖_F² / m.
        data_centered = data["q"] - np.mean(data["q"], axis=0)
        lifted = decomposition.DelayEmbeddingLift(embedding_dim).apply(data_centered)
        weights = np.tile(np.ones(Nspace), embedding_dim)
        data_weighted = lifted * np.sqrt(weights)
        m = lifted.shape[0]
        independent_total = float(np.linalg.norm(data_weighted, "fro") ** 2 / m)

        retained = float(np.sum(analyzer.eigenvalues))
        assert independent_total > retained
        expected_fraction = retained / independent_total
        np.testing.assert_allclose(
            analyzer.energy_captured_fraction,
            expected_fraction,
            rtol=1e-12,
            atol=0.0,
        )

        denom, suffix = analyzer._energy_denominator()
        assert suffix == ""
        np.testing.assert_allclose(denom, independent_total, rtol=1e-12, atol=0.0)
        percentages = 100.0 * analyzer.eigenvalues / denom
        pct_sum = float(np.sum(percentages))
        assert pct_sum < 100.0
        np.testing.assert_allclose(
            pct_sum,
            100.0 * expected_fraction,
            rtol=1e-12,
            atol=0.0,
        )

    def test_energy_plot_titles_report_the_true_total(self, tmp_path, monkeypatch):
        """Plot axis labels use the true-total denominator, not retained-only.

        Renders through the real plotting path and reads label text off the
        live figure object before it is closed. A retained-sum denominator
        reintroduces the "retained modes only" suffix and must fail this check.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import openmodalpy.stpod as stpod_mod

        rng = np.random.default_rng(17)
        Ns, Nspace = 24, 6
        embedding_dim = 4
        n_modes = 2
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_energy_plot",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        labels: list[str] = []
        real_savefig = stpod_mod.plt.savefig

        def capture_savefig(*args, **kwargs):
            fig = stpod_mod.plt.gcf()
            for ax in fig.axes:
                labels.append(ax.get_ylabel() or "")
                labels.append(ax.get_title() or "")
                labels.append(ax.get_xlabel() or "")
            if getattr(fig, "_suptitle", None) is not None:
                labels.append(fig._suptitle.get_text() or "")
            return real_savefig(*args, **kwargs)

        monkeypatch.setattr(stpod_mod.plt, "savefig", capture_savefig)
        try:
            analyzer.plot_eigenvalues()
        finally:
            plt.close("all")

        assert labels, "expected to capture axis text from the live figure"
        joined = "\n".join(labels)
        assert any("Normalized Eigenvalue" in text for text in labels)
        assert "retained" not in joined.lower()

    def test_total_energy_save_load_roundtrip(self, tmp_path):
        """Both total_energy and energy_captured_fraction survive save → load."""
        rng = np.random.default_rng(13)
        Ns, Nspace = 18, 4
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_energy_rt",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.save_results("stpod_energy_rt.hdf5")

        with h5py.File(tmp_path / "stpod_energy_rt.hdf5", "r") as handle:
            assert "total_energy" in handle.attrs
            assert "energy_captured_fraction" in handle.attrs
            assert handle.attrs["total_energy"] == analyzer.total_energy

        reloaded = STPODAnalyzer(
            file_path="test_stpod_energy_rt",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        reloaded.load_results("stpod_energy_rt.hdf5")
        assert reloaded.total_energy == analyzer.total_energy
        assert reloaded.energy_captured_fraction == analyzer.energy_captured_fraction
        denom, suffix = reloaded._energy_denominator()
        assert denom == analyzer.total_energy
        assert suffix == ""

    def test_legacy_results_fallback_to_retained_sum(self, tmp_path):
        """Files without total_energy still load and fall back honestly."""
        rng = np.random.default_rng(14)
        Ns, Nspace = 16, 4
        data = {
            "q": rng.standard_normal((Ns, Nspace)),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }
        analyzer = STPODAnalyzer(
            file_path="test_stpod_legacy",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.save_results("stpod_legacy.hdf5")

        # Strip the new attrs so the file looks pre-change.
        path = tmp_path / "stpod_legacy.hdf5"
        with h5py.File(path, "a") as handle:
            del handle.attrs["total_energy"]
            del handle.attrs["energy_captured_fraction"]

        reloaded = STPODAnalyzer(
            file_path="test_stpod_legacy",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        reloaded.load_results("stpod_legacy.hdf5")
        assert not np.isfinite(reloaded.total_energy)
        denom, suffix = reloaded._energy_denominator()
        retained = float(np.sum(reloaded.eigenvalues))
        assert denom == retained
        assert "retained" in suffix
        # Percentages against the fallback still sum to 100%.
        percentages = 100.0 * reloaded.eigenvalues / denom
        np.testing.assert_allclose(float(np.sum(percentages)), 100.0, rtol=1e-12)

    def test_legacy_load_clears_a_previous_runs_total(self, tmp_path):
        """Loading a legacy file must not inherit the total from an earlier run.

        Reusing one analyzer would otherwise denominate against another
        dataset's total while the label claimed it was the true one.
        """
        rng = np.random.default_rng(15)
        Ns, Nspace = 16, 4

        def make_data(seed_rng):
            return {
                "q": seed_rng.standard_normal((Ns, Nspace)),
                "x": np.arange(Nspace),
                "y": np.array([0.0]),
                "dt": 0.1,
                "Nx": Nspace,
                "Ny": 1,
                "Ns": Ns,
            }

        def make_analyzer(data):
            return STPODAnalyzer(
                file_path="test_stpod_stale",
                embedding_dim=3,
                n_modes_save=2,
                results_dir=tmp_path,
                figures_dir=tmp_path,
                data_loader=lambda _: data,
                spatial_weight_type="uniform",
            )

        writer = make_analyzer(make_data(rng))
        writer.load_and_preprocess()
        writer.perform_stpod()
        writer.save_results("stpod_stale.hdf5")
        with h5py.File(tmp_path / "stpod_stale.hdf5", "a") as handle:
            del handle.attrs["total_energy"]
            del handle.attrs["energy_captured_fraction"]

        # Same analyzer object: run on its own data first, then load the legacy file.
        reused = make_analyzer(make_data(rng))
        reused.load_and_preprocess()
        reused.perform_stpod()
        assert np.isfinite(reused.total_energy)
        reused.load_results("stpod_stale.hdf5")

        assert not np.isfinite(reused.total_energy)
        assert not np.isfinite(reused.energy_captured_fraction)
        denom, suffix = reused._energy_denominator()
        assert denom == float(np.sum(reused.eigenvalues))
        assert "retained" in suffix


def test_stpod_keeps_every_mode_the_lift_supports_in_the_spatial_regime(caplog):
    """ST-POD's own rank cap must not drop a genuine mode (spatial lift).

    Ns=40, Nspace=2, embedding_dim=3 lifts to 38x6. The lifted matrix supports
    6 modes (full column rank); the caller cap must allow all of them.
    Guards ``stpod.py``'s caller-side cap, which the low-level ``_solve_svd``
    test cannot reach.
    """
    rng = np.random.default_rng(11)
    Ns, Nspace, embedding_dim = 40, 2, 3
    data = {
        "q": rng.standard_normal((Ns, Nspace)),
        "x": np.arange(float(Nspace)),
        "y": np.array([0.0]),
        "dt": 0.1,
        "Nx": Nspace,
        "Ny": 1,
        "Ns": Ns,
    }
    analyzer = STPODAnalyzer(
        file_path="stpod_spatial_regime",
        embedding_dim=embedding_dim,
        n_modes_save=10,  # deliberately above the bound so the cap decides
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    with caplog.at_level(logging.WARNING, logger="openmodalpy.stpod"):
        analyzer.perform_stpod()
    assert "Only 6 modes available, requested 10" in caplog.text

    n_samples_lift = Ns - embedding_dim + 1
    n_space_lift = Nspace * embedding_dim
    want = min(n_samples_lift, n_space_lift)
    assert want == 6, "fixture drifted: it must sit in the spatial regime"
    assert analyzer.eigenvalues.size == want
    assert analyzer.modes.shape == (n_space_lift, want)


def test_stpod_kept_count_equals_lifted_matrix_rank():
    """ST-POD keeps every mode ``matrix_rank`` of the lifted matrix supports.

    Temporal-lift regime (``m <= d*Nx``): snapshots are centered before the
    delay embed, but a window of a zero-mean series is not zero-mean, so the
    lifted matrix has full row rank. Restoring the centered-input caller cap
    ``min(m - 1, n)`` drops one genuine mode and fails here.
    """
    from openmodalpy.core.decomposition import DelayEmbeddingLift

    rng = np.random.default_rng(4205)
    # The first three are temporal-lift (m = Ns - d + 1 <= d * Nx), where the
    # old m-1 cap dropped a mode. The last (m=38, d*Nx=6) is spatial: the
    # feature dimension binds and both caps give 6, so it is the control.
    cases = [(30, 3, 10), (16, 5, 4), (24, 6, 5), (40, 2, 3)]
    for Ns, Nx, d in cases:
        q = rng.standard_normal((Ns, Nx))
        data = {
            "q": q,
            "x": np.arange(Nx, dtype=float),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": 1,
            "Ns": Ns,
        }
        m = Ns - d + 1
        analyzer = STPODAnalyzer(
            file_path=f"stpod_rank_{Ns}_{Nx}_{d}",
            embedding_dim=d,
            n_modes_save=m + 5,
            data_loader=lambda _, _data=data: _data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        lifted = DelayEmbeddingLift(d).apply(q - q.mean(axis=0))
        want = int(np.linalg.matrix_rank(lifted, tol=1e-10))
        got = int(np.asarray(analyzer.eigenvalues).size)
        assert got == want, f"Ns={Ns} Nx={Nx} d={d} lifted={lifted.shape}: kept {got}, rank {want}"
        assert analyzer.modes.shape[1] == got


def _uniform_field(ns: int = 24, nx: int = 6, ny: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "q": rng.standard_normal((ns, nx * ny)),
        "x": np.linspace(0.5, 2.0, nx),
        "y": np.linspace(0.5, 1.5, ny),
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }


def _ramp_uniform_weights(x, y, z=None, n_space=None):
    if n_space is None:
        n_space = int(np.asarray(x).shape[0] * np.asarray(y).shape[0])
    return np.linspace(0.2, 3.0, int(n_space)).reshape(-1, 1)


def test_stpod_uniform_metric_moves_eigenvalues_and_matches_saved_file(monkeypatch, tmp_path):
    """The uniform path must use the metric load_and_preprocess built.

    _get_weight_vector used to return ones whenever the type was uniform, so
    the eigenproblem ignored the built metric while self.W — and the saved
    file — still named it. Patch the builder to a ramp: the eigenvalues must
    move, and a prescribed rerun on the W stored in the results file must
    reproduce that spectrum.
    """
    field = _uniform_field()
    n_space = field["Nx"] * field["Ny"]
    ramp = _ramp_uniform_weights(field["x"], field["y"], n_space=n_space)
    common = dict(
        file_path="dummy",
        data_loader=lambda _: field,
        embedding_dim=3,
        n_modes_save=4,
        use_parallel=False,
    )
    plain = STPODAnalyzer(
        spatial_weight_type="uniform",
        results_dir=str(tmp_path / "plain"),
        figures_dir=str(tmp_path / "plain"),
        **common,
    )
    plain.load_and_preprocess()
    plain.perform_stpod()

    monkeypatch.setattr(
        "openmodalpy.core.base.calculate_uniform_weights",
        _ramp_uniform_weights,
    )
    ramped = STPODAnalyzer(
        spatial_weight_type="uniform",
        results_dir=str(tmp_path / "ramp"),
        figures_dir=str(tmp_path / "ramp"),
        **common,
    )
    ramped.load_and_preprocess()
    ramped.perform_stpod()
    ramped.save_results("stpod_ramp.hdf5")

    assert not np.allclose(plain.eigenvalues, ramped.eigenvalues), (
        "ST-POD eigenvalues did not move when calculate_uniform_weights "
        "returned a ramp; the uniform path discarded the metric"
    )

    with h5py.File(tmp_path / "ramp" / "stpod_ramp.hdf5", "r") as handle:
        saved_w = np.asarray(handle["W"])
    np.testing.assert_allclose(saved_w.ravel(), ramp.ravel())

    from_file = STPODAnalyzer(
        spatial_weights=saved_w,
        results_dir=str(tmp_path / "from_file"),
        figures_dir=str(tmp_path / "from_file"),
        **common,
    )
    from_file.load_and_preprocess()
    from_file.perform_stpod()
    np.testing.assert_allclose(from_file.eigenvalues, ramped.eigenvalues)
