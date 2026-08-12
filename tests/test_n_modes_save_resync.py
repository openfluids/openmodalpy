"""n_modes_save must never outlive the mode arrays (openmodalpy-eyk)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from openmodalpy import MPODAnalyzer, PODAnalyzer, STPODAnalyzer


def _rank_deficient_field(n_snap: int = 40, n_space: int = 64, rank: int = 1, seed: int = 0) -> dict:
    """Exact-rank field so the solver drops modes below the caller's cap."""
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal((rank, n_space))
    amps = rng.standard_normal((n_snap, rank))
    q = amps @ pattern
    nx = int(np.sqrt(n_space))
    ny = n_space // nx
    assert nx * ny == n_space
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": np.linspace(0.0, 1.0, nx),
        "y": np.linspace(0.0, 1.0, ny),
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": n_snap,
    }


def _assert_count_matches_arrays(analyzer) -> None:
    n = int(analyzer.n_modes_save)
    assert n == int(analyzer.eigenvalues.size)
    assert n == int(analyzer.modes.shape[1])
    assert n == int(analyzer.time_coefficients.shape[1])
    # Cap was 12; rank-1 (after centering) field must drop below that.
    assert n < 12


@pytest.mark.parametrize(
    "cls, kwargs, run",
    [
        (PODAnalyzer, {}, "perform_pod"),
        (STPODAnalyzer, {"embedding_dim": 3}, "perform_stpod"),
        (MPODAnalyzer, {}, "perform_mpod"),
        (
            MPODAnalyzer,
            {"band_edges": [0.0, 0.25, 0.5, 1.0], "band_scale": "normalized_nyquist"},
            "perform_mpod",
        ),
    ],
    ids=["pod", "stpod", "mpod-1band", "mpod-multiband"],
)
def test_n_modes_save_resyncs_on_rank_deficient_field(cls, kwargs, run, tmp_path):
    data = _rank_deficient_field()
    analyzer = cls(
        file_path="rank_def",
        n_modes_save=12,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        **kwargs,
    )
    analyzer.load_and_preprocess()
    getattr(analyzer, run)()
    _assert_count_matches_arrays(analyzer)


@pytest.mark.parametrize(
    "cls, kwargs, run",
    [
        (STPODAnalyzer, {"embedding_dim": 3}, "perform_stpod"),
        (
            MPODAnalyzer,
            {"band_edges": [0.0, 0.25, 0.5, 1.0], "band_scale": "normalized_nyquist"},
            "perform_mpod",
        ),
    ],
    ids=["stpod", "mpod-multiband"],
)
def test_saved_mode_count_matches_saved_modes(cls, kwargs, run, tmp_path):
    """A results file must not overstate how many modes it holds.

    Kept apart from the in-memory check above so it is not shadowed by it:
    when the counter is stale, this failure names the written file.
    """
    data = _rank_deficient_field()
    analyzer = cls(
        file_path="rank_def_save",
        n_modes_save=12,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        **kwargs,
    )
    analyzer.load_and_preprocess()
    getattr(analyzer, run)()
    analyzer.save_results("resync.hdf5")

    with h5py.File(tmp_path / "resync.hdf5") as handle:
        declared = int(handle.attrs["n_modes_saved"])
        stored = int(handle["modes"].shape[1])
    assert declared == stored, f"file declares {declared} modes but holds {stored}"
    assert stored < 12, "field did not force a drop; the case no longer tests anything"


def test_stpod_plot_modes_guards_by_array_width(tmp_path, monkeypatch):
    """Plot guards must consult modes.shape[1], not only n_modes_save."""
    import matplotlib

    matplotlib.use("Agg")

    data = _rank_deficient_field()
    analyzer = STPODAnalyzer(
        file_path="rank_def_plot",
        embedding_dim=3,
        n_modes_save=12,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.perform_stpod()
    avail = int(analyzer.modes.shape[1])
    # Force the stale counter the resync removes, so the guard is tested alone.
    analyzer.n_modes_save = avail + 9
    analyzer.plot_modes(plot_n_modes=avail + 9)
