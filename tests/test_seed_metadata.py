"""Seed propagation into saved result metadata."""

from __future__ import annotations

import h5py

from openmodalpy import PODAnalyzer
from openmodalpy.example_data import generate_example_dataset


def test_seed_recorded_in_saved_metadata(tmp_path):
    """The seed the generator used must land in the saved result file as data_seed."""
    seed = 7
    data = generate_example_dataset(
        "cylinder_wake",
        {"Nx": 20, "Ny": 12, "Nt": 40, "seed": seed},
    )
    assert data.get("seed") == seed

    analyzer = PODAnalyzer(
        file_path="seed_meta",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=4,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.save_results("seed_meta.hdf5")

    with h5py.File(tmp_path / "seed_meta.hdf5", "r") as handle:
        assert "data_seed" in handle.attrs
        assert int(handle.attrs["data_seed"]) == seed
