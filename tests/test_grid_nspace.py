"""Grid product and snapshot width must agree at the analyzer boundary."""

import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.core.base import _reported_grid


def _analyzer(loader, tmp_path):
    return PODAnalyzer(
        file_path="custom",
        n_modes_save=2,
        data_loader=loader,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )


def test_custom_loader_grid_mismatch_is_rejected(tmp_path):
    """A data_loader is any (str) -> dict; a wrong grid must not reach the metric."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 4), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 4),
            "Nx": 4,
            "Ny": 4,
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    with pytest.raises(ValueError, match=r"Nx\*Ny\*Nz") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "16" in msg
    assert "4" in msg
    assert "Nx=4" in msg
    assert "Ny=4" in msg
    assert "Nz=1" in msg
    assert "q.shape[1]" in msg


def test_custom_loader_consistent_grid_is_accepted(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 3),
            "Nx": 4,
            "Ny": 3,
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 12
    assert int(np.asarray(analyzer.data["q"]).shape[1]) == 12


def test_dataset_without_grid_metadata_is_not_rejected(tmp_path):
    """No Nx/Ny/Nz means no grid to check — do not invent 1*1*1 and reject."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((8, 6), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 6),
            "y": np.array([0.0]),
            "Ns": 8,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) is None
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 6


def test_reported_grid_treats_a_missing_axis_as_extent_one():
    assert _reported_grid({"Nx": 4, "Ny": 3}) == (4, 3, 1)
    assert _reported_grid({"Ny": 5}) == (1, 5, 1)
    assert _reported_grid({}) is None


def test_reported_grid_all_ones_is_not_a_claim():
    """Extents that are all 1 say nothing about width."""
    assert _reported_grid({"Nz": 1}) is None
    assert _reported_grid({"Nx": 1}) is None
    assert _reported_grid({"Nx": 1, "Ny": 1, "Nz": 1}) is None
    assert _reported_grid({"Nx": 12}) == (12, 1, 1)


def test_lone_degenerate_grid_key_is_not_a_claim(tmp_path):
    """A leftover Nz: 1 is not a 1*1*1 claim against a wider matrix."""

    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 4), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 4),
            "y": np.linspace(0.0, 1.0, 1),
            "Nz": 1,
            "Ns": 6,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) is None
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 4


def test_lone_nx_matching_claim_is_accepted(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 12),
            "y": np.array([0.0]),
            "Nx": 12,
            "Ns": 6,
            "dt": 0.1,
        }

    assert _reported_grid(loader("custom")) == (12, 1, 1)
    analyzer = _analyzer(loader, tmp_path)
    analyzer.load_and_preprocess()
    assert int(np.asarray(analyzer.W).size) == 12


def test_lone_nx_mismatching_claim_is_rejected(tmp_path):
    def loader(_: str) -> dict:
        return {
            "q": np.ones((6, 12), dtype=np.float32),
            "x": np.linspace(0.0, 1.0, 12),
            "y": np.array([0.0]),
            "Nx": 7,
            "Ns": 6,
            "dt": 0.1,
        }

    analyzer = _analyzer(loader, tmp_path)
    with pytest.raises(ValueError, match=r"Nx\*Ny\*Nz") as info:
        analyzer.load_and_preprocess()
    msg = str(info.value)
    assert "7" in msg
    assert "12" in msg
