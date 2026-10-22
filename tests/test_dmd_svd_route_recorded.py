"""The DMD result file records which SVD route the run took.

The routing rule sends a small-rank request on a large matrix to ARPACK and
everything else to a dense LAPACK solve. The two differ by an order of
magnitude in cost on a delay-embedded case, so a result file that does not
name the route cannot explain the time it took. These tests check that the
field is written, that it survives a save and a reload, and that it is not
the same word every time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openmodalpy import DMDAnalyzer, generate_double_gyre


@pytest.fixture(scope="module")
def gyre() -> dict:
    """A double gyre with enough snapshots to reach the ARPACK route."""
    return generate_double_gyre(Nx=60, Ny=30, Nt=400)


def _run(data: dict, rank: object, embedding_dim: int) -> DMDAnalyzer:
    """Run DMD once and return the analyzer."""
    analyzer = DMDAnalyzer(data=data, n_modes_save=5, rank=rank)
    analyzer.load_and_preprocess()
    analyzer.perform_dmd(embedding_dim=embedding_dim, method="ls")
    return analyzer


def test_a_small_rank_request_records_the_iterative_route(gyre: dict) -> None:
    """Rank 10 on 400 snapshots is below the ARPACK rank fraction."""
    analyzer = _run(gyre, 10, 4)
    assert analyzer._get_algorithm_metadata()["dmd_svd_route"] == "iterative"


def test_a_near_full_rank_request_records_the_dense_route(gyre: dict) -> None:
    """The svht rank on this case is far above the ARPACK rank fraction."""
    with pytest.warns(RuntimeWarning, match="effective rank"):
        analyzer = _run(gyre, "svht", 4)
    assert analyzer._get_algorithm_metadata()["dmd_svd_route"] == "dense"


def test_the_route_survives_a_save_and_a_reload(gyre: dict, tmp_path: Path) -> None:
    """A reloaded result must report the route of the run that made it."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        written = _run(gyre, 10, 4)
        assert written._get_algorithm_metadata()["dmd_svd_route"] == "iterative"
        written.save_results()

        reloaded = DMDAnalyzer(data=gyre, n_modes_save=5, rank=10)
        reloaded.load_and_preprocess()
        reloaded.load_results()
    finally:
        os.chdir(cwd)

    assert reloaded._get_algorithm_metadata()["dmd_svd_route"] == "iterative"
