"""Pytest configuration for the doctests in ``src/openmodalpy``.

The examples in the docstrings construct analyzers. An analyzer makes its
results and figures directories when it is built, so a doctest that runs in
the repository root would leave those directories there. This fixture gives
every doctest its own temporary working directory.

The test suite has its own directory fixture in ``tests/conftest.py``; this
file only covers items collected from ``src``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _doctest_runs_in_tmp_cwd(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Run each doctest in its own temporary directory."""
    if isinstance(request.node, pytest.DoctestItem):
        request.getfixturevalue("monkeypatch").chdir(tmp_path)
