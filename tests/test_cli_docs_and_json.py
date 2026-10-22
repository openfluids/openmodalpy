"""Check the machine-readable CLI output and the shipped technical reference.

An assistant or a script reads these paths. Text meant for a terminal is not a
contract, so the JSON shape and the presence of DOC.md are asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmodalpy import PODAnalyzer
from openmodalpy.cli import main
from openmodalpy.commands import METHOD_REGISTRY, documentation_path, documentation_text
from openmodalpy.example_data import generate_example_dataset

DOC_TITLE = "# OpenModalPy technical reference"


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> Any:
    """Run one CLI command and return its stdout parsed as JSON."""
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_documentation_ships_with_the_package() -> None:
    """Check that DOC.md is readable through the package, not only the checkout."""
    path = documentation_path()
    assert path.is_file()
    assert documentation_text().startswith(DOC_TITLE)


def test_docs_command_prints_the_reference(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that `openmodalpy docs` prints the same text the package holds."""
    assert main(["docs"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(DOC_TITLE)
    assert "## Data Contract" in printed


def test_docs_path_flag_prints_an_existing_file(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that `openmodalpy docs --path` names a file that exists."""
    assert main(["docs", "--path"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()


def test_methods_list_json_covers_every_registered_method(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that the JSON list holds one entry per registered method."""
    payload = _run(capsys, ["methods", "list", "--json"])
    assert {entry["method_id"] for entry in payload} == set(METHOD_REGISTRY)
    assert all(entry["cli_name"] and entry["description"] for entry in payload)


def test_methods_show_json_carries_the_parameter_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that one method prints as a JSON object with its parameters."""
    payload = _run(capsys, ["methods", "show", "pod", "--json"])
    assert payload["method_id"] == "pod"
    assert "solver" in payload["parameter_help"]


def test_examples_list_json_omits_the_payload(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that the example list stays a summary and names each config path."""
    payload = _run(capsys, ["examples", "list", "--json"])
    assert payload
    for entry in payload:
        assert "payload" not in entry
        assert entry["name"] and entry["config_path"]


def test_results_inspect_json_is_loadable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check that a saved result file inspects as JSON with its dataset names."""
    data = generate_example_dataset("double_gyre", {"Nx": 16, "Ny": 8, "Nt": 64})
    pod = PODAnalyzer(
        data=data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results("pod.hdf5")
    saved = tmp_path / "results" / "pod.hdf5"

    payload = _run(capsys, ["results", "inspect", str(saved), "--json"])
    assert "modes" in json.dumps(payload)
    assert np.asarray(pod.eigenvalues).size == 2
