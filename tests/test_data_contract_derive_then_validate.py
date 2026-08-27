"""``data=`` must derive Nx/Ny/Nz/Ns and validate the rest, like a file load.

Before this test existed, the README example failed with a bare
``KeyError: 'Ns'`` because the constructor checked only that ``data`` was a
non-empty dict, never which keys it held. These tests pin the fix: the
README example runs as printed, the shipped template runs as a test, and a
genuinely missing key names itself in the error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from openmodalpy import PODAnalyzer, generate_double_gyre

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "examples" / "my_data_template.py"


def _load_template_module() -> Any:
    """Import the shipped template file by path, the way a copied script runs."""
    spec = importlib.util.spec_from_file_location("my_data_template", TEMPLATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_data_example_runs_verbatim() -> None:
    """README.md, Data Format section: hand-built dict straight into data=."""
    p = generate_double_gyre()
    d = {"q": p["q"], "dt": p["dt"], "Nx": p["Nx"], "Ny": p["Ny"], "x": p["x"], "y": p["y"]}
    pod = PODAnalyzer(data=d)
    pod.load_and_preprocess()
    assert pod.data["Ns"] == p["q"].shape[0]


def test_data_dict_without_grid_counts_derives_them() -> None:
    """Ns, Nx, Ny, Nz are all absent; every one must come from the array shapes."""
    p = generate_double_gyre()
    d = {"q": p["q"], "dt": p["dt"], "x": p["x"], "y": p["y"]}
    pod = PODAnalyzer(data=d)
    pod.load_and_preprocess()
    assert pod.data["Ns"] == p["q"].shape[0]
    assert pod.data["Nx"] == p["Nx"]
    assert pod.data["Ny"] == p["Ny"]
    assert pod.data["Nz"] == 1


def test_missing_underivable_key_names_itself() -> None:
    """dt cannot be derived from shapes; the error must name it."""
    p = generate_double_gyre()
    d = {"q": p["q"], "x": p["x"], "y": p["y"]}
    with pytest.raises(ValueError, match=r"dt") as exc_info:
        PODAnalyzer(data=d)
    msg = str(exc_info.value)
    assert "dt" in msg
    assert "q, x, y, dt" in msg


def test_missing_q_names_itself() -> None:
    """Any of the four required keys must be named, not just dt."""
    p = generate_double_gyre()
    d = {"x": p["x"], "y": p["y"], "dt": p["dt"]}
    with pytest.raises(ValueError, match=r"q") as exc_info:
        PODAnalyzer(data=d)
    assert "q" in str(exc_info.value)


def test_template_loads_and_pod_runs(tmp_path) -> None:
    """The shipped template, run as-is, produces a dataset a POD run accepts."""
    module = _load_template_module()
    data = module.load_my_data("unused/path")
    pod = PODAnalyzer(
        data=data, n_modes_save=2, results_dir=str(tmp_path / "results"), figures_dir=str(tmp_path / "figures")
    )
    pod.load_and_preprocess()
    pod.perform_pod()
    assert pod.modes.shape[0] == data["q"].shape[1]
    assert pod.modes.shape[1] <= 2


def test_doc_supported_formats_agree_with_list_supported_formats() -> None:
    """DOC.md must name every extension `list_supported_formats` reports."""
    from openmodalpy.core.io import DataInterfaceManager

    reported = DataInterfaceManager().list_supported_formats()
    doc_text = (ROOT / "DOC.md").read_text(encoding="utf-8")
    section = doc_text.split("### Supported input formats", 1)[1].split("###", 1)[0]
    for extension in reported:
        needle = extension if extension == "directory" else f"`{extension}`"
        assert needle in section, f"DOC.md 'Supported input formats' does not mention {extension!r}"
