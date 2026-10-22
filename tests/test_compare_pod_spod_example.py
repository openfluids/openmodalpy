"""``examples/compare_pod_spod.py`` must run and produce real POD/SPOD output.

There is no CI gate that runs ``examples/*.py`` scripts; the smoke job only
runs the CLI (`openmodalpy examples run taylor_green`). This test is the
gate for the compare script, following the same import-by-path precedent as
``tests/test_data_contract_derive_then_validate.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "compare_pod_spod.py"


def _load_script_module() -> Any:
    """Import the shipped script by path, the way a copied script runs."""
    spec = importlib.util.spec_from_file_location("compare_pod_spod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_pod_spod_runs_and_writes_the_figure(tmp_path: Path) -> None:
    """Both analyzers run on the same in-memory dataset and produce real modes."""
    module = _load_script_module()
    pod, spod = module.main(tmp_path)

    # POD: energy-ranked modes over the whole grid.
    assert pod.modes.shape[0] == pod.data["Nx"] * pod.data["Ny"]
    assert pod.eigenvalues.ndim == 1
    assert pod.eigenvalues.size > 0

    # SPOD: one eigenvalue set and one mode set per frequency bin.
    assert spod.freq.size > 0
    assert spod.eigenvalues.shape[0] == spod.freq.size
    assert spod.modes.shape[0] == spod.freq.size
    assert spod.modes.shape[1] == pod.data["Nx"] * pod.data["Ny"]

    # The example exists to show that SPOD resolves a frequency. Pin that.
    # An earlier version used a generator whose only tone sat below one bin
    # width, so the leading eigenvalue peaked in the zero-frequency bin and
    # the SPOD panel simply repeated the POD one. Shapes alone did not catch
    # it: every assertion above passed.
    leading = np.real(np.asarray(spod.eigenvalues)[:, 0])
    peak_bin = int(np.argmax(leading))
    assert peak_bin != 0, "leading SPOD energy sits in the zero-frequency bin"
    assert spod.freq[peak_bin] > 0.0
    # The peak must stand clear of the zero-frequency bin, not merely beat it.
    assert leading[peak_bin] > 100.0 * leading[0], (
        f"peak {leading[peak_bin]:.3e} at {spod.freq[peak_bin]:.4f} Hz is not "
        f"clear of the zero-frequency bin {leading[0]:.3e}"
    )

    figure_path = tmp_path / "figures" / "compare_pod_spod.png"
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0
