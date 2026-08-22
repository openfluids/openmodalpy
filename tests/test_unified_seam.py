"""Library path and CLI path must produce the same outputs.

One analysis sequence means one set of artifacts: the same case run through
``analyzer.run_analysis()`` and through ``analyze_from_spec`` must land the
same result and figure FILE NAMES in their respective directories. Contents
are not compared byte-for-byte — matplotlib embeds timestamps — but a figure
present on one path and missing on the other is exactly the drift this test
exists to kill. DMD and PSD-POD are mandatory cases: DMD changed behaviour
(plots now default-on) and PSD-POD had no plotting at all before the seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openmodalpy import DMDAnalyzer, PODAnalyzer, PSDPODAnalyzer, SPODAnalyzer
from openmodalpy.commands import _loader_from_case, analyze_from_spec
from openmodalpy.specs import AnalyzeSpec, CaseSpec, DataSourceSpec


def _double_gyre_spec(tmp_path: Path, method: str, tag: str, nfft: int) -> AnalyzeSpec:
    data = DataSourceSpec(kind="generator", name="double_gyre", params={"Nx": 8, "Ny": 4, "Nt": 24})
    case = CaseSpec(
        name="parity",
        description="library/CLI parity case",
        case_type="double_gyre",
        data=data,
        spatial_weight_type="uniform",
        n_modes_save=3,
        rank=1,
        nfft=nfft,
        overlap=0.5,
        use_parallel=False,
        generate_plots=True,
        # Explicit roots: the defaults resolve outside tmp_path, which let
        # runs from different parameter sets see each other's files.
        results_root=tmp_path / "cli" / "results",
        figures_root=tmp_path / "cli" / "figures",
    )
    return AnalyzeSpec(run_id=tag, method=method, case=case)


def _artifact_names(root: Path) -> set[str]:
    return {p.name for p in root.rglob("*") if p.is_file()}


def _parity_case(method: str) -> tuple[type, dict[str, Any], dict[str, Any]]:
    """Return (class, ctor kwargs from the case, perform kwargs) per method."""
    if method == "dmd":
        return DMDAnalyzer, {"rank": 1}, {}
    if method == "psd_pod":
        return PSDPODAnalyzer, {"blockwise_mean": False}, {}
    return SPODAnalyzer if method == "spod" else PODAnalyzer, {}, {}


@pytest.mark.filterwarnings("ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning")
@pytest.mark.parametrize(
    ("method", "nfft"),
    [("pod", 8), ("dmd", 8), ("spod", 8), ("psd_pod", 8)],
    ids=["POD", "DMD", "SPOD", "PSD-POD"],
)
def test_library_and_cli_produce_same_files(method: str, nfft: int, tmp_path: Path) -> None:
    analyzer_cls, extra_ctor, perform_kwargs = _parity_case(method)
    spec = _double_gyre_spec(tmp_path, method, "run", nfft)

    # Library path: construct by hand with the CLI's own loader and roots.
    file_path, data_loader = _loader_from_case(spec.case)
    lib_results = tmp_path / "lib" / "results"
    lib_figures = tmp_path / "lib" / "figures"
    lib_results.mkdir(parents=True)
    lib_figures.mkdir(parents=True)

    common: dict[str, Any] = {
        "file_path": file_path,
        "results_dir": str(lib_results),
        "figures_dir": str(lib_figures),
        "data_loader": data_loader,
        "spatial_weight_type": spec.case.spatial_weight_type,
        "use_parallel": False,
    }
    if analyzer_cls is not SPODAnalyzer:
        common["n_modes_save"] = spec.case.n_modes_save
    if analyzer_cls in (SPODAnalyzer, PSDPODAnalyzer):
        common["nfft"] = nfft
        common["overlap"] = 0.5
    analyzer = analyzer_cls(**common, **extra_ctor)
    # Same run id as the spec: figure names key on it where supported.
    _ = analyzer.run_analysis(plots=True, run_id="run", **perform_kwargs)

    # CLI path: same spec through the orchestration layer.
    outcome = analyze_from_spec(spec)

    assert outcome.success and outcome.executed
    cli_results = Path(outcome.results_dir)
    cli_figures = Path(outcome.figures_dir)

    assert _artifact_names(lib_results) == _artifact_names(cli_results), f"{method}: result files diverged"
    assert _artifact_names(lib_figures) == _artifact_names(cli_figures), f"{method}: figures diverged"
