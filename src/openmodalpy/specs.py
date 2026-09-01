"""Typed specs and command metadata for the OpenModalPy CLI/API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DataSourceSpec:
    """Describe where a case gets its snapshots."""

    kind: Literal["file", "generator", "dnami"]
    path: Path | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    """Shared case-level settings used by one or more modal runs."""

    name: str
    description: str
    case_type: str
    data: DataSourceSpec
    spatial_weight_type: str = "uniform"
    n_modes_save: int = 10
    # DMD truncation rank: positive int, "svht", or "energy" (required for DMD;
    # None reaches DMDAnalyzer and raises). n_modes_save only bounds saved/plotted
    # output and never sets the operator rank.
    rank: int | str | None = None
    # Cumulative energy target for rank="energy". None means leave the analyzer
    # default (0.999) alone — one source of truth, not re-hardcoded here.
    energy_fraction: float | None = None
    nfft: int = 128
    overlap: float = 0.5
    embedding_dim: int = 10
    use_parallel: bool = True
    generate_plots: bool = True
    results_root: Path | None = None
    figures_root: Path | None = None


@dataclass(frozen=True)
class AnalyzeSpec:
    """Describe one concrete analysis run."""

    run_id: str
    method: str
    case: CaseSpec
    params: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None


@dataclass(frozen=True)
class RunCollectionSpec:
    """Describe either a direct set of analyses or a suite of nested configs."""

    name: str
    description: str
    config_path: Path
    analyses: list[AnalyzeSpec] = field(default_factory=list)
    nested_configs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class MethodInfo:
    """Metadata exposed through ``openmodalpy methods``."""

    method_id: str
    cli_name: str
    display_name: str
    description: str
    parameter_help: dict[str, str] = field(default_factory=dict)
    implementation_scope: str = "analyzer"


@dataclass(frozen=True)
class ExampleInfo:
    """Discovered example config metadata."""

    name: str
    config_path: Path
    kind: str
    title: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    """Execution record returned by the command core."""

    run_id: str
    method: str
    case_name: str
    results_dir: Path
    figures_dir: Path
    results_path: Path | None
    success: bool
    executed: bool
    message: str = ""


METHOD_REGISTRY: dict[str, MethodInfo] = {
    "pod": MethodInfo(
        method_id="pod",
        cli_name="pod",
        display_name="POD",
        description="Proper orthogonal decomposition on the snapshot ensemble.",
        parameter_help={
            "solver": (
                "Second-order route: eigh (default, correlation/Gram kernel) or "
                "svd (weighted snapshot matrix; better dynamic range on weak modes)."
            ),
        },
    ),
    "mpod": MethodInfo(
        method_id="mpod",
        cli_name="mpod",
        display_name="mPOD",
        description="Multiscale second-order POD with non-overlapping temporal scale bands.",
        parameter_help={
            "band_edges": "Band edges defining the non-overlapping mPOD intervals.",
            "band_scale": "Interpret band edges in Hz or as fractions of Nyquist.",
            "filter_kind": "Band filter type; currently rectangular.",
        },
    ),
    "psd_pod": MethodInfo(
        method_id="psd_pod",
        cli_name="psd-pod",
        display_name="PSD-POD",
        description="POD on the ensemble of blockwise Fourier realizations.",
        parameter_help={
            "nfft": "FFT block size used to build the Fourier ensemble.",
            "overlap": "Block overlap fraction used in the FFT blocking.",
        },
    ),
    "dmd": MethodInfo(
        method_id="dmd",
        cli_name="dmd",
        display_name="DMD",
        description="Lift-and-regress dynamic mode decomposition on paired snapshots.",
        parameter_help={
            "method": "Regression model: ls or tls.",
            "embedding_dim": "Embedding depth; >1 gives Hankel / HODMD-style coordinates.",
        },
    ),
    "hodmd": MethodInfo(
        method_id="hodmd",
        cli_name="hodmd",
        display_name="HODMD",
        description="Higher-order / Hankel DMD using a delay embedding before DMD regression.",
        parameter_help={
            "embedding_dim": "Embedding depth; defaults to the case embedding dimension.",
        },
    ),
    "tls_hodmd": MethodInfo(
        method_id="tls_hodmd",
        cli_name="tls-hodmd",
        display_name="TLS-HODMD",
        description="Higher-order / Hankel DMD with total least-squares regression.",
        parameter_help={
            "embedding_dim": "Embedding depth; defaults to the case embedding dimension.",
        },
    ),
    "spod": MethodInfo(
        method_id="spod",
        cli_name="spod",
        display_name="SPOD",
        description="Welch-block spectral proper orthogonal decomposition.",
        parameter_help={
            "nfft": "FFT block size.",
            "overlap": "Block overlap fraction.",
        },
    ),
    "bsmd": MethodInfo(
        method_id="bsmd",
        cli_name="bsmd",
        display_name="BSMD",
        description="Bispectral mode decomposition for triadic interactions.",
        parameter_help={
            "nfft": "FFT block size.",
            "overlap": "Block overlap fraction.",
        },
    ),
    "stpod": MethodInfo(
        method_id="stpod",
        cli_name="stpod",
        display_name="ST-POD",
        description="Delay-embedded space-time POD in a block-Hankel lift.",
        parameter_help={
            "embedding_dim": "Delay embedding dimension.",
        },
    ),
}


def display_name_for(analysis_type: str) -> str:
    """Return the user-facing display name for an ``analysis_type`` key.

    Looks up ``METHOD_REGISTRY``. Unknown types fall back to ``.upper()`` so a
    new analyzer still gets a readable label before it is registered.
    """
    info = METHOD_REGISTRY.get(analysis_type)
    if info is None:
        return analysis_type.upper()
    return info.display_name
