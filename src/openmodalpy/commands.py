"""Shared command core for the OpenModalPy CLI and Python API."""

from __future__ import annotations

import argparse
import importlib.resources
import json
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openmodalpy.bsmd import BSMDAnalyzer
from openmodalpy.config_io import load_jsonc, resolve_path
from openmodalpy.core.base import (
    add_inset_colorbar,
    get_fig_aspect_ratio,
    get_robust_clim,
    plot_isometric_slices_3d,
    plot_orthogonal_slices_3d,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.io import load_data
from openmodalpy.core.welch import welch_nblocks
from openmodalpy.dmd import DMDAnalyzer
from openmodalpy.example_data import generate_example_dataset
from openmodalpy.mpod import MPODAnalyzer
from openmodalpy.pod import PODAnalyzer
from openmodalpy.psd_pod import PSDPODAnalyzer
from openmodalpy.specs import (
    AnalyzeSpec,
    CaseSpec,
    DataSourceSpec,
    ExampleInfo,
    MethodInfo,
    RunCollectionSpec,
    RunOutcome,
)
from openmodalpy.spod import SPODAnalyzer
from openmodalpy.stpod import STPODAnalyzer

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
            "delays": "Delay embedding depth; >1 gives Hankel / HODMD-style coordinates.",
        },
    ),
    "hodmd": MethodInfo(
        method_id="hodmd",
        cli_name="hodmd",
        display_name="HODMD",
        description="Higher-order / Hankel DMD using a delay embedding before DMD regression.",
        parameter_help={
            "delays": "Delay embedding depth; defaults to the case embedding dimension.",
        },
    ),
    "tls_hodmd": MethodInfo(
        method_id="tls_hodmd",
        cli_name="tls-hodmd",
        display_name="TLS-HODMD",
        description="Higher-order / Hankel DMD with total least-squares regression.",
        parameter_help={
            "delays": "Delay embedding depth; defaults to the case embedding dimension.",
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

METHOD_ALIASES = {
    "psd-pod": "psd_pod",
    "tls-hodmd": "tls_hodmd",
}


def repo_root() -> Path:
    """Return the repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def source_checkout_root() -> Path | None:
    """Return the repo root only when running from a source checkout."""
    root = repo_root()
    if (root / "pyproject.toml").is_file() and (root / "examples").is_dir():
        return root
    return None


def examples_root() -> Path:
    """Return the example-config root."""
    return repo_root() / "examples"


def packaged_examples_root() -> Traversable:
    """Return the packaged example-config root."""
    return importlib.resources.files("openmodalpy.examples")


def normalize_method_name(name: str) -> str:
    """Map CLI-style names to canonical internal method IDs."""
    normalized = name.strip().lower().replace("-", "_")
    normalized = METHOD_ALIASES.get(normalized, normalized)
    if normalized not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY)}")
    return normalized


def list_methods() -> list[MethodInfo]:
    """Return the supported method registry."""
    return list(METHOD_REGISTRY.values())


def get_method_spec(name: str) -> MethodInfo:
    """Return one method spec by name or alias."""
    return METHOD_REGISTRY[normalize_method_name(name)]


def _default_results_root(case_name: str) -> Path:
    base = source_checkout_root() or Path.cwd()
    return base / "results" / case_name


def _default_figures_root(case_name: str) -> Path:
    base = source_checkout_root() or Path.cwd()
    return base / "figures" / case_name


def _coerce_rank(value: Any) -> int | str | None:
    """Read a DMD truncation rank from config: null, a positive int, or a criterion name.

    null/None is accepted here and passed through; DMDAnalyzer refuses None at
    construction so an omitted rank fails with one message, not two.
    A JSON boolean is rejected rather than treated as missing.
    """
    if isinstance(value, bool):
        raise ValueError(f"rank must be null, a positive int, 'svht', or 'energy'; got {value!r}")
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower()
    if text in ("", "none", "null"):
        return None
    if text in ("svht", "energy"):
        return text
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"rank must be null, a positive int, 'svht', or 'energy'; got {value!r}") from None


def _coerce_energy_fraction(value: Any) -> float | None:
    """Read the energy-rank fraction from config: null, or a float in (0, 1].

    null/None means "do not override" — DMDAnalyzer keeps its own 0.999 default.
    A JSON boolean is rejected rather than treated as missing.
    """
    if isinstance(value, bool):
        raise ValueError(f"energy_fraction must be null or a float in (0, 1]; got {value!r}")
    if value is None:
        return None
    try:
        frac = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"energy_fraction must be null or a float in (0, 1]; got {value!r}") from None
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"energy_fraction must be null or a float in (0, 1]; got {value!r}")
    return frac


def _load_case_spec_from_payload(payload: dict[str, Any], config_path: Path) -> CaseSpec:
    case_payload = payload.get("case")
    if not isinstance(case_payload, dict):
        raise ValueError(f"{config_path} must define a 'case' object.")

    name = str(case_payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"{config_path} case block is missing 'name'.")

    data_payload = case_payload.get("data")
    if not isinstance(data_payload, dict):
        raise ValueError(f"{config_path} case block is missing a 'data' object.")

    data_kind = str(data_payload.get("kind", "")).strip().lower()
    if data_kind == "file":
        path_value = data_payload.get("path")
        if not path_value:
            raise ValueError(f"{config_path} file-backed case '{name}' is missing data.path.")
        data = DataSourceSpec(kind="file", path=resolve_path(str(path_value), config_path))
        case_type = str(case_payload.get("case_type", "experimental")).strip().lower() or "experimental"
    elif data_kind == "dnami":
        path_value = data_payload.get("path")
        if not path_value:
            raise ValueError(f"{config_path} dNami-backed case '{name}' is missing data.path.")
        schema = data_payload.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"{config_path} dNami-backed case '{name}' must define data.schema as an object.")
        data = DataSourceSpec(kind="dnami", path=resolve_path(str(path_value), config_path), params={"schema": schema})
        case_type = str(case_payload.get("case_type", "experimental")).strip().lower() or "experimental"
    elif data_kind == "generator":
        generator_name = str(data_payload.get("name", "")).strip()
        if not generator_name:
            raise ValueError(f"{config_path} generator-backed case '{name}' is missing data.name.")
        params = data_payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{config_path} generator params must be a JSON object.")
        data = DataSourceSpec(kind="generator", name=generator_name, params=params)
        case_type = str(case_payload.get("case_type", "analytical")).strip().lower() or "analytical"
    else:
        raise ValueError(f"{config_path} has unsupported case.data.kind '{data_kind}'.")

    results_root_value = case_payload.get("results_root")
    figures_root_value = case_payload.get("figures_root")

    return CaseSpec(
        name=name,
        description=str(payload.get("description") or case_payload.get("description") or name),
        case_type=case_type,
        data=data,
        spatial_weight_type=str(case_payload.get("spatial_weight_type", "uniform")),
        n_modes_save=int(case_payload.get("n_modes_save", 10)),
        rank=_coerce_rank(case_payload.get("rank")),
        energy_fraction=_coerce_energy_fraction(case_payload.get("energy_fraction")),
        nfft=int(case_payload.get("nfft", 128)),
        overlap=float(case_payload.get("overlap", 0.5)),
        embedding_dim=int(case_payload.get("embedding_dim", 10)),
        use_parallel=bool(case_payload.get("use_parallel", True)),
        generate_plots=bool(case_payload.get("generate_plots", True)),
        results_root=(
            resolve_path(str(results_root_value), config_path)
            if results_root_value is not None
            else _default_results_root(name)
        ),
        figures_root=(
            resolve_path(str(figures_root_value), config_path)
            if figures_root_value is not None
            else _default_figures_root(name)
        ),
    )


def load_case_spec(config_path: str | Path) -> CaseSpec:
    """Load only the case-level settings from one example config."""
    resolved = Path(config_path).expanduser().resolve()
    payload = load_jsonc(resolved)
    return _load_case_spec_from_payload(payload, resolved)


def _apply_case_overrides(case: CaseSpec, overrides: dict[str, Any]) -> CaseSpec:
    """Return a copy of ``case`` with supported CLI/Python overrides applied."""
    return CaseSpec(
        name=case.name,
        description=case.description,
        case_type=case.case_type,
        data=case.data,
        spatial_weight_type=str(overrides.get("spatial_weight_type", case.spatial_weight_type)),
        n_modes_save=int(overrides.get("n_modes_save", case.n_modes_save)),
        # Rebuilt field-by-field, so an omitted rank would silently drop it.
        rank=_coerce_rank(overrides["rank"]) if "rank" in overrides else case.rank,
        energy_fraction=(
            _coerce_energy_fraction(overrides["energy_fraction"])
            if "energy_fraction" in overrides
            else case.energy_fraction
        ),
        nfft=int(overrides.get("nfft", case.nfft)),
        overlap=float(overrides.get("overlap", case.overlap)),
        embedding_dim=int(overrides.get("embedding_dim", case.embedding_dim)),
        use_parallel=bool(overrides.get("use_parallel", case.use_parallel)),
        generate_plots=bool(overrides.get("generate_plots", case.generate_plots)),
        results_root=(
            Path(overrides["results_root"]).expanduser().resolve() if "results_root" in overrides else case.results_root
        ),
        figures_root=(
            Path(overrides["figures_root"]).expanduser().resolve() if "figures_root" in overrides else case.figures_root
        ),
    )


def _load_run_collection(config_path: Path) -> RunCollectionSpec:
    payload = load_jsonc(config_path)
    name = str(payload.get("name") or config_path.stem)
    description = str(payload.get("description") or name)

    configs_payload = payload.get("configs")
    if isinstance(configs_payload, list):
        nested = [resolve_path(str(item), config_path) for item in configs_payload]
        return RunCollectionSpec(
            name=name,
            description=description,
            config_path=config_path,
            nested_configs=nested,
        )

    case = _load_case_spec_from_payload(payload, config_path)
    runs_payload = payload.get("runs")
    if not isinstance(runs_payload, list) or not runs_payload:
        raise ValueError(f"{config_path} must define a non-empty 'runs' list.")

    analyses: list[AnalyzeSpec] = []
    for index, run_payload in enumerate(runs_payload, start=1):
        if not isinstance(run_payload, dict):
            raise ValueError(f"Run entry #{index} in {config_path} must be a JSON object.")
        if run_payload.get("enabled", True) is False:
            continue
        run_id = str(run_payload.get("id") or f"run_{index}")
        method = normalize_method_name(str(run_payload.get("method", "")))
        params = run_payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Run '{run_id}' in {config_path} must define params as an object.")
        analyses.append(
            AnalyzeSpec(
                run_id=run_id,
                method=method,
                case=case,
                params=params,
                config_path=config_path,
            )
        )

    if not analyses:
        raise ValueError(f"{config_path} does not contain any enabled runs.")

    return RunCollectionSpec(
        name=name,
        description=description,
        config_path=config_path,
        analyses=analyses,
    )


def _loader_from_case(case: CaseSpec) -> tuple[str, Any]:
    """Create the analyzer input tuple ``(file_path, data_loader)``."""
    if case.data.kind == "file":
        return str(case.data.path), None
    if case.data.kind == "dnami":
        schema = dict(case.data.params.get("schema", {}))

        def loader(_: str) -> dict[str, Any]:
            return load_data(str(case.data.path), loader_type="dnami", schema=schema)

        return case.name, loader
    if case.data.kind == "generator":
        generator_name = str(case.data.name)
        generator_params = dict(case.data.params)

        def loader(_: str) -> dict[str, Any]:
            return generate_example_dataset(generator_name, generator_params)

        return case.name, loader
    raise ValueError(f"Unsupported data source kind '{case.data.kind}'.")


def _run_directories(spec: AnalyzeSpec) -> tuple[Path, Path]:
    results_dir = (spec.case.results_root or _default_results_root(spec.case.name)) / spec.run_id
    figures_dir = (spec.case.figures_root or _default_figures_root(spec.case.name)) / spec.run_id
    return results_dir, figures_dir


def _find_latest_result_file(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("*.hdf5"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _prepare_common_run(spec: AnalyzeSpec, *, dry_run: bool) -> tuple[str, Any, Path, Path]:
    file_path, data_loader = _loader_from_case(spec.case)
    results_dir, figures_dir = _run_directories(spec)
    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
    return file_path, data_loader, results_dir, figures_dir


def _apply_snapshot_limit(analyzer: Any, spec: AnalyzeSpec) -> None:
    """Optionally truncate the loaded snapshot matrix for heavy example runs."""
    limit_value = spec.params.get("max_snapshots")
    if limit_value is None:
        return
    if "q" not in analyzer.data:
        return
    limit = int(limit_value)
    q = analyzer.data["q"]
    if limit < 2 or limit >= q.shape[0]:
        return
    analyzer.data["q"] = q[:limit, :]
    analyzer.data["Ns"] = limit
    if hasattr(analyzer, "novlap") and hasattr(analyzer, "nfft") and analyzer.nfft > 1:
        # Same floor formula as BaseAnalyzer.load_and_preprocess (welch_nblocks).
        # The old ceil here overwrote a correct floor value and requested more
        # blocks than fit after truncation (e.g. Ns=400, nfft=128, ovl=0.5).
        Ns = int(analyzer.data["Ns"])
        nblocks = welch_nblocks(Ns, analyzer.nfft, analyzer.novlap)
        if nblocks < 1:
            raise ValueError(
                f"Cannot form Welch blocks: Ns={Ns}, nfft={analyzer.nfft} "
                f"(novlap={analyzer.novlap}) yield nblocks={nblocks}"
            )
        analyzer.nblocks = nblocks


def _make_dry_run_outcome(
    spec: AnalyzeSpec, results_dir: Path, figures_dir: Path, results_path: Path | None = None
) -> RunOutcome:
    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=results_path,
        success=True,
        executed=False,
        message="Dry run only.",
    )


def _plot_psd_pod_eigenvalues(eigenvalues: np.ndarray, figures_dir: Path, run_id: str) -> list[Path]:
    saved: list[Path] = []
    if eigenvalues.size == 0:
        return saved

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, len(eigenvalues) + 1), np.maximum(eigenvalues.real, 1e-16), marker="o")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("PSD-POD eigenvalue")
    ax.set_title("PSD-POD eigenvalues")
    path = figures_dir / f"{run_id}_eigenvalues.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    cumulative = np.cumsum(eigenvalues.real)
    total = cumulative[-1] if cumulative.size else 0.0
    if total > 0:
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(eigenvalues) + 1), cumulative / total * 100.0, marker="o")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Cumulative energy [%]")
        ax.set_title("PSD-POD cumulative energy")
        path = figures_dir / f"{run_id}_cumulative_energy.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def _plot_psd_pod_modes(
    *,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    data: dict[str, Any],
    figures_dir: Path,
    run_id: str,
    plot_n_modes: int = 2,
) -> list[Path]:
    """Plot the leading PSD-POD spatial modes with the same 2D styling as other analyzers."""
    saved: list[Path] = []
    if modes.size == 0:
        return saved

    nx = int(data.get("Nx", 0))
    ny = int(data.get("Ny", 0))
    if nx <= 1 or ny <= 1 or modes.shape[0] != nx * ny:
        return saved

    x_coords = data.get("x", np.arange(nx))
    y_coords = data.get("y", np.arange(ny))
    if np.ndim(x_coords) == 1 and np.ndim(y_coords) == 1:
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
    else:
        x_mesh, y_mesh = x_coords, y_coords

    total_energy = float(np.sum(np.real(eigenvalues)))
    fig_aspect = get_fig_aspect_ratio(data)
    n_modes = min(plot_n_modes, modes.shape[1])
    ncols = n_modes
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols * fig_aspect, 4), squeeze=False)
    axes = axes.ravel()
    var_name = data.get("metadata", {}).get("var_name", "q")

    for idx in range(n_modes):
        ax = axes[idx]
        mode = np.asarray(modes[:, idx].real).reshape(nx, ny)
        vmin, vmax = get_robust_clim(mode, method="percentile")
        levels = np.linspace(vmin, vmax, 21)
        cf = ax.contourf(x_mesh, y_mesh, mode, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(x_mesh, y_mesh, mode, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
        style_spatial_axes(ax, data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
        energy_pct = 100.0 * float(np.real(eigenvalues[idx])) / total_energy if total_energy > 0 else 0.0
        ax.set_title(f"PSD-POD Mode {idx + 1} [{var_name}] | E={energy_pct:.2f}%")
        add_inset_colorbar(
            fig,
            ax,
            cf,
            data,
            ticks=[vmin, 0, vmax],
            ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
        )

    with plt.rc_context():
        fig.tight_layout()
    path = figures_dir / f"{run_id}_modes_1_to_{n_modes}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    return saved


def _plot_psd_pod_modes_3d(
    *,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    data: dict[str, Any],
    figures_dir: Path,
    run_id: str,
    plot_n_modes: int = 2,
) -> list[Path]:
    """Plot the leading PSD-POD modes with the shared 3D helpers."""
    saved: list[Path] = []
    if modes.size == 0 or resolve_volume_layout(data, modes.shape[0]) is None:
        return saved

    x_coords = data.get("x")
    y_coords = data.get("y")
    z_coords = data.get("z")
    total_energy = float(np.sum(np.real(eigenvalues))) if eigenvalues.size else 0.0
    n_modes = min(plot_n_modes, modes.shape[1])
    for idx in range(n_modes):
        mode_3d = reshape_mode_to_volume(np.asarray(modes[:, idx]).real, data)
        energy_pct = 100.0 * float(np.real(eigenvalues[idx])) / total_energy if total_energy > 0 else 0.0
        title = f"PSD-POD Mode {idx + 1} | E={energy_pct:.2f}%"
        slice_path = figures_dir / f"{run_id}_mode_{idx + 1}_slices.png"
        iso_path = figures_dir / f"{run_id}_mode_{idx + 1}_isometric.png"
        plot_orthogonal_slices_3d(
            mode_3d,
            x_coords,
            y_coords,
            z_coords,
            output_path=str(slice_path),
            title_prefix=title,
            data=data,
            scalar_name="psd_pod_mode",
        )
        plot_isometric_slices_3d(
            mode_3d,
            x_coords,
            y_coords,
            z_coords,
            output_path=str(iso_path),
            title_prefix=title,
            data=data,
            scalar_name="psd_pod_mode",
        )
        saved.extend([slice_path, iso_path])
    return saved


def _maybe_plot_volumetric_modes(
    analyzer: Any,
    *,
    plot_n_modes: int,
    slices_kwargs: dict[str, Any] | None = None,
    iso_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Use analyzer-specific 3D plot hooks when volumetric data is present."""
    if int(analyzer.data.get("Nz", 1)) <= 1:
        return False
    used = False
    if hasattr(analyzer, "plot_modes_3d_slices"):
        kwargs = {"plot_n_modes": plot_n_modes}
        kwargs.update(slices_kwargs or {})
        analyzer.plot_modes_3d_slices(**kwargs)
        used = True
    if hasattr(analyzer, "plot_modes_3d_isometric"):
        kwargs = {"plot_n_modes": plot_n_modes}
        kwargs.update(iso_kwargs or {})
        analyzer.plot_modes_3d_isometric(**kwargs)
        used = True
    return used


def _run_psd_pod(spec: AnalyzeSpec, *, dry_run: bool) -> RunOutcome:
    """Thin CLI/API runner: construct PSDPODAnalyzer, run, optionally plot."""
    file_path, data_loader, results_dir, figures_dir = _prepare_common_run(spec, dry_run=dry_run)
    if dry_run:
        return _make_dry_run_outcome(spec, results_dir, figures_dir, results_dir / "dry_run_psd_pod.hdf5")

    # Same flags compute_fft_blocks / blocksfft will use. blocksfft always
    # removes a mean; blockwise_mean only chooses global vs per-block.
    blockwise_mean = bool(spec.params.get("blockwise_mean", False))
    analyzer = PSDPODAnalyzer(
        file_path=file_path,
        nfft=int(spec.params.get("nfft", spec.case.nfft)),
        overlap=float(spec.params.get("overlap", spec.case.overlap)),
        results_dir=str(results_dir),
        figures_dir=str(figures_dir),
        data_loader=data_loader,
        spatial_weight_type=spec.case.spatial_weight_type,
        use_parallel=spec.case.use_parallel,
        blockwise_mean=blockwise_mean,
        n_modes_save=spec.case.n_modes_save,
    )
    analyzer.load_and_preprocess()
    _apply_snapshot_limit(analyzer, spec)
    analyzer.compute_fft_blocks()
    analyzer.perform_psd_pod()
    analyzer.save_results()
    save_path = Path(analyzer.results_path) if analyzer.results_path else _find_latest_result_file(results_dir)

    if spec.case.generate_plots:
        modes = np.asarray(analyzer.modes)
        eigenvalues = np.asarray(analyzer.eigenvalues)
        _plot_psd_pod_eigenvalues(eigenvalues, figures_dir, spec.run_id)
        if resolve_volume_layout(analyzer.data, modes.shape[0]) is not None:
            _plot_psd_pod_modes_3d(
                modes=modes,
                eigenvalues=eigenvalues,
                data=analyzer.data,
                figures_dir=figures_dir,
                run_id=spec.run_id,
                plot_n_modes=min(2, spec.case.n_modes_save),
            )
        else:
            _plot_psd_pod_modes(
                modes=modes,
                eigenvalues=eigenvalues,
                data=analyzer.data,
                figures_dir=figures_dir,
                run_id=spec.run_id,
                plot_n_modes=min(2, spec.case.n_modes_save),
            )

    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=save_path,
        success=True,
        executed=True,
    )


def _run_pod_like(
    spec: AnalyzeSpec,
    analyzer_cls: Any,
    compute_fn: str,
    *,
    dry_run: bool,
    extra_kwargs: dict[str, Any] | None = None,
    compute_kwargs: dict[str, Any] | None = None,
) -> RunOutcome:
    file_path, data_loader, results_dir, figures_dir = _prepare_common_run(spec, dry_run=dry_run)
    if dry_run:
        return _make_dry_run_outcome(spec, results_dir, figures_dir)

    common_kwargs = {
        "file_path": file_path,
        "results_dir": str(results_dir),
        "figures_dir": str(figures_dir),
        "data_loader": data_loader,
        "spatial_weight_type": spec.case.spatial_weight_type,
        "use_parallel": spec.case.use_parallel,
    }
    analyzer = analyzer_cls(**common_kwargs, **(extra_kwargs or {}))

    analyzer.load_and_preprocess()
    _apply_snapshot_limit(analyzer, spec)
    getattr(analyzer, compute_fn)(**(compute_kwargs or {}))
    analyzer.save_results()

    if spec.case.generate_plots:
        analyzer.plot_eigenvalues()
        plotted_volumetric = _maybe_plot_volumetric_modes(
            analyzer,
            plot_n_modes=min(2, spec.case.n_modes_save),
            slices_kwargs={"delay_idx": 0} if isinstance(analyzer, STPODAnalyzer) else None,
            iso_kwargs={"delay_idx": 0} if isinstance(analyzer, STPODAnalyzer) else None,
        )
        if not plotted_volumetric:
            if isinstance(analyzer, STPODAnalyzer):
                analyzer.plot_modes(plot_n_modes=min(2, spec.case.n_modes_save))
            else:
                analyzer.plot_modes(plot_n_modes=min(2, spec.case.n_modes_save), modes_per_fig=2)
        if hasattr(analyzer, "plot_time_coefficients"):
            analyzer.plot_time_coefficients(n_coeffs_to_plot=min(2, spec.case.n_modes_save))
        if hasattr(analyzer, "plot_cumulative_energy"):
            analyzer.plot_cumulative_energy()

    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=_find_latest_result_file(results_dir),
        success=True,
        executed=True,
    )


def _run_dmd(spec: AnalyzeSpec, *, dry_run: bool) -> RunOutcome:
    file_path, data_loader, results_dir, figures_dir = _prepare_common_run(spec, dry_run=dry_run)
    if dry_run:
        return _make_dry_run_outcome(spec, results_dir, figures_dir)

    dmd_kwargs: dict[str, Any] = {
        "file_path": file_path,
        "results_dir": str(results_dir),
        "figures_dir": str(figures_dir),
        "data_loader": data_loader,
        "spatial_weight_type": spec.case.spatial_weight_type,
        "n_modes_save": int(spec.params.get("n_modes_save", spec.case.n_modes_save)),
        "rank": _coerce_rank(spec.params.get("rank", spec.case.rank)),
        "use_parallel": spec.case.use_parallel,
    }
    # None on the case means leave the analyzer default alone (one source of truth).
    energy_fraction = _coerce_energy_fraction(spec.params.get("energy_fraction", spec.case.energy_fraction))
    if energy_fraction is not None:
        dmd_kwargs["energy_fraction"] = energy_fraction
    analyzer = DMDAnalyzer(**dmd_kwargs)
    analyzer.load_and_preprocess()
    _apply_snapshot_limit(analyzer, spec)

    _hodmd_variants = {"hodmd": "ls", "tls_hodmd": "tls"}
    if spec.method in _hodmd_variants:
        delays = int(spec.params.get("delays", spec.case.embedding_dim))
        if delays < 2:
            raise ValueError(f"{spec.method} requires delays >= 2.")
        analyzer.perform_dmd(
            method=_hodmd_variants[spec.method],
            delays=delays,
            named_variant=spec.method,
        )
    else:
        analyzer.perform_dmd(
            method=str(spec.params.get("method", "ls")),
            delays=int(spec.params.get("delays", 1)),
        )
    analyzer.save_results()

    if spec.case.generate_plots:
        analyzer.plot_eigenvalues()
        if not _maybe_plot_volumetric_modes(analyzer, plot_n_modes=min(2, spec.case.n_modes_save)):
            analyzer.plot_modes(plot_n_modes=min(2, spec.case.n_modes_save), modes_per_fig=2)
        analyzer.plot_time_coefficients(n_coeffs_to_plot=min(2, spec.case.n_modes_save))
        analyzer.plot_cumulative_energy()

    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=_find_latest_result_file(results_dir),
        success=True,
        executed=True,
    )


def _run_spod(spec: AnalyzeSpec, *, dry_run: bool) -> RunOutcome:
    file_path, data_loader, results_dir, figures_dir = _prepare_common_run(spec, dry_run=dry_run)
    if dry_run:
        return _make_dry_run_outcome(spec, results_dir, figures_dir)

    analyzer = SPODAnalyzer(
        file_path=file_path,
        nfft=int(spec.params.get("nfft", spec.case.nfft)),
        overlap=float(spec.params.get("overlap", spec.case.overlap)),
        results_dir=str(results_dir),
        figures_dir=str(figures_dir),
        data_loader=data_loader,
        spatial_weight_type=spec.case.spatial_weight_type,
        use_parallel=spec.case.use_parallel,
    )
    analyzer.load_and_preprocess()
    _apply_snapshot_limit(analyzer, spec)
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.save_results()

    if spec.case.generate_plots:
        analyzer.plot_eigenvalues()
        dominant_idx = int(np.argmax(analyzer.eigenvalues[:, 0]))
        if not _maybe_plot_volumetric_modes(
            analyzer,
            plot_n_modes=min(2, analyzer.modes.shape[2]),
            slices_kwargs={"freqs_to_plot": [dominant_idx]},
            iso_kwargs={"freqs_to_plot": [dominant_idx]},
        ):
            analyzer.plot_modes(
                freqs_to_plot=[dominant_idx],
                plot_n_modes=min(2, analyzer.modes.shape[2]),
                modes_per_fig=2,
            )
        analyzer.plot_cumulative_energy()

    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=_find_latest_result_file(results_dir),
        success=True,
        executed=True,
    )


def _run_bsmd(spec: AnalyzeSpec, *, dry_run: bool) -> RunOutcome:
    file_path, data_loader, results_dir, figures_dir = _prepare_common_run(spec, dry_run=dry_run)
    if dry_run:
        return _make_dry_run_outcome(spec, results_dir, figures_dir)

    analyzer = BSMDAnalyzer(
        file_path=file_path,
        nfft=int(spec.params.get("nfft", spec.case.nfft)),
        overlap=float(spec.params.get("overlap", spec.case.overlap)),
        results_dir=str(results_dir),
        figures_dir=str(figures_dir),
        data_loader=data_loader,
        spatial_weight_type=spec.case.spatial_weight_type,
        use_parallel=spec.case.use_parallel,
        max_qhat_gb=float(spec.params.get("max_qhat_gb", 4.0)),
    )
    analyzer.load_and_preprocess()
    _apply_snapshot_limit(analyzer, spec)
    analyzer.compute_fft_blocks()
    analyzer.perform_bsmd()
    analyzer.save_results()

    if spec.case.generate_plots:
        analyzer.plot_energy_map()
        if not _maybe_plot_volumetric_modes(analyzer, plot_n_modes=2):
            analyzer.plot_modes(plot_n_modes=2)

    return RunOutcome(
        run_id=spec.run_id,
        method=spec.method,
        case_name=spec.case.name,
        results_dir=results_dir,
        figures_dir=figures_dir,
        results_path=_find_latest_result_file(results_dir),
        success=True,
        executed=True,
    )


def analyze_from_spec(spec: AnalyzeSpec, *, dry_run: bool = False) -> RunOutcome:
    """Run one analysis spec or print the dry-run plan."""
    dispatch = {
        "pod": lambda: _run_pod_like(
            spec,
            PODAnalyzer,
            "perform_pod",
            dry_run=dry_run,
            extra_kwargs={"n_modes_save": spec.case.n_modes_save},
            compute_kwargs={"solver": str(spec.params.get("solver", "eigh"))},
        ),
        "mpod": lambda: _run_pod_like(
            spec,
            MPODAnalyzer,
            "perform_mpod",
            dry_run=dry_run,
            extra_kwargs={
                "n_modes_save": spec.case.n_modes_save,
                "band_edges": spec.params.get("band_edges"),
                "band_scale": str(spec.params.get("band_scale", "hz")),
                "filter_kind": str(spec.params.get("filter_kind", "rectangular")),
            },
        ),
        "psd_pod": lambda: _run_psd_pod(spec, dry_run=dry_run),
        "dmd": lambda: _run_dmd(spec, dry_run=dry_run),
        "hodmd": lambda: _run_dmd(spec, dry_run=dry_run),
        "tls_hodmd": lambda: _run_dmd(spec, dry_run=dry_run),
        "spod": lambda: _run_spod(spec, dry_run=dry_run),
        "bsmd": lambda: _run_bsmd(spec, dry_run=dry_run),
        "stpod": lambda: _run_pod_like(
            spec,
            STPODAnalyzer,
            "perform_stpod",
            dry_run=dry_run,
            extra_kwargs={
                "embedding_dim": int(spec.params.get("embedding_dim", spec.case.embedding_dim)),
                "n_modes_save": spec.case.n_modes_save,
            },
        ),
    }
    outcome = dispatch[spec.method]()
    state = "DRY-RUN" if not outcome.executed else "DONE"
    print(
        f"[{state}] {spec.case.name}:{spec.run_id} -> {METHOD_REGISTRY[spec.method].cli_name} "
        f"(results={outcome.results_dir}, figures={outcome.figures_dir})"
    )
    return outcome


def analyze_from_config(
    config_path: str | Path,
    *,
    method: str,
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> RunOutcome:
    """Create one analysis spec from a case config plus CLI/Python overrides."""
    resolved = Path(config_path).expanduser().resolve()
    overrides = dict(overrides or {})
    case = _apply_case_overrides(load_case_spec(resolved), overrides)
    params = dict(overrides)
    for case_key in (
        "generate_plots",
        "results_root",
        "figures_root",
        "spatial_weight_type",
        "n_modes_save",
        "nfft",
        "overlap",
        "embedding_dim",
        "use_parallel",
    ):
        params.pop(case_key, None)
    normalized_method = normalize_method_name(method)
    resolved_run_id = run_id or f"{normalized_method}_cli"
    spec = AnalyzeSpec(
        run_id=resolved_run_id,
        method=normalized_method,
        case=case,
        params=params,
        config_path=resolved,
    )
    return analyze_from_spec(spec, dry_run=dry_run)


def _print_run_plan(collection: RunCollectionSpec) -> None:
    print("=" * 72)
    print(collection.name)
    print("=" * 72)
    print(collection.description)
    if collection.nested_configs:
        print("Nested configs:")
        for config_path in collection.nested_configs:
            print(f"  - {config_path}")
        return

    print(f"Case: {collection.analyses[0].case.name}")
    print("Runs:")
    for spec in collection.analyses:
        extras = ", ".join(f"{k}={v}" for k, v in sorted(spec.params.items()))
        line = f"  - {spec.run_id}: {METHOD_REGISTRY[spec.method].cli_name}"
        if extras:
            line += f" ({extras})"
        print(line)


def run_from_config(config_path: str | Path, *, dry_run: bool = False) -> list[RunOutcome]:
    """Run a config-defined set of analyses, including suites of nested configs."""
    resolved = Path(config_path).expanduser().resolve()
    collection = _load_run_collection(resolved)
    _print_run_plan(collection)

    if collection.nested_configs:
        if dry_run:
            print("Dry run only; no analyses executed.")
            return []
        outcomes: list[RunOutcome] = []
        for nested_path in collection.nested_configs:
            outcomes.extend(run_from_config(nested_path, dry_run=False))
        return outcomes

    if dry_run:
        for spec in collection.analyses:
            analyze_from_spec(spec, dry_run=True)
        print("Dry run only; no analyses executed.")
        return []

    outcomes = [analyze_from_spec(spec, dry_run=False) for spec in collection.analyses]
    return outcomes


def discover_examples(root: str | Path | None = None) -> list[ExampleInfo]:
    """Discover example configs under the repo ``examples/`` directory."""
    examples: list[ExampleInfo] = []
    if root is not None:
        search_root = Path(root).expanduser().resolve()
        if not search_root.is_dir():
            return []
        paths = sorted(search_root.glob("*.jsonc"))
    else:
        repo_examples = examples_root()
        if repo_examples.is_dir() and (repo_examples / "run_benchmarks.jsonc").is_file():
            paths = sorted(repo_examples.glob("*.jsonc"))
        else:
            paths = sorted(
                Path(str(resource))
                for resource in packaged_examples_root().iterdir()
                if resource.name.endswith(".jsonc")
            )

    for config_path in paths:
        payload = load_jsonc(config_path)
        kind = "suite" if isinstance(payload.get("configs"), list) else "case"
        examples.append(
            ExampleInfo(
                name=config_path.stem,
                config_path=config_path,
                kind=kind,
                title=str(payload.get("name") or config_path.stem),
                description=str(payload.get("description") or config_path.stem),
                payload=payload,
            )
        )
    return examples


def get_example_info(name: str, root: str | Path | None = None) -> ExampleInfo:
    """Return metadata for one discovered example config."""
    for info in discover_examples(root):
        if info.name == name:
            return info
    raise ValueError(f"Unknown example '{name}'.")


def load_example_payload(name: str, root: str | Path | None = None) -> dict[str, Any]:
    """Load one discovered example config as a plain dictionary."""
    return get_example_info(name, root=root).payload


def inspect_results(path: str | Path) -> dict[str, Any]:
    """Inspect one result file or result directory and return a plain summary.

    HDF5 paths go through :func:`openmodalpy.core.results.read_results` so
    legacy capitalised dataset names appear under their canonical keys.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        files = sorted([candidate for candidate in resolved.iterdir() if candidate.suffix in {".hdf5", ".h5", ".json"}])
        return {
            "path": str(resolved),
            "type": "directory",
            "entries": [candidate.name for candidate in files],
        }

    if resolved.suffix == ".json":
        return {
            "path": str(resolved),
            "type": "json",
            "payload": json.loads(resolved.read_text()),
        }

    if resolved.suffix not in {".hdf5", ".h5"}:
        raise ValueError(f"Unsupported result file type: {resolved}")

    from openmodalpy.core.results import read_results

    res = read_results(resolved)
    datasets: dict[str, dict[str, Any]] = {}
    for name in (
        "modes",
        "eigenvalues",
        "time_coefficients",
        "freq",
        "st",
        "modes1",
        "modes2",
        "triads",
        "amplitudes",
        "omega",
        "x",
        "y",
        "z",
        "W",
        "temporal_mean",
        "energy_map",
        "FFTBlocks",
    ):
        value = getattr(res, name, None)
        if value is not None:
            datasets[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    for name, value in res.extra.items():
        datasets[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    return {
        "path": str(resolved),
        "type": "hdf5",
        "datasets": datasets,
        "attrs": res.attrs,
    }


def print_results_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a result summary returned by :func:`inspect_results`."""
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def run_config_entrypoint(default_config: Path, description: str | None = None) -> None:
    """Small argparse frontend used by the example wrapper scripts."""
    parser = argparse.ArgumentParser(description=description or "Run one OpenModalPy config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the JSONC config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and print the planned run without executing it.",
    )
    args = parser.parse_args()
    run_from_config(args.config.resolve(), dry_run=args.dry_run)
