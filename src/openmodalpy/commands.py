"""Shared command core for the OpenModalPy CLI and Python API."""

from __future__ import annotations

import argparse
import importlib.resources
import json
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, TypedDict

from openmodalpy.bsmd import BSMDAnalyzer
from openmodalpy.config_io import load_jsonc, resolve_path
from openmodalpy.core.io import load_data
from openmodalpy.dmd import DMDAnalyzer
from openmodalpy.example_data import generate_example_dataset
from openmodalpy.mpod import MPODAnalyzer
from openmodalpy.pod import PODAnalyzer
from openmodalpy.psd_pod import PSDPODAnalyzer
from openmodalpy.specs import (
    METHOD_REGISTRY,
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


# Accepted keys, written once per level next to the readers below.
# CASE_FIELD_KEYS is the overlap between the case payload and CLI/Python overrides.
TOP_LEVEL_KEYS = frozenset({"kind", "name", "description", "case", "runs", "configs"})
CASE_FIELD_KEYS = frozenset(
    {
        "spatial_weight_type",
        "n_modes_save",
        "rank",
        "energy_fraction",
        "nfft",
        "overlap",
        "embedding_dim",
        "use_parallel",
        "generate_plots",
        "results_root",
        "figures_root",
    }
)
CASE_KEYS = CASE_FIELD_KEYS | frozenset({"name", "description", "case_type", "data"})
DATA_KEYS = frozenset({"kind", "path", "name", "params", "schema"})
RUN_KEYS = frozenset({"id", "method", "params", "enabled"})
VALID_CONFIG_KINDS = frozenset({"analysis-suite", "config-suite"})


def _reject_unknown_keys(
    mapping: dict[str, Any],
    accepted: frozenset[str],
    *,
    config_path: Path,
    level: str,
) -> None:
    """Raise if ``mapping`` carries a key nothing at this level reads."""
    unknown = [key for key in mapping if key not in accepted]
    if not unknown:
        return
    if "spatial_weights" in unknown:
        raise ValueError(
            f"{config_path} key 'spatial_weights' cannot prescribe a metric from a "
            "config file. Pass spatial_weights= to the analyzer through the library "
            "API instead, or set spatial_weight_type to choose a built-in metric."
        )
    shown = ", ".join(repr(key) for key in unknown)
    label = "keys" if len(unknown) > 1 else "key"
    accepted_shown = ", ".join(sorted(accepted))
    raise ValueError(f"{config_path} has unknown {level} {label} {shown}. Accepted keys: {accepted_shown}.")


def _validate_config_kind(payload: dict[str, Any], config_path: Path) -> None:
    """Read top-level ``kind`` and require it to agree with the file contents."""
    if "kind" not in payload:
        return
    kind = payload["kind"]
    accepted = "'analysis-suite', 'config-suite'"
    if kind not in VALID_CONFIG_KINDS:
        raise ValueError(f"{config_path} has invalid kind {kind!r}. Accepted values: {accepted}.")
    has_configs = isinstance(payload.get("configs"), list)
    if kind == "config-suite" and not has_configs:
        raise ValueError(f"{config_path} kind is 'config-suite' but has no 'configs' list.")
    if kind == "analysis-suite" and has_configs:
        raise ValueError(f"{config_path} kind is 'analysis-suite' but contains a 'configs' list.")


def _load_case_spec_from_payload(payload: dict[str, Any], config_path: Path) -> CaseSpec:
    _reject_unknown_keys(payload, TOP_LEVEL_KEYS, config_path=config_path, level="top-level")
    _validate_config_kind(payload, config_path)

    case_payload = payload.get("case")
    if not isinstance(case_payload, dict):
        raise ValueError(f"{config_path} must define a 'case' object.")
    _reject_unknown_keys(case_payload, CASE_KEYS, config_path=config_path, level="case")

    name = str(case_payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"{config_path} case block is missing 'name'.")

    data_payload = case_payload.get("data")
    if not isinstance(data_payload, dict):
        raise ValueError(f"{config_path} case block is missing a 'data' object.")
    _reject_unknown_keys(data_payload, DATA_KEYS, config_path=config_path, level="data")

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
    """Return a copy of ``case`` with supported CLI/Python overrides applied.

    The names read here are ``CASE_FIELD_KEYS`` — the same set the case-payload
    reader accepts for these fields. Method-specific override keys are left for
    ``params`` and are not rejected here.
    """
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
    _reject_unknown_keys(payload, TOP_LEVEL_KEYS, config_path=config_path, level="top-level")
    _validate_config_kind(payload, config_path)
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
        _reject_unknown_keys(run_payload, RUN_KEYS, config_path=config_path, level="run")
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


def _run_psd_pod(spec: AnalyzeSpec, *, dry_run: bool) -> RunOutcome:
    """Thin CLI runner: construct, hand the spec to the analyzer seam."""
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
    analyzer.run_analysis(
        plots=spec.case.generate_plots,
        run_id=spec.run_id,
        snapshot_limit=spec.params.get("max_snapshots"),
    )
    save_path = Path(analyzer.results_path) if analyzer.results_path else _find_latest_result_file(results_dir)

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
    analyzer.run_analysis(
        plots=spec.case.generate_plots,
        snapshot_limit=spec.params.get("max_snapshots"),
        **(compute_kwargs or {}),
    )

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


class _DMDPerformKwargs(TypedDict, total=False):
    """Keyword arguments accepted by ``DMDAnalyzer.perform_dmd``."""

    method: str
    delays: int
    named_variant: str


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

    _hodmd_variants = {"hodmd": "ls", "tls_hodmd": "tls"}
    perform_kwargs: _DMDPerformKwargs
    if spec.method in _hodmd_variants:
        delays = int(spec.params.get("delays", spec.case.embedding_dim))
        if delays < 2:
            raise ValueError(f"{spec.method} requires delays >= 2.")
        perform_kwargs = dict(method=_hodmd_variants[spec.method], delays=delays, named_variant=spec.method)
    else:
        perform_kwargs = dict(
            method=str(spec.params.get("method", "ls")),
            delays=int(spec.params.get("delays", 1)),
        )

    analyzer.run_analysis(
        plots=spec.case.generate_plots,
        snapshot_limit=spec.params.get("max_snapshots"),
        **perform_kwargs,
    )
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
    analyzer.run_analysis(
        plots=spec.case.generate_plots,
        snapshot_limit=spec.params.get("max_snapshots"),
    )
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
    analyzer.run_analysis(
        plots=spec.case.generate_plots,
        snapshot_limit=spec.params.get("max_snapshots"),
    )
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
            dry_run=dry_run,
            extra_kwargs={"n_modes_save": spec.case.n_modes_save},
            compute_kwargs={"solver": str(spec.params.get("solver", "eigh"))},
        ),
        "mpod": lambda: _run_pod_like(
            spec,
            MPODAnalyzer,
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

    # TODO: each spec
    # still loads its case from disk. Cache the loader result per data source
    # here and hand it through data= once the pipelines accept a cache.
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
        _validate_config_kind(payload, config_path)
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
