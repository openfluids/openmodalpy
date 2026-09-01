#!/usr/bin/env python3
"""Subcommand CLI frontend for OpenModalPy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from openmodalpy.commands import (
    METHOD_REGISTRY,
    analyze_from_config,
    discover_examples,
    get_example_info,
    get_method_spec,
    inspect_results,
    list_methods,
    load_example_payload,
    normalize_method_name,
    print_results_summary,
    run_from_config,
)


def _configure_cli_logging() -> None:
    """Install a single INFO handler so library log lines reach the terminal.

    The library itself never configures logging; only this CLI entry point does.
    Message-only format keeps CLI output free of level/name decoration.

    The handler goes on the ``openmodalpy`` logger, not the root logger, so the
    CLI shows this package's own progress lines and does not start relaying INFO
    records from matplotlib, h5py or any other dependency.
    """
    pkg_logger = logging.getLogger("openmodalpy")
    if pkg_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)


def _analysis_method(value: str) -> str:
    """Normalize a method name, keeping the registry's own error text.

    argparse throws away the message of a ValueError raised by a ``type``
    callable and prints ``invalid <callable name> value`` instead, which would
    show the user an internal function name rather than the list of methods.
    ArgumentTypeError is printed verbatim, so re-raise as that.
    """
    try:
        return normalize_method_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse tree."""
    parser = argparse.ArgumentParser(
        description="OpenModalPy: a unified Python workflow for modal decomposition of spatiotemporal data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Method list is derived from METHOD_REGISTRY so top-level help and the
    # analysis_method arg stay in lockstep with the registry (no hardcoded names).
    method_names = ", ".join(info.cli_name for info in METHOD_REGISTRY.values())
    analyze = subparsers.add_parser(
        "analyze",
        help=f"Run one analysis family from a case config ({method_names}).",
    )
    analyze.add_argument(
        "analysis_method",
        type=_analysis_method,
        choices=sorted(METHOD_REGISTRY),
        help=f"Method to run ({method_names}).",
    )
    analyze.add_argument("--config", type=Path, required=True, help="Path to the case JSONC config file.")
    analyze.add_argument("--run-id", type=str, default=None, help="Custom run id for outputs.")
    analyze.add_argument("--dry-run", action="store_true", help="Print the resolved analysis without executing it.")
    analyze.add_argument("--no-plots", action="store_true", help="Disable figure generation for this run.")
    analyze.add_argument("--results-dir", type=Path, default=None, help="Override the case results root.")
    analyze.add_argument("--figures-dir", type=Path, default=None, help="Override the case figures root.")
    analyze.add_argument("--weight-type", type=str, default=None, help="Override the spatial weight type.")
    analyze.add_argument("--n-modes", type=int, default=None, help="Override the number of saved modes.")
    analyze.add_argument("--nfft", type=int, default=None, help="Override the FFT block size.")
    analyze.add_argument("--overlap", type=float, default=None, help="Override the FFT overlap fraction.")
    analyze.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override the delay embedding dimension (DMD, HODMD, ST-POD).",
    )
    analyze.add_argument(
        "--band-edges",
        type=str,
        default=None,
        help="Comma-separated mPOD band edges, e.g. 0,0.15,0.35,1.0.",
    )
    analyze.add_argument(
        "--band-scale",
        type=str,
        default=None,
        help="mPOD band scale: 'hz' or 'normalized_nyquist'.",
    )
    analyze.add_argument(
        "--filter-kind",
        type=str,
        default=None,
        help="mPOD filter kind; currently 'rectangular'.",
    )
    analyze.add_argument(
        "--method",
        dest="dmd_method",
        choices=("ls", "tls"),
        default=None,
        help="DMD regression model (used only when analyze dmd).",
    )
    analyze.add_argument(
        "--solver",
        choices=("eigh", "svd"),
        default=None,
        help="POD eigen-solver: 'eigh' or 'svd' (used only when analyze pod).",
    )

    run = subparsers.add_parser("run", help="Run one config-defined analysis suite or example suite.")
    run.add_argument("--config", type=Path, required=True, help="Path to the JSONC config file.")
    run.add_argument("--dry-run", action="store_true", help="Resolve the config and print the planned runs only.")

    examples = subparsers.add_parser("examples", help="Inspect or execute bundled example configs.")
    examples_subparsers = examples.add_subparsers(dest="examples_command", required=True)
    examples_subparsers.add_parser("list", help="List discovered example configs.")
    examples_show = examples_subparsers.add_parser("show", help="Show one example config.")
    examples_show.add_argument("name", type=str, help="Example name, e.g. cavity.")
    examples_run = examples_subparsers.add_parser("run", help="Run one discovered example config.")
    examples_run.add_argument("name", type=str, help="Example name, e.g. cavity.")
    examples_run.add_argument("--dry-run", action="store_true", help="Resolve the example without executing it.")

    methods = subparsers.add_parser("methods", help="Inspect supported method families.")
    methods_subparsers = methods.add_subparsers(dest="methods_command", required=True)
    methods_subparsers.add_parser("list", help="List supported methods.")
    methods_show = methods_subparsers.add_parser("show", help="Show one method.")
    methods_show.add_argument("name", type=str, help="Method name or alias.")

    results = subparsers.add_parser("results", help="Inspect saved result files or directories.")
    results_subparsers = results.add_subparsers(dest="results_command", required=True)
    results_inspect = results_subparsers.add_parser("inspect", help="Inspect one result path.")
    results_inspect.add_argument("path", type=Path, help="Path to a result file or directory.")

    return parser


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.no_plots:
        overrides["generate_plots"] = False
    if args.results_dir is not None:
        overrides["results_root"] = str(args.results_dir)
    if args.figures_dir is not None:
        overrides["figures_root"] = str(args.figures_dir)
    if args.weight_type is not None:
        overrides["spatial_weight_type"] = args.weight_type
    if args.n_modes is not None:
        overrides["n_modes_save"] = args.n_modes
    if args.nfft is not None:
        overrides["nfft"] = args.nfft
    if args.overlap is not None:
        overrides["overlap"] = args.overlap
    if args.embedding_dim is not None:
        overrides["embedding_dim"] = args.embedding_dim
    if args.band_edges is not None:
        overrides["band_edges"] = [float(item) for item in args.band_edges.split(",") if item.strip()]
    if args.band_scale is not None:
        overrides["band_scale"] = args.band_scale
    if args.filter_kind is not None:
        overrides["filter_kind"] = args.filter_kind
    if args.dmd_method is not None:
        overrides["method"] = args.dmd_method
    if args.solver is not None:
        overrides["solver"] = args.solver
    return overrides


def _print_methods_list() -> None:
    for spec in list_methods():
        print(f"{spec.cli_name:8s}  {spec.display_name:8s}  {spec.description}")


def _print_method_show(name: str) -> None:
    spec = get_method_spec(name)
    print(spec.display_name)
    print("-" * len(spec.display_name))
    print(spec.description)
    if spec.parameter_help:
        print("Parameters:")
        for key, text in spec.parameter_help.items():
            print(f"  {key}: {text}")
    print(f"Scope: {spec.implementation_scope}")


def _print_examples_list() -> None:
    examples = discover_examples()
    if not examples:
        print("No example configs were found.")
        return
    for info in examples:
        print(f"{info.name:24s}  {info.kind:5s}  {info.description}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by both the console script and ``python -m openmodalpy``.

    Returns 0 on success and a non-zero exit code on failure of the internal
    unhandled-command fallback. Usage and argument errors still go through
    argparse and raise ``SystemExit`` (typically with code 2); that channel is
    separate from the integer return path.
    """
    _configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        analyze_from_config(
            args.config.resolve(),
            method=args.analysis_method,
            run_id=args.run_id,
            overrides=_collect_overrides(args),
            dry_run=args.dry_run,
        )
        return 0

    if args.command == "run":
        run_from_config(args.config.resolve(), dry_run=args.dry_run)
        return 0

    if args.command == "examples":
        if args.examples_command == "list":
            _print_examples_list()
            return 0
        if args.examples_command == "show":
            info = get_example_info(args.name)
            print(f"# {info.title}")
            print(f"path: {info.config_path}")
            print(json.dumps(load_example_payload(args.name), indent=2))
            return 0
        if args.examples_command == "run":
            info = get_example_info(args.name)
            run_from_config(info.config_path, dry_run=args.dry_run)
            return 0

    if args.command == "methods":
        if args.methods_command == "list":
            _print_methods_list()
            return 0
        if args.methods_command == "show":
            _print_method_show(args.name)
            return 0

    if args.command == "results" and args.results_command == "inspect":
        print_results_summary(inspect_results(args.path))
        return 0

    # Defensive fallback: required subparsers make this unreachable from argv.
    parser.print_usage(sys.stderr)
    sys.stderr.write(f"{parser.prog}: error: Unhandled command tree: {args}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
