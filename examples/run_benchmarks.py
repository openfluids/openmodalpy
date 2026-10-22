#!/usr/bin/env python3
"""Thin wrapper around the unified ``openmodalpy run`` command core."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openmodalpy.commands import run_config_entrypoint  # noqa: E402

DEFAULT_CONFIG = Path(__file__).with_suffix(".jsonc")


def run_single_case_cli(default_config: Path, description: str | None = None) -> None:
    """Run one example case from its JSONC config. Each per-case wrapper calls it."""
    run_config_entrypoint(default_config=default_config, description=description)


def main() -> None:
    """Run the example suite defined by ``run_benchmarks.jsonc``."""
    run_config_entrypoint(
        default_config=DEFAULT_CONFIG,
        description="Run the OpenModalPy example suite from a JSONC config.",
    )


if __name__ == "__main__":
    main()
