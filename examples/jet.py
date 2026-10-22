#!/usr/bin/env python3
"""Run the OpenModalPy jet experimental case."""

from pathlib import Path

import run_benchmarks

if __name__ == "__main__":
    run_benchmarks.run_single_case_cli(
        default_config=Path(__file__).with_suffix(".jsonc"),
        description="Run the OpenModalPy jet experimental case.",
    )
