"""Speed and memory tripwire for DMD, POD, and SPOD on the cylinder wake.

Nothing else in this suite measures speed or memory, so a large regression can
land unseen. These three cases record wall time and peak Python heap use, and
they fail only when a number goes far past its ceiling.

This is not a benchmark. The numbers move with the machine and with load, and
a single run is not a measurement.

WHAT THE MEMORY NUMBER COUNTS. ``tracemalloc.get_traced_memory()[1]`` counts
Python heap allocations. It does not see the scratch space BLAS and LAPACK
allocate, and it is not the resident set. ``resource.getrusage`` was tried
first and rejected: ``ru_maxrss`` is a high-water mark for the whole process,
so it reports a delta of zero once the interpreter has passed these sizes.

BASELINE. Seven runs of this file on an idle 24-core AMD Ryzen 9 9900X, Linux,
OpenBLAS, at the package default of one BLAS thread, 2026-09-02:

    case                best    median   worst    peak
    dmd_embed4_rank10   0.836   0.927    1.098 s  163.9 MB
    pod_nmodes10        0.055   0.057    0.070 s   48.2 MB
    spod_nfft64_ov50    0.824   0.838    0.851 s   70.0 MB

The numbers come from the tests themselves, not from a separate script. A
script that leaves out one step measures a different thing: an early SPOD
measurement that skipped ``compute_fft_blocks`` gave 0.207 s, four times
below what the test does.

THE CEILINGS AND WHY THEY HAVE THAT SHAPE. Time is ``max(10 * median, 5 s)``.
The multiplier alone is not safe for POD: ten times a 57 ms median is 0.57 s,
and a shared runner can spend that much on noise. The 5 s floor keeps POD far
from an accidental failure. DMD and SPOD are slow enough that the multiplier
decides, at 9.3 s and 8.4 s.

Memory uses 3x the peak, which can be tighter because the Python heap total
does not move between runs at all: all seven gave the same three numbers.

These ceilings catch a 25x-class regression, which is the size of the DMD
slowdown that commit f319be3 removed. They will not catch a 30 percent drift,
and they are not meant to.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

from openmodalpy import DMDAnalyzer, PODAnalyzer, SPODAnalyzer
from openmodalpy.example_data import generate_example_dataset

# One record per case, drained by ``pytest_terminal_summary`` in conftest.py.
# The summary is written after the tests, so a progress bar cannot overwrite it
# and pytest does not capture it.
PERF_RECORDS: list[str] = []

TIME_FLOOR_SECONDS = 5.0
MEMORY_MARGIN = 3.0
TIME_MARGIN = 10.0


def _time_ceiling(median_seconds: float) -> float:
    """Return the wall-time ceiling for a case with this measured median."""
    return max(TIME_MARGIN * median_seconds, TIME_FLOOR_SECONDS)


def _record(name: str, run: Callable[[], None], median_seconds: float, peak_mb: float) -> None:
    """Run one case, record its numbers, and check both ceilings."""
    tracemalloc.start()
    started = time.perf_counter()
    run()
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    measured_mb = peak_bytes / 1e6
    PERF_RECORDS.append(f"PERF {name} seconds={elapsed:.3f} peak_mb={measured_mb:.1f}")

    time_ceiling = _time_ceiling(median_seconds)
    memory_ceiling = MEMORY_MARGIN * peak_mb
    assert elapsed < time_ceiling, (
        f"{name}: {elapsed:.3f} s is past the {time_ceiling:.2f} s ceiling (baseline median {median_seconds:.3f} s)"
    )
    assert measured_mb < memory_ceiling, (
        f"{name}: {measured_mb:.1f} MB is past the {memory_ceiling:.1f} MB ceiling (baseline peak {peak_mb:.1f} MB)"
    )


def test_dmd_delay_embedded_stays_fast(tmp_path: Path) -> None:
    """DMD with four delays holds the speed commit f319be3 gave it."""
    data = generate_example_dataset("cylinder_wake")

    def run() -> None:
        analyzer = DMDAnalyzer(
            rank=10,
            n_modes_save=10,
            data=data,
            spatial_weight_type="uniform",
            results_dir=str(tmp_path),
            figures_dir=str(tmp_path),
        )
        analyzer.load_and_preprocess()
        analyzer.perform_dmd(embedding_dim=4)

    _record("dmd_embed4_rank10", run, median_seconds=0.927, peak_mb=163.9)


def test_pod_stays_fast(tmp_path: Path) -> None:
    """POD on the same field stays far below its ceiling."""
    data = generate_example_dataset("cylinder_wake")

    def run() -> None:
        analyzer = PODAnalyzer(
            n_modes_save=10,
            data=data,
            spatial_weight_type="uniform",
            results_dir=str(tmp_path),
            figures_dir=str(tmp_path),
        )
        analyzer.load_and_preprocess()
        analyzer.perform_pod()

    _record("pod_nmodes10", run, median_seconds=0.057, peak_mb=48.2)


def test_spod_stays_fast(tmp_path: Path) -> None:
    """SPOD over Welch blocks stays far below its ceiling."""
    data = generate_example_dataset("cylinder_wake")

    def run() -> None:
        analyzer = SPODAnalyzer(
            nfft=64,
            overlap=0.5,
            n_modes_save=10,
            data=data,
            spatial_weight_type="uniform",
            results_dir=str(tmp_path),
            figures_dir=str(tmp_path),
        )
        analyzer.load_and_preprocess()
        analyzer.compute_fft_blocks()
        analyzer.perform_spod()

    _record("spod_nfft64_ov50", run, median_seconds=0.838, peak_mb=70.0)
