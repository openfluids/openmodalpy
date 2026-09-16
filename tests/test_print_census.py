"""Upper-bound census of ``print(`` under ``src/openmodalpy/``.

Phase 1 of the logging seam cleared ``core/`` (was 30). Phase 2 may only lower
the remaining per-module counts; this test fails if a new ``print(`` appears.
"""

from __future__ import annotations

import re
from pathlib import Path

# Upper bounds measured after phase-1 conversion (core/ → 0). Counts are
# inclusive of the one removed SPOD qhat-guard print (spod.py was 31).
PRINT_UPPER_BOUNDS: dict[str, int] = {
    "bsmd.py": 0,
    # Every command writes through cli._emit, which holds the only print.
    "cli.py": 1,
    "commands.py": 13,
    "core/base.py": 0,
    "core/io.py": 0,
    "core/parallel.py": 0,
    "dmd.py": 0,
    "pod.py": 0,
    "psd_pod.py": 0,
    "spod.py": 0,
    "stpod.py": 0,
}

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "openmodalpy"
_PRINT_RE = re.compile(r"print\(")


def _count_prints(path: Path) -> int:
    return len(_PRINT_RE.findall(path.read_text(encoding="utf-8")))


def _module_print_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        n = _count_prints(path)
        if n:
            rel = str(path.relative_to(_SRC_ROOT))
            counts[rel] = n
    return counts


def test_print_census_per_module_upper_bounds():
    """No module may gain prints; core/ must stay at zero."""
    counts = _module_print_counts()

    for rel, upper in PRINT_UPPER_BOUNDS.items():
        actual = counts.get(rel, 0)
        assert actual <= upper, f"{rel}: {actual} print( exceeds upper bound {upper}"

    # Any new file with prints must be added to PRINT_UPPER_BOUNDS deliberately.
    unexpected = sorted(set(counts) - set(PRINT_UPPER_BOUNDS))
    assert not unexpected, f"modules with print( not in census: {unexpected}"


def test_print_census_core_is_zero():
    """core/ is fully on the logging seam after phase 1."""
    core = _SRC_ROOT / "core"
    total = sum(_count_prints(p) for p in core.glob("*.py"))
    assert total == 0


def test_print_census_total_does_not_grow():
    """Whole-tree total can only fall from the post-phase-1 baseline of 204."""
    total = sum(_module_print_counts().values())
    assert total <= 204, f"src/openmodalpy print( total grew to {total} (cap 204)"
