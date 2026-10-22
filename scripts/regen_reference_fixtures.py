#!/usr/bin/env python3
"""Regenerate committed analytic reference fixtures under tests/fixtures/reference/.

No CLI flags (project convention). Writes one JSON file per built-in generator
with the POD energy-fraction spectrum and DMD |λ|/phase on the grid the packaged
example config for that generator runs.
Byte-identical on a clean checkout for a fixed environment (single-thread BLAS).

Not shipped in the wheel — developers only:
    uv run python scripts/regen_reference_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Shared helpers live next to the comparison test (one definition for both).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from openmodalpy.config_io import load_jsonc  # noqa: E402
from openmodalpy.example_data import GENERATORS  # noqa: E402
from tests.reference_helpers import compute_reference_spectra  # noqa: E402

OUT_DIR = ROOT / "tests" / "fixtures" / "reference"
PACKAGED_EXAMPLES_DIR = ROOT / "src" / "openmodalpy" / "examples"

# Names which generators have fixtures. Non-config extras live here; grids
# and seed (Nx/Ny/Nt/seed) are read from the packaged example configs — single
# source. cylinder_wake states seed 42 in its packaged config so the fixture
# records the full generation contract without a hard-coded merge.
_GENERATOR_EXTRAS: dict[str, dict[str, Any]] = {
    "double_gyre": {},
    "taylor_green": {},
    "cylinder_wake": {},
}


def _params_from_packaged(generator: str) -> dict[str, Any]:
    """Build generator_params: packaged data.params plus any non-config extras."""
    path = PACKAGED_EXAMPLES_DIR / f"{generator}.jsonc"
    if not path.is_file():
        raise FileNotFoundError(f"no packaged config at {path}")
    doc = load_jsonc(path)
    case = doc.get("case", doc)
    params = dict(case.get("data", {}).get("params", {}))
    params.update(_GENERATOR_EXTRAS.get(generator, {}))
    return params


# Populated at import from the packaged configs (gate and regen both read this).
GENERATOR_PARAMS: dict[str, dict[str, Any]] = {name: _params_from_packaged(name) for name in _GENERATOR_EXTRAS}

# Rank is per generator, set by conditioning rather than by taste.
#
# Exact DMD forms Atilde = (u^H X2 v) / s, so it DIVIDES by the kept singular
# values. Keeping a mode with s_r/s_0 = q amplifies machine round-off by 1/q,
# giving an eigenvalue error of about eps/q. A fixture may only assert to rtol
# if that amplified error stays below it:
#
#     eps / (s_r/s_0)  <  rtol          (eps = 2.2e-16)
#
# double_gyre decays fastest: s[9]/s[0] = 1.9e-12, so rank 10 implies an error
# near 1e-4 -- a hundred times looser than rtol=1e-6. That is why the fixture
# reproduced perfectly on the machine that wrote it and failed on every CI
# platform: the arithmetic is not reproducible there at 1e-6, and the recorded
# values were partly round-off. Rank 8 keeps s[7]/s[0] = 4.2e-9, implying 5.2e-8
# and leaving rtol=1e-6 honest.
#
# cylinder_wake at rank 10 is fine: s9/s0 ~ 2e-3, so the amplified error is
# eps/(s9/s0) ~ 1e-13, far inside rtol. taylor_green is not: its
# s9/s0 is ~ 3e-17, so eps/(s9/s0) is order 1 and rank 10 fails the same
# inequality. The recorded taylor_green fixture is still valid because the
# analyzer truncates to effective rank and warns before those modes are used —
# the fixture records that truncated spectrum, not ten fully-conditioned modes.
N_MODES_BY_GENERATOR: dict[str, int] = {
    "double_gyre": 8,
    "taylor_green": 10,
    "cylinder_wake": 10,
}
N_MODES = 10  # default for a generator not listed above
RTOL = 1e-6
ATOL = 1e-12


def compute_reference(generator: str, params: dict[str, Any], n_modes: int = N_MODES) -> dict[str, Any]:
    """Run POD + DMD on a generator payload; return serialisable spectrum arrays."""
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator!r}; available: {sorted(GENERATORS)}")

    spectra = compute_reference_spectra(generator, params, n_modes=n_modes, rtol=RTOL)

    # Key order: spectrum arrays first so an out-of-band probe that walks leaves
    # hits a spectrum entry (gate step 4 uses named keys; order still kept stable).
    return {
        "generator": generator,
        "pod_energy_fractions": [float(v) for v in spectra["pod_energy_fractions"]],
        "dmd_abs_lambda": [float(v) for v in spectra["dmd_abs_lambda"]],
        "dmd_phase": [float(v) for v in spectra["dmd_phase"]],
        "energy_captured_fraction": float(spectra["energy_captured_fraction"]),
        "generator_params": dict(params),
        "n_modes": int(n_modes),
        "rtol": float(RTOL),
        "atol": float(ATOL),
    }


def write_fixture(doc: dict[str, Any], path: Path) -> None:
    """Write JSON with stable formatting (trailing newline, LF, no sort_keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    if set(GENERATOR_PARAMS) != set(GENERATORS):
        missing = set(GENERATORS) - set(GENERATOR_PARAMS)
        extra = set(GENERATOR_PARAMS) - set(GENERATORS)
        raise SystemExit(f"GENERATOR_PARAMS mismatch: missing={missing}, extra={extra}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, params in GENERATOR_PARAMS.items():
        doc = compute_reference(name, params, n_modes=N_MODES_BY_GENERATOR.get(name, N_MODES))
        out = OUT_DIR / f"{name}.json"
        write_fixture(doc, out)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
