"""Guard: oracle tests must not import the library sign/phase or order rules.

The oracle side deliberately duplicates the canonical sign rule and the DMD
spectrum-order rule so a change to the library turns those tests red. Importing
the library helper makes both sides agree by construction and the comparison
stops checking anything. This meta-test fails loudly at the moment of re-coupling.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Module that OWNS the canonical sign/phase rule and is allowed to import it
# from the library. Every other tests/*.py module must express the rule itself
# (or call the test-side copy in reference_helpers) so a library change is visible.
RULE_OWNER = "tests/test_decomposition.py"

_FORBIDDEN = frozenset(
    {
        "canonicalize_modes",
        "canonical_pivot_index",
        "CANONICAL_TIE_RTOL",
        "canonical_eigenvalue_order",
    }
)

_WHY = (
    "oracle tests must express the sign rule themselves, so that changing the "
    "library rule turns them red; importing it makes them agree with the library "
    "by construction and they stop checking anything"
)

_TESTS_DIR = Path(__file__).resolve().parent


def _violations_in(path: Path) -> list[str]:
    """Return human-readable import/attribute hits for the forbidden names."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _FORBIDDEN:
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    hits.append(f"line {node.lineno}: from {node.module or '?'} import {alias.name}{as_part}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # ``import pkg.canonicalize_modes`` (pathological) or a bare name.
                leaf = alias.name.rsplit(".", 1)[-1]
                if leaf in _FORBIDDEN:
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    hits.append(f"line {node.lineno}: import {alias.name}{as_part}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN:
            hits.append(f"line {node.lineno}: attribute access .{node.attr}")
    return hits


def test_oracle_tests_do_not_import_the_library_sign_rule():
    """No tests/*.py module except the rule owner may import the library sign rule."""
    offenders: list[str] = []
    for path in sorted(_TESTS_DIR.glob("*.py")):
        rel = f"tests/{path.name}"
        if rel == RULE_OWNER:
            continue
        hits = _violations_in(path)
        if hits:
            detail = "; ".join(hits)
            offenders.append(f"{rel}: {detail}")

    assert not offenders, (
        f"forbidden library sign-rule import(s) outside {RULE_OWNER}. {_WHY}. Offenders: " + " | ".join(offenders)
    )


# --- Generalised rule ---
# A file that verifies module X may call X's PUBLIC entry points (exercising
# the code under test is the point), but must not import X's PRIVATE
# (underscore) helpers: those are the implementation's own formula carriers,
# and building expected values from them is exactly how twins regrow.

VERIFIED_MODULE_REGISTRY: dict[str, str] = {
    "test_psd_pod_numerics.py": "openmodalpy.psd_pod",
    "test_bsmd_core.py": "openmodalpy.bsmd",
    "test_spod_function.py": "openmodalpy.spod",
    "test_dmd.py": "openmodalpy.dmd",
    "test_welch_analytical.py": "openmodalpy.core.welch",
}

# Exceptions where the private symbol is the SUBJECT UNDER TEST (unit tests of
# the helper itself), not an expectation builder. Every entry needs a reason;
# anything absent from here is a violation.
ALLOWED_PRIVATE_IMPORTS: dict[tuple[str, str], str] = {
    ("test_dmd.py", "_delay_embed"): (
        "test_delay_embed_{shape,d1_identity,values} are unit tests of the "
        "helper itself; expected values there are built by hand (vstack of "
        "slices), not by the helper."
    ),
}


def _private_imports_from(path: Path, module: str) -> list[str]:
    """Report imports of underscore-prefixed names from ``module`` in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                if alias.name.startswith("_"):
                    hits.append(f"line {node.lineno}: from {module} import {alias.name}")
    return hits


def test_verified_module_files_do_not_import_their_private_helpers():
    """No file verifying module X imports X's private numerical helpers."""
    offenders: list[str] = []
    for rel, module in sorted(VERIFIED_MODULE_REGISTRY.items()):
        path = _TESTS_DIR / rel
        if not path.exists():
            continue
        for hit in _private_imports_from(path, module):
            symbol = hit.rsplit(" import ", 1)[-1]
            if (rel, symbol) in ALLOWED_PRIVATE_IMPORTS:
                continue
            offenders.append(f"{rel}: {hit}")

    assert not offenders, (
        "files verifying a module must not import that module's private "
        "helpers — build expectations independently instead. Offenders: " + " | ".join(offenders)
    )
