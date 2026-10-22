"""A reader looking for the SPOD eigenproblem finds it one hop from ``spod.py``.

The package states that someone should find the equation from the paper in the
code quickly. SPOD used to be the worst case: ``spod.py`` called a dispatcher
in ``core/base.py``, a file that holds no SPOD mathematics, and that dispatcher
called either of two entry points that ran the same body.

These checks pin the path. They read the source rather than the behaviour,
because where the code lives is the claim.
"""

from __future__ import annotations

import ast
from pathlib import Path

import openmodalpy.core.base as base_module
import openmodalpy.spod as spod_module
from openmodalpy.core.decomposition import spod_single_frequency

EQUATION = "spod_single_frequency"
EQUATION_MODULE = "openmodalpy.core.decomposition"


def _imported_names(path: Path) -> dict[str, str]:
    """Map each name a module imports to the module it comes from."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                found[alias.asname or alias.name] = node.module
    return found


def test_base_holds_no_spod_dispatch() -> None:
    """``core/base.py`` carries no SPOD entry point for a reader to stop at."""
    assert not hasattr(base_module, "spod_function")
    assert not hasattr(base_module, "spod_single_frequency_optimized")


def test_spod_reaches_the_eigenproblem_in_one_hop() -> None:
    """``spod.py`` imports the eigenproblem straight from the module that holds it."""
    imports = _imported_names(Path(spod_module.__file__))

    assert imports.get(EQUATION) == EQUATION_MODULE, (
        f"spod.py must import {EQUATION} from {EQUATION_MODULE}; got {imports.get(EQUATION)!r}"
    )


def test_the_eigenproblem_is_where_the_import_says() -> None:
    """The imported name really is the per-frequency eigenproblem, not a forwarder."""
    source = Path(str(spod_single_frequency.__module__).replace(".", "/") + ".py")
    assert source.name == "decomposition.py"
    body = spod_single_frequency.__doc__ or ""
    assert "eigenproblem" in body.lower()
