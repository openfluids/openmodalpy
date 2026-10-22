"""Entry-point exit-code wiring for ``python -m openmodalpy`` and the console script.

``__main__.py`` historically discarded ``main()``'s return value. These tests pin
process exit status so a future non-zero ``return`` cannot fail silently under
``python -m`` only.
"""

from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
import textwrap

import pytest


def test_module_entry_propagates_main_return_code_subprocess() -> None:
    """Process exit status must equal the int returned by ``cli.main``.

    Behavioural (subprocess + patch), not a source grep. Expected to FAIL when
    ``__main__.py`` calls ``main()`` without ``raise SystemExit(...)``.
    """
    code = textwrap.dedent(
        """
        import runpy
        import openmodalpy.cli as c
        c.main = lambda argv=None: 37
        runpy.run_module("openmodalpy", run_name="__main__")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 37, (
        f"__main__ discarded main()'s return value (expected 37, got {proc.returncode}); stderr={proc.stderr!r}"
    )


def test_module_entry_raises_system_exit_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process path: covers ``__main__.py`` under pytest-cov and pins SystemExit."""
    import openmodalpy.cli as cli_mod

    monkeypatch.setattr(cli_mod, "main", lambda argv=None: 37)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("openmodalpy", run_name="__main__")
    assert exc_info.value.code == 37


def test_module_methods_list_exits_zero_and_prints_table() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "openmodalpy", "methods", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "spod" in combined
    assert "pod" in combined


def test_module_no_args_exits_nonzero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "openmodalpy"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_console_script_methods_list_exits_zero_and_prints_table() -> None:
    script = shutil.which("openmodalpy")
    assert script is not None, "console script 'openmodalpy' not on PATH"
    proc = subprocess.run(
        [script, "methods", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "spod" in combined


def test_console_script_no_args_exits_nonzero() -> None:
    script = shutil.which("openmodalpy")
    assert script is not None, "console script 'openmodalpy' not on PATH"
    proc = subprocess.run(
        [script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
