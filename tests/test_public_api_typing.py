"""Guards for the public typed surface: py.typed, exported return types, CLI help."""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import numpy as np

import openmodalpy as om
from openmodalpy import cli
from openmodalpy.commands import METHOD_REGISTRY


def test_py_typed_marker_exists() -> None:
    """Installed packages need the PEP 561 marker next to the package root."""
    package_root = Path(om.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
    # Also assert the source-tree marker when running editable, so a bare
    # delete of src/openmodalpy/py.typed fails the suite (gate mutation).
    src_marker = Path(__file__).resolve().parents[1] / "src" / "openmodalpy" / "py.typed"
    assert src_marker.is_file()


def test_all_public_return_types_are_exported() -> None:
    """Walk every name in ``__all__``; classes returned by public functions must be public."""
    required = ("MethodInfo", "ExampleInfo", "RunCollectionSpec")
    for name in required:
        assert name in om.__all__, f"{name} missing from __all__"
        assert hasattr(om, name), f"{name} not importable from openmodalpy"

    public_classes = {getattr(om, name) for name in om.__all__ if inspect.isclass(getattr(om, name, None))}
    unexported: list[str] = []
    for name in om.__all__:
        obj = getattr(om, name)
        if not inspect.isfunction(obj):
            continue
        hints = typing.get_type_hints(obj)
        ret = hints.get("return")
        parts = getattr(ret, "__args__", ()) or (ret,)
        for part in parts:
            if (
                inspect.isclass(part)
                and getattr(part, "__module__", "").startswith("openmodalpy")
                and part not in public_classes
            ):
                unexported.append(f"{name} -> {part.__name__}")
    assert not unexported, f"public functions returning unexported types: {unexported}"


def test_example_generators_are_exported_and_callable() -> None:
    """The three synthetic-dataset generators and the dispatcher are public and callable."""
    dg = om.generate_double_gyre(Nx=8, Ny=6, Nt=10)
    assert dg["q"].shape == (10, 8 * 6)

    tg = om.generate_taylor_green(Nx=8, Ny=6, Nt=10)
    assert tg["q"].shape == (10, 8 * 6)

    cw = om.generate_cylinder_wake(Nx=8, Ny=6, Nt=10)
    assert cw["q"].shape == (10, 8 * 6)

    via_dispatch = om.generate_example_dataset("double_gyre", {"Nx": 8, "Ny": 6, "Nt": 10})
    assert via_dispatch["q"].shape == (10, 8 * 6)


def test_example_payload_accessors_are_exported_and_callable() -> None:
    """``get_example_info``/``load_example_payload`` are public and resolve a real config."""
    discovered = om.discover_examples()
    assert discovered, "no example configs discovered"
    name = discovered[0].name

    info = om.get_example_info(name)
    assert info.name == name

    payload = om.load_example_payload(name)
    assert payload == info.payload


def test_end_to_end_pod_from_generator_using_public_api_only(tmp_path) -> None:
    """A user touching only ``import openmodalpy`` names can go generator -> POD -> saved file."""
    data = om.generate_double_gyre(Nx=8, Ny=6, Nt=10)
    analyzer = om.PODAnalyzer(data=data, results_dir=tmp_path, figures_dir=tmp_path)
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.save_results("pod_from_generator.hdf5")

    saved_path = tmp_path / "pod_from_generator.hdf5"
    assert saved_path.exists()

    results = om.read_results(saved_path)
    eigenvalues = results.eigenvalues
    assert eigenvalues is not None
    assert eigenvalues.size > 0
    assert np.all(np.isfinite(eigenvalues))
    assert bool((eigenvalues[:-1] >= eigenvalues[1:]).all())


def test_all_public_names_include_examples_and_resolve() -> None:
    """The six new example-generator/accessor names are exported and resolve via getattr."""
    required = {
        "generate_double_gyre",
        "generate_taylor_green",
        "generate_cylinder_wake",
        "generate_example_dataset",
        "get_example_info",
        "load_example_payload",
    }
    assert required <= set(om.__all__)
    for name in om.__all__:
        assert hasattr(om, name), f"{name} in __all__ but not resolvable via getattr"


def test_cli_analyze_help_matches_method_registry(monkeypatch) -> None:
    """Top-level parser help must list every METHOD_REGISTRY cli_name."""
    # Argparse wraps help text to the terminal width, and it will break a long line
    # mid-token ("tls-\nhodmd"), which no amount of whitespace collapsing puts back
    # together. Pin a wide terminal so the check does not depend on the window the
    # suite happens to run in.
    monkeypatch.setenv("COLUMNS", "400")
    help_text = " ".join(cli.build_parser().format_help().split())
    registry_names = {info.cli_name for info in METHOD_REGISTRY.values()}
    assert registry_names, "METHOD_REGISTRY is empty"
    missing = sorted(name for name in registry_names if name not in help_text)
    assert not missing, f"CLI help missing registry method names: {missing}"
