from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pytest

from openmodalpy import analyze_from_config
from openmodalpy.cli import build_parser, main
from openmodalpy.commands import (
    METHOD_REGISTRY,
    discover_examples,
    get_method_spec,
    inspect_results,
    normalize_method_name,
    run_from_config,
)
from openmodalpy.core.base import BaseAnalyzer


def _write_jsonc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def test_method_registry_exposes_mpod_psd_pod_and_hodmd() -> None:
    mpod = get_method_spec("mpod")
    psd_pod = get_method_spec("psd-pod")
    hodmd = get_method_spec("hodmd")
    tls_hodmd = get_method_spec("tls-hodmd")

    assert mpod.method_id == "mpod"
    assert mpod.cli_name == "mpod"
    assert "second-order" in mpod.description
    assert psd_pod.method_id == "psd_pod"
    assert psd_pod.cli_name == "psd-pod"
    assert "Fourier realizations" in psd_pod.description
    assert hodmd.method_id == "hodmd"
    assert "Hankel" in hodmd.description
    assert "embedding_dim" in hodmd.parameter_help
    assert tls_hodmd.method_id == "tls_hodmd"


def test_analyze_from_config_routes_hodmd_aliases(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy analytical case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "embedding_dim": 5,
                "spatial_weight_type": "uniform",
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [],
        },
    )

    captured: list[dict[str, object]] = []

    class FakeDMDAnalyzer:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def run_analysis(self, *, plots, run_id=None, snapshot_limit=None, **kwargs):
            self.load_and_preprocess()
            self.perform_dmd(**kwargs)
            self.save_results()

        def load_and_preprocess(self):
            return None

        def perform_dmd(self, *, method: str, embedding_dim: int, named_variant: str | None = None):
            captured.append(
                {
                    "method": method,
                    "embedding_dim": embedding_dim,
                    "named_variant": named_variant,
                    "results_dir": self._kwargs["results_dir"],
                }
            )

        def save_results(self):
            Path(self._kwargs["results_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(self._kwargs["results_dir"]) / "fake.hdf5").write_text("fake")

        def plot_eigenvalues(self):
            raise AssertionError("plots should be disabled in this test")

        def plot_modes(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_time_coefficients(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_cumulative_energy(self):
            raise AssertionError("plots should be disabled in this test")

    monkeypatch.setattr("openmodalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)

    outcome_hodmd = analyze_from_config(config_path, method="hodmd", overrides={"generate_plots": False})
    outcome_tls = analyze_from_config(config_path, method="tls-hodmd", overrides={"generate_plots": False})

    assert captured[0]["method"] == "ls"
    assert captured[0]["embedding_dim"] == 5
    assert captured[0]["named_variant"] == "hodmd"
    assert captured[1]["method"] == "tls"
    assert captured[1]["embedding_dim"] == 5
    assert captured[1]["named_variant"] == "tls_hodmd"
    assert outcome_hodmd.method == "hodmd"
    assert outcome_tls.method == "tls_hodmd"


def test_analyze_from_config_forwards_dmd_variant_options(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy analytical case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "spatial_weight_type": "uniform",
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [],
        },
    )

    captured: dict[str, object] = {}

    class FakeDMDAnalyzer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run_analysis(self, *, plots, run_id=None, snapshot_limit=None, **kwargs):
            self.load_and_preprocess()
            self.perform_dmd(**kwargs)
            self.save_results()

        def load_and_preprocess(self):
            captured["loaded"] = True

        def perform_dmd(self, *, method: str, embedding_dim: int, named_variant: str | None = None):
            captured["perform"] = {"method": method, "embedding_dim": embedding_dim}

        def save_results(self):
            results_dir = Path(captured["init"]["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "fake.hdf5").write_text("fake")

        def plot_eigenvalues(self):
            raise AssertionError("plots should be disabled in this test")

        def plot_modes(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_time_coefficients(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_cumulative_energy(self):
            raise AssertionError("plots should be disabled in this test")

    monkeypatch.setattr("openmodalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)

    outcome = analyze_from_config(
        config_path,
        method="dmd",
        overrides={
            "method": "tls",
            "embedding_dim": 4,
            "generate_plots": False,
            "results_root": str(tmp_path / "custom_results"),
            "figures_root": str(tmp_path / "custom_figures"),
        },
    )

    init_kwargs = captured["init"]
    assert captured["loaded"] is True
    assert captured["perform"] == {"method": "tls", "embedding_dim": 4}
    assert init_kwargs["file_path"] == "toy_case"
    assert callable(init_kwargs["data_loader"])
    # Compare path components, not a substring: on Windows the separator is "\",
    # so endswith("custom_results/dmd_cli") fails against a perfectly correct path.
    assert Path(init_kwargs["results_dir"]).parts[-2:] == ("custom_results", "dmd_cli")
    assert Path(init_kwargs["figures_dir"]).parts[-2:] == ("custom_figures", "dmd_cli")
    assert outcome.method == "dmd"
    assert outcome.executed is True


def test_config_params_embedding_dim_reaches_dmd_analyzer(tmp_path: Path, monkeypatch) -> None:
    """Check run params embedding_dim reaches perform_dmd."""
    config_path = tmp_path / "suite.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy suite",
            "description": "Toy suite",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "spatial_weight_type": "uniform",
                "generate_plots": False,
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [
                {
                    "id": "tls",
                    "method": "dmd",
                    "params": {"method": "tls", "embedding_dim": 3},
                },
            ],
        },
    )

    captured: dict[str, object] = {}

    class FakeDMDAnalyzer:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def run_analysis(self, *, plots, run_id=None, snapshot_limit=None, **kwargs):
            self.load_and_preprocess()
            self.perform_dmd(**kwargs)
            self.save_results()

        def load_and_preprocess(self):
            return None

        def perform_dmd(self, *, method: str, embedding_dim: int, named_variant: str | None = None):
            captured["perform"] = {"method": method, "embedding_dim": embedding_dim}

        def save_results(self):
            results_dir = Path(self._kwargs["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "fake.hdf5").write_text("fake")

    monkeypatch.setattr("openmodalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)
    run_from_config(config_path)
    assert captured["perform"] == {"method": "tls", "embedding_dim": 3}


def test_config_rank_reaches_the_dmd_analyzer(tmp_path: Path, monkeypatch) -> None:
    """A rank chosen in the config must arrive at DMDAnalyzer, not be silently dropped.

    ``CaseSpec.rank`` shipped briefly as dead config: nothing read it from the payload,
    ``_apply_case_overrides`` rebuilt the spec without it, and ``_run_dmd`` never passed
    it on. Asserting the field merely exists cannot catch that -- this pins the value
    the analyzer actually receives.
    """

    def build(case_rank, overrides):
        config_path = tmp_path / f"case_{case_rank}_{len(overrides)}.jsonc"
        case: dict[str, object] = {
            "name": "toy_case",
            "case_type": "analytical",
            "data": {
                "kind": "generator",
                "name": "double_gyre",
                "params": {"Nx": 8, "Ny": 4, "Nt": 12},
            },
            "spatial_weight_type": "uniform",
            "results_root": str(tmp_path / "results"),
            "figures_root": str(tmp_path / "figures"),
        }
        if case_rank is not None:
            case["rank"] = case_rank
        _write_jsonc(
            config_path,
            {
                "name": "Toy case",
                "description": "Toy analytical case",
                "case": case,
                "runs": [],
            },
        )
        captured: dict[str, object] = {}

        class FakeDMDAnalyzer:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def run_analysis(self, *, plots, run_id=None, snapshot_limit=None, **kwargs):
                self.load_and_preprocess()
                self.perform_dmd(**kwargs)
                self.save_results()

            def load_and_preprocess(self):
                pass

            def perform_dmd(self, **kwargs):
                pass

            def save_results(self):
                results_dir = Path(captured["init"]["results_dir"])
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "fake.hdf5").write_text("fake")

        monkeypatch.setattr("openmodalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)
        analyze_from_config(
            config_path,
            method="dmd",
            overrides={"generate_plots": False, **overrides},
        )
        return captured["init"]["rank"]

    # From the case payload, for each accepted spelling.
    assert build("svht", {}) == "svht"
    assert build("energy", {}) == "energy"
    assert build(12, {}) == 12
    # Absent from the config: carried through as None. The refusal lives in
    # DMDAnalyzer, so this fake never sees it -- see
    # test_config_without_rank_refuses_through_the_real_analyzer.
    assert build(None, {}) is None
    # A per-run override must win over the case value.
    assert build("svht", {"rank": 5}) == 5
    # And an override must not wipe a case value it does not mention.
    assert build(7, {"embedding_dim": 1}) == 7


def test_config_without_rank_refuses_through_the_real_analyzer(tmp_path: Path) -> None:
    """A config that omits ``rank`` must fail for a user, not just in the constructor.

    The test above uses a fake analyzer that accepts ``rank=None``, so it pins the
    pass-through and cannot see the refusal. This runs the real DMDAnalyzer through
    the config path -- the route a user actually takes -- so 'DMD refuses to guess a
    rank' is verified end to end rather than at the API boundary only.
    """
    config_path = tmp_path / "no_rank.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "No rank",
            "description": "DMD case that omits the required rank",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "spatial_weight_type": "uniform",
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [],
        },
    )
    with pytest.raises(ValueError, match=r"rank is required"):
        analyze_from_config(config_path, method="dmd", overrides={"generate_plots": False})


def test_config_energy_fraction_reaches_the_dmd_analyzer(tmp_path: Path, monkeypatch) -> None:
    """A non-default energy_fraction in the config must arrive at DMDAnalyzer.

    Presence on CaseSpec alone is not enough — that is how CaseSpec.rank shipped
    unreachable. Pin the value the analyzer constructor actually receives.
    """

    def build(case_energy_fraction, overrides):
        config_path = tmp_path / f"ef_{case_energy_fraction}_{len(overrides)}.jsonc"
        case: dict[str, object] = {
            "name": "toy_case",
            "case_type": "analytical",
            "data": {
                "kind": "generator",
                "name": "double_gyre",
                "params": {"Nx": 8, "Ny": 4, "Nt": 12},
            },
            "spatial_weight_type": "uniform",
            "rank": "energy",
            "results_root": str(tmp_path / "results"),
            "figures_root": str(tmp_path / "figures"),
        }
        if case_energy_fraction is not None:
            case["energy_fraction"] = case_energy_fraction
        _write_jsonc(
            config_path,
            {
                "name": "Toy case",
                "description": "Toy analytical case",
                "case": case,
                "runs": [],
            },
        )
        captured: dict[str, object] = {}

        class FakeDMDAnalyzer:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            def run_analysis(self, *, plots, run_id=None, snapshot_limit=None, **kwargs):
                self.load_and_preprocess()
                self.perform_dmd(**kwargs)
                self.save_results()

            def load_and_preprocess(self):
                pass

            def perform_dmd(self, **kwargs):
                pass

            def save_results(self):
                results_dir = Path(captured["init"]["results_dir"])
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "fake.hdf5").write_text("fake")

        monkeypatch.setattr("openmodalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)
        analyze_from_config(
            config_path,
            method="dmd",
            overrides={"generate_plots": False, **overrides},
        )
        return captured["init"]

    # From the case payload: the configured fraction reaches the constructor.
    init = build(0.95, {})
    assert init["rank"] == "energy"
    assert init["energy_fraction"] == 0.95
    # Omitted key: analyzer keeps its own default (constructor arg not passed).
    init_default = build(None, {})
    assert "energy_fraction" not in init_default
    # A per-run override must win over the case value.
    init_ov = build(0.95, {"energy_fraction": 0.8})
    assert init_ov["energy_fraction"] == 0.8
    # An override that does not mention energy_fraction must not drop the case value.
    init_keep = build(0.92, {"embedding_dim": 1})
    assert init_keep["energy_fraction"] == 0.92


def test_energy_fraction_parse_rejects_out_of_range(tmp_path: Path) -> None:
    """energy_fraction outside (0, 1] fails at parse time with one message."""
    from openmodalpy.commands import load_case_spec

    config_path = tmp_path / "bad_ef.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Bad fraction",
            "description": "energy_fraction out of range",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "rank": "energy",
                "energy_fraction": 1.5,
            },
            "runs": [],
        },
    )
    with pytest.raises(ValueError, match=r"energy_fraction must be null or a float in \(0, 1\]"):
        load_case_spec(config_path)


def test_run_from_config_executes_runs_schema(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "suite.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy suite",
            "description": "Toy suite",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
            },
            "runs": [
                {"id": "pod", "method": "pod"},
                {"id": "mpod", "method": "mpod"},
                {"id": "psd", "method": "psd-pod"},
                {"id": "hodmd", "method": "hodmd"},
                {"id": "tls", "method": "dmd", "params": {"method": "tls", "embedding_dim": 3}},
            ],
        },
    )

    seen: list[tuple[str, str, dict[str, object]]] = []

    def fake_analyze(spec, *, dry_run: bool = False):
        seen.append((spec.run_id, spec.method, dict(spec.params)))
        return object()

    monkeypatch.setattr("openmodalpy.commands.analyze_from_spec", fake_analyze)

    run_from_config(config_path)

    assert seen == [
        ("pod", "pod", {}),
        ("mpod", "mpod", {}),
        ("psd", "psd_pod", {}),
        ("hodmd", "hodmd", {}),
        ("tls", "dmd", {"method": "tls", "embedding_dim": 3}),
    ]


def test_run_from_config_executes_nested_config_suite(tmp_path: Path, monkeypatch) -> None:
    case_a = tmp_path / "a.jsonc"
    case_b = tmp_path / "b.jsonc"
    suite = tmp_path / "suite.jsonc"

    payload = {
        "name": "Toy case",
        "description": "Toy case",
        "case": {
            "name": "toy_case",
            "case_type": "analytical",
            "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
        },
        "runs": [{"id": "pod", "method": "pod"}],
    }
    _write_jsonc(case_a, payload)
    _write_jsonc(case_b, payload | {"name": "Toy case B"})
    _write_jsonc(
        suite,
        {
            "name": "Nested suite",
            "description": "Nested suite",
            "configs": [str(case_a), str(case_b)],
        },
    )

    seen: list[str] = []

    def fake_analyze(spec, *, dry_run: bool = False):
        seen.append(spec.config_path.name)
        return object()

    monkeypatch.setattr("openmodalpy.commands.analyze_from_spec", fake_analyze)

    run_from_config(suite)

    assert seen == ["a.jsonc", "b.jsonc"]


def test_discover_examples_lists_repo_configs() -> None:
    example_names = {info.name for info in discover_examples()}

    assert "run_benchmarks" in example_names
    assert "cavity" in example_names
    assert "cylinder" in example_names
    assert "jet" in example_names


def test_discover_examples_falls_back_to_packaged_resources(monkeypatch) -> None:
    monkeypatch.setattr("openmodalpy.commands.examples_root", lambda: Path("/definitely/missing/examples"))

    example_names = {info.name for info in discover_examples()}

    assert "run_benchmarks" in example_names
    assert "double_gyre" in example_names


def test_command_core_uses_volumetric_plot_hooks_when_available() -> None:
    calls = []

    class FakeAnalyzer:
        data = {"Nz": 2}

        def plot_modes_3d_slices(self, **kwargs):
            calls.append(("slices", kwargs))

        def plot_modes_3d_isometric(self, **kwargs):
            calls.append(("iso", kwargs))

    analyzer = FakeAnalyzer()
    used = BaseAnalyzer._maybe_plot_volumetric_modes(analyzer, plot_n_modes=2, slices_kwargs={"freqs_to_plot": [1]})

    assert used is True
    assert calls == [
        ("slices", {"plot_n_modes": 2, "freqs_to_plot": [1]}),
        ("iso", {"plot_n_modes": 2}),
    ]


def test_inspect_results_reads_hdf5_metadata(tmp_path: Path) -> None:
    result_path = tmp_path / "toy.hdf5"
    with h5py.File(result_path, "w") as handle:
        handle.create_dataset("modes", data=[[1.0, 2.0]])
        handle.attrs["analysis_type"] = "pod"
        handle.attrs["Ns"] = 5

    summary = inspect_results(result_path)

    assert summary["type"] == "hdf5"
    assert summary["datasets"]["modes"]["shape"] == [1, 2]
    assert summary["attrs"]["analysis_type"] == "pod"
    assert summary["attrs"]["Ns"] == 5


def test_unhandled_command_tree_returns_2(monkeypatch, capsys) -> None:
    # Defensive branch unreachable from the command line: every subparser is
    # required=True, so ordinary argv never lands here. Monkeypatch parse_args
    # to hand back an unrecognised command and exercise the fallback.
    def fake_parse_args(self, args=None, namespace=None):
        return argparse.Namespace(command="not-a-real-command")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)

    exit_code = main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Unhandled command tree" in err
    assert "not-a-real-command" in err


def _analysis_method_action() -> argparse.Action:
    parser = build_parser()
    command_action = next(a for a in parser._actions if a.dest == "command")
    analyze = command_action.choices["analyze"]
    return next(a for a in analyze._actions if a.dest == "analysis_method")


def test_analyze_method_choices_match_registry() -> None:
    action = _analysis_method_action()
    assert action.choices is not None
    assert set(action.choices) == set(METHOD_REGISTRY)


@pytest.mark.parametrize(
    "spelling",
    [
        "pod",
        "POD",
        "psd-pod",
        "psd_pod",
        "PSD-POD",
        "tls-hodmd",
        "tls_hodmd",
        " pod ",
    ],
)
def test_analyze_method_spellings_normalize_at_parse(spelling: str) -> None:
    parser = build_parser()
    args = parser.parse_args(["analyze", spelling, "--config", "/nonexistent.jsonc"])
    assert args.analysis_method == normalize_method_name(spelling)
    assert args.analysis_method in METHOD_REGISTRY


def test_analyze_unknown_method_rejected_at_parse() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["analyze", "not-a-method", "--config", "/nonexistent.jsonc"])
    assert excinfo.value.code == 2


def test_analyze_unknown_method_error_lists_the_methods(capsys) -> None:
    """The rejection tells the user what to type, not an internal name.

    argparse discards a ValueError's message and prints
    'invalid <callable name> value'; only ArgumentTypeError survives verbatim.
    """
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "psdpod", "--config", "/nonexistent.jsonc"])
    err = capsys.readouterr().err
    assert "psdpod" in err
    assert "normalize_method_name" not in err
    for method in METHOD_REGISTRY:
        assert method in err


def test_cli_analyze_subcommand_routes_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
            },
            "runs": [{"id": "pod", "method": "pod"}],
        },
    )

    captured: dict[str, object] = {}

    def fake_analyze_from_config(config_path, *, method, run_id=None, overrides=None, dry_run=False):
        captured["config_path"] = Path(config_path)
        captured["method"] = method
        captured["run_id"] = run_id
        captured["overrides"] = dict(overrides or {})
        captured["dry_run"] = dry_run
        return object()

    monkeypatch.setattr("openmodalpy.cli.analyze_from_config", fake_analyze_from_config)

    exit_code = main(
        [
            "analyze",
            "dmd",
            "--config",
            str(config_path),
            "--method",
            "tls",
            "--embedding-dim",
            "4",
            "--no-plots",
            "--run-id",
            "custom_run",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == config_path.resolve()
    assert captured["method"] == "dmd"
    assert captured["run_id"] == "custom_run"
    assert captured["dry_run"] is False
    assert captured["overrides"]["method"] == "tls"
    assert captured["overrides"]["embedding_dim"] == 4
    assert captured["overrides"]["generate_plots"] is False
    assert capsys.readouterr().out == ""


def test_cli_analyze_subcommand_routes_solver_override(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
            },
            "runs": [{"id": "pod", "method": "pod"}],
        },
    )

    captured: dict[str, object] = {}

    def fake_analyze_from_config(config_path, *, method, run_id=None, overrides=None, dry_run=False):
        captured["config_path"] = Path(config_path)
        captured["method"] = method
        captured["run_id"] = run_id
        captured["overrides"] = dict(overrides or {})
        captured["dry_run"] = dry_run
        return object()

    monkeypatch.setattr("openmodalpy.cli.analyze_from_config", fake_analyze_from_config)

    exit_code = main(
        [
            "analyze",
            "pod",
            "--config",
            str(config_path),
            "--solver",
            "svd",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == config_path.resolve()
    assert captured["method"] == "pod"
    assert captured["dry_run"] is False
    assert captured["overrides"]["solver"] == "svd"
    assert capsys.readouterr().out == ""

    # Omitting --solver must leave the key absent: the CLI must not inject a
    # default into overrides, so downstream code keeps its own 'eigh' default.
    captured.clear()
    exit_code = main(["analyze", "pod", "--config", str(config_path)])
    assert exit_code == 0
    assert "solver" not in captured["overrides"]


def test_run_from_config_executes_real_psd_pod(tmp_path: Path) -> None:
    config_path = tmp_path / "psd_pod_case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "PSD-POD toy case",
            "description": "Real PSD-POD command-core execution",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 20},
                },
                "spatial_weight_type": "uniform",
                "n_modes_save": 4,
                "nfft": 8,
                "overlap": 0.5,
                "embedding_dim": 4,
                "generate_plots": False,
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [{"id": "psd", "method": "psd-pod"}],
        },
    )

    outcomes = run_from_config(config_path)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.method == "psd_pod"
    assert outcome.executed is True
    assert outcome.results_path is not None
    assert outcome.results_path.is_file()

    summary = inspect_results(outcome.results_path)
    assert summary["attrs"]["analysis_type"] == "psd_pod"
    assert "eigenvalues" in summary["datasets"]
    assert "modes" in summary["datasets"]
    assert "time_coefficients" in summary["datasets"]


def test_psd_pod_uses_mean_subtraction_matches_blocksfft_modes(tmp_path: Path) -> None:
    """PSD-POD must record the centering blocksfft actually applied.

    blocksfft always removes a mean: global when blockwise_mean is False,
    per-block when True. Both modes must write uses_mean_subtraction=True
    and echo the configured blockwise_mean flag into the result file.
    """
    for blockwise_mean in (False, True):
        config_path = tmp_path / f"psd_pod_mean_{int(blockwise_mean)}.jsonc"
        results_root = tmp_path / f"results_bm_{int(blockwise_mean)}"
        figures_root = tmp_path / f"figures_bm_{int(blockwise_mean)}"
        _write_jsonc(
            config_path,
            {
                "name": f"PSD-POD mean mode {blockwise_mean}",
                "description": "uses_mean_subtraction vs blocksfft contract",
                "case": {
                    "name": "toy_case",
                    "case_type": "analytical",
                    "data": {
                        "kind": "generator",
                        "name": "double_gyre",
                        "params": {"Nx": 8, "Ny": 4, "Nt": 20},
                    },
                    "spatial_weight_type": "uniform",
                    "n_modes_save": 4,
                    "nfft": 8,
                    "overlap": 0.5,
                    "generate_plots": False,
                    "results_root": str(results_root),
                    "figures_root": str(figures_root),
                },
                "runs": [
                    {
                        "id": "psd",
                        "method": "psd-pod",
                        "params": {"blockwise_mean": blockwise_mean},
                    }
                ],
            },
        )

        outcomes = run_from_config(config_path)
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.results_path is not None and outcome.results_path.is_file()

        with h5py.File(outcome.results_path, "r") as handle:
            assert bool(handle.attrs["uses_mean_subtraction"]) is True
            assert bool(handle.attrs["blockwise_mean"]) is blockwise_mean
            assert handle.attrs["analysis_type"] == "psd_pod"


def test_coerce_rank_rejects_boolean():
    """A JSON boolean must not be silently treated as a missing rank."""
    from openmodalpy.commands import _coerce_rank

    with pytest.raises(ValueError, match=r"rank"):
        _coerce_rank(True)
    with pytest.raises(ValueError, match=r"rank"):
        _coerce_rank(False)


def test_coerce_energy_fraction_rejects_boolean():
    """A JSON boolean must not be silently treated as a missing energy_fraction."""
    from openmodalpy.commands import _coerce_energy_fraction

    with pytest.raises(ValueError, match=r"energy_fraction"):
        _coerce_energy_fraction(True)
    with pytest.raises(ValueError, match=r"energy_fraction"):
        _coerce_energy_fraction(False)


def _minimal_analysis_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "name": "Toy case",
        "description": "Toy case",
        "case": {
            "name": "toy_case",
            "case_type": "analytical",
            "data": {
                "kind": "generator",
                "name": "double_gyre",
                "params": {"Nx": 8, "Ny": 4, "Nt": 12},
            },
            "results_root": str(tmp_path / "results"),
            "figures_root": str(tmp_path / "figures"),
        },
        "runs": [{"id": "pod", "method": "pod"}],
    }


def _mapping_keys_read(func: object, names: set[str]) -> set[str]:
    """String keys read from the named mappings in ``func`` via get / [] / in."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.In | ast.NotIn) and isinstance(comparator, ast.Name) and comparator.id in names:
                    keys.add(node.left.value)
    return keys


def test_accepted_key_constants_match_readers() -> None:
    """The accepted-key constants must match what the loaders actually read."""
    from openmodalpy.commands import (
        CASE_FIELD_KEYS,
        CASE_KEYS,
        DATA_KEYS,
        RUN_KEYS,
        TOP_LEVEL_KEYS,
        _apply_case_overrides,
        _load_case_spec_from_payload,
        _load_run_collection,
        _validate_config_kind,
    )

    assert CASE_KEYS == CASE_FIELD_KEYS | {"name", "description", "case_type", "data"}
    assert _mapping_keys_read(_load_case_spec_from_payload, {"case_payload"}) == set(CASE_KEYS)
    assert _mapping_keys_read(_load_case_spec_from_payload, {"data_payload"}) == set(DATA_KEYS)
    assert _mapping_keys_read(_apply_case_overrides, {"overrides"}) == set(CASE_FIELD_KEYS)
    assert _mapping_keys_read(_load_run_collection, {"run_payload"}) == set(RUN_KEYS)
    top_keys = (
        _mapping_keys_read(_load_case_spec_from_payload, {"payload"})
        | _mapping_keys_read(_load_run_collection, {"payload"})
        | _mapping_keys_read(_validate_config_kind, {"payload"})
    )
    assert top_keys == set(TOP_LEVEL_KEYS)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "typo_top.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    payload["nfft_"] = 64
    _write_jsonc(config_path, payload)
    with pytest.raises(ValueError, match=r"nfft_") as excinfo:
        run_from_config(config_path, dry_run=True)
    message = str(excinfo.value)
    assert config_path.name in message
    assert "Accepted keys" in message
    assert "kind" in message


def test_unknown_case_key_raises(tmp_path: Path) -> None:
    from openmodalpy.commands import load_case_spec

    config_path = tmp_path / "typo_case.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    case = payload["case"]
    assert isinstance(case, dict)
    case["n_modes_sav"] = 8
    _write_jsonc(config_path, payload)
    with pytest.raises(ValueError, match=r"n_modes_sav") as excinfo:
        load_case_spec(config_path)
    message = str(excinfo.value)
    assert config_path.name in message
    assert "Accepted keys" in message
    assert "n_modes_save" in message


def test_unknown_run_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "typo_run.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    payload["runs"] = [{"id": "pod", "method": "pod", "nfft_": 64}]
    _write_jsonc(config_path, payload)
    with pytest.raises(ValueError, match=r"nfft_") as excinfo:
        run_from_config(config_path, dry_run=True)
    message = str(excinfo.value)
    assert config_path.name in message
    assert "Accepted keys" in message
    assert "method" in message


def test_spatial_weights_key_in_config_raises(tmp_path: Path) -> None:
    """A spatial_weights array in a config must raise, not run under uniform weights."""
    config_path = tmp_path / "spatial_weights.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    case = payload["case"]
    assert isinstance(case, dict)
    case["spatial_weights"] = [1.0, 1.0, 1.0]
    _write_jsonc(config_path, payload)
    with pytest.raises(ValueError, match=r"spatial_weights.*cannot prescribe a metric.*library API"):
        run_from_config(config_path, dry_run=True)


def test_kind_must_agree_with_contents(tmp_path: Path) -> None:
    """Top-level kind is read and must match whether the file has a configs list."""
    lying_analysis = tmp_path / "lie_analysis.jsonc"
    _write_jsonc(
        lying_analysis,
        {
            "kind": "analysis-suite",
            "name": "Lie",
            "description": "Lie",
            "configs": [str(tmp_path / "missing.jsonc")],
        },
    )
    with pytest.raises(ValueError, match=r"analysis-suite"):
        run_from_config(lying_analysis, dry_run=True)

    lying_suite = tmp_path / "lie_suite.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    payload["kind"] = "config-suite"
    _write_jsonc(lying_suite, payload)
    with pytest.raises(ValueError, match=r"config-suite"):
        run_from_config(lying_suite, dry_run=True)

    bad_kind = tmp_path / "bad_kind.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    payload["kind"] = "nope"
    _write_jsonc(bad_kind, payload)
    with pytest.raises(ValueError, match=r"analysis-suite.*config-suite"):
        run_from_config(bad_kind, dry_run=True)

    honest = tmp_path / "honest.jsonc"
    payload = _minimal_analysis_payload(tmp_path)
    payload["kind"] = "analysis-suite"
    _write_jsonc(honest, payload)
    run_from_config(honest, dry_run=True)

    child = tmp_path / "child.jsonc"
    _write_jsonc(child, _minimal_analysis_payload(tmp_path) | {"kind": "analysis-suite"})
    suite = tmp_path / "suite.jsonc"
    _write_jsonc(
        suite,
        {
            "kind": "config-suite",
            "name": "Nested",
            "description": "Nested",
            "configs": [str(child)],
        },
    )
    run_from_config(suite, dry_run=True)


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.jsonc")),
    ids=lambda p: p.stem,
)
def test_shipped_example_config_loads(path: Path) -> None:
    from openmodalpy.commands import _load_run_collection

    collection = _load_run_collection(path)
    assert collection.name
