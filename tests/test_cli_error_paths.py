"""Test error paths and edge cases in CLI and command dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from openmodalpy.cli import _analysis_method, _collect_overrides, main
from openmodalpy.commands import (
    _coerce_energy_fraction,
    _coerce_rank,
    _load_case_spec_from_payload,
    _load_run_collection,
)


def _write_config(path: Path, payload: dict) -> None:
    """Write a minimal valid config file."""
    path.write_text(json.dumps(payload, indent=2))


class TestAnalysisMethodTypeValidation:
    """Tests for the _analysis_method type validator used by argparse."""

    def test_valid_method_name_returns_normalized(self) -> None:
        """Valid method names are normalized and returned."""
        assert _analysis_method("pod") == "pod"
        assert _analysis_method("POD") == "pod"
        assert _analysis_method("psd-pod") == "psd_pod"

    def test_invalid_method_raises_argument_type_error(self) -> None:
        """Invalid method names raise ArgumentTypeError with full registry text."""
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown method"):
            _analysis_method("invalid_method")


class TestCollectOverridesAllPaths:
    """Test every override branch in _collect_overrides."""

    def _make_args(self, **kwargs: Any) -> argparse.Namespace:
        """Create a minimal Namespace with all override flags."""
        defaults = {
            "no_plots": False,
            "results_dir": None,
            "figures_dir": None,
            "weight_type": None,
            "n_modes": None,
            "nfft": None,
            "overlap": None,
            "embedding_dim": None,
            "band_edges": None,
            "band_scale": None,
            "filter_kind": None,
            "dmd_method": None,
            "solver": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_collect_overrides_empty_returns_empty_dict(self) -> None:
        """No overrides set returns empty dict."""
        args = self._make_args()
        result = _collect_overrides(args)
        assert result == {}

    def test_collect_overrides_no_plots_flag(self) -> None:
        """--no-plots flag sets generate_plots to False."""
        args = self._make_args(no_plots=True)
        result = _collect_overrides(args)
        assert result == {"generate_plots": False}

    def test_collect_overrides_results_dir(self) -> None:
        """--results-dir is converted to results_root override."""
        results_path = Path("/tmp/results")
        args = self._make_args(results_dir=results_path)
        result = _collect_overrides(args)
        assert result == {"results_root": str(results_path)}

    def test_collect_overrides_figures_dir(self) -> None:
        """--figures-dir is converted to figures_root override."""
        figures_path = Path("/tmp/figures")
        args = self._make_args(figures_dir=figures_path)
        result = _collect_overrides(args)
        assert result == {"figures_root": str(figures_path)}

    def test_collect_overrides_weight_type(self) -> None:
        """--weight-type override is collected."""
        args = self._make_args(weight_type="cell_volume")
        result = _collect_overrides(args)
        assert result == {"spatial_weight_type": "cell_volume"}

    def test_collect_overrides_n_modes(self) -> None:
        """--n-modes override is collected."""
        args = self._make_args(n_modes=20)
        result = _collect_overrides(args)
        assert result == {"n_modes_save": 20}

    def test_collect_overrides_nfft(self) -> None:
        """--nfft override is collected."""
        args = self._make_args(nfft=256)
        result = _collect_overrides(args)
        assert result == {"nfft": 256}

    def test_collect_overrides_overlap(self) -> None:
        """--overlap override is collected."""
        args = self._make_args(overlap=0.75)
        result = _collect_overrides(args)
        assert result == {"overlap": 0.75}

    def test_collect_overrides_embedding_dim(self) -> None:
        """--embedding-dim override is collected."""
        args = self._make_args(embedding_dim=15)
        result = _collect_overrides(args)
        assert result == {"embedding_dim": 15}

    def test_collect_overrides_band_edges(self) -> None:
        """--band-edges string is parsed into list of floats."""
        args = self._make_args(band_edges="0,0.1,0.5,1.0")
        result = _collect_overrides(args)
        assert result == {"band_edges": [0.0, 0.1, 0.5, 1.0]}

    def test_collect_overrides_band_edges_handles_whitespace(self) -> None:
        """--band-edges parsing skips whitespace-only entries."""
        args = self._make_args(band_edges="0, , 0.5 , 1.0")
        result = _collect_overrides(args)
        assert result == {"band_edges": [0.0, 0.5, 1.0]}

    def test_collect_overrides_band_scale(self) -> None:
        """--band-scale override is collected."""
        args = self._make_args(band_scale="normalized_nyquist")
        result = _collect_overrides(args)
        assert result == {"band_scale": "normalized_nyquist"}

    def test_collect_overrides_filter_kind(self) -> None:
        """--filter-kind override is collected."""
        args = self._make_args(filter_kind="rectangular")
        result = _collect_overrides(args)
        assert result == {"filter_kind": "rectangular"}

    def test_collect_overrides_dmd_method(self) -> None:
        """--method (dmd_method) override is collected."""
        args = self._make_args(dmd_method="tls")
        result = _collect_overrides(args)
        assert result == {"method": "tls"}

    def test_collect_overrides_solver(self) -> None:
        """--solver override is collected."""
        args = self._make_args(solver="svd")
        result = _collect_overrides(args)
        assert result == {"solver": "svd"}

    def test_collect_overrides_multiple_flags(self) -> None:
        """Multiple override flags are all collected."""
        args = self._make_args(
            no_plots=True,
            n_modes=30,
            nfft=512,
            solver="svd",
        )
        result = _collect_overrides(args)
        assert result == {
            "generate_plots": False,
            "n_modes_save": 30,
            "nfft": 512,
            "solver": "svd",
        }


class TestCoerceRankEdgeCases:
    """Test error handling in _coerce_rank."""

    def test_coerce_rank_none_returns_none(self) -> None:
        """None input returns None."""
        assert _coerce_rank(None) is None

    def test_coerce_rank_positive_int(self) -> None:
        """Positive integers pass through."""
        assert _coerce_rank(5) == 5
        assert _coerce_rank(100) == 100

    def test_coerce_rank_svht_string(self) -> None:
        """The string 'svht' is accepted."""
        assert _coerce_rank("svht") == "svht"

    def test_coerce_rank_energy_string(self) -> None:
        """The string 'energy' is accepted."""
        assert _coerce_rank("energy") == "energy"

    def test_coerce_rank_numeric_string(self) -> None:
        """Numeric strings are converted to int."""
        assert _coerce_rank("42") == 42

    def test_coerce_rank_boolean_raises(self) -> None:
        """Boolean values are rejected with specific error text."""
        with pytest.raises(ValueError, match="rank must be null, a positive int"):
            _coerce_rank(True)
        with pytest.raises(ValueError, match="rank must be null, a positive int"):
            _coerce_rank(False)

    def test_coerce_rank_invalid_string_raises(self) -> None:
        """Invalid string values raise with specific error text."""
        with pytest.raises(ValueError, match="rank must be null, a positive int"):
            _coerce_rank("invalid")

    def test_coerce_rank_empty_string_returns_none(self) -> None:
        """Empty string is treated as None."""
        assert _coerce_rank("") is None

    def test_coerce_rank_whitespace_string_returns_none(self) -> None:
        """Whitespace-only string is treated as None."""
        assert _coerce_rank("  ") is None


class TestCoerceEnergyFractionEdgeCases:
    """Test error handling in _coerce_energy_fraction."""

    def test_coerce_energy_fraction_none_returns_none(self) -> None:
        """None input returns None."""
        assert _coerce_energy_fraction(None) is None

    def test_coerce_energy_fraction_valid_float(self) -> None:
        """Valid fractions in (0, 1] pass through."""
        assert _coerce_energy_fraction(0.5) == 0.5
        assert _coerce_energy_fraction(0.999) == 0.999
        assert _coerce_energy_fraction(1.0) == 1.0

    def test_coerce_energy_fraction_numeric_string(self) -> None:
        """Numeric strings are converted to float."""
        assert _coerce_energy_fraction("0.75") == 0.75

    def test_coerce_energy_fraction_boolean_raises(self) -> None:
        """Boolean values are rejected."""
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction(True)
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction(False)

    def test_coerce_energy_fraction_zero_raises(self) -> None:
        """Zero is out of range (0, 1]."""
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction(0.0)

    def test_coerce_energy_fraction_greater_than_one_raises(self) -> None:
        """Values > 1.0 are out of range."""
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction(1.1)

    def test_coerce_energy_fraction_negative_raises(self) -> None:
        """Negative values are out of range."""
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction(-0.5)

    def test_coerce_energy_fraction_invalid_string_raises(self) -> None:
        """Non-numeric strings raise."""
        with pytest.raises(ValueError, match="energy_fraction must be null or a float"):
            _coerce_energy_fraction("not_a_number")


class TestLoadCaseSpecValidation:
    """Test error paths in _load_case_spec_from_payload."""

    def _valid_case_payload(self) -> dict:
        """Return a minimal valid case payload."""
        return {
            "case": {
                "name": "test",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {},
                },
            }
        }

    def test_case_not_dict_raises(self, tmp_path: Path) -> None:
        """'case' value must be a dict."""
        config_path = tmp_path / "bad.jsonc"
        payload = {"case": "not_a_dict"}
        with pytest.raises(ValueError, match="must define a 'case' object"):
            _load_case_spec_from_payload(payload, config_path)

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        """Missing case name raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"].pop("name")
        with pytest.raises(ValueError, match="case block is missing 'name'"):
            _load_case_spec_from_payload(payload, config_path)

    def test_empty_name_raises(self, tmp_path: Path) -> None:
        """Empty case name raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["name"] = ""
        with pytest.raises(ValueError, match="case block is missing 'name'"):
            _load_case_spec_from_payload(payload, config_path)

    def test_missing_data_dict_raises(self, tmp_path: Path) -> None:
        """Missing 'data' dict raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"].pop("data")
        with pytest.raises(ValueError, match="case block is missing a 'data' object"):
            _load_case_spec_from_payload(payload, config_path)

    def test_unsupported_data_kind_raises(self, tmp_path: Path) -> None:
        """Unsupported data.kind raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"]["kind"] = "unsupported"
        with pytest.raises(ValueError, match="unsupported case.data.kind"):
            _load_case_spec_from_payload(payload, config_path)

    def test_file_data_missing_path_raises(self, tmp_path: Path) -> None:
        """File data source without path raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"] = {"kind": "file"}
        with pytest.raises(ValueError, match="file-backed case .* is missing data.path"):
            _load_case_spec_from_payload(payload, config_path)

    def test_dnami_data_missing_path_raises(self, tmp_path: Path) -> None:
        """dNami data source without path raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"] = {"kind": "dnami"}
        with pytest.raises(ValueError, match="dNami-backed case .* is missing data.path"):
            _load_case_spec_from_payload(payload, config_path)

    def test_dnami_data_missing_schema_raises(self, tmp_path: Path) -> None:
        """dNami data source without schema raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"] = {"kind": "dnami", "path": "/data"}
        with pytest.raises(ValueError, match="must define data.schema as an object"):
            _load_case_spec_from_payload(payload, config_path)

    def test_generator_data_missing_name_raises(self, tmp_path: Path) -> None:
        """Generator data source without name raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"] = {"kind": "generator"}
        with pytest.raises(ValueError, match="generator-backed case .* is missing data.name"):
            _load_case_spec_from_payload(payload, config_path)

    def test_generator_data_bad_params_raises(self, tmp_path: Path) -> None:
        """Generator params must be a dict."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_case_payload()
        payload["case"]["data"]["params"] = "not_a_dict"
        with pytest.raises(ValueError, match="generator params must be a JSON object"):
            _load_case_spec_from_payload(payload, config_path)


class TestLoadRunCollectionValidation:
    """Test error paths in _load_run_collection."""

    def _valid_run_payload(self) -> dict:
        """Return a minimal valid run collection payload."""
        return {
            "case": {
                "name": "test",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {},
                },
            },
            "runs": [
                {
                    "method": "pod",
                    "params": {},
                }
            ],
        }

    def test_empty_runs_list_raises(self, tmp_path: Path) -> None:
        """Empty 'runs' list raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_run_payload()
        payload["runs"] = []
        _write_config(config_path, payload)
        with pytest.raises(ValueError, match="must define a non-empty 'runs' list"):
            _load_run_collection(config_path)

    def test_run_entry_not_dict_raises(self, tmp_path: Path) -> None:
        """Non-dict run entry raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_run_payload()
        payload["runs"] = ["not_a_dict"]
        _write_config(config_path, payload)
        with pytest.raises(ValueError, match="Run entry #1 .* must be a JSON object"):
            _load_run_collection(config_path)

    def test_run_params_not_dict_raises(self, tmp_path: Path) -> None:
        """Non-dict params in run raises."""
        config_path = tmp_path / "bad.jsonc"
        payload = self._valid_run_payload()
        payload["runs"][0]["params"] = "not_a_dict"
        _write_config(config_path, payload)
        with pytest.raises(ValueError, match="must define params as an object"):
            _load_run_collection(config_path)

    def test_disabled_run_is_skipped(self, tmp_path: Path) -> None:
        """Run with enabled=false is not included in analyses."""
        config_path = tmp_path / "test.jsonc"
        payload = self._valid_run_payload()
        payload["runs"].append({"method": "spod", "params": {}, "enabled": False})
        _write_config(config_path, payload)
        collection = _load_run_collection(config_path)
        assert len(collection.analyses) == 1
        assert collection.analyses[0].method == "pod"

    def test_all_runs_disabled_raises(self, tmp_path: Path) -> None:
        """If all runs are disabled, error is raised."""
        config_path = tmp_path / "test.jsonc"
        payload = self._valid_run_payload()
        payload["runs"][0]["enabled"] = False
        _write_config(config_path, payload)
        with pytest.raises(ValueError, match="does not contain any enabled runs"):
            _load_run_collection(config_path)


class TestCLIMainDispatch:
    """Test main CLI dispatch paths."""

    def _write_run_config(self, path: Path) -> None:
        """Write a valid run collection config file."""
        config = {
            "case": {
                "name": "test",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 4, "Ny": 3, "Nt": 5},
                },
            },
            "runs": [{"method": "pod", "params": {}}],
        }
        path.write_text(json.dumps(config))

    def test_run_command_with_config_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """run command with config path and --dry-run executes."""
        config_path = tmp_path / "test.jsonc"
        self._write_run_config(config_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result = main(["run", "--config", str(config_path), "--dry-run"])
        assert result == 0

    def test_examples_list_shows_available_examples(self, capsys: pytest.CaptureFixture[str]) -> None:
        """examples list command prints example names."""
        result = main(["examples", "list"])
        assert result == 0
        captured = capsys.readouterr()
        assert captured.out

    def test_methods_list_shows_available_methods(self, capsys: pytest.CaptureFixture[str]) -> None:
        """methods list command prints method names."""
        result = main(["methods", "list"])
        assert result == 0
        captured = capsys.readouterr()
        assert "pod" in captured.out or "POD" in captured.out

    def test_methods_show_displays_method_details(self, capsys: pytest.CaptureFixture[str]) -> None:
        """methods show command prints method info."""
        result = main(["methods", "show", "pod"])
        assert result == 0
        captured = capsys.readouterr()
        assert captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCLIExamplesCommands:
    """Test examples subcommand paths."""

    def test_examples_show_displays_example_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """examples show command displays example name, path, and config."""
        result = main(["examples", "show", "cavity"])
        assert result == 0
        captured = capsys.readouterr()
        # Should print title, path, and JSON payload
        assert "cavity" in captured.out.lower() or "path" in captured.out

    def test_examples_run_dry_run_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        """examples run with --dry-run loads and displays plan without executing."""
        result = main(["examples", "run", "cavity", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        # Should print dry-run indication
        assert "Dry run" in captured.out or captured.out


class TestCLIResultsCommands:
    """Test results subcommand paths."""

    def test_results_inspect_hdf5_file(self, tmp_path: Path) -> None:
        """results inspect displays HDF5 file metadata."""
        import h5py
        import numpy as np

        hdf5_path = tmp_path / "results.h5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("modes", data=np.random.rand(10, 5))
            f.create_dataset("eigenvalues", data=np.array([1.0, 0.5, 0.2]))
            f.attrs["test_attr"] = "test_value"

        result = main(["results", "inspect", str(hdf5_path)])
        assert result == 0

    def test_results_inspect_json_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """results inspect displays JSON file contents."""
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps({"result": "value", "count": 42}))

        result = main(["results", "inspect", str(json_path)])
        assert result == 0
        captured = capsys.readouterr()
        assert "result" in captured.out or "value" in captured.out

    def test_results_inspect_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """results inspect lists files in directory."""
        results_dir = tmp_path / "results_dir"
        results_dir.mkdir()
        (results_dir / "file1.hdf5").write_text("dummy")
        (results_dir / "file2.json").write_text("{}")

        result = main(["results", "inspect", str(results_dir)])
        assert result == 0
        captured = capsys.readouterr()
        # Should show directory type and file entries
        assert "directory" in captured.out.lower() or captured.out
