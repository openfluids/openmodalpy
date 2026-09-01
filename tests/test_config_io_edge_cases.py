"""Tests for config_io edge cases and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmodalpy.config_io import (
    load_jsonc,
    parse_name_list,
    resolve_path,
    strip_jsonc_comments,
)


def test_strip_jsonc_comments_line_comment() -> None:
    """strip_jsonc_comments removes line comments."""
    text = '{"key": "value"}  // this is a comment'
    result = strip_jsonc_comments(text)
    assert "//" not in result
    assert '"key"' in result


def test_strip_jsonc_comments_block_comment() -> None:
    """strip_jsonc_comments removes block comments."""
    text = '{"key": /* comment */ "value"}'
    result = strip_jsonc_comments(text)
    assert "/*" not in result
    assert "comment" not in result
    assert '"key"' in result


def test_strip_jsonc_comments_multiline_block() -> None:
    """strip_jsonc_comments removes multiline block comments."""
    text = """{"key": /* this is
    a multiline
    comment */ "value"}"""
    result = strip_jsonc_comments(text)
    assert "multiline" not in result
    assert '"key"' in result


def test_load_jsonc_valid(tmp_path: Path) -> None:
    """load_jsonc parses valid JSONC file."""
    config_file = tmp_path / "config.jsonc"
    config_file.write_text('{"name": "test", "value": 42} // comment')

    result = load_jsonc(config_file)

    assert result["name"] == "test"
    assert result["value"] == 42


def test_load_jsonc_not_dict_raises(tmp_path: Path) -> None:
    """load_jsonc raises when file contains non-dict JSON."""
    config_file = tmp_path / "config.jsonc"
    config_file.write_text('["not", "a", "dict"]')

    with pytest.raises(ValueError, match="Config file must define a JSON object"):
        load_jsonc(config_file)


def test_load_jsonc_expanduser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_jsonc expands ~ in paths."""
    home = tmp_path
    monkeypatch.setenv("HOME", str(home))

    config_file = home / "config.jsonc"
    config_file.write_text('{"key": "value"}')

    result = load_jsonc("~/config.jsonc")
    assert result["key"] == "value"


def test_resolve_path_relative() -> None:
    """resolve_path resolves relative paths against base."""
    base = Path("/base/dir/config.json")
    result = resolve_path("../data/file.txt", base)

    assert result == Path("/base/data/file.txt")
    assert result.is_absolute()


def test_resolve_path_absolute() -> None:
    """resolve_path returns absolute paths unchanged."""
    result = resolve_path("/absolute/path/file.txt", "/any/base")

    assert result == Path("/absolute/path/file.txt")


def test_resolve_path_expandvars() -> None:
    """resolve_path expands environment variables."""
    import os

    os.environ["TEST_VAR"] = "/test/dir"

    result = resolve_path("$TEST_VAR/file.txt", "/base")

    assert result == Path("/test/dir/file.txt")


def test_resolve_path_expanduser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """resolve_path expands ~ in paths."""
    home = tmp_path
    monkeypatch.setenv("HOME", str(home))

    result = resolve_path("~/config.json", "/any/base")

    assert result == home / "config.json"


def test_parse_name_list_string() -> None:
    """parse_name_list splits comma-separated strings."""
    result = parse_name_list("name1, name2, name3")

    assert result == ["name1", "name2", "name3"]


def test_parse_name_list_string_with_empty() -> None:
    """parse_name_list ignores empty strings."""
    result = parse_name_list("name1, , name2, ")

    assert result == ["name1", "name2"]


def test_parse_name_list_list() -> None:
    """parse_name_list accepts list of names."""
    result = parse_name_list(["name1", "name2", "name3"])

    assert result == ["name1", "name2", "name3"]


def test_parse_name_list_list_with_empty() -> None:
    """parse_name_list filters empty items from list."""
    result = parse_name_list(["name1", "", "name2", " "])

    assert result == ["name1", "name2"]


def test_parse_name_list_list_coerces_to_string() -> None:
    """parse_name_list coerces non-string list items to strings."""
    result = parse_name_list([1, 2, "three"])

    assert result == ["1", "2", "three"]


def test_parse_name_list_none() -> None:
    """parse_name_list returns empty list for None."""
    result = parse_name_list(None)

    assert result == []


def test_parse_name_list_invalid_type_raises() -> None:
    """parse_name_list rejects invalid types."""
    with pytest.raises(TypeError, match="Expected a string or list"):
        parse_name_list({"invalid": "dict"})

    with pytest.raises(TypeError, match="Expected a string or list"):
        parse_name_list(42)
