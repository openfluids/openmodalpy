"""Shared JSONC and path helpers for OpenModalPy command/config flows."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def strip_jsonc_comments(text: str) -> str:
    """Remove line and block comments from a JSONC string."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    cleaned_lines = []
    for line in text.splitlines():
        if "//" in line:
            line = line.split("//", 1)[0]
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def load_jsonc(path: str | Path) -> dict[str, Any]:
    """Load a JSONC file into a dictionary."""
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(strip_jsonc_comments(resolved.read_text()))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must define a JSON object: {resolved}")
    return payload


def resolve_path(path_value: str | Path, base_path: str | Path) -> Path:
    """Resolve env vars, ``~``, and relative paths against a base path."""
    base = Path(base_path)
    base_dir = base if base.is_dir() else base.parent
    candidate = Path(os.path.expandvars(str(path_value))).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def parse_name_list(value: Any) -> list[str]:
    """Normalize a string or list config value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Expected a string or list of strings, got {type(value).__name__}")
