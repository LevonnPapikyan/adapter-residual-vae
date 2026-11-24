"""JSON input and output, and command-line parsing with a JSON config.

What belongs here
-----------------
- ``parse_args_with_config``: command defaults, then a JSON file, then the command line.
- ``read_json``, ``write_json``, and the guard that keeps outputs out of protected folders.

What does NOT belong here
-------------------------
- What any particular command's settings are or mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    """Read one JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, indent: int | None = 1) -> None:
    """Write one JSON file, keeping key order and non-ASCII characters as they are."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")


def parse_args_with_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse with the parser's defaults, overridden by ``--config FILE``, overridden by the command line.

    An unknown key in the JSON file is an error, so a misspelled setting cannot be
    silently ignored.
    """

    parser.add_argument("--config", type=Path, default=None, help="JSON configuration file.")
    arguments = list(sys.argv[1:] if argv is None else argv)
    first_pass = argparse.ArgumentParser(add_help=False)
    first_pass.add_argument("--config", type=Path, default=None)
    config_path = first_pass.parse_known_args(arguments)[0].config
    if config_path is None:
        return parser.parse_args(arguments)

    try:
        values = read_json(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON config {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"JSON config must hold an object: {config_path}")
    known = {action.dest for action in parser._actions} - {"help", "config"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown keys in JSON config {config_path}: {unknown}. Known keys: {sorted(known)}")
    parser.set_defaults(**values)
    return parser.parse_args(arguments)


def refuse_protected_path(path: str | Path, *, protected_dirs: Iterable[str | Path]) -> Path:
    """Return ``path`` resolved, or raise if it lies inside a protected folder."""

    resolved = Path(path).resolve()
    for protected in protected_dirs:
        protected = Path(protected).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"Refusing to write into {resolved}: {protected} is read-only for this code.")
    return resolved
