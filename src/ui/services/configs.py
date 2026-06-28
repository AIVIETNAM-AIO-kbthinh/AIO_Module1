"""Resolve run YAML configs by run name."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ...utils.config import RunConfig, load_config
from ..config import resolve_path

DEFAULT_CONFIG_DIRS: tuple[str, ...] = (
    "configs/core",
    "configs",
)


def _read_run_name(path: Path) -> str | None:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        return None
    run_section = raw.get("run")
    if not isinstance(run_section, dict):
        return None
    name = run_section.get("name")
    return str(name) if name else None


@lru_cache(maxsize=1)
def build_config_index(
    dirs: tuple[str, ...] = DEFAULT_CONFIG_DIRS,
) -> dict[str, Path]:
    """Map ``run.name`` → config path by scanning YAML files once."""
    index: dict[str, Path] = {}
    for directory in dirs:
        root = resolve_path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.yaml")):
            try:
                run_name = _read_run_name(path)
            except (OSError, yaml.YAMLError):
                continue
            if run_name and run_name not in index:
                index[run_name] = path
    return index


def find_config_path(
    run_name: str,
    *,
    search_dirs: tuple[str, ...] | None = None,
) -> Path | None:
    """Return the config file path for ``run_name``, if discoverable."""
    dirs = search_dirs if search_dirs is not None else DEFAULT_CONFIG_DIRS
    for directory in dirs:
        candidate = resolve_path(directory) / f"{run_name}.yaml"
        if candidate.is_file():
            return candidate

    index = build_config_index(dirs)
    return index.get(run_name)


def load_run_config(
    run_name: str,
    *,
    search_dirs: tuple[str, ...] | None = None,
) -> RunConfig | None:
    """Load a typed ``RunConfig`` for ``run_name``."""
    path = find_config_path(run_name, search_dirs=search_dirs)
    if path is None:
        return None
    return load_config(str(path))


def load_run_config_raw(
    run_name: str,
    *,
    search_dirs: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Load the raw YAML mapping for ``run_name``."""
    path = find_config_path(run_name, search_dirs=search_dirs)
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return raw if isinstance(raw, dict) else None
