"""Per-run artifacts under ``runs/<run_name>/``."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DEFAULT_RUNS_DIR, resolve_path
from .configs import find_config_path


@dataclass(frozen=True)
class RunDetail:
    """Metrics and artifact paths for a single completed run."""

    run_name: str
    run_dir: Path
    metrics: dict[str, Any]
    metrics_path: Path
    checkpoint_path: Path | None
    config_path: Path | None


def run_directory(runs_dir: Path, run_name: str) -> Path:
    return runs_dir / run_name


def get_run_detail(
    run_name: str,
    *,
    runs_dir: Path | str | None = None,
    config_path: Path | str | None = None,
) -> RunDetail | None:
    """Load ``metrics.json`` for ``run_name``; return ``None`` if not found."""
    root = resolve_path(runs_dir) if runs_dir is not None else resolve_path(DEFAULT_RUNS_DIR)
    directory = run_directory(root, run_name)
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        return None

    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    checkpoint = directory / "checkpoint.pt"
    if config_path is not None:
        resolved_config = resolve_path(config_path)
    else:
        resolved_config = find_config_path(run_name)

    return RunDetail(
        run_name=run_name,
        run_dir=directory,
        metrics=metrics,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint if checkpoint.is_file() else None,
        config_path=resolved_config,
    )
