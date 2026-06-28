"""Matrix config generation and batch execution for the UI."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from ...experiments.matrix import assign_slices, generate_configs, write_configs
from ...training.runner import run_experiment
from ...utils.config import RunConfig, load_config
from ..config import (
    CORE_MATRIX_SPEC,
    DEFAULT_ASSIGNMENT_CSV,
    DEFAULT_MATRIX_CONFIG_DIR,
    DEFAULT_RESULTS_CSV,
    resolve_path,
    to_relative_path
)
from .experiment import run_experiment_with_logs


@dataclass(frozen=True)
class MatrixGenerationResult:
    config_count: int
    out_dir: str
    assignment_path: str
    config_paths: tuple[str, ...]


@dataclass
class BatchRunResult:
    total: int
    completed: int
    skipped: int
    failed: int
    errors: list[str] = field(default_factory=list)
    last_run: str | None = None


def load_matrix_spec(spec_path: str = CORE_MATRIX_SPEC) -> dict[str, Any]:
    path = resolve_path(spec_path)
    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Matrix spec at '{spec_path}' must be a mapping.")
    return spec


def generate_matrix_configs(
    *,
    matrix_spec: str = CORE_MATRIX_SPEC,
    out_dir: str = DEFAULT_MATRIX_CONFIG_DIR,
    n_accounts: int = 5,
    assignment_path: str | None = None,
) -> MatrixGenerationResult:
    """Generate YAML configs and an assignment manifest."""
    spec = load_matrix_spec(matrix_spec)
    configs = generate_configs(spec)
    out_abs = resolve_path(out_dir)
    paths = write_configs(configs, str(out_abs))
    assignments = assign_slices([config["run"]["name"] for config in configs], n_accounts)

    manifest_rel = assignment_path or DEFAULT_ASSIGNMENT_CSV
    manifest_abs = resolve_path(manifest_rel)
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_abs, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "config_path", "account"])
        writer.writeheader()
        for assignment, path in zip(assignments, paths):
            writer.writerow(
                {
                    "run": assignment["run"],
                    "config_path": to_relative_path(path),
                    "account": assignment["account"],
                }
            )

    return MatrixGenerationResult(
        config_count=len(configs),
        out_dir=out_dir,
        assignment_path=manifest_rel,
        config_paths=tuple(to_relative_path(path) for path in paths),
    )


def load_assignment(assignment_path: str = DEFAULT_ASSIGNMENT_CSV) -> pd.DataFrame:
    path = resolve_path(assignment_path)
    if not path.is_file():
        return pd.DataFrame(columns=["run", "config_path", "account"])
    frame = pd.read_csv(path)
    if "config_path" in frame.columns:
        frame["config_path"] = frame["config_path"].astype(str)
    return frame


def completed_run_names(results_csv: str = DEFAULT_RESULTS_CSV) -> set[str]:
    path = resolve_path(results_csv)
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if "run" not in frame.columns:
        return set()
    return set(frame["run"].astype(str).unique())


def slice_for_account(assignment: pd.DataFrame, account: int) -> pd.DataFrame:
    if assignment.empty:
        return assignment
    return assignment[assignment["account"].astype(int) == int(account)].reset_index(drop=True)


def run_batch(
    config_paths: list[str],
    *,
    skip_completed: bool = True,
    results_csv: str = DEFAULT_RESULTS_CSV,
    log_lines: list[str] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BatchRunResult:
    """Run a list of config paths sequentially."""
    done = completed_run_names(results_csv) if skip_completed else set()
    logs = log_lines if log_lines is not None else []
    result = BatchRunResult(total=len(config_paths), completed=0, skipped=0, failed=0)

    for index, config_rel in enumerate(config_paths, start=1):
        config_abs = resolve_path(config_rel)
        config = load_config(str(config_abs))
        run_name = config.run.name
        result.last_run = run_name

        if skip_completed and run_name in done:
            result.skipped += 1
            logs.append(f"[skip] {run_name} already in {results_csv}")
            if on_progress:
                on_progress(index, len(config_paths), run_name)
            continue

        logs.append(f"[start] {to_relative_path(config_abs)}")
        if on_progress:
            on_progress(index, len(config_paths), run_name)
        try:
            run_experiment_with_logs(config, logs)
            result.completed += 1
            done.add(run_name)
            logs.append(f"[done] {run_name}")
        except Exception as error:  # noqa: BLE001 — surface batch failures in UI
            result.failed += 1
            message = f"[failed] {run_name}: {error}"
            result.errors.append(message)
            logs.append(message)

    return result
