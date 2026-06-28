"""Tests for repository-relative path helpers."""

from pathlib import Path

from src.ui.paths import REPO_ROOT, resolve_path, to_relative_path


def test_resolve_relative_path():
    assert resolve_path("runs/results.csv") == REPO_ROOT / "runs" / "results.csv"


def test_to_relative_path_round_trip():
    absolute = REPO_ROOT / "runs" / "results.csv"
    assert to_relative_path(absolute) == "runs/results.csv"


def test_to_relative_path_accepts_relative_input():
    assert to_relative_path("configs/example_run.yaml") == "configs/example_run.yaml"


def test_resolve_absolute_path_unchanged(tmp_path: Path):
    file_path = tmp_path / "external.csv"
    file_path.write_text("x", encoding="utf-8")
    assert resolve_path(file_path) == file_path.resolve()
