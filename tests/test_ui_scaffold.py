"""Smoke tests for the Phase 0 UI scaffold."""

from pathlib import Path

from src.ui.bootstrap import REPO_ROOT, ensure_repo_on_path
from src.ui.config import CORE_MATRIX_RUN_COUNT, DEFAULT_RUNS_DIR, DISCLAIMER
from src.ui.paths import resolve_path, to_relative_path


def test_bootstrap_puts_repo_on_path():
    ensure_repo_on_path()
    assert str(REPO_ROOT) in __import__("sys").path


def test_ui_config_paths_exist():
    assert (REPO_ROOT / "configs" / "matrix" / "core.yaml").is_file()
    assert (REPO_ROOT / "configs" / "example_run.yaml").is_file()
    assert DEFAULT_RUNS_DIR == "runs"
    assert resolve_path(DEFAULT_RUNS_DIR) == REPO_ROOT / "runs"


def test_core_matrix_run_count():
    assert CORE_MATRIX_RUN_COUNT == 162


def test_disclaimer_is_non_empty():
    assert "not a clinical" in DISCLAIMER.lower()


def test_streamlit_pages_exist():
    pages = Path(__file__).resolve().parents[1] / "src" / "ui" / "pages"
    names = {path.name for path in pages.glob("*.py")}
    assert "1_Results.py" in names
    assert "4_Matrix.py" in names
