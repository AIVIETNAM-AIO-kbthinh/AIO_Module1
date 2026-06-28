"""UI-level constants and defaults.

Paths shown in the dashboard are repository-relative strings (for example
``runs/results.csv``). Use ``src.ui.paths.resolve_path`` before filesystem access.
"""

from .paths import REPO_ROOT, resolve_path, to_relative_path

# Repository-relative defaults (safe to show in text inputs).
DEFAULT_RUNS_DIR = "runs"
DEFAULT_RESULTS_CSV = "runs/results.csv"
DEFAULT_DATA_ROOT = "data"
CORE_MATRIX_SPEC = "configs/matrix/core.yaml"
EXAMPLE_RUN_CONFIG = "configs/example_run.yaml"
DEFAULT_MATRIX_CONFIG_DIR = "configs/core"
DEFAULT_ASSIGNMENT_CSV = "configs/core/assignment.csv"
CORE_MATRIX_RUN_COUNT = 162

DISCLAIMER = (
    "Research prototype only. This tool supports experiment analysis on MedMNIST. "
    "It is not a clinical diagnostic system and must not be used for patient care."
)

APP_TITLE = "AIO Module 1 — Experiment Dashboard"
APP_TAGLINE = (
    "Generalization levers on low-data medical image classification "
    "(preprocessing × transfer strategy × data regime)."
)

__all__ = [
    "APP_TAGLINE",
    "APP_TITLE",
    "CORE_MATRIX_RUN_COUNT",
    "CORE_MATRIX_SPEC",
    "DEFAULT_ASSIGNMENT_CSV",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MATRIX_CONFIG_DIR",
    "DEFAULT_RESULTS_CSV",
    "DEFAULT_RUNS_DIR",
    "DISCLAIMER",
    "EXAMPLE_RUN_CONFIG",
    "REPO_ROOT",
    "resolve_path",
    "to_relative_path",
]
