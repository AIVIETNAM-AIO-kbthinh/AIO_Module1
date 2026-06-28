"""Load and validate ``results.csv`` for the dashboard."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_RESULTS_CSV, resolve_path, to_relative_path
from .schema import ALL_KNOWN_COLUMNS, REQUIRED_COLUMNS


@dataclass(frozen=True)
class ResultsTable:
    """Parsed experiment results, with load metadata for the UI."""

    frame: pd.DataFrame
    source_path: Path | None
    found: bool
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]

    @property
    def source_path_display(self) -> str | None:
        if self.source_path is None:
            return None
        return to_relative_path(self.source_path)

    @property
    def row_count(self) -> int:
        return len(self.frame)

    @property
    def run_names(self) -> list[str]:
        if self.frame.empty or "run" not in self.frame.columns:
            return []
        return self.frame["run"].astype(str).tolist()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALL_KNOWN_COLUMNS))


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ("regime", "seed", "trainable_params", "wall_clock_s", *(
        column for column in frame.columns if column in {"auroc", "accuracy", "macro_f1", "ece"}
    ))
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_results(path: Path | str | None = None) -> ResultsTable:
    """Load ``results.csv``; return an empty table when the file is missing."""
    relative = path if path is not None else DEFAULT_RESULTS_CSV
    source = resolve_path(relative)
    if not source.is_file():
        return ResultsTable(
            frame=_empty_frame(),
            source_path=None,
            found=False,
            missing_columns=REQUIRED_COLUMNS,
            extra_columns=(),
        )

    frame = pd.read_csv(source)
    present = set(frame.columns)
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in present)
    known = set(ALL_KNOWN_COLUMNS)
    extra = tuple(sorted(column for column in present if column not in known))

    frame = _coerce_numeric_columns(frame)
    if "run" in frame.columns:
        frame = frame.sort_values("run").reset_index(drop=True)

    return ResultsTable(
        frame=frame,
        source_path=source,
        found=True,
        missing_columns=missing,
        extra_columns=extra,
    )


def filter_results(
    table: ResultsTable,
    *,
    dataset: str | None = None,
    preprocessing: str | None = None,
    transfer: str | None = None,
    regime: float | None = None,
    seed: int | None = None,
    backbone: str | None = None,
) -> pd.DataFrame:
    """Return a filtered view of the results table."""
    frame = table.frame.copy()
    if frame.empty:
        return frame

    filters = {
        "dataset": dataset,
        "preprocessing": preprocessing,
        "transfer": transfer,
        "backbone": backbone,
    }
    for column, value in filters.items():
        if value is not None and column in frame.columns:
            frame = frame[frame[column].astype(str) == str(value)]

    if regime is not None and "regime" in frame.columns:
        frame = frame[frame["regime"].astype(float) == float(regime)]
    if seed is not None and "seed" in frame.columns:
        frame = frame[frame["seed"].astype(int) == int(seed)]

    return frame.reset_index(drop=True)
