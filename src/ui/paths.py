"""Resolve repository-relative paths for the UI."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve ``path`` against the repository root when it is not absolute."""
    candidate = Path(path)
    root = (base or REPO_ROOT).resolve()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def to_relative_path(path: str | Path, *, base: Path | None = None) -> str:
    """Return a repo-relative POSIX path for display; fall back to the input."""
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = resolve_path(absolute, base=base)
    else:
        absolute = absolute.resolve()
    root = (base or REPO_ROOT).resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        return absolute.as_posix()
