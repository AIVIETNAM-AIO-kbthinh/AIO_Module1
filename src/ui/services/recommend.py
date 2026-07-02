"""Rule-based decision guide over aggregated experiment results."""

from dataclasses import dataclass, replace
from enum import Enum

import pandas as pd

from .aggregation import aggregate_results
from .schema import DEFAULT_AGGREGATE_METRICS


class Priority(str, Enum):
    """What to optimize when recommending a lever combination."""

    AUROC = "auroc"
    FEW_PARAMS = "few_params"
    FAST = "fast"


@dataclass(frozen=True)
class Recommendation:
    dataset: str
    regime: float
    preprocessing: str
    transfer: str
    auroc_mean: float | None
    trainable_params_mean: float | None
    wall_clock_s_mean: float | None
    n_seeds: int
    rationale: str


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = list(DEFAULT_AGGREGATE_METRICS)
    for extra in ("trainable_params", "wall_clock_s"):
        if extra in frame.columns and extra not in columns:
            columns.append(extra)
    return tuple(columns)


_AUROC_TOLERANCE = 0.02


def _pick_auroc(row: pd.Series) -> Recommendation:
    return Recommendation(
        dataset=str(row["dataset"]),
        regime=float(row["regime"]),
        preprocessing=str(row["preprocessing"]),
        transfer=str(row["transfer"]),
        auroc_mean=float(row["auroc_mean"]) if pd.notna(row.get("auroc_mean")) else None,
        trainable_params_mean=(
            float(row["trainable_params_mean"])
            if pd.notna(row.get("trainable_params_mean"))
            else None
        ),
        wall_clock_s_mean=(
            float(row["wall_clock_s_mean"]) if pd.notna(row.get("wall_clock_s_mean")) else None
        ),
        n_seeds=int(row["n_seeds"]),
        rationale="Highest mean AUROC for this dataset and regime.",
    )


def _pick_by_cost(
    aggregated: pd.DataFrame,
    *,
    cost_column: str,
    rationale: str,
) -> Recommendation | None:
    """Pick the cheapest combo by ``cost_column``, restricted to combos within
    ``_AUROC_TOLERANCE`` of the best mean AUROC when that metric is available."""
    if cost_column not in aggregated.columns:
        return None

    if "auroc_mean" in aggregated.columns:
        best_auroc = float(aggregated["auroc_mean"].max())
        candidates = aggregated[aggregated["auroc_mean"] >= best_auroc - _AUROC_TOLERANCE]
        sort_columns = [cost_column, "auroc_mean"]
        ascending = [True, False]
    else:
        candidates = aggregated
        sort_columns = [cost_column]
        ascending = [True]

    best = candidates.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
    return replace(_pick_auroc(best), rationale=rationale)


def recommend(
    frame: pd.DataFrame,
    *,
    dataset: str,
    regime: float,
    priority: Priority,
) -> Recommendation | None:
    """Return the best preprocessing + transfer combo for the given context."""
    if frame.empty:
        return None

    subset = frame[
        (frame["dataset"].astype(str) == dataset)
        & (frame["regime"].astype(float) == float(regime))
    ]
    if subset.empty:
        return None

    aggregated = aggregate_results(subset, metric_columns=_metric_columns(subset))
    if aggregated.empty:
        return None

    if priority is Priority.AUROC:
        if "auroc_mean" not in aggregated.columns:
            return None
        best = aggregated.sort_values("auroc_mean", ascending=False, kind="mergesort").iloc[0]
        return _pick_auroc(best)

    if priority is Priority.FEW_PARAMS:
        return _pick_by_cost(
            aggregated,
            cost_column="trainable_params_mean",
            rationale=(
                f"Fewest trainable parameters among combos within {_AUROC_TOLERANCE:g} "
                "AUROC of the best."
            ),
        )

    if priority is Priority.FAST:
        return _pick_by_cost(
            aggregated,
            cost_column="wall_clock_s_mean",
            rationale=(
                f"Shortest mean wall-clock time among combos within {_AUROC_TOLERANCE:g} "
                "AUROC of the best."
            ),
        )

    return None
