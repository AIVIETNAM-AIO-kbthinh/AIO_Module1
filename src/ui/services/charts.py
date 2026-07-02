"""Chart data preparation and Plotly figures for the Analysis page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .schema import DEFAULT_AGGREGATE_METRICS


def _metric_mean_column(metric: str) -> str:
    return f"{metric}_mean"


def filter_aggregated(
    aggregated: pd.DataFrame,
    *,
    dataset: str | None = None,
    regime: float | None = None,
    transfer: str | None = None,
) -> pd.DataFrame:
    frame = aggregated.copy()
    if frame.empty:
        return frame
    if dataset is not None and "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str) == dataset]
    if regime is not None and "regime" in frame.columns:
        frame = frame[frame["regime"].astype(float) == float(regime)]
    if transfer is not None and "transfer" in frame.columns:
        frame = frame[frame["transfer"].astype(str) == transfer]
    return frame.reset_index(drop=True)


def pivot_preprocessing_regime(
    aggregated: pd.DataFrame,
    *,
    dataset: str,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Pivot mean metric with preprocessing as rows and regime as columns."""
    subset = filter_aggregated(aggregated, dataset=dataset)
    value_column = _metric_mean_column(metric)
    if subset.empty or value_column not in subset.columns:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index="preprocessing",
        columns="regime",
        values=value_column,
        aggfunc="mean",
    )
    pivot.columns = [f"{int(round(float(col) * 100))}%" for col in pivot.columns]
    return pivot.sort_index()


def pivot_preprocessing_transfer(
    aggregated: pd.DataFrame,
    *,
    dataset: str,
    regime: float,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Pivot mean metric with preprocessing as rows and transfer as columns."""
    subset = filter_aggregated(aggregated, dataset=dataset, regime=regime)
    value_column = _metric_mean_column(metric)
    if subset.empty or value_column not in subset.columns:
        return pd.DataFrame()

    return subset.pivot_table(
        index="preprocessing",
        columns="transfer",
        values=value_column,
        aggfunc="mean",
    ).sort_index()


def regime_line_frame(
    aggregated: pd.DataFrame,
    *,
    dataset: str,
    transfer: str | None = None,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Long-format data for regime trend lines."""
    subset = filter_aggregated(aggregated, dataset=dataset, transfer=transfer)
    value_column = _metric_mean_column(metric)
    std_column = f"{metric}_std"
    if subset.empty or value_column not in subset.columns:
        return pd.DataFrame()

    columns = ["regime", "preprocessing", "transfer", value_column]
    if std_column in subset.columns:
        columns.append(std_column)
    frame = subset[columns].copy()
    frame["combo"] = frame["preprocessing"].astype(str) + " / " + frame["transfer"].astype(str)
    frame["regime_pct"] = frame["regime"].astype(float) * 100.0
    return frame.sort_values("regime_pct")


def pareto_frame(
    raw_frame: pd.DataFrame,
    *,
    dataset: str | None = None,
    regime: float | None = None,
) -> pd.DataFrame:
    """Per-run AUROC vs trainable parameters for a Pareto scatter."""
    frame = raw_frame.copy()
    if frame.empty:
        return frame
    if dataset is not None:
        frame = frame[frame["dataset"].astype(str) == dataset]
    if regime is not None:
        frame = frame[frame["regime"].astype(float) == float(regime)]
    required = ("auroc", "trainable_params", "preprocessing", "transfer")
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()

    extra = [column for column in ("run", "regime") if column in frame.columns]
    frame = frame[[*required, *extra]].copy()
    frame["combo"] = frame["preprocessing"].astype(str) + " / " + frame["transfer"].astype(str)
    return frame.reset_index(drop=True)


def pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated points (maximize AUROC, minimize trainable params)."""
    if frame.empty:
        return frame

    ordered = frame.sort_values(
        ["trainable_params", "auroc"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    frontier_rows = []
    best_auroc = float("-inf")
    for _, row in ordered.iterrows():
        score = float(row["auroc"])
        if score > best_auroc:
            frontier_rows.append(row)
            best_auroc = score
    if not frontier_rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.DataFrame(frontier_rows).reset_index(drop=True)


def heatmap_figure(pivot: pd.DataFrame, *, title: str, zmin: float = 0.0, zmax: float = 1.0) -> go.Figure:
    if pivot.empty:
        return go.Figure().update_layout(title=title)

    figure = px.imshow(
        pivot,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=zmin,
        zmax=zmax,
        labels=dict(color="mean"),
    )
    figure.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="preprocessing",
    )
    return figure


def regime_line_figure(line_frame: pd.DataFrame, *, title: str, metric: str = "auroc") -> go.Figure:
    value_column = _metric_mean_column(metric)
    if line_frame.empty:
        return go.Figure().update_layout(title=title)

    figure = px.line(
        line_frame,
        x="regime_pct",
        y=value_column,
        color="combo",
        markers=True,
        labels={"regime_pct": "Regime (%)", value_column: f"{metric} (mean)"},
        title=title,
    )
    return figure


def pareto_figure(points: pd.DataFrame, frontier: pd.DataFrame, *, title: str) -> go.Figure:
    if points.empty:
        return go.Figure().update_layout(title=title)

    scatter = px.scatter(
        points,
        x="trainable_params",
        y="auroc",
        color="combo",
        hover_data=["run", "regime"],
        labels={"trainable_params": "Trainable parameters", "auroc": "AUROC"},
        title=title,
    )
    if not frontier.empty:
        frontier_sorted = frontier.sort_values("trainable_params")
        scatter.add_trace(
            go.Scatter(
                x=frontier_sorted["trainable_params"],
                y=frontier_sorted["auroc"],
                mode="lines+markers",
                name="Pareto frontier",
                line=dict(color="black", dash="dash"),
                marker=dict(size=8),
            )
        )
    scatter.update_xaxes(type="log")
    return scatter


def available_metrics(aggregated: pd.DataFrame) -> list[str]:
    return [
        metric
        for metric in DEFAULT_AGGREGATE_METRICS
        if _metric_mean_column(metric) in aggregated.columns
    ]
