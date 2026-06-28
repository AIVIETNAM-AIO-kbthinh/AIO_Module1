"""Charts, aggregated table, and decision guide."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.components.formatting import format_regime, format_metric
from src.ui.components.paths import path_input
from src.ui.config import APP_TITLE, DEFAULT_RESULTS_CSV
from src.ui.services import aggregate_results, best_combinations, load_results
from src.ui.services.charts import (
    available_metrics,
    heatmap_figure,
    pareto_figure,
    pareto_frame,
    pareto_frontier,
    pivot_preprocessing_regime,
    pivot_preprocessing_transfer,
    regime_line_figure,
    regime_line_frame,
)
from src.ui.services.recommend import Priority, recommend

st.set_page_config(page_title=f"Analysis — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Analysis & decision guide")

with st.sidebar:
    results_path = path_input("Results CSV", DEFAULT_RESULTS_CSV, key="analysis_results_csv")
table = load_results(results_path)

if not table.found or table.frame.empty:
    st.info(
        f"No results at `{results_path}`. Complete some runs before using analysis tools."
    )
    st.stop()

frame = table.frame
aggregated = aggregate_results(frame)
metrics = available_metrics(aggregated) or ["auroc"]

datasets = sorted(frame["dataset"].astype(str).unique())
regimes = sorted(frame["regime"].dropna().unique(), reverse=True)
transfers = sorted(frame["transfer"].astype(str).unique())

st.sidebar.subheader("Chart context")
dataset = st.sidebar.selectbox("Dataset", options=datasets)
regime = st.sidebar.selectbox(
    "Regime (for transfer heatmap / Pareto)",
    options=regimes,
    format_func=format_regime,
)
transfer = st.sidebar.selectbox(
    "Transfer (optional, for regime lines)",
    options=[None, *transfers],
    format_func=lambda value: "All" if value is None else str(value),
)
metric = st.sidebar.selectbox("Primary metric", options=metrics)

tab_charts, tab_table, tab_guide = st.tabs(["Charts", "Aggregated table", "Decision guide"])

with tab_charts:
    col_left, col_right = st.columns(2)

    with col_left:
        prep_regime = pivot_preprocessing_regime(aggregated, dataset=dataset, metric=metric)
        st.plotly_chart(
            heatmap_figure(
                prep_regime,
                title=f"{metric.upper()} — preprocessing × regime ({dataset})",
            ),
            use_container_width=True,
        )

    with col_right:
        prep_transfer = pivot_preprocessing_transfer(
            aggregated,
            dataset=dataset,
            regime=float(regime),
            metric=metric,
        )
        st.plotly_chart(
            heatmap_figure(
                prep_transfer,
                title=(
                    f"{metric.upper()} — preprocessing × transfer "
                    f"({dataset}, {format_regime(regime)})"
                ),
            ),
            use_container_width=True,
        )

    line_data = regime_line_frame(
        aggregated,
        dataset=dataset,
        transfer=transfer,
        metric=metric,
    )
    st.plotly_chart(
        regime_line_figure(
            line_data,
            title=f"{metric.upper()} vs regime ({dataset})",
            metric=metric,
        ),
        use_container_width=True,
    )

    pareto_points = pareto_frame(frame, dataset=dataset, regime=float(regime))
    frontier = pareto_frontier(pareto_points)
    st.plotly_chart(
        pareto_figure(
            pareto_points,
            frontier,
            title=f"Pareto frontier — AUROC vs params ({dataset}, {format_regime(regime)})",
        ),
        use_container_width=True,
    )

with tab_table:
    st.caption("Mean ± std over seeds for each preprocessing × transfer × regime combo.")
    st.dataframe(aggregated, use_container_width=True, hide_index=True)

    st.subheader("Best combo per dataset × regime (by AUROC)")
    st.dataframe(best_combinations(aggregated, metric=metric), use_container_width=True)

with tab_guide:
    st.markdown(
        "Pick your dataset, label budget (regime), and priority. "
        "The guide ranks lever combinations from completed runs."
    )

    guide_cols = st.columns(3)
    guide_dataset = guide_cols[0].selectbox(
        "Dataset",
        options=datasets,
        key="guide_dataset",
    )
    guide_regime = guide_cols[1].selectbox(
        "Regime",
        options=regimes,
        format_func=format_regime,
        key="guide_regime",
    )
    priority_label = guide_cols[2].selectbox(
        "Priority",
        options=[
            (Priority.AUROC, "Best AUROC"),
            (Priority.FEW_PARAMS, "Fewest trainable parameters"),
            (Priority.FAST, "Fastest training"),
        ],
        format_func=lambda item: item[1],
        key="guide_priority",
    )
    priority = priority_label[0]

    suggestion = recommend(
        frame,
        dataset=guide_dataset,
        regime=float(guide_regime),
        priority=priority,
    )

    if suggestion is None:
        st.warning("No completed runs match that dataset and regime.")
    else:
        st.success("Recommended configuration")
        rec_cols = st.columns(4)
        rec_cols[0].metric("Preprocessing", suggestion.preprocessing)
        rec_cols[1].metric("Transfer", suggestion.transfer)
        rec_cols[2].metric("AUROC (mean)", format_metric(suggestion.auroc_mean))
        rec_cols[3].metric("Seeds", suggestion.n_seeds)

        detail_cols = st.columns(2)
        if suggestion.trainable_params_mean is not None:
            detail_cols[0].metric(
                "Trainable params (mean)",
                f"{int(suggestion.trainable_params_mean):,}",
            )
        if suggestion.wall_clock_s_mean is not None:
            detail_cols[1].metric(
                "Wall clock (mean, s)",
                format_metric(suggestion.wall_clock_s_mean, digits=1),
            )
        st.caption(suggestion.rationale)
