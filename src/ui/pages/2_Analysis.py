"""Charts, aggregated table, and decision guide."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components.layout import (
    init_page,
    page_title_suffix,
    render_disclaimer,
    render_page_header,
    render_sidebar_appearance,
    sidebar_section,
)
from src.ui.components.formatting import format_regime, format_metric
from src.ui.components.paths import path_input
from src.ui.config import DEFAULT_RESULTS_CSV
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

init_page(page_title_suffix("Analysis"))

with st.sidebar:
    render_sidebar_appearance()
    sidebar_section("Data")
    results_path = path_input("Results CSV", DEFAULT_RESULTS_CSV, key="analysis_results_csv")

render_disclaimer()
render_page_header(
    "Analysis & decision guide",
    subtitle="Compare lever interactions, inspect aggregated metrics, and get recommendations.",
)

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

with st.sidebar:
    sidebar_section("Chart context")
    dataset = st.selectbox("Dataset", options=datasets)
    regime = st.selectbox(
        "Regime (transfer heatmap / Pareto)",
        options=regimes,
        format_func=format_regime,
    )
    transfer = st.selectbox(
        "Transfer (regime lines)",
        options=[None, *transfers],
        format_func=lambda value: "All" if value is None else str(value),
    )
    metric = st.selectbox("Primary metric", options=metrics)

plotly_template = "plotly_dark" if st.session_state.get("ui_theme") == "dark" else "plotly_white"

tab_charts, tab_table, tab_guide = st.tabs(["Charts", "Aggregated table", "Decision guide"])

with tab_charts:
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        prep_regime = pivot_preprocessing_regime(aggregated, dataset=dataset, metric=metric)
        figure = heatmap_figure(
            prep_regime,
            title=f"{metric.upper()} — preprocessing × regime ({dataset})",
        )
        figure.update_layout(template=plotly_template)
        st.plotly_chart(figure, width="stretch")

    with col_right:
        prep_transfer = pivot_preprocessing_transfer(
            aggregated,
            dataset=dataset,
            regime=float(regime),
            metric=metric,
        )
        figure = heatmap_figure(
            prep_transfer,
            title=(
                f"{metric.upper()} — preprocessing × transfer "
                f"({dataset}, {format_regime(regime)})"
            ),
        )
        figure.update_layout(template=plotly_template)
        st.plotly_chart(figure, width="stretch")

    line_data = regime_line_frame(
        aggregated,
        dataset=dataset,
        transfer=transfer,
        metric=metric,
    )
    line_figure = regime_line_figure(
        line_data,
        title=f"{metric.upper()} vs regime ({dataset})",
        metric=metric,
    )
    line_figure.update_layout(template=plotly_template)
    st.plotly_chart(line_figure, width="stretch")

    pareto_points = pareto_frame(frame, dataset=dataset, regime=float(regime))
    frontier = pareto_frontier(pareto_points)
    pareto_chart = pareto_figure(
        pareto_points,
        frontier,
        title=f"Pareto frontier — AUROC vs params ({dataset}, {format_regime(regime)})",
    )
    pareto_chart.update_layout(template=plotly_template)
    st.plotly_chart(pareto_chart, width="stretch")

with tab_table:
    with st.container(border=True):
        st.caption("Mean ± std over seeds for each preprocessing × transfer × regime combo.")
        st.dataframe(aggregated, width="stretch", hide_index=True)

    with st.container(border=True):
        st.subheader("Best combo per dataset × regime (by AUROC)")
        st.dataframe(best_combinations(aggregated, metric=metric), width="stretch")

with tab_guide:
    st.markdown(
        "Pick your dataset, label budget (regime), and priority. "
        "The guide ranks lever combinations from completed runs."
    )

    with st.container(border=True):
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
