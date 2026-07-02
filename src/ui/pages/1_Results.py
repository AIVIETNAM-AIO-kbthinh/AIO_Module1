"""Results dashboard — filterable table and run detail."""

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
from src.ui.components.filters import apply_filters, render_results_filters
from src.ui.components.formatting import format_regime
from src.ui.components.paths import path_input
from src.ui.components.run_detail import render_run_detail
from src.ui.config import DEFAULT_RESULTS_CSV, DEFAULT_RUNS_DIR
from src.ui.services import load_results, summarize_completion

init_page(page_title_suffix("Results"))

with st.sidebar:
    render_sidebar_appearance()
    sidebar_section("Data source")
    results_path = path_input("Results CSV", DEFAULT_RESULTS_CSV, key="results_csv")
    runs_path = path_input("Runs directory", DEFAULT_RUNS_DIR, key="results_runs_dir")

render_disclaimer()
render_page_header(
    "Results",
    subtitle="Filter completed runs and inspect metrics, artifacts, and configuration.",
)

table = load_results(results_path)

if not table.found:
    st.info(
        f"No results file at `{results_path}`. Run an experiment first:\n\n"
        "`python scripts/run_experiment.py --config configs/example_run.yaml`"
    )
    st.stop()

if table.missing_columns:
    st.error(
        "Results file is missing required columns: "
        + ", ".join(table.missing_columns)
    )
    st.stop()

completion = summarize_completion(table.frame)
with st.container(border=True):
    summary_cols = st.columns(4)
    summary_cols[0].metric("Unique runs", completion.unique_runs)
    summary_cols[1].metric("Matrix progress", f"{completion.unique_runs}/{completion.expected}")
    summary_cols[2].metric("Rows in CSV", completion.completed)
    summary_cols[3].metric("Datasets", len(completion.datasets))

filter_state = render_results_filters(table.frame, key_prefix="results_")
filtered = apply_filters(table, filter_state)

if filtered.empty:
    st.warning("No runs match the current filters.")
    st.stop()

display = filtered.copy()
if "regime" in display.columns:
    display["regime_label"] = display["regime"].map(format_regime)

with st.container(border=True):
    st.subheader("Experiment runs")
    display_path = table.source_path_display or results_path
    st.caption(f"{len(filtered)} run(s) shown · source: `{display_path}`")

    selection = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

selected_rows = selection.selection.rows if selection.selection else []
if not selected_rows:
    st.info("Select a row in the table to inspect run details.")
    st.stop()

selected = filtered.iloc[selected_rows[0]]
run_name = str(selected["run"])

with st.container(border=True):
    st.subheader(f"Run detail — `{run_name}`")
    render_run_detail(run_name, runs_dir=runs_path)
