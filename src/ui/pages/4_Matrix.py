"""Core matrix orchestration — generate configs and run account slices."""

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
from src.ui.components.paths import path_input
from src.ui.config import (
    CORE_MATRIX_RUN_COUNT,
    CORE_MATRIX_SPEC,
    DEFAULT_ASSIGNMENT_CSV,
    DEFAULT_MATRIX_CONFIG_DIR,
    DEFAULT_RESULTS_CSV,
)
from src.ui.services.batch import (
    completed_run_names,
    generate_matrix_configs,
    load_assignment,
    load_matrix_spec,
    run_batch,
    slice_for_account,
)

init_page(page_title_suffix("Matrix"))

if "matrix_logs" not in st.session_state:
    st.session_state.matrix_logs = []
if "matrix_generation" not in st.session_state:
    st.session_state.matrix_generation = None

with st.sidebar:
    render_sidebar_appearance()
    sidebar_section("Paths")
    matrix_spec = path_input("Matrix spec", CORE_MATRIX_SPEC, key="matrix_spec")
    config_dir = path_input("Config output dir", DEFAULT_MATRIX_CONFIG_DIR, key="matrix_out_dir")
    assignment_path = path_input("Assignment CSV", DEFAULT_ASSIGNMENT_CSV, key="matrix_assignment")
    results_csv = path_input("Results CSV", DEFAULT_RESULTS_CSV, key="matrix_results_csv")
    n_accounts = st.number_input("Kaggle accounts", min_value=1, value=5, step=1)

render_disclaimer()
render_page_header(
    "Experiment matrix",
    subtitle=f"Generate and run the locked {CORE_MATRIX_RUN_COUNT}-run core matrix.",
)

try:
    spec = load_matrix_spec(matrix_spec)
    axis_summary = ", ".join(
        f"{axis}×{len(values)}" for axis, values in sorted(spec.get("axes", {}).items())
    )
    st.info(f"Axes: {axis_summary} · spec `{CORE_MATRIX_SPEC}`")
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.container(border=True):
    if st.button("Generate configs", type="primary", use_container_width=True):
        try:
            st.session_state.matrix_generation = generate_matrix_configs(
                matrix_spec=matrix_spec,
                out_dir=config_dir,
                n_accounts=int(n_accounts),
                assignment_path=assignment_path,
            )
            st.session_state.matrix_logs.append(
                f"Generated {st.session_state.matrix_generation.config_count} configs "
                f"in `{config_dir}`."
            )
            st.success("Matrix configs generated.")
        except Exception as error:  # noqa: BLE001
            st.error(f"Generation failed: {error}")

assignment = load_assignment(assignment_path)
completed = completed_run_names(results_csv)

if assignment.empty:
    st.warning(
        f"No assignment file at `{assignment_path}`. Generate configs first."
    )
else:
    with st.container(border=True):
        st.subheader("Assignment")
        progress = len(completed & set(assignment["run"].astype(str)))
        st.metric("Matrix progress", f"{progress} / {len(assignment)} runs in results")
        st.dataframe(assignment, use_container_width=True, hide_index=True)

    with st.container(border=True):
        account = st.selectbox(
            "Account slice",
            sorted(assignment["account"].astype(int).unique()),
        )
        slice_df = slice_for_account(assignment, int(account))
        pending = slice_df[~slice_df["run"].astype(str).isin(completed)]
        st.caption(
            f"Account {account}: {len(slice_df)} runs · "
            f"{len(pending)} pending · {len(slice_df) - len(pending)} already recorded"
        )

        max_runs = st.number_input(
            "Max runs this batch (0 = all pending)",
            min_value=0,
            value=0,
            step=1,
            help="Limit batch size for smoke tests on CPU.",
        )

        skip_completed = st.checkbox("Skip runs already in results.csv", value=True)

        if st.button(
            "Run account slice",
            type="primary",
            use_container_width=True,
            disabled=slice_df.empty,
        ):
            paths = slice_df["config_path"].astype(str).tolist()
            if skip_completed:
                paths = [
                    path
                    for path, run_name in zip(paths, slice_df["run"].astype(str))
                    if run_name not in completed
                ]
            if max_runs > 0:
                paths = paths[: int(max_runs)]

            if not paths:
                st.info("No pending runs for this slice.")
            else:
                st.session_state.matrix_logs = [f"Batch: {len(paths)} run(s) queued."]
                progress_bar = st.progress(0.0)
                status = st.empty()

                def _on_progress(current: int, total: int, run_name: str) -> None:
                    progress_bar.progress(current / total)
                    status.caption(f"Running {current}/{total}: `{run_name}`")

                batch_result = run_batch(
                    paths,
                    skip_completed=False,
                    results_csv=results_csv,
                    log_lines=st.session_state.matrix_logs,
                    on_progress=_on_progress,
                )
                progress_bar.progress(1.0)
                st.success(
                    f"Batch finished: {batch_result.completed} completed, "
                    f"{batch_result.skipped} skipped, {batch_result.failed} failed."
                )
                if batch_result.errors:
                    st.error("\n".join(batch_result.errors))

if st.session_state.matrix_generation:
    with st.container(border=True):
        gen = st.session_state.matrix_generation
        st.subheader("Last generation")
        st.markdown(
            f"- Configs: `{gen.config_count}` in `{gen.out_dir}`\n"
            f"- Assignment: `{gen.assignment_path}`"
        )

if st.session_state.matrix_logs:
    with st.container(border=True):
        st.subheader("Log")
        st.code("\n".join(st.session_state.matrix_logs), language=None)
