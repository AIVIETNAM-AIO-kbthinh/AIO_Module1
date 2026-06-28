"""Streamlit home page — experiment dashboard entry point."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.config import (
    APP_TAGLINE,
    APP_TITLE,
    CORE_MATRIX_RUN_COUNT,
    DEFAULT_RESULTS_CSV,
    DEFAULT_RUNS_DIR,
)
from src.ui.services import load_results, summarize_completion

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_disclaimer()

st.title(APP_TITLE)
st.caption(APP_TAGLINE)

st.markdown(
    """
This dashboard supports a **controlled study** of generalization when labeled medical
images are scarce. Use the sidebar to inspect results, compare lever combinations,
launch single runs, or orchestrate the full experiment matrix.
"""
)

overview_col, progress_col = st.columns([3, 2])

with overview_col:
    st.subheader("About the study")

    st.markdown(
        """
We ask how two levers interact as training labels shrink on the public
**MedMNIST** benchmark (PneumoniaMNIST, DermaMNIST, and related flags — **not**
for clinical use).
"""
    )

    lever_col, matrix_col = st.columns(2)

    with lever_col:
        st.markdown("**Input-space lever (preprocessing)**")
        st.markdown(
            """
| Arm | Description |
|-----|-------------|
| `p0` | No extra contrast |
| `p_global` | Global histogram equalization |
| `p_local` | Local contrast (CLAHE) |
"""
        )
        st.markdown("**Parameter-space lever (transfer)**")
        st.markdown(
            """
| Arm | Description |
|-----|-------------|
| `lp` | Linear probe — train head only |
| `ft` | Full fine-tune |
| `lora` | LoRA adapters + head |
"""
        )

    with matrix_col:
        st.markdown("**Locked core matrix**")
        st.markdown(
            f"""
- **{CORE_MATRIX_RUN_COUNT} runs** total
- 3 preprocessing × 3 transfer × 3 regimes
- Regimes: **100%**, **10%**, **5%** of training labels
- 2 datasets × 3 random seeds
- Backbone: **ResNet-50** (ImageNet), **64×64**
"""
        )
        st.markdown("**Evaluation**")
        st.markdown(
            """
- Primary metric: **AUROC** on the official test split
- Also tracked: accuracy, macro-F1, ECE
- Outputs: per-run checkpoint, `metrics.json`, consolidated `results.csv`
"""
        )

    st.subheader("Dashboard pages")
    st.markdown(
        """
| Page | Purpose |
|------|---------|
| **Results** | Filter completed runs and inspect metrics, artifacts, and config |
| **Analysis** | Interaction charts, aggregated tables, and the decision guide |
| **Run experiment** | Configure and launch a single run from a form |
| **Matrix** | Generate the 162-run configs and execute an account slice in batch |
"""
    )

with progress_col:
    st.subheader("Current progress")
    results_table = load_results()
    completion = summarize_completion(results_table.frame)

    if results_table.found:
        st.metric("Unique runs", f"{completion.unique_runs} / {completion.expected}")
        st.metric("Rows in CSV", completion.completed)
        if completion.datasets:
            st.markdown("**Datasets recorded**")
            for name in completion.datasets:
                st.markdown(f"- `{name}`")
        st.caption(f"Source: `{DEFAULT_RESULTS_CSV}`")
    else:
        st.info(
            f"No `{DEFAULT_RESULTS_CSV}` yet. Start from **Run experiment** or run via CLI:"
        )
        st.code(
            "python scripts/run_experiment.py --config configs/example_run.yaml",
            language="bash",
        )

    st.markdown("**Default paths**")
    st.markdown(f"- Results: `{DEFAULT_RESULTS_CSV}`")
    st.markdown(f"- Runs: `{DEFAULT_RUNS_DIR}/<run_name>/`")

    if completion.missing > 0 and results_table.found:
        st.progress(
            completion.unique_runs / completion.expected,
            text=f"{completion.missing} core matrix run(s) remaining",
        )
