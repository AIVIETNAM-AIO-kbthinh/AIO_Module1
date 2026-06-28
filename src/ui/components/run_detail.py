"""Run detail panel for a single experiment."""

from pathlib import Path

import streamlit as st
import yaml

from ..config import to_relative_path
from ..services.configs import load_run_config_raw
from ..services.runs import get_run_detail
from .formatting import format_metric


def render_run_detail(run_name: str, *, runs_dir: str | Path | None = None) -> None:
    """Show metrics, artifact paths, and config for one run."""
    detail = get_run_detail(run_name, runs_dir=runs_dir)
    if detail is None:
        st.warning(f"No artifacts found for run `{run_name}`.")
        return

    metrics = detail.metrics
    cols = st.columns(4)
    for column, key in zip(cols, ("auroc", "accuracy", "macro_f1", "ece")):
        if key in metrics:
            column.metric(key.upper(), format_metric(metrics[key]))

    meta = st.columns(3)
    if "trainable_params" in metrics:
        meta[0].metric("Trainable params", f"{int(metrics['trainable_params']):,}")
    if "wall_clock_s" in metrics:
        meta[1].metric("Wall clock (s)", format_metric(metrics["wall_clock_s"], digits=1))
    if "seed" in metrics:
        meta[2].metric("Seed", str(int(metrics["seed"])))

    st.markdown("**Artifacts**")
    artifact_lines = [f"- Metrics: `{to_relative_path(detail.metrics_path)}`"]
    if detail.checkpoint_path is not None:
        artifact_lines.append(f"- Checkpoint: `{to_relative_path(detail.checkpoint_path)}`")
    else:
        artifact_lines.append("- Checkpoint: not found")
    if detail.config_path is not None:
        artifact_lines.append(f"- Config: `{to_relative_path(detail.config_path)}`")
    st.markdown("\n".join(artifact_lines))

    raw_config = load_run_config_raw(run_name)
    if raw_config is not None:
        with st.expander("Run config (YAML)"):
            st.code(yaml.safe_dump(raw_config, sort_keys=False), language="yaml")
