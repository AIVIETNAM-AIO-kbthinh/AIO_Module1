"""Single-run launcher — configure and execute one experiment."""

import streamlit as st

from src.ui.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from src.ui.components import render_disclaimer
from src.ui.components.paths import path_input
from src.ui.config import (
    APP_TITLE,
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_CSV,
    DEFAULT_RUNS_DIR,
    EXAMPLE_RUN_CONFIG,
)
from src.ui.services.experiment import (
    FORM_METRICS,
    FORM_PREPROCESSING,
    FORM_SPLITS,
    FORM_TRANSFER,
    build_config_dict,
    build_run_name,
    check_gpu_environment,
    config_to_yaml,
    parse_run_config,
    run_experiment_with_logs,
)

st.set_page_config(page_title=f"Run experiment — {APP_TITLE}", layout="wide")
render_disclaimer()

st.header("Run experiment")
st.caption(f"Form fields mirror `{EXAMPLE_RUN_CONFIG}`. Paths are relative to the repo root.")

if "experiment_logs" not in st.session_state:
    st.session_state.experiment_logs = []
if "last_run_results" not in st.session_state:
    st.session_state.last_run_results = None

with st.sidebar:
    st.subheader("Paths")
    output_dir = path_input("Output directory", DEFAULT_RUNS_DIR, key="run_output_dir")
    data_root = path_input("Data root", DEFAULT_DATA_ROOT, key="run_data_root")
    results_csv = path_input("Results CSV", DEFAULT_RESULTS_CSV, key="run_results_csv")

    if st.button("Check GPU", use_container_width=True):
        status = check_gpu_environment()
        if status.available:
            st.success(status.message)
        else:
            st.warning(status.message)

gpu_status = check_gpu_environment()
if gpu_status.available:
    st.success(gpu_status.message)
else:
    st.warning(gpu_status.message)

tab_run, tab_data, tab_model, tab_train = st.tabs(["Run", "Data", "Model", "Train & eval"])

with tab_run:
    run_cols = st.columns(3)
    seed = run_cols[0].number_input("Seed", min_value=0, value=0, step=1)
    auto_name = st.checkbox("Auto-generate run name", value=True)
    custom_name = ""
    if not auto_name:
        custom_name = st.text_input("Run name", value="my_custom_run")

with tab_data:
    data_cols = st.columns(3)
    dataset = data_cols[0].selectbox(
        "Dataset",
        ["pneumoniamnist", "dermamnist", "retinamnist", "bloodmnist"],
    )
    preprocessing = data_cols[1].selectbox("Preprocessing", FORM_PREPROCESSING)
    regime = data_cols[2].selectbox(
        "Regime",
        [1.0, 0.25, 0.10, 0.05],
        format_func=lambda value: f"{int(round(value * 100))}%",
    )
    data_cols2 = st.columns(3)
    resolution = data_cols2[0].selectbox("Resolution", [28, 64, 128, 224], index=1)
    as_rgb = data_cols2[1].checkbox("Expand to RGB", value=True)
    if preprocessing == "p_local":
        clahe_cols = st.columns(2)
        clahe_clip = clahe_cols[0].number_input("CLAHE clip limit", value=2.0, min_value=0.1)
        clahe_tile = clahe_cols[1].number_input("CLAHE tile size", value=8, min_value=2, step=1)
    else:
        clahe_clip, clahe_tile = 2.0, 8
    st.markdown("**Augmentation (train only)**")
    aug_cols = st.columns(3)
    hflip = aug_cols[0].checkbox("Horizontal flip", value=True)
    vflip = aug_cols[1].checkbox("Vertical flip", value=False)
    rotation = aug_cols[2].checkbox("Rotation", value=True)
    rotation_degrees = st.number_input("Rotation degrees", value=10.0, min_value=0.0)

with tab_model:
    model_cols = st.columns(3)
    backbone = model_cols[0].text_input("Backbone (timm id)", value="resnet50")
    transfer = model_cols[1].selectbox("Transfer strategy", FORM_TRANSFER)
    pretrained = model_cols[2].checkbox("ImageNet pretrained", value=True)
    if transfer == "lora":
        lora_cols = st.columns(3)
        lora_rank = lora_cols[0].number_input("LoRA rank", value=8, min_value=1)
        lora_alpha = lora_cols[1].number_input("LoRA alpha", value=16, min_value=1)
        lora_conv_target = lora_cols[2].selectbox("LoRA conv target", ["kernel3", "all"])
    else:
        lora_rank, lora_alpha, lora_conv_target = 8, 16, "kernel3"

with tab_train:
    train_cols = st.columns(3)
    epochs = train_cols[0].number_input("Epochs", value=30, min_value=1)
    batch_size = train_cols[1].number_input("Batch size", value=64, min_value=1)
    num_workers = train_cols[2].number_input("DataLoader workers", value=0, min_value=0)
    opt_cols = st.columns(3)
    optimizer_name = opt_cols[0].text_input("Optimizer", value="adamw")
    learning_rate = opt_cols[1].number_input("Learning rate", value=0.001, format="%.4f")
    weight_decay = opt_cols[2].number_input("Weight decay", value=0.0001, format="%.4f")
    sel_cols = st.columns(3)
    monitor = sel_cols[0].selectbox("Selection monitor", FORM_METRICS)
    patience = sel_cols[1].number_input("Early stopping patience (0=off)", value=0, min_value=0)
    eval_split = sel_cols[2].selectbox("Eval split", FORM_SPLITS)
    eval_metrics = st.multiselect("Eval metrics", FORM_METRICS, default=FORM_METRICS)
    ece_bins = st.number_input("ECE bins", value=15, min_value=1)

run_name = (
    build_run_name(dataset, preprocessing, transfer, float(regime), int(seed))
    if auto_name
    else custom_name
)

config_dict = build_config_dict(
    dataset=dataset,
    preprocessing=preprocessing,
    transfer=transfer,
    regime=float(regime),
    seed=int(seed),
    output_dir=output_dir,
    data_root=data_root,
    resolution=int(resolution),
    as_rgb=as_rgb,
    clahe_clip_limit=float(clahe_clip),
    clahe_tile_grid_size=int(clahe_tile),
    hflip_enabled=hflip,
    vflip_enabled=vflip,
    rotation_enabled=rotation,
    rotation_degrees=float(rotation_degrees),
    backbone=backbone,
    pretrained=pretrained,
    lora_rank=int(lora_rank),
    lora_alpha=int(lora_alpha),
    lora_conv_target=lora_conv_target,
    epochs=int(epochs),
    batch_size=int(batch_size),
    num_workers=int(num_workers),
    optimizer_name=optimizer_name,
    learning_rate=float(learning_rate),
    weight_decay=float(weight_decay),
    monitor=monitor,
    early_stopping_patience=int(patience),
    eval_metrics=eval_metrics,
    eval_split=eval_split,
    ece_bins=int(ece_bins),
    run_name=run_name,
)

st.subheader("YAML preview")
st.code(config_to_yaml(config_dict), language="yaml")

validation_error: str | None = None
try:
    parsed_config = parse_run_config(config_dict)
except ValueError as error:
    validation_error = str(error)
    parsed_config = None

if validation_error:
    st.error(f"Config validation failed: {validation_error}")

run_disabled = validation_error is not None or st.session_state.get("experiment_running", False)
if st.button("Run experiment", type="primary", disabled=run_disabled, use_container_width=True):
    st.session_state.experiment_running = True
    st.session_state.experiment_logs = [f"Starting run `{run_name}`..."]
    try:
        results = run_experiment_with_logs(parsed_config, st.session_state.experiment_logs)
        st.session_state.last_run_results = results
        st.session_state.experiment_logs.append(f"Finished `{run_name}`.")
        st.success(f"Run `{run_name}` completed.")
    except Exception as error:  # noqa: BLE001 — show training failures in UI
        st.session_state.experiment_logs.append(f"Error: {error}")
        st.error(f"Run failed: {error}")
    finally:
        st.session_state.experiment_running = False

if st.session_state.last_run_results:
    st.subheader("Last run metrics")
    st.json(st.session_state.last_run_results)

if st.session_state.experiment_logs:
    st.subheader("Log")
    st.code("\n".join(st.session_state.experiment_logs), language=None)
