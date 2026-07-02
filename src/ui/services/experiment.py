"""Build run configs from UI form values and execute experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import yaml

from ...training.runner import run_experiment
from ...utils.config import RunConfig, build_run_config
from ...utils.constants import Metric, Preprocessing, Split, TransferStrategy
from ...utils.logging import get_logger


def build_run_name(
    dataset: str,
    preprocessing: str,
    transfer: str,
    regime: float,
    seed: int,
) -> str:
    regime_token = f"r{int(round(float(regime) * 100))}"
    return f"{dataset}_{preprocessing}_{transfer}_{regime_token}_s{seed}"


def build_config_dict(
    *,
    dataset: str,
    preprocessing: str,
    transfer: str,
    regime: float,
    seed: int,
    output_dir: str = "runs",
    data_root: str = "data",
    resolution: int = 64,
    as_rgb: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    hflip_enabled: bool = True,
    hflip_p: float = 0.5,
    vflip_enabled: bool = False,
    vflip_p: float = 0.5,
    rotation_enabled: bool = True,
    rotation_degrees: float = 10.0,
    backbone: str = "resnet50",
    pretrained: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_conv_target: str = "kernel3",
    epochs: int = 30,
    batch_size: int = 64,
    num_workers: int = 0,
    optimizer_name: str = "adamw",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    monitor: str = "auroc",
    early_stopping_patience: int = 0,
    eval_metrics: list[str] | None = None,
    eval_split: str = "test",
    ece_bins: int = 15,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Build a run-config mapping from form fields."""
    name = run_name or build_run_name(dataset, preprocessing, transfer, regime, seed)
    metrics = eval_metrics or ["auroc", "accuracy", "macro_f1", "ece"]
    return {
        "run": {"name": name, "seed": seed, "output_dir": output_dir},
        "data": {
            "dataset": dataset,
            "resolution": resolution,
            "regime": regime,
            "root": data_root,
            "as_rgb": as_rgb,
            "preprocessing": preprocessing,
            "clahe": {
                "clip_limit": clahe_clip_limit,
                "tile_grid_size": clahe_tile_grid_size,
            },
            "augmentation": {
                "horizontal_flip": {"enabled": hflip_enabled, "p": hflip_p},
                "vertical_flip": {"enabled": vflip_enabled, "p": vflip_p},
                "rotation": {"enabled": rotation_enabled, "degrees": rotation_degrees},
            },
        },
        "model": {
            "backbone": backbone,
            "pretrained": pretrained,
            "transfer": transfer,
            "lora": {
                "rank": lora_rank,
                "alpha": lora_alpha,
                "conv_target": lora_conv_target,
            },
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "optimizer": {
                "name": optimizer_name,
                "lr": learning_rate,
                "kwargs": {"weight_decay": weight_decay},
            },
            "selection": {
                "monitor": monitor,
                "early_stopping_patience": early_stopping_patience,
            },
        },
        "eval": {"metrics": metrics, "split": eval_split, "ece_bins": ece_bins},
    }


def config_to_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def parse_run_config(config: dict[str, Any]) -> RunConfig:
    return build_run_config(config)


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    message: str
    device_count: int = 0


def check_gpu_environment() -> GpuStatus:
    """Report whether CUDA is available for training."""
    if not torch.cuda.is_available():
        return GpuStatus(
            available=False,
            message="No CUDA device detected. Training will run on CPU (slow).",
        )
    count = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(index) for index in range(count))
    return GpuStatus(
        available=True,
        message=f"CUDA available ({count} GPU(s)): {names}",
        device_count=count,
    )


class _ListLogHandler(logging.Handler):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append(self.format(record))


def run_experiment_with_logs(
    config: RunConfig,
    log_lines: list[str],
) -> dict[str, Any]:
    """Execute one run and append training log lines to ``log_lines``."""
    logger = get_logger("training")
    handler = _ListLogHandler(log_lines)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        return run_experiment(config)
    finally:
        logger.removeHandler(handler)


# Re-export enums for form widgets.
FORM_PREPROCESSING = [member.value for member in Preprocessing]
FORM_TRANSFER = [member.value for member in TransferStrategy]
FORM_METRICS = [member.value for member in Metric]
FORM_SPLITS = [member.value for member in Split]
