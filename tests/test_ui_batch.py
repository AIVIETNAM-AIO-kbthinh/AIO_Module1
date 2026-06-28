"""Tests for matrix generation and batch helpers."""

from pathlib import Path

import pandas as pd
import yaml

from src.ui.services.batch import (
    completed_run_names,
    generate_matrix_configs,
    load_assignment,
    slice_for_account,
)


def test_generate_matrix_configs_writes_files(tmp_path):
    spec = {
        "base": {
            "run": {"output_dir": "runs"},
            "data": {
                "resolution": 64,
                "root": "data",
                "as_rgb": True,
                "preprocessing": "p0",
            },
            "model": {"backbone": "resnet50", "pretrained": True, "transfer": "lp"},
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "optimizer": {
                    "name": "adamw",
                    "lr": 0.001,
                    "kwargs": {"weight_decay": 0.0001},
                },
            },
            "eval": {"metrics": ["auroc"]},
        },
        "axes": {
            "dataset": ["pneumoniamnist"],
            "preprocessing": ["p0", "p_global"],
            "transfer": ["lp"],
            "regime": [1.0],
            "seed": [0],
        },
    }
    spec_path = tmp_path / "spec.yaml"
    with open(spec_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle)

    out_dir = tmp_path / "configs"
    assignment_path = tmp_path / "assignment.csv"
    result = generate_matrix_configs(
        matrix_spec=str(spec_path),
        out_dir=str(out_dir),
        n_accounts=2,
        assignment_path=str(assignment_path),
    )
    assert result.config_count == 2
    assert len(list(out_dir.glob("*.yaml"))) == 2

    assignment = load_assignment(str(assignment_path))
    assert len(assignment) == 2
    assert set(assignment["account"].astype(int)) == {0, 1}


def test_slice_for_account_filters_rows():
    frame = pd.DataFrame(
        {
            "run": ["a", "b", "c"],
            "config_path": ["x.yaml", "y.yaml", "z.yaml"],
            "account": [0, 1, 0],
        }
    )
    sliced = slice_for_account(frame, 0)
    assert list(sliced["run"]) == ["a", "c"]


def test_completed_run_names_reads_fixture():
    names = completed_run_names("tests/fixtures/ui/results.csv")
    assert "pneumoniamnist_p0_lp_r100_s0" in names
