"""Tests for experiment form builders."""

import pytest

from src.ui.services.experiment import (
    build_config_dict,
    build_run_name,
    parse_run_config,
)


def test_build_run_name_matches_matrix_convention():
    assert build_run_name("pneumoniamnist", "p0", "lp", 1.0, 0) == "pneumoniamnist_p0_lp_r100_s0"
    assert build_run_name("dermamnist", "p_local", "lora", 0.05, 1) == "dermamnist_p_local_lora_r5_s1"


def test_build_config_dict_validates():
    raw = build_config_dict(
        dataset="pneumoniamnist",
        preprocessing="p0",
        transfer="lp",
        regime=1.0,
        seed=0,
        epochs=1,
        batch_size=8,
    )
    config = parse_run_config(raw)
    assert config.run.name == "pneumoniamnist_p0_lp_r100_s0"
    assert config.data.preprocessing.value == "p0"


def test_build_config_dict_rejects_bad_preprocessing():
    raw = build_config_dict(
        dataset="pneumoniamnist",
        preprocessing="bad",
        transfer="lp",
        regime=1.0,
        seed=0,
        epochs=1,
        batch_size=8,
    )
    with pytest.raises(ValueError):
        parse_run_config(raw)
