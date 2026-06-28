"""Tests for the Phase 1 UI data services."""

from pathlib import Path

import pandas as pd
import pytest

from src.ui.services import (
    aggregate_results,
    best_combinations,
    build_config_index,
    filter_results,
    find_config_path,
    get_run_detail,
    load_results,
    load_run_config,
    summarize_completion,
)
from src.ui.services.configs import build_config_index as _build_config_index

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ui"
_RESULTS_CSV = _FIXTURES / "results.csv"
_CONFIG_DIR = _FIXTURES / "configs" / "core"
_RUNS_DIR = _FIXTURES / "runs"


@pytest.fixture(autouse=True)
def clear_config_index_cache():
    _build_config_index.cache_clear()
    yield
    _build_config_index.cache_clear()


def test_load_results_missing_file(tmp_path):
    table = load_results(tmp_path / "missing.csv")
    assert not table.found
    assert table.source_path is None
    assert table.row_count == 0
    assert table.missing_columns


def test_load_results_parses_fixture():
    table = load_results(_RESULTS_CSV)
    assert table.found
    assert table.row_count == 6
    assert table.missing_columns == ()
    assert table.frame["regime"].dtype.kind == "f"
    assert list(table.run_names)[0] == "dermamnist_p_global_ft_r5_s0"


def test_filter_results_by_axes():
    table = load_results(_RESULTS_CSV)
    filtered = filter_results(
        table,
        dataset="pneumoniamnist",
        preprocessing="p0",
        transfer="lp",
        regime=1.0,
    )
    assert len(filtered) == 3
    assert set(filtered["seed"].astype(int)) == {0, 1, 2}


def test_get_run_detail_loads_metrics():
    detail = get_run_detail(
        "pneumoniamnist_p0_lp_r100_s0",
        runs_dir="tests/fixtures/ui/runs",
        config_path="tests/fixtures/ui/configs/core/pneumoniamnist_p0_lp_r100_s0.yaml",
    )
    assert detail is not None
    assert detail.metrics["auroc"] == 0.81
    assert detail.checkpoint_path is None
    assert detail.config_path is not None


def test_get_run_detail_missing_returns_none(tmp_path):
    assert get_run_detail("missing_run", runs_dir=tmp_path) is None


def test_find_config_by_filename(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "my_run.yaml"
    path.write_text("run:\n  name: my_run\n", encoding="utf-8")
    assert find_config_path("my_run", search_dirs=(config_dir,)) == path


def test_build_config_index_reads_run_name_from_yaml():
    index = build_config_index(("tests/fixtures/ui/configs/core",))
    assert "pneumoniamnist_p0_lp_r100_s0" in index
    assert index["pneumoniamnist_p0_lp_r100_s0"].name == "pneumoniamnist_p0_lp_r100_s0.yaml"


def test_load_run_config_returns_typed_config():
    config = load_run_config(
        "pneumoniamnist_p0_lp_r100_s0",
        search_dirs=("tests/fixtures/ui/configs/core",),
    )
    assert config is not None
    assert config.run.name == "pneumoniamnist_p0_lp_r100_s0"
    assert config.data.preprocessing.value == "p0"


def test_summarize_completion():
    table = load_results(_RESULTS_CSV)
    stats = summarize_completion(table.frame, expected=162)
    assert stats.completed == 6
    assert stats.unique_runs == 6
    assert stats.missing == 156
    assert stats.fraction == pytest.approx(6 / 162)
    assert "pneumoniamnist" in stats.datasets


def test_aggregate_results_mean_std():
    table = load_results(_RESULTS_CSV)
    aggregated = aggregate_results(table.frame)
    lp_row = aggregated[
        (aggregated["dataset"] == "pneumoniamnist")
        & (aggregated["preprocessing"] == "p0")
        & (aggregated["transfer"] == "lp")
        & (aggregated["regime"] == 1.0)
    ]
    assert len(lp_row) == 1
    assert lp_row.iloc[0]["n_seeds"] == 3
    assert lp_row.iloc[0]["auroc_mean"] == pytest.approx(0.82, abs=1e-9)
    assert lp_row.iloc[0]["auroc_std"] == pytest.approx(0.01, abs=1e-9)


def test_best_combinations_picks_highest_auroc_per_regime():
    table = load_results(_RESULTS_CSV)
    aggregated = aggregate_results(table.frame)
    best = best_combinations(aggregated, metric="auroc")
    pneumo = best[best["dataset"] == "pneumoniamnist"]
    assert len(pneumo) == 2
    full_data = pneumo[pneumo["regime"] == 1.0].iloc[0]
    low_data = pneumo[pneumo["regime"] == 0.1].iloc[0]
    assert full_data["transfer"] == "lp"
    assert low_data["transfer"] == "lora"
    assert full_data["auroc_mean"] > low_data["auroc_mean"]


def test_aggregate_results_rejects_missing_group_columns():
    with pytest.raises(ValueError):
        aggregate_results(pd.DataFrame({"run": ["a"]}))
