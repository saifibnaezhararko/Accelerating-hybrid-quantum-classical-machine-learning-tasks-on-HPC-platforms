"""Tests for sweep planning and aggregation (Week 3, Software Lead)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qnlp_hpc.sweep import (
    SweepError,
    aggregate,
    collect,
    confidence_interval,
    parse_assignments,
    parse_grid,
    parse_seeds,
    plan_runs,
)


def test_parse_seeds_range() -> None:
    assert parse_seeds("0-4") == [0, 1, 2, 3, 4]


def test_parse_seeds_list_and_mixed() -> None:
    assert parse_seeds("1,2,3") == [1, 2, 3]
    assert parse_seeds("0-2, 7") == [0, 1, 2, 7]


def test_parse_seeds_allows_negative_values() -> None:
    # '-3' must read as a seed, not as a malformed range.
    assert parse_seeds("-3,4") == [-3, 4]


@pytest.mark.parametrize("text", ["", "abc", "5-1", "3-x"])
def test_parse_seeds_rejects_bad_input(text: str) -> None:
    with pytest.raises(SweepError):
        parse_seeds(text)


def test_parse_grid() -> None:
    assert parse_grid(["sentence_dim=2,4,8"]) == {"sentence_dim": ["2", "4", "8"]}


def test_parse_grid_rejects_missing_equals() -> None:
    with pytest.raises(SweepError, match="FIELD=VALUE"):
        parse_grid(["sentence_dim"])


def test_parse_assignments() -> None:
    assert parse_assignments(["epochs=5", "seed=1"], "--set") == {"epochs": "5", "seed": "1"}


def test_plan_runs_without_a_grid(tmp_path: Path) -> None:
    runs = plan_runs([0, 1], {}, {}, tmp_path)

    assert [run.run_id for run in runs] == ["seed0", "seed1"]
    assert runs[0].overrides["seed"] == "0"
    assert runs[0].output_dir == tmp_path / "seed0"


def test_plan_runs_crosses_grid_with_seeds(tmp_path: Path) -> None:
    runs = plan_runs([0, 1], {"sentence_dim": ["2", "4"]}, {"epochs": "3"}, tmp_path)

    assert len(runs) == 4
    assert [run.run_id for run in runs] == [
        "sentence_dim2_seed0",
        "sentence_dim2_seed1",
        "sentence_dim4_seed0",
        "sentence_dim4_seed1",
    ]
    # Fixed overrides ride along on every run.
    assert all(run.overrides["epochs"] == "3" for run in runs)
    # Each run writes somewhere different, otherwise they overwrite each other.
    assert len({run.output_dir for run in runs}) == 4


def test_plan_runs_multi_dimensional_grid(tmp_path: Path) -> None:
    runs = plan_runs([0], {"noun_dim": ["2", "4"], "sentence_dim": ["2", "4"]}, {}, tmp_path)
    assert len(runs) == 4


def test_seed_override_wins_over_a_fixed_one(tmp_path: Path) -> None:
    """--set seed=9 must not silently pin every repetition to one seed."""
    runs = plan_runs([0, 1], {}, {"seed": "9"}, tmp_path)
    assert [run.overrides["seed"] for run in runs] == ["0", "1"]


def test_confidence_interval_matches_the_t_distribution() -> None:
    values = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
    half_width, critical = confidence_interval(values)

    # t(0.975, df=4) = 2.776; s/sqrt(n) = 0.158114/sqrt(5) = 0.070711
    assert critical == pytest.approx(2.776445, rel=1e-4)
    assert half_width == pytest.approx(0.196326, rel=1e-4)


def test_confidence_interval_needs_two_points() -> None:
    half_width, critical = confidence_interval(pd.Series([0.5]))
    assert half_width != half_width  # NaN
    assert critical != critical


def test_collect_joins_summary_csvs(tmp_path: Path) -> None:
    runs = plan_runs([0, 1], {}, {}, tmp_path)
    for index, run in enumerate(runs):
        run.output_dir.mkdir(parents=True)
        pd.DataFrame([{"test_accuracy": 0.5 + index / 10, "training_seconds": 1.0}]).to_csv(
            run.summary_path, index=False
        )

    results = [
        {"run_id": "seed0", "ok": True, "wall_seconds": 1.0, "returncode": 0},
        {"run_id": "seed1", "ok": True, "wall_seconds": 2.0, "returncode": 0},
    ]
    frame = collect(runs, results)

    assert list(frame["run_id"]) == ["seed0", "seed1"]
    assert list(frame["test_accuracy"]) == [0.5, 0.6]
    assert list(frame["wall_seconds"]) == [1.0, 2.0]


def test_collect_keeps_rows_for_runs_that_produced_nothing(tmp_path: Path) -> None:
    runs = plan_runs([0], {}, {}, tmp_path)
    results = [{"run_id": "seed0", "ok": False, "wall_seconds": 0.5, "returncode": 1}]

    frame = collect(runs, results)

    assert len(frame) == 1
    assert "test_accuracy" not in frame.columns


def test_aggregate_groups_by_grid_dimension() -> None:
    frame = pd.DataFrame(
        [
            {"sentence_dim": "2", "ok": True, "test_accuracy": 0.50},
            {"sentence_dim": "2", "ok": True, "test_accuracy": 0.60},
            {"sentence_dim": "4", "ok": True, "test_accuracy": 0.70},
            {"sentence_dim": "4", "ok": True, "test_accuracy": 0.80},
        ]
    )
    summary = aggregate(frame, ["sentence_dim"])
    accuracy = summary[summary["metric"] == "test_accuracy"].set_index("sentence_dim")

    assert accuracy.loc["2", "mean"] == pytest.approx(0.55)
    assert accuracy.loc["4", "mean"] == pytest.approx(0.75)
    assert accuracy.loc["2", "runs"] == 2


def test_aggregate_excludes_failed_runs() -> None:
    frame = pd.DataFrame(
        [
            {"ok": True, "test_accuracy": 0.5},
            {"ok": True, "test_accuracy": 0.7},
            {"ok": False, "test_accuracy": 0.0},
        ]
    )
    summary = aggregate(frame, [])
    accuracy = summary[summary["metric"] == "test_accuracy"].iloc[0]

    assert accuracy["runs"] == 2
    assert accuracy["mean"] == pytest.approx(0.6)


def test_aggregate_on_an_empty_frame() -> None:
    assert aggregate(pd.DataFrame(), []).empty
