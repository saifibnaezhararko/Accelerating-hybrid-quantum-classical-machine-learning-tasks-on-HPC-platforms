"""Sweep planning and aggregation for repeated experiment runs.

The benchmark plan calls for ~30 repetitions per dataset with 95% confidence
intervals, plus ansatz comparisons. Two constraints shape this module:

* Repetitions cannot share a process — the pipeline binds its hyperparameters at
  import time — so each run is a subprocess of ``scripts/run_experiment.py``.
* Results have to be joinable afterwards, so every run gets its own output
  directory and its ``mc1_spider_training_summary.csv`` is collected by run id.

The CLI lives in ``scripts/run_sweep.py``; everything here is import-safe and
testable without launching anything.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qnlp_hpc.paths import REPO_ROOT

RUNNER = Path("scripts/run_experiment.py")
SUMMARY_FILENAME = "mc1_spider_training_summary.csv"

# Columns worth aggregating; the rest of the summary records constant settings.
METRIC_COLUMNS = (
    "test_accuracy",
    "best_development_accuracy",
    "best_development_loss",
    "best_train_accuracy",
    "best_train_loss",
    "training_seconds",
    "best_epoch",
    "epochs_completed",
)


class SweepError(ValueError):
    """Raised for a malformed sweep specification."""


@dataclass(frozen=True)
class Run:
    run_id: str
    overrides: dict[str, str]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        return self.output_dir / SUMMARY_FILENAME


def parse_seeds(text: str) -> list[int]:
    """Accept ``0-29``, ``1,2,3``, or a mix. Duplicates are preserved in order."""
    seeds: list[int] = []
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        # lstrip('-') so a negative seed is not read as a range.
        if "-" in piece.lstrip("-"):
            start, _, end = piece.partition("-")
            try:
                first, last = int(start), int(end)
            except ValueError:
                raise SweepError(f"Bad seed range {piece!r}; expected e.g. 0-29") from None
            if last < first:
                raise SweepError(f"Bad seed range {piece!r}: end is before start.")
            seeds.extend(range(first, last + 1))
        else:
            try:
                seeds.append(int(piece))
            except ValueError:
                raise SweepError(f"Bad seed {piece!r}") from None
    if not seeds:
        raise SweepError("No seeds parsed.")
    return seeds


def parse_assignments(entries: list[str], flag: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SweepError(f"{flag} expects FIELD=VALUE, got {entry!r}")
        key, _, value = entry.partition("=")
        values[key.strip()] = value.strip()
    return values


def parse_grid(entries: list[str]) -> dict[str, list[str]]:
    """``sentence_dim=2,4,8`` -> ``{"sentence_dim": ["2", "4", "8"]}``."""
    grid: dict[str, list[str]] = {}
    for key, value in parse_assignments(entries, "--grid").items():
        options = [item.strip() for item in value.split(",") if item.strip()]
        if not options:
            raise SweepError(f"--grid {key} has no values.")
        grid[key] = options
    return grid


def plan_runs(
    seeds: list[int],
    grid: dict[str, list[str]],
    fixed: dict[str, str],
    sweep_dir: Path,
) -> list[Run]:
    """Cartesian product of the grid, crossed with every seed."""
    keys = sorted(grid)
    combinations = list(itertools.product(*(grid[key] for key in keys))) if keys else [()]

    runs: list[Run] = []
    for combination in combinations:
        combination_overrides = dict(zip(keys, combination, strict=True))
        label = "_".join(f"{key}{value}" for key, value in combination_overrides.items())
        for seed in seeds:
            run_id = f"{label}_seed{seed}" if label else f"seed{seed}"
            output_dir = sweep_dir / run_id
            runs.append(
                Run(
                    run_id=run_id,
                    overrides={
                        **fixed,
                        **combination_overrides,
                        "seed": str(seed),
                        "output_dir": str(output_dir),
                    },
                    output_dir=output_dir,
                )
            )
    return runs


def execute(run: Run, config_path: Path, quiet: bool = False) -> dict[str, object]:
    """Run one repetition in its own interpreter.

    A fresh process per run costs a few seconds of import overhead, which is the
    price of the pipeline's import-time constants. It also isolates crashes: one
    bad seed cannot take the sweep down.
    """
    command = [sys.executable, str(RUNNER), "--config", str(config_path)]
    for key, value in run.overrides.items():
        command += ["--set", f"{key}={value}"]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    ok = completed.returncode == 0
    if not ok:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-5:]
        print(f"  [{run.run_id}] FAILED (exit {completed.returncode})")
        for line in tail:
            print(f"      {line}")
    elif not quiet:
        print(f"  [{run.run_id}] ok in {elapsed:.1f}s")

    return {
        "run_id": run.run_id,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "ok": ok,
    }


def collect(runs: list[Run], results: list[dict[str, object]]) -> pd.DataFrame:
    """Join each run's summary CSV with the sweep's own bookkeeping."""
    by_id = {result["run_id"]: result for result in results}
    rows = []
    for run in runs:
        record: dict[str, object] = {"run_id": run.run_id, **run.overrides}
        record.update(by_id.get(run.run_id, {}))
        if run.summary_path.is_file():
            summary = pd.read_csv(run.summary_path)
            if not summary.empty:
                record.update(summary.iloc[0].to_dict())
        rows.append(record)
    return pd.DataFrame(rows)


def confidence_interval(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Half-width of the CI on the mean, and the critical value used.

    Student's t, not 1.96: at 30 repetitions the difference is small but real, and
    ansatz comparisons are often run with far fewer.
    """
    n = int(values.count())
    if n < 2:
        return float("nan"), float("nan")

    standard_error = float(values.std(ddof=1)) / (n**0.5)
    try:
        from scipy import stats

        critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except ImportError:  # pragma: no cover - scipy ships with scikit-learn
        critical = 1.96
    return critical * standard_error, critical


def aggregate(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Mean / std / 95% CI per metric, grouped by the grid dimensions."""
    if frame.empty:
        return pd.DataFrame()

    successful = frame[frame["ok"].astype(bool)] if "ok" in frame.columns else frame
    metrics = [column for column in METRIC_COLUMNS if column in successful.columns]
    if successful.empty or not metrics:
        return pd.DataFrame()

    groups = successful.groupby(group_keys) if group_keys else [((), successful)]
    rows = []
    for key, group in groups:
        keys_tuple = key if isinstance(key, tuple) else (key,)
        label = dict(zip(group_keys, keys_tuple, strict=True))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            half_width, _ = confidence_interval(values)
            count = int(values.count())
            rows.append(
                {
                    **label,
                    "metric": metric,
                    "runs": count,
                    "mean": float(values.mean()) if count else float("nan"),
                    "std": float(values.std(ddof=1)) if count > 1 else float("nan"),
                    "ci95_half_width": half_width,
                    "min": float(values.min()) if count else float("nan"),
                    "max": float(values.max()) if count else float("nan"),
                }
            )
    return pd.DataFrame(rows)
