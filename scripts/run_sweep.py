"""Run an experiment many times and aggregate the results.

::

    # 30 seeds of the baseline
    python scripts/run_sweep.py --seeds 0-29

    # ansatz comparison: 3 dimensions x 5 seeds = 15 runs
    python scripts/run_sweep.py --seeds 0-4 --grid sentence_dim=2,4,8

    # see the plan without running anything
    python scripts/run_sweep.py --seeds 0-29 --dry-run

Each repetition runs as its own subprocess and writes to
``outputs/sweeps/<sweep>/<run-id>/``; the sweep then collects every
``mc1_spider_training_summary.csv`` into one table with mean, std and a 95%
confidence interval. Planning and aggregation live in ``qnlp_hpc.sweep``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from qnlp_hpc.config import ConfigError, apply_overrides, load_config
from qnlp_hpc.paths import OUTPUTS_DIR, resolve
from qnlp_hpc.sweep import (
    SweepError,
    aggregate,
    collect,
    execute,
    parse_assignments,
    parse_grid,
    parse_seeds,
    plan_runs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an experiment across seeds/parameters and aggregate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mc1_spider.yaml"))
    parser.add_argument(
        "--seeds",
        default="0-29",
        help="Seeds as a range and/or list, e.g. '0-29' or '1,2,3' (default: 0-29).",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="FIELD=V1,V2",
        help="Sweep a field across values, crossed with every seed. Repeatable.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="fixed",
        metavar="FIELD=VALUE",
        help="Override held constant across the sweep. Repeatable.",
    )
    parser.add_argument("--name", help="Sweep name (default: config name + timestamp).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Repetitions to run concurrently (default: 1). Each is a subprocess.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    parser.add_argument("--quiet", action="store_true", help="Only report failures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config_path = resolve(args.config)
        fixed = parse_assignments(args.fixed, "--set")
        grid = parse_grid(args.grid)
        seeds = parse_seeds(args.seeds)
    except SweepError as exc:
        raise SystemExit(str(exc)) from None

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1.")

    # Validate every combination up front, rather than failing on run 17.
    try:
        base = load_config(config_path)
        for key, options in grid.items():
            for option in options:
                apply_overrides(base, {**fixed, key: option})
        if not grid:
            apply_overrides(base, fixed)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from None

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    sweep_name = args.name or f"{base.name}_{stamp}"
    sweep_dir = OUTPUTS_DIR / "sweeps" / sweep_name

    runs = plan_runs(seeds, grid, fixed, sweep_dir)
    settings = len(runs) // len(seeds)

    print(f"Sweep:   {sweep_name}")
    print(f"Config:  {config_path}")
    print(f"Runs:    {len(runs)} ({len(seeds)} seed(s) x {settings} setting(s))")
    print(f"Output:  {sweep_dir}")
    if grid:
        print(f"Grid:    {grid}")
    if fixed:
        print(f"Fixed:   {fixed}")

    if args.dry_run:
        for run in runs[:10]:
            print(f"  {run.run_id}: {run.overrides}")
        if len(runs) > 10:
            print(f"  ... and {len(runs) - 10} more")
        print("--dry-run: nothing executed.")
        return 0

    sweep_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda run: execute(run, config_path, args.quiet), runs))
    else:
        results = [execute(run, config_path, args.quiet) for run in runs]

    total_seconds = (datetime.now(UTC) - started).total_seconds()
    failures = [result for result in results if not result["ok"]]

    frame = collect(runs, results)
    runs_csv = sweep_dir / "sweep_runs.csv"
    frame.to_csv(runs_csv, index=False)

    summary = aggregate(frame, sorted(grid))
    summary_csv = sweep_dir / "sweep_summary.csv"
    if not summary.empty:
        summary.to_csv(summary_csv, index=False)

    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps(
            {
                "sweep": sweep_name,
                "config": str(config_path),
                "seeds": seeds,
                "grid": grid,
                "fixed": fixed,
                "runs": len(runs),
                "failures": len(failures),
                "jobs": args.jobs,
                "wall_seconds": total_seconds,
                "finished_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nCompleted {len(runs) - len(failures)}/{len(runs)} runs in {total_seconds:.1f}s")
    print(f"Per-run results -> {runs_csv}")

    if summary.empty:
        print("No summary CSVs were produced — check the per-run output above.")
    else:
        print(f"Aggregate       -> {summary_csv}\n")
        headline = summary[summary["metric"] == "test_accuracy"]
        if not headline.empty:
            with pd.option_context("display.width", 120, "display.max_columns", None):
                print(headline.to_string(index=False))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
