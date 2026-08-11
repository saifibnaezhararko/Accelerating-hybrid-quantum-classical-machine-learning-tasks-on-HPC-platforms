"""Run the MC1 SpiderAnsatz experiment from a config file.

The contributed entrypoint ``scripts/train_mc1_spider.py`` runs the constants baked
into ``qnlp_hpc.mc1_spider.config``. This one runs the same pipeline from a YAML
config plus command-line overrides, so seeds and ansatz dimensions can be varied
without editing tracked source.

Examples
--------
::

    python scripts/run_experiment.py                                  # baseline
    python scripts/run_experiment.py --config configs/mc1_spider_smoke.yaml
    python scripts/run_experiment.py --set seed=7 --set sentence_dim=8
    python scripts/run_experiment.py --print-config --dry-run

Nothing from ``qnlp_hpc.mc1_spider`` is imported until the config has been applied:
those modules copy the constants at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run as `python scripts/run_experiment.py` without an
# editable install (`poetry install` / `pip install -e .`). No-op once installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qnlp_hpc.config import ConfigError, apply_overrides, apply_to_mc1_spider, load_config

DEFAULT_CONFIG = Path("configs/mc1_spider.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MC1 SpiderAnsatz experiment from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment config YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="FIELD=VALUE",
        help="Override a config field, e.g. --set seed=7. Repeatable.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config as JSON before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the config, then stop without training.",
    )
    return parser


def parse_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--set expects FIELD=VALUE, got {entry!r}")
        key, _, value = entry.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        if args.overrides:
            config = apply_overrides(config, parse_overrides(args.overrides))
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from None

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))

    print(f"Experiment: {config.name}")
    print(f"Data:       {config.resolved_data_path}")
    print(f"Outputs:    {config.resolved_output_dir}")

    if not config.resolved_data_path.is_file():
        raise SystemExit(
            f"Data file not found: {config.resolved_data_path}\n"
            "Build it first:  python scripts/prepare_data.py --input <source> "
            "--output data/processed/<name>.txt"
        )

    if args.dry_run:
        print("--dry-run: config is valid, stopping before training.")
        return 0

    try:
        apply_to_mc1_spider(config)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from None

    # Imported only now — see the module docstring.
    from qnlp_hpc.mc1_spider.experiment import run_experiment

    results = run_experiment()
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
