"""Data pipeline CLI: acquire -> convert -> validate -> write ``data/processed/``.

Examples
--------
Re-validate the dataset already in the repo::

    python scripts/prepare_data.py --input data/processed/MC1.txt --check-only

Convert a CSV the NLP lead produced into the MC1 format the trainer reads::

    python scripts/prepare_data.py \
        --input data/raw/pairs.csv \
        --sentence-1-col first --sentence-2-col second --label-col same_topic \
        --output data/processed/pairs_v2.txt

Pull from Kaggle first (needs ``poetry install --with data`` and credentials)::

    python scripts/prepare_data.py --kaggle owner/dataset-name --input-name pairs.csv \
        --output data/processed/pairs_v2.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run as `python scripts/prepare_data.py` without an
# editable install (`poetry install` / `pip install -e .`). No-op once installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qnlp_hpc.data.convert import SUFFIX_FORMATS, ColumnMapping, convert_file
from qnlp_hpc.data.schema import DatasetValidationError, check_dataset, deduplicate, write_pairs
from qnlp_hpc.paths import resolve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire, convert and validate sentence-pair datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_argument_group("source")
    source.add_argument(
        "--kaggle",
        metavar="OWNER/DATASET",
        help="Kaggle dataset reference to download into data/raw/ first.",
    )
    source.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if the target directory already has files.",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Input file to convert. Relative paths resolve against the repo root.",
    )
    source.add_argument(
        "--input-name",
        help="Filename to pick out of the downloaded Kaggle directory.",
    )
    source.add_argument(
        "--format",
        choices=sorted(set(SUFFIX_FORMATS.values())),
        help="Override the format inferred from the file extension.",
    )

    mapping = parser.add_argument_group("column mapping")
    mapping.add_argument("--sentence-1-col", default="sentence_1")
    mapping.add_argument("--sentence-2-col", default="sentence_2")
    mapping.add_argument("--label-col", default="label")
    mapping.add_argument(
        "--label-map",
        action="append",
        default=[],
        metavar="VALUE=0|1",
        help="Map a non-numeric label, e.g. --label-map same=1 --label-map other=0.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        type=Path,
        help="Destination .txt in MC1 format. Required unless --check-only.",
    )
    output.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and report, write nothing.",
    )
    output.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate pairs (they leak across the train/test split).",
    )
    output.add_argument(
        "--summary-json",
        type=Path,
        help="Also write the validation summary to this JSON file.",
    )

    return parser


def parse_label_map(entries: list[str]) -> dict[str, int]:
    label_map: dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--label-map expects VALUE=0|1, got {entry!r}")
        key, _, value = entry.partition("=")
        try:
            label_map[key.strip()] = int(value)
        except ValueError:
            raise SystemExit(f"--label-map value must be an integer, got {value!r}") from None
    return label_map


def resolve_input(args: argparse.Namespace) -> Path:
    if args.kaggle:
        # Imported lazily: the kaggle package is an optional dependency.
        from qnlp_hpc.data.acquire import download_kaggle_dataset

        directory = download_kaggle_dataset(args.kaggle, force=args.force_download)
        if args.input:
            return resolve(args.input)
        if not args.input_name:
            raise SystemExit(
                "--kaggle downloaded the dataset; now pass --input-name (or --input) "
                f"to select a file from {directory}."
            )
        matches = sorted(directory.rglob(args.input_name))
        if not matches:
            available = sorted(p.name for p in directory.rglob("*") if p.is_file())
            raise SystemExit(f"{args.input_name!r} not found in {directory}. Found: {available}")
        return matches[0]

    if not args.input:
        raise SystemExit("Nothing to do: pass --input (and/or --kaggle).")
    return resolve(args.input)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.check_only and not args.output:
        raise SystemExit("--output is required unless --check-only is given.")

    source_path = resolve_input(args)
    print(f"Reading: {source_path}")

    mapping = ColumnMapping(
        sentence_1=args.sentence_1_col,
        sentence_2=args.sentence_2_col,
        label=args.label_col,
        label_map=parse_label_map(args.label_map),
    )

    try:
        pairs = convert_file(source_path, mapping, args.format)
    except DatasetValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if not args.keep_duplicates:
        pairs, dropped = deduplicate(pairs)
        if dropped:
            print(f"Dropped {dropped} exact duplicate pair(s).")

    try:
        summary = check_dataset(pairs)
    except DatasetValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print("Dataset summary:")
    for key, value in summary.items():
        print(f"  {key:>18}: {value}")

    if args.check_only:
        print("--check-only: nothing written.")
    else:
        destination = resolve(args.output)
        write_pairs(pairs, destination)
        print(f"Wrote {len(pairs)} pair(s) -> {destination}")

    if args.summary_json:
        summary_path = resolve(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"source": str(source_path), **summary}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
