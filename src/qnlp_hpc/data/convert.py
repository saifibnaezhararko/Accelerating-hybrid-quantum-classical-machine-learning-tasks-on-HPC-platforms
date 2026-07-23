"""Unified format conversion: CSV / TSV / JSON / JSONL / MC1 text -> canonical pairs.

Every dataset the NLP lead brings in arrives in a different shape. This module is
the single funnel: point it at a file, say which columns hold the two sentences and
the label, and it emits the canonical MC1 format that
``qnlp_hpc.mc1_spider.data.load_mc1`` already reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from qnlp_hpc.data.schema import (
    DatasetValidationError,
    SentencePair,
    make_pair,
    read_pairs,
)

SUFFIX_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".txt": "mc1",
}


@dataclass
class ColumnMapping:
    """Which source columns carry the canonical fields.

    ``label_map`` translates non-numeric labels, e.g. ``{"same": 1, "other": 0}``.
    Comparison is case-insensitive on the stripped string form.
    """

    sentence_1: str = "sentence_1"
    sentence_2: str = "sentence_2"
    label: str = "label"
    label_map: dict[str, int] = field(default_factory=dict)

    def translate_label(self, value: object) -> object:
        if not self.label_map:
            return value
        key = str(value).strip().lower()
        normalised = {k.strip().lower(): v for k, v in self.label_map.items()}
        if key not in normalised:
            raise DatasetValidationError(
                f"Label {value!r} is not in label_map {sorted(normalised)}."
            )
        return normalised[key]


def detect_format(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUFFIX_FORMATS:
        raise DatasetValidationError(
            f"Unsupported input extension {suffix!r}. " f"Supported: {sorted(SUFFIX_FORMATS)}"
        )
    return SUFFIX_FORMATS[suffix]


def load_frame(path: Path, source_format: str | None = None) -> pd.DataFrame:
    """Read any supported input into a DataFrame with canonical column names."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    fmt = source_format or detect_format(source)

    if fmt == "csv":
        return pd.read_csv(source, skipinitialspace=True)
    if fmt == "tsv":
        return pd.read_csv(source, sep="\t")
    if fmt == "json":
        return pd.read_json(source)
    if fmt == "jsonl":
        return pd.read_json(source, lines=True)
    if fmt == "mc1":
        # Already canonical — round-trip it so the file still gets validated.
        pairs = read_pairs(source)
        return pd.DataFrame(
            {
                "sentence_1": [pair.sentence_1 for pair in pairs],
                "sentence_2": [pair.sentence_2 for pair in pairs],
                "label": [pair.label for pair in pairs],
            }
        )

    raise DatasetValidationError(f"Unknown source format {fmt!r}.")


def frame_to_pairs(
    frame: pd.DataFrame,
    mapping: ColumnMapping | None = None,
) -> list[SentencePair]:
    """Project a DataFrame onto the canonical schema, validating every row."""
    column_mapping = mapping or ColumnMapping()

    required = (
        column_mapping.sentence_1,
        column_mapping.sentence_2,
        column_mapping.label,
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise DatasetValidationError(
            f"Input is missing column(s) {missing}. " f"Available columns: {list(frame.columns)}"
        )

    pairs: list[SentencePair] = []
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        values = dict(zip(frame.columns, row, strict=True))
        try:
            pairs.append(
                make_pair(
                    values[column_mapping.sentence_1],
                    values[column_mapping.sentence_2],
                    column_mapping.translate_label(values[column_mapping.label]),
                )
            )
        except DatasetValidationError as exc:
            raise DatasetValidationError(f"row {position}: {exc}") from exc

    return pairs


def convert_file(
    source: Path,
    mapping: ColumnMapping | None = None,
    source_format: str | None = None,
) -> list[SentencePair]:
    """Read ``source`` in any supported format and return canonical pairs."""
    return frame_to_pairs(load_frame(source, source_format), mapping)
