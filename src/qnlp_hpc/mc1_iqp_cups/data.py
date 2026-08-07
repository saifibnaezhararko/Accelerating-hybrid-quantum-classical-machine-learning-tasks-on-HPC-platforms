"""Load, split, and validate the TREC sentence-pair dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from qnlp_hpc.mc1_iqp_cups_gpu.config import (
    DEVELOPMENT_RATIO,
    OUTPUT_DIR,
    SEED,
    TEST_RATIO,
)


def load_pairs(path: Path) -> pd.DataFrame:
    """Load and validate TREC sentence pairs."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Place trec_pairs_1000.txt in data/processed "
            "or update DATA_PATH in config.py."
        )

    rows: list[dict[str, object]] = []

    # csv.reader is important because TREC questions may contain commas.
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)

        for line_number, parts in enumerate(reader, start=1):

            if not parts:
                continue

            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_number} must contain exactly 3 columns: "
                    f"sentence_1, sentence_2, label. Got: {parts!r}"
                )

            sentence_1 = parts[0].strip()
            sentence_2 = parts[1].strip()
            label_text = parts[2].strip()

            if not sentence_1 or not sentence_2:
                raise ValueError(
                    f"Line {line_number} contains an empty sentence."
                )

            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} has a non-integer label: "
                    f"{label_text!r}"
                ) from exc

            if label not in (0, 1):
                raise ValueError(
                    f"Line {line_number} label must be 0 or 1, got {label}."
                )

            rows.append(
                {
                    "sentence_1": sentence_1,
                    "sentence_2": sentence_2,
                    "label": label,
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError(f"{path} contains no usable examples.")

    labels = set(frame["label"].astype(int))

    if labels != {0, 1}:
        raise ValueError(
            f"Dataset must contain labels 0 and 1; found {sorted(labels)}."
        )

    return frame


def sentence_set(frame: pd.DataFrame) -> set[str]:
    """Return all sentences appearing in a pair dataframe."""

    return set(frame["sentence_1"]).union(frame["sentence_2"])


def token_vocabulary(sentences: set[str]) -> set[str]:
    """Return whitespace-token vocabulary."""

    return {
        token
        for sentence in sentences
        for token in sentence.split()
    }


def split_stratified(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic stratified train/dev/test splits.

    For 1000 balanced pairs:

        train       = 810
        development = 90
        test        = 100

    Test is 10% of the full dataset.
    Development is 10% of the remaining 90%.
    """

    # --------------------------------------------------
    # First split:
    # 90% train+development
    # 10% test
    # --------------------------------------------------

    train_dev, test = train_test_split(
        frame,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=frame["label"],
        shuffle=True,
    )

    # --------------------------------------------------
    # Second split:
    # 90% training
    # 10% development
    # from the remaining train_dev dataset
    # --------------------------------------------------

    training, development = train_test_split(
        train_dev,
        test_size=DEVELOPMENT_RATIO,
        random_state=SEED,
        stratify=train_dev["label"],
        shuffle=True,
    )

    # --------------------------------------------------
    # Validate labels
    # --------------------------------------------------

    for name, split in (
        ("training", training),
        ("development", development),
        ("test", test),
    ):

        if split.empty:
            raise RuntimeError(f"The {name} split is empty.")

        labels = set(split["label"].astype(int))

        if labels != {0, 1}:
            raise RuntimeError(
                f"The {name} split must contain labels 0 and 1; "
                f"found {sorted(labels)}."
            )

    return (
        training.reset_index(drop=True),
        development.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_split_diagnostics(
    full_frame: pd.DataFrame,
    training: pd.DataFrame,
    development: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Write split datasets and statistics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save actual splits
    training.to_csv(
        output_dir / "training_split.csv",
        index=False,
    )

    development.to_csv(
        output_dir / "development_split.csv",
        index=False,
    )

    test.to_csv(
        output_dir / "test_split.csv",
        index=False,
    )

    # Save statistics
    summary_rows: list[dict[str, object]] = []

    for split_name, split in (
        ("training", training),
        ("development", development),
        ("test", test),
        ("full_input", full_frame),
    ):

        counts = (
            split["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        summary_rows.append(
            {
                "split": split_name,
                "pairs": len(split),
                "unique_sentences": len(sentence_set(split)),
                "label_0": int(counts.get(0, 0)),
                "label_1": int(counts.get(1, 0)),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )