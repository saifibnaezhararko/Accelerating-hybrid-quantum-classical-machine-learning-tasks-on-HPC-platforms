"""Load, split, and validate the MC1 dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qnlp_hpc.mc1_iqp_cups_gpu.config import (
    DEVELOPMENT_SENTENCES,
    OUTPUT_DIR,
    TEST_SENTENCES,
)


def load_mc1(path: Path) -> pd.DataFrame:
    """Load and validate MC1 sentence pairs from ``path``."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Place MC1.txt in data/processed or update DATA_PATH."
        )

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.rsplit(",", maxsplit=2)]
            if len(parts) != 3:
                raise ValueError(
                    f"Line {line_number} must have the format "
                    f"'sentence 1, sentence 2, label': {line!r}"
                )

            sentence_1, sentence_2, label_text = parts
            if not sentence_1 or not sentence_2:
                raise ValueError(f"Line {line_number} contains an empty sentence: {line!r}")

            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} has a non-integer label: {label_text!r}"
                ) from exc

            if label not in (0, 1):
                raise ValueError(f"Line {line_number} label must be 0 or 1, got {label}.")

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

    missing_labels = {0, 1}.difference(frame["label"].unique())
    if missing_labels:
        raise ValueError(
            "MC1.txt must contain both labels 0 and 1; " f"missing {sorted(missing_labels)}."
        )

    return frame


def sentence_set(frame: pd.DataFrame) -> set[str]:
    """Return all complete sentences appearing in a pair dataframe."""
    return set(frame["sentence_1"]).union(frame["sentence_2"])


def token_vocabulary(sentences: set[str]) -> set[str]:
    """Return the whitespace-token vocabulary of a sentence collection."""
    return {token for sentence in sentences for token in sentence.split()}


def split_sentence_disjoint(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a deterministic sentence-disjoint train/dev/test split.

    A pair is retained only when both sentences belong to the same split.
    The fourth returned dataframe contains excluded cross-split pairs.
    """
    all_sentences = sentence_set(frame)

    unknown_development = DEVELOPMENT_SENTENCES.difference(all_sentences)
    unknown_test = TEST_SENTENCES.difference(all_sentences)
    if unknown_development or unknown_test:
        raise ValueError(
            "The sentence split does not match this MC1 file. "
            f"Missing development sentences={sorted(unknown_development)}, "
            f"missing test sentences={sorted(unknown_test)}."
        )

    if DEVELOPMENT_SENTENCES.intersection(TEST_SENTENCES):
        raise RuntimeError("Development and test sentence sets overlap.")

    training_sentences = all_sentences.difference(DEVELOPMENT_SENTENCES.union(TEST_SENTENCES))
    if not training_sentences:
        raise RuntimeError("The split produced no training sentences.")

    split_rows: dict[str, list[dict[str, object]]] = {
        "training": [],
        "development": [],
        "test": [],
        "dropped_cross_split": [],
    }

    for row in frame.itertuples(index=False):
        left = row.sentence_1
        right = row.sentence_2

        if left in training_sentences and right in training_sentences:
            split_name = "training"
        elif left in DEVELOPMENT_SENTENCES and right in DEVELOPMENT_SENTENCES:
            split_name = "development"
        elif left in TEST_SENTENCES and right in TEST_SENTENCES:
            split_name = "test"
        else:
            split_name = "dropped_cross_split"

        split_rows[split_name].append(
            {
                "sentence_1": left,
                "sentence_2": right,
                "label": int(row.label),
            }
        )

    training = pd.DataFrame(split_rows["training"])
    development = pd.DataFrame(split_rows["development"])
    test = pd.DataFrame(split_rows["test"])
    dropped = pd.DataFrame(split_rows["dropped_cross_split"])

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
                f"The {name} split must contain labels 0 and 1; " f"found {sorted(labels)}."
            )

    training_observed = sentence_set(training)
    development_observed = sentence_set(development)
    test_observed = sentence_set(test)

    overlaps = {
        "train_dev": training_observed.intersection(development_observed),
        "train_test": training_observed.intersection(test_observed),
        "dev_test": development_observed.intersection(test_observed),
    }
    if any(overlaps.values()):
        formatted_overlaps = {name: sorted(sentences) for name, sentences in overlaps.items()}
        raise RuntimeError(f"Sentence-disjointness check failed: {formatted_overlaps!r}")

    training_vocab = token_vocabulary(training_observed)
    held_out_vocab = token_vocabulary(development_observed.union(test_observed))
    unseen_tokens = held_out_vocab.difference(training_vocab)
    if unseen_tokens:
        raise RuntimeError(
            "Held-out splits contain tokens absent from training: " f"{sorted(unseen_tokens)}"
        )

    return (
        training.reset_index(drop=True),
        development.reset_index(drop=True),
        test.reset_index(drop=True),
        dropped.reset_index(drop=True),
    )


def save_split_diagnostics(
    full_frame: pd.DataFrame,
    training: pd.DataFrame,
    development: pd.DataFrame,
    test: pd.DataFrame,
    dropped: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Write sentence assignments, excluded pairs, and split statistics."""
    assignments: list[dict[str, str]] = []
    for split_name, split in (
        ("training", training),
        ("development", development),
        ("test", test),
    ):
        for sentence in sorted(sentence_set(split)):
            assignments.append({"sentence": sentence, "split": split_name})

    pd.DataFrame(assignments).to_csv(
        output_dir / "sentence_disjoint_manifest.csv",
        index=False,
    )
    dropped.to_csv(
        output_dir / "sentence_disjoint_dropped_pairs.csv",
        index=False,
    )

    summary_rows: list[dict[str, object]] = []
    for split_name, split in (
        ("training", training),
        ("development", development),
        ("test", test),
        ("dropped_cross_split", dropped),
    ):
        counts = split["label"].value_counts().sort_index().to_dict()
        sentences = sentence_set(split) if not split.empty else set()
        summary_rows.append(
            {
                "split": split_name,
                "pairs": len(split),
                "unique_sentences": len(sentences),
                "label_0": int(counts.get(0, 0)),
                "label_1": int(counts.get(1, 0)),
            }
        )

    summary_rows.append(
        {
            "split": "full_input",
            "pairs": len(full_frame),
            "unique_sentences": len(sentence_set(full_frame)),
            "label_0": int((full_frame["label"] == 0).sum()),
            "label_1": int((full_frame["label"] == 1).sum()),
        }
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "sentence_disjoint_split_summary.csv",
        index=False,
    )
