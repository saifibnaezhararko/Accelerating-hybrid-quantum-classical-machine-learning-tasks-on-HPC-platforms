import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from qnlp_hpc.mc1_spider.config import (
    SEED,
    TEST_SIZE,
    VALIDATION_SIZE_WITHIN_TRAIN,
)

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mc1(path: Path) -> pd.DataFrame:
    """Load MC1 sentence pairs from ``path``.

    ``rsplit(',', maxsplit=2)`` treats the final two commas as separators.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Place MC1.txt in the directory from which you run this script."
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
                raise ValueError(
                    f"Line {line_number} contains an empty sentence: {line!r}"
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

    class_counts = frame["label"].value_counts()
    missing_labels = {0, 1}.difference(class_counts.index)
    if missing_labels:
        raise ValueError(
            "MC1.txt must contain both labels 0 and 1; "
            f"missing {sorted(missing_labels)}."
        )

    return frame


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/development/test splits."""
    train_val, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        stratify=frame["label"],
        random_state=SEED,
        shuffle=True,
    )
    train, development = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        stratify=train_val["label"],
        random_state=SEED,
        shuffle=True,
    )
    return train, development, test