"""Canonical sentence-pair schema shared by every dataset in the project.

One record is ``(sentence_1, sentence_2, label)`` with ``label`` in ``{0, 1}``.

The on-disk form is the MC1 text format inherited from the original prototype::

    cook creates complicated dish, experienced chef prepares complicated dish, 1

Parsing splits on the **last two** commas (``rsplit(',', 2)``), matching
``qnlp_hpc.mc1_spider.data.load_mc1``, which is the training-side reader. This
module is the write/convert side: it produces files that reader accepts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = (0, 1)
_WHITESPACE = re.compile(r"\s+")


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be represented in the canonical format."""


@dataclass(frozen=True)
class SentencePair:
    sentence_1: str
    sentence_2: str
    label: int

    def to_line(self) -> str:
        return f"{self.sentence_1}, {self.sentence_2}, {self.label}"


def normalise_sentence(text: str) -> str:
    """Strip and collapse internal whitespace (incl. tabs and newlines)."""
    return _WHITESPACE.sub(" ", str(text)).strip()


def make_pair(sentence_1: object, sentence_2: object, label: object) -> SentencePair:
    """Build a validated :class:`SentencePair` from raw values.

    Commas are rejected in both sentences: ``sentence_2`` containing one would
    shift the ``rsplit`` boundary and silently corrupt the label, and allowing
    them only in ``sentence_1`` would make the format asymmetric.
    """
    first = normalise_sentence(sentence_1)
    second = normalise_sentence(sentence_2)

    if not first or not second:
        raise DatasetValidationError(f"Empty sentence in pair: {first!r}, {second!r}")
    for sentence in (first, second):
        if "," in sentence:
            raise DatasetValidationError(
                "Sentences must not contain commas (the MC1 format is "
                f"comma-delimited): {sentence!r}"
            )

    try:
        parsed_label = int(str(label).strip())
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"Non-integer label: {label!r}") from exc

    if parsed_label not in VALID_LABELS:
        raise DatasetValidationError(
            f"Label must be one of {list(VALID_LABELS)}, got {parsed_label}."
        )

    return SentencePair(first, second, parsed_label)


def read_pairs(path: Path) -> list[SentencePair]:
    """Read an MC1-format file, reporting the offending line on failure."""
    pairs: list[SentencePair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(",", maxsplit=2)
            if len(parts) != 3:
                raise DatasetValidationError(
                    f"{path}:{line_number} is not " f"'sentence 1, sentence 2, label': {line!r}"
                )
            try:
                pairs.append(make_pair(*parts))
            except DatasetValidationError as exc:
                raise DatasetValidationError(f"{path}:{line_number} — {exc}") from exc
    return pairs


def deduplicate(pairs: Iterable[SentencePair]) -> tuple[list[SentencePair], int]:
    """Drop exact duplicate records, preserving order.

    Returns the kept records and the number dropped. Duplicates matter here: the
    dataset is only 100 rows, so a repeated pair that lands on both sides of the
    train/test split leaks.
    """
    seen: set[SentencePair] = set()
    kept: list[SentencePair] = []
    duplicates = 0
    for pair in pairs:
        if pair in seen:
            duplicates += 1
            continue
        seen.add(pair)
        kept.append(pair)
    return kept, duplicates


def check_dataset(pairs: list[SentencePair]) -> dict[str, object]:
    """Validate a whole dataset and return a summary for the pipeline log."""
    if not pairs:
        raise DatasetValidationError("Dataset is empty.")

    counts = Counter(pair.label for pair in pairs)
    missing = [label for label in VALID_LABELS if label not in counts]
    if missing:
        raise DatasetValidationError(
            f"Dataset must contain both labels {list(VALID_LABELS)}; " f"missing {missing}."
        )

    # Stratified splitting needs at least 2 members per class; scikit-learn
    # raises a late, confusing error otherwise.
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        raise DatasetValidationError(
            f"Labels {too_small} have fewer than 2 examples; " "stratified splitting will fail."
        )

    sentences = {pair.sentence_1 for pair in pairs} | {pair.sentence_2 for pair in pairs}
    word_counts = [len(sentence.split()) for sentence in sentences]

    return {
        "records": len(pairs),
        "label_counts": dict(sorted(counts.items())),
        "unique_sentences": len(sentences),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "mean_words": round(sum(word_counts) / len(word_counts), 2),
    }


def write_pairs(pairs: list[SentencePair], path: Path) -> dict[str, object]:
    """Validate then write ``pairs`` in MC1 format; return the summary."""
    summary = check_dataset(pairs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(pair.to_line() for pair in pairs) + "\n",
        encoding="utf-8",
    )
    return summary


def iter_lines(pairs: Iterable[SentencePair]) -> Iterator[str]:
    for pair in pairs:
        yield pair.to_line()
