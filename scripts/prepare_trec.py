"""Filter the TREC question dataset down to a QNLP-sized subset.

Dataset-level dimension reduction for the lambeq pipeline: circuit width
scales with sentence length and parameter count with vocabulary size, so
the raw TREC set (5,452 questions, 8,677-word vocabulary, up to 37 words
per question) is reduced with length, class, and word-frequency filters.

By default the raw train and test files are pooled, filtered together,
and re-split (stratified, seeded). TREC's official split is unusable
after filtering: almost no official test question is fully covered by
the filtered train vocabulary, and a model cannot evaluate words it has
no parameters for. ``--keep-original-split`` preserves the official
split instead, restricting test to train-vocabulary sentences (expect a
tiny test set).

Usage (defaults reproduce the committed modified_trec_dataset/):
    python scripts/prepare_trec.py
    python scripts/prepare_trec.py --max-words 6 --classes 0,3,5 --no-balance
"""

from __future__ import annotations

import argparse
import collections
import json
import string
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMN = "label-coarse"
COARSE_LABEL_NAMES = {0: "DESC", 1: "ENTY", 2: "ABBR", 3: "HUM", 4: "NUM", 5: "LOC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train",
        type=Path,
        default=REPO_ROOT / "trec dataset" / "train.csv",
        help="Path to the raw TREC train CSV.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=REPO_ROOT / "trec dataset" / "test.csv",
        help="Path to the raw TREC test CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "modified_trec_dataset",
        help="Directory for the filtered train.csv / test.csv / summary.json.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=8,
        help="Keep sentences with at most this many words (after cleaning).",
    )
    parser.add_argument(
        "--classes",
        default="0,3",
        help="Comma-separated coarse labels to keep (0=DESC 1=ENTY 2=ABBR " "3=HUM 4=NUM 5=LOC).",
    )
    parser.add_argument(
        "--min-word-freq",
        type=int,
        default=2,
        help="Drop sentences containing any word occurring fewer than this "
        "many times in the filtered pool (single pass).",
    )
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample every class to the smallest class size.",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase all tokens (merges 'What'/'what' into one symbol).",
    )
    parser.add_argument(
        "--strip-punct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop punctuation-only tokens such as the trailing '?'.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Test fraction for the stratified re-split (pooled mode only).",
    )
    parser.add_argument(
        "--keep-original-split",
        action="store_true",
        help="Keep TREC's official train/test split instead of pooling and "
        "re-splitting. The test set shrinks drastically: only sentences "
        "fully covered by the filtered train vocabulary survive.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for balancing and re-splitting.")
    return parser.parse_args()


def clean_text(text: str, lowercase: bool, strip_punct: bool) -> str:
    tokens = text.split()
    if strip_punct:
        tokens = [t for t in tokens if not all(c in string.punctuation for c in t)]
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return " ".join(tokens)


def base_filter(frame: pd.DataFrame, args: argparse.Namespace, classes: list[int]) -> pd.DataFrame:
    """Cleaning, length, class, and duplicate filters."""
    frame = frame.copy()
    frame["text"] = frame["text"].apply(lambda s: clean_text(s, args.lowercase, args.strip_punct))
    frame = frame[frame["text"].str.split().str.len().between(1, args.max_words)]
    frame = frame[frame[LABEL_COLUMN].isin(classes)]
    return frame.drop_duplicates(subset="text")


def apply_min_word_freq(frame: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Drop sentences containing words rarer than ``min_freq``.

    Single pass on purpose: word counts are taken once, on the incoming
    frame. Recounting after each removal (a fixed point) cascades to an
    empty set on a corpus this small and lexically diverse.
    """
    counts = collections.Counter(w for s in frame["text"] for w in s.split())
    return frame[frame["text"].apply(lambda s: all(counts[w] >= min_freq for w in s.split()))]


def balance_classes(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    smallest = frame[LABEL_COLUMN].value_counts().min()
    parts = [
        group.sample(n=smallest, random_state=seed) for _, group in frame.groupby(LABEL_COLUMN)
    ]
    return pd.concat(parts).sort_index()


def stratified_split(
    frame: pd.DataFrame, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified, vocabulary-aware test draw.

    A shuffled candidate goes to test only if every one of its words
    still occurs in the rows left for train, so the test set has no
    out-of-vocabulary words by construction (a naive draw loses ~half
    the test rows to OOV afterwards).
    """
    counts = collections.Counter(w for s in frame["text"] for w in s.split())
    test_indices: list[int] = []
    for _, group in frame.groupby(LABEL_COLUMN):
        quota = round(len(group) * test_frac)
        taken = 0
        shuffled = group.sample(frac=1, random_state=seed)
        for idx, text in shuffled["text"].items():
            if taken >= quota:
                break
            word_counts = collections.Counter(text.split())
            if all(counts[w] > c for w, c in word_counts.items()):
                counts.subtract(word_counts)
                test_indices.append(idx)
                taken += 1
    test = frame.loc[sorted(test_indices)]
    train = frame.drop(test_indices)
    return train, test


def vocabulary(frame: pd.DataFrame) -> set[str]:
    return {w for s in frame["text"] for w in s.split()}


def covered_by(frame: pd.DataFrame, vocab: set[str]) -> pd.Series:
    return frame["text"].apply(lambda s: all(w in vocab for w in s.split()))


def label_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
    return {
        f"{label} ({COARSE_LABEL_NAMES.get(label, '?')})": int(count)
        for label, count in counts.items()
    }


def split_stats(frame: pd.DataFrame) -> dict[str, object]:
    lengths = frame["text"].str.split().str.len()
    return {
        "sentences": int(len(frame)),
        "labels": label_distribution(frame),
        "vocab_size": len(vocabulary(frame)),
        "words_min": int(lengths.min()),
        "words_max": int(lengths.max()),
    }


def main() -> None:
    args = parse_args()
    classes = sorted(int(c) for c in args.classes.split(","))
    label_map = {original: new for new, original in enumerate(classes)}

    train_raw = pd.read_csv(args.train).assign(trec_split="train")
    test_raw = pd.read_csv(args.test).assign(trec_split="test")

    if args.keep_original_split:
        train = base_filter(train_raw, args, classes)
        train = apply_min_word_freq(train, args.min_word_freq)
        if args.balance and len(train):
            train = balance_classes(train, args.seed)
        test = base_filter(test_raw, args, classes)
        test = test[~test["text"].isin(set(train["text"]))]  # leakage guard
        test = test[covered_by(test, vocabulary(train))]
    else:
        pool = pd.concat([train_raw, test_raw], ignore_index=True)
        pool = base_filter(pool, args, classes)
        pool = apply_min_word_freq(pool, args.min_word_freq)
        if args.balance and len(pool):
            pool = balance_classes(pool, args.seed)
        train, test = stratified_split(pool, args.test_frac, args.seed)

    if train.empty or train[LABEL_COLUMN].nunique() < len(classes):
        raise SystemExit(
            "Filters left a class empty — relax --max-words, " "--min-word-freq, or --classes."
        )

    # Contiguous class IDs for training code (e.g. CrossEntropyLoss).
    train["label"] = train[LABEL_COLUMN].map(label_map)
    test["label"] = test[LABEL_COLUMN].map(label_map)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)

    oov_words = vocabulary(test) - vocabulary(train)
    summary = {
        "filters": {
            "max_words": args.max_words,
            "classes": {str(c): COARSE_LABEL_NAMES.get(c, "?") for c in classes},
            "min_word_freq": args.min_word_freq,
            "balance": args.balance,
            "lowercase": args.lowercase,
            "strip_punct": args.strip_punct,
            "split": (
                "original-trec"
                if args.keep_original_split
                else f"pooled-stratified (test_frac={args.test_frac})"
            ),
            "seed": args.seed,
        },
        "label_column_map": {str(c): label_map[c] for c in classes},
        "train": split_stats(train),
        "test": split_stats(test),
        "test_words_missing_from_train": sorted(oov_words),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_dir / 'train.csv'}")
    print(f"Wrote {args.out_dir / 'test.csv'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
