"""Tests for the dataset acquisition/conversion pipeline (Week 2, Software Lead)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qnlp_hpc.data.convert import ColumnMapping, convert_file, detect_format
from qnlp_hpc.data.schema import (
    DatasetValidationError,
    check_dataset,
    deduplicate,
    make_pair,
    read_pairs,
    write_pairs,
)
from qnlp_hpc.paths import PROCESSED_DIR, find_repo_root


def test_make_pair_normalises_whitespace() -> None:
    pair = make_pair("  cook   creates\tdish ", "chef prepares meal\n", " 1 ")
    assert pair.sentence_1 == "cook creates dish"
    assert pair.sentence_2 == "chef prepares meal"
    assert pair.label == 1


def test_make_pair_rejects_comma_in_sentence() -> None:
    # A comma in sentence_2 would move the rsplit boundary and eat the label.
    with pytest.raises(DatasetValidationError, match="must not contain commas"):
        make_pair("cook creates dish", "chef prepares meal, quickly", 1)


@pytest.mark.parametrize("label", ["2", "-1", "yes", ""])
def test_make_pair_rejects_bad_label(label: str) -> None:
    with pytest.raises(DatasetValidationError):
        make_pair("cook creates dish", "chef prepares meal", label)


def test_make_pair_rejects_empty_sentence() -> None:
    with pytest.raises(DatasetValidationError, match="Empty sentence"):
        make_pair("   ", "chef prepares meal", 1)


def _sample_pairs() -> list:
    """Two examples per class — the minimum a stratified split can work with."""
    return [
        make_pair("cook creates dish", "chef prepares meal", 1),
        make_pair("hacker writes code", "programmer writes code", 1),
        make_pair("hacker writes code", "chef prepares meal", 0),
        make_pair("cook bakes bread", "programmer creates code", 0),
    ]


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    pairs = _sample_pairs()
    destination = tmp_path / "out.txt"
    summary = write_pairs(pairs, destination)

    assert summary["records"] == 4
    assert summary["label_counts"] == {0: 2, 1: 2}
    assert read_pairs(destination) == pairs


def test_written_file_matches_the_trainer_side_reader(tmp_path: Path) -> None:
    """The file we emit must parse under mc1_spider's own loader."""
    from qnlp_hpc.mc1_spider.data import load_mc1

    pairs = _sample_pairs()
    destination = tmp_path / "out.txt"
    write_pairs(pairs, destination)

    frame = load_mc1(destination)
    assert list(frame.columns) == ["sentence_1", "sentence_2", "label"]
    assert frame["sentence_1"].tolist() == [pair.sentence_1 for pair in pairs]
    assert frame["label"].tolist() == [pair.label for pair in pairs]


def test_deduplicate_drops_exact_repeats() -> None:
    pair = make_pair("cook creates dish", "chef prepares meal", 1)
    other = make_pair("hacker writes code", "chef prepares meal", 0)
    kept, dropped = deduplicate([pair, other, pair])

    assert kept == [pair, other]
    assert dropped == 1


def test_check_dataset_requires_both_labels() -> None:
    with pytest.raises(DatasetValidationError, match="missing"):
        check_dataset([make_pair("cook creates dish", "chef prepares meal", 1)] * 2)


def test_check_dataset_rejects_class_too_small_to_stratify() -> None:
    pairs = [
        make_pair("cook creates dish", "chef prepares meal", 1),
        make_pair("cook bakes bread", "chef prepares soup", 1),
        make_pair("hacker writes code", "chef prepares meal", 0),
    ]
    with pytest.raises(DatasetValidationError, match="fewer than 2"):
        check_dataset(pairs)


def test_convert_csv_with_column_mapping(tmp_path: Path) -> None:
    source = tmp_path / "pairs.csv"
    source.write_text(
        "first,second,same_topic\n"
        "cook creates dish,chef prepares meal,same\n"
        "hacker writes code,chef prepares meal,other\n",
        encoding="utf-8",
    )
    mapping = ColumnMapping(
        sentence_1="first",
        sentence_2="second",
        label="same_topic",
        label_map={"same": 1, "other": 0},
    )

    pairs = convert_file(source, mapping)

    assert [pair.label for pair in pairs] == [1, 0]
    assert pairs[0].sentence_1 == "cook creates dish"


def test_convert_reports_the_offending_row(tmp_path: Path) -> None:
    source = tmp_path / "pairs.csv"
    source.write_text(
        "sentence_1,sentence_2,label\n"
        "cook creates dish,chef prepares meal,1\n"
        "hacker writes code,chef prepares meal,7\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="row 2"):
        convert_file(source)


def test_convert_rejects_unknown_label_value(tmp_path: Path) -> None:
    source = tmp_path / "pairs.csv"
    source.write_text(
        "sentence_1,sentence_2,label\ncook creates dish,chef prepares meal,maybe\n",
        encoding="utf-8",
    )
    mapping = ColumnMapping(label_map={"same": 1, "other": 0})
    with pytest.raises(DatasetValidationError, match="label_map"):
        convert_file(source, mapping)


def test_convert_missing_column_lists_what_is_available(tmp_path: Path) -> None:
    source = tmp_path / "pairs.csv"
    source.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Available columns"):
        convert_file(source)


def test_convert_tsv(tmp_path: Path) -> None:
    source = tmp_path / "pairs.tsv"
    source.write_text(
        "sentence_1\tsentence_2\tlabel\n"
        "cook creates dish\tchef prepares meal\t1\n"
        "hacker writes code\tchef prepares meal\t0\n",
        encoding="utf-8",
    )
    assert len(convert_file(source)) == 2


def test_convert_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    records = [
        {"sentence_1": "cook creates dish", "sentence_2": "chef prepares meal", "label": 1},
        {"sentence_1": "hacker writes code", "sentence_2": "chef prepares meal", "label": 0},
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    assert [pair.label for pair in convert_file(source)] == [1, 0]


def test_detect_format_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Unsupported input extension"):
        detect_format(tmp_path / "pairs.parquet")


def test_repo_root_is_found_from_the_package() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()


def test_shipped_mc1_dataset_is_valid() -> None:
    """Guards the tracked dataset against a bad edit or a stray comma."""
    summary = check_dataset(read_pairs(PROCESSED_DIR / "MC1.txt"))
    assert summary["records"] == 100
    assert summary["label_counts"] == {0: 47, 1: 53}
