
import pytest

from pathlib import Path

from qnlp_hpc.mc1_spider.data import load_mc1


def test_load_mc1(tmp_path: Path) -> None:
    data_path = tmp_path / "MC1.txt"
    data_path.write_text(
        "cats chase mice, dogs chase balls, 0\n"
        "cats chase mice, kittens chase mice, 1\n",
        encoding="utf-8",
    )

    frame = load_mc1(data_path)

    assert len(frame) == 2
    assert list(frame.columns) == ["sentence_1", "sentence_2", "label"]
    assert frame["label"].tolist() == [0, 1]

def test_load_mc1_rejects_invalid_label(tmp_path: Path) -> None:
    data_path = tmp_path / "MC1.txt"
    data_path.write_text(
        "sentence one, sentence two, 7\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label must be 0 or 1"):
        load_mc1(data_path)
