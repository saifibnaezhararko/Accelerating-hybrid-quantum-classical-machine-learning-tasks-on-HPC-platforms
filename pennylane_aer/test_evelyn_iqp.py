from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qnlp_hpc.mc1_iqp_cups import circuits, config, data, model

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def write_mc1(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "MC1.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_mc1_parses_pairs_and_labels(tmp_path: Path) -> None:
    path = write_mc1(
        tmp_path,
        [
            "cook prepares meal, chef creates dish, 1",
            "programmer writes code, hacker writes code, 1",
            "cook prepares meal, programmer writes code, 0",
        ],
    )
    frame = data.load_mc1(path)

    assert list(frame.columns) == ["sentence_1", "sentence_2", "label"]
    assert len(frame) == 3
    assert frame.loc[0, "sentence_1"] == "cook prepares meal"
    assert frame.loc[0, "sentence_2"] == "chef creates dish"
    assert sorted(frame["label"].unique()) == [0, 1]


def test_load_mc1_skips_blank_lines(tmp_path: Path) -> None:
    path = write_mc1(
        tmp_path,
        [
            "cook prepares meal, chef creates dish, 1",
            "",
            "   ",
            "cook prepares meal, programmer writes code, 0",
        ],
    )
    assert len(data.load_mc1(path)) == 2


def test_load_mc1_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        data.load_mc1(tmp_path / "absent.txt")


def test_load_mc1_rejects_non_integer_label(tmp_path: Path) -> None:
    path = write_mc1(tmp_path, ["cook prepares meal, chef creates dish, yes"])
    with pytest.raises(ValueError, match="non-integer label"):
        data.load_mc1(path)


def test_load_mc1_rejects_out_of_range_label(tmp_path: Path) -> None:
    path = write_mc1(tmp_path, ["cook prepares meal, chef creates dish, 2"])
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        data.load_mc1(path)


def test_load_mc1_rejects_empty_sentence(tmp_path: Path) -> None:
    path = write_mc1(tmp_path, [" , chef creates dish, 1"])
    with pytest.raises(ValueError, match="empty sentence"):
        data.load_mc1(path)


def test_load_mc1_requires_both_classes(tmp_path: Path) -> None:
    path = write_mc1(
        tmp_path,
        [
            "cook prepares meal, chef creates dish, 1",
            "programmer writes code, hacker writes code, 1",
        ],
    )
    with pytest.raises(ValueError, match="must contain both labels"):
        data.load_mc1(path)


def test_load_mc1_comma_inside_sentence_shifts_the_delimiter(tmp_path: Path) -> None:
    path = write_mc1(
        tmp_path,
        [
            "cook prepares meal, and dish, chef creates dish, 1",
            "cook prepares meal, programmer writes code, 0",
        ],
    )
    frame = data.load_mc1(path)
    assert frame.loc[0, "sentence_1"] == "cook prepares meal, and dish"
    assert frame.loc[0, "sentence_2"] == "chef creates dish"


@pytest.fixture(scope="module")
def mc1_frame() -> pd.DataFrame:
    if not config.DATA_PATH.is_file():
        pytest.skip(f"MC1 dataset not present at {config.DATA_PATH}")
    return data.load_mc1(config.DATA_PATH)


@pytest.fixture(scope="module")
def splits(mc1_frame: pd.DataFrame):
    return data.split_sentence_disjoint(mc1_frame)


def test_split_conserves_every_pair(mc1_frame: pd.DataFrame, splits) -> None:
    train, development, test, dropped = splits
    assert len(train) + len(development) + len(test) + len(dropped) == len(mc1_frame)


def test_split_sentences_are_disjoint(splits) -> None:
    train, development, test, _ = splits
    train_sentences = data.sentence_set(train)
    development_sentences = data.sentence_set(development)
    test_sentences = data.sentence_set(test)

    assert not train_sentences & development_sentences
    assert not train_sentences & test_sentences
    assert not development_sentences & test_sentences


def test_held_out_vocabulary_is_covered_by_training(splits) -> None:
    train, development, test, _ = splits
    train_vocabulary = data.token_vocabulary(data.sentence_set(train))
    held_out = data.token_vocabulary(data.sentence_set(development) | data.sentence_set(test))
    assert not held_out - train_vocabulary


def test_every_split_carries_both_labels(splits) -> None:
    train, development, test, _ = splits
    for split in (train, development, test):
        assert set(split["label"]) == {0, 1}


def test_split_rejects_a_frame_missing_the_declared_sentences() -> None:
    frame = pd.DataFrame(
        [
            {"sentence_1": "cook prepares meal", "sentence_2": "chef creates dish", "label": 1},
            {"sentence_1": "cook prepares meal", "sentence_2": "hacker writes code", "label": 0},
        ]
    )
    with pytest.raises(ValueError, match="does not match this MC1 file"):
        data.split_sentence_disjoint(frame)


@pytest.fixture(scope="module")
def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sentence_1": "cook prepares meal", "sentence_2": "chef creates dish", "label": 1},
            {
                "sentence_1": "programmer writes code",
                "sentence_2": "hacker writes code",
                "label": 1,
            },
            {"sentence_1": "cook prepares meal", "sentence_2": "hacker writes code", "label": 0},
        ]
    )


@pytest.fixture(scope="module")
def circuit_lookup(small_frame: pd.DataFrame) -> dict[str, object]:
    return circuits.build_quantum_circuits(small_frame)


def test_build_quantum_circuits_covers_every_unique_sentence(
    small_frame: pd.DataFrame, circuit_lookup: dict[str, object]
) -> None:
    expected = set(small_frame["sentence_1"]) | set(small_frame["sentence_2"])
    assert set(circuit_lookup) == expected


def test_circuits_carry_trainable_symbols(circuit_lookup: dict[str, object]) -> None:
    symbols = circuits.collect_circuit_symbols(list(circuit_lookup.values()))
    assert symbols, "IQPAnsatz produced no free symbols - nothing would train"


def test_circuits_use_the_configured_qubit_count(circuit_lookup: dict[str, object]) -> None:
    for circuit in circuit_lookup.values():
        assert len(circuit.cod) == config.SENTENCE_QUBITS


def test_make_pairs_and_targets_preserves_row_order(
    small_frame: pd.DataFrame, circuit_lookup: dict[str, object]
) -> None:
    pairs, targets = circuits.make_pairs_and_targets(small_frame, circuit_lookup)

    assert len(pairs) == len(small_frame)
    assert targets.tolist() == small_frame["label"].tolist()
    for pair, row in zip(pairs, small_frame.itertuples(index=False), strict=True):
        assert pair[0] is circuit_lookup[row.sentence_1]
        assert pair[1] is circuit_lookup[row.sentence_2]


@pytest.fixture(scope="module")
def trained_pair_model(circuit_lookup: dict[str, object]):
    flat = [circuit for circuit in circuit_lookup.values()]
    pair_model = model.IQPPairModel.from_diagrams(flat)
    pair_model.initialise_weights()
    return pair_model


def test_forward_returns_two_logits_per_pair(
    trained_pair_model, small_frame: pd.DataFrame, circuit_lookup: dict[str, object]
) -> None:
    pairs, _ = circuits.make_pairs_and_targets(small_frame, circuit_lookup)
    logits = trained_pair_model(pairs)
    assert logits.shape == (len(pairs), 2)


def test_forward_rejects_an_empty_batch(trained_pair_model) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        trained_pair_model([])


def test_pair_model_is_symmetric(trained_pair_model, circuit_lookup: dict[str, object]) -> None:
    left = circuit_lookup["cook prepares meal"]
    right = circuit_lookup["hacker writes code"]

    trained_pair_model.eval()
    with torch.no_grad():
        forward = trained_pair_model([(left, right)])
        reversed_order = trained_pair_model([(right, left)])

    torch.testing.assert_close(forward, reversed_order)


def test_identical_sentences_give_a_zero_difference_feature(
    trained_pair_model, circuit_lookup: dict[str, object]
) -> None:
    circuit = circuit_lookup["cook prepares meal"]
    trained_pair_model.eval()
    with torch.no_grad():
        first = trained_pair_model([(circuit, circuit)])
        second = trained_pair_model([(circuit, circuit)])
    torch.testing.assert_close(first, second)


def test_gradients_reach_the_lambeq_circuit_symbols(
    circuit_lookup: dict[str, object], small_frame: pd.DataFrame
) -> None:
    flat = list(circuit_lookup.values())
    pair_model = model.IQPPairModel.from_diagrams(flat)
    pair_model.initialise_weights()
    pair_model.train()

    pairs, targets = circuits.make_pairs_and_targets(small_frame, circuit_lookup)
    logits = pair_model(pairs)
    loss = model.LambeqCrossEntropyLoss()(logits, torch.tensor(targets))
    loss.backward()

    weights = pair_model.weights
    assert weights.grad is not None, "no gradient on the lambeq symbol weights"
    assert float(weights.grad.abs().sum()) > 0.0, "circuit parameters received zero gradient"


def test_loss_accepts_class_ids_and_one_hot() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7]])
    loss_function = model.LambeqCrossEntropyLoss()

    class_id_loss = loss_function(logits, torch.tensor([0, 1]))
    one_hot_loss = loss_function(logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    torch.testing.assert_close(class_id_loss, one_hot_loss)


def test_accuracy_matches_hand_computed_value() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7], [3.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    assert model.accuracy(logits, targets) == pytest.approx(2 / 3)


def test_class_ids_handles_both_target_encodings() -> None:
    assert model.class_ids(torch.tensor([0, 1, 1])).tolist() == [0, 1, 1]
    one_hot = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert model.class_ids(one_hot).tolist() == [0, 1, 1]
