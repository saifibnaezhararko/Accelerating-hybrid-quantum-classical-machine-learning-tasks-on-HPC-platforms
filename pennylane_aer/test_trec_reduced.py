"""Tests for the TREC dimensionality-reduction route.

Run from the repo root:  pytest pennylane_aer/test_trec_reduced.py

These target the claims the pipeline makes about itself - the official split is
preserved, labels stay contiguous, the encoder never sees held-out questions,
the stratified development split keeps the rare classes, and the loss actually
moves the circuit - rather than restating the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_ROOT / "src"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from trec_reduced import (  # noqa: E402
    COARSE_CLASS_NAMES,
    ClassicalSentenceClassifier,
    NoBottleneckSentenceClassifier,
    QuantumSentenceClassifier,
    SentenceAngleEncoder,
    build_split,
    class_names,
    confusion_matrix,
    evaluate_split,
    load_trec,
    macro_f1,
    mean_confidence_interval,
    stratified_split,
    train_model,
)

DATA_DIR = _ROOT / "trec dataset"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").is_file(), reason="raw TREC dataset not present"
)


@pytest.fixture(scope="module")
def frames():
    return load_trec(DATA_DIR)


@pytest.fixture(scope="module")
def train_frame(frames):
    return frames[0]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_official_split_sizes_are_preserved(frames):
    """TREC ships 5452/500; published numbers are quoted against that split."""
    train, test = frames
    assert len(train) == 5452
    assert len(test) == 500


def test_coarse_labels_are_contiguous_and_complete(frames):
    train, test = frames
    assert sorted(train["label"].unique()) == list(range(6))
    assert set(test["label"]).issubset(set(train["label"]))


def test_class_filter_remaps_labels_to_a_contiguous_range():
    """Cross-entropy needs 0..k-1; a class filter leaves gaps unless remapped."""
    train, test = load_trec(DATA_DIR, classes=[0, 3])
    assert sorted(train["label"].unique()) == [0, 1]
    assert sorted(test["label"].unique()) == [0, 1]
    assert train.attrs["label_remap"] == {0: 0, 3: 1}


def test_max_words_filter_bounds_question_length():
    train, _ = load_trec(DATA_DIR, max_words=8)
    assert train["text"].str.split().str.len().max() <= 8
    assert len(train) < 5452


def test_missing_directory_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_trec(tmp_path)


def test_unknown_label_column_is_rejected():
    with pytest.raises(ValueError, match="label_column"):
        load_trec(DATA_DIR, label_column="nonsense")


def test_fine_labels_have_more_classes_than_coarse(frames):
    fine_train, _ = load_trec(DATA_DIR, label_column="label-fine")
    assert fine_train["label"].nunique() > frames[0]["label"].nunique()


def test_class_names_follow_remapped_label_order():
    assert class_names({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, "label-coarse") == list(
        COARSE_CLASS_NAMES
    )
    assert class_names({0: 0, 3: 1}, "label-coarse") == ["DESC", "HUM"]


def test_class_names_match_the_question_content(train_frame):
    """Pin the label order against the data, not against a remembered mapping.

    This file's labels are not the ABBR-first order some TREC distributions
    use, and a wrong constant mislabels every class in the report while every
    accuracy stays correct - silent and easy to miss.  Each probe is a phrase
    that only its own category would contain.
    """
    probes = {
        "DESC": "what are liver enzymes",
        "ABBR": "stand for",
        "NUM": "when was",
        "LOC": "waterfall",
    }
    lowered = train_frame["text"].str.lower()
    for name, phrase in probes.items():
        labels = train_frame.loc[lowered.str.contains(phrase, regex=False), "label"]
        assert len(labels) > 0, phrase
        dominant = int(labels.mode().iloc[0])
        assert COARSE_CLASS_NAMES[dominant] == name, (
            f"questions containing {phrase!r} are mostly label {dominant}, "
            f"which this file names {COARSE_CLASS_NAMES[dominant]!r}, not {name!r}"
        )


def test_the_rare_class_is_abbr(train_frame):
    """ABBR is 86 of 5,452 rows; the stratified split exists because of it."""
    counts = train_frame["label"].value_counts()
    assert COARSE_CLASS_NAMES[int(counts.idxmin())] == "ABBR"


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def test_stratified_split_preserves_class_proportions(train_frame):
    kept, held_out = stratified_split(train_frame, 0.15, seed=0)
    assert len(kept) + len(held_out) == len(train_frame)
    for label in sorted(train_frame["label"].unique()):
        original = (train_frame["label"] == label).mean()
        observed = (held_out["label"] == label).mean()
        assert abs(original - observed) < 0.03, label


def test_stratified_split_keeps_every_class_in_development(train_frame):
    """TREC's ABBR class is 86 of 5,452 rows; a uniform draw can miss it."""
    _, held_out = stratified_split(train_frame, 0.15, seed=0)
    assert set(held_out["label"]) == set(train_frame["label"])


def test_stratified_split_never_empties_a_class_out_of_training():
    tiny = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 0, 1]})
    kept, held_out = stratified_split(tiny, 0.9, seed=0)
    assert set(kept["label"]) == {0, 1}
    assert len(held_out) >= 1


def test_stratified_split_is_deterministic_per_seed(train_frame):
    first = stratified_split(train_frame, 0.15, seed=3)[1]["text"].tolist()
    second = stratified_split(train_frame, 0.15, seed=3)[1]["text"].tolist()
    assert first == second
    other = stratified_split(train_frame, 0.15, seed=4)[1]["text"].tolist()
    assert first != other


def test_invalid_fraction_is_rejected(train_frame):
    with pytest.raises(ValueError, match="fraction"):
        stratified_split(train_frame, 1.5, seed=0)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_corpus(train_frame):
    return train_frame["text"].astype(str).tolist()[:400]


def test_encoder_is_fitted_on_training_questions_only(small_corpus, frames):
    """A held-out question must not change the feature map it is scored under."""
    encoder = SentenceAngleEncoder(n_qubits=4).fit(small_corpus)
    held_out = frames[1]["text"].astype(str).tolist()[:50]

    first = encoder.transform(held_out)
    encoder.transform(frames[1]["text"].astype(str).tolist()[50:100])
    second = encoder.transform(held_out)

    np.testing.assert_allclose(first, second)


def test_angles_stay_inside_the_bloch_range(small_corpus, frames):
    """Wrapping past +-pi would alias distinct questions onto the same angle."""
    held_out = frames[1]["text"].astype(str).tolist()[:100]
    for scaling in ("global", "per-component"):
        encoder = SentenceAngleEncoder(n_qubits=4, scaling=scaling).fit(small_corpus)
        assert np.all(np.abs(encoder.transform(held_out)) < np.pi)


@pytest.mark.parametrize("reducer", ["tsvd", "pca"])
def test_per_component_scaling_flattens_the_component_spreads(small_corpus, reducer):
    """The mechanism behind the scaling ablation, stated as a measurement.

    `per-component` divides each component by its own standard deviation, so
    the angles arrive at the circuit with near-equal spread and a low-variance
    direction is presented as loudly as the leading one.  `global` keeps them
    unequal.  Flatness is min/max spread: 1.0 means fully equalised.
    """
    spreads = {
        scaling: SentenceAngleEncoder(n_qubits=4, reducer=reducer, scaling=scaling)
        .fit_transform(small_corpus)
        .std(axis=0)
        for scaling in ("global", "per-component")
    }
    flatness = {name: float(s.min() / s.max()) for name, s in spreads.items()}
    assert flatness["per-component"] > 0.9
    assert flatness["global"] < flatness["per-component"]


def test_global_scaling_keeps_a_centred_basis_ordered_by_variance(small_corpus):
    """PCA centres before decomposing, so component 0 carries the most variance.

    TruncatedSVD does not centre: on TF-IDF its leading direction is the
    average document, which loses most of its spread once the reduced features
    are centred.  That is why this invariant is asserted against PCA, and why
    the ordering claim must not be assumed for the tsvd default.
    """
    spread = (
        SentenceAngleEncoder(n_qubits=4, reducer="pca", scaling="global")
        .fit_transform(small_corpus)
        .std(axis=0)
    )
    assert int(np.argmax(spread)) == 0
    assert spread[0] > spread[-1]


def test_tsvd_consumes_sparse_input_without_densifying(small_corpus, monkeypatch):
    """TREC's TF-IDF matrix is 5,452 x 8,460; densifying it wastes ~370 MB."""
    encoder = SentenceAngleEncoder(n_qubits=4, reducer="tsvd")

    def explode(matrix):
        raise AssertionError("sparse matrix was densified for TruncatedSVD")

    monkeypatch.setattr(SentenceAngleEncoder, "_densify", staticmethod(explode))
    encoder.fit(small_corpus)
    assert encoder.transform(small_corpus[:10]).shape == (10, 4)


@pytest.mark.parametrize("reducer", ["tsvd", "pca"])
@pytest.mark.parametrize("embedding", ["count", "tfidf"])
def test_offline_encoder_combinations_produce_finite_angles(small_corpus, embedding, reducer):
    encoder = SentenceAngleEncoder(n_qubits=3, embedding=embedding, reducer=reducer)
    angles = encoder.fit(small_corpus).transform(small_corpus[:20])
    assert angles.shape == (20, 3)
    assert np.isfinite(angles).all()


def test_encoder_rejects_impossible_widths(small_corpus):
    with pytest.raises(ValueError, match="Cannot reduce"):
        SentenceAngleEncoder(n_qubits=100000).fit(small_corpus[:5])


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit must be called"):
        SentenceAngleEncoder(n_qubits=2).transform(["what is a test ?"])


def test_unknown_encoder_options_are_rejected():
    with pytest.raises(ValueError, match="embedding"):
        SentenceAngleEncoder(n_qubits=2, embedding="word2vec")
    with pytest.raises(ValueError, match="reducer"):
        SentenceAngleEncoder(n_qubits=2, reducer="nmf")
    with pytest.raises(ValueError, match="scaling"):
        SentenceAngleEncoder(n_qubits=2, scaling="minmax")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_macro_f1_ignores_class_size():
    """A model that abandons a rare class keeps accuracy but loses macro-F1."""
    # 98 of class 0 correct, both class-1 rows predicted as class 0.
    matrix = np.array([[98, 0], [2, 0]])
    accuracy = np.trace(matrix) / matrix.sum()
    assert accuracy == pytest.approx(0.98)
    assert macro_f1(matrix) < 0.5


def test_macro_f1_of_a_perfect_matrix_is_one():
    assert macro_f1(np.array([[5, 0], [0, 7]])) == pytest.approx(1.0)


def test_confusion_matrix_counts_true_by_predicted():
    labels = torch.tensor([0, 0, 1, 2])
    predictions = torch.tensor([0, 1, 1, 0])
    matrix = confusion_matrix(predictions, labels, 3)
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1 and matrix[2, 0] == 1
    assert matrix.sum() == len(labels)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_split(small_corpus, train_frame):
    encoder = SentenceAngleEncoder(n_qubits=3).fit(small_corpus)
    return build_split("tiny", train_frame.head(24), encoder)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: QuantumSentenceClassifier(n_qubits=3, n_classes=6, n_layers=1, reuploads=1),
        lambda: ClassicalSentenceClassifier(3, 6),
        lambda: NoBottleneckSentenceClassifier(3, 6),
    ],
    ids=["quantum", "classical", "no-bottleneck"],
)
def test_models_emit_one_logit_row_per_class(factory, tiny_split):
    torch.manual_seed(0)
    model = factory().eval()
    with torch.no_grad():
        logits = model(tiny_split.angles)
    assert logits.shape == (len(tiny_split), 6)
    assert torch.isfinite(logits).all()


def test_gradients_reach_the_circuit_weights(tiny_split):
    """Without this the circuit could be a frozen random feature map."""
    torch.manual_seed(0)
    model = QuantumSentenceClassifier(n_qubits=3, n_classes=6, n_layers=1, reuploads=1)
    torch.nn.functional.cross_entropy(model(tiny_split.angles), tiny_split.labels).backward()

    weights = model.quantum.weights
    assert weights.grad is not None
    assert float(weights.grad.abs().sum()) > 0.0


def test_reuploads_do_not_change_the_parameter_count():
    """Re-uploading buys expressivity with depth, not with extra parameters."""
    torch.manual_seed(0)
    single = QuantumSentenceClassifier(n_qubits=3, n_classes=6, n_layers=2, reuploads=1)
    double = QuantumSentenceClassifier(n_qubits=3, n_classes=6, n_layers=2, reuploads=2)
    assert single.quantum.weights.shape == double.quantum.weights.shape


def test_reuploads_must_fit_in_the_layer_budget():
    with pytest.raises(ValueError, match="cannot be split"):
        QuantumSentenceClassifier(n_qubits=3, n_classes=6, n_layers=1, reuploads=2)


def test_controls_are_close_to_the_circuit_in_parameter_count():
    """A control that differs in capacity would confound the comparison."""
    quantum = QuantumSentenceClassifier(n_qubits=8, n_classes=6, n_layers=2, reuploads=2)
    classical = ClassicalSentenceClassifier(8, 6)
    quantum_total = sum(p.numel() for p in quantum.parameters())
    classical_total = sum(p.numel() for p in classical.parameters())
    assert abs(quantum_total - classical_total) <= 32


def test_training_selects_the_best_development_epoch(tiny_split):
    torch.manual_seed(0)
    model = ClassicalSentenceClassifier(3, 6)
    record = train_model(
        model, tiny_split, tiny_split, 6, epochs=5, batch_size=8, learning_rate=0.05
    )
    assert 1 <= record["selected_epoch"] <= 5
    assert len(record["history"]) == 5
    losses = [entry["development_loss"] for entry in record["history"]]
    assert record["best_development_loss"] == pytest.approx(min(losses))


def test_early_stopping_halts_before_the_epoch_budget(tiny_split):
    torch.manual_seed(0)
    model = ClassicalSentenceClassifier(3, 6)
    record = train_model(
        model,
        tiny_split,
        tiny_split,
        6,
        epochs=200,
        batch_size=8,
        learning_rate=0.0,  # loss cannot improve, so patience must trigger
        patience=3,
    )
    assert record["epochs_run"] < 200


def test_evaluate_reports_accuracy_f1_and_a_full_confusion_matrix(tiny_split):
    torch.manual_seed(0)
    model = NoBottleneckSentenceClassifier(3, 6)
    scores = evaluate_split(model, tiny_split, 6)
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert 0.0 <= scores["macro_f1"] <= 1.0
    matrix = np.array(scores["confusion"])
    assert matrix.shape == (6, 6)
    assert matrix.sum() == len(tiny_split)


def test_evaluation_batching_does_not_change_the_result(tiny_split):
    """The Aer path evaluates in chunks; chunking must be invisible."""
    torch.manual_seed(0)
    model = ClassicalSentenceClassifier(3, 6)
    whole = evaluate_split(model, tiny_split, 6, batch_size=1024)
    chunked = evaluate_split(model, tiny_split, 6, batch_size=5)
    assert whole["accuracy"] == pytest.approx(chunked["accuracy"])
    assert whole["confusion"] == chunked["confusion"]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_confidence_interval_matches_a_hand_computed_t_value():
    # n=5, mean=3, sd=sqrt(2.5); t(0.975, 4) = 2.776445
    summary = mean_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["std"] == pytest.approx(np.sqrt(2.5))
    assert summary["half_width"] == pytest.approx(2.776445 * np.sqrt(2.5) / np.sqrt(5), rel=1e-5)


def test_confidence_interval_of_a_single_run_has_no_width():
    summary = mean_confidence_interval([0.75])
    assert summary["n"] == 1
    assert summary["ci_low"] == summary["ci_high"] == pytest.approx(0.75)
