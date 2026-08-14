"""Tests for the MC1 dimensionality-reduction route.

Run from the repo root:  PYTHONPATH=src pytest pennylane_aer/test_mc1_reduced.py

These target the claims the pipeline makes about itself - derived topics agree
with every shipped label, the encoder never sees held-out sentences, pair
features are swap-invariant, and the loss actually moves the circuit - rather
than restating the implementation.
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

from mc1_reduced import (  # noqa: E402
    ClassicalPairClassifier,
    NoBottleneckPairClassifier,
    QuantumPairClassifier,
    SentenceAngleEncoder,
    augmented_pairs,
    build_split,
    derive_sentence_topics,
    evaluate_split,
    frame_pairs,
    mean_confidence_interval,
    train_pair_model,
)

from qnlp_hpc.mc1_iqp_cups import config as evelyn_config  # noqa: E402
from qnlp_hpc.mc1_iqp_cups import data as evelyn_data  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return evelyn_data.load_mc1(evelyn_config.DATA_PATH)


@pytest.fixture(scope="module")
def topics(frame) -> dict[str, int]:
    return derive_sentence_topics(frame)


@pytest.fixture(scope="module")
def train_sentences(frame) -> list[str]:
    all_sentences = set(frame["sentence_1"]).union(frame["sentence_2"])
    return sorted(
        all_sentences - evelyn_config.TEST_SENTENCES - evelyn_config.DEVELOPMENT_SENTENCES
    )


# --------------------------------------------------------------------------
# Derived topics
# --------------------------------------------------------------------------


def test_derived_topics_reproduce_every_shipped_label(frame, topics):
    """The whole augmentation rests on this: if it fails, the extra pairs lie."""
    for row in frame.itertuples(index=False):
        same = topics[row.sentence_1] == topics[row.sentence_2]
        assert same == (int(row.label) == 1), (row.sentence_1, row.sentence_2, row.label)


def test_derived_topics_cover_every_sentence_and_use_both_labels(frame, topics):
    sentences = set(frame["sentence_1"]).union(frame["sentence_2"])
    assert set(topics) == sentences
    assert set(topics.values()) == {0, 1}


def test_contradictory_labels_are_rejected():
    """a==b, b==c, but a!=c is not two-colourable and must not pass silently."""
    contradictory = pd.DataFrame(
        [
            {"sentence_1": "a", "sentence_2": "b", "label": 1},
            {"sentence_1": "b", "sentence_2": "c", "label": 1},
            {"sentence_1": "a", "sentence_2": "c", "label": 0},
        ]
    )
    with pytest.raises(ValueError, match="not consistent"):
        derive_sentence_topics(contradictory)


def test_augmented_pairs_are_complete_and_unordered(topics):
    sentences = sorted(evelyn_config.TEST_SENTENCES)
    pairs = augmented_pairs(sentences, topics)
    n = len(sentences)
    assert len(pairs) == n * (n - 1) // 2
    assert len({tuple(sorted(pair[:2])) for pair in pairs}) == len(pairs)
    assert all(left != right for left, right, _ in pairs)


def test_augmentation_agrees_with_the_shipped_test_pairs(frame, topics):
    """Augmentation must be a superset of MC1's own labels, not a replacement."""
    _, _, test_frame, _ = evelyn_data.split_sentence_disjoint(frame)
    derived = {
        tuple(sorted((left, right))): label
        for left, right, label in augmented_pairs(sorted(evelyn_config.TEST_SENTENCES), topics)
    }
    shipped = {
        tuple(sorted((left, right))): label for left, right, label in frame_pairs(test_frame)
    }
    assert shipped, "expected MC1 to ship test pairs"
    for key, label in shipped.items():
        assert derived[key] == label


# --------------------------------------------------------------------------
# Encoder: leakage and scaling
# --------------------------------------------------------------------------


def test_encoder_is_fitted_on_training_sentences_only(train_sentences):
    """A held-out sentence must not change the feature map it is scored under."""
    encoder = SentenceAngleEncoder(n_qubits=4).fit(train_sentences)
    held_out = sorted(evelyn_config.TEST_SENTENCES)

    first = encoder.transform(held_out)
    encoder.transform(sorted(evelyn_config.DEVELOPMENT_SENTENCES))
    second = encoder.transform(held_out)

    np.testing.assert_allclose(first, second)


def test_angles_stay_inside_the_bloch_range(train_sentences):
    """Wrapping past +-pi would alias distinct sentences onto the same angle."""
    for scaling in ("global", "per-component"):
        encoder = SentenceAngleEncoder(n_qubits=4, scaling=scaling).fit(train_sentences)
        angles = encoder.transform(sorted(evelyn_config.TEST_SENTENCES))
        assert np.all(np.abs(angles) < np.pi)


def test_global_scaling_preserves_the_component_variance_ordering(train_sentences):
    """The point of `global`: PCA's variance ordering must survive into angles.

    `per-component` divides each component by its own standard deviation, which
    lifts the lowest-variance component to the amplitude of the highest.
    """
    held_out = sorted(evelyn_config.TEST_SENTENCES)

    global_angles = (
        SentenceAngleEncoder(n_qubits=4, scaling="global").fit(train_sentences).transform(held_out)
    )
    per_component_angles = (
        SentenceAngleEncoder(n_qubits=4, scaling="per-component")
        .fit(train_sentences)
        .transform(held_out)
    )

    global_spread = global_angles.std(axis=0)
    per_component_spread = per_component_angles.std(axis=0)

    assert global_spread[0] > global_spread[-1]
    assert per_component_spread[-1] / per_component_spread[0] > global_spread[-1] / global_spread[0]


def test_encoder_rejects_impossible_widths(train_sentences):
    with pytest.raises(ValueError, match="Cannot reduce"):
        SentenceAngleEncoder(n_qubits=64).fit(train_sentences)


@pytest.mark.parametrize("reducer", ["pca", "tsvd"])
@pytest.mark.parametrize("embedding", ["count", "tfidf"])
def test_offline_encoder_combinations_produce_finite_angles(train_sentences, embedding, reducer):
    encoder = SentenceAngleEncoder(n_qubits=3, embedding=embedding, reducer=reducer)
    angles = encoder.fit(train_sentences).transform(sorted(evelyn_config.TEST_SENTENCES))
    assert angles.shape == (len(evelyn_config.TEST_SENTENCES), 3)
    assert np.isfinite(angles).all()


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit must be called"):
        SentenceAngleEncoder(n_qubits=2).transform(["chef creates dish"])


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_splits_are_sentence_disjoint(frame, topics, train_sentences):
    encoder = SentenceAngleEncoder(n_qubits=3).fit(train_sentences)
    groups = {
        "train": train_sentences,
        "development": sorted(evelyn_config.DEVELOPMENT_SENTENCES),
        "test": sorted(evelyn_config.TEST_SENTENCES),
    }
    splits = {
        name: build_split(name, augmented_pairs(sentences, topics), encoder, sentences)
        for name, sentences in groups.items()
    }
    for left in splits:
        for right in splits:
            if left < right:
                assert not set(splits[left].sentences) & set(splits[right].sentences)


def test_split_indices_address_the_right_sentences(topics, train_sentences):
    encoder = SentenceAngleEncoder(n_qubits=3).fit(train_sentences)
    sentences = sorted(evelyn_config.TEST_SENTENCES)
    pairs = augmented_pairs(sentences, topics)
    split = build_split("test", pairs, encoder, sentences)

    for position, (left, right, label) in enumerate(pairs):
        assert split.sentences[int(split.left[position])] == left
        assert split.sentences[int(split.right[position])] == right
        assert int(split.labels[position]) == label


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_split(topics, train_sentences):
    encoder = SentenceAngleEncoder(n_qubits=3).fit(train_sentences)
    sentences = sorted(evelyn_config.TEST_SENTENCES)[:5]
    return build_split("small", augmented_pairs(sentences, topics), encoder, sentences)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: QuantumPairClassifier(n_qubits=3, n_layers=1, reuploads=1),
        lambda: ClassicalPairClassifier(3),
        lambda: NoBottleneckPairClassifier(3),
    ],
    ids=["quantum", "classical", "no-bottleneck"],
)
def test_pair_models_are_symmetric(factory, small_split):
    """MC1's label is a property of the pair, so order must not matter."""
    torch.manual_seed(0)
    model = factory().eval()
    with torch.no_grad():
        forward = model(small_split.angles, small_split.left, small_split.right)
        reversed_ = model(small_split.angles, small_split.right, small_split.left)
    torch.testing.assert_close(forward, reversed_, atol=1e-6, rtol=1e-5)


def test_quantum_features_are_evaluated_once_per_distinct_sentence(small_split, monkeypatch):
    """The cost claim: circuit work scales with sentences, not with pairs.

    5 sentences make 10 pairs, i.e. 20 sentence slots.  A model that
    featurised pairs directly would push 20 rows through the circuit; this one
    must push 5.
    """
    torch.manual_seed(0)
    model = QuantumPairClassifier(n_qubits=3, n_layers=1, reuploads=1)

    seen: list[int] = []
    original = QuantumPairClassifier.sentence_features

    def recording(self, angles):
        seen.append(int(angles.shape[0]))
        return original(self, angles)

    monkeypatch.setattr(QuantumPairClassifier, "sentence_features", recording)
    model.eval()
    with torch.no_grad():
        model(small_split.angles, small_split.left, small_split.right)

    assert len(small_split) == 10
    assert seen == [5]


def test_gradients_reach_the_circuit_weights(small_split):
    """Without this the circuit could be a frozen random feature map."""
    torch.manual_seed(0)
    model = QuantumPairClassifier(n_qubits=3, n_layers=1, reuploads=1)
    logits = model(small_split.angles, small_split.left, small_split.right)
    torch.nn.functional.cross_entropy(logits, small_split.labels).backward()

    weights = model.quantum.weights
    assert weights.grad is not None
    assert float(weights.grad.abs().sum()) > 0.0


def test_reuploads_do_not_change_the_parameter_count():
    """Re-uploading buys expressivity with depth, not with extra parameters."""
    torch.manual_seed(0)
    single = QuantumPairClassifier(n_qubits=3, n_layers=2, reuploads=1)
    double = QuantumPairClassifier(n_qubits=3, n_layers=2, reuploads=2)
    assert single.quantum.weights.shape == double.quantum.weights.shape


def test_reuploads_must_fit_in_the_layer_budget():
    with pytest.raises(ValueError, match="cannot be split"):
        QuantumPairClassifier(n_qubits=3, n_layers=1, reuploads=2)


def test_training_selects_the_best_development_epoch(small_split):
    torch.manual_seed(0)
    model = ClassicalPairClassifier(3)
    record = train_pair_model(
        model, small_split, small_split, epochs=5, batch_size=4, learning_rate=0.05
    )
    assert 1 <= record["selected_epoch"] <= 5
    assert len(record["history"]) == 5
    losses = [entry["development_loss"] for entry in record["history"]]
    assert record["best_development_loss"] == pytest.approx(min(losses))
    accuracy, _ = evaluate_split(model, small_split)
    assert 0.0 <= accuracy <= 1.0


def test_early_stopping_halts_before_the_epoch_budget(small_split):
    torch.manual_seed(0)
    model = ClassicalPairClassifier(3)
    record = train_pair_model(
        model,
        small_split,
        small_split,
        epochs=200,
        batch_size=4,
        learning_rate=0.0,  # loss cannot improve, so patience must trigger
        patience=3,
    )
    assert record["epochs_run"] < 200


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
