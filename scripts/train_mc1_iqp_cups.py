"""Train and evaluate an MC1 sentence-pair classifier.

The model uses ``cups_reader`` to create sentence diagrams, ``IQPAnsatz`` to
convert them to quantum circuits, and a small PyTorch classifier on top of the
quantum outputs.

Every complete sentence is assigned to exactly one of training, development,
or test.

A sentence pair is retained only when both of its sentences belong to the
same split.  Cross-split pairs are excluded and written to a diagnostic CSV.
Consequently, no complete sentence appearing in development or test appears
in training.

The split is deterministic for the supplied MC1 dataset and was chosen to:
* retain as many of the 100 pairs as possible;
* preserve both labels in every split;
* keep all held-out word tokens represented in training, so this evaluates
  compositional generalisation to unseen complete sentences rather than
  out-of-vocabulary handling.

The quantum model's symbol table is constructed from training circuits only.
Development and test labels are never used for optimisation.

This script is intended to live in the repository's ``scripts/`` directory.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, SupportsFloat, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lambeq import (
    AtomicType,
    Dataset,
    IQPAnsatz,
    PytorchQuantumModel,
    PytorchTrainer,
    cups_reader,
)
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    """Find the project root, falling back to the working directory."""
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


SEED = 2

BATCH_SIZE = 8
EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
CLASSIFIER_HIDDEN_DIM = 16
CLASSIFIER_DROPOUT = 0.05

# cups_reader maps sentence-type wires to a genuine two-qubit IQP circuit.
SENTENCE_QUBITS = 2
IQP_LAYERS = 1
N_SINGLE_QUBIT_PARAMS = 3
CIRCUIT_OUTPUT_DIM = 2**SENTENCE_QUBITS

REPO_ROOT = find_repo_root()
DATA_PATH = REPO_ROOT / "data" / "processed" / "MC1.txt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mc1_iqp_cups"
LOG_DIR = OUTPUT_DIR / "training_logs"


# These held-out sentence sets are deterministic and disjoint.
# Training receives every sentence not listed here.
DEVELOPMENT_SENTENCES = frozenset(
    {
        "chef creates meal",
        "chef prepares tasty dish",
        "cook prepares meal",
        "devoted programmer creates advanced code",
        "experienced chef creates complicated meal",
        "experienced cook prepares complicated dish",
        "programmer writes complicated code",
        "skilful chef prepares complicated dish",
        "skilful chef prepares tasty dish",
        "skilful cook creates tasty dish",
        "skilful cook prepares meal",
        "skilful programmer creates complicated code",
        "skilful programmer writes advanced code",
        "skilful programmer writes code",
    }
)

TEST_SENTENCES = frozenset(
    {
        "chef creates dish",
        "chef creates tasty meal",
        "cook prepares complicated dish",
        "cook prepares complicated meal",
        "experienced chef creates complicated dish",
        "experienced chef prepares meal",
        "experienced cook prepares meal",
        "hacker creates advanced code",
        "hacker writes advanced code",
        "programmer creates code",
        "programmer writes advanced code",
    }
)


# ---------------------------------------------------------------------------
# Reproducibility and data loading
# ---------------------------------------------------------------------------


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mc1(path: Path) -> pd.DataFrame:
    """Load MC1 sentence pairs from ``path``."""
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
                    f"Line {line_number} has a non-integer label: " f"{label_text!r}"
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

    missing_labels = {0, 1}.difference(frame["label"].unique())
    if missing_labels:
        raise ValueError(
            "MC1.txt must contain both labels 0 and 1; "
            f"missing {sorted(missing_labels)}."
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

    Rows whose two endpoints belong to different sentence partitions are
    excluded.  The returned fourth dataframe contains those excluded rows.
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
        raise RuntimeError("Development and test sentence manifests overlap.")

    training_sentences = all_sentences.difference(
        DEVELOPMENT_SENTENCES.union(TEST_SENTENCES)
    )
    if not training_sentences:
        raise RuntimeError("The strict split produced no training sentences.")

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
                f"The {name} split must contain labels 0 and 1; "
                f"found {sorted(labels)}."
            )

    observed_training_sentences = sentence_set(training)
    observed_development_sentences = sentence_set(development)
    observed_test_sentences = sentence_set(test)

    overlaps = {
        "train_dev": observed_training_sentences.intersection(
            observed_development_sentences
        ),
        "train_test": observed_training_sentences.intersection(observed_test_sentences),
        "dev_test": observed_development_sentences.intersection(
            observed_test_sentences
        ),
    }
    if any(overlaps.values()):
        raise RuntimeError(
            "Sentence-disjointness check failed: "
            + repr({key: sorted(value) for key, value in overlaps.items()})
        )

    training_vocab = token_vocabulary(observed_training_sentences)
    held_out_vocab = token_vocabulary(
        observed_development_sentences.union(observed_test_sentences)
    )
    unseen_tokens = held_out_vocab.difference(training_vocab)
    if unseen_tokens:
        raise RuntimeError(
            "Held-out splits contain tokens absent from training: "
            f"{sorted(unseen_tokens)}"
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
) -> None:
    """Write the split manifest, excluded pairs, and summary statistics."""
    assignments: list[dict[str, str]] = []
    for split_name, split in (
        ("training", training),
        ("development", development),
        ("test", test),
    ):
        for sentence in sorted(sentence_set(split)):
            assignments.append(
                {
                    "sentence": sentence,
                    "split": split_name,
                }
            )

    pd.DataFrame(assignments).to_csv(
        OUTPUT_DIR / "sentence_disjoint_manifest.csv",
        index=False,
    )
    dropped.to_csv(
        OUTPUT_DIR / "sentence_disjoint_dropped_pairs.csv",
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
        OUTPUT_DIR / "sentence_disjoint_split_summary.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Diagram construction and model
# ---------------------------------------------------------------------------


def build_quantum_circuits(frame: pd.DataFrame) -> dict[str, object]:
    """Create cups_reader diagrams and map them to two-qubit IQP circuits."""
    unique_sentences = pd.unique(
        pd.concat(
            [frame["sentence_1"], frame["sentence_2"]],
            ignore_index=True,
        )
    ).tolist()

    parse_start = time.perf_counter()
    parsed_diagrams = cups_reader.sentences2diagrams(unique_sentences)
    parse_seconds = time.perf_counter() - parse_start

    failed_sentences = [
        sentence
        for sentence, diagram in zip(
            unique_sentences,
            parsed_diagrams,
            strict=True,
        )
        if diagram is None
    ]
    if failed_sentences:
        raise RuntimeError(
            "cups_reader could not create diagrams for: " + repr(failed_sentences)
        )

    iqp_ansatz = IQPAnsatz(
        {AtomicType.SENTENCE: SENTENCE_QUBITS},
        n_layers=IQP_LAYERS,
        n_single_qubit_params=N_SINGLE_QUBIT_PARAMS,
    )

    ansatz_start = time.perf_counter()
    quantum_circuits = {
        text: iqp_ansatz(diagram)
        for text, diagram in zip(
            unique_sentences,
            parsed_diagrams,
            strict=True,
        )
    }
    ansatz_seconds = time.perf_counter() - ansatz_start

    print(
        f"Built {len(unique_sentences)} cups_reader diagrams in "
        f"{parse_seconds:.2f} s."
    )
    print(
        f"Applied {IQP_LAYERS}-layer two-qubit lambeq IQPAnsatz to "
        f"{len(unique_sentences)} diagrams in {ansatz_seconds:.2f} s."
    )
    return quantum_circuits


def make_pairs_and_targets(
    split: pd.DataFrame,
    quantum_circuits: dict[str, object],
) -> tuple[list[tuple[object, object]], np.ndarray]:
    """Convert a dataframe split into circuit pairs and writable class IDs."""
    pairs = [
        (
            quantum_circuits[row.sentence_1],
            quantum_circuits[row.sentence_2],
        )
        for row in split.itertuples(index=False)
    ]
    targets = np.array(split["label"], dtype=np.int64, copy=True)
    return pairs, targets


def collect_circuit_symbols(circuits: Sequence[Any]) -> set[Any]:
    """Return every trainable symbol appearing in a circuit collection."""
    symbols: set[Any] = set()
    for circuit in circuits:
        symbols.update(circuit.free_symbols)
    return symbols


class IQPPairModel(PytorchQuantumModel):
    """Symmetric Siamese classifier over two IQP sentence circuits."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        # A two-qubit output gives four probabilities. The symmetric pair
        # representation concatenates |left-right| and left*right.
        pair_feature_dim = 2 * CIRCUIT_OUTPUT_DIM
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pair_feature_dim, CLASSIFIER_HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(CLASSIFIER_DROPOUT),
            torch.nn.Linear(CLASSIFIER_HIDDEN_DIM, 2),
        )

    def forward(
        self,
        diagram_pairs: Sequence[tuple[object, object]],
    ) -> torch.Tensor:
        if not diagram_pairs:
            raise ValueError("diagram_pairs cannot be empty.")

        flat_circuits = [circuit for pair in diagram_pairs for circuit in pair]

        sentence_probabilities = self.get_diagram_output(flat_circuits)
        sentence_probabilities = sentence_probabilities.reshape(
            len(diagram_pairs),
            2,
            CIRCUIT_OUTPUT_DIM,
        )

        # Centre each probability component from [0, 1] to [-1, 1].
        sentence_features = 2.0 * (sentence_probabilities - 0.5)
        left = sentence_features[:, 0, :]
        right = sentence_features[:, 1, :]

        absolute_difference = torch.abs(left - right)
        elementwise_product = left * right
        pair_features = torch.cat(
            (absolute_difference, elementwise_product),
            dim=1,
        )

        expected_width = 2 * CIRCUIT_OUTPUT_DIM
        if pair_features.shape[1] != expected_width:
            raise RuntimeError(
                "Pair-feature dimension mismatch: "
                f"expected {expected_width}, got {pair_features.shape[1]}."
            )

        # PytorchQuantumModel's tensor contraction commonly returns float64,
        # while torch.nn.Linear parameters default to float32. Matrix
        # multiplication requires matching dtypes, so align the quantum
        # features with the classifier without detaching the computation graph.
        classifier_parameter = next(self.classifier.parameters())
        pair_features = pair_features.to(
            device=classifier_parameter.device,
            dtype=classifier_parameter.dtype,
        )

        return self.classifier(pair_features)


class LambeqCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy accepting integer, float, or one-hot lambeq targets."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(logits, targets.long())


def class_ids(y: torch.Tensor) -> torch.Tensor:
    if y.ndim > 1:
        return torch.argmax(y, dim=1).long()
    return y.long()


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    predicted = torch.argmax(y_hat, dim=1)
    expected = class_ids(y)
    return float((predicted == expected).float().mean().item())


def evaluation_loss(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    expected = class_ids(y)
    return float(F.cross_entropy(y_hat, expected).detach().cpu().item())


# ---------------------------------------------------------------------------
# Diagnostics and output helpers
# ---------------------------------------------------------------------------


def prediction_counts(
    model: torch.nn.Module,
    pairs: Sequence[tuple[object, object]],
) -> dict[int, int]:
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(pairs), dim=1).detach().cpu().numpy()
    labels, counts = np.unique(predictions, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def to_float_list(values: Sequence[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            result.append(float(value.detach().cpu().item()))
        else:
            result.append(float(cast(SupportsFloat, value)))
    return result


def save_benchmark_plot(
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Path,
    selected_epoch: int,
) -> None:
    epochs = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, values)

    if 1 <= selected_epoch <= len(values):
        selected_value = values[selected_epoch - 1]
        ax.axvline(selected_epoch, linestyle="--", alpha=0.45)
        ax.scatter(
            [selected_epoch],
            [selected_value],
            facecolors="none",
            edgecolors="black",
        )
        ax.annotate(
            f"selected epoch = {selected_epoch}\n(min dev loss)",
            xy=(selected_epoch, selected_value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel == "Accuracy":
        ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_four_benchmark_plots(
    history: pd.DataFrame,
    selected_epoch: int,
) -> None:
    plot_specs = (
        ("train_loss", "Loss", "Training set — Loss", "benchmark_train_loss.png"),
        (
            "development_loss",
            "Loss",
            "Development set — Loss",
            "benchmark_development_loss.png",
        ),
        (
            "train_accuracy",
            "Accuracy",
            "Training set — Accuracy",
            "benchmark_train_accuracy.png",
        ),
        (
            "development_accuracy",
            "Accuracy",
            "Development set — Accuracy",
            "benchmark_development_accuracy.png",
        ),
    )
    for column, ylabel, title, filename in plot_specs:
        save_benchmark_plot(
            history[column].tolist(),
            ylabel=ylabel,
            title=title,
            output_path=OUTPUT_DIR / filename,
            selected_epoch=selected_epoch,
        )


def get_best_epoch(history: pd.DataFrame) -> int:
    ranked = history.sort_values(
        ["development_loss", "development_accuracy", "epoch"],
        ascending=[True, False, True],
    )
    return int(ranked.iloc[0]["epoch"])


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    set_random_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent an incompatible checkpoint from a previous run being reloaded.
    stale_checkpoint = LOG_DIR / "best_model.lt"
    if stale_checkpoint.is_file():
        stale_checkpoint.unlink()

    device_id = -1
    device_name = "CPU (lambeq PytorchQuantumModel tensor contraction)"

    print(f"PyTorch: {torch.__version__}")
    print(f"Training device: {device_name}")
    print(f"Reading: {DATA_PATH}")
    print(
        "Training configuration: "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, hidden_dim={CLASSIFIER_HIDDEN_DIM}, "
        f"dropout={CLASSIFIER_DROPOUT}, "
        f"patience={EARLY_STOPPING_PATIENCE}, "
        f"iqp_layers={IQP_LAYERS}, "
        f"sentence_qubits={SENTENCE_QUBITS}"
    )

    frame = load_mc1(DATA_PATH)
    print(f"Loaded {len(frame)} examples.")

    (
        train_frame,
        development_frame,
        test_frame,
        dropped_frame,
    ) = split_sentence_disjoint(frame)

    print("Complete sentences are disjoint across " "training/development/test.")
    print(
        f"Retained {len(train_frame) + len(development_frame) + len(test_frame)} "
        f"of {len(frame)} pairs; dropped {len(dropped_frame)} cross-split pairs."
    )
    save_split_diagnostics(
        frame,
        train_frame,
        development_frame,
        test_frame,
        dropped_frame,
    )

    for name, split in (
        ("training", train_frame),
        ("development", development_frame),
        ("test", test_frame),
    ):
        counts = split["label"].value_counts().sort_index().to_dict()
        print(f"{name:>11}: {len(split):3d} examples | labels {counts}")

    retained_frame = pd.concat(
        [train_frame, development_frame, test_frame],
        ignore_index=True,
    )
    quantum_circuits = build_quantum_circuits(retained_frame)

    train_pairs, train_targets = make_pairs_and_targets(train_frame, quantum_circuits)
    development_pairs, development_targets = make_pairs_and_targets(
        development_frame, quantum_circuits
    )
    test_pairs, test_targets = make_pairs_and_targets(test_frame, quantum_circuits)

    training_quantum_circuits = [circuit for pair in train_pairs for circuit in pair]

    evaluation_quantum_circuits = [
        circuit for pair in development_pairs + test_pairs for circuit in pair
    ]
    missing_symbols = collect_circuit_symbols(evaluation_quantum_circuits).difference(
        collect_circuit_symbols(training_quantum_circuits)
    )
    if missing_symbols:
        raise RuntimeError(
            "Development/test circuits contain symbols that never appear in "
            "training: "
            f"{sorted(map(str, missing_symbols))}"
        )

    # Construct trainable symbols from training circuits only. Held-out complete
    # sentences do not participate in model initialisation.
    model = IQPPairModel.from_diagrams(training_quantum_circuits)
    model.initialise_weights()
    print(f"Trainable lambeq symbols: {len(model.symbols)}")
    print(f"Circuit output dimension: {CIRCUIT_OUTPUT_DIM}")

    trainer = PytorchTrainer(
        model=model,
        loss_function=LambeqCrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_args={"weight_decay": WEIGHT_DECAY},
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device_id,
        evaluate_functions={
            "acc": accuracy,
            "loss": evaluation_loss,
        },
        evaluate_on_train=True,
        log_dir=LOG_DIR,
        verbose="text",
        seed=SEED,
    )

    train_dataset = Dataset(
        train_pairs,
        train_targets,
        batch_size=min(BATCH_SIZE, len(train_pairs)),
        shuffle=True,
    )
    development_dataset = Dataset(
        development_pairs,
        development_targets,
        batch_size=len(development_pairs),
        shuffle=False,
    )

    training_start = time.perf_counter()
    trainer.fit(
        train_dataset,
        development_dataset,
        eval_interval=1,
        log_interval=5,
        early_stopping_criterion="loss",
        early_stopping_interval=EARLY_STOPPING_PATIENCE,
        minimize_criterion=True,
    )
    training_seconds = time.perf_counter() - training_start
    print(f"Training completed in {training_seconds:.2f} s.")

    # Use full-set evaluation loss for both training and development curves.
    train_loss = to_float_list(trainer.train_eval_results["loss"])
    development_loss = to_float_list(trainer.val_eval_results["loss"])
    train_accuracy = to_float_list(trainer.train_eval_results["acc"])
    development_accuracy = to_float_list(trainer.val_eval_results["acc"])

    lengths = {
        len(train_loss),
        len(development_loss),
        len(train_accuracy),
        len(development_accuracy),
    }
    if len(lengths) != 1:
        raise RuntimeError(
            "Training histories have inconsistent lengths: "
            f"train_loss={len(train_loss)}, "
            f"development_loss={len(development_loss)}, "
            f"train_accuracy={len(train_accuracy)}, "
            f"development_accuracy={len(development_accuracy)}"
        )

    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_loss) + 1),
            "train_loss": train_loss,
            "development_loss": development_loss,
            "train_accuracy": train_accuracy,
            "development_accuracy": development_accuracy,
        }
    )
    best_epoch = get_best_epoch(history)
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    history.to_csv(
        OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_history.csv",
        index=False,
    )
    save_four_benchmark_plots(history, best_epoch)

    trainer_best_model = Path(trainer.log_dir) / "best_model.lt"
    if trainer_best_model.is_file():
        model.load(str(trainer_best_model))
        print(f"Loaded best checkpoint: {trainer_best_model}")
    else:
        print(
            "Warning: best_model.lt was not found; evaluating the final "
            "epoch model instead."
        )

    train_prediction_counts = prediction_counts(model, train_pairs)
    development_prediction_counts = prediction_counts(model, development_pairs)
    test_prediction_counts = prediction_counts(model, test_pairs)
    print(f"Training prediction counts: {train_prediction_counts}")
    print(f"Development prediction counts: {development_prediction_counts}")
    print(f"Test prediction counts: {test_prediction_counts}")

    model.eval()
    with torch.no_grad():
        test_logits = model(test_pairs)
        test_probabilities = torch.softmax(test_logits, dim=1)
        test_predictions = (
            torch.argmax(test_probabilities, dim=1).detach().cpu().numpy()
        )

    test_true = test_targets.astype(np.int64, copy=False)
    test_accuracy_value = float(np.mean(test_predictions == test_true))

    print(f"Selected epoch: {best_epoch} (minimum development loss)")
    print(
        "Selected development metrics: "
        f"accuracy={best_row['development_accuracy']:.4f}, "
        f"loss={best_row['development_loss']:.4f}"
    )
    print(f"Test accuracy: {test_accuracy_value:.4f}")
    print(
        classification_report(
            test_true,
            test_predictions,
            labels=[0, 1],
            target_names=[
                "different domain (0)",
                "same domain (1)",
            ],
            digits=4,
            zero_division=0,
        )
    )

    results = test_frame.reset_index(drop=True).copy()
    results["predicted_label"] = test_predictions
    results["correct"] = results["label"] == results["predicted_label"]
    results["probability_label_0"] = test_probabilities[:, 0].detach().cpu().numpy()
    results["probability_label_1"] = test_probabilities[:, 1].detach().cpu().numpy()
    results.to_csv(
        OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_test_predictions.csv",
        index=False,
    )

    final_lambeq_model_path = (
        OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_best_model.lt"
    )
    model.save(str(final_lambeq_model_path))
    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_pair_model.pt",
    )

    summary = pd.DataFrame(
        [
            {
                "reader": "cups_reader",
                "sentence_overlap_train_dev": 0,
                "sentence_overlap_train_test": 0,
                "sentence_overlap_dev_test": 0,
                "dropped_cross_split_pairs": len(dropped_frame),
                "training_pairs": len(train_frame),
                "development_pairs": len(development_frame),
                "test_pairs": len(test_frame),
                "seed": SEED,
                "epochs_completed": len(history),
                "selected_epoch": best_epoch,
                "selected_train_loss": float(best_row["train_loss"]),
                "selected_development_loss": float(best_row["development_loss"]),
                "selected_train_accuracy": float(best_row["train_accuracy"]),
                "selected_development_accuracy": float(
                    best_row["development_accuracy"]
                ),
                "test_accuracy": test_accuracy_value,
                "training_seconds": training_seconds,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "sentence_qubits": SENTENCE_QUBITS,
                "iqp_layers": IQP_LAYERS,
                "n_single_qubit_params": N_SINGLE_QUBIT_PARAMS,
                "circuit_output_dim": CIRCUIT_OUTPUT_DIM,
                "quantum_backend": "PytorchQuantumModel tensor contraction",
                "classifier_hidden_dim": CLASSIFIER_HIDDEN_DIM,
                "classifier_dropout": CLASSIFIER_DROPOUT,
                "early_stopping_criterion": "development_loss",
                "training_prediction_counts": str(train_prediction_counts),
                "development_prediction_counts": str(development_prediction_counts),
                "test_prediction_counts": str(test_prediction_counts),
            }
        ]
    )
    summary.to_csv(
        OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_training_summary.csv",
        index=False,
    )

    print(f"Outputs saved in: {OUTPUT_DIR}")
    for filename in (
        "benchmark_train_loss.png",
        "benchmark_development_loss.png",
        "benchmark_train_accuracy.png",
        "benchmark_development_accuracy.png",
        "sentence_disjoint_manifest.csv",
        "sentence_disjoint_dropped_pairs.csv",
        "sentence_disjoint_split_summary.csv",
        "mc1_iqp_cups_sentence_disjoint_best_model.lt",
        "mc1_iqp_cups_sentence_disjoint_pair_model.pt",
        "mc1_iqp_cups_sentence_disjoint_history.csv",
        "mc1_iqp_cups_sentence_disjoint_test_predictions.csv",
        "mc1_iqp_cups_sentence_disjoint_training_summary.csv",
        "training_logs/best_model.lt",
    ):
        print(f"  {filename}")


if __name__ == "__main__":
    main()
