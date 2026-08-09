"""Classical baseline and angle-encoded VQC training."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qnlp_hpc.trec_pca_vqc.model import AngleEncodedVQC
from qnlp_hpc.trec_pca_vqc.prepare_data import ID_TO_LABEL, TREC_CLASSES


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    """Calculate six-class classification metrics."""
    labels = list(range(len(TREC_CLASSES)))

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=labels,
        ),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=labels,
            target_names=list(TREC_CLASSES),
            output_dict=True,
            zero_division=0,
        ),
    }


def build_prediction_frame(
    source_frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Combine source questions with model predictions."""
    result = source_frame[
        ["text", "coarse_label", "fine_label", "label"]
    ].reset_index(drop=True).copy()

    result["predicted_label"] = predictions
    result["predicted_class"] = [
        ID_TO_LABEL[int(label)]
        for label in predictions
    ]
    result["correct"] = result["label"] == result["predicted_label"]

    for class_id, class_name in ID_TO_LABEL.items():
        result[f"probability_{class_name}"] = probabilities[:, class_id]

    return result


def run_classical_baseline(
    training_features: np.ndarray,
    development_features: np.ndarray,
    test_features: np.ndarray,
    training_frame: pd.DataFrame,
    development_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int = 42,
) -> dict[str, object]:
    """Train PCA + multinomial logistic regression."""
    training_labels = training_frame["label"].to_numpy(dtype=np.int64)
    development_labels = development_frame["label"].to_numpy(dtype=np.int64)
    test_labels = test_frame["label"].to_numpy(dtype=np.int64)

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )

    start = time.perf_counter()
    classifier.fit(training_features, training_labels)
    training_seconds = time.perf_counter() - start

    development_predictions = classifier.predict(development_features)
    test_predictions = classifier.predict(test_features)
    test_probabilities = classifier.predict_proba(test_features)

    development_metrics = calculate_metrics(
        development_labels,
        development_predictions,
    )
    test_metrics = calculate_metrics(
        test_labels,
        test_predictions,
    )

    print("\nClassical baseline")
    print("------------------")
    print(f"Training time: {training_seconds:.4f} s")
    print(
        "Development: "
        f"accuracy={development_metrics['accuracy']:.4f}, "
        f"macro_f1={development_metrics['macro_f1']:.4f}"
    )
    print(
        "Official test: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    return {
        "model": classifier,
        "training_seconds": training_seconds,
        "development_metrics": development_metrics,
        "test_metrics": test_metrics,
        "test_predictions": build_prediction_frame(
            test_frame,
            test_predictions,
            test_probabilities,
        ),
    }


def evaluate_vqc(
    model: AngleEncodedVQC,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a VQC without updating its parameters."""
    dataset = TensorDataset(features, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    model.eval()

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            total_loss += float(loss.item()) * len(batch_labels)
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels.cpu())

    combined_logits = torch.cat(all_logits, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)

    probabilities = torch.softmax(combined_logits, dim=1).numpy()
    predictions = torch.argmax(combined_logits, dim=1).numpy()
    expected = combined_labels.numpy()

    metrics = calculate_metrics(expected, predictions)

    return {
        "loss": total_loss / len(dataset),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "predictions": predictions,
        "probabilities": probabilities,
    }


def run_vqc_experiment(
    training_angles: np.ndarray,
    development_angles: np.ndarray,
    test_angles: np.ndarray,
    training_frame: pd.DataFrame,
    development_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    n_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    patience: int = 8,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, object]:
    """Train, select, and evaluate an angle-encoded VQC."""
    set_random_seeds(seed)

    if training_angles.ndim != 2:
        raise ValueError("training_angles must be a two-dimensional array.")

    n_qubits = training_angles.shape[1]

    if development_angles.shape[1] != n_qubits:
        raise ValueError("Development feature dimension does not match training.")

    if test_angles.shape[1] != n_qubits:
        raise ValueError("Test feature dimension does not match training.")

    selected_device = device_name or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(selected_device)

    print("\nAngle-encoded VQC")
    print("-----------------")
    print(f"Device: {device}")
    print(f"Qubits: {n_qubits}")
    print(f"Variational layers: {n_layers}")
    print(f"Statevector dimension: {1 << n_qubits}")
    print(f"Training examples: {len(training_frame)}")
    print(f"Development examples: {len(development_frame)}")
    print(f"Official test examples: {len(test_frame)}")

    training_features = torch.tensor(
        training_angles,
        dtype=torch.float32,
    )
    development_features = torch.tensor(
        development_angles,
        dtype=torch.float32,
    )
    test_features = torch.tensor(
        test_angles,
        dtype=torch.float32,
    )

    training_labels = torch.tensor(
        training_frame["label"].to_numpy(),
        dtype=torch.long,
    )
    development_labels = torch.tensor(
        development_frame["label"].to_numpy(),
        dtype=torch.long,
    )
    test_labels = torch.tensor(
        test_frame["label"].to_numpy(),
        dtype=torch.long,
    )

    training_dataset = TensorDataset(
        training_features,
        training_labels,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=min(batch_size, len(training_dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = AngleEncodedVQC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=len(TREC_CLASSES),
        seed=seed,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_development_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_predictions: list[torch.Tensor] = []
        epoch_labels: list[torch.Tensor] = []

        for batch_features, batch_labels in training_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch_labels)
            epoch_predictions.append(torch.argmax(logits, dim=1).cpu())
            epoch_labels.append(batch_labels.cpu())

        training_loss = epoch_loss / len(training_dataset)

        combined_training_predictions = torch.cat(
            epoch_predictions
        ).numpy()
        combined_training_labels = torch.cat(epoch_labels).numpy()

        training_accuracy = float(
            accuracy_score(
                combined_training_labels,
                combined_training_predictions,
            )
        )

        development_result = evaluate_vqc(
            model,
            development_features,
            development_labels,
            batch_size=batch_size,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_accuracy": training_accuracy,
                "development_loss": development_result["loss"],
                "development_accuracy": development_result["accuracy"],
                "development_macro_f1": development_result["macro_f1"],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={training_loss:.4f} | "
            f"train_acc={training_accuracy:.4f} | "
            f"dev_loss={development_result['loss']:.4f} | "
            f"dev_acc={development_result['accuracy']:.4f} | "
            f"dev_macro_f1={development_result['macro_f1']:.4f}"
        )

        if development_result["loss"] < best_development_loss - 1e-6:
            best_development_loss = float(development_result["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    training_seconds = time.perf_counter() - training_start

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    development_result = evaluate_vqc(
        model,
        development_features,
        development_labels,
        batch_size=batch_size,
        device=device,
    )
    test_result = evaluate_vqc(
        model,
        test_features,
        test_labels,
        batch_size=batch_size,
        device=device,
    )

    print(f"Selected epoch: {best_epoch}")
    print(f"Training time: {training_seconds:.2f} s")
    print(
        "Selected development: "
        f"accuracy={development_result['accuracy']:.4f}, "
        f"macro_f1={development_result['macro_f1']:.4f}"
    )
    print(
        "Official test: "
        f"accuracy={test_result['accuracy']:.4f}, "
        f"macro_f1={test_result['macro_f1']:.4f}"
    )

    print(
        classification_report(
            test_labels.numpy(),
            test_result["predictions"],
            labels=list(range(len(TREC_CLASSES))),
            target_names=list(TREC_CLASSES),
            digits=4,
            zero_division=0,
        )
    )

    return {
        "model": model,
        "history": pd.DataFrame(history_rows),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "device": str(device),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "development_metrics": development_result,
        "test_metrics": test_result,
        "test_predictions": build_prediction_frame(
            test_frame,
            test_result["predictions"],
            test_result["probabilities"],
        ),
    }