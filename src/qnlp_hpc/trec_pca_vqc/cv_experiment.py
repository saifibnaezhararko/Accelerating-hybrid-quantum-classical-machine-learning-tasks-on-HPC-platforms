"""Leakage-safe fixed-epoch models for reduction cross-validation."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qnlp_hpc.trec_pca_vqc.experiment import calculate_metrics
from qnlp_hpc.trec_pca_vqc.model import AngleEncodedVQC


def set_fold_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fold_logistic_regression(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Train and evaluate the classical fold baseline."""
    model = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        class_weight="balanced",
    )

    start = time.perf_counter()
    model.fit(training_features, training_labels)
    training_seconds = time.perf_counter() - start

    predictions = model.predict(validation_features)
    metrics = calculate_metrics(validation_labels, predictions)

    return {
        "model": model,
        "training_seconds": training_seconds,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def evaluate_fold_vqc(
    model: AngleEncodedVQC,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the fixed-epoch model on an untouched outer fold."""
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    logits_parts: list[torch.Tensor] = []

    model.eval()

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            total_loss += float(loss.item()) * len(batch_labels)
            logits_parts.append(logits.cpu())

    logits = torch.cat(logits_parts, dim=0)
    predictions = torch.argmax(logits, dim=1).numpy()
    metrics = calculate_metrics(labels.numpy(), predictions)

    return {
        "loss": total_loss / len(labels),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def run_fixed_epoch_vqc(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    n_layers: int = 2,
    epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device_name: str = "cpu",
    measurement_mode: str = "z",
    data_reuploading: bool = False,
) -> dict[str, object]:
    """Train a VQC for a fixed budget and evaluate once on the outer fold."""
    set_fold_seed(seed)

    n_qubits = training_features.shape[1]
    device = torch.device(device_name)

    training_tensor = torch.tensor(
        training_features,
        dtype=torch.float32,
    )
    training_label_tensor = torch.tensor(
        training_labels,
        dtype=torch.long,
    )
    validation_tensor = torch.tensor(
        validation_features,
        dtype=torch.float32,
    )
    validation_label_tensor = torch.tensor(
        validation_labels,
        dtype=torch.long,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        TensorDataset(training_tensor, training_label_tensor),
        batch_size=min(batch_size, len(training_tensor)),
        shuffle=True,
        generator=generator,
    )

    model = AngleEncodedVQC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=6,
        seed=seed,
        measurement_mode=measurement_mode,
        data_reuploading=data_reuploading,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    class_counts = np.bincount(
        training_labels,
        minlength=6,
    )

    if np.any(class_counts == 0):
        raise ValueError(
            f"Training fold is missing a class: {class_counts.tolist()}"
        )

    class_weights = (
            len(training_labels)
            / (6.0 * class_counts)
    )

    class_weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
    )

    print(
        "    training class weights:",
        np.round(class_weights, 4).tolist(),
    )

    start = time.perf_counter()
    final_training_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_labels)

        final_training_loss = total_loss / len(training_tensor)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:03d}/{epochs} "
                f"training_loss={final_training_loss:.4f}"
            )

    training_seconds = time.perf_counter() - start

    validation_result = evaluate_fold_vqc(
        model,
        validation_tensor,
        validation_label_tensor,
        batch_size=batch_size,
        device=device,
    )

    return {
        "model": model,
        "training_seconds": training_seconds,
        "final_training_loss": final_training_loss,
        "validation_loss": validation_result["loss"],
        "accuracy": validation_result["accuracy"],
        "macro_f1": validation_result["macro_f1"],
        "confusion_matrix": validation_result["confusion_matrix"],
    }