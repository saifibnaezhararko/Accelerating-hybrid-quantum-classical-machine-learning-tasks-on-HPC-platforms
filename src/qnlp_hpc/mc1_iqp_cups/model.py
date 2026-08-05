"""Define the IQP sentence-pair model, loss, and evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lambeq import PytorchQuantumModel

from qnlp_hpc.mc1_iqp_cups.config import (
    CIRCUIT_OUTPUT_DIM,
    CLASSIFIER_DROPOUT,
    CLASSIFIER_HIDDEN_DIM,
)


class IQPPairModel(PytorchQuantumModel):
    """Symmetric classifier over pairs of IQP sentence circuits."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        # A two-qubit circuit produces four probabilities. Concatenating the
        # absolute difference and elementwise product gives eight pair features.
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

        # Tensor contraction commonly returns float64, while Linear parameters
        # default to float32. Match dtypes without detaching the gradient graph.
        classifier_parameter = next(self.classifier.parameters())
        pair_features = pair_features.to(
            device=classifier_parameter.device,
            dtype=classifier_parameter.dtype,
        )

        return self.classifier(pair_features)


class LambeqCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy accepting class IDs or one-hot lambeq targets."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(logits, targets.long())


def class_ids(targets: torch.Tensor) -> torch.Tensor:
    """Convert class-ID or one-hot targets to integer class IDs."""
    if targets.ndim > 1:
        return torch.argmax(targets, dim=1).long()
    return targets.long()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy for a batch."""
    predictions = torch.argmax(logits, dim=1)
    expected = class_ids(targets)
    return float((predictions == expected).float().mean().item())


def evaluation_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return detached cross-entropy for evaluation and early stopping."""
    expected = class_ids(targets)
    return float(F.cross_entropy(logits, expected).detach().cpu().item())


def prediction_counts(
    model: torch.nn.Module,
    pairs: Sequence[tuple[object, object]],
) -> dict[int, int]:
    """Count each class predicted by ``model`` for a collection of pairs."""
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(pairs), dim=1).detach().cpu().numpy()

    labels, counts = np.unique(predictions, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}
