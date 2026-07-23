from typing import Sequence

import torch
import torch.nn.functional as F
from lambeq import PytorchModel

from qnlp_hpc.mc1_spider.config import (
    CLASSIFIER_DROPOUT,
    CLASSIFIER_HIDDEN_DIM,
    SENTENCE_DIM,
)


class SpiderPairModel(PytorchModel):
    """Symmetric Siamese classifier over two SpiderAnsatz sentence vectors."""

    def __init__(self) -> None:
        super().__init__()
        # Symmetric pair features used below:
        #   |left-right| -> SENTENCE_DIM
        #   left*right   -> SENTENCE_DIM
        # Therefore the classifier input width is 2*SENTENCE_DIM.
        pair_feature_dim = 2 * SENTENCE_DIM
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

        flat_diagrams = [
            diagram
            for pair in diagram_pairs
            for diagram in pair
        ]

        sentence_vectors = self.get_diagram_output(flat_diagrams)
        sentence_vectors = sentence_vectors.reshape(
            len(diagram_pairs),
            2,
            SENTENCE_DIM,
        )

        left = sentence_vectors[:, 0, :]
        right = sentence_vectors[:, 1, :]

        # These features are invariant when the sentence order is swapped.
        absolute_difference = torch.abs(left - right)
        elementwise_product = left * right

        pair_features = torch.cat(
            (
                absolute_difference,
                elementwise_product,
            ),
            dim=1,
        )

        expected_width = 2 * SENTENCE_DIM
        if pair_features.shape[1] != expected_width:
            raise RuntimeError(
                "Pair-feature dimension mismatch: "
                f"expected {expected_width}, got {pair_features.shape[1]}."
            )

        return self.classifier(pair_features)


class LambeqCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy that accepts lambeq float or one-hot label tensors."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.ndim > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(logits, targets.long())


def class_ids(y: torch.Tensor) -> torch.Tensor:
    """Return integer class IDs from integer, float, or one-hot targets."""
    if y.ndim > 1:
        return torch.argmax(y, dim=1).long()
    return y.long()


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    predicted = torch.argmax(y_hat, dim=1)
    expected = class_ids(y)
    return float((predicted == expected).float().mean().item())


def evaluation_loss(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Cross-entropy metric used by lambeq for validation early stopping.

    lambeq only accepts names registered in ``evaluate_functions`` as an
    ``early_stopping_criterion``.  Its internally recorded validation cost is
    not automatically exposed under the name ``"loss"``.
    """
    expected = class_ids(y)
    loss = F.cross_entropy(y_hat, expected)
    return float(loss.detach().cpu().item())
