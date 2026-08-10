"""Fold-local dimensionality reduction for TREC experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import MinMaxScaler

SUPPORTED_METHODS = ("pca", "lda", "nca")


@dataclass
class ReducedFold:
    """Features and fitted preprocessing objects for one CV fold."""

    method: str
    training_angles: np.ndarray
    validation_angles: np.ndarray
    reducer: object
    angle_scaler: MinMaxScaler
    explained_variance: float | None


def create_reducer(
    method: str,
    n_components: int = 5,
    seed: int = 42,
) -> object:
    """Construct a dimensionality reducer without fitting it."""
    normalised_method = method.lower()

    if normalised_method == "pca":
        return PCA(
            n_components=n_components,
            random_state=seed,
        )

    if normalised_method == "lda":
        return LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage="auto",
            n_components=n_components,
        )

    if normalised_method == "nca":
        return NeighborhoodComponentsAnalysis(
            n_components=n_components,
            init="pca",
            max_iter=200,
            tol=1e-5,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown reduction method {method!r}. "
        f"Supported methods: {SUPPORTED_METHODS}."
    )


def fit_transform_fold(
    method: str,
    training_embeddings: np.ndarray,
    training_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    n_components: int = 5,
    seed: int = 42,
) -> ReducedFold:
    """Fit reduction and angle scaling on one fold's training data only."""
    if training_embeddings.ndim != 2:
        raise ValueError("training_embeddings must be two-dimensional.")

    if validation_embeddings.ndim != 2:
        raise ValueError("validation_embeddings must be two-dimensional.")

    if len(training_embeddings) != len(training_labels):
        raise ValueError(
            "Training embeddings and labels have different lengths."
        )

    unique_labels = np.unique(training_labels)

    if method.lower() == "lda":
        maximum_lda_components = len(unique_labels) - 1

        if n_components > maximum_lda_components:
            raise ValueError(
                f"LDA with {len(unique_labels)} classes supports at most "
                f"{maximum_lda_components} components."
            )

    reducer = create_reducer(
        method=method,
        n_components=n_components,
        seed=seed,
    )

    # Every reducer is fitted only on the current fold's training partition.
    training_reduced = reducer.fit_transform(
        training_embeddings,
        training_labels,
    )
    validation_reduced = reducer.transform(validation_embeddings)

    # Angle scaling is also fitted only on the fold's training partition.
    angle_scaler = MinMaxScaler(
        feature_range=(-np.pi, np.pi),
        clip=True,
    )

    training_angles = angle_scaler.fit_transform(training_reduced)
    validation_angles = angle_scaler.transform(validation_reduced)

    explained_variance: float | None = None

    if isinstance(reducer, PCA):
        explained_variance = float(
            reducer.explained_variance_ratio_.sum()
        )

    return ReducedFold(
        method=method.lower(),
        training_angles=np.asarray(
            training_angles,
            dtype=np.float32,
        ),
        validation_angles=np.asarray(
            validation_angles,
            dtype=np.float32,
        ),
        reducer=reducer,
        angle_scaler=angle_scaler,
        explained_variance=explained_variance,
    )