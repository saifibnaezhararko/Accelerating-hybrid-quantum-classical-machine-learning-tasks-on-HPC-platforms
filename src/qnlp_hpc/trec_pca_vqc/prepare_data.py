"""TREC data loading and preprocessing for the PCA-VQC experiment."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

TREC_CLASSES = ("ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM")
LABEL_TO_ID = {label: index for index, label in enumerate(TREC_CLASSES)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase text and collapse repeated whitespace."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def load_trec(path: Path) -> pd.DataFrame:
    """Read a standard TREC .label file.

    Expected format:

        DESC:manner How did serfdom develop in Russia ?
    """
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"TREC file not found: {source}")

    rows: list[dict[str, object]] = []

    with source.open("r", encoding="latin-1") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                label_token, question = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{line_number} does not contain a question."
                ) from exc

            coarse_label, separator, fine_label = label_token.partition(":")

            if not separator:
                raise ValueError(
                    f"{source}:{line_number} has no coarse:fine label: "
                    f"{label_token!r}"
                )

            if coarse_label not in LABEL_TO_ID:
                raise ValueError(
                    f"{source}:{line_number} has unknown coarse label "
                    f"{coarse_label!r}."
                )

            cleaned_question = clean_text(question)

            if not cleaned_question:
                raise ValueError(
                    f"{source}:{line_number} contains an empty question."
                )

            rows.append(
                {
                    "text": cleaned_question,
                    "coarse_label": coarse_label,
                    "fine_label": fine_label,
                    "label": LABEL_TO_ID[coarse_label],
                    "source_line": line_number,
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError(f"TREC file contains no examples: {source}")

    return frame


def stratified_sample(
    frame: pd.DataFrame,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Select the same number of examples from every coarse class."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than zero.")

    sampled_groups: list[pd.DataFrame] = []

    for class_name in TREC_CLASSES:
        class_rows = frame[frame["coarse_label"] == class_name]

        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Class {class_name} contains only {len(class_rows)} examples; "
                f"requested {samples_per_class}."
            )

        sampled_groups.append(
            class_rows.sample(
                n=samples_per_class,
                random_state=seed,
                replace=False,
            )
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)

    return sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def build_splits(
    training_path: Path,
    test_path: Path,
    samples_per_class: int = 50,
    development_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build sampled training/development splits and retain official test data."""
    if not 0.0 < development_ratio < 1.0:
        raise ValueError("development_ratio must be between zero and one.")

    full_training = load_trec(training_path)
    official_test = load_trec(test_path)

    sampled = stratified_sample(
        full_training,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    training, development = train_test_split(
        sampled,
        test_size=development_ratio,
        random_state=seed,
        shuffle=True,
        stratify=sampled["label"],
    )

    training = training.reset_index(drop=True)
    development = development.reset_index(drop=True)
    official_test = official_test.reset_index(drop=True)

    return training, development, official_test


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return counts in the fixed TREC class order."""
    counts = frame["coarse_label"].value_counts()

    return {
        class_name: int(counts.get(class_name, 0))
        for class_name in TREC_CLASSES
    }

def encode_texts(
    frame: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode questions as normalized 384-dimensional MiniLM embeddings."""
    embeddings = model.encode(
        frame["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def reduce_and_scale_features(
    training_embeddings: np.ndarray,
    development_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    n_components: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PCA,
    MinMaxScaler,
]:
    """Fit PCA and angle scaling using training data only."""
    if n_components <= 0:
        raise ValueError("n_components must be greater than zero.")

    maximum_components = min(training_embeddings.shape)

    if n_components > maximum_components:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum "
            f"possible value {maximum_components}."
        )

    pca = PCA(n_components=n_components)

    training_reduced = pca.fit_transform(training_embeddings)
    development_reduced = pca.transform(development_embeddings)
    test_reduced = pca.transform(test_embeddings)

    # Fit only on training data. Values outside the training range are clipped.
    angle_scaler = MinMaxScaler(
        feature_range=(-np.pi, np.pi),
        clip=True,
    )

    training_angles = angle_scaler.fit_transform(training_reduced)
    development_angles = angle_scaler.transform(development_reduced)
    test_angles = angle_scaler.transform(test_reduced)

    return (
        np.asarray(training_angles, dtype=np.float32),
        np.asarray(development_angles, dtype=np.float32),
        np.asarray(test_angles, dtype=np.float32),
        pca,
        angle_scaler,
    )


def prepare_features(
    training: pd.DataFrame,
    development: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int = 8,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, object]:
    """Create MiniLM embeddings and PCA angle features for all splits."""
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SentenceTransformer: {model_name}")
    print(f"Embedding device: {selected_device}")

    encoder = SentenceTransformer(
        model_name,
        device=selected_device,
    )

    print("Encoding training questions...")
    training_embeddings = encode_texts(training, encoder, batch_size)

    print("Encoding development questions...")
    development_embeddings = encode_texts(development, encoder, batch_size)

    print("Encoding test questions...")
    test_embeddings = encode_texts(test, encoder, batch_size)

    (
        training_angles,
        development_angles,
        test_angles,
        pca,
        angle_scaler,
    ) = reduce_and_scale_features(
        training_embeddings,
        development_embeddings,
        test_embeddings,
        n_components=n_components,
    )

    retained_variance = float(pca.explained_variance_ratio_.sum())

    print(f"Embedding dimension: {training_embeddings.shape[1]}")
    print(f"PCA components: {n_components}")
    print(f"Retained training variance: {retained_variance:.4%}")

    return {
        "training_embeddings": training_embeddings,
        "development_embeddings": development_embeddings,
        "test_embeddings": test_embeddings,
        "training_angles": training_angles,
        "development_angles": development_angles,
        "test_angles": test_angles,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "retained_variance": retained_variance,
        "embedding_model": model_name,
        "embedding_device": selected_device,
    }
