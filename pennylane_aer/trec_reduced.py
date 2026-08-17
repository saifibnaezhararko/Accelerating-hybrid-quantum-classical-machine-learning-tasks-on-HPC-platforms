"""TREC through dimensionality reduction into a narrow PennyLane circuit.

The lambeq route ties circuit width to sentence length, which is why it was
demonstrated on MC1's 3-5 word sentences: TREC questions run to 37 words, and a
qubit per pregroup type is not reachable.  This module takes the other route
the project methodology describes - *embedding -> quantum layer -> classifier* - and
applies it to the TREC question-type dataset as shipped:

    question -> TF-IDF / bag-of-words / BERT -> TSVD | PCA | UMAP -> n_qubits
             -> angle embedding -> entangling layers -> <Z_i> -> class logits

Circuit width becomes a hyperparameter in the 2-8 qubit range rather than a
function of sentence length, and nothing is post-selected, so shot-based
simulation and real hardware both work.

TREC is single-sentence multi-class classification, so unlike the MC1 pair task
the classifier reads one circuit output per example.  The 6-way coarse label is
the default; ``--label-column label-fine`` selects the 47-way task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

EMBEDDINGS = ("count", "tfidf", "bert")
REDUCERS = ("tsvd", "pca", "umap")
SCALINGS = ("global", "per-component")
LABEL_COLUMNS = ("label-coarse", "label-fine")

TOKEN_PATTERN = r"[a-z0-9']+"
DEFAULT_BERT_MODEL = "bert-base-uncased"

# The six coarse TREC categories, in this file's label order.  Verified against
# the questions themselves rather than assumed: label 0 is "What are liver
# enzymes ?" (DESC), 2 is "What does INRI stand for ?" (ABBR), 4 is "When was
# Ozzy Osbourne born ?" (NUM) and 5 is "What is the highest waterfall ?" (LOC).
# This is *not* the ABBR-first order some TREC distributions use, and
# test_class_names_match_the_question_content pins it against the data.
COARSE_CLASS_NAMES = ("DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC")


# --------------------------------------------------------------------------
# Loading TREC as shipped
# --------------------------------------------------------------------------


def load_trec(
    directory: Path,
    label_column: str = "label-coarse",
    max_words: int | None = None,
    classes: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``train.csv`` / ``test.csv`` from the raw TREC folder.

    The official split is kept: TREC ships a 5,452/500 train/test division and
    published numbers are quoted against it.  ``max_words`` and ``classes`` are
    optional scaling knobs and are off by default, so the dataset is used as
    distributed unless a study asks otherwise.
    """
    if label_column not in LABEL_COLUMNS:
        raise ValueError(f"label_column must be one of {LABEL_COLUMNS}, got {label_column!r}.")

    frames = []
    for name in ("train", "test"):
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {path}\n"
                "Point --data-dir at the folder holding TREC's train.csv and test.csv."
            )
        frame = pd.read_csv(path)
        missing = {"text", label_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

        frame = frame.rename(columns={label_column: "label"})[["text", "label"]]
        frame["text"] = frame["text"].astype(str).str.strip()
        frame = frame[frame["text"].str.len() > 0]

        if max_words is not None:
            frame = frame[frame["text"].str.split().str.len() <= max_words]
        if classes is not None:
            frame = frame[frame["label"].isin(list(classes))]

        if frame.empty:
            raise ValueError(f"{path.name} has no rows left after filtering.")
        frames.append(frame.reset_index(drop=True))

    train, test = frames

    # Labels must be a contiguous 0..k-1 range for cross-entropy; after a
    # `classes` filter the original ids are not, so remap and keep the mapping.
    present = sorted(set(train["label"]).union(test["label"]))
    remap = {original: index for index, original in enumerate(present)}
    train["label"] = train["label"].map(remap)
    test["label"] = test["label"].map(remap)

    unseen = set(test["label"]) - set(train["label"])
    if unseen:
        raise ValueError(f"Test split contains classes absent from training: {sorted(unseen)}.")

    train.attrs["label_remap"] = remap
    test.attrs["label_remap"] = remap
    return train, test


def class_names(remap: dict[int, int], label_column: str) -> list[str]:
    """Human-readable class names in remapped label order."""
    if label_column != "label-coarse":
        return [f"fine-{original}" for original in sorted(remap, key=remap.get)]
    return [
        COARSE_CLASS_NAMES[original] if original < len(COARSE_CLASS_NAMES) else str(original)
        for original in sorted(remap, key=remap.get)
    ]


def stratified_split(
    frame: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a stratified fraction, preserving the class proportions.

    TREC's coarse classes are badly unbalanced (ABBR is 86 of 5,452 training
    rows), so a uniform random split can leave a class unrepresented in
    development and make model selection blind to it.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")

    generator = np.random.default_rng(seed)
    held_out_positions: list[int] = []
    for label in sorted(frame["label"].unique()):
        positions = np.flatnonzero(frame["label"].to_numpy() == label)
        generator.shuffle(positions)
        take = max(1, int(round(len(positions) * fraction)))
        take = min(take, len(positions) - 1)  # never empty a class out of training
        held_out_positions.extend(positions[:take].tolist())

    mask = np.zeros(len(frame), dtype=bool)
    mask[held_out_positions] = True
    return (
        frame.loc[~mask].reset_index(drop=True),
        frame.loc[mask].reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Embedding + dimensionality reduction
# --------------------------------------------------------------------------


def _bert_embeddings(sentences: Sequence[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Mean-pooled final hidden states.  Requires a downloadable checkpoint."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - transformers ships with lambeq
        raise ImportError("embedding='bert' requires transformers.") from exc

    try:
        tokeniser = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{model_name}'. BERT embeddings need the checkpoint in the "
            "HuggingFace cache or network access; use --embedding tfidf offline."
        ) from exc

    model.eval()
    pooled_batches = []
    for start in range(0, len(sentences), batch_size):
        chunk = list(sentences[start : start + batch_size])
        encoded = tokeniser(chunk, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled_batches.append(
            ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).numpy()
        )
    return np.concatenate(pooled_batches).astype(np.float64)


class SentenceAngleEncoder:
    """Fit on training questions only; map any question to ``n_qubits`` angles.

    Everything stateful - vocabulary, reduction basis, angle scale - is fitted
    in :meth:`fit` and reused in :meth:`transform`.  Fitting on the held-out
    questions would leak the test distribution into the feature map.
    """

    def __init__(
        self,
        n_qubits: int,
        embedding: str = "tfidf",
        reducer: str = "tsvd",
        scaling: str = "global",
        seed: int = 0,
        bert_model: str = DEFAULT_BERT_MODEL,
        min_document_frequency: int = 1,
    ) -> None:
        if embedding not in EMBEDDINGS:
            raise ValueError(f"embedding must be one of {EMBEDDINGS}, got {embedding!r}.")
        if reducer not in REDUCERS:
            raise ValueError(f"reducer must be one of {REDUCERS}, got {reducer!r}.")
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}, got {scaling!r}.")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}.")

        self.n_qubits = n_qubits
        self.embedding = embedding
        self.reducer = reducer
        self.scaling = scaling
        self.seed = seed
        self.bert_model = bert_model
        self.min_document_frequency = min_document_frequency

        self._vectoriser: Any = None
        self._reducer: Any = None
        self._needs_dense = False
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | float | None = None
        self.explained_variance_ratio: float | None = None
        self.vocabulary_size: int | None = None

    # -- embedding ---------------------------------------------------------

    def _embed(self, sentences: Sequence[str], fit: bool) -> Any:
        """Sparse for the count-based embeddings, dense for BERT.

        TREC's TF-IDF matrix is 5,452 x 8,460; densifying it costs ~370 MB for
        no benefit, and TruncatedSVD consumes sparse input directly.
        """
        if self.embedding == "bert":
            return _bert_embeddings(sentences, self.bert_model)

        if fit:
            if self.embedding == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectoriser = TfidfVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    lowercase=True,
                    min_df=self.min_document_frequency,
                )
            else:
                from sklearn.feature_extraction.text import CountVectorizer

                self._vectoriser = CountVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    lowercase=True,
                    binary=True,
                    min_df=self.min_document_frequency,
                )
            matrix = self._vectoriser.fit_transform(list(sentences))
            self.vocabulary_size = len(self._vectoriser.vocabulary_)
        else:
            matrix = self._vectoriser.transform(list(sentences))
        return matrix

    # -- reduction ---------------------------------------------------------

    def _build_reducer(self) -> tuple[Any, bool]:
        """Return the reducer and whether it needs dense input."""
        if self.reducer == "tsvd":
            from sklearn.decomposition import TruncatedSVD

            return TruncatedSVD(n_components=self.n_qubits, random_state=self.seed), False
        if self.reducer == "pca":
            from sklearn.decomposition import PCA

            return PCA(n_components=self.n_qubits, random_state=self.seed), True

        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "reducer='umap' requires umap-learn (`pip install umap-learn`); "
                "tsvd and pca need only scikit-learn."
            ) from exc
        return umap.UMAP(n_components=self.n_qubits, random_state=self.seed), True

    @staticmethod
    def _densify(matrix: Any) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            return matrix.toarray().astype(np.float64)
        return np.asarray(matrix, dtype=np.float64)

    # -- public API --------------------------------------------------------

    def fit(self, sentences: Sequence[str]) -> SentenceAngleEncoder:
        features = self._embed(sentences, fit=True)
        self._reducer, self._needs_dense = self._build_reducer()
        if self._needs_dense:
            features = self._densify(features)

        if self.n_qubits > min(features.shape):
            raise ValueError(
                f"Cannot reduce to {self.n_qubits} components from a "
                f"{features.shape[0]}x{features.shape[1]} training matrix."
            )

        reduced = np.asarray(self._reducer.fit_transform(features), dtype=np.float64)

        ratio = getattr(self._reducer, "explained_variance_ratio_", None)
        self.explained_variance_ratio = float(np.sum(ratio)) if ratio is not None else None

        self._centre = reduced.mean(axis=0)
        centred = reduced - self._centre

        if self.scaling == "global":
            # One scalar for every component, so the variance ordering the
            # reducer produced survives into the angles.  Scaling each
            # component to unit variance instead amplifies the low-variance
            # components up to the amplitude of the leading one, and the
            # circuit then sees noise and signal at equal strength.
            self._scale = float(np.abs(centred).max()) or 1.0
        else:
            self._scale = centred.std(axis=0) + 1e-12

        return self

    def transform(self, sentences: Sequence[str]) -> np.ndarray:
        if self._reducer is None or self._centre is None or self._scale is None:
            raise RuntimeError("SentenceAngleEncoder.fit must be called before transform.")

        features = self._embed(sentences, fit=False)
        if self._needs_dense:
            features = self._densify(features)

        reduced = np.asarray(self._reducer.transform(features), dtype=np.float64)
        # tanh bounds the angles to (-pi, pi) so an unseen question can never
        # wrap around the Bloch sphere and alias onto a different question.
        return np.pi * np.tanh((reduced - self._centre) / self._scale)

    def fit_transform(self, sentences: Sequence[str]) -> np.ndarray:
        return self.fit(sentences).transform(sentences)


# --------------------------------------------------------------------------
# Split container
# --------------------------------------------------------------------------


@dataclass
class SentenceSplit:
    """One split: angles for every question and its class label."""

    name: str
    sentences: list[str]
    angles: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def label_counts(self) -> dict[int, int]:
        values, counts = torch.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=True)}


def build_split(name: str, frame: pd.DataFrame, encoder: SentenceAngleEncoder) -> SentenceSplit:
    sentences = frame["text"].astype(str).tolist()
    return SentenceSplit(
        name=name,
        sentences=sentences,
        angles=torch.tensor(encoder.transform(sentences), dtype=torch.float32),
        labels=torch.tensor(frame["label"].to_numpy(), dtype=torch.long),
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def build_sentence_circuit(
    n_qubits: int,
    n_layers: int,
    reuploads: int = 1,
    backend_config: dict[str, Any] | None = None,
    diff_method: str = "best",
) -> tuple[Callable[..., Any], Any, tuple[int, ...]]:
    """A re-uploading angle-embedding circuit and its weight shape.

    ``reuploads`` re-applies the angle embedding between entangling blocks.
    Re-uploading raises the expressivity of a narrow circuit without spending
    extra qubits, which is the resource that actually binds here.
    """
    import pennylane as qml

    if backend_config is None:
        device = qml.device("default.qubit", wires=n_qubits)
    else:
        config = dict(backend_config)
        plugin = config.pop("backend")
        aer_backend = config.pop("device", None)
        if aer_backend is not None:
            config["backend"] = aer_backend
        device = qml.device(plugin, wires=n_qubits, **config)

    layers_per_block, extra = divmod(n_layers, reuploads)
    if layers_per_block < 1:
        raise ValueError(f"n_layers={n_layers} cannot be split across {reuploads} re-uploads.")
    block_sizes = [layers_per_block + (1 if i < extra else 0) for i in range(reuploads)]

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        start = 0
        for size in block_sizes:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[start : start + size], wires=range(n_qubits))
            start += size
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
    return circuit, device, weight_shape


class ClassifierHead(torch.nn.Module):
    """The same head for every model, so only the bottleneck differs."""

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        hidden_dim: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class _SentenceClassifier(torch.nn.Module):
    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        return self.head(self.sentence_features(angles))


class QuantumSentenceClassifier(_SentenceClassifier):
    """reduced angles -> PennyLane circuit -> <Z_i> -> class logits."""

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        reuploads: int = 1,
        hidden_dim: int = 16,
        dropout: float = 0.05,
        backend_config: dict[str, Any] | None = None,
        diff_method: str = "best",
        broadcast: bool | None = None,
    ) -> None:
        super().__init__()
        import pennylane as qml

        self.n_qubits = n_qubits
        circuit, self.device, weight_shape = build_sentence_circuit(
            n_qubits, n_layers, reuploads, backend_config, diff_method
        )
        self.quantum = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})
        self.head = ClassifierHead(n_qubits, n_classes, hidden_dim, dropout)

        # default.qubit differentiates by backprop through the simulator and
        # handles a broadcast batch.  External simulators fall back to
        # parameter-shift, which cannot differentiate a broadcasted tape
        # (PennyLane #4462), so they must be fed one question at a time.
        self.broadcast = backend_config is None if broadcast is None else broadcast

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        if self.broadcast:
            features = self.quantum(angles)
        else:
            features = torch.stack([self.quantum(row) for row in angles])
        if isinstance(features, (list, tuple)):
            features = torch.stack(features, dim=-1)
        return features.reshape(angles.shape[0], self.n_qubits).to(dtype=angles.dtype)


class ClassicalSentenceClassifier(_SentenceClassifier):
    """Control: the circuit replaced by a bounded map of the same shape.

    ``Linear(q, q) -> tanh`` matches the circuit's input and output width and
    is close to it in parameter count, so a difference in accuracy is
    attributable to the layer rather than to capacity elsewhere.
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        hidden_dim: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.bottleneck = torch.nn.Linear(n_qubits, n_qubits)
        self.head = ClassifierHead(n_qubits, n_classes, hidden_dim, dropout)

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.bottleneck(angles))


class NoBottleneckSentenceClassifier(_SentenceClassifier):
    """Second control: no learned per-question layer at all.

    Separates "the quantum layer helps" from "the reduced features and the
    classifier head were already sufficient".
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        hidden_dim: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.head = ClassifierHead(n_qubits, n_classes, hidden_dim, dropout)

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.tanh(angles)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        matrix[true, predicted] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    """Unweighted mean per-class F1.

    TREC's coarse classes span 86 to 1,250 training rows, so plain accuracy is
    dominated by the large classes; macro-F1 shows whether the rare ones are
    predicted at all.
    """
    scores = []
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        predicted = matrix[:, index].sum()
        actual = matrix[index, :].sum()
        if actual == 0:
            continue
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module, split: SentenceSplit, n_classes: int, batch_size: int = 1024
) -> dict[str, Any]:
    model.eval()
    logits = torch.cat(
        [
            model(split.angles[start : start + batch_size])
            for start in range(0, len(split), batch_size)
        ]
    )
    predictions = logits.argmax(dim=1)
    matrix = confusion_matrix(predictions, split.labels, n_classes)
    return {
        "accuracy": float((predictions == split.labels).float().mean()),
        "loss": float(torch.nn.functional.cross_entropy(logits, split.labels)),
        "macro_f1": macro_f1(matrix),
        "confusion": matrix.tolist(),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_model(
    model: torch.nn.Module,
    train: SentenceSplit,
    development: SentenceSplit,
    n_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    patience: int | None = None,
    verbose_every: int = 0,
) -> dict[str, Any]:
    """Train with dev-loss model selection, as the MC1 experiments did."""
    import time

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    n_train = len(train)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_train)
        batch_losses = []
        for offset in range(0, n_train, batch_size):
            batch = permutation[offset : offset + batch_size]
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(train.angles[batch]), train.labels[batch]
            )
            loss.backward()
            optimiser.step()
            batch_losses.append(float(loss.detach()))

        scores = evaluate_split(model, development, n_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "development_loss": scores["loss"],
                "development_accuracy": scores["accuracy"],
                "development_macro_f1": scores["macro_f1"],
            }
        )

        if scores["loss"] < best_loss:
            best_loss = scores["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            print(
                f"      epoch {epoch:3d}/{epochs}  train_loss={history[-1]['train_loss']:.4f}  "
                f"dev_loss={scores['loss']:.4f}  dev_acc={scores['accuracy']:.3f}"
            )
        if patience and epoch - best_epoch >= patience:
            break

    seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "seconds": seconds,
        "epochs_run": len(history),
        "seconds_per_epoch": seconds / max(1, len(history)),
        "selected_epoch": best_epoch,
        "best_development_loss": best_loss,
        "history": history,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def mean_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean, standard deviation and a Student-t interval, as sweep.py reports."""
    array = np.asarray(list(values), dtype=float)
    n = int(array.size)
    mean = float(array.mean()) if n else float("nan")
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "std": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "half_width": 0.0,
        }

    from scipy import stats

    std = float(array.std(ddof=1))
    half_width = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "half_width": half_width,
    }
