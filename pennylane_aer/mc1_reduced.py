"""MC1 through dimensionality reduction into a narrow PennyLane circuit.

The lambeq route ties circuit width to sentence length: MC1 at
``SENTENCE_QUBITS=2`` needs 14-18 qubits, of which 12-16 are post-selected, so
shot-based simulation and real hardware are both ruled out (see README traps 1
and 2).  This module takes the other route CLAUDE.md section 3 describes -
*embedding -> quantum layer -> classifier* - and applies it to MC1 rather than
TREC:

    sentence -> bag-of-words / TF-IDF / BERT -> PCA | TSVD | UMAP -> n_qubits
             -> angle embedding -> entangling layers -> <Z_i>

Circuit width becomes a hyperparameter in the 3-8 qubit range the NLP lead
targets, and nothing is post-selected.

MC1 is a *pair* task, so per-sentence features are combined with the same
swap-invariant features Evelyn's ``IQPPairModel`` uses (``|l-r|``, ``l*r``),
keeping the classifier head comparable.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

EMBEDDINGS = ("count", "tfidf", "bert")
REDUCERS = ("pca", "tsvd", "umap")
SCALINGS = ("global", "per-component")

TOKEN_PATTERN = r"[a-z']+"
DEFAULT_BERT_MODEL = "bert-base-uncased"


# --------------------------------------------------------------------------
# Topic labels, derived from the dataset rather than assumed
# --------------------------------------------------------------------------


def derive_sentence_topics(frame: pd.DataFrame) -> dict[str, int]:
    """Recover a per-sentence topic id from MC1's pairwise labels.

    MC1 labels a pair 1 when both sentences share a topic and 0 when they do
    not, which makes the sentence graph a 2-colouring problem: label-1 edges
    force equal colours, label-0 edges force opposite ones.  A consistent
    colouring is a per-sentence topic assignment that no external lexicon had
    to supply.

    Raises ``ValueError`` if the colouring is contradictory, or if it fails to
    reproduce any label in ``frame`` - so a dataset where this framing does not
    hold fails loudly instead of silently producing wrong topics.
    """
    adjacency: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        same_topic = int(row.label) == 1
        adjacency[row.sentence_1].append((row.sentence_2, same_topic))
        adjacency[row.sentence_2].append((row.sentence_1, same_topic))

    sentences = sorted(set(frame["sentence_1"]).union(frame["sentence_2"]))
    topics: dict[str, int] = {}

    for start in sentences:
        if start in topics:
            continue
        topics[start] = 0
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour, same_topic in adjacency[current]:
                expected = topics[current] if same_topic else 1 - topics[current]
                if neighbour not in topics:
                    topics[neighbour] = expected
                    stack.append(neighbour)
                elif topics[neighbour] != expected:
                    raise ValueError(
                        "MC1 labels are not consistent with a two-topic assignment: "
                        f"{current!r} and {neighbour!r} cannot both be satisfied."
                    )

    disagreements = [
        (row.sentence_1, row.sentence_2, int(row.label))
        for row in frame.itertuples(index=False)
        if (topics[row.sentence_1] == topics[row.sentence_2]) != (int(row.label) == 1)
    ]
    if disagreements:
        raise ValueError(
            f"Derived topics disagree with {len(disagreements)} of {len(frame)} labels; "
            f"first: {disagreements[0]!r}"
        )

    return topics


def augmented_pairs(sentences: Sequence[str], topics: dict[str, int]) -> list[tuple[str, str, int]]:
    """Every unordered sentence pair drawn from ``sentences``, labelled by topic.

    Applied *inside* a split, this expands the supervision without crossing the
    sentence-disjoint boundary: MC1 ships 13 test pairs over 11 test sentences,
    where all 55 possible pairs carry a label the dataset already determined.
    """
    return [
        (left, right, int(topics[left] == topics[right]))
        for left, right in itertools.combinations(sorted(sentences), 2)
    ]


def frame_pairs(frame: pd.DataFrame) -> list[tuple[str, str, int]]:
    """The pairs MC1 actually ships for a split, in file order."""
    return [
        (row.sentence_1, row.sentence_2, int(row.label)) for row in frame.itertuples(index=False)
    ]


# --------------------------------------------------------------------------
# Embedding + dimensionality reduction
# --------------------------------------------------------------------------


def _bert_embeddings(sentences: Sequence[str], model_name: str) -> np.ndarray:
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
    encoded = tokeniser(list(sentences), padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.numpy().astype(np.float64)


class SentenceAngleEncoder:
    """Fit on training sentences only; map any sentence to ``n_qubits`` angles.

    Everything stateful - vocabulary, reduction basis, angle scale - is fitted
    in :meth:`fit` and reused in :meth:`transform`.  Fitting on the held-out
    sentences would leak the test distribution into the feature map, which on a
    dataset this small is enough to manufacture the result.
    """

    def __init__(
        self,
        n_qubits: int,
        embedding: str = "tfidf",
        reducer: str = "pca",
        scaling: str = "global",
        seed: int = 0,
        bert_model: str = DEFAULT_BERT_MODEL,
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

        self._vectoriser: Any = None
        self._reducer: Any = None
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | float | None = None
        self.explained_variance_ratio: float | None = None

    # -- embedding ---------------------------------------------------------

    def _embed(self, sentences: Sequence[str], fit: bool) -> np.ndarray:
        if self.embedding == "bert":
            return _bert_embeddings(sentences, self.bert_model)

        if fit:
            if self.embedding == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectoriser = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)
            else:
                from sklearn.feature_extraction.text import CountVectorizer

                self._vectoriser = CountVectorizer(
                    token_pattern=TOKEN_PATTERN, lowercase=True, binary=True
                )
            matrix = self._vectoriser.fit_transform(list(sentences))
        else:
            matrix = self._vectoriser.transform(list(sentences))
        return np.asarray(matrix.todense(), dtype=np.float64)

    # -- reduction ---------------------------------------------------------

    def _build_reducer(self, n_features: int) -> Any:
        if self.reducer == "pca":
            from sklearn.decomposition import PCA

            return PCA(n_components=self.n_qubits, random_state=self.seed)
        if self.reducer == "tsvd":
            from sklearn.decomposition import TruncatedSVD

            return TruncatedSVD(n_components=self.n_qubits, random_state=self.seed)

        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "reducer='umap' requires umap-learn (`pip install umap-learn`); "
                "pca and tsvd need only scikit-learn."
            ) from exc
        return umap.UMAP(n_components=self.n_qubits, random_state=self.seed)

    # -- public API --------------------------------------------------------

    def fit(self, sentences: Sequence[str]) -> SentenceAngleEncoder:
        features = self._embed(sentences, fit=True)
        if self.n_qubits > min(features.shape):
            raise ValueError(
                f"Cannot reduce to {self.n_qubits} components from a "
                f"{features.shape[0]}x{features.shape[1]} training matrix."
            )

        self._reducer = self._build_reducer(features.shape[1])
        reduced = np.asarray(self._reducer.fit_transform(features), dtype=np.float64)

        ratio = getattr(self._reducer, "explained_variance_ratio_", None)
        self.explained_variance_ratio = float(np.sum(ratio)) if ratio is not None else None

        self._centre = reduced.mean(axis=0)
        centred = reduced - self._centre

        if self.scaling == "global":
            # One scalar for every component, so the variance ordering the
            # reducer produced survives into the angles.  Scaling each
            # component to unit variance instead (`per-component`) amplifies
            # the low-variance components - which on MC1 carry syntax, not
            # topic - up to the amplitude of the topic-carrying component, and
            # the circuit then sees noise and signal at equal strength.
            self._scale = float(np.abs(centred).max()) or 1.0
        else:
            self._scale = centred.std(axis=0) + 1e-12

        return self

    def transform(self, sentences: Sequence[str]) -> np.ndarray:
        if self._reducer is None or self._centre is None or self._scale is None:
            raise RuntimeError("SentenceAngleEncoder.fit must be called before transform.")

        reduced = np.asarray(
            self._reducer.transform(self._embed(sentences, fit=False)), dtype=np.float64
        )
        # tanh bounds the angles to (-pi, pi) so an unseen sentence can never
        # wrap around the Bloch sphere and alias onto a different sentence.
        return np.pi * np.tanh((reduced - self._centre) / self._scale)

    def fit_transform(self, sentences: Sequence[str]) -> np.ndarray:
        return self.fit(sentences).transform(sentences)


# --------------------------------------------------------------------------
# Split containers
# --------------------------------------------------------------------------


@dataclass
class PairSplit:
    """One split: distinct sentences, their angles, and pairs as index arrays.

    Pairs are stored as indices into ``angles`` rather than as their own
    feature rows.  The quantum layer then costs one circuit evaluation per
    *distinct sentence*, not two per pair - 11 test sentences cover all 55 of
    their pairs.
    """

    name: str
    sentences: list[str]
    angles: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def label_counts(self) -> dict[int, int]:
        values, counts = torch.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=True)}


def build_split(
    name: str,
    pairs: Sequence[tuple[str, str, int]],
    encoder: SentenceAngleEncoder,
    sentences: Sequence[str] | None = None,
) -> PairSplit:
    ordered = sorted({s for pair in pairs for s in pair[:2]} if sentences is None else sentences)
    index = {sentence: position for position, sentence in enumerate(ordered)}
    return PairSplit(
        name=name,
        sentences=ordered,
        angles=torch.tensor(encoder.transform(ordered), dtype=torch.float32),
        left=torch.tensor([index[left] for left, _, _ in pairs], dtype=torch.long),
        right=torch.tensor([index[right] for _, right, _ in pairs], dtype=torch.long),
        labels=torch.tensor([label for _, _, label in pairs], dtype=torch.long),
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


class PairHead(torch.nn.Module):
    """Swap-invariant pair features, then Evelyn's classifier head.

    ``|l-r|`` and ``l*r`` are both symmetric under swapping the two sentences,
    so ``forward(a, b) == forward(b, a)`` holds by construction - MC1's label
    is a property of the pair, not of its order.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((torch.abs(left - right), left * right), dim=1))


class _PairClassifier(torch.nn.Module):
    """Shared plumbing: featurise distinct sentences once, then index pairs."""

    feature_dim: int

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, angles: torch.Tensor, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        used, inverse = torch.unique(torch.cat((left, right)), return_inverse=True)
        features = self.sentence_features(angles.index_select(0, used))
        split = left.shape[0]
        return self.head(features[inverse[:split]], features[inverse[split:]])


class QuantumPairClassifier(_PairClassifier):
    """reduced angles -> PennyLane circuit -> swap-invariant features -> head."""

    def __init__(
        self,
        n_qubits: int,
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
        self.feature_dim = n_qubits
        circuit, self.device, weight_shape = build_sentence_circuit(
            n_qubits, n_layers, reuploads, backend_config, diff_method
        )
        self.quantum = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})
        self.head = PairHead(n_qubits, hidden_dim, dropout)

        # default.qubit differentiates by backprop through the simulator and
        # handles a broadcast batch.  External simulators fall back to
        # parameter-shift, which cannot differentiate a broadcasted tape
        # (PennyLane #4462), so they must be fed one sentence at a time.
        self.broadcast = backend_config is None if broadcast is None else broadcast

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        if self.broadcast:
            features = self.quantum(angles)
        else:
            features = torch.stack([self.quantum(row) for row in angles])
        if isinstance(features, (list, tuple)):
            features = torch.stack(features, dim=-1)
        return features.reshape(angles.shape[0], self.n_qubits).to(dtype=angles.dtype)


class ClassicalPairClassifier(_PairClassifier):
    """Control: the circuit replaced by a bounded map of the same shape.

    ``Linear(q, q) -> tanh`` matches the circuit's input and output width and
    is close to it in parameter count, so a difference in accuracy is
    attributable to the layer rather than to capacity elsewhere in the model.
    """

    def __init__(
        self,
        n_qubits: int,
        hidden_dim: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_dim = n_qubits
        self.bottleneck = torch.nn.Linear(n_qubits, n_qubits)
        self.head = PairHead(n_qubits, hidden_dim, dropout)

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.bottleneck(angles))


class NoBottleneckPairClassifier(_PairClassifier):
    """Second control: no learned per-sentence layer at all.

    Separates "the quantum layer helps" from "the reduced features and the
    classifier head were already sufficient".
    """

    def __init__(self, n_qubits: int, hidden_dim: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.feature_dim = n_qubits
        self.head = PairHead(n_qubits, hidden_dim, dropout)

    def sentence_features(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.tanh(angles)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_split(model: torch.nn.Module, split: PairSplit) -> tuple[float, float]:
    model.eval()
    logits = model(split.angles, split.left, split.right)
    loss = float(torch.nn.functional.cross_entropy(logits, split.labels))
    accuracy = float((logits.argmax(dim=1) == split.labels).float().mean())
    return accuracy, loss


def train_pair_model(
    model: torch.nn.Module,
    train: PairSplit,
    development: PairSplit,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    patience: int | None = None,
    verbose_every: int = 0,
) -> dict[str, Any]:
    """Train with dev-loss model selection, mirroring the mc1_iqp_cups protocol."""
    import time

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    n_pairs = len(train)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_pairs)
        batch_losses = []
        for offset in range(0, n_pairs, batch_size):
            batch = permutation[offset : offset + batch_size]
            optimiser.zero_grad()
            logits = model(train.angles, train.left[batch], train.right[batch])
            loss = torch.nn.functional.cross_entropy(logits, train.labels[batch])
            loss.backward()
            optimiser.step()
            batch_losses.append(float(loss.detach()))

        development_accuracy, development_loss = evaluate_split(model, development)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "development_loss": development_loss,
                "development_accuracy": development_accuracy,
            }
        )

        if development_loss < best_loss:
            best_loss = development_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
            print(
                f"      epoch {epoch:3d}/{epochs}  train_loss={history[-1]['train_loss']:.4f}  "
                f"dev_loss={development_loss:.4f}  dev_acc={development_accuracy:.3f}"
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
