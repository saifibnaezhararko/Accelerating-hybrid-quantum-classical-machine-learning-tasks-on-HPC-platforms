from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn.functional as F

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


@dataclass
class TrecDataset:

    train_sequences: torch.Tensor
    train_labels: torch.Tensor
    test_sequences: torch.Tensor
    test_labels: torch.Tensor
    vocabulary: dict[str, int]
    max_length: int

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def build_vocabulary(texts: list[str], min_frequency: int = 1) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenise(text))
    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_frequency:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(texts: list[str], vocabulary: dict[str, int], max_length: int) -> torch.Tensor:
    unknown = vocabulary[UNKNOWN_TOKEN]
    encoded = np.zeros((len(texts), max_length), dtype=np.int64)
    for row, text in enumerate(texts):
        ids = [vocabulary.get(token, unknown) for token in tokenise(text)][:max_length]
        encoded[row, : len(ids)] = ids
    return torch.from_numpy(encoded)


def load_trec(directory: Path, min_frequency: int = 1) -> TrecDataset:
    train_frame = pd.read_csv(directory / "train.csv")
    test_frame = pd.read_csv(directory / "test.csv")

    for name, frame in (("train", train_frame), ("test", test_frame)):
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {sorted(missing)}")

    train_texts = train_frame["text"].astype(str).tolist()
    test_texts = test_frame["text"].astype(str).tolist()

    vocabulary = build_vocabulary(train_texts, min_frequency=min_frequency)
    max_length = max(len(tokenise(text)) for text in train_texts + test_texts)

    return TrecDataset(
        train_sequences=encode(train_texts, vocabulary, max_length),
        train_labels=torch.tensor(train_frame["label"].to_numpy(), dtype=torch.long),
        test_sequences=encode(test_texts, vocabulary, max_length),
        test_labels=torch.tensor(test_frame["label"].to_numpy(), dtype=torch.long),
        vocabulary=vocabulary,
        max_length=max_length,
    )


class TextCNN(torch.nn.Module):

    def __init__(
        self,
        vocabulary_size: int,
        n_qubits: int,
        embedding_dim: int = 32,
        n_filters: int = 16,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(embedding_dim, n_filters, kernel_size=size, padding=size - 1)
                for size in kernel_sizes
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.project = torch.nn.Linear(n_filters * len(kernel_sizes), n_qubits)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequences).transpose(1, 2)
        pooled = [
            F.relu(convolution(embedded)).max(dim=2).values for convolution in self.convolutions
        ]
        features = self.dropout(torch.cat(pooled, dim=1))

        return torch.pi * torch.tanh(self.project(features))


def build_quantum_layer(
    n_qubits: int,
    n_layers: int,
    backend_config: dict[str, Any] | None = None,
    diff_method: str = "best",
    gradient_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, qml.devices.Device]:
    if backend_config is None:
        device = qml.device("default.qubit", wires=n_qubits)
    else:
        config = dict(backend_config)
        plugin = config.pop("backend")
        aer_backend = config.pop("device", None)
        if aer_backend is not None:
            config["backend"] = aer_backend
        device = qml.device(plugin, wires=n_qubits, **config)

    qnode_options: dict[str, Any] = {}
    if gradient_kwargs:
        # PennyLane <=0.41 accepted gradient-transform kwargs directly on
        # qml.qnode. From 0.42 onward they must be nested under
        # ``gradient_kwargs``. The GPU server is pinned to 0.40.
        version = tuple(int(part) for part in qml.__version__.split(".")[:2])
        if version >= (0, 42):
            qnode_options["gradient_kwargs"] = gradient_kwargs
        else:
            qnode_options.update(gradient_kwargs)

    @qml.qnode(device, interface="torch", diff_method=diff_method, **qnode_options)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes), device


class HybridTextClassifier(torch.nn.Module):

    def __init__(
        self,
        vocabulary_size: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        embedding_dim: int = 32,
        n_filters: int = 16,
        dropout: float = 0.3,
        backend_config: dict[str, Any] | None = None,
        diff_method: str = "best",
        gradient_kwargs: dict[str, Any] | None = None,
        broadcast: bool | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.cnn = TextCNN(
            vocabulary_size,
            n_qubits=n_qubits,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            dropout=dropout,
        )
        self.quantum, self.device = build_quantum_layer(
            n_qubits,
            n_layers,
            backend_config,
            diff_method,
            gradient_kwargs,
        )
        self.head = torch.nn.Linear(n_qubits, 2)

        self.broadcast = backend_config is None if broadcast is None else broadcast

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        angles = self.cnn(sequences)

        if self.broadcast:
            quantum_features = self.quantum(angles)
        else:
            quantum_features = torch.stack(
                [self.quantum(angles[row]) for row in range(angles.shape[0])]
            )

        if isinstance(quantum_features, (list, tuple)):
            quantum_features = torch.stack(quantum_features, dim=-1)
        quantum_features = quantum_features.reshape(sequences.shape[0], self.n_qubits)
        return self.head(quantum_features.to(dtype=angles.dtype))


class ClassicalTextClassifier(torch.nn.Module):

    def __init__(
        self,
        vocabulary_size: int,
        n_qubits: int = 4,
        embedding_dim: int = 32,
        n_filters: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = TextCNN(
            vocabulary_size,
            n_qubits=n_qubits,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            dropout=dropout,
        )
        self.bottleneck = torch.nn.Linear(n_qubits, n_qubits)
        self.head = torch.nn.Linear(n_qubits, 2)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.bottleneck(self.cnn(sequences))))
