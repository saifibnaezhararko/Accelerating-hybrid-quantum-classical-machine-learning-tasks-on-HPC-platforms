"""Build sentence diagrams and IQP quantum circuits."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from lambeq import AtomicType, IQPAnsatz, cups_reader

from qnlp_hpc.mc1_iqp_cups.config import (
    IQP_LAYERS,
    N_SINGLE_QUBIT_PARAMS,
    SENTENCE_QUBITS,
)


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
        raise RuntimeError("cups_reader could not create diagrams for: " + repr(failed_sentences))

    iqp_ansatz = IQPAnsatz(
        {AtomicType.SENTENCE: SENTENCE_QUBITS},
        n_layers=IQP_LAYERS,
        n_single_qubit_params=N_SINGLE_QUBIT_PARAMS,
    )

    ansatz_start = time.perf_counter()
    quantum_circuits = {
        sentence: iqp_ansatz(diagram)
        for sentence, diagram in zip(
            unique_sentences,
            parsed_diagrams,
            strict=True,
        )
    }
    ansatz_seconds = time.perf_counter() - ansatz_start

    print(f"Built {len(unique_sentences)} cups_reader diagrams in " f"{parse_seconds:.2f} s.")
    print(
        f"Applied {IQP_LAYERS}-layer two-qubit lambeq IQPAnsatz to "
        f"{len(unique_sentences)} diagrams in {ansatz_seconds:.2f} s."
    )
    return quantum_circuits


def make_pairs_and_targets(
    split: pd.DataFrame,
    quantum_circuits: dict[str, object],
) -> tuple[list[tuple[object, object]], np.ndarray]:
    """Convert a dataframe split into circuit pairs and class IDs."""
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
