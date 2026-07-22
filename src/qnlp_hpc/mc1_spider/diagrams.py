import time

import numpy as np
import pandas as pd

from lambeq import AtomicType, SpiderAnsatz, spiders_reader
from lambeq.backend.tensor import Dim

from qnlp_hpc.mc1_spider.config import (
    MAX_ORDER,
    NOUN_DIM,
    SENTENCE_DIM,
)

def build_tensor_diagrams(frame: pd.DataFrame) -> dict[str, object]:
    """Create offline spiders_reader diagrams and apply SpiderAnsatz."""
    unique_sentences = pd.unique(
        pd.concat(
            [frame["sentence_1"], frame["sentence_2"]],
            ignore_index=True,
        )
    ).tolist()

    parse_start = time.perf_counter()
    parsed_diagrams = spiders_reader.sentences2diagrams(unique_sentences)
    parse_seconds = time.perf_counter() - parse_start

    failed_sentences = [
        sentence
        for sentence, diagram in zip(unique_sentences, parsed_diagrams)
        if diagram is None
    ]
    if failed_sentences:
        raise RuntimeError(
            "spiders_reader could not create diagrams for: "
            + repr(failed_sentences)
        )

    noun, sentence = AtomicType.NOUN, AtomicType.SENTENCE
    spider_ansatz = SpiderAnsatz(
        {
            noun: Dim(NOUN_DIM),
            sentence: Dim(SENTENCE_DIM),
        },
        max_order=MAX_ORDER,
    )

    ansatz_start = time.perf_counter()
    tensor_diagrams = {
        text: spider_ansatz(diagram)
        for text, diagram in zip(unique_sentences, parsed_diagrams)
    }
    ansatz_seconds = time.perf_counter() - ansatz_start

    print(
        f"Built {len(unique_sentences)} spiders_reader diagrams in "
        f"{parse_seconds:.2f} s."
    )
    print(
        f"Applied SpiderAnsatz to {len(unique_sentences)} diagrams in "
        f"{ansatz_seconds:.2f} s."
    )
    return tensor_diagrams


def make_pairs_and_targets(
    split: pd.DataFrame,
    tensor_diagrams: dict[str, object],
) -> tuple[list[tuple[object, object]], np.ndarray]:
    """Convert a dataframe split into tensor-diagram pairs and class IDs."""
    pairs = [
        (
            tensor_diagrams[row.sentence_1],
            tensor_diagrams[row.sentence_2],
        )
        for row in split.itertuples(index=False)
    ]

    # CrossEntropyLoss expects class IDs rather than one-hot vectors.
    # lambeq Dataset may convert these values to FloatTensor; the custom loss
    # below converts them back to torch.long inside the training step.
    targets = split["label"].to_numpy(dtype=np.int64)
    return pairs, targets