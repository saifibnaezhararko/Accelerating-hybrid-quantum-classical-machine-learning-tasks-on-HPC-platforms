"""Data acquisition + format conversion (Kaggle API, CSV/TSV/JSON).

``acquire`` is deliberately not re-exported: it imports the optional ``kaggle``
package, and the training pipeline must import this package without it.
"""

from qnlp_hpc.data.convert import ColumnMapping, convert_file, load_frame
from qnlp_hpc.data.schema import (
    DatasetValidationError,
    SentencePair,
    check_dataset,
    deduplicate,
    make_pair,
    read_pairs,
    write_pairs,
)

__all__ = [
    "ColumnMapping",
    "DatasetValidationError",
    "SentencePair",
    "check_dataset",
    "convert_file",
    "deduplicate",
    "load_frame",
    "make_pair",
    "read_pairs",
    "write_pairs",
]
