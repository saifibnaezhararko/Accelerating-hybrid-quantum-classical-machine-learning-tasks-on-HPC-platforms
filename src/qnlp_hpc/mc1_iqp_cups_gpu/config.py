"""Configuration for the TREC pair cups_reader and IQP experiment."""

from qnlp_hpc.paths import OUTPUTS_DIR, PROCESSED_DIR

SEED = 2
GPU_DEVICE = 0

BATCH_SIZE = 8
EPOCHS = 120
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
CLASSIFIER_HIDDEN_DIM = 16
CLASSIFIER_DROPOUT = 0.05

# cups_reader maps sentence-type wires to a two-qubit IQP circuit.
SENTENCE_QUBITS = 2
IQP_LAYERS = 1
N_SINGLE_QUBIT_PARAMS = 3
CIRCUIT_OUTPUT_DIM = 2**SENTENCE_QUBITS

# TREC sentence-pair dataset
DATA_PATH = PROCESSED_DIR / "trec_pairs_500.txt"

# 90% train+development, 10% final test
TEST_RATIO = 0.10

# Take 10% of the remaining 90% as development
# Final sizes for 1000 samples:
# train = 810
# development = 90
# test = 100
DEVELOPMENT_RATIO = 0.10

OUTPUT_DIR = OUTPUTS_DIR / "trec_iqp_cups_gpu"
LOG_DIR = OUTPUT_DIR / "training_logs"
