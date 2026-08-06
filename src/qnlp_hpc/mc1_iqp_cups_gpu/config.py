"""Configuration for the MC1 cups_reader and IQP experiment."""

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

DATA_PATH = PROCESSED_DIR / "MC1.txt"
OUTPUT_DIR = OUTPUTS_DIR / "mc1_iqp_cups_gpu"
LOG_DIR = OUTPUT_DIR / "training_logs"

# Every sentence not listed here is assigned to training.
DEVELOPMENT_SENTENCES = frozenset(
    {
        "chef creates meal",
        "chef prepares tasty dish",
        "cook prepares meal",
        "devoted programmer creates advanced code",
        "experienced chef creates complicated meal",
        "experienced cook prepares complicated dish",
        "programmer writes complicated code",
        "skilful chef prepares complicated dish",
        "skilful chef prepares tasty dish",
        "skilful cook creates tasty dish",
        "skilful cook prepares meal",
        "skilful programmer creates complicated code",
        "skilful programmer writes advanced code",
        "skilful programmer writes code",
    }
)

TEST_SENTENCES = frozenset(
    {
        "chef creates dish",
        "chef creates tasty meal",
        "cook prepares complicated dish",
        "cook prepares complicated meal",
        "experienced chef creates complicated dish",
        "experienced chef prepares meal",
        "experienced cook prepares meal",
        "hacker creates advanced code",
        "hacker writes advanced code",
        "programmer creates code",
        "programmer writes advanced code",
    }
)
