from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from lambeq import (
    AtomicType,
    Dataset,
    PytorchModel,
    PytorchTrainer,
    SpiderAnsatz,
    spiders_reader,
)
from lambeq.backend.tensor import Dim



os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 0
TEST_SIZE = 0.20       # 80% train、20% test
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 3e-2

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "MC1.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)



df = pd.read_csv(
    DATA_FILE,
    header=None,
    sep=",",
    skipinitialspace=True,
    names=["sentence_1", "sentence_2", "label"],
)

df["sentence_1"] = df["sentence_1"].astype(str).str.strip()
df["sentence_2"] = df["sentence_2"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"]).astype(int)



print("Data Length:", len(df))
print("label count:")
print(df["label"].value_counts().sort_index())



indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label"],
)

print("\n Training set size:", len(train_indices))
print("Test set size:", len(test_indices))

print("\n Training set label count:")
print(df.iloc[train_indices]["label"].value_counts().sort_index())

print("\n Test set label count:")
print(df.iloc[test_indices]["label"].value_counts().sort_index())



all_sentences = []

for row in df.itertuples(index=False):
    all_sentences.append(row.sentence_1)
    all_sentences.append(row.sentence_2)

print("\n Using spiders_reader ...")
all_diagrams = spiders_reader.sentences2diagrams(all_sentences)


# sentence_1(row 0), sentence_2(row 0),
# sentence_1(row 1), sentence_2(row 1), ...
left_diagrams = all_diagrams[0::2]
right_diagrams = all_diagrams[1::2]


# ============================================================
# SpiderAnsatz
# ============================================================
ansatz = SpiderAnsatz(
    {
        AtomicType.SENTENCE: Dim(2),
    },
    max_order=2,
)
print("\nApplying SpiderAnsatz……")

left_circuits = [
    ansatz(diagram)
    for diagram in left_diagrams
]

right_circuits = [
    ansatz(diagram)
    for diagram in right_diagrams
]


pair_circuits = [
    left @ right
    for left, right in zip(left_circuits, right_circuits)
]

train_circuits = [
    pair_circuits[i]
    for i in train_indices
]

test_circuits = [
    pair_circuits[i]
    for i in test_indices
]



train_labels_integer = (
    df.iloc[train_indices]["label"]
    .to_numpy(dtype=int)
)

test_labels_integer = (
    df.iloc[test_indices]["label"]
    .to_numpy(dtype=int)
)

# label 0 -> [1, 0]
# label 1 -> [0, 1]
train_labels = np.eye(
    2,
    dtype=np.float32,
)[train_labels_integer].tolist()

test_labels = np.eye(
    2,
    dtype=np.float32,
)[test_labels_integer].tolist()



class SpiderPairClassifier(PytorchModel):

    def __init__(self) -> None:
        super().__init__()

        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, diagrams):
        features = self.get_diagram_output(diagrams)

        features = features.reshape(
            features.shape[0],
            -1,
        ).float()


        logits = self.classifier(features)

        return logits


model = SpiderPairClassifier.from_diagrams(
    pair_circuits
)



def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    predicted_classes = torch.argmax(
        predictions,
        dim=1,
    )

    true_classes = torch.argmax(
        targets,
        dim=1,
    )

    return (
        predicted_classes == true_classes
    ).float().mean()



train_dataset = Dataset(
    train_circuits,
    train_labels,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


trainer = PytorchTrainer(
    model=model,

    # two-dimensional one-hot labels
    loss_function=torch.nn.BCEWithLogitsLoss(),

    optimizer=torch.optim.AdamW,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,

    evaluate_functions={
        "accuracy": accuracy,
    },

    evaluate_on_train=True,
    verbose="text",
    seed=SEED,


    device=-1,
)


# ============================================================
# Training
# ============================================================
print("\nStart Training……")


trainer.fit(
    train_dataset,
    val_dataset=None,
    log_interval=5,
)

train_accuracy_history = trainer.train_eval_results["accuracy"]


max_train_accuracy = max(train_accuracy_history)

best_train_epoch = (
    np.argmax(train_accuracy_history) + 1
)

print("\n====================================")
print(
    f"Maximum training accuracy: "
    f"{max_train_accuracy:.4f}"
)
print(
    f"Best training epoch: "
    f"{best_train_epoch}"
)
print("====================================")


model.eval()

with torch.no_grad():

    test_logits = model(test_circuits)

    test_probabilities = torch.softmax(
        test_logits,
        dim=1,
    )

    test_predictions = torch.argmax(
        test_probabilities,
        dim=1,
    ).cpu().numpy()

test_accuracy = np.mean(
    test_predictions == test_labels_integer
)

print("\n====================================")
print(f"Test accuracy: {test_accuracy:.4f}")
print("====================================")

print("\nConfusion matrix:")
print(
    confusion_matrix(
        test_labels_integer,
        test_predictions,
        labels=[0, 1],
    )
)

print("\nClassification report:")
print(
    classification_report(
        test_labels_integer,
        test_predictions,
        labels=[0, 1],
        target_names=["class 0", "class 1"],
        digits=4,
        zero_division=0,
    )
)



result_df = (
    df.iloc[test_indices]
    .copy()
    .reset_index(drop=True)
)

result_df["predicted_label"] = test_predictions

result_df["probability_label_0"] = (
    test_probabilities[:, 0]
    .cpu()
    .numpy()
)

result_df["probability_label_1"] = (
    test_probabilities[:, 1]
    .cpu()
    .numpy()
)

result_df.to_csv(
    BASE_DIR / "spider_test_predictions.csv",
    index=False,
)

torch.save(
    model.state_dict(),
    BASE_DIR / "spider_pair_model_state.pt",
)



plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    trainer.train_epoch_costs,
)

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("SpiderAnsatz sentence-pair training")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "spider_training_curve.png",
    dpi=200,
)

plt.show()

print("\nSaved:")
print("spider_pair_model_state.pt")
print("spider_test_predictions.csv")
print("spider_training_curve.png")