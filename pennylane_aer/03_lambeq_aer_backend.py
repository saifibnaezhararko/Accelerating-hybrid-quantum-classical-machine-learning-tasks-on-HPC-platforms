from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_ROOT / "src"), str(_ROOT / "pennylane_aer")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aer_backend import (
    aer_backend_config,
    aer_gpu_available,
    copy_weights,
    describe_backend,
)
from lambeq import PennyLaneModel

from qnlp_hpc.mc1_iqp_cups import circuits as evelyn_circuits
from qnlp_hpc.mc1_iqp_cups import config as evelyn_config
from qnlp_hpc.mc1_iqp_cups import data as evelyn_data
from qnlp_hpc.mc1_iqp_cups import model as evelyn_model

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class AerIQPPairModel(PennyLaneModel):

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        pair_feature_dim = 2 * evelyn_config.CIRCUIT_OUTPUT_DIM
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pair_feature_dim, evelyn_config.CLASSIFIER_HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(evelyn_config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(evelyn_config.CLASSIFIER_HIDDEN_DIM, 2),
        )

    def forward(self, diagram_pairs) -> torch.Tensor:
        if not diagram_pairs:
            raise ValueError("diagram_pairs cannot be empty.")

        flat_circuits = [circuit for pair in diagram_pairs for circuit in pair]
        probabilities = self.get_diagram_output(flat_circuits)
        probabilities = probabilities.reshape(
            len(diagram_pairs), 2, evelyn_config.CIRCUIT_OUTPUT_DIM
        )

        features = 2.0 * (probabilities - 0.5)
        left, right = features[:, 0, :], features[:, 1, :]
        pair_features = torch.cat((torch.abs(left - right), left * right), dim=1)

        parameter = next(self.classifier.parameters())
        pair_features = pair_features.to(device=parameter.device, dtype=parameter.dtype)
        return self.classifier(pair_features)


def build_shared_head_state(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    reference = torch.nn.Sequential(
        torch.nn.Linear(2 * evelyn_config.CIRCUIT_OUTPUT_DIM, evelyn_config.CLASSIFIER_HIDDEN_DIM),
        torch.nn.GELU(),
        torch.nn.Dropout(evelyn_config.CLASSIFIER_DROPOUT),
        torch.nn.Linear(evelyn_config.CLASSIFIER_HIDDEN_DIM, 2),
    )
    return {key: value.clone() for key, value in reference.state_dict().items()}


def evaluate(pair_model, pairs, batch_size: int) -> tuple[np.ndarray, float]:
    pair_model.eval()
    outputs = []
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(0, len(pairs), batch_size):
            outputs.append(pair_model(pairs[index : index + batch_size]))
    seconds = time.perf_counter() - start
    return torch.cat(outputs).detach().cpu().numpy(), seconds


def train_briefly(pair_model, pairs, targets, epochs: int, batch_size: int) -> list[float]:
    optimiser = torch.optim.Adam(
        pair_model.parameters(),
        lr=evelyn_config.LEARNING_RATE,
        weight_decay=evelyn_config.WEIGHT_DECAY,
    )
    loss_function = evelyn_model.LambeqCrossEntropyLoss()
    target_tensor = torch.as_tensor(targets)
    losses = []

    pair_model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for index in range(0, len(pairs), batch_size):
            batch = pairs[index : index + batch_size]
            batch_targets = target_tensor[index : index + batch_size]
            optimiser.zero_grad()
            loss = loss_function(pair_model(batch), batch_targets)
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"    epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}")
    return losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=12, help="Test pairs to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=evelyn_config.SEED)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--gpu", action="store_true", help="Use Aer's GPU statevector.")
    parser.add_argument(
        "--demo-shot-failure",
        action="store_true",
        help="Also run shot-based Aer, to show post-selection collapsing to NaN.",
    )
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    frame = evelyn_data.load_mc1(evelyn_config.DATA_PATH)
    train_frame, _, test_frame, _ = evelyn_data.split_sentence_disjoint(frame)
    subset = test_frame.head(arguments.pairs)

    circuit_lookup = evelyn_circuits.build_quantum_circuits(
        __import__("pandas").concat([train_frame, subset], ignore_index=True)
    )
    pairs, targets = evelyn_circuits.make_pairs_and_targets(subset, circuit_lookup)
    training_circuits = [
        circuit
        for pair in evelyn_circuits.make_pairs_and_targets(train_frame, circuit_lookup)[0]
        for circuit in pair
    ]
    evaluation_circuits = [circuit for pair in pairs for circuit in pair]

    all_circuits = training_circuits + evaluation_circuits
    training_symbols = evelyn_circuits.collect_circuit_symbols(training_circuits)
    evaluation_symbols = evelyn_circuits.collect_circuit_symbols(evaluation_circuits)
    unseen = evaluation_symbols - training_symbols
    if unseen:
        raise RuntimeError(
            f"Evaluation circuits introduce untrained symbols: {sorted(map(str, unseen))}"
        )

    sample_circuit = next(iter(circuit_lookup.values()))
    print(
        f"Evaluating {len(pairs)} MC1 test pairs; sentence qubits={evelyn_config.SENTENCE_QUBITS}"
    )

    head_state = build_shared_head_state(arguments.seed)

    baseline = evelyn_model.IQPPairModel.from_diagrams(all_circuits)
    baseline.initialise_weights()
    baseline.classifier.load_state_dict(head_state)

    probe = PennyLaneModel.from_diagrams([sample_circuit])
    n_qubits = probe.circuit_map[sample_circuit]._n_qubits
    n_post_selected = len(probe.circuit_map[sample_circuit]._post_selection)
    print(
        f"Circuit width: {n_qubits} qubits, {n_post_selected} post-selected "
        f"(post-selection succeeds with probability ~2^-{n_post_selected})"
    )

    configurations: list[tuple[str, dict | None]] = [
        ("pennylane-default", None),
        ("pennylane-aer", aer_backend_config(gpu=arguments.gpu)),
    ]
    if arguments.demo_shot_failure:
        configurations.append(("pennylane-aer-1024-shots", aer_backend_config(shots=1024)))

    if arguments.gpu and not aer_gpu_available():
        print("  --gpu requested but this Aer build has no GPU device; using CPU.")

    results = []
    baseline_logits, baseline_seconds = evaluate(baseline, pairs, arguments.batch_size)
    print(f"\n[tensor-contraction] baseline  {baseline_seconds:6.2f}s")
    results.append(
        {
            "backend": "tensor-contraction (PytorchQuantumModel)",
            "seconds": baseline_seconds,
            "max_abs_logit_difference": 0.0,
            "predictions_match_baseline": True,
            "finite": True,
        }
    )
    baseline_predictions = baseline_logits.argmax(axis=1)

    for label, backend_config in configurations:
        print(f"\n[{label}] {describe_backend(backend_config)}")
        quantum_model = AerIQPPairModel.from_diagrams(all_circuits, backend_config=backend_config)
        quantum_model.initialise_weights()
        copy_weights(baseline, quantum_model)
        quantum_model.classifier.load_state_dict(head_state)

        logits, seconds = evaluate(quantum_model, pairs, arguments.batch_size)
        finite = bool(np.isfinite(logits).all())
        difference = float(np.abs(logits - baseline_logits).max()) if finite else float("nan")
        predictions = logits.argmax(axis=1)
        matches = bool(finite and (predictions == baseline_predictions).all())

        print(f"    time: {seconds:6.2f}s  ({seconds / baseline_seconds:5.1f}x baseline)")
        print(f"    finite outputs: {finite}")
        print(f"    max |logit difference| vs baseline: {difference:.3e}")
        print(f"    identical predictions: {matches}")

        record = {
            "backend": label,
            "backend_config": describe_backend(backend_config),
            "seconds": seconds,
            "slowdown_vs_baseline": seconds / baseline_seconds,
            "max_abs_logit_difference": difference,
            "predictions_match_baseline": matches,
            "finite": finite,
        }

        if arguments.train_epochs and finite:
            print(f"    training {arguments.train_epochs} epochs on this backend:")
            record["training_losses"] = train_briefly(
                quantum_model,
                pairs,
                targets,
                arguments.train_epochs,
                arguments.batch_size,
            )

        results.append(record)

    aer = next((r for r in results if r["backend"] == "pennylane-aer"), None)
    agreed = bool(aer and aer["predictions_match_baseline"])
    print("\n=== verdict ===")
    if agreed:
        print(
            "Qiskit Aer reproduces the baseline predictions exactly.\n"
            "The tensor-contraction model and the Aer simulator agree, so the port\n"
            "is validated and Aer can stand in as the quantum layer."
        )
    else:
        print("Aer did NOT reproduce the baseline predictions - see the table above.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "03_lambeq_aer_backend.json"
    report_path.write_text(
        json.dumps(
            {
                "pairs": len(pairs),
                "sentence_qubits": evelyn_config.SENTENCE_QUBITS,
                "circuit_qubits": n_qubits,
                "post_selected_qubits": n_post_selected,
                "aer_gpu_available": aer_gpu_available(),
                "results": results,
                "aer_matches_baseline": agreed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {report_path}")
    return 0 if agreed else 1


if __name__ == "__main__":
    raise SystemExit(main())
