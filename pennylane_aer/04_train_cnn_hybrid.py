from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from aer_backend import aer_backend_config, aer_gpu_available, describe_backend
from hybrid_cnn_quantum import (
    ClassicalTextClassifier,
    HybridTextClassifier,
    load_trec,
)

ROOT = _HERE.parent
DATA_DIR = ROOT / "modified_trec_dataset"
OUTPUT_DIR = _HERE / "outputs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, sequences: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    model.eval()
    logits = model(sequences)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    return accuracy, float(loss)


def train_model(
    model: torch.nn.Module,
    dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    label: str,
    gradient_clip: float | None = None,
) -> dict[str, object]:
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    n_train = len(dataset.train_sequences)

    history: list[dict[str, float]] = []
    best_accuracy = 0.0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(n_train)
        epoch_losses = []
        epoch_gradient_norms = []

        for index in range(0, n_train, batch_size):
            batch = permutation[index : index + batch_size]
            optimiser.zero_grad()
            logits = model(dataset.train_sequences[batch])
            loss = torch.nn.functional.cross_entropy(logits, dataset.train_labels[batch])
            loss.backward()
            if gradient_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
                epoch_gradient_norms.append(float(gradient_norm))
            optimiser.step()
            epoch_losses.append(float(loss.detach()))

        train_accuracy, _ = evaluate(model, dataset.train_sequences, dataset.train_labels)
        test_accuracy, test_loss = evaluate(model, dataset.test_sequences, dataset.test_labels)
        best_accuracy = max(best_accuracy, test_accuracy)
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "seconds": epoch_seconds,
                "gradient_norm_mean_before_clipping": (
                    float(np.mean(epoch_gradient_norms)) if epoch_gradient_norms else None
                ),
            }
        )
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(
                f"    epoch {epoch:3d}/{epochs}  loss={np.mean(epoch_losses):.4f}  "
                f"train_acc={train_accuracy:.3f}  test_acc={test_accuracy:.3f}  "
                f"{epoch_seconds:.1f}s"
            )

    seconds = time.perf_counter() - start
    # The final epoch already evaluates the unchanged model on the full test set.
    # Re-running an external simulator here is expensive and, for stochastic
    # differentiation methods, can produce a confusing second reported value.
    final_accuracy = float(history[-1]["test_accuracy"])
    final_loss = float(history[-1]["test_loss"])
    quantum_parameters = sum(p.numel() for n, p in model.named_parameters() if "quantum" in n)
    quantum_gradient = sum(
        float(parameter.grad.detach().pow(2).sum())
        for name, parameter in model.named_parameters()
        if "quantum" in name and parameter.grad is not None
    ) ** 0.5
    cnn_gradient = sum(
        float(parameter.grad.detach().pow(2).sum())
        for name, parameter in model.named_parameters()
        if name.startswith("cnn.") and parameter.grad is not None
    ) ** 0.5

    print(
        f"    done in {seconds:.1f}s | final test acc {final_accuracy:.4f} "
        f"| best {best_accuracy:.4f}"
    )
    print(
        f"    final-batch gradients: cnn={cnn_gradient:.6f}  "
        f"quantum={quantum_gradient:.6f}"
    )
    return {
        "configuration": label,
        "epochs": epochs,
        "seconds": seconds,
        "seconds_per_epoch": seconds / epochs,
        "final_test_accuracy": final_accuracy,
        "final_test_loss": final_loss,
        "best_test_accuracy": best_accuracy,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "quantum_parameters": quantum_parameters,
        "cnn_gradient_norm": cnn_gradient,
        "quantum_gradient_norm": quantum_gradient,
        "gradient_clip": gradient_clip,
        "history": history,
    }


def benchmark_backend(
    model: torch.nn.Module,
    dataset,
    steps: int,
    batch_size: int,
    learning_rate: float,
    label: str,
    epochs_for_projection: int,
) -> dict[str, object]:
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    n_train = len(dataset.train_sequences)

    print(f"    measuring {steps} optimisation steps at batch size {batch_size}")
    losses = []
    step_seconds = []
    permutation = torch.randperm(n_train)

    model.train()
    for step in range(steps):
        batch = permutation[step * batch_size : (step + 1) * batch_size]
        if len(batch) == 0:
            break
        start = time.perf_counter()
        optimiser.zero_grad()
        logits = model(dataset.train_sequences[batch])
        loss = torch.nn.functional.cross_entropy(logits, dataset.train_labels[batch])
        loss.backward()
        optimiser.step()
        elapsed = time.perf_counter() - start
        step_seconds.append(elapsed)
        losses.append(float(loss.detach()))
        print(f"      step {step + 1}/{steps}  loss={losses[-1]:.4f}  {elapsed:.2f}s")

    mean_step = float(np.mean(step_seconds))
    per_sample = mean_step / batch_size
    projected_epoch = per_sample * n_train
    projected_total = projected_epoch * epochs_for_projection

    quantum_gradient = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if "quantum" in name and parameter.grad is not None
    )

    print(f"    mean step: {mean_step:.2f}s ({per_sample:.2f}s per sentence)")
    print(f"    projected: {projected_epoch / 60:.1f} min/epoch, ")
    print(f"               {projected_total / 3600:.1f} h for {epochs_for_projection} epochs")
    print(f"    gradient reaching circuit weights: {quantum_gradient:.6f}")

    return {
        "configuration": label,
        "mode": "benchmark",
        "steps": len(step_seconds),
        "batch_size": batch_size,
        "mean_step_seconds": mean_step,
        "seconds_per_sentence": per_sample,
        "seconds_per_epoch": projected_epoch,
        "projected_epoch_minutes": projected_epoch / 60,
        "projected_total_hours": projected_total / 3600,
        "losses": losses,
        "quantum_gradient_norm": quantum_gradient,
        "quantum_parameters": sum(p.numel() for n, p in model.named_parameters() if "quantum" in n),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "final_test_accuracy": float("nan"),
        "best_test_accuracy": float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4, help="Quantum bottleneck width.")
    parser.add_argument("--layers", type=int, default=2, help="StronglyEntanglingLayers depth.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="CNN token embedding width.")
    parser.add_argument("--filters", type=int, default=16, help="CNN filters per kernel size.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--aer-mode",
        choices=("benchmark", "train"),
        default="benchmark",
        help="Benchmark a few Aer steps (default) or fully train the Aer hybrid model.",
    )
    parser.add_argument(
        "--aer-epochs",
        type=int,
        default=None,
        help="Aer training epochs; defaults to --epochs.",
    )
    parser.add_argument("--aer-steps", type=int, default=4, help="Measured Aer optimiser steps.")
    parser.add_argument("--aer-batch-size", type=int, default=4)
    parser.add_argument(
        "--aer-learning-rate",
        type=float,
        default=None,
        help="Aer learning rate; defaults to --learning-rate.",
    )
    parser.add_argument(
        "--aer-diff-method",
        choices=("parameter-shift", "finite-diff", "spsa"),
        default="parameter-shift",
        help="PennyLane differentiation method for the external Aer simulator.",
    )
    parser.add_argument(
        "--aer-spsa-directions",
        type=int,
        default=1,
        help="SPSA perturbation directions averaged per gradient (higher is stabler/slower).",
    )
    parser.add_argument(
        "--aer-gradient-clip",
        type=float,
        default=None,
        help="Optional maximum total gradient L2 norm before each Aer optimiser step.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-aer", action="store_true")
    parser.add_argument(
        "--only-aer",
        action="store_true",
        help="Skip the classical and default.qubit controls and run only the Aer stage.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use Aer's GPU statevector.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    arguments = parser.parse_args()

    if arguments.only_aer and arguments.skip_aer:
        parser.error("--only-aer and --skip-aer cannot be used together")
    if arguments.epochs <= 0:
        parser.error("--epochs must be positive")
    if arguments.embedding_dim <= 0:
        parser.error("--embedding-dim must be positive")
    if arguments.filters <= 0:
        parser.error("--filters must be positive")
    if arguments.aer_epochs is not None and arguments.aer_epochs <= 0:
        parser.error("--aer-epochs must be positive")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if arguments.aer_batch_size <= 0:
        parser.error("--aer-batch-size must be positive")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.aer_learning_rate is not None and arguments.aer_learning_rate <= 0:
        parser.error("--aer-learning-rate must be positive")
    if arguments.aer_spsa_directions <= 0:
        parser.error("--aer-spsa-directions must be positive")
    if arguments.aer_gradient_clip is not None and arguments.aer_gradient_clip <= 0:
        parser.error("--aer-gradient-clip must be positive")

    if not arguments.data_dir.is_dir():
        print(f"Dataset not found: {arguments.data_dir}\nRun scripts/prepare_trec.py first.")
        return 2

    dataset = load_trec(arguments.data_dir)
    print(
        f"TREC: {len(dataset.train_sequences)} train / {len(dataset.test_sequences)} test, "
        f"vocab {dataset.vocabulary_size}, max length {dataset.max_length}"
    )
    print(f"Quantum bottleneck: {arguments.qubits} qubits x {arguments.layers} layers")
    print(
        f"CNN: embedding_dim={arguments.embedding_dim}, "
        f"filters={arguments.filters} per kernel"
    )

    results = []

    if not arguments.only_aer:
        print("\n[classical] CNN -> linear bottleneck -> head")
        set_seed(arguments.seed)
        classical = ClassicalTextClassifier(
            dataset.vocabulary_size,
            n_qubits=arguments.qubits,
            embedding_dim=arguments.embedding_dim,
            n_filters=arguments.filters,
        )
        results.append(
            train_model(
                classical,
                dataset,
                arguments.epochs,
                arguments.batch_size,
                arguments.learning_rate,
                "classical",
            )
        )

        print("\n[hybrid] CNN -> PennyLane circuit -> head  (default.qubit)")
        set_seed(arguments.seed)
        hybrid = HybridTextClassifier(
            dataset.vocabulary_size,
            n_qubits=arguments.qubits,
            n_layers=arguments.layers,
            embedding_dim=arguments.embedding_dim,
            n_filters=arguments.filters,
        )
        results.append(
            train_model(
                hybrid,
                dataset,
                arguments.epochs,
                arguments.batch_size,
                arguments.learning_rate,
                "hybrid-default.qubit",
            )
        )

    if not arguments.skip_aer:
        gpu_available = aer_gpu_available()
        use_gpu = arguments.gpu and gpu_available
        if arguments.gpu and not gpu_available:
            print("\nERROR: --gpu was requested, but this Aer build exposes no GPU device.")
            return 2
        backend_config = aer_backend_config(gpu=use_gpu)
        print(
            f"\n[hybrid-aer] CNN -> PennyLane circuit -> head  ({describe_backend(backend_config)})"
        )
        set_seed(arguments.seed)
        gradient_kwargs = None
        if arguments.aer_diff_method == "spsa":
            gradient_kwargs = {
                "sampler_rng": np.random.default_rng(arguments.seed),
                "num_directions": arguments.aer_spsa_directions,
            }
        hybrid_aer = HybridTextClassifier(
            dataset.vocabulary_size,
            n_qubits=arguments.qubits,
            n_layers=arguments.layers,
            embedding_dim=arguments.embedding_dim,
            n_filters=arguments.filters,
            backend_config=backend_config,
            diff_method=arguments.aer_diff_method,
            gradient_kwargs=gradient_kwargs,
        )
        aer_epochs = arguments.aer_epochs or arguments.epochs
        aer_learning_rate = arguments.aer_learning_rate or arguments.learning_rate
        aer_label = "hybrid-qiskit-aer-gpu" if use_gpu else "hybrid-qiskit-aer-cpu"
        if arguments.aer_mode == "train":
            print(
                f"    full Aer training: {aer_epochs} epochs, "
                f"batch size {arguments.aer_batch_size}, "
                f"diff_method={arguments.aer_diff_method}"
            )
            results.append(
                train_model(
                    hybrid_aer,
                    dataset,
                    epochs=aer_epochs,
                    batch_size=arguments.aer_batch_size,
                    learning_rate=aer_learning_rate,
                    label=aer_label,
                    gradient_clip=arguments.aer_gradient_clip,
                )
            )
        else:
            results.append(
                benchmark_backend(
                    hybrid_aer,
                    dataset,
                    steps=arguments.aer_steps,
                    batch_size=arguments.aer_batch_size,
                    learning_rate=aer_learning_rate,
                    label=aer_label,
                    epochs_for_projection=aer_epochs,
                )
            )

    print("\n=== summary ===")
    print(f"{'configuration':24s} {'test acc':>9s} {'best':>7s} {'s/epoch':>10s} {'q-params':>9s}")
    for record in results:
        accuracy = record["final_test_accuracy"]
        best = record["best_test_accuracy"]
        measured = "" if record.get("mode") != "benchmark" else "  (projected)"
        accuracy_text = "        -" if np.isnan(accuracy) else f"{accuracy:9.4f}"
        best_text = "      -" if np.isnan(best) else f"{best:7.4f}"
        print(
            f"{record['configuration']:24s} {accuracy_text} {best_text} "
            f"{record['seconds_per_epoch']:10.2f} {record['quantum_parameters']:9d}{measured}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "04_cnn_hybrid_quantum.json"
    report_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "train": len(dataset.train_sequences),
                    "test": len(dataset.test_sequences),
                    "vocabulary_size": dataset.vocabulary_size,
                    "max_length": dataset.max_length,
                },
                "qubits": arguments.qubits,
                "layers": arguments.layers,
                "embedding_dim": arguments.embedding_dim,
                "filters": arguments.filters,
                "seed": arguments.seed,
                "aer_gpu_available": aer_gpu_available(),
                "aer_requested": not arguments.skip_aer,
                "aer_mode": arguments.aer_mode,
                "aer_epochs": arguments.aer_epochs or arguments.epochs,
                "aer_batch_size": arguments.aer_batch_size,
                "aer_learning_rate": arguments.aer_learning_rate or arguments.learning_rate,
                "aer_diff_method": arguments.aer_diff_method,
                "aer_spsa_directions": arguments.aer_spsa_directions,
                "aer_gradient_clip": arguments.aer_gradient_clip,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
