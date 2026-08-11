from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pennylane as qml
import torch

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_circuit(device: qml.devices.Device, n_qubits: int):

    @qml.qnode(device, interface="torch")
    def circuit(weights: torch.Tensor):
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)

        for wire in range(n_qubits - 1):
            qml.CRZ(weights[0, wire], wires=[wire, wire + 1])

        for wire in range(n_qubits):
            qml.RX(weights[1, wire], wires=wire)
            qml.RZ(weights[2, wire], wires=wire)

        return qml.probs(wires=range(n_qubits))

    return circuit


def build_device(
    name: str,
    n_qubits: int,
    shots: int | None,
    use_gpu: bool,
    aer_backend: str | None = None,
) -> qml.devices.Device:
    if name == "default.qubit":
        return qml.device("default.qubit", wires=n_qubits, shots=shots)

    extra = {"backend": aer_backend} if aer_backend else {}
    device = qml.device("qiskit.aer", wires=n_qubits, shots=shots, **extra)
    if use_gpu:
        device.backend.set_options(device="GPU")
    return device


def describe_aer(use_gpu: bool) -> dict[str, object]:
    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    devices = list(simulator.available_devices())
    return {
        "aer_available_devices": devices,
        "gpu_requested": use_gpu,
        "gpu_supported": "GPU" in devices,
    }


def run_backend(
    label: str,
    device: qml.devices.Device,
    n_qubits: int,
    weights: torch.Tensor,
    repeats: int,
) -> dict[str, object]:
    circuit = make_circuit(device, n_qubits)

    circuit(weights)

    start = time.perf_counter()
    for _ in range(repeats):
        probabilities = circuit(weights)
    forward_seconds = (time.perf_counter() - start) / repeats

    parameters = weights.detach().clone().requires_grad_(True)
    circuit_for_grad = make_circuit(device, n_qubits)
    start = time.perf_counter()
    output = circuit_for_grad(parameters)
    output[0].backward()
    backward_seconds = time.perf_counter() - start

    gradient = parameters.grad
    gradient_norm = float(gradient.norm()) if gradient is not None else 0.0

    probability_vector = probabilities.detach().cpu().numpy()
    return {
        "backend": label,
        "qubits": n_qubits,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "probability_sum": float(probability_vector.sum()),
        "first_probabilities": [float(value) for value in probability_vector[:4]],
        "gradient_flows": gradient is not None and gradient_norm > 0.0,
        "gradient_norm": gradient_norm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=4096, help="Shot count for the sampling run.")
    parser.add_argument("--repeats", type=int, default=5, help="Forward passes per timing.")
    parser.add_argument("--gpu", action="store_true", help="Ask Aer for its GPU statevector.")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    n_qubits = arguments.qubits
    weights = torch.rand(3, n_qubits, dtype=torch.float64) * 2 * torch.pi

    aer_report = describe_aer(arguments.gpu)
    print(f"PennyLane {qml.__version__} | qubits={n_qubits}")
    print(f"Aer devices on this machine: {aer_report['aer_available_devices']}")
    if arguments.gpu and not aer_report["gpu_supported"]:
        print("  --gpu requested but this Aer build exposes no GPU device; running on CPU.")
        arguments.gpu = False

    statevector = "aer_simulator_statevector"
    plan = [
        ("default.qubit", "default.qubit", None, False, None),
        ("qiskit.aer default backend", "qiskit.aer", None, arguments.gpu, None),
        ("qiskit.aer statevector", "qiskit.aer", None, arguments.gpu, statevector),
        (
            f"qiskit.aer statevector, {arguments.shots} shots",
            "qiskit.aer",
            arguments.shots,
            arguments.gpu,
            statevector,
        ),
    ]

    results = []
    for label, device_name, shots, use_gpu, aer_backend in plan:
        print(f"\n--- {label} ---")
        device = build_device(device_name, n_qubits, shots, use_gpu, aer_backend)
        record = run_backend(label, device, n_qubits, weights, arguments.repeats)
        results.append(record)
        print(f"forward : {record['forward_seconds'] * 1000:8.2f} ms")
        print(f"backward: {record['backward_seconds'] * 1000:8.2f} ms")
        print(f"probs[:4]: {[round(value, 4) for value in record['first_probabilities']]}")
        print(f"gradient flows: {record['gradient_flows']} (norm {record['gradient_norm']:.4f})")

    def deviation(record: dict[str, object]) -> float:
        reference = results[0]["first_probabilities"]
        return max(
            abs(a - b) for a, b in zip(reference, record["first_probabilities"], strict=True)
        )

    default_backend_deviation = deviation(results[1])
    statevector_deviation = deviation(results[2])
    shot_deviation = deviation(results[3])

    print("\n=== agreement with default.qubit (max |dp| over the first 4 outcomes) ===")
    print(f"Aer default backend        : {default_backend_deviation:.2e}  <- silently sampled")
    print(f"Aer statevector            : {statevector_deviation:.2e}  <- exact, as it should be")
    print(f"Aer statevector {arguments.shots:5d} shots: {shot_deviation:.2e}  <- ~1/sqrt(shots)")

    agrees = statevector_deviation < 1e-9
    print(
        "\nExact Aer reproduces default.qubit."
        if agrees
        else "\nMISMATCH: Aer statevector disagrees with default.qubit."
    )
    if default_backend_deviation > 1e-9:
        print(
            "Confirmed the trap: the default `aer_simulator` backend does NOT give\n"
            "exact probabilities even at shots=None. Always pass\n"
            '  backend="aer_simulator_statevector"'
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "pennylane_version": qml.__version__,
        "qubits": n_qubits,
        "shots": arguments.shots,
        "aer": aer_report,
        "results": results,
        "default_backend_max_deviation": default_backend_deviation,
        "statevector_max_deviation": statevector_deviation,
        "shot_max_deviation": shot_deviation,
        "exact_backends_agree": agrees,
        "default_aer_backend_silently_samples": default_backend_deviation > 1e-9,
    }
    report_path = OUTPUT_DIR / "01_pennylane_aer_circuit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0 if agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
