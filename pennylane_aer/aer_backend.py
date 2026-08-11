from __future__ import annotations

from typing import Any

AER_STATEVECTOR = "aer_simulator_statevector"


def aer_backend_config(
    *,
    gpu: bool = False,
    shots: int | None = None,
    precision: str = "double",
) -> dict[str, Any]:
    if gpu or shots is not None or precision != "double":
        from qiskit_aer import AerSimulator

        options: dict[str, Any] = {"method": "statevector", "precision": precision}
        if gpu:
            options["device"] = "GPU"
        simulator = AerSimulator(**options)
        config: dict[str, Any] = {"backend": "qiskit.aer", "device": simulator}
    else:
        config = {"backend": "qiskit.aer", "device": AER_STATEVECTOR}

    if shots is not None:
        config["shots"] = shots
    return config


def aer_gpu_available() -> bool:
    try:
        from qiskit_aer import AerSimulator

        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


def describe_backend(config: dict[str, Any] | None) -> str:
    if config is None:
        return "pennylane:default.qubit (exact)"

    device = config.get("device")
    device_name = getattr(device, "name", device)
    if callable(device_name):
        device_name = device_name()
    shots = config.get("shots")
    suffix = f", {shots} shots" if shots else ", exact"
    return f"{config.get('backend')}:{device_name}{suffix}"


def copy_weights(source, target) -> None:
    import torch

    source_by_name = {
        str(symbol): weight for symbol, weight in zip(source.symbols, source.weights, strict=True)
    }

    missing = [str(symbol) for symbol in target.symbols if str(symbol) not in source_by_name]
    if missing:
        raise ValueError(f"Source model is missing symbols required by the target: {missing}")

    with torch.no_grad():
        for symbol, weight in zip(target.symbols, target.weights, strict=True):
            value = source_by_name[str(symbol)]
            weight.copy_(torch.as_tensor(value, dtype=weight.dtype).reshape(weight.shape))
