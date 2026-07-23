"""Capability probing — what quantum/ML backends this machine can actually run.

The point is to answer "is the GPU simulator operational?" with evidence rather
than with an install log. Every check is defensive: a missing or broken optional
package produces a reason string, never an exception, so the probe runs on a bare
laptop and on a fully provisioned HPC node alike.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capability:
    """One probed capability."""

    name: str
    available: bool
    detail: str = ""
    version: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "yes" if self.available else "no"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_import(module: str, distribution: str | None = None) -> Capability:
    """Import a module, reporting the distribution version if it succeeds."""
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return Capability(module, False, f"not installed ({exc})")
    except Exception as exc:  # pragma: no cover - broken install
        return Capability(module, False, f"import failed: {type(exc).__name__}: {exc}")
    return Capability(module, True, version=_version(distribution or module))


def probe_torch() -> Capability:
    try:
        import torch
    except ImportError as exc:
        return Capability("torch", False, f"not installed ({exc})")

    cuda = bool(torch.cuda.is_available())
    devices = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
    )
    detail = (
        f"CUDA {torch.version.cuda}, {len(devices)} device(s): {devices}" if cuda else "CPU only"
    )
    return Capability(
        "torch",
        True,
        detail,
        version=torch.__version__,
        extras={"cuda": cuda, "devices": devices},
    )


def probe_aer() -> Capability:
    """Qiskit Aer, and which simulation devices it reports.

    ``available_devices()`` is the honest GPU test: qiskit-aer and qiskit-aer-gpu
    install the same import name, so only the device list distinguishes them.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        return Capability("qiskit-aer", False, f"not installed ({exc})")

    try:
        devices = list(AerSimulator().available_devices())
    except Exception as exc:  # pragma: no cover - driver/runtime failure
        return Capability(
            "qiskit-aer",
            True,
            f"installed but device query failed: {type(exc).__name__}: {exc}",
            version=_version("qiskit-aer"),
        )

    return Capability(
        "qiskit-aer",
        True,
        f"devices: {devices}",
        version=_version("qiskit-aer"),
        extras={"devices": devices, "gpu": "GPU" in devices},
    )


def probe_pennylane_device(device_name: str) -> Capability:
    """Try to construct a PennyLane device — the plugin entry point can lie."""
    try:
        import pennylane as qml
    except ImportError as exc:
        return Capability(device_name, False, f"pennylane not installed ({exc})")

    try:
        qml.device(device_name, wires=1)
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return Capability(device_name, False, message[:120])
    return Capability(device_name, True, version=_version("pennylane"))


PENNYLANE_DEVICES = (
    "default.qubit",
    "lightning.qubit",
    "lightning.gpu",
    "qiskit.aer",
)


def probe_all() -> dict[str, Capability]:
    """Probe every capability the project can make use of."""
    capabilities: dict[str, Capability] = {
        "lambeq": _probe_import("lambeq"),
        "torch": probe_torch(),
        "qiskit": _probe_import("qiskit"),
        "qiskit-aer": probe_aer(),
        "pytket": _probe_import("pytket"),
        "pytket-qiskit": _probe_import("pytket.extensions.qiskit", "pytket-qiskit"),
        "pennylane": _probe_import("pennylane"),
        "cuquantum": _probe_import("cuquantum", "cuquantum-python"),
    }
    for device in PENNYLANE_DEVICES:
        capabilities[f"pennylane:{device}"] = probe_pennylane_device(device)
    return capabilities


def gpu_simulation_available(capabilities: dict[str, Capability] | None = None) -> bool:
    """True when at least one GPU-backed quantum simulator is usable."""
    caps = capabilities or probe_all()
    aer_gpu = bool(caps["qiskit-aer"].extras.get("gpu"))
    lightning_gpu = caps["pennylane:lightning.gpu"].available
    return aer_gpu or lightning_gpu
