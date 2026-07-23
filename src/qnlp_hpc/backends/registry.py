"""Backend registry — the seam between the quantum layer and the classical ML side.

Each entry names a lambeq model, the trainer it must be paired with, the diagram
kind it consumes, and the capabilities it needs. Nothing here imports an optional
package at module level: the registry can be listed on a machine where none of the
simulators are installed, which is the whole point of ``scripts/check_backends.py``.

Two axes matter when reading this table:

* **kind** — ``tensor`` backends consume ``SpiderAnsatz``/``TensorAnsatz`` diagrams
  (the current MC1 pipeline); ``circuit`` backends consume ``IQPAnsatz``-style
  quantum circuits. They are not interchangeable: switching axis means re-running
  the ansatz, which is the Quantum ML Lead's port.
* **trainer** — ``pytorch`` backends are ``torch.nn.Module``s and differentiate
  through the simulator (autograd/adjoint), so they slot into ``PytorchTrainer``.
  ``quantum`` backends are gradient-free and need ``QuantumTrainer`` with SPSA.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qnlp_hpc.backends.probe import Capability, probe_all


@dataclass(frozen=True)
class BackendSpec:
    """Declarative description of one runnable backend."""

    name: str
    kind: str  # "tensor" | "circuit"
    model: str  # lambeq class name
    trainer: str  # "pytorch" | "quantum"
    device: str  # "cpu" | "gpu"
    requires: tuple[str, ...]
    description: str
    install_hint: str = ""
    model_kwargs: Callable[[], dict[str, Any]] = field(default=dict)

    @property
    def differentiable(self) -> bool:
        """Whether gradients flow back into PyTorch (vs gradient-free SPSA)."""
        return self.trainer == "pytorch"


def _pennylane_kwargs(
    pennylane_device: str,
    diff_method: str | None = None,
    **device_options: Any,
) -> Callable[[], dict[str, Any]]:
    """Build the kwargs for ``PennyLaneModel.from_diagrams``.

    Two different destinations, easy to confuse: ``diff_method`` is a parameter of
    the lambeq model itself, while ``device_options`` go inside ``backend_config``
    and are forwarded to ``qml.device`` — that is how the Aer plugin is switched to
    its GPU statevector (``device="GPU"``).
    """

    def build() -> dict[str, Any]:
        kwargs: dict[str, Any] = {"backend_config": {"backend": pennylane_device, **device_options}}
        if diff_method is not None:
            kwargs["diff_method"] = diff_method
        return kwargs

    return build


def _tket_aer_kwargs() -> dict[str, Any]:
    # Imported lazily — pytket is an optional dependency.
    from pytket.extensions.qiskit import AerBackend

    backend = AerBackend()
    return {
        "backend_config": {
            "backend": backend,
            "compilation": backend.default_compilation_pass(2),
            "shots": 8192,
        }
    }


BACKENDS: dict[str, BackendSpec] = {
    "pytorch-tensor": BackendSpec(
        name="pytorch-tensor",
        kind="tensor",
        model="PytorchModel",
        trainer="pytorch",
        device="cpu",
        requires=("lambeq", "torch"),
        description=(
            "Tensor contraction in PyTorch. What the MC1 pipeline runs today; "
            "uses CUDA automatically when torch reports a device."
        ),
    ),
    "numpy": BackendSpec(
        name="numpy",
        kind="circuit",
        model="NumpyModel",
        trainer="quantum",
        device="cpu",
        requires=("lambeq",),
        description=(
            "Exact statevector simulation in NumPy. No shot noise, no external "
            "simulator — the reference the Aer backends are checked against."
        ),
    ),
    "pennylane-default": BackendSpec(
        name="pennylane-default",
        kind="circuit",
        model="PennyLaneModel",
        trainer="pytorch",
        device="cpu",
        requires=("pennylane", "pennylane:default.qubit", "torch"),
        description="PennyLane reference simulator with torch autograd.",
        install_hint="poetry install --with quantum",
        model_kwargs=_pennylane_kwargs("default.qubit"),
    ),
    "pennylane-lightning": BackendSpec(
        name="pennylane-lightning",
        kind="circuit",
        model="PennyLaneModel",
        trainer="pytorch",
        device="cpu",
        requires=("pennylane", "pennylane:lightning.qubit", "torch"),
        description="Lightning C++ CPU simulator, faster than default.qubit.",
        install_hint="poetry install --with quantum",
        # Not adjoint: lambeq's circuits post-select, and lightning rejects
        # adjoint differentiation for them ("does not support adjoint with
        # requested circuit"). lambeq's default 'best' falls back to
        # parameter-shift. Verified by scripts/check_backends.py --verify.
        model_kwargs=_pennylane_kwargs("lightning.qubit"),
    ),
    "pennylane-lightning-gpu": BackendSpec(
        name="pennylane-lightning-gpu",
        kind="circuit",
        model="PennyLaneModel",
        trainer="pytorch",
        device="gpu",
        requires=("pennylane", "pennylane:lightning.gpu", "torch"),
        description=(
            "Lightning GPU (cuStateVec/cuQuantum) — a target configuration for "
            "the HPC benchmark. Untested locally: no GPU on the dev box."
        ),
        install_hint="pip install pennylane-lightning[gpu]  # needs CUDA 12.x",
        # See the lightning.qubit note: adjoint is left off until it is verified
        # against post-selected circuits on an actual GPU node.
        model_kwargs=_pennylane_kwargs("lightning.gpu"),
    ),
    "pennylane-qiskit-aer": BackendSpec(
        name="pennylane-qiskit-aer",
        kind="circuit",
        model="PennyLaneModel",
        trainer="pytorch",
        device="cpu",
        requires=("pennylane", "pennylane:qiskit.aer", "qiskit-aer", "torch"),
        description=(
            "Qiskit Aer driven through PennyLane — the PyTorch<->Qiskit seam: "
            "Aer simulates, torch autograd trains (parameter-shift)."
        ),
        install_hint="poetry install --with quantum",
        model_kwargs=_pennylane_kwargs("qiskit.aer"),
    ),
    "pennylane-qiskit-aer-gpu": BackendSpec(
        name="pennylane-qiskit-aer-gpu",
        kind="circuit",
        model="PennyLaneModel",
        trainer="pytorch",
        device="gpu",
        requires=("pennylane", "pennylane:qiskit.aer", "qiskit-aer:gpu", "torch"),
        description="Same seam, with Aer's GPU statevector device (Experiment 2).",
        install_hint="poetry install --with gpu  # qiskit-aer-gpu, CUDA 12.4-12.6",
        model_kwargs=_pennylane_kwargs("qiskit.aer", device="GPU"),
    ),
    "tket-aer": BackendSpec(
        name="tket-aer",
        kind="circuit",
        model="TketModel",
        trainer="quantum",
        device="cpu",
        requires=("lambeq", "pytket", "pytket-qiskit"),
        description=(
            "The original prototype path: pytket compiles to Qiskit Aer, "
            "shot-based, trained gradient-free with SPSA."
        ),
        install_hint="poetry install --with quantum",
        model_kwargs=_tket_aer_kwargs,
    ),
}

DEFAULT_BACKEND = "pytorch-tensor"


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend cannot run on this machine."""


def get_spec(name: str) -> BackendSpec:
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(f"Unknown backend {name!r}. Known backends: {sorted(BACKENDS)}") from None


def _capability_met(requirement: str, capabilities: dict[str, Capability]) -> tuple[bool, str]:
    """Resolve one requirement, including the ``name:flag`` form."""
    if requirement == "qiskit-aer:gpu":
        aer = capabilities["qiskit-aer"]
        if not aer.available:
            return False, f"qiskit-aer {aer.detail}"
        if not aer.extras.get("gpu"):
            return False, f"qiskit-aer has no GPU device ({aer.detail})"
        return True, ""

    capability = capabilities.get(requirement)
    if capability is None:
        return False, f"unknown requirement {requirement!r}"
    if not capability.available:
        return False, f"{requirement}: {capability.detail or 'unavailable'}"
    return True, ""


def check(spec: BackendSpec, capabilities: dict[str, Capability] | None = None) -> tuple[bool, str]:
    """Return ``(available, reason)`` for one backend."""
    caps = capabilities if capabilities is not None else probe_all()
    reasons = []
    for requirement in spec.requires:
        met, reason = _capability_met(requirement, caps)
        if not met:
            reasons.append(reason)
    if reasons:
        return False, "; ".join(reasons)
    return True, "ready"


def available_backends(
    capabilities: dict[str, Capability] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Availability and reason for every registered backend."""
    caps = capabilities if capabilities is not None else probe_all()
    return {name: check(spec, caps) for name, spec in BACKENDS.items()}


def require(name: str, capabilities: dict[str, Capability] | None = None) -> BackendSpec:
    """Return the spec, raising a message with the install hint if unusable."""
    spec = get_spec(name)
    ok, reason = check(spec, capabilities)
    if not ok:
        hint = f"\nTry: {spec.install_hint}" if spec.install_hint else ""
        raise BackendUnavailableError(f"Backend {name!r} is not available: {reason}{hint}")
    return spec


def build_model(name: str, diagrams: list[Any], capabilities: dict[str, Capability] | None = None):
    """Instantiate the lambeq model for ``name`` over ``diagrams``.

    No return annotation: the concrete class depends on which optional packages
    are installed on this machine.
    """
    import lambeq

    spec = require(name, capabilities)
    model_class = getattr(lambeq, spec.model)
    model = model_class.from_diagrams(diagrams, **spec.model_kwargs())
    model.initialise_weights()
    return model
