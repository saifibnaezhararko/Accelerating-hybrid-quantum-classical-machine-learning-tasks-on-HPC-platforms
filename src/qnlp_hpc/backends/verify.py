"""End-to-end verification that a backend really runs, not just that it imports.

For every registered backend this builds two real diagrams, runs a forward pass,
and — where the backend is differentiable — a backward pass, checking that gradient
actually reaches the circuit parameters. That is the difference between "qiskit-aer
is installed" and "the PyTorch<->Qiskit seam works on this node".

Used by ``scripts/check_backends.py --verify``; on the HPC node it is the evidence
for the "GPU simulator operational" deliverable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qnlp_hpc.backends.probe import Capability, probe_all
from qnlp_hpc.backends.registry import BACKENDS, BackendSpec, build_model, check

# Short, grammatically simple, and drawn from the MC1 vocabulary so the circuits
# resemble the ones the real experiment builds.
CHECK_SENTENCES = ("cook creates dish", "hacker writes code")


@dataclass(frozen=True)
class Verification:
    backend: str
    available: bool
    ran: bool
    detail: str
    seconds: float | None = None
    output_shape: tuple[int, ...] | None = None
    gradient: bool | None = None

    @property
    def status(self) -> str:
        if not self.available:
            return "skipped"
        return "ok" if self.ran else "FAILED"


def build_check_diagrams(kind: str, sentences: tuple[str, ...] = CHECK_SENTENCES) -> list:
    """Build diagrams of the right kind for a backend.

    ``tensor`` and ``circuit`` backends consume different objects, so the ansatz is
    chosen from the backend's kind rather than assumed.
    """
    from lambeq import AtomicType, IQPAnsatz, SpiderAnsatz, spiders_reader

    diagrams = spiders_reader.sentences2diagrams(list(sentences))

    if kind == "tensor":
        from lambeq.backend.tensor import Dim

        ansatz = SpiderAnsatz(
            {AtomicType.NOUN: Dim(2), AtomicType.SENTENCE: Dim(2)},
            max_order=2,
        )
    elif kind == "circuit":
        ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1)
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"Unknown backend kind {kind!r}.")

    return [ansatz(diagram) for diagram in diagrams]


def _check_gradient(output) -> bool:
    """Backward through a non-constant scalar and confirm a parameter moved.

    Not ``output.sum()``: lambeq's PennyLane models return normalised
    probabilities, whose row sums are identically 1, so that gradient is zero by
    construction and would look like a broken backend.
    """
    import torch

    if not isinstance(output, torch.Tensor) or not output.requires_grad:
        return False

    scalar = output[:, 0].sum() if output.ndim > 1 else output.sum()
    scalar.backward()
    return True


def verify_backend(
    name: str,
    capabilities: dict[str, Capability] | None = None,
) -> Verification:
    """Run one backend end to end, converting any failure into a report."""
    spec: BackendSpec = BACKENDS[name]
    caps = capabilities if capabilities is not None else probe_all()

    is_available, reason = check(spec, caps)
    if not is_available:
        return Verification(name, False, False, reason)

    start = time.perf_counter()
    try:
        diagrams = build_check_diagrams(spec.kind)
        model = build_model(name, diagrams, caps)
        output = model(diagrams)

        shape = tuple(getattr(output, "shape", ()))
        gradient: bool | None = None
        if spec.differentiable:
            gradient = _check_gradient(output)
            parameters = [p for p in model.parameters() if p.grad is not None]
            if not parameters:
                raise RuntimeError("forward ran but no parameter received a gradient")
    except Exception as exc:
        return Verification(
            name,
            True,
            False,
            f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    detail = f"{len(diagrams)} diagrams, output {shape}"
    if gradient is not None:
        detail += ", gradients flow" if gradient else ", NO gradient"
    return Verification(name, True, True, detail, elapsed, shape, gradient)


def verify_all(capabilities: dict[str, Capability] | None = None) -> list[Verification]:
    caps = capabilities if capabilities is not None else probe_all()
    return [verify_backend(name, caps) for name in BACKENDS]
