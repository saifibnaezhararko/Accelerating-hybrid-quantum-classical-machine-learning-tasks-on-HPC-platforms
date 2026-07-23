"""Quantum <-> classical interop (Qiskit Aer, PyTorch, JAX, PennyLane).

``registry`` declares the backends, ``probe`` reports what this machine can run,
and ``verify`` proves it by running a forward/backward pass. Nothing imports an
optional simulator at module level, so this package is safe to import anywhere.
"""

from qnlp_hpc.backends.probe import Capability, gpu_simulation_available, probe_all
from qnlp_hpc.backends.registry import (
    BACKENDS,
    DEFAULT_BACKEND,
    BackendSpec,
    BackendUnavailableError,
    available_backends,
    build_model,
    check,
    get_spec,
    require,
)
from qnlp_hpc.backends.verify import Verification, verify_all, verify_backend

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "BackendSpec",
    "BackendUnavailableError",
    "Capability",
    "Verification",
    "available_backends",
    "build_model",
    "check",
    "get_spec",
    "gpu_simulation_available",
    "probe_all",
    "require",
    "verify_all",
    "verify_backend",
]
