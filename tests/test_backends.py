"""Tests for the backend registry, probe and verification (Week 3, Software Lead)."""

from __future__ import annotations

import pytest

from qnlp_hpc.backends import (
    BACKENDS,
    DEFAULT_BACKEND,
    BackendUnavailableError,
    Capability,
    available_backends,
    check,
    get_spec,
    gpu_simulation_available,
    probe_all,
    require,
    verify_backend,
)
from qnlp_hpc.backends.probe import PENNYLANE_DEVICES
from qnlp_hpc.backends.registry import _capability_met
from qnlp_hpc.backends.verify import build_check_diagrams

VALID_KINDS = {"tensor", "circuit"}
VALID_TRAINERS = {"pytorch", "quantum"}
VALID_DEVICES = {"cpu", "gpu"}


def _capabilities(**overrides: bool) -> dict[str, Capability]:
    """A fully-available capability map, minus whatever the test turns off."""
    names = [
        "lambeq",
        "torch",
        "qiskit",
        "qiskit-aer",
        "pytket",
        "pytket-qiskit",
        "pennylane",
        "cuquantum",
        *(f"pennylane:{device}" for device in PENNYLANE_DEVICES),
    ]
    caps = {
        name: Capability(name, overrides.get(name, True), "test", extras={"gpu": True})
        for name in names
    }
    return caps


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_every_spec_is_well_formed(name: str) -> None:
    spec = BACKENDS[name]
    assert spec.name == name
    assert spec.kind in VALID_KINDS
    assert spec.trainer in VALID_TRAINERS
    assert spec.device in VALID_DEVICES
    assert spec.requires, "a backend with no requirements cannot be probed"
    assert spec.description


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_every_requirement_is_resolvable(name: str) -> None:
    """A typo in `requires` would otherwise mark a backend permanently unusable."""
    capabilities = _capabilities()
    for requirement in BACKENDS[name].requires:
        met, reason = _capability_met(requirement, capabilities)
        assert met, f"{name} -> {requirement}: {reason}"


def test_default_backend_is_registered() -> None:
    assert DEFAULT_BACKEND in BACKENDS


def test_get_spec_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown backend"):
        get_spec("does-not-exist")


def test_check_reports_the_missing_requirement() -> None:
    capabilities = _capabilities(pytket=False)
    ok, reason = check(BACKENDS["tket-aer"], capabilities)
    assert not ok
    assert "pytket" in reason


def test_aer_gpu_requirement_needs_a_gpu_device() -> None:
    capabilities = _capabilities()
    capabilities["qiskit-aer"] = Capability(
        "qiskit-aer", True, "devices: ['CPU']", extras={"devices": ["CPU"], "gpu": False}
    )
    ok, reason = check(BACKENDS["pennylane-qiskit-aer-gpu"], capabilities)
    assert not ok
    assert "no GPU device" in reason


def test_require_raises_with_the_install_hint() -> None:
    capabilities = _capabilities(pytket=False)
    with pytest.raises(BackendUnavailableError, match="poetry install"):
        require("tket-aer", capabilities)


def test_available_backends_covers_the_whole_registry() -> None:
    availability = available_backends(_capabilities())
    assert set(availability) == set(BACKENDS)
    assert all(ok for ok, _ in availability.values())


def test_gpu_detection_reads_the_aer_device_list() -> None:
    capabilities = _capabilities()
    capabilities["qiskit-aer"] = Capability(
        "qiskit-aer", True, "devices: ['CPU', 'GPU']", extras={"gpu": True}
    )
    assert gpu_simulation_available(capabilities)

    capabilities["qiskit-aer"] = Capability("qiskit-aer", True, "", extras={"gpu": False})
    capabilities["pennylane:lightning.gpu"] = Capability("lightning.gpu", False, "missing")
    assert not gpu_simulation_available(capabilities)


def test_probe_covers_every_component_the_registry_asks_for() -> None:
    capabilities = probe_all()
    for spec in BACKENDS.values():
        for requirement in spec.requires:
            key = requirement.split(":")[0] if requirement.endswith(":gpu") else requirement
            assert key in capabilities, f"probe_all() is missing {key!r}"


def test_probe_never_raises_on_this_machine() -> None:
    capabilities = probe_all()
    assert capabilities["lambeq"].available
    assert all(isinstance(capability, Capability) for capability in capabilities.values())


@pytest.mark.parametrize("kind", sorted(VALID_KINDS))
def test_check_diagrams_build_for_both_kinds(kind: str) -> None:
    diagrams = build_check_diagrams(kind)
    assert len(diagrams) == 2


def test_build_check_diagrams_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown backend kind"):
        build_check_diagrams("holographic")


def test_pytorch_tensor_backend_verifies_end_to_end() -> None:
    """The one backend guaranteed present wherever the package installs."""
    result = verify_backend(DEFAULT_BACKEND)

    assert result.available, result.detail
    assert result.ran, result.detail
    assert result.gradient is True, result.detail
    assert result.output_shape == (2, 2)


def test_unavailable_backend_is_skipped_not_failed() -> None:
    capabilities = _capabilities(pytket=False)
    result = verify_backend("tket-aer", capabilities)

    assert not result.available
    assert result.status == "skipped"
