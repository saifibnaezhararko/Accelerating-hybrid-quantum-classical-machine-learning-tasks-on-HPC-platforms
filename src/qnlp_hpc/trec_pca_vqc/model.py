"""Differentiable angle-encoded variational quantum classifier."""

from __future__ import annotations

import math

import torch
from torch import nn


def apply_ry(
    state: torch.Tensor,
    angles: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a batched Ry rotation to one qubit."""
    batch_size = state.shape[0]
    stride = 1 << (n_qubits - qubit - 1)

    paired = state.reshape(batch_size, -1, 2, stride)
    amplitude_0 = paired[:, :, 0, :]
    amplitude_1 = paired[:, :, 1, :]

    angles = angles.reshape(-1, 1, 1)
    cosine = torch.cos(angles / 2)
    sine = torch.sin(angles / 2)

    rotated_0 = cosine * amplitude_0 - sine * amplitude_1
    rotated_1 = sine * amplitude_0 + cosine * amplitude_1

    return torch.stack(
        (rotated_0, rotated_1),
        dim=2,
    ).reshape(batch_size, -1)


def apply_rz(
    state: torch.Tensor,
    angles: torch.Tensor,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a batched Rz rotation to one qubit."""
    batch_size = state.shape[0]
    stride = 1 << (n_qubits - qubit - 1)

    paired = state.reshape(batch_size, -1, 2, stride)
    amplitude_0 = paired[:, :, 0, :]
    amplitude_1 = paired[:, :, 1, :]

    angles = angles.reshape(-1, 1, 1)

    phase_0 = torch.exp(-0.5j * angles)
    phase_1 = torch.exp(0.5j * angles)

    rotated_0 = phase_0 * amplitude_0
    rotated_1 = phase_1 * amplitude_1

    return torch.stack(
        (rotated_0, rotated_1),
        dim=2,
    ).reshape(batch_size, -1)


def cnot_permutation(
    n_qubits: int,
    control: int,
    target: int,
) -> torch.Tensor:
    """Return the basis-state permutation for a CNOT gate."""
    if control == target:
        raise ValueError("CNOT control and target must be different.")

    basis = torch.arange(1 << n_qubits, dtype=torch.long)

    control_mask = 1 << (n_qubits - control - 1)
    target_mask = 1 << (n_qubits - target - 1)

    control_is_one = (basis & control_mask) != 0

    return torch.where(
        control_is_one,
        basis ^ target_mask,
        basis,
    )


def z_measurement_signs(n_qubits: int) -> torch.Tensor:
    """Construct signs used to calculate every single-qubit Z expectation."""
    basis = torch.arange(1 << n_qubits, dtype=torch.long)
    signs: list[torch.Tensor] = []

    for qubit in range(n_qubits):
        mask = 1 << (n_qubits - qubit - 1)
        bit_is_one = (basis & mask) != 0

        signs.append(
            torch.where(
                bit_is_one,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            )
        )

    return torch.stack(signs, dim=0)


class AngleEncodedVQC(nn.Module):
    """Angle encoding, hardware-efficient VQC, and six-class dense head."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classes: int = 6,
        seed: int = 42,
        measurement_mode: str = "z",
        data_reuploading: bool = False,
    ) -> None:
        super().__init__()

        if n_qubits <= 0:
            raise ValueError("n_qubits must be greater than zero.")

        if n_layers <= 0:
            raise ValueError("n_layers must be greater than zero.")

        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        if measurement_mode not in ("z", "z_zz"):
            raise ValueError(
                "measurement_mode must be either 'z' or 'z_zz'."
            )

        self.measurement_mode = measurement_mode
        self.data_reuploading = data_reuploading

        generator = torch.Generator()
        generator.manual_seed(seed)

        initial_parameters = (
            torch.rand(
                n_layers,
                n_qubits,
                2,
                generator=generator,
            )
            * 0.02
            - 0.01
        )

        # Index 0 is Ry; index 1 is Rz.
        self.quantum_parameters = nn.Parameter(initial_parameters)

        measurement_dimension = (
            n_qubits
            if measurement_mode == "z"
            else 2 * n_qubits
        )

        self.classifier = nn.Linear(
            measurement_dimension,
            n_classes,
        )

        for layer in range(n_layers):
            for control in range(n_qubits):
                target = (control + 1) % n_qubits
                self.register_buffer(
                    f"cnot_{layer}_{control}",
                    cnot_permutation(
                        n_qubits,
                        control,
                        target,
                    ),
                )

        self.register_buffer(
            "z_signs",
            z_measurement_signs(n_qubits),
        )

        self.register_buffer(
            "zz_signs",
            self.z_signs
            * torch.roll(
                self.z_signs,
                shifts=-1,
                dims=0,
            ),
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a batch of |00...0> statevectors."""
        state = torch.zeros(
            batch_size,
            1 << self.n_qubits,
            dtype=torch.complex64,
            device=device,
        )
        state[:, 0] = 1.0 + 0.0j
        return state

    def quantum_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the VQC and return one Z expectation per qubit."""
        if inputs.ndim != 2:
            raise ValueError(
                f"Expected a 2D input tensor, got shape {tuple(inputs.shape)}."
            )

        if inputs.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} angle features, "
                f"got {inputs.shape[1]}."
            )

        inputs = inputs.to(dtype=torch.float32)
        state = self.initial_state(
            batch_size=inputs.shape[0],
            device=inputs.device,
        )

        # Angle encoding: one PCA feature per qubit.
        for qubit in range(self.n_qubits):
            state = apply_ry(
                state,
                inputs[:, qubit],
                qubit,
                self.n_qubits,
            )

        # Hardware-efficient variational layers.
        for layer in range(self.n_layers):
            # The first encoding happened before the loop. Re-upload before
            # every subsequent variational layer when enabled.
            if self.data_reuploading and layer > 0:
                for qubit in range(self.n_qubits):
                    state = apply_ry(
                        state,
                        inputs[:, qubit],
                        qubit,
                        self.n_qubits,
                    )
            for qubit in range(self.n_qubits):
                ry_angle = self.quantum_parameters[layer, qubit, 0]
                rz_angle = self.quantum_parameters[layer, qubit, 1]

                state = apply_ry(
                    state,
                    ry_angle,
                    qubit,
                    self.n_qubits,
                )
                state = apply_rz(
                    state,
                    rz_angle,
                    qubit,
                    self.n_qubits,
                )

            for control in range(self.n_qubits):
                permutation = getattr(
                    self,
                    f"cnot_{layer}_{control}",
                )
                state = state[:, permutation]

        probabilities = state.abs().square()

        z_expectations = (
                probabilities
                @ self.z_signs.transpose(0, 1)
        )

        if self.measurement_mode == "z":
            quantum_features = z_expectations
        else:
            zz_expectations = (
                    probabilities
                    @ self.zz_signs.transpose(0, 1)
            )
            quantum_features = torch.cat(
                (z_expectations, zz_expectations),
                dim=1,
            )

        return quantum_features.to(dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return six unnormalised class logits."""
        quantum_features = self.quantum_features(inputs)
        return self.classifier(quantum_features)