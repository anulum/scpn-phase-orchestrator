# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entanglement-aware hybrid order parameter monitor

"""Classical+quantum co-simulation order monitor.

Computes Kuramoto synchrony and qubit-partition entanglement entropy from either
statevectors or density matrices using NumPy only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

__all__ = [
    "HybridOrderParameterResult",
    "compute_hybrid_entanglement_order_parameter",
]

CLAIM_BOUNDARY = "quantum_cosimulation_monitor_not_qpu_execution"
BACKEND = "numpy_statevector_density_matrix"


@dataclass(frozen=True)
class HybridOrderParameterResult:
    """Result of a hybrid classical-quantum order-parameter evaluation."""

    R: float
    Psi: float
    entanglement_entropy: float
    normalised_entanglement_entropy: float
    participation_ratio: float
    qubit_count: int
    bipartition: tuple[tuple[int, ...], tuple[int, ...]]
    backend: str
    claim_boundary: str
    non_actuating: bool
    execution_disabled: bool
    record_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record."""
        record: dict[str, object] = {
            "R": self.R,
            "Psi": self.Psi,
            "entanglement_entropy": self.entanglement_entropy,
            "normalised_entanglement_entropy": self.normalised_entanglement_entropy,
            "participation_ratio": self.participation_ratio,
            "qubit_count": self.qubit_count,
            "bipartition": [list(self.bipartition[0]), list(self.bipartition[1])],
            "backend": self.backend,
            "claim_boundary": self.claim_boundary,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
        }
        record["record_hash"] = _deterministic_record_hash(record)
        return record


def compute_hybrid_entanglement_order_parameter(
    phases: FloatArray,
    quantum_state: object,
    *,
    qubit_count: int | None = None,
    bipartition: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> HybridOrderParameterResult:
    """Compute classical R/Psi and the entanglement-aware hybrid order metric.

    Args:
        phases: Classical phase data.
        quantum_state: Vector of length ``2**n`` or density matrix shape
            ``(2**n, 2**n)``.
        qubit_count: Optional explicit qubit-count override; must match the state.
        bipartition: Optional pair of qubit index groups for reduced entropy.

    Returns:
        HybridOrderParameterResult with a deterministic audit record hash.
    """
    phases_clean = _require_finite_float_array(phases, name="phases")
    r_value, psi_value = compute_order_parameter(phases_clean)

    n_qubits, density_matrix = _validate_quantum_state(quantum_state)

    if qubit_count is None:
        qubit_count = n_qubits
    else:
        if not isinstance(qubit_count, int) or qubit_count < 1:
            raise ValueError("qubit_count must be a positive integer")
        if qubit_count != n_qubits:
            raise ValueError("qubit_count is inconsistent with quantum_state size")

    partition = _validate_bipartition(bipartition=bipartition, n_qubits=qubit_count)
    reduced = _reduced_density_matrix(
        density_matrix=density_matrix,
        subsystem_a=partition[0],
        n_qubits=qubit_count,
    )
    entropy, participation_ratio = _von_neumann_entropy(reduced)
    max_entropy = float(min(len(partition[0]), len(partition[1])))
    normalised_entropy = 0.0 if max_entropy <= 0.0 else entropy / max_entropy
    normalised_entropy = float(np.clip(normalised_entropy, 0.0, 1.0))

    result_payload = {
        "R": r_value,
        "Psi": psi_value,
        "entanglement_entropy": entropy,
        "normalised_entanglement_entropy": normalised_entropy,
        "participation_ratio": participation_ratio,
        "qubit_count": qubit_count,
        "bipartition": [list(partition[0]), list(partition[1])],
        "backend": BACKEND,
        "claim_boundary": CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
    }
    result_payload["record_hash"] = _deterministic_record_hash(result_payload)
    return HybridOrderParameterResult(
        R=float(result_payload["R"]),
        Psi=float(result_payload["Psi"]),
        entanglement_entropy=float(result_payload["entanglement_entropy"]),
        normalised_entanglement_entropy=float(
            result_payload["normalised_entanglement_entropy"],
        ),
        participation_ratio=float(result_payload["participation_ratio"]),
        qubit_count=int(result_payload["qubit_count"]),
        bipartition=(tuple(partition[0]), tuple(partition[1])),
        backend=str(result_payload["backend"]),
        claim_boundary=str(result_payload["claim_boundary"]),
        non_actuating=bool(result_payload["non_actuating"]),
        execution_disabled=bool(result_payload["execution_disabled"]),
        record_hash=str(result_payload["record_hash"]),
    )


def _require_finite_float_array(values: object, *, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_quantum_state(
    quantum_state: object,
) -> tuple[int, NDArray[np.complex128]]:
    raw_state = np.asarray(quantum_state)

    if raw_state.ndim == 0:
        raise ValueError("quantum_state must be a vector or density matrix")
    if raw_state.ndim == 2 and (raw_state.shape[0] == 1 or raw_state.shape[1] == 1):
        raw_state = raw_state.reshape(-1)

    if raw_state.ndim == 1:
        return _normalise_statevector(raw_state)

    if raw_state.ndim != 2:
        raise ValueError("quantum_state must be a vector or density matrix")
    if raw_state.shape[0] != raw_state.shape[1]:
        raise ValueError("density matrix must be square")
    if raw_state.dtype == np.bool_:
        raise ValueError("quantum_state must be numeric")

    density = raw_state.astype(np.complex128, copy=True)
    if not np.all(np.isfinite(density)):
        raise ValueError("quantum_state must contain finite values")
    if not np.allclose(density, density.conj().T, atol=1e-12, rtol=1e-10):
        raise ValueError("density matrix must be Hermitian")
    if not np.isfinite(np.trace(density)):
        raise ValueError("quantum_state density trace must be finite")

    trace = float(np.trace(density).real)
    if trace <= 0.0:
        raise ValueError("quantum_state density trace must be positive")

    if not np.isclose(trace, 1.0, atol=1e-12, rtol=1e-10):
        density = density / trace

    n_qubits = _qubit_count_from_dimension(density.shape[0])
    return n_qubits, density


def _normalise_statevector(
    vector: object,
) -> tuple[int, ComplexArray]:
    if vector.dtype == np.bool_:
        raise ValueError("quantum_state must be numeric")
    vector_complex = np.asarray(vector, dtype=np.complex128)
    if not np.all(np.isfinite(vector_complex)):
        raise ValueError("quantum_state must contain finite values")
    if vector_complex.ndim != 1:
        raise ValueError("quantum_state vector must be one-dimensional")

    dim = vector_complex.size
    if dim < 2:
        raise ValueError("quantum_state vector must contain at least two amplitudes")
    n_qubits = _qubit_count_from_dimension(dim)

    norm = float(np.linalg.norm(vector_complex))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quantum_state vector must have non-zero norm")
    vector_complex = vector_complex / norm
    density = np.outer(vector_complex, np.conj(vector_complex))
    return n_qubits, density


def _validate_bipartition(
    *,
    bipartition: tuple[tuple[int, ...], tuple[int, ...]] | None,
    n_qubits: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if bipartition is None:
        if n_qubits < 2:
            raise ValueError("bipartition requires at least two qubits")
        left = tuple(range(n_qubits // 2))
        right = tuple(range(n_qubits // 2, n_qubits))
        return left, right

    if not isinstance(bipartition, tuple) and not isinstance(bipartition, list):
        raise ValueError("bipartition must be two index groups")
    if len(bipartition) != 2:
        raise ValueError("bipartition must be two index groups")

    left_raw, right_raw = bipartition
    left: list[int] = []
    right: list[int] = []
    for idx in left_raw:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError("bipartition indices must be integers")
        if idx < 0 or idx >= n_qubits:
            raise ValueError("bipartition index out of range")
        left.append(int(idx))
    for idx in right_raw:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError("bipartition indices must be integers")
        if idx < 0 or idx >= n_qubits:
            raise ValueError("bipartition index out of range")
        right.append(int(idx))

    if not left or not right:
        raise ValueError("bipartition groups must be non-empty")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("bipartition groups must contain unique indices")
    left_set = set(left)
    right_set = set(right)
    if left_set & right_set:
        raise ValueError("bipartition groups must be disjoint")
    all_indices = left_set | right_set
    if len(all_indices) != n_qubits:
        raise ValueError("bipartition must cover every qubit exactly once")

    return tuple(sorted(left_set)), tuple(sorted(right_set))


def _reduced_density_matrix(
    density_matrix: NDArray[np.complex128],
    *,
    subsystem_a: tuple[int, ...],
    n_qubits: int,
) -> NDArray[np.complex128]:
    subsystem_b = tuple(i for i in range(n_qubits) if i not in subsystem_a)
    if len(subsystem_a) < 1 or len(subsystem_b) < 1:
        raise ValueError("bipartition must split into two non-empty groups")

    dim_a = 1 << len(subsystem_a)
    dim_b = 1 << len(subsystem_b)

    tensor = density_matrix.reshape((2,) * (2 * n_qubits))
    ordering = tuple(subsystem_a) + tuple(subsystem_b)
    permutation = ordering + tuple(index + n_qubits for index in ordering)
    tensor = np.transpose(tensor, permutation)
    tensor = tensor.reshape((dim_a, dim_b, dim_a, dim_b))
    return np.einsum("abcb->ac", tensor)


def _von_neumann_entropy(
    reduced_density_matrix: NDArray[np.complex128],
) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvalsh(reduced_density_matrix)
    eigvals = np.real(eigenvalues)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    total = float(np.sum(eigvals))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("reduced density matrix could not be normalised")
    eigvals = eigvals / total
    mask = eigvals > 0.0
    if not np.any(mask):
        return 0.0, 0.0

    entropy = -float(np.sum(eigvals[mask] * np.log2(eigvals[mask])))
    if not np.isfinite(entropy):
        entropy = 0.0
    purity = float(np.sum(eigvals**2))
    participation_ratio = 0.0 if purity <= 0.0 else 1.0 / purity
    return entropy, participation_ratio


def _deterministic_record_hash(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _qubit_count_from_dimension(dim: int) -> int:
    if dim < 2:
        raise ValueError("quantum_state must contain at least two amplitudes")
    qubits = int(np.log2(dim))
    if (1 << qubits) != dim:
        raise ValueError("quantum_state Hilbert dimension must be a power of two")
    return qubits
