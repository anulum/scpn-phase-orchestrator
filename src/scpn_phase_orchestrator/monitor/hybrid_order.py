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
from numbers import Complex, Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

__all__ = [
    "HybridOrderParameterResult",
    "compute_hybrid_entanglement_order_parameter",
]

CLAIM_BOUNDARY = "quantum_cosimulation_monitor_not_qpu_execution"
BACKEND = "numpy_statevector_density_matrix"
SUPPORTED_SIMULATOR_BACKENDS = frozenset(
    {
        BACKEND,
        "numpy_statevector",
        "numpy_density_matrix",
    }
)


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

    def __post_init__(self) -> None:
        """Validate and normalize immutable published evidence."""
        r_value = _finite_real(self.R, name="R")
        if not 0.0 <= r_value <= 1.0:
            raise ValueError("R must be a finite number in [0, 1]")
        psi_value = _finite_real(self.Psi, name="Psi")
        if not -np.pi <= psi_value <= np.pi:
            raise ValueError("Psi must be a finite phase in [-pi, pi]")
        entropy = _finite_real(self.entanglement_entropy, name="entanglement_entropy")
        if entropy < 0.0:
            raise ValueError("entanglement_entropy must be finite and non-negative")
        normalised = _finite_real(
            self.normalised_entanglement_entropy,
            name="normalised_entanglement_entropy",
        )
        if not 0.0 <= normalised <= 1.0:
            raise ValueError(
                "normalised_entanglement_entropy must be finite and in [0, 1]"
            )
        participation = _finite_real(
            self.participation_ratio, name="participation_ratio"
        )
        qubit_count = _positive_int(self.qubit_count, name="qubit_count")
        partition = _validate_bipartition(
            bipartition=self.bipartition, n_qubits=qubit_count
        )
        max_entropy = float(min(len(partition[0]), len(partition[1])))
        if entropy > max_entropy + 1e-12:
            raise ValueError("entanglement_entropy exceeds the bipartition maximum")
        expected_normalised = entropy / max_entropy
        if not np.isclose(normalised, expected_normalised, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "normalised_entanglement_entropy contradicts entanglement_entropy"
            )
        max_participation = float(1 << len(partition[0]))
        if not 1.0 - 1e-12 <= participation <= max_participation + 1e-12:
            raise ValueError("participation_ratio is outside its bipartition bounds")
        backend = _validate_simulator_backend(self.backend)
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise ValueError("claim_boundary must preserve the no-QPU boundary")
        if self.non_actuating is not True:
            raise ValueError("non_actuating must be exactly True")
        if self.execution_disabled is not True:
            raise ValueError("execution_disabled must be exactly True")

        record = _audit_record_body(
            r_value=r_value,
            psi_value=psi_value,
            entropy=entropy,
            normalised_entropy=normalised,
            participation_ratio=participation,
            qubit_count=qubit_count,
            bipartition=partition,
            backend=backend,
        )
        if not isinstance(self.record_hash, str) or self.record_hash != (
            expected_hash := _deterministic_record_hash(record)
        ):
            raise ValueError("record_hash does not match the canonical evidence")

        object.__setattr__(self, "R", r_value)
        object.__setattr__(self, "Psi", psi_value)
        object.__setattr__(self, "entanglement_entropy", entropy)
        object.__setattr__(self, "normalised_entanglement_entropy", normalised)
        object.__setattr__(self, "participation_ratio", participation)
        object.__setattr__(self, "qubit_count", qubit_count)
        object.__setattr__(self, "bipartition", partition)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "record_hash", expected_hash)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe audit record.
        """
        record = _audit_record_body(
            r_value=self.R,
            psi_value=self.Psi,
            entropy=self.entanglement_entropy,
            normalised_entropy=self.normalised_entanglement_entropy,
            participation_ratio=self.participation_ratio,
            qubit_count=self.qubit_count,
            bipartition=self.bipartition,
            backend=self.backend,
        )
        record["record_hash"] = self.record_hash
        return record


def compute_hybrid_entanglement_order_parameter(
    phases: FloatArray,
    quantum_state: object,
    *,
    qubit_count: int | None = None,
    bipartition: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    simulator_backend: str = BACKEND,
) -> HybridOrderParameterResult:
    """Compute classical R/Psi and the entanglement-aware hybrid order metric.

    Parameters
    ----------
    phases : FloatArray
        Classical phase data.
    quantum_state : object
        Vector of length ``2**n`` or density matrix shape ``(2**n, 2**n)``.
    qubit_count : int | None
        Optional explicit qubit-count override; must match the state.
    bipartition : tuple[tuple[int, ...], tuple[int, ...]] | None
        Optional pair of qubit index groups for reduced entropy.
    simulator_backend : str
        Explicit local simulator contract. The default accepts either statevector or
        density-matrix NumPy inputs; ``"numpy_statevector"`` and
        ``"numpy_density_matrix"`` require the corresponding payload shape and record
        that backend explicitly.

    Returns
    -------
    HybridOrderParameterResult
        HybridOrderParameterResult with a deterministic audit record hash.

    Raises
    ------
    ValueError
        If the quantum state or bipartition is invalid.
    """
    phases_clean = _require_finite_float_array(phases, name="phases")
    r_value, psi_value = compute_order_parameter(phases_clean)

    backend = _validate_simulator_backend(simulator_backend)
    n_qubits, density_matrix, state_kind = _validate_quantum_state(quantum_state)
    if backend == "numpy_statevector" and state_kind != "statevector":
        raise ValueError("simulator_backend numpy_statevector requires a statevector")
    if backend == "numpy_density_matrix" and state_kind != "density_matrix":
        raise ValueError(
            "simulator_backend numpy_density_matrix requires a density matrix"
        )

    if qubit_count is None:
        qubit_count = n_qubits
    else:
        qubit_count = _positive_int(qubit_count, name="qubit_count")
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

    result_payload = _audit_record_body(
        r_value=float(r_value),
        psi_value=float(psi_value),
        entropy=float(entropy),
        normalised_entropy=float(normalised_entropy),
        participation_ratio=float(participation_ratio),
        qubit_count=int(qubit_count),
        bipartition=partition,
        backend=backend,
    )
    result_payload["record_hash"] = _deterministic_record_hash(result_payload)
    return HybridOrderParameterResult(
        R=float(r_value),
        Psi=float(psi_value),
        entanglement_entropy=float(entropy),
        normalised_entanglement_entropy=float(normalised_entropy),
        participation_ratio=float(participation_ratio),
        qubit_count=int(qubit_count),
        bipartition=(tuple(partition[0]), tuple(partition[1])),
        backend=str(result_payload["backend"]),
        claim_boundary=str(result_payload["claim_boundary"]),
        non_actuating=bool(result_payload["non_actuating"]),
        execution_disabled=bool(result_payload["execution_disabled"]),
        record_hash=str(result_payload["record_hash"]),
    )


def _finite_real(value: object, *, name: str) -> float:
    """Return a finite, non-boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_int(value: object, *, name: str) -> int:
    """Return a positive, non-boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _raw_numeric_array(
    values: object, *, name: str, allow_complex: bool
) -> NDArray[np.generic]:
    """Return a non-coercive numeric array with a stable error boundary."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if raw.dtype.kind == "O":
        numeric_type = Complex if allow_complex else Real
        if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
            raise ValueError(f"{name} must not contain boolean values")
        if any(not isinstance(item, numeric_type) for item in raw.flat):
            raise ValueError(f"{name} must be numeric")
    elif raw.dtype.kind == "b":
        raise ValueError(f"{name} must not contain boolean values")
    elif not allow_complex and raw.dtype.kind == "c":
        raise ValueError(f"{name} must be real-valued")
    elif raw.dtype.kind not in ("iufc" if allow_complex else "iuf"):
        raise ValueError(f"{name} must be numeric")
    return raw


def _require_finite_float_array(values: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite float array, else raise."""
    raw = _raw_numeric_array(values, name=name, allow_complex=False)
    array = raw.astype(np.float64, copy=True).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_quantum_state(
    quantum_state: object,
) -> tuple[int, NDArray[np.complex128], str]:
    """Validate a quantum state vector of finite, normalisable amplitudes."""
    raw_state = _raw_numeric_array(
        quantum_state, name="quantum_state", allow_complex=True
    )

    if raw_state.ndim == 0:
        raise ValueError("quantum_state must be a vector or density matrix")
    if raw_state.ndim == 2 and (raw_state.shape[0] == 1 or raw_state.shape[1] == 1):
        raw_state = raw_state.reshape(-1)

    if raw_state.ndim == 1:
        n_qubits, density = _normalise_statevector(raw_state)
        return n_qubits, density, "statevector"

    if raw_state.ndim != 2:
        raise ValueError("quantum_state must be a vector or density matrix")
    if raw_state.shape[0] != raw_state.shape[1]:
        raise ValueError("density matrix must be square")
    density = raw_state.astype(np.complex128, copy=True)
    if not np.all(np.isfinite(density)):
        raise ValueError("quantum_state must contain finite values")
    if not np.allclose(density, density.conj().T, atol=1e-12, rtol=1e-10):
        raise ValueError("density matrix must be Hermitian")
    trace = float(np.trace(density).real)
    if trace <= 0.0:
        raise ValueError("quantum_state density trace must be positive")

    if not np.isclose(trace, 1.0, atol=1e-12, rtol=1e-10):
        density = density / trace

    eigenvalues = np.linalg.eigvalsh(density)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("density matrix eigenvalues must be finite")
    if float(np.min(eigenvalues)) < -1e-12:
        raise ValueError("density matrix must be positive semidefinite")
    purity = float(np.trace(density @ density).real)
    if not np.isclose(purity, 1.0, atol=1e-12, rtol=1e-10):
        raise ValueError(
            "density matrix must represent a pure state for entanglement entropy"
        )

    n_qubits = _qubit_count_from_dimension(density.shape[0])
    return n_qubits, density, "density_matrix"


def _validate_simulator_backend(simulator_backend: object) -> str:
    """Return the supported quantum-simulator backend name, else raise."""
    if not isinstance(simulator_backend, str) or not simulator_backend.strip():
        raise ValueError("simulator_backend must be a supported backend name")
    backend = simulator_backend.strip()
    if backend not in SUPPORTED_SIMULATOR_BACKENDS:
        raise ValueError(
            "simulator_backend must be one of: "
            + ", ".join(sorted(SUPPORTED_SIMULATOR_BACKENDS))
        )
    return backend


def _normalise_statevector(
    vector: NDArray[np.generic],
) -> tuple[int, ComplexArray]:
    """Return the state vector normalised to unit norm, else raise."""
    vector_complex = np.asarray(vector, dtype=np.complex128)
    if not np.all(np.isfinite(vector_complex)):
        raise ValueError("quantum_state must contain finite values")
    dim = vector_complex.size
    if dim < 2:
        raise ValueError("quantum_state vector must contain at least two amplitudes")
    n_qubits = _qubit_count_from_dimension(dim)

    norm = float(np.linalg.norm(vector_complex))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quantum_state vector must have non-zero norm")
    vector_complex = vector_complex / norm
    density: ComplexArray = np.asarray(
        np.outer(vector_complex, np.conj(vector_complex)),
        dtype=np.complex128,
    )
    return n_qubits, density


def _validate_bipartition(
    *,
    bipartition: tuple[tuple[int, ...], tuple[int, ...]] | None,
    n_qubits: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a qubit bipartition covers every qubit exactly once."""
    if bipartition is None:
        if n_qubits < 2:
            raise ValueError("bipartition requires at least two qubits")
        default_left = tuple(range(n_qubits // 2))
        default_right = tuple(range(n_qubits // 2, n_qubits))
        return default_left, default_right

    if not isinstance(bipartition, tuple) and not isinstance(bipartition, list):
        raise ValueError("bipartition must be two index groups")
    if len(bipartition) != 2:
        raise ValueError("bipartition must be two index groups")

    left_raw, right_raw = bipartition
    if not isinstance(left_raw, (tuple, list)) or not isinstance(
        right_raw, (tuple, list)
    ):
        raise ValueError("bipartition must be two index groups")
    left_indices: list[int] = []
    right_indices: list[int] = []
    for idx in left_raw:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral):
            raise ValueError("bipartition indices must be integers")
        idx = int(idx)
        if idx < 0 or idx >= n_qubits:
            raise ValueError("bipartition index out of range")
        left_indices.append(idx)
    for idx in right_raw:
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral):
            raise ValueError("bipartition indices must be integers")
        idx = int(idx)
        if idx < 0 or idx >= n_qubits:
            raise ValueError("bipartition index out of range")
        right_indices.append(idx)

    if not left_indices or not right_indices:
        raise ValueError("bipartition groups must be non-empty")
    if len(set(left_indices)) != len(left_indices) or len(set(right_indices)) != len(
        right_indices
    ):
        raise ValueError("bipartition groups must contain unique indices")
    left_set = set(left_indices)
    right_set = set(right_indices)
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
    """Return the reduced density matrix over a qubit subsystem."""
    subsystem_b = tuple(i for i in range(n_qubits) if i not in subsystem_a)
    if len(subsystem_a) < 1 or len(subsystem_b) < 1:
        raise ValueError("bipartition must split into two non-empty groups")

    dim_a = 1 << len(subsystem_a)
    dim_b = 1 << len(subsystem_b)

    tensor = density_matrix.reshape((2,) * (2 * n_qubits))
    ordering = tuple(subsystem_a) + tuple(subsystem_b)
    permutation = list(ordering + tuple(index + n_qubits for index in ordering))
    tensor = np.transpose(tensor, permutation)
    tensor = tensor.reshape((dim_a, dim_b, dim_a, dim_b))
    return np.asarray(np.einsum("abcb->ac", tensor), dtype=np.complex128)


def _von_neumann_entropy(
    reduced_density_matrix: NDArray[np.complex128],
) -> tuple[float, float]:
    """Return the von Neumann entropy of a density matrix."""
    eigenvalues = np.linalg.eigvalsh(reduced_density_matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("reduced density matrix eigenvalues must be finite")
    eigvals = np.real(eigenvalues)
    eigvals = np.clip(eigvals, 0.0, 1.0)
    total = float(np.sum(eigvals))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("reduced density matrix could not be normalised")
    eigvals = eigvals / total
    mask = eigvals > 0.0
    entropy = -float(np.sum(eigvals[mask] * np.log2(eigvals[mask])))
    purity = float(np.sum(eigvals**2))
    participation_ratio = 1.0 / purity
    return entropy, participation_ratio


def _audit_record_body(
    *,
    r_value: float,
    psi_value: float,
    entropy: float,
    normalised_entropy: float,
    participation_ratio: float,
    qubit_count: int,
    bipartition: tuple[tuple[int, ...], tuple[int, ...]],
    backend: str,
) -> dict[str, object]:
    """Return the canonical unhashed audit-record body."""
    return {
        "R": r_value,
        "Psi": psi_value,
        "entanglement_entropy": entropy,
        "normalised_entanglement_entropy": normalised_entropy,
        "participation_ratio": participation_ratio,
        "qubit_count": qubit_count,
        "bipartition": [list(bipartition[0]), list(bipartition[1])],
        "backend": backend,
        "claim_boundary": CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
    }


def _deterministic_record_hash(record: dict[str, object]) -> str:
    """Return the canonical-JSON SHA-256 hash of a record."""
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _qubit_count_from_dimension(dim: int) -> int:
    """Return the qubit count for a power-of-two state dimension, else raise."""
    if dim < 2:
        raise ValueError("quantum_state must contain at least two amplitudes")
    qubits = int(np.log2(dim))
    if (1 << qubits) != dim:
        raise ValueError("quantum_state Hilbert dimension must be a power of two")
    return qubits
