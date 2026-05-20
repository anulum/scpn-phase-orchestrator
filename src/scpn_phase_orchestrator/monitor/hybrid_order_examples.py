# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid order-parameter audit scenarios

"""Deterministic scenario fixtures for quantum co-simulation audit evidence.

The fixtures model hybrid order-parameter audits (entanglement entropy plus
classical synchrony metrics) for non-actuating review workflows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

HybridBoundary: str = "quantum_cosimulation_monitor_not_qpu_execution"
AllowedDomains = ("quantum_simulation", "power_grid", "cardiac_rhythm")


def _ensure_float64_vector(values: object, *, label: str) -> FloatArray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be convertible to float64 array") from exc

    if arr.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _summary(values: FloatArray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _summary_from_complex(values: ComplexArray) -> dict[str, float]:
    magnitudes = np.abs(values)
    return {
        "count": int(values.size),
        "min": float(np.min(magnitudes)),
        "max": float(np.max(magnitudes)),
        "mean": float(np.mean(magnitudes)),
        "std": float(np.std(magnitudes)),
    }


@dataclass
class HybridStateCandidate:
    """Deterministic candidate state description for a scenario."""

    state_id: str
    candidate_type: str
    amplitudes: ComplexArray
    entanglement_entropy: float
    order_metric_r: float
    order_metric_psi: float
    objective_labels: tuple[str, ...]
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = HybridBoundary


@dataclass
class HybridOrderScenario:
    """One deterministic scenario with review-safe outputs."""

    domain: str
    scenario_id: str
    phases: FloatArray
    qubit_count: int
    bipartition: tuple[tuple[int, ...], tuple[int, ...]]
    state_candidates: tuple[HybridStateCandidate, ...]
    objective_labels: tuple[str, ...]
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = HybridBoundary
    scenario_hash: str = ""


def _validate_state_candidate(
    candidate: HybridStateCandidate,
    *,
    scenario_id: str,
    qubit_count: int,
) -> None:
    if not isinstance(candidate.state_id, str) or not candidate.state_id.strip():
        raise ValueError(f"candidate in {scenario_id} requires non-empty state_id")
    if candidate.candidate_type not in {"product", "entangled"}:
        raise ValueError(
            f"candidate {candidate.state_id} requires type product or entangled"
        )
    if not isinstance(candidate.amplitudes, np.ndarray):
        raise ValueError(
            f"candidate {candidate.state_id} amplitudes must be an ndarray"
        )
    if candidate.amplitudes.ndim != 1:
        raise ValueError(
            f"candidate {candidate.state_id} amplitudes must be one-dimensional"
        )
    if candidate.amplitudes.dtype.kind not in {"c", "f"}:
        raise ValueError(
            f"candidate {candidate.state_id} amplitudes must be complex64/complex128"
        )
    if candidate.amplitudes.size != 2**qubit_count:
        raise ValueError(
            f"candidate {candidate.state_id} amplitudes must have size 2**qubit_count"
        )
    if not np.all(np.isfinite(np.abs(candidate.amplitudes))):
        raise ValueError(f"candidate {candidate.state_id} amplitudes must be finite")
    if candidate.non_actuating is not True:
        raise ValueError(f"candidate {candidate.state_id} must set non_actuating=True")
    if candidate.execution_disabled is not True:
        raise ValueError(
            f"candidate {candidate.state_id} must set execution_disabled=True"
        )
    if candidate.claim_boundary != HybridBoundary:
        raise ValueError(f"candidate {candidate.state_id} has invalid claim_boundary")
    if not math.isfinite(candidate.entanglement_entropy):
        raise ValueError(
            f"candidate {candidate.state_id} entanglement_entropy must be finite"
        )
    if not math.isfinite(candidate.order_metric_r):
        raise ValueError(
            f"candidate {candidate.state_id} order_metric_r must be finite"
        )
    if not math.isfinite(candidate.order_metric_psi):
        raise ValueError(
            f"candidate {candidate.state_id} order_metric_psi must be finite"
        )
    if not candidate.objective_labels:
        raise ValueError(f"candidate {candidate.state_id} requires objective labels")
    if not all(
        isinstance(label, str) and label.strip() for label in candidate.objective_labels
    ):
        raise ValueError(f"candidate {candidate.state_id} objective labels invalid")


def _validate_bipartition(
    bipartition: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    qubit_count: int,
    scenario_id: str,
) -> None:
    if (
        not isinstance(bipartition, tuple)
        or len(bipartition) != 2
        or not isinstance(bipartition[0], tuple)
        or not isinstance(bipartition[1], tuple)
    ):
        raise ValueError(f"{scenario_id} bipartition must be a pair of tuples")
    left, right = bipartition
    merged = left + right
    if len(merged) != qubit_count:
        raise ValueError(f"{scenario_id} bipartition must cover all qubits")
    if len(set(merged)) != qubit_count:
        raise ValueError(f"{scenario_id} bipartition must contain unique indices")
    if any(i < 0 or i >= qubit_count for i in merged):
        raise ValueError(f"{scenario_id} bipartition contains invalid qubit index")


def _validate_scenario(scenario: HybridOrderScenario) -> None:
    if not isinstance(scenario.domain, str) or scenario.domain not in AllowedDomains:
        raise ValueError("invalid scenario domain")
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    if scenario.qubit_count < 2:
        raise ValueError("qubit_count must be at least two")

    phases = _ensure_float64_vector(
        scenario.phases, label=f"{scenario.scenario_id}.phases"
    )
    if phases.size != scenario.qubit_count:
        raise ValueError(f"{scenario.scenario_id} requires len(phases)=qubit_count")

    _validate_bipartition(
        scenario.bipartition,
        qubit_count=scenario.qubit_count,
        scenario_id=scenario.scenario_id,
    )

    if scenario.non_actuating is not True:
        raise ValueError(f"scenario {scenario.scenario_id} must set non_actuating=True")
    if scenario.execution_disabled is not True:
        raise ValueError(
            f"scenario {scenario.scenario_id} must set execution_disabled=True"
        )
    if scenario.claim_boundary != HybridBoundary:
        raise ValueError(f"scenario {scenario.scenario_id} has invalid claim_boundary")

    if not scenario.state_candidates:
        raise ValueError(f"scenario {scenario.scenario_id} requires candidates")
    if len(scenario.state_candidates) < 2:
        raise ValueError(
            f"scenario {scenario.scenario_id} must include at least two candidates"
        )
    candidate_types = [
        candidate.candidate_type for candidate in scenario.state_candidates
    ]
    if "product" not in candidate_types:
        raise ValueError(
            f"scenario {scenario.scenario_id} must include a product state"
        )
    if "entangled" not in candidate_types:
        raise ValueError(
            f"scenario {scenario.scenario_id} must include an entangled state"
        )
    for candidate in scenario.state_candidates:
        _validate_state_candidate(
            candidate,
            scenario_id=scenario.scenario_id,
            qubit_count=scenario.qubit_count,
        )

    if not scenario.objective_labels:
        raise ValueError(f"scenario {scenario.scenario_id} requires objective labels")
    if not all(
        isinstance(label, str) and label.strip() for label in scenario.objective_labels
    ):
        raise ValueError(f"scenario {scenario.scenario_id} objective labels invalid")

    expected_hash = _compute_scenario_hash(scenario)
    if scenario.scenario_hash and scenario.scenario_hash != expected_hash:
        raise ValueError(
            f"scenario {scenario.scenario_id} has mismatched scenario_hash"
        )


def _compute_scenario_hash(scenario: HybridOrderScenario) -> str:
    payload = {
        "domain": scenario.domain,
        "scenario_id": scenario.scenario_id,
        "qubit_count": scenario.qubit_count,
        "bipartition": [list(part) for part in scenario.bipartition],
        "phases": [float(v) for v in scenario.phases.tolist()],
        "state_candidates": [
            {
                "state_id": candidate.state_id,
                "candidate_type": candidate.candidate_type,
                "amplitudes": [
                    [float(v.real), float(v.imag)]
                    for v in candidate.amplitudes.tolist()
                ],
                "entanglement_entropy": float(candidate.entanglement_entropy),
                "order_metric_r": float(candidate.order_metric_r),
                "order_metric_psi": float(candidate.order_metric_psi),
                "objective_labels": list(candidate.objective_labels),
                "non_actuating": candidate.non_actuating,
                "execution_disabled": candidate.execution_disabled,
                "claim_boundary": candidate.claim_boundary,
            }
            for candidate in sorted(scenario.state_candidates, key=lambda c: c.state_id)
        ],
        "objective_labels": list(scenario.objective_labels),
        "non_actuating": scenario.non_actuating,
        "execution_disabled": scenario.execution_disabled,
        "claim_boundary": scenario.claim_boundary,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _to_record(scenario: HybridOrderScenario) -> dict[str, object]:
    return {
        "domain": scenario.domain,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "qubit_count": scenario.qubit_count,
        "bipartition": [list(part) for part in scenario.bipartition],
        "phases": [float(v) for v in scenario.phases.tolist()],
        "phases_summary": _summary(
            _ensure_float64_vector(scenario.phases, label="phases")
        ),
        "non_actuating": scenario.non_actuating,
        "execution_disabled": scenario.execution_disabled,
        "claim_boundary": scenario.claim_boundary,
        "objective_labels": list(scenario.objective_labels),
        "state_candidates": [
            {
                "state_id": candidate.state_id,
                "state_type": candidate.candidate_type,
                "amplitudes": [
                    [float(value.real), float(value.imag)]
                    for value in candidate.amplitudes.tolist()
                ],
                "amplitude_summary": _summary_from_complex(candidate.amplitudes),
                "entanglement_entropy": float(candidate.entanglement_entropy),
                "order_metric_r": float(candidate.order_metric_r),
                "order_metric_psi": float(candidate.order_metric_psi),
                "objective_labels": list(candidate.objective_labels),
                "non_actuating": candidate.non_actuating,
                "execution_disabled": candidate.execution_disabled,
                "claim_boundary": candidate.claim_boundary,
            }
            for candidate in scenario.state_candidates
        ],
    }


def _compute_order_metrics(phases: FloatArray) -> tuple[float, float]:
    order_r = float(np.abs(np.mean(np.exp(1j * phases))))
    order_psi = float(0.5 + 0.5 * math.cos(float(np.mean(phases))))
    return (order_r, order_psi)


def _normalize_probability(amplitudes: ComplexArray) -> NDArray[np.float64]:
    probabilities = np.abs(amplitudes) ** 2
    total = float(np.sum(probabilities))
    return probabilities / total


def _compute_entanglement_entropy(amplitudes: ComplexArray) -> float:
    probs = _normalize_probability(amplitudes)
    nz = probs > 0
    return -float(np.sum(probs[nz] * np.log2(probs[nz])))


def _product_state_vector(qubit_count: int) -> ComplexArray:
    size = 2**qubit_count
    vec = np.zeros(size, dtype=np.complex128)
    vec[0] = 1.0 + 0.0j
    return vec


def _entangled_state_vector(qubit_count: int) -> ComplexArray:
    size = 2**qubit_count
    vec = np.zeros(size, dtype=np.complex128)
    vec[0] = 1.0 / math.sqrt(2.0)
    vec[-1] = 1.0 / math.sqrt(2.0)
    return vec


def _build_scenario(
    *,
    domain: str,
    scenario_id: str,
    qubit_count: int,
    phase_offset: float,
    objective_labels: tuple[str, ...],
) -> HybridOrderScenario:
    phases = np.linspace(
        0.0, 2.0 * math.pi, qubit_count, endpoint=False, dtype=np.float64
    )
    phases = phases + phase_offset
    phases = np.mod(phases, 2.0 * math.pi).astype(np.float64)

    bipartition: tuple[tuple[int, ...], tuple[int, ...]]
    if qubit_count == 2:
        bipartition = ((0,), (1,))
    elif qubit_count == 3:
        bipartition = ((0,), (1, 2))
    else:
        bipartition = ((0, 1), (2, 3))

    order_r, order_psi = _compute_order_metrics(phases)

    product = HybridStateCandidate(
        state_id=f"{scenario_id}_product_state",
        candidate_type="product",
        amplitudes=_product_state_vector(qubit_count),
        entanglement_entropy=_compute_entanglement_entropy(
            _product_state_vector(qubit_count)
        ),
        order_metric_r=order_r,
        order_metric_psi=order_psi,
        objective_labels=("low_entanglement", "classical_synchrony"),
    )
    entangled = HybridStateCandidate(
        state_id=f"{scenario_id}_entangled_state",
        candidate_type="entangled",
        amplitudes=_entangled_state_vector(qubit_count),
        entanglement_entropy=_compute_entanglement_entropy(
            _entangled_state_vector(qubit_count)
        ),
        order_metric_r=order_r * 0.9,
        order_metric_psi=order_psi * 0.9,
        objective_labels=("high_entanglement", "robust_rollback"),
    )

    return HybridOrderScenario(
        domain=domain,
        scenario_id=scenario_id,
        phases=_ensure_float64_vector(phases, label=f"{scenario_id}.phases"),
        qubit_count=qubit_count,
        bipartition=bipartition,
        state_candidates=(product, entangled),
        objective_labels=objective_labels,
    )


def build_hybrid_order_parameter_scenarios() -> tuple[dict[str, object], ...]:
    """Return deterministic, JSON-safe hybrid order-parameter scenarios."""
    scenarios = (
        _build_scenario(
            domain="quantum_simulation",
            scenario_id="hybrid_order_quantum_simulation_v1",
            qubit_count=2,
            phase_offset=0.15,
            objective_labels=(
                "quantum_cosimulation_validation",
                "entanglement_audit",
                "classical_phase_coherence",
            ),
        ),
        _build_scenario(
            domain="power_grid",
            scenario_id="hybrid_order_power_grid_v1",
            qubit_count=3,
            phase_offset=0.47,
            objective_labels=(
                "islanding_resilience",
                "frequency_lock",
                "quantum_readout_alignment",
            ),
        ),
        _build_scenario(
            domain="cardiac_rhythm",
            scenario_id="hybrid_order_cardiac_rhythm_v1",
            qubit_count=4,
            phase_offset=1.03,
            objective_labels=(
                "phase_stabilisation",
                "entanglement_envelope",
                "rhythm_quality",
            ),
        ),
    )

    records: list[dict[str, object]] = []
    for scenario in scenarios:
        _validate_scenario(scenario)
        scenario.scenario_hash = _compute_scenario_hash(scenario)
        _validate_scenario(scenario)
        records.append(_to_record(scenario))

    return tuple(records)


__all__ = [
    "HybridOrderScenario",
    "HybridStateCandidate",
    "build_hybrid_order_parameter_scenarios",
    "_validate_scenario",
    "_compute_scenario_hash",
]
