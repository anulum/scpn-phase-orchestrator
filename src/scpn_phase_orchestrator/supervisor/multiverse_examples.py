# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multiverse counterfactual domain scenarios

"""Multiverse counterfactual scenario examples for branch review."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

CounterfactualBoundary: Final[str] = "counterfactual_branch_rollout_not_live_actuation"
SupportedCounterfactualKnobs: Final[frozenset[str]] = frozenset(
    {"K", "alpha", "zeta", "Psi"}
)


def _ensure_float64_vector(
    values: Iterable[float], *, label: str
) -> NDArray[np.float64]:
    if isinstance(values, (bool, np.bool_)) or _contains_boolean_alias(values):
        raise ValueError(f"{label} must contain numeric values")
    if _contains_complex_alias(values):
        raise ValueError(f"{label} must contain real-valued numeric values")
    arr = np.asarray(tuple(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{label} must have float dtype")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _summary(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _ensure_real_scalar(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(
        value, (complex, np.complexfloating)
    ):
        raise ValueError(f"{label} must be a real-valued numeric scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a real-valued numeric scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class BranchCandidate:
    """Single deterministic branch candidate for a counterfactual rollout scenario."""

    candidate_id: str
    knob_variations: tuple[tuple[str, float], ...]
    topology_variations: tuple[str, ...]
    objective_labels: tuple[str, ...]
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = CounterfactualBoundary

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record."""
        return {
            "candidate_id": self.candidate_id,
            "knob_variations": [[name, value] for name, value in self.knob_variations],
            "topology_variations": list(self.topology_variations),
            "objective_labels": list(self.objective_labels),
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DomainScenario:
    """Deterministic scenario definition for aggregate multiverse benchmarks."""

    domain: str
    scenario_id: str
    initial_phases: NDArray[np.float64]
    initial_omegas: NDArray[np.float64]
    branch_candidates: tuple[BranchCandidate, ...]
    objective_labels: tuple[str, ...]
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = CounterfactualBoundary

    def scenario_hash(self) -> str:
        """Return the deterministic scenario digest."""
        return _compute_scenario_hash(
            domain=self.domain,
            scenario_id=self.scenario_id,
            initial_phases=self.initial_phases,
            initial_omegas=self.initial_omegas,
            branch_candidates=self.branch_candidates,
            objective_labels=self.objective_labels,
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record."""
        return {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash(),
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
            "initial_phases_summary": _summary(self.initial_phases),
            "initial_omegas_summary": _summary(self.initial_omegas),
            "initial_phases": self.initial_phases.tolist(),
            "initial_omegas": self.initial_omegas.tolist(),
            "branch_candidates": [
                candidate.to_audit_record() for candidate in self.branch_candidates
            ],
            "objective_labels": list(self.objective_labels),
        }


def _compute_scenario_hash(
    *,
    domain: str,
    scenario_id: str,
    initial_phases: NDArray[np.float64],
    initial_omegas: NDArray[np.float64],
    branch_candidates: tuple[BranchCandidate, ...],
    objective_labels: tuple[str, ...],
) -> str:
    canonical = {
        "domain": domain,
        "scenario_id": scenario_id,
        "claim_boundary": CounterfactualBoundary,
        "initial_phases": [float(x) for x in initial_phases.tolist()],
        "initial_omegas": [float(x) for x in initial_omegas.tolist()],
        "objective_labels": list(objective_labels),
        "branch_candidates": [
            {
                "candidate_id": branch.candidate_id,
                "knob_variations": [[k, float(v)] for k, v in branch.knob_variations],
                "topology_variations": list(branch.topology_variations),
                "objective_labels": list(branch.objective_labels),
                "non_actuating": branch.non_actuating,
                "execution_disabled": branch.execution_disabled,
                "claim_boundary": branch.claim_boundary,
            }
            for branch in sorted(branch_candidates, key=lambda item: item.candidate_id)
        ],
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_branch_candidate(candidate: BranchCandidate) -> None:
    if (
        not isinstance(candidate.candidate_id, str)
        or not candidate.candidate_id.strip()
    ):
        raise ValueError("branch candidate id must be a non-empty string")

    if (
        not isinstance(candidate.knob_variations, tuple)
        or not candidate.knob_variations
    ):
        raise ValueError(
            f"candidate '{candidate.candidate_id}' requires knob_variations"
        )
    for knob_name, knob_value in candidate.knob_variations:
        if not isinstance(knob_name, str) or not knob_name.strip():
            raise ValueError(
                f"candidate '{candidate.candidate_id}' has invalid knob name"
            )
        if knob_name not in SupportedCounterfactualKnobs:
            raise ValueError(
                f"candidate '{candidate.candidate_id}' knob "
                f"'{knob_name}' is not supported by counterfactual rollouts"
            )
        _ensure_real_scalar(
            knob_value,
            label=f"candidate '{candidate.candidate_id}' knob '{knob_name}'",
        )

    if not candidate.topology_variations:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' requires topology variations"
        )
    if not all(isinstance(v, str) and v.strip() for v in candidate.topology_variations):
        raise ValueError(
            f"candidate '{candidate.candidate_id}' topology variations invalid"
        )

    if not candidate.objective_labels:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' requires objective labels"
        )
    if not all(
        isinstance(lbl, str) and lbl.strip() for lbl in candidate.objective_labels
    ):
        raise ValueError(
            f"candidate '{candidate.candidate_id}' has invalid objective labels"
        )

    if candidate.non_actuating is not True:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' must have non_actuating=True"
        )
    if candidate.execution_disabled is not True:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' must have execution_disabled=True"
        )
    if candidate.claim_boundary != CounterfactualBoundary:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' must use claim_boundary="
            f"{CounterfactualBoundary}"
        )


def _validate_scenario(scenario: DomainScenario) -> None:
    if not isinstance(scenario.domain, str) or not scenario.domain.strip():
        raise ValueError("scenario domain must be a non-empty string")
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id.strip():
        raise ValueError(
            f"scenario for {scenario.domain!r} must have a non-empty scenario_id"
        )

    _ = _ensure_float64_vector(
        scenario.initial_phases, label=f"{scenario.scenario_id}.initial_phases"
    )
    _ = _ensure_float64_vector(
        scenario.initial_omegas, label=f"{scenario.scenario_id}.initial_omegas"
    )

    if scenario.initial_phases.shape != scenario.initial_omegas.shape:
        raise ValueError(
            f"scenario {scenario.scenario_id} initial phases/omegas "
            "must have same shape"
        )

    if scenario.non_actuating is not True:
        raise ValueError(f"scenario {scenario.scenario_id} must set non_actuating=True")
    if scenario.execution_disabled is not True:
        raise ValueError(
            f"scenario {scenario.scenario_id} must set execution_disabled=True"
        )
    if scenario.claim_boundary != CounterfactualBoundary:
        raise ValueError(
            f"scenario {scenario.scenario_id} must use claim_boundary="
            f"{CounterfactualBoundary}"
        )

    if not scenario.branch_candidates:
        raise ValueError(
            f"scenario {scenario.scenario_id} requires at least one branch"
        )
    for candidate in scenario.branch_candidates:
        _validate_branch_candidate(candidate)

    if not scenario.objective_labels:
        raise ValueError(f"scenario {scenario.scenario_id} requires objective labels")
    if not all(
        isinstance(lbl, str) and lbl.strip() for lbl in scenario.objective_labels
    ):
        raise ValueError(
            f"scenario {scenario.scenario_id} has invalid objective labels"
        )


def _build_static_scenarios() -> tuple[DomainScenario, ...]:
    return (
        DomainScenario(
            domain="power_grid",
            scenario_id="power_grid_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.00, 0.33, 0.67, 1.05, 1.58, 2.14],
                label="power_grid.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [59.98, 60.02, 59.99, 60.01, 59.95, 60.05],
                label="power_grid.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="power_grid_load_shed_margin",
                    knob_variations=(("zeta", 0.03), ("K", 0.12)),
                    topology_variations=("ring_redundant", "mesh_reinforced"),
                    objective_labels=("load_stability", "frequency_regulation"),
                ),
                BranchCandidate(
                    candidate_id="power_grid_regional_islanding",
                    knob_variations=(("alpha", 0.04), ("Psi", 0.02)),
                    topology_variations=("sector_islands", "hierarchical_loop"),
                    objective_labels=("islanding_resilience", "power_flow_safety"),
                ),
            ),
            objective_labels=(
                "load_stability",
                "frequency_regulation",
                "islanding_resilience",
            ),
        ),
        DomainScenario(
            domain="cardiac_rhythm",
            scenario_id="cardiac_rhythm_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.12, 0.58, 0.94, 1.20, 1.68],
                label="cardiac_rhythm.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [0.95, 1.02, 0.98, 1.01, 1.05],
                label="cardiac_rhythm.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="cardiac_refractory_brake",
                    knob_variations=(("K", 0.08), ("zeta", 0.04)),
                    topology_variations=(
                        "node_reconnect_lowpass",
                        "dual_loop_bradyzone",
                    ),
                    objective_labels=("arrhythmia_suppression", "heartbeat_stability"),
                ),
                BranchCandidate(
                    candidate_id="cardiac_autonomic_probe",
                    knob_variations=(("alpha", -0.03), ("zeta", 0.02)),
                    topology_variations=(
                        "pacemaker_safe_override",
                        "layered_autonomic",
                    ),
                    objective_labels=(
                        "cycle_variability_reduction",
                        "oxygenation_support",
                    ),
                ),
            ),
            objective_labels=(
                "arrhythmia_suppression",
                "cardio_stability",
                "oxygenation_support",
            ),
        ),
        DomainScenario(
            domain="cyber_industrial",
            scenario_id="cyber_industrial_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.21, 0.43, 0.70, 1.02, 1.38, 1.70],
                label="cyber_industrial.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [0.98, 1.03, 1.01, 1.00, 1.02, 1.05],
                label="cyber_industrial.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="cyber_isolation_containment",
                    knob_variations=(("zeta", 0.06), ("Psi", 0.03)),
                    topology_variations=(
                        "zonal_segmentation",
                        "trust_graph_hardening",
                    ),
                    objective_labels=(
                        "attack_surface_reduction",
                        "service_containment",
                    ),
                ),
                BranchCandidate(
                    candidate_id="cyber_traffic_cushion",
                    knob_variations=(("K", 0.05), ("alpha", -0.01)),
                    topology_variations=(
                        "flow_reroute_bypass",
                        "priority_queueing",
                    ),
                    objective_labels=("latency_regulation", "availability_guardrail"),
                ),
            ),
            objective_labels=(
                "attack_surface_reduction",
                "service_containment",
                "latency_regulation",
            ),
        ),
        DomainScenario(
            domain="traffic_flow",
            scenario_id="traffic_flow_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.05, 0.31, 0.62, 0.96, 1.37, 1.83, 2.41],
                label="traffic_flow.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [0.82, 0.88, 0.79, 0.91, 0.86, 0.93, 0.84],
                label="traffic_flow.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="traffic_ramp_meter_damping",
                    knob_variations=(("K", 0.07), ("alpha", -0.015)),
                    topology_variations=("corridor_metering", "arterial_feedback"),
                    objective_labels=(
                        "congestion_wave_damping",
                        "green_wave_stability",
                    ),
                ),
                BranchCandidate(
                    candidate_id="traffic_spillback_diversion",
                    knob_variations=(("zeta", 0.025), ("Psi", 0.18)),
                    topology_variations=("queue_spillback_bypass", "adaptive_detour"),
                    objective_labels=(
                        "spillback_prevention",
                        "network_throughput_guardrail",
                    ),
                ),
            ),
            objective_labels=(
                "congestion_wave_damping",
                "spillback_prevention",
                "green_wave_stability",
            ),
        ),
        DomainScenario(
            domain="manufacturing_spc",
            scenario_id="manufacturing_spc_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.08, 0.27, 0.52, 0.88, 1.22, 1.61],
                label="manufacturing_spc.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [1.00, 1.04, 0.97, 1.02, 0.95, 1.06],
                label="manufacturing_spc.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="manufacturing_drift_recoupling",
                    knob_variations=(("K", 0.06), ("zeta", 0.018)),
                    topology_variations=("line_rebalance", "sensor_loop_reweight"),
                    objective_labels=(
                        "process_drift_recovery",
                        "line_balance_stability",
                    ),
                ),
                BranchCandidate(
                    candidate_id="manufacturing_scrap_phase_guard",
                    knob_variations=(("alpha", 0.025), ("Psi", -0.12)),
                    topology_variations=("quality_gate_feedback", "rework_buffer"),
                    objective_labels=(
                        "scrap_rate_reduction",
                        "quality_variance_guardrail",
                    ),
                ),
            ),
            objective_labels=(
                "process_drift_recovery",
                "scrap_rate_reduction",
                "line_balance_stability",
            ),
        ),
        DomainScenario(
            domain="plasma_control",
            scenario_id="plasma_control_counterfactual_rollout_v1",
            initial_phases=_ensure_float64_vector(
                [0.02, 0.44, 0.91, 1.33, 1.79, 2.26, 2.78, 3.10],
                label="plasma_control.initial_phases",
            ),
            initial_omegas=_ensure_float64_vector(
                [2.08, 2.01, 2.13, 2.05, 1.96, 2.10, 2.03, 2.16],
                label="plasma_control.initial_omegas",
            ),
            branch_candidates=(
                BranchCandidate(
                    candidate_id="plasma_mode_locking_phase_scan",
                    knob_variations=(("alpha", -0.02), ("K", 0.09)),
                    topology_variations=("edge_harmonic_feedback", "coil_phase_scan"),
                    objective_labels=(
                        "mode_locking_stability",
                        "confinement_margin",
                    ),
                ),
                BranchCandidate(
                    candidate_id="plasma_elm_mitigation_drive",
                    knob_variations=(("zeta", 0.035), ("Psi", 0.24)),
                    topology_variations=(
                        "resonant_magnetic_perturbation",
                        "q_profile_guard",
                    ),
                    objective_labels=(
                        "edge_localised_mode_mitigation",
                        "divertor_heat_flux_guardrail",
                    ),
                ),
            ),
            objective_labels=(
                "mode_locking_stability",
                "edge_localised_mode_mitigation",
                "confinement_margin",
            ),
        ),
    )


def build_multiverse_domain_scenarios() -> tuple[dict[str, object], ...]:
    """Build deterministic multiverse domain scenario records."""
    scenarios = _build_static_scenarios()
    records: list[dict[str, object]] = []

    for scenario in scenarios:
        _validate_scenario(scenario)
        record = scenario.to_audit_record()
        _verify_record_hash(scenario, record)
        records.append(record)

    return tuple(records)


def _verify_record_hash(scenario: DomainScenario, record: dict[str, object]) -> None:
    expected = scenario.scenario_hash()
    if record["scenario_hash"] != expected:
        raise ValueError(
            f"scenario {scenario.scenario_id} has mismatched scenario_hash"
        )


__all__ = [
    "BranchCandidate",
    "CounterfactualBoundary",
    "DomainScenario",
    "SupportedCounterfactualKnobs",
    "build_multiverse_domain_scenarios",
    "_validate_branch_candidate",
    "_validate_scenario",
]
