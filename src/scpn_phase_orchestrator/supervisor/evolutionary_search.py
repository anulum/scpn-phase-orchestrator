# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline evolutionary supervisor policy search

"""Deterministic offline evolutionary supervisor policy search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, TypedDict

from scpn_phase_orchestrator.monitor import STLMonitor

__all__ = [
    "EvolutionaryCandidate",
    "EvolutionarySearchConfig",
    "EvolutionarySearchReport",
    "run_offline_evolutionary_supervisor_search",
]


@dataclass(frozen=True)
class EvolutionarySearchConfig:
    """Configuration for deterministic offline candidate evolution."""

    generation_count: int = 2
    population_size: int = 8
    mutation_step: float = 0.05
    minimum_replay_reward: float = 0.0
    minimum_safety_margin: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_int(self.generation_count, "generation_count")
        _require_positive_int(self.population_size, "population_size")
        _require_finite_positive(self.mutation_step, "mutation_step")
        _require_finite_real(self.minimum_replay_reward, "minimum_replay_reward")
        _require_finite_real(self.minimum_safety_margin, "minimum_safety_margin")


class _ReplaySummary(TypedDict):
    """Reduced summary of a candidate's replay evaluation."""

    replay_count: int
    mean_reward: float
    min_reward: float
    mean_safety_margin: float
    min_safety_margin: float
    violation_count: int


@dataclass(frozen=True)
class EvolutionaryCandidate:
    """One offline candidate snapshot from a deterministic mutation step."""

    candidate_id: str
    generation: int
    knob: str
    parent_value: float
    candidate_value: float
    mutation_delta: float
    genome: tuple[tuple[str, float], ...]
    replay_fitness: float
    stl_robustness: float
    stl_satisfied: bool
    replay_violation_count: int
    blocked_reasons: tuple[str, ...]
    candidate_hash: str
    review_required: bool = True
    live_merge_permitted: bool = False
    hot_patch_permitted: bool = False
    actuation_permitted: bool = False

    @property
    def accepted(self) -> bool:
        """Whether the candidate passed all guard and replay gates.

        Returns
        -------
        bool
            Whether the candidate passed all guard and replay gates.
        """
        return not self.blocked_reasons

    @property
    def status(self) -> str:
        """Return an export-friendly status string.

        Returns
        -------
        str
            Return an export-friendly status string.
        """
        return "accepted_for_review" if self.accepted else "rejected"

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe candidate evidence for audit transport.

        Returns
        -------
        dict[str, object]
            Return JSON-safe candidate evidence for audit transport.
        """
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "knob": self.knob,
            "parent_value": self.parent_value,
            "candidate_value": self.candidate_value,
            "mutation_delta": self.mutation_delta,
            "genome": [[key, value] for key, value in self.genome],
            "replay_fitness": self.replay_fitness,
            "stl_robustness": self.stl_robustness,
            "stl_satisfied": self.stl_satisfied,
            "replay_violation_count": self.replay_violation_count,
            "blocked_reasons": list(self.blocked_reasons),
            "status": self.status,
            "review_required": self.review_required,
            "live_merge_permitted": self.live_merge_permitted,
            "hot_patch_permitted": self.hot_patch_permitted,
            "actuation_permitted": self.actuation_permitted,
            "candidate_hash": self.candidate_hash,
        }


@dataclass(frozen=True)
class EvolutionarySearchReport:
    """Deterministic, offline-only audit report for evolutionary search."""

    schema_name: str
    schema_version: str
    config: EvolutionarySearchConfig
    parent_policy_hash: str
    replay_summary: _ReplaySummary
    stl_spec: str
    stl_monitoring: dict[str, object]
    candidate_count: int
    accepted_count: int
    rejected_count: int
    candidates: tuple[EvolutionaryCandidate, ...]
    best_candidate: EvolutionaryCandidate | None
    claim_boundary: str
    non_actuating: bool
    execution_disabled: bool
    hot_patch_permitted: bool
    live_merge_permitted: bool
    operator_review_required: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record for review tooling.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe audit record for review tooling.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "generation_count": self.config.generation_count,
            "population_size": self.config.population_size,
            "mutation_step": self.config.mutation_step,
            "minimum_replay_reward": self.config.minimum_replay_reward,
            "minimum_safety_margin": self.config.minimum_safety_margin,
            "parent_policy_hash": self.parent_policy_hash,
            "replay_summary": self.replay_summary,
            "stl_spec": self.stl_spec,
            "stl_monitoring": self.stl_monitoring,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "best_candidate": self.best_candidate.to_audit_record()
            if self.best_candidate
            else None,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
            "claim_boundary": self.claim_boundary,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "hot_patch_permitted": self.hot_patch_permitted,
            "live_merge_permitted": self.live_merge_permitted,
            "operator_review_required": self.operator_review_required,
            "report_hash": self.report_hash,
        }


def run_offline_evolutionary_supervisor_search(
    parent_policy: Mapping[str, object],
    audit_replays: Sequence[Mapping[str, object]],
    *,
    stl_spec: str,
    trace: Mapping[str, Sequence[object]],
    generation_count: int = 2,
    population_size: int = 8,
    mutation_step: float = 0.05,
    minimum_replay_reward: float = 0.0,
    minimum_safety_margin: float = 0.0,
) -> EvolutionarySearchReport:
    """Run deterministic offline evolutionary policy mutation search.

    Returns review-only candidates plus guards that block any live merge/hot patch.

    Parameters
    ----------
    parent_policy : Mapping[str, object]
        The parent policy genome.
    audit_replays : Sequence[Mapping[str, object]]
        Audit replay records used to score candidates.
    stl_spec : str
        An STL specification string used as a safety gate.
    trace : Mapping[str, Sequence[object]]
        Signal trace keyed by variable name, each a sequence of floats.
    generation_count : int
        Number of search generations.
    population_size : int
        Number of candidates per generation.
    mutation_step : float
        Mutation step size applied per generation.
    minimum_replay_reward : float
        Minimum replay reward a candidate must reach.
    minimum_safety_margin : float
        Minimum safety margin a candidate must preserve.

    Returns
    -------
    EvolutionarySearchReport
        The offline evolutionary search report.
    """
    config = EvolutionarySearchConfig(
        generation_count=generation_count,
        population_size=population_size,
        mutation_step=mutation_step,
        minimum_replay_reward=minimum_replay_reward,
        minimum_safety_margin=minimum_safety_margin,
    )
    parent = _validate_parent_policy(parent_policy)
    replays = _validate_replays(audit_replays)
    replay_summary = _summarise_replays(replays)
    stl_monitor, stl_result, validated_trace = _validate_and_evaluate_stl(
        stl_spec=stl_spec,
        trace=trace,
    )
    genome_keys = tuple(sorted(parent))

    def _candidate_genome(
        mutated: Mapping[str, float],
    ) -> tuple[tuple[str, float], ...]:
        """Return the mutated candidate genome from the parent."""
        return tuple((key, float(mutated[key])) for key in sorted(mutated))

    candidates: list[EvolutionaryCandidate] = []
    for generation in range(config.generation_count):
        for local_idx in range(config.population_size):
            knob = genome_keys[(generation + local_idx) % len(genome_keys)]
            parent_value = parent[knob]
            direction = 1.0 if (generation + local_idx) % 2 == 0 else -1.0
            magnitude = _deterministic_mutation_magnitude(
                config=config,
                generation=generation,
                local_index=local_idx,
            )
            mutation_delta = direction * magnitude
            mutated_policy = dict(parent)
            mutated_policy[knob] = parent_value + mutation_delta

            blocked = _candidate_blocked_reasons(
                replay_summary,
                stl_result["robustness"],
                mutation_delta=mutation_delta,
                minimum_replay_reward=config.minimum_replay_reward,
                minimum_safety_margin=config.minimum_safety_margin,
            )
            # Candidate fitness is replay-weighted so selection stays offline,
            # replay-summary deterministic, and biased toward smaller drift.
            replay_fitness = float(replay_summary["mean_reward"]) - 0.5 * abs(
                mutation_delta
            )
            candidate = EvolutionaryCandidate(
                candidate_id=f"g{generation + 1:03d}-c{local_idx + 1:03d}",
                generation=generation + 1,
                knob=knob,
                parent_value=parent_value,
                candidate_value=float(mutated_policy[knob]),
                mutation_delta=float(mutation_delta),
                genome=_candidate_genome(mutated_policy),
                replay_fitness=replay_fitness,
                stl_robustness=float(stl_result["robustness"]),
                stl_satisfied=bool(stl_result["satisfied"]),
                replay_violation_count=int(replay_summary["violation_count"]),
                blocked_reasons=tuple(blocked),
                candidate_hash="",
            )
            candidates.append(
                replace(
                    candidate,
                    candidate_hash=_build_stable_hash(candidate.to_audit_record()),
                )
            )

    accepted = [candidate for candidate in candidates if candidate.accepted]
    rejected = [candidate for candidate in candidates if not candidate.accepted]
    best_candidate = max(
        accepted,
        key=lambda candidate: candidate.replay_fitness + candidate.stl_robustness,
        default=None,
    )
    stl_monitor_record = stl_monitor.evaluate_result(validated_trace).to_audit_record()
    report = EvolutionarySearchReport(
        schema_name="evolutionary_supervisor_policy_search",
        schema_version="0.1.0",
        config=config,
        parent_policy_hash=_build_stable_hash(parent),
        replay_summary=replay_summary,
        stl_spec=stl_spec,
        stl_monitoring=stl_monitor_record,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        candidates=tuple(candidates),
        best_candidate=best_candidate,
        claim_boundary="offline_evolutionary_supervisor_review_not_live_actuation",
        non_actuating=True,
        execution_disabled=True,
        hot_patch_permitted=False,
        live_merge_permitted=False,
        operator_review_required=True,
        report_hash="",
    )

    report_record = report.to_audit_record()
    report_hash = _build_stable_hash(report_record)
    return EvolutionarySearchReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        config=report.config,
        parent_policy_hash=report.parent_policy_hash,
        replay_summary=report.replay_summary,
        stl_spec=report.stl_spec,
        stl_monitoring=report.stl_monitoring,
        candidate_count=report.candidate_count,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        candidates=report.candidates,
        best_candidate=report.best_candidate,
        claim_boundary=report.claim_boundary,
        non_actuating=report.non_actuating,
        execution_disabled=report.execution_disabled,
        hot_patch_permitted=report.hot_patch_permitted,
        live_merge_permitted=report.live_merge_permitted,
        operator_review_required=report.operator_review_required,
        report_hash=report_hash,
    )


def _validate_and_evaluate_stl(
    *, stl_spec: str, trace: Mapping[str, Sequence[object]]
) -> tuple[STLMonitor, dict[str, float | bool], dict[str, list[float]]]:
    """Validate and evaluate the STL specification over the replays."""
    validated_spec = _non_empty_string(stl_spec, "stl_spec")
    validated_trace = _validate_trace(trace)

    monitor = STLMonitor(validated_spec)
    try:
        robustness = monitor.evaluate(validated_trace)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise ValueError(
            "stl_spec and trace must be valid for offline monitoring"
        ) from exc

    return (
        monitor,
        {
            "robustness": float(robustness),
            "satisfied": bool(robustness >= 0.0),
        },
        validated_trace,
    )


def _validate_parent_policy(policy: Mapping[str, object]) -> dict[str, float]:
    """Validate the parent policy, else raise."""
    if not isinstance(policy, Mapping):
        raise ValueError("parent_policy must be a mapping")
    if not policy:
        raise ValueError("parent_policy must be non-empty")
    validated: dict[str, float] = {}
    for key, value in policy.items():
        if not isinstance(key, str) or not key:
            raise ValueError("parent_policy keys must be non-empty strings")
        numeric = _require_finite_real(value, f"parent_policy[{key}]")
        validated[key] = numeric
    return validated


def _validate_replays(
    audit_replays: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate the replay records, else raise."""
    if not isinstance(audit_replays, Sequence) or isinstance(
        audit_replays, (str, bytes, bytearray)
    ):
        raise ValueError("audit_replays must be a sequence of mappings")
    if not audit_replays:
        raise ValueError("audit_replays must contain at least one replay")

    out: list[dict[str, object]] = []
    for index, replay in enumerate(audit_replays):
        if not isinstance(replay, Mapping):
            raise ValueError(f"audit_replays[{index}] must be a mapping")
        reward = _require_finite_real(
            replay.get("reward"), f"audit_replays[{index}].reward"
        )
        safety_margin = _require_finite_real(
            replay.get("safety_margin"),
            f"audit_replays[{index}].safety_margin",
        )
        if "violations" not in replay:
            raise ValueError(
                f"audit_replays[{index}].violations must be a sequence of strings"
            )
        violations_raw = replay.get("violations")
        if not isinstance(violations_raw, Sequence) or isinstance(
            violations_raw, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"audit_replays[{index}].violations must be a sequence of strings"
            )
        violations: list[str] = []
        for item in violations_raw:
            if not isinstance(item, str):
                raise ValueError(
                    f"audit_replays[{index}].violations must be a sequence of strings"
                )
            violations.append(item)
        replay_id = replay.get("replay_id", str(index))
        out.append(
            {
                "replay_id": str(replay_id),
                "reward": float(reward),
                "safety_margin": float(safety_margin),
                "violations": violations,
            }
        )
    return out


def _summarise_replays(replays: Sequence[Mapping[str, object]]) -> _ReplaySummary:
    """Return the reduced summary of the replay records."""
    rewards: list[float] = [
        _require_finite_real(replay["reward"], "reward") for replay in replays
    ]
    margins: list[float] = [
        _require_finite_real(replay["safety_margin"], "safety_margin")
        for replay in replays
    ]
    violation_count = 0
    for replay in replays:
        violations = replay["violations"]
        if not isinstance(violations, list):
            raise ValueError("violations must be a list")
        violation_count += len(violations)
    return {
        "replay_count": len(replays),
        "mean_reward": float(sum(rewards) / len(rewards)),
        "min_reward": float(min(rewards)),
        "mean_safety_margin": float(sum(margins) / len(margins)),
        "min_safety_margin": float(min(margins)),
        "violation_count": int(violation_count),
    }


def _candidate_blocked_reasons(
    replay_summary: _ReplaySummary,
    stl_robustness: float,
    *,
    mutation_delta: float,
    minimum_replay_reward: float,
    minimum_safety_margin: float,
) -> list[str]:
    """Return the reasons blocking a candidate."""
    reasons: list[str] = []
    if replay_summary["mean_reward"] < float(minimum_replay_reward):
        reasons.append("replay_reward_below_minimum")
    if replay_summary["mean_safety_margin"] < float(minimum_safety_margin):
        reasons.append("safety_margin_below_minimum")
    if replay_summary["violation_count"] > 0:
        reasons.append("replay_violations_present")
    if stl_robustness < 0.0:
        reasons.append("stl_spec_not_satisfied")
    if abs(float(mutation_delta)) > replay_summary["min_safety_margin"]:
        reasons.append("counterfactual_safety_delta_exceeds_replay_margin")
    return reasons


def _deterministic_mutation_magnitude(
    *,
    config: EvolutionarySearchConfig,
    generation: int,
    local_index: int,
) -> float:
    """Return the deterministic mutation magnitude for a step."""
    step = config.mutation_step
    index = generation * config.population_size + local_index
    # Keep the bounded mutation step in (0, mutation_step].
    raw = (index % config.population_size) + 1
    return step * (raw / (config.population_size + 1))


def _validate_trace(trace: Mapping[str, Sequence[object]]) -> dict[str, list[float]]:
    """Validate the search trace, else raise."""
    if not isinstance(trace, Mapping):
        raise ValueError("trace must be a mapping of signal names to sequences")
    if not trace:
        raise ValueError("trace must contain at least one signal")
    validated: dict[str, list[float]] = {}
    expected_length: int | None = None
    for signal, values in trace.items():
        if not isinstance(signal, str) or not signal:
            raise ValueError("trace signal names must be non-empty strings")
        if isinstance(values, (str, bytes, bytearray)):
            raise ValueError("trace values must be a sequence of finite numeric values")
        if not isinstance(values, Sequence):
            raise ValueError("trace values must be a sequence of finite numeric values")
        values_list = list(values)
        if not values_list:
            raise ValueError("trace signals must be non-empty")
        finite_values: list[float] = []
        for value in values_list:
            finite_values.append(_require_finite_real(value, f"trace[{signal}]"))
        if expected_length is None:
            expected_length = len(finite_values)
        elif len(finite_values) != expected_length:
            raise ValueError("all signals in trace must have equal length")
        if expected_length == 0:
            raise ValueError("trace signals must be non-empty")
        validated[signal] = finite_values
    return validated


def _non_empty_string(value: object, field: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_finite_real(value: object, field: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))):
        raise ValueError(f"{field} must be finite")
    return number


def _require_finite_positive(value: object, field: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    number = _require_finite_real(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _require_positive_int(value: object, field: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return number


def _build_stable_hash(payload: Mapping[str, Any] | object) -> str:
    """Return a stable SHA-256 hash of the inputs."""
    clean_payload: object
    if isinstance(payload, dict):
        clean_payload = dict(payload)
        clean_payload.pop("candidate_hash", None)
        clean_payload.pop("report_hash", None)
    else:
        clean_payload = payload
    blob = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
