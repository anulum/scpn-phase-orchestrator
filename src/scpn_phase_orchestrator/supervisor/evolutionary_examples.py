# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Evolutionary supervisor policy examples

"""Deterministic examples for offline evolutionary supervisor policy search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final, cast

from scpn_phase_orchestrator.supervisor.evolutionary_search import (
    run_offline_evolutionary_supervisor_search,
)

EvolutionaryBoundary: Final[str] = "evolutionary_supervisor_search_not_live_actuation"
SUPPORTED_DOMAINS: Final[tuple[str, ...]] = (
    "power_grid",
    "cardiac_rhythm",
    "cyber_industrial",
    "traffic_flow",
)

__all__ = [
    "EvolutionaryBoundary",
    "SUPPORTED_DOMAINS",
    "_validate_evolutionary_supervisor_search_record",
    "build_evolutionary_supervisor_search_examples",
    "build_evolutionary_supervisor_search_examples_from_worker_a_api",
]


def build_evolutionary_supervisor_search_examples() -> tuple[dict[str, object], ...]:
    """Return deterministic offline-search example inputs for reference gates.

    Returns
    -------
    tuple[dict[str, object], ...]
        Return deterministic offline-search example inputs for reference gates.
    """
    records = _build_examples()
    for record in records:
        _validate_evolutionary_supervisor_search_record(record)
    return records


def build_evolutionary_supervisor_search_examples_from_worker_a_api() -> tuple[
    dict[str, object], ...
]:
    """Return examples enriched with core offline-search report counts.

    Returns
    -------
    tuple[dict[str, object], ...]
        Return examples enriched with core offline-search report counts.
    """
    enriched: list[dict[str, object]] = []
    for record in build_evolutionary_supervisor_search_examples():
        report = run_offline_evolutionary_supervisor_search(
            _as_mapping(record["parent_policy"], label="parent_policy"),
            _as_mapping_sequence(record["audit_replays"], label="audit_replays"),
            stl_spec=str(record["stl_spec"]),
            trace=_as_trace(record["trace"]),
            generation_count=_as_int(
                record["generation_count"],
                label="generation_count",
            ),
            population_size=_as_int(record["population_size"], label="population_size"),
            mutation_step=_as_float(record["mutation_step"], label="mutation_step"),
            minimum_replay_reward=_as_float(
                record["minimum_replay_reward"], label="minimum_replay_reward"
            ),
            minimum_safety_margin=_as_float(
                record["minimum_safety_margin"], label="minimum_safety_margin"
            ),
        )
        merged = dict(record)
        merged.update(
            {
                "candidate_count": report.candidate_count,
                "accepted_candidate_count": report.accepted_count,
                "rejected_candidate_count": report.rejected_count,
                "report_hash": report.report_hash,
            }
        )
        enriched.append(merged)
    return tuple(enriched)


def _build_examples() -> tuple[dict[str, object], ...]:
    """Build the deterministic evolutionary supervisor search examples."""
    examples: tuple[dict[str, object], ...] = (
        {
            "domain": "power_grid",
            "scenario_id": "power_grid_offline_evolutionary_search_v1",
            "parent_policy": {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
            "audit_replays": (
                {
                    "replay_id": "grid_nominal",
                    "reward": 0.92,
                    "safety_margin": 0.07,
                    "violations": [],
                },
                {
                    "replay_id": "grid_disturbance",
                    "reward": 0.84,
                    "safety_margin": 0.06,
                    "violations": [],
                },
            ),
            "stl_spec": "always (R >= 0.82)",
            "trace": {"R": (0.91, 0.90, 0.89, 0.88)},
            "generation_count": 2,
            "population_size": 4,
            "mutation_step": 0.05,
            "minimum_replay_reward": 0.70,
            "minimum_safety_margin": 0.04,
        },
        {
            "domain": "cardiac_rhythm",
            "scenario_id": "cardiac_rhythm_offline_evolutionary_search_v1",
            "parent_policy": {"K": 0.36, "alpha": 0.19, "zeta": 0.04},
            "audit_replays": (
                {
                    "replay_id": "cardiac_nominal",
                    "reward": 0.78,
                    "safety_margin": 0.05,
                    "violations": [],
                },
                {
                    "replay_id": "cardiac_arrhythmia_replay",
                    "reward": 0.72,
                    "safety_margin": 0.03,
                    "violations": [],
                },
            ),
            "stl_spec": "always (R >= 0.60)",
            "trace": {"R": (0.72, 0.70, 0.69, 0.67)},
            "generation_count": 1,
            "population_size": 5,
            "mutation_step": 0.06,
            "minimum_replay_reward": 0.65,
            "minimum_safety_margin": 0.02,
        },
        {
            "domain": "cyber_industrial",
            "scenario_id": "cyber_industrial_offline_evolutionary_search_v1",
            "parent_policy": {"K": 0.28, "alpha": 0.11, "gamma": 0.07},
            "audit_replays": (
                {
                    "replay_id": "cyber_latency_stress",
                    "reward": 0.31,
                    "safety_margin": 0.01,
                    "violations": ["latency_breach", "service_stall"],
                },
                {
                    "replay_id": "cyber_packet_loss",
                    "reward": 0.28,
                    "safety_margin": 0.01,
                    "violations": ["packet_loss"],
                },
            ),
            "stl_spec": "always (R >= 0.88)",
            "trace": {"R": (0.61, 0.58, 0.55, 0.53)},
            "generation_count": 1,
            "population_size": 4,
            "mutation_step": 0.05,
            "minimum_replay_reward": 0.70,
            "minimum_safety_margin": 0.05,
        },
        {
            "domain": "traffic_flow",
            "scenario_id": "traffic_flow_offline_evolutionary_search_v1",
            "parent_policy": {"K": 0.61, "alpha": 0.08, "eta": 0.03},
            "audit_replays": (
                {
                    "replay_id": "traffic_nominal",
                    "reward": 0.88,
                    "safety_margin": 0.08,
                    "violations": [],
                },
                {
                    "replay_id": "traffic_incident",
                    "reward": 0.76,
                    "safety_margin": 0.05,
                    "violations": [],
                },
            ),
            "stl_spec": "always (R >= 0.70)",
            "trace": {"R": (0.79, 0.80, 0.78, 0.77)},
            "generation_count": 1,
            "population_size": 4,
            "mutation_step": 0.04,
            "minimum_replay_reward": 0.70,
            "minimum_safety_margin": 0.04,
        },
    )
    records: list[dict[str, object]] = []
    for example in examples:
        record = dict(example)
        record.update(
            {
                "operator_review_required": True,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "actuation_permitted": False,
                "claim_boundary": EvolutionaryBoundary,
            }
        )
        record["scenario_hash"] = _scenario_hash(record)
        records.append(record)
    return tuple(records)


def _validate_evolutionary_supervisor_search_record(
    record: Mapping[str, object],
) -> None:
    """Validate an evolutionary supervisor search record, else raise."""
    mapping = _as_mapping(record, label="record")
    for field in (
        "domain",
        "scenario_id",
        "scenario_hash",
        "claim_boundary",
        "parent_policy",
        "audit_replays",
        "stl_spec",
        "trace",
        "generation_count",
        "population_size",
        "mutation_step",
        "minimum_replay_reward",
        "minimum_safety_margin",
        "operator_review_required",
        "execution_disabled",
        "hot_patch_permitted",
        "live_merge_permitted",
        "actuation_permitted",
    ):
        if field not in mapping:
            raise ValueError(f"record missing required field '{field}'")
    domain = _as_non_empty_str(mapping["domain"], label="record.domain")
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported domain '{domain}'")
    _as_non_empty_str(mapping["scenario_id"], label="record.scenario_id")
    _validate_hash(mapping["scenario_hash"], label="scenario_hash")
    if mapping["claim_boundary"] != EvolutionaryBoundary:
        raise ValueError("record has invalid claim_boundary")
    _validate_parent_policy(
        _as_mapping(mapping["parent_policy"], label="parent_policy")
    )
    _validate_audit_replays(mapping["audit_replays"])
    _as_non_empty_str(mapping["stl_spec"], label="stl_spec")
    _validate_trace(_as_mapping(mapping["trace"], label="trace"))
    if _as_int(mapping["generation_count"], label="generation_count") <= 0:
        raise ValueError("generation_count must be positive")
    if _as_int(mapping["population_size"], label="population_size") <= 0:
        raise ValueError("population_size must be positive")
    if _as_float(mapping["mutation_step"], label="mutation_step") <= 0.0:
        raise ValueError("mutation_step must be positive")
    _as_float(mapping["minimum_replay_reward"], label="minimum_replay_reward")
    _as_float(mapping["minimum_safety_margin"], label="minimum_safety_margin")
    for field, expected in (
        ("operator_review_required", True),
        ("execution_disabled", True),
        ("hot_patch_permitted", False),
        ("live_merge_permitted", False),
        ("actuation_permitted", False),
    ):
        if _as_bool(mapping[field], label=field) is not expected:
            raise ValueError(f"{field} must be {expected}")
    if _scenario_hash(mapping) != mapping["scenario_hash"]:
        raise ValueError("record has invalid scenario_hash")


def _validate_parent_policy(parent_policy: Mapping[str, object]) -> None:
    """Validate the parent policy, else raise."""
    if not parent_policy:
        raise ValueError("record requires non-empty parent_policy")
    for key, value in parent_policy.items():
        _as_non_empty_str(key, label="parent_policy key")
        _as_float(value, label=f"parent_policy[{key}]")


def _validate_audit_replays(value: object) -> None:
    """Validate the audit replay records, else raise."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("audit_replays must be a sequence")
    if not value:
        raise ValueError("audit_replays must be non-empty")
    for replay in value:
        replay_map = _as_mapping(replay, label="audit_replay")
        _as_non_empty_str(replay_map.get("replay_id"), label="audit_replay.replay_id")
        _as_float(replay_map.get("reward"), label="audit_replay.reward")
        _as_float(replay_map.get("safety_margin"), label="audit_replay.safety_margin")
        violations = replay_map.get("violations")
        if not isinstance(violations, Sequence) or isinstance(
            violations, (str, bytes, bytearray)
        ):
            raise ValueError("audit_replay.violations must be a sequence")
        for violation in violations:
            _as_non_empty_str(violation, label="audit_replay.violation")


def _validate_trace(trace: Mapping[str, object]) -> None:
    """Validate the search trace, else raise."""
    if not trace:
        raise ValueError("trace must be non-empty")
    for signal, values in trace.items():
        _as_non_empty_str(signal, label="trace signal")
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise ValueError("trace values must be sequences")
        if not values:
            raise ValueError("trace values must be non-empty")
        for value in values:
            _as_float(value, label=f"trace[{signal}]")


def _scenario_hash(record: Mapping[str, object]) -> str:
    """Return the canonical-JSON SHA-256 hash of a scenario."""
    audit_replays = record["audit_replays"]
    if not isinstance(audit_replays, Sequence) or isinstance(
        audit_replays, (str, bytes, bytearray)
    ):
        raise ValueError("audit_replays must be a sequence")
    payload = {
        "domain": record["domain"],
        "scenario_id": record["scenario_id"],
        "parent_policy": record["parent_policy"],
        "audit_replays": list(audit_replays),
        "stl_spec": record["stl_spec"],
        "trace": record["trace"],
        "generation_count": record["generation_count"],
        "population_size": record["population_size"],
        "mutation_step": record["mutation_step"],
        "minimum_replay_reward": record["minimum_replay_reward"],
        "minimum_safety_margin": record["minimum_safety_margin"],
    }
    return _hash_payload(payload)


def _validate_hash(value: object, *, label: str) -> None:
    """Assert a hash matches its payload, else raise."""
    text = _as_non_empty_str(value, label=label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"record has invalid {label}")


def _as_mapping(value: object, *, label: str) -> dict[str, object]:
    """Return ``value`` as a mapping, else raise ``ValueError``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _as_mapping_sequence(
    value: object,
    *,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    """Return ``value`` as a sequence of mappings, else raise."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of mappings")
    return tuple(_as_mapping(item, label=f"{label} item") for item in value)


def _as_trace(value: object) -> dict[str, Sequence[object]]:
    """Return ``value`` as a validated trace, else raise."""
    mapping = _as_mapping(value, label="trace")
    return {key: cast(Sequence[object], item) for key, item in mapping.items()}


def _as_non_empty_str(value: object, *, label: str) -> str:
    """Return ``value`` as a non-empty string, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _as_float(value: object, *, label: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))):
        raise ValueError(f"{label} must be a finite number")
    return number


def _as_int(value: object, *, label: str) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _as_bool(value: object, *, label: str) -> bool:
    """Return ``value`` as a real boolean, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _hash_payload(payload: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 hash of a payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
