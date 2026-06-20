# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy reduced-evidence boundary core

"""Reduced child-summary, escalation, and sync-envelope types with validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


_REGIME_CRITICAL = "critical"


_REGIME_DEGRADED = "degraded"


_REGIME_NOMINAL = "nominal"


_AUDIT_SCOPE_REDUCED_SUMMARIES = "reduced_child_summaries_only"


_DEFAULT_HIERARCHY_SYNC_PROTOCOL = "spo-hierarchy-sync/v1"


_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1


_HIERARCHY_RAW_MARKER_TERMS = frozenset(
    {
        "evidence",
        "history",
        "payload",
        "raw",
    }
)


_HIERARCHY_RAW_DATA_TERMS = frozenset(
    {
        "actuator",
        "actuators",
        "coupling",
        "couplings",
        "event",
        "events",
        "graph",
        "graphs",
        "observation",
        "observations",
        "phase",
        "phases",
        "series",
        "signal",
        "signals",
        "time",
        "timeseries",
    }
)


_FORBIDDEN_RAW_HIERARCHY_KEYS = frozenset(
    {
        "actuator",
        "actuators",
        "actuator_handle",
        "actuator_handles",
        "actuator_target",
        "actuator_targets",
        "child_evidence",
        "child_observations",
        "coupling",
        "coupling_matrix",
        "coupling_matrices",
        "couplings",
        "event",
        "events",
        "evidence",
        "graph",
        "knm",
        "local_coupling_matrix",
        "local_coupling_matrices",
        "observation",
        "observations",
        "phase",
        "phases",
        "raw",
        "raw_coupling",
        "raw_coupling_matrix",
        "raw_event",
        "raw_events",
        "raw_graph",
        "raw_observation",
        "raw_observations",
        "raw_time_series",
        "raw_actuator",
        "raw_actuator_target",
        "raw_actuator_targets",
        "raw_actuators",
        "raw_child_observations",
        "raw_evidence",
        "raw_phases",
        "raw_signal",
        "raw_signals",
        "signal",
        "signals",
        "time_series",
    }
)


@dataclass(frozen=True)
class ChildSupervisorSummary:
    """Bounded child-supervisor evidence for parent orchestration.

    The summary intentionally carries reduced coherence evidence only. Raw
    child phases, time series, local coupling matrices, and actuator targets do
    not cross the hierarchy boundary in this foundation slice.
    """

    name: str
    channel: str
    R: float
    psi: float
    regime: str = _REGIME_NOMINAL
    confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_child_summary_reduced_only(self)
        object.__setattr__(
            self,
            "metadata",
            _normalise_metadata_value(self.metadata, "summary.metadata"),
        )

    @property
    def weighted_R(self) -> float:
        """Return coherence weighted by summary confidence.

        Returns
        -------
        float
            Return coherence weighted by summary confidence.
        """
        return float(self.R * self.confidence)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe reduced child summary.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe reduced child summary.
        """
        return {
            "name": self.name,
            "channel": self.channel,
            "R": float(self.R),
            "psi": float(self.psi),
            "regime": self.regime,
            "confidence": float(self.confidence),
            "weighted_R": self.weighted_R,
            "metadata": _metadata_to_audit_record(self.metadata),
        }


@dataclass(frozen=True)
class HierarchyEscalation:
    """Bounded evidence escalated from a child to the parent supervisor."""

    child: str
    channel: str
    severity: str
    reason: str
    R: float
    confidence: float
    child_regime: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe escalation record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe escalation record.
        """
        return {
            "child": self.child,
            "channel": self.channel,
            "severity": self.severity,
            "reason": self.reason,
            "R": float(self.R),
            "confidence": float(self.confidence),
            "child_regime": self.child_regime,
        }


@dataclass(frozen=True)
class HierarchySyncEnvelope:
    """Transport-neutral hierarchy summary exchanged by edge/cloud nodes."""

    protocol_version: str
    source_node: str
    sequence: int
    summary: ChildSupervisorSummary
    monotonic_time_s: float | None = None

    def __post_init__(self) -> None:
        _validate_envelope_reduced_only(self)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe transport envelope audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe transport envelope audit record.
        """
        record: dict[str, object] = {
            "protocol_version": self.protocol_version,
            "source_node": self.source_node,
            "sequence": self.sequence,
            "summary": self.summary.to_audit_record(),
        }
        if self.monotonic_time_s is not None:
            record["monotonic_time_s"] = float(self.monotonic_time_s)
        return record

    def to_json(self) -> str:
        """Serialise the envelope with deterministic key ordering.

        Returns
        -------
        str
            Serialise the envelope with deterministic key ordering.
        """
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


def _child_escalations(
    child: ChildSupervisorSummary,
    *,
    degraded_threshold: float,
    critical_threshold: float,
    min_confidence: float,
) -> tuple[HierarchyEscalation, ...]:
    records: list[HierarchyEscalation] = []
    regime = child.regime.lower()
    if child.confidence < min_confidence:
        records.append(
            _escalation(child, "degraded", "child_summary_below_min_confidence")
        )
    if critical_threshold > child.R:
        records.append(_escalation(child, "critical", "child_coherence_below_critical"))
    elif degraded_threshold > child.R:
        records.append(_escalation(child, "degraded", "child_coherence_below_degraded"))
    if "critical" in regime and critical_threshold <= child.R:
        records.append(_escalation(child, "critical", "child_regime_escalation"))
    elif "degraded" in regime and degraded_threshold <= child.R:
        records.append(_escalation(child, "degraded", "child_regime_escalation"))
    return tuple(records)


def _dummy_summary() -> ChildSupervisorSummary:
    return ChildSupervisorSummary("validation", "validation", 1.0, 0.0)


def _validate_envelope_reduced_only(envelope: HierarchySyncEnvelope) -> None:
    if not isinstance(envelope, HierarchySyncEnvelope):
        raise ValueError("envelope must be a HierarchySyncEnvelope")
    _reject_raw_instance_attributes(envelope, "envelope")
    _require_non_empty(envelope.protocol_version, "protocol_version")
    _require_non_empty(envelope.source_node, "source_node")
    _validate_child_summary_reduced_only(envelope.summary)
    sequence = _require_integer(envelope.sequence, "sequence")
    if sequence < 0:
        raise ValueError("sequence must be >= 0")
    if envelope.monotonic_time_s is not None:
        monotonic_time_s = _require_float(
            envelope.monotonic_time_s,
            "monotonic_time_s",
        )
        if monotonic_time_s < 0.0:
            raise ValueError("monotonic_time_s must be finite and non-negative")


def _canonical_hierarchy_sync_envelope(
    envelope: HierarchySyncEnvelope,
) -> HierarchySyncEnvelope:
    _validate_envelope_reduced_only(envelope)
    return HierarchySyncEnvelope(
        protocol_version=envelope.protocol_version,
        source_node=envelope.source_node,
        sequence=envelope.sequence,
        summary=_canonical_child_summary(envelope.summary),
        monotonic_time_s=envelope.monotonic_time_s,
    )


def _canonical_child_summary(summary: ChildSupervisorSummary) -> ChildSupervisorSummary:
    _validate_child_summary_reduced_only(summary)
    return ChildSupervisorSummary(
        name=summary.name,
        channel=summary.channel,
        R=summary.R,
        psi=summary.psi,
        regime=summary.regime,
        confidence=summary.confidence,
        metadata=summary.metadata,
    )


def _validate_reduced_summary_metadata(summary: ChildSupervisorSummary) -> None:
    _validate_child_summary_reduced_only(summary)


def _validate_child_summary_reduced_only(summary: ChildSupervisorSummary) -> None:
    if not isinstance(summary, ChildSupervisorSummary):
        raise ValueError("summary must be a ChildSupervisorSummary")
    _reject_raw_instance_attributes(summary, "summary")
    _require_non_empty(summary.name, "name")
    _require_non_empty(summary.channel, "channel")
    _require_unit_interval(summary.R, "R")
    _require_finite(summary.psi, "psi")
    _require_unit_interval(summary.confidence, "confidence")
    _require_non_empty(summary.regime, "regime")
    metadata = summary.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a decoded mapping")
    _validate_metadata_value(metadata, "summary.metadata")


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if abs(value) > _MAX_JSON_SAFE_INTEGER:
        raise ValueError(
            f"{name} must be between -{_MAX_JSON_SAFE_INTEGER} "
            f"and {_MAX_JSON_SAFE_INTEGER}"
        )
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    _require_finite_number(value, name)
    return float(value)


def _validate_metadata_value(value: object, path: str) -> None:
    _normalise_metadata_value(value, path)


def _normalise_metadata_value(value: object, path: str) -> object:
    if isinstance(value, Mapping):
        normalised: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            child_path = f"{path}.{key}"
            if _is_forbidden_hierarchy_key(key):
                raise ValueError(f"{child_path} contains raw child evidence: {key}")
            normalised[key] = _normalise_metadata_value(nested, child_path)
        return MappingProxyType(normalised)
    if isinstance(value, list | tuple):
        return tuple(
            _normalise_metadata_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        try:
            _require_integer(value, path)
        except ValueError as exc:
            raise ValueError(f"{path} must be JSON-safe metadata") from exc
        return value
    if isinstance(value, float):
        try:
            _require_finite_number(value, path)
        except ValueError as exc:
            raise ValueError(f"{path} must be finite JSON-safe metadata") from exc
        return value
    raise ValueError(f"{path} must be JSON-safe metadata")


def _metadata_to_audit_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _metadata_to_audit_record(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [_metadata_to_audit_record(nested) for nested in value]
    return value


def _reject_raw_instance_attributes(instance: object, location: str) -> None:
    try:
        names = vars(instance)
    except TypeError:
        return
    forbidden = {name for name in names if _is_forbidden_hierarchy_key(name)}
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"{location} contains raw child evidence attributes: {keys}")


def _is_forbidden_hierarchy_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in _FORBIDDEN_RAW_HIERARCHY_KEYS:
        return True
    if lowered.startswith("raw_"):
        return True
    if lowered.startswith("raw") and any(
        term in lowered for term in _HIERARCHY_RAW_DATA_TERMS
    ):
        return True
    tokens = {token for token in lowered.split("_") if token}
    return bool(
        tokens.intersection(_HIERARCHY_RAW_MARKER_TERMS)
        and tokens.intersection(_HIERARCHY_RAW_DATA_TERMS)
    )


def _normalise_previous_sequences(
    previous_sequences: Mapping[str, int] | None,
) -> dict[str, int]:
    if previous_sequences is not None and not isinstance(previous_sequences, Mapping):
        raise ValueError("previous_sequences must be a mapping")
    normalised: dict[str, int] = {}
    for source_node, sequence in (previous_sequences or {}).items():
        _require_non_empty(source_node, "source_node")
        normalised[source_node] = _require_integer(sequence, "sequence")
        if normalised[source_node] < 0:
            raise ValueError("sequence must be >= 0")
    return normalised


def _duplicate_sequence_conflict_sources(
    envelopes: Sequence[HierarchySyncEnvelope],
) -> set[str]:
    grouped: dict[tuple[str, int], list[HierarchySyncEnvelope]] = {}
    for envelope in envelopes:
        grouped.setdefault((envelope.source_node, envelope.sequence), []).append(
            envelope
        )
    conflict_sources: set[str] = set()
    for (source_node, _sequence), records in grouped.items():
        if len(records) < 2:
            continue
        fingerprints = {
            json.dumps(
                record.to_audit_record(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        }
        if len(fingerprints) > 1:
            conflict_sources.add(source_node)
    return conflict_sources


def _rejection(
    envelope: HierarchySyncEnvelope,
    reason: str,
) -> dict[str, object]:
    return {
        "source_node": envelope.source_node,
        "sequence": envelope.sequence,
        "reason": reason,
        "protocol_version": envelope.protocol_version,
    }


def _rejection_sort_key(record: Mapping[str, object]) -> tuple[str, int, str]:
    return (
        str(record["source_node"]),
        _require_integer(record["sequence"], "sequence"),
        str(record["reason"]),
    )


def _escalation(
    child: ChildSupervisorSummary,
    severity: str,
    reason: str,
) -> HierarchyEscalation:
    return HierarchyEscalation(
        child=child.name,
        channel=child.channel,
        severity=severity,
        reason=reason,
        R=float(child.R),
        confidence=float(child.confidence),
        child_regime=child.regime,
    )


def _weighted_order_parameter(
    weights: FloatArray,
    phases: FloatArray,
) -> tuple[float, float]:
    if float(np.sum(weights)) == 0.0:
        return 0.0, 0.0
    vector = np.mean(weights * np.exp(1j * phases))
    return float(np.abs(vector)), float(np.angle(vector))


def _cross_child_alignment(phases: FloatArray) -> FloatArray:
    delta = phases[:, None] - phases[None, :]
    return np.asarray((1.0 + np.cos(delta)) / 2.0, dtype=np.float64)


def _parent_regime(
    parent_r: float,
    *,
    degraded_threshold: float,
    critical_threshold: float,
) -> str:
    if parent_r < critical_threshold:
        return _REGIME_CRITICAL
    if parent_r < degraded_threshold:
        return _REGIME_DEGRADED
    return _REGIME_NOMINAL


def _validate_plan_inputs(
    *,
    children: Sequence[ChildSupervisorSummary],
    hierarchy: str,
    degraded_threshold: float,
    critical_threshold: float,
    min_confidence: float,
) -> None:
    if not children:
        raise ValueError("children must contain at least one child summary")
    for child in children:
        _validate_reduced_summary_metadata(child)
    _require_non_empty(hierarchy, "hierarchy")
    _require_unit_interval(degraded_threshold, "degraded_threshold")
    _require_unit_interval(critical_threshold, "critical_threshold")
    _require_unit_interval(min_confidence, "min_confidence")
    if critical_threshold > degraded_threshold:
        raise ValueError("critical_threshold must be <= degraded_threshold")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")


def _require_finite(value: float, name: str) -> None:
    _require_finite_number(value, name)


def _require_unit_interval(value: float, name: str) -> None:
    _require_finite_number(value, name)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
