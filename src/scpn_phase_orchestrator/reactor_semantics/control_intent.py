# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor research ControlIntent

"""Digest-sealed, review-only reactor control hypotheses.

This module deliberately has no dependency on supervisor, actuation, CONTROL,
device adapters, or hardware transports.  An intent is evidence for a later
admission decision; it is never an executable command.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from .observability_profiles import DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
from .regime_ontology import (
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    AxisApplicability,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .vocabulary import (
    REVIEW_ONLY_AUTHORITY,
    ClockKind,
    EvidenceClass,
    QualityState,
    ValidityState,
    finite_real,
    non_negative_integer,
    non_negative_real,
    probability,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
)

REACTOR_CONTROL_INTENT_SCHEMA = (
    "scpn-phase-orchestrator.reactor-research-control-intent.v1"
)
REACTOR_CONTROL_INTENT_VERSION = "1.0.0"
MAX_REACTOR_CONTROL_INTENT_BYTES = 1024 * 1024
SPO_PROJECT = "SCPN-PHASE-ORCHESTRATOR"

_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TARGETS = {
    "confinement_or_assembly": frozenset({"established"}),
    "stability_or_symmetry": frozenset({"symmetric_or_quiescent"}),
    "driver_synchronization": frozenset({"synchronized"}),
    "power_or_burn": frozenset({"rising", "sustained"}),
    "exhaust_or_boundary": frozenset({"conditioned", "regulated"}),
}


class ReactorControlObjective(StrEnum):
    """Ontology axis that a research hypothesis proposes to influence."""

    CONFINEMENT_OR_ASSEMBLY = "confinement_or_assembly"
    STABILITY_OR_SYMMETRY = "stability_or_symmetry"
    DRIVER_SYNCHRONIZATION = "driver_synchronization"
    POWER_OR_BURN = "power_or_burn"
    EXHAUST_OR_BOUNDARY = "exhaust_or_boundary"


class ControlVariableDirection(StrEnum):
    """Direction of a bounded candidate relative to the reviewed baseline."""

    DECREASE = "decrease"
    HOLD = "hold"
    INCREASE = "increase"


@dataclass(frozen=True, slots=True)
class ControlVariableEnvelope:
    """Device-contract-bound candidate value that cannot execute itself."""

    variable_id: str
    units: str
    lower_bound: float
    upper_bound: float
    max_abs_delta: float
    max_abs_rate_per_s: float
    baseline_value: float
    proposed_value: float
    proposed_delta: float
    proposed_rate_per_s: float
    rate_horizon_s: float
    baseline_evidence_id: str
    baseline_timestamp_ns: int
    direction: ControlVariableDirection

    def __post_init__(self) -> None:
        """Validate and normalise the bounded variable proposal."""
        object.__setattr__(
            self,
            "variable_id",
            require_identifier(self.variable_id, field="control variable_id"),
        )
        object.__setattr__(self, "units", require_text(self.units, field="units"))
        lower = finite_real(self.lower_bound, field="lower_bound")
        upper = finite_real(self.upper_bound, field="upper_bound")
        if lower >= upper:
            raise ValueError("control variable lower_bound must be below upper_bound")
        max_delta = non_negative_real(self.max_abs_delta, field="max_abs_delta")
        max_rate = non_negative_real(
            self.max_abs_rate_per_s,
            field="max_abs_rate_per_s",
        )
        if max_delta == 0.0 or max_rate == 0.0:
            raise ValueError("control variable delta and rate limits must be positive")
        baseline = finite_real(self.baseline_value, field="baseline_value")
        value = finite_real(self.proposed_value, field="proposed_value")
        delta = finite_real(self.proposed_delta, field="proposed_delta")
        rate = finite_real(self.proposed_rate_per_s, field="proposed_rate_per_s")
        horizon = non_negative_real(self.rate_horizon_s, field="rate_horizon_s")
        if horizon == 0.0:
            raise ValueError("control variable rate_horizon_s must be positive")
        baseline_evidence = require_identifier(
            self.baseline_evidence_id,
            field="baseline_evidence_id",
        )
        baseline_timestamp = non_negative_integer(
            self.baseline_timestamp_ns,
            field="baseline_timestamp_ns",
        )
        if not lower <= baseline <= upper:
            raise ValueError("baseline_value lies outside the device contract bounds")
        if not lower <= value <= upper:
            raise ValueError("proposed_value lies outside the device contract bounds")
        if abs(delta) > max_delta:
            raise ValueError("proposed_delta exceeds the device contract limit")
        if abs(rate) > max_rate:
            raise ValueError("proposed_rate_per_s exceeds the device contract limit")
        if not _close(value, baseline + delta):
            raise ValueError(
                "proposed_value must equal baseline_value plus proposed_delta"
            )
        direction = require_enum(
            self.direction,
            ControlVariableDirection,
            field="control variable direction",
        )
        if direction is ControlVariableDirection.HOLD and delta != 0.0:
            raise ValueError("hold direction requires zero proposed_delta")
        if direction is ControlVariableDirection.HOLD and rate != 0.0:
            raise ValueError("hold direction requires zero proposed_rate_per_s")
        if direction is ControlVariableDirection.INCREASE and delta <= 0.0:
            raise ValueError("increase direction requires positive proposed_delta")
        if direction is ControlVariableDirection.INCREASE and rate <= 0.0:
            raise ValueError("increase direction requires positive proposed_rate_per_s")
        if direction is ControlVariableDirection.DECREASE and delta >= 0.0:
            raise ValueError("decrease direction requires negative proposed_delta")
        if direction is ControlVariableDirection.DECREASE and rate >= 0.0:
            raise ValueError("decrease direction requires negative proposed_rate_per_s")
        if not _close(delta, rate * horizon):
            raise ValueError(
                "proposed_delta must equal proposed_rate_per_s times rate_horizon_s"
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "max_abs_delta", max_delta)
        object.__setattr__(self, "max_abs_rate_per_s", max_rate)
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "proposed_value", value)
        object.__setattr__(self, "proposed_delta", delta)
        object.__setattr__(self, "proposed_rate_per_s", rate)
        object.__setattr__(self, "rate_horizon_s", horizon)
        object.__setattr__(self, "baseline_evidence_id", baseline_evidence)
        object.__setattr__(self, "baseline_timestamp_ns", baseline_timestamp)

    def to_record(self) -> dict[str, object]:
        """Return a complete JSON-compatible variable envelope."""
        return {
            "baseline_evidence_id": self.baseline_evidence_id,
            "baseline_timestamp_ns": self.baseline_timestamp_ns,
            "baseline_value": self.baseline_value,
            "direction": self.direction.value,
            "lower_bound": self.lower_bound,
            "max_abs_delta": self.max_abs_delta,
            "max_abs_rate_per_s": self.max_abs_rate_per_s,
            "proposed_delta": self.proposed_delta,
            "proposed_rate_per_s": self.proposed_rate_per_s,
            "proposed_value": self.proposed_value,
            "rate_horizon_s": self.rate_horizon_s,
            "units": self.units,
            "upper_bound": self.upper_bound,
            "variable_id": self.variable_id,
        }

    @classmethod
    def from_record(cls, raw: object) -> ControlVariableEnvelope:
        """Decode one strict variable envelope."""
        record = require_exact_keys(
            raw,
            required=frozenset(
                {
                    "baseline_evidence_id",
                    "baseline_timestamp_ns",
                    "baseline_value",
                    "direction",
                    "lower_bound",
                    "max_abs_delta",
                    "max_abs_rate_per_s",
                    "proposed_delta",
                    "proposed_rate_per_s",
                    "proposed_value",
                    "rate_horizon_s",
                    "units",
                    "upper_bound",
                    "variable_id",
                }
            ),
            field="control variable envelope",
        )
        try:
            direction = ControlVariableDirection(cast(str, record["direction"]))
        except ValueError as exc:
            raise ValueError("unknown control variable direction") from exc
        return cls(
            variable_id=cast(str, record["variable_id"]),
            units=cast(str, record["units"]),
            lower_bound=cast(float, record["lower_bound"]),
            upper_bound=cast(float, record["upper_bound"]),
            max_abs_delta=cast(float, record["max_abs_delta"]),
            max_abs_rate_per_s=cast(float, record["max_abs_rate_per_s"]),
            baseline_value=cast(float, record["baseline_value"]),
            proposed_value=cast(float, record["proposed_value"]),
            proposed_delta=cast(float, record["proposed_delta"]),
            proposed_rate_per_s=cast(float, record["proposed_rate_per_s"]),
            rate_horizon_s=cast(float, record["rate_horizon_s"]),
            baseline_evidence_id=cast(str, record["baseline_evidence_id"]),
            baseline_timestamp_ns=cast(int, record["baseline_timestamp_ns"]),
            direction=direction,
        )


@dataclass(frozen=True, slots=True)
class ReactorResearchControlIntent:
    """One non-executable reactor control hypothesis for CONTROL review."""

    intent_id: str
    reactor_context_id: str
    configuration: str
    event_id: str
    producer_project: str
    producer_revision: str
    producer_artifact_sha256: str
    source_handoff_schema: str
    source_handoff_sha256: str
    source_revision: str
    source_admission_schema: str
    source_admission_sha256: str
    source_admission_decision_digest: str
    source_regime_id: str
    source_regime_assignment_sha256: str
    source_regime_label: str
    source_semantic_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_class: EvidenceClass
    source_validity: ValidityState
    source_quality: QualityState
    objective: ReactorControlObjective
    hypothesized_target_regime_label: str
    effect_hypothesis: str
    device_control_contract_id: str
    device_control_contract_schema: str
    device_control_contract_sha256: str
    variable: ControlVariableEnvelope
    clock_domain: str
    clock_kind: ClockKind
    clock_epoch: str
    evidence_timestamp_ns: int
    sample_rate_hz: float
    latency_s: float
    timestamp_offset_ps: int
    issued_at_ns: int
    valid_until_ns: int
    confidence_subject_id: str
    confidence: float
    observability: float
    uncertainty_abs: float
    uncertainty_units: str
    uncertainty_basis: str
    reactor_registry_version: str
    reactor_registry_digest: str
    observability_registry_version: str
    observability_registry_digest: str
    ontology_version: str
    ontology_digest: str
    downstream_control_review_required: bool = True
    device_adapter_required: bool = True
    operator_approval_required: bool = True
    machine_protection_veto_required: bool = True
    execution_permitted: bool = False
    authority: str = REVIEW_ONLY_AUTHORITY
    actionable: bool = False
    schema: str = REACTOR_CONTROL_INTENT_SCHEMA
    schema_version: str = REACTOR_CONTROL_INTENT_VERSION

    def __post_init__(self) -> None:
        """Validate and normalise the review-only intent contract."""
        if self.schema != REACTOR_CONTROL_INTENT_SCHEMA:
            raise ValueError("unsupported reactor ControlIntent schema")
        if (
            require_semver(
                self.schema_version,
                field="ControlIntent schema_version",
            )
            != REACTOR_CONTROL_INTENT_VERSION
        ):
            raise ValueError("unsupported reactor ControlIntent version")
        for field_name in (
            "intent_id",
            "reactor_context_id",
            "event_id",
            "source_admission_schema",
            "source_handoff_schema",
            "source_regime_id",
            "device_control_contract_id",
            "device_control_contract_schema",
            "clock_domain",
            "clock_epoch",
            "confidence_subject_id",
        ):
            object.__setattr__(
                self,
                field_name,
                require_identifier(getattr(self, field_name), field=field_name),
            )
        configuration = DEFAULT_REACTOR_REGISTRY.resolve(self.configuration).identifier
        object.__setattr__(self, "configuration", configuration)
        if self.producer_project != SPO_PROJECT:
            raise ValueError("reactor ControlIntent producer must be SPO")
        for field_name in ("producer_revision", "source_revision"):
            object.__setattr__(
                self,
                field_name,
                _git_revision(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "producer_artifact_sha256",
            "source_handoff_sha256",
            "source_admission_sha256",
            "source_admission_decision_digest",
            "source_regime_assignment_sha256",
            "device_control_contract_sha256",
            "reactor_registry_digest",
            "observability_registry_digest",
            "ontology_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        semantics = _identifiers(self.source_semantic_ids, field="source_semantic_id")
        evidence = _identifiers(self.evidence_ids, field="evidence_id")
        if not semantics or not evidence:
            raise ValueError("ControlIntent requires semantic and evidence identifiers")
        if self.variable.baseline_evidence_id not in evidence:
            raise ValueError("baseline_evidence_id must be present in evidence_ids")
        object.__setattr__(self, "source_semantic_ids", semantics)
        object.__setattr__(self, "evidence_ids", evidence)
        objective = require_enum(
            self.objective,
            ReactorControlObjective,
            field="control objective",
        )
        ontology = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
        axis = ontology.resolve_axis(objective.value)
        if ontology.applicability_for(axis.axis_id, configuration) is not (
            AxisApplicability.APPLICABLE
        ):
            raise ValueError("control objective is not applicable to configuration")
        source_label = require_identifier(
            self.source_regime_label,
            field="source_regime_label",
        )
        if source_label in {"unknown", "not_applicable"}:
            raise ValueError("ControlIntent requires a classified source regime")
        if source_label not in axis.labels:
            raise ValueError(
                "source_regime_label is not defined for the objective axis"
            )
        target_label = require_identifier(
            self.hypothesized_target_regime_label,
            field="hypothesized_target_regime_label",
        )
        if target_label not in _SAFE_TARGETS[objective.value]:
            raise ValueError(
                "hypothesized target is not in the research-safe vocabulary"
            )
        object.__setattr__(self, "source_regime_label", source_label)
        object.__setattr__(self, "hypothesized_target_regime_label", target_label)
        evidence_class = require_enum(
            self.evidence_class,
            EvidenceClass,
            field="evidence_class",
        )
        if evidence_class not in {
            EvidenceClass.OBSERVED,
            EvidenceClass.EXPERIMENTAL,
            EvidenceClass.SIMULATION,
        }:
            raise ValueError(
                "ControlIntent requires observed, experimental, or simulation evidence"
            )
        if (
            require_enum(
                self.source_validity,
                ValidityState,
                field="source_validity",
            )
            is not ValidityState.VALID
        ):
            raise ValueError("ControlIntent requires valid source evidence")
        if (
            require_enum(
                self.source_quality,
                QualityState,
                field="source_quality",
            )
            is not QualityState.VALID
        ):
            raise ValueError("ControlIntent requires valid source quality")
        object.__setattr__(
            self,
            "effect_hypothesis",
            require_text(self.effect_hypothesis, field="effect_hypothesis"),
        )
        clock_kind = require_enum(self.clock_kind, ClockKind, field="clock_kind")
        if clock_kind is ClockKind.UNKNOWN:
            raise ValueError("ControlIntent requires a known clock kind")
        evidence_timestamp = non_negative_integer(
            self.evidence_timestamp_ns,
            field="evidence_timestamp_ns",
        )
        sample_rate = non_negative_real(self.sample_rate_hz, field="sample_rate_hz")
        if sample_rate == 0.0:
            raise ValueError("ControlIntent sample_rate_hz must be positive")
        latency = non_negative_real(self.latency_s, field="latency_s")
        offset = non_negative_integer(
            self.timestamp_offset_ps,
            field="timestamp_offset_ps",
        )
        if offset > 999:
            raise ValueError("timestamp_offset_ps must be in [0, 999]")
        issued = non_negative_integer(self.issued_at_ns, field="issued_at_ns")
        valid_until = non_negative_integer(
            self.valid_until_ns,
            field="valid_until_ns",
        )
        if not evidence_timestamp <= self.variable.baseline_timestamp_ns <= issued:
            raise ValueError(
                "baseline timestamp must lie between evidence and issue time"
            )
        if not evidence_timestamp <= issued <= valid_until:
            raise ValueError(
                "ControlIntent evidence, issue, and validity times are inconsistent"
            )
        object.__setattr__(self, "evidence_timestamp_ns", evidence_timestamp)
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "latency_s", latency)
        object.__setattr__(self, "timestamp_offset_ps", offset)
        object.__setattr__(self, "issued_at_ns", issued)
        object.__setattr__(self, "valid_until_ns", valid_until)
        confidence = probability(self.confidence, field="confidence")
        observability = probability(self.observability, field="observability")
        if confidence == 0.0 or observability == 0.0:
            raise ValueError(
                "ControlIntent confidence and observability must be non-zero"
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(
            self,
            "uncertainty_abs",
            non_negative_real(self.uncertainty_abs, field="uncertainty_abs"),
        )
        uncertainty_units = require_text(
            self.uncertainty_units,
            field="uncertainty_units",
        )
        if uncertainty_units != self.variable.units:
            raise ValueError(
                "uncertainty_units must match the candidate variable units"
            )
        object.__setattr__(self, "uncertainty_units", uncertainty_units)
        object.__setattr__(
            self,
            "uncertainty_basis",
            require_text(self.uncertainty_basis, field="uncertainty_basis"),
        )
        self._validate_registry_bindings()
        required_true = {
            "downstream_control_review_required": (
                self.downstream_control_review_required
            ),
            "device_adapter_required": self.device_adapter_required,
            "operator_approval_required": self.operator_approval_required,
            "machine_protection_veto_required": (self.machine_protection_veto_required),
        }
        if any(value is not True for value in required_true.values()):
            raise ValueError("all downstream ControlIntent safety gates are mandatory")
        if self.execution_permitted is not False:
            raise ValueError("reactor ControlIntent can never permit execution")
        if self.authority != REVIEW_ONLY_AUTHORITY or self.actionable is not False:
            raise ValueError("reactor ControlIntent must remain review-only")

    def _validate_registry_bindings(self) -> None:
        """Require exact bindings to the installed semantic registries."""
        if (
            self.reactor_registry_version != DEFAULT_REACTOR_REGISTRY.version
            or self.reactor_registry_digest != DEFAULT_REACTOR_REGISTRY.digest
        ):
            raise ValueError("ControlIntent reactor registry binding mismatch")
        observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        if (
            self.observability_registry_version != observability.version
            or self.observability_registry_digest != observability.digest
        ):
            raise ValueError("ControlIntent observability registry binding mismatch")
        ontology = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
        if (
            self.ontology_version != ontology.version
            or self.ontology_digest != ontology.digest
        ):
            raise ValueError("ControlIntent ontology binding mismatch")

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic ControlIntent payload."""
        return {
            "actionable": self.actionable,
            "authority": self.authority,
            "clock_domain": self.clock_domain,
            "clock_epoch": self.clock_epoch,
            "clock_kind": self.clock_kind.value,
            "confidence": self.confidence,
            "confidence_subject_id": self.confidence_subject_id,
            "configuration": self.configuration,
            "downstream_control_review_required": (
                self.downstream_control_review_required
            ),
            "device_adapter_required": self.device_adapter_required,
            "device_control_contract_id": self.device_control_contract_id,
            "device_control_contract_schema": self.device_control_contract_schema,
            "device_control_contract_sha256": self.device_control_contract_sha256,
            "effect_hypothesis": self.effect_hypothesis,
            "evidence_class": self.evidence_class.value,
            "event_id": self.event_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_timestamp_ns": self.evidence_timestamp_ns,
            "execution_permitted": self.execution_permitted,
            "intent_id": self.intent_id,
            "issued_at_ns": self.issued_at_ns,
            "latency_s": self.latency_s,
            "machine_protection_veto_required": (self.machine_protection_veto_required),
            "objective": self.objective.value,
            "observability": self.observability,
            "observability_registry_digest": self.observability_registry_digest,
            "observability_registry_version": self.observability_registry_version,
            "ontology_digest": self.ontology_digest,
            "ontology_version": self.ontology_version,
            "operator_approval_required": self.operator_approval_required,
            "producer_artifact_sha256": self.producer_artifact_sha256,
            "producer_project": self.producer_project,
            "producer_revision": self.producer_revision,
            "reactor_context_id": self.reactor_context_id,
            "reactor_registry_digest": self.reactor_registry_digest,
            "reactor_registry_version": self.reactor_registry_version,
            "sample_rate_hz": self.sample_rate_hz,
            "source_admission_decision_digest": (self.source_admission_decision_digest),
            "source_admission_schema": self.source_admission_schema,
            "source_admission_sha256": self.source_admission_sha256,
            "source_handoff_schema": self.source_handoff_schema,
            "source_handoff_sha256": self.source_handoff_sha256,
            "source_quality": self.source_quality.value,
            "source_regime_id": self.source_regime_id,
            "source_regime_assignment_sha256": (self.source_regime_assignment_sha256),
            "source_regime_label": self.source_regime_label,
            "source_revision": self.source_revision,
            "source_semantic_ids": list(self.source_semantic_ids),
            "source_validity": self.source_validity.value,
            "hypothesized_target_regime_label": (self.hypothesized_target_regime_label),
            "timestamp_offset_ps": self.timestamp_offset_ps,
            "uncertainty_abs": self.uncertainty_abs,
            "uncertainty_basis": self.uncertainty_basis,
            "uncertainty_units": self.uncertainty_units,
            "valid_until_ns": self.valid_until_ns,
            "variable": self.variable.to_record(),
        }


_INTENT_PAYLOAD_FIELDS = frozenset(
    {
        "actionable",
        "authority",
        "clock_domain",
        "clock_epoch",
        "clock_kind",
        "confidence",
        "confidence_subject_id",
        "configuration",
        "downstream_control_review_required",
        "device_adapter_required",
        "device_control_contract_id",
        "device_control_contract_schema",
        "device_control_contract_sha256",
        "effect_hypothesis",
        "evidence_class",
        "event_id",
        "evidence_ids",
        "evidence_timestamp_ns",
        "execution_permitted",
        "intent_id",
        "issued_at_ns",
        "latency_s",
        "machine_protection_veto_required",
        "objective",
        "observability",
        "observability_registry_digest",
        "observability_registry_version",
        "ontology_digest",
        "ontology_version",
        "operator_approval_required",
        "producer_artifact_sha256",
        "producer_project",
        "producer_revision",
        "reactor_context_id",
        "reactor_registry_digest",
        "reactor_registry_version",
        "sample_rate_hz",
        "source_admission_decision_digest",
        "source_admission_schema",
        "source_admission_sha256",
        "source_handoff_schema",
        "source_handoff_sha256",
        "source_quality",
        "source_regime_id",
        "source_regime_assignment_sha256",
        "source_regime_label",
        "source_revision",
        "source_semantic_ids",
        "source_validity",
        "hypothesized_target_regime_label",
        "timestamp_offset_ps",
        "uncertainty_abs",
        "uncertainty_basis",
        "uncertainty_units",
        "valid_until_ns",
        "variable",
    }
)


def control_intent_to_record(intent: ReactorResearchControlIntent) -> dict[str, object]:
    """Return a digest-sealed portable ControlIntent envelope."""
    payload = intent.to_record()
    return {
        "payload": payload,
        "payload_sha256": _canonical_digest(payload),
        "schema": intent.schema,
        "schema_version": intent.schema_version,
    }


def control_intent_from_record(raw: object) -> ReactorResearchControlIntent:
    """Decode a strict record and verify its complete identity bindings."""
    envelope = require_exact_keys(
        raw,
        required=frozenset({"payload", "payload_sha256", "schema", "schema_version"}),
        field="reactor ControlIntent envelope",
    )
    if envelope["schema"] != REACTOR_CONTROL_INTENT_SCHEMA:
        raise ValueError("unsupported reactor ControlIntent schema")
    if envelope["schema_version"] != REACTOR_CONTROL_INTENT_VERSION:
        raise ValueError("unsupported reactor ControlIntent version")
    payload = require_exact_keys(
        envelope["payload"],
        required=_INTENT_PAYLOAD_FIELDS,
        field="reactor ControlIntent payload",
    )
    supplied_digest = require_sha256(
        envelope["payload_sha256"],
        field="payload_sha256",
    )
    if supplied_digest != _canonical_digest(payload):
        raise ValueError("reactor ControlIntent payload digest mismatch")
    try:
        objective = ReactorControlObjective(cast(str, payload["objective"]))
        evidence_class = EvidenceClass(cast(str, payload["evidence_class"]))
        source_validity = ValidityState(cast(str, payload["source_validity"]))
        source_quality = QualityState(cast(str, payload["source_quality"]))
        clock_kind = ClockKind(cast(str, payload["clock_kind"]))
    except ValueError as exc:
        raise ValueError("unknown reactor ControlIntent enum value") from exc
    return ReactorResearchControlIntent(
        intent_id=cast(str, payload["intent_id"]),
        reactor_context_id=cast(str, payload["reactor_context_id"]),
        configuration=cast(str, payload["configuration"]),
        event_id=cast(str, payload["event_id"]),
        producer_project=cast(str, payload["producer_project"]),
        producer_revision=cast(str, payload["producer_revision"]),
        producer_artifact_sha256=cast(str, payload["producer_artifact_sha256"]),
        source_handoff_schema=cast(str, payload["source_handoff_schema"]),
        source_handoff_sha256=cast(str, payload["source_handoff_sha256"]),
        source_revision=cast(str, payload["source_revision"]),
        source_admission_schema=cast(str, payload["source_admission_schema"]),
        source_admission_sha256=cast(str, payload["source_admission_sha256"]),
        source_admission_decision_digest=cast(
            str,
            payload["source_admission_decision_digest"],
        ),
        source_regime_id=cast(str, payload["source_regime_id"]),
        source_regime_assignment_sha256=cast(
            str,
            payload["source_regime_assignment_sha256"],
        ),
        source_regime_label=cast(str, payload["source_regime_label"]),
        source_semantic_ids=_string_tuple(
            payload["source_semantic_ids"],
            field="source_semantic_ids",
        ),
        evidence_ids=_string_tuple(payload["evidence_ids"], field="evidence_ids"),
        evidence_class=evidence_class,
        source_validity=source_validity,
        source_quality=source_quality,
        objective=objective,
        hypothesized_target_regime_label=cast(
            str,
            payload["hypothesized_target_regime_label"],
        ),
        effect_hypothesis=cast(str, payload["effect_hypothesis"]),
        device_control_contract_id=cast(str, payload["device_control_contract_id"]),
        device_control_contract_schema=cast(
            str,
            payload["device_control_contract_schema"],
        ),
        device_control_contract_sha256=cast(
            str,
            payload["device_control_contract_sha256"],
        ),
        variable=ControlVariableEnvelope.from_record(payload["variable"]),
        clock_domain=cast(str, payload["clock_domain"]),
        clock_kind=clock_kind,
        clock_epoch=cast(str, payload["clock_epoch"]),
        evidence_timestamp_ns=cast(int, payload["evidence_timestamp_ns"]),
        sample_rate_hz=cast(float, payload["sample_rate_hz"]),
        latency_s=cast(float, payload["latency_s"]),
        timestamp_offset_ps=cast(int, payload["timestamp_offset_ps"]),
        issued_at_ns=cast(int, payload["issued_at_ns"]),
        valid_until_ns=cast(int, payload["valid_until_ns"]),
        confidence_subject_id=cast(str, payload["confidence_subject_id"]),
        confidence=cast(float, payload["confidence"]),
        observability=cast(float, payload["observability"]),
        uncertainty_abs=cast(float, payload["uncertainty_abs"]),
        uncertainty_units=cast(str, payload["uncertainty_units"]),
        uncertainty_basis=cast(str, payload["uncertainty_basis"]),
        reactor_registry_version=cast(str, payload["reactor_registry_version"]),
        reactor_registry_digest=cast(str, payload["reactor_registry_digest"]),
        observability_registry_version=cast(
            str,
            payload["observability_registry_version"],
        ),
        observability_registry_digest=cast(
            str,
            payload["observability_registry_digest"],
        ),
        ontology_version=cast(str, payload["ontology_version"]),
        ontology_digest=cast(str, payload["ontology_digest"]),
        downstream_control_review_required=cast(
            bool,
            payload["downstream_control_review_required"],
        ),
        device_adapter_required=cast(bool, payload["device_adapter_required"]),
        operator_approval_required=cast(bool, payload["operator_approval_required"]),
        machine_protection_veto_required=cast(
            bool,
            payload["machine_protection_veto_required"],
        ),
        execution_permitted=cast(bool, payload["execution_permitted"]),
        authority=cast(str, payload["authority"]),
        actionable=cast(bool, payload["actionable"]),
        schema=envelope["schema"],
        schema_version=envelope["schema_version"],
    )


def control_intent_to_bytes(intent: ReactorResearchControlIntent) -> bytes:
    """Serialize an intent to its unique canonical UTF-8 representation."""
    return _canonical_json(control_intent_to_record(intent)).encode("utf-8")


def control_intent_from_bytes(payload: bytes) -> ReactorResearchControlIntent:
    """Decode only the canonical, duplicate-free, size-bounded representation."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("ControlIntent bytes must be non-empty bytes")
    if len(payload) > MAX_REACTOR_CONTROL_INTENT_BYTES:
        raise ValueError("ControlIntent bytes exceed the maximum size")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("ControlIntent bytes must be strict UTF-8") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("ControlIntent JSON is invalid") from exc
    intent = control_intent_from_record(raw)
    if control_intent_to_bytes(intent) != payload:
        raise ValueError("ControlIntent bytes must use canonical JSON")
    return intent


def control_intent_digest(intent: ReactorResearchControlIntent) -> str:
    """Return SHA-256 of canonical ControlIntent bytes."""
    return hashlib.sha256(control_intent_to_bytes(intent)).hexdigest()


def _identifiers(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    """Validate a canonically sorted tuple of unique identifiers."""
    parsed = tuple(require_identifier(item, field=field) for item in values)
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError(f"{field} values must be unique and sorted")
    return parsed


def _git_revision(value: object, *, field: str) -> str:
    """Validate and return a lowercase 40-character Git revision."""
    revision = require_text(value, field=field)
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git revision")
    return revision


def _string_tuple(raw: object, *, field: str) -> tuple[str, ...]:
    """Decode an array of strings without coercion."""
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(raw)


def _canonical_json(payload: object) -> str:
    """Encode a value as canonical JSON text."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_digest(payload: object) -> str:
    """Return SHA-256 of canonical JSON text."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _close(left: float, right: float) -> bool:
    """Compare finite values with a scale-aware relative tolerance."""
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1e-12 * scale


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key: {key}")
        record[key] = value
    return record
