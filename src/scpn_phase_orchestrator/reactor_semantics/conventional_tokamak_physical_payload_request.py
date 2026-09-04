# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Conventional-tokamak physical payload request
"""Define the exact L1 producer request for conventional-tokamak evidence.

The request binds the existing simulation-only FUSION review adapter, but does
not reinterpret its TORAX output as a physical sample. It names the evidence a
new producer payload must carry before SPO can review a physical observation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, cast

from .observability_profiles import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    ReactorSignalCandidateProfile,
)
from .producer_evidence_state import (
    PRODUCER_EVIDENCE_STATE_POLICIES,
    ProducerEvidenceStatePolicy,
)
from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY

CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA: Final = (
    "scpn-phase-orchestrator.conventional-tokamak-physical-payload-request.v1"
)
CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION: Final = "1.1.0"
MAX_CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_BYTES: Final = 1024 * 1024

_CONFIGURATION: Final = "conventional_tokamak"
_DEVICE_PROJECT: Final = "SCPN-TOKAMAK-CORE"
_PRODUCER_PROJECT: Final = "SCPN-FUSION-CORE"
_INTAKE_LANE: Final = "L1_extend_exercised_review_adapter"
_CURRENT_EVIDENCE_STATE: Final = "verified_review_adapter_simulation"


class ConventionalTokamakPhysicalPayloadRequirementId(StrEnum):
    """Stable identifiers for producer-owned physical evidence obligations."""

    PHYSICAL_SAMPLE_IDENTITY = "physical_sample_identity"
    CONFIGURATION_SPECIFIC_DIAGNOSTIC_IDENTITY = (
        "configuration_specific_diagnostic_identity"
    )
    PHENOMENON_IDENTITY = "phenomenon_identity"
    PHYSICAL_REFERENCE_IDENTITY = "physical_reference_identity"
    PHYSICAL_CLOCK_EPOCH_CORRELATION = "physical_clock_epoch_correlation"
    OBSERVATION_OPERATOR_OR_CALIBRATION = "observation_operator_or_calibration"
    UNCERTAINTY = "uncertainty"
    VALIDITY = "validity"
    PRODUCER_EVIDENCE_STATE_SEMANTICS = "producer_evidence_state_semantics"
    QUALITY = "quality"
    PROVENANCE_AND_REPRODUCIBILITY = "provenance_and_reproducibility"
    OBSERVABILITY_GATE = "observability_gate"
    INDEPENDENT_VALIDATION = "independent_validation"


@dataclass(frozen=True, slots=True)
class ConventionalTokamakPhysicalPayloadRequirement:
    """One missing producer prerequisite and its acceptance condition."""

    requirement_id: ConventionalTokamakPhysicalPayloadRequirementId
    evidence_subject: str
    acceptance_condition: str
    immutable_artifact_binding_required: bool = True
    missing: bool = True


@dataclass(frozen=True, slots=True)
class ConventionalTokamakPhysicalCandidateRequirement:
    """One applicable observability candidate, recorded without selection."""

    candidate_id: str
    phenomenon: str
    observability_class: str
    admissible_carriers: tuple[str, ...]
    required_evidence: tuple[str, ...]
    unmet_evidence: str
    reference_required: bool
    observation_operator_required: bool
    repeated_cycle_required: bool
    evidence_claimed: bool = False


@dataclass(frozen=True, slots=True)
class ConventionalTokamakAdapterBinding:
    """Exact identity of the exercised adapter that this L1 request extends."""

    ingress_state: str
    producer_project: str
    source_schema: str
    adapter_api: str
    handoff_schema: str
    semantic_profile: str
    semantic_profile_version: str
    evidence_state: str = _CURRENT_EVIDENCE_STATE
    source_kind: str = "simulation"
    physical_source_present: bool = False
    reusable_as_physical_evidence: bool = False


_REQUIREMENTS: Final = (
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PHYSICAL_SAMPLE_IDENTITY,
        "physical sample series and immutable observation identity",
        "Supply sampled physical values with diagnostic channel, shot or discharge, "
        "interval, units, array shape, missing-data semantics, source revision, and "
        "content digest; simulation or reconstructed design-plan values do not "
        "qualify.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.CONFIGURATION_SPECIFIC_DIAGNOSTIC_IDENTITY,
        "diagnostic identity bound specifically to conventional_tokamak",
        "Name the facility, device, conventional-tokamak configuration revision, "
        "diagnostic system, channel inventory, geometry and frame; generic tokamak "
        "or spherical-tokamak inheritance is forbidden.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PHENOMENON_IDENTITY,
        "controlled physical phenomenon and semantic carrier",
        "Select one registered candidate or propose a versioned registry change, then "
        "bind its physical meaning, carrier, sign, units, frame and non-applicability "
        "conditions without inferring phase from topology or a diagnostic name.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PHYSICAL_REFERENCE_IDENTITY,
        "physical reference signal, state or event",
        "Identify the measured or reconstructed reference, its convention, source, "
        "uncertainty and validity; numerical solver time and synthetic TORAX frames "
        "cannot substitute for a facility reference.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PHYSICAL_CLOCK_EPOCH_CORRELATION,
        "diagnostic acquisition clock correlated to facility and event epochs",
        "Supply clock identifiers, epochs, offset and drift model, correlation method, "
        "resolution, uncertainty, validity interval and immutable provenance for every "
        "sample-to-reference time mapping.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.OBSERVATION_OPERATOR_OR_CALIBRATION,
        "validated measurement operator or calibration lineage",
        "Bind raw channels to physical quantities and the selected candidate through "
        "a versioned calibration or observation operator, geometry, transfer response, "
        "units, coverage, uncertainty and independent validation evidence.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.UNCERTAINTY,
        "measurement, timing, reference and operator uncertainty",
        "Quantify uncertainty with method, units, confidence or coverage, correlations "
        "and propagation rules for the final observable and any derived phase.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.VALIDITY,
        "sample, interval, channel, calibration and model validity",
        "Declare exact validity domains and invalidation rules, including "
        "out-of-domain, "
        "missing-channel and stale-calibration refusal conditions.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PRODUCER_EVIDENCE_STATE_SEMANTICS,
        "producer-owned evidence disposition about current plant truth",
        "Supply distinct, non-overlapping unknown, out_of_distribution, "
        "low_observability, and stale dispositions with versioned classification "
        "criteria, precedence, transitions, and interval semantics. Bind every "
        "disposition to physical sample identity, qualified clock correlation, "
        "calibration or observation-operator revision, validity domain, and the "
        "predeclared observability-gate result. Provider quality is orthogonal and "
        "cannot replace or erase the evidence cause. Each disposition must use the "
        "shared U0 validity mapping and force an unclassified UNKNOWN physical "
        "regime; none is a plasma or reactor-regime label.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.QUALITY,
        "provider and derived quality semantics",
        "Supply versioned sample, channel and interval quality flags and a fail-closed "
        "mapping to accepted, degraded, rejected and unknown states.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.PROVENANCE_AND_REPRODUCIBILITY,
        "immutable source, package and transformation provenance",
        "Bind a clean immutable source revision or complete digest-bound dirty "
        "snapshot, "
        "producer package and wheel digest, environment, commands, source files, "
        "transformations, licences and generated artifact digests for reproduction.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.OBSERVABILITY_GATE,
        "predeclared candidate-specific observability decision",
        "Declare channels, interval, statistic, band, threshold, uncertainty rule, "
        "false-positive control and minimum evidence before evaluating the target, and "
        "return an explicit fail-closed gate result.",
    ),
    ConventionalTokamakPhysicalPayloadRequirement(
        ConventionalTokamakPhysicalPayloadRequirementId.INDEPENDENT_VALIDATION,
        "independent validation of diagnostic mapping and extracted meaning",
        "Supply a separately owned multi-shot, held-out or cross-diagnostic validation "
        "record with non-overlapping evaluation custody and same-shot circularity "
        "excluded.",
    ),
)


class ConventionalTokamakPhysicalPayloadRequestRefusalCode(StrEnum):
    """Stable refusal categories for request-envelope intake."""

    INVALID_INPUT = "invalid_input"
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NONCANONICAL_BYTES = "noncanonical_bytes"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    REQUEST_CONTRACT_MISMATCH = "request_contract_mismatch"


class ConventionalTokamakPhysicalPayloadRequestRefusalError(ValueError):
    """Raised when bytes cannot reconstruct the exact producer request."""

    def __init__(
        self,
        code: ConventionalTokamakPhysicalPayloadRequestRefusalCode,
        detail: str,
    ) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ConventionalTokamakPhysicalPayloadRequest:
    """Digest-bound L1 request that carries no physical evidence itself."""

    request_id: str = field(init=False)
    requested_owner_project: str = field(init=False, default=_PRODUCER_PROJECT)
    device_project: str = field(init=False, default=_DEVICE_PROJECT)
    configuration: str = field(init=False, default=_CONFIGURATION)
    intake_lane: str = field(init=False, default=_INTAKE_LANE)
    current_adapter: ConventionalTokamakAdapterBinding = field(init=False)
    semantic_profile_registry_version: str = field(init=False)
    semantic_profile_registry_sha256: str = field(init=False)
    observability_registry_version: str = field(init=False)
    observability_registry_sha256: str = field(init=False)
    candidate_requirements: tuple[
        ConventionalTokamakPhysicalCandidateRequirement, ...
    ] = field(init=False)
    requirements: tuple[ConventionalTokamakPhysicalPayloadRequirement, ...] = field(
        init=False, default=_REQUIREMENTS
    )
    producer_evidence_state_policies: tuple[ProducerEvidenceStatePolicy, ...] = field(
        init=False, default=PRODUCER_EVIDENCE_STATE_POLICIES
    )
    producer_evidence_state_contract_required: bool = field(init=False, default=True)
    producer_evidence_state_contract_present: bool = field(init=False, default=False)
    quality_state_may_substitute_for_evidence_state: bool = field(
        init=False, default=False
    )
    selected_candidate_id: None = field(init=False, default=None)
    physical_payload_schema_allocated: bool = field(init=False, default=False)
    physical_source_present: bool = field(init=False, default=False)
    observation_admitted: bool = field(init=False, default=False)
    phase_inference_eligible: bool = field(init=False, default=False)
    phase_inference_performed: bool = field(init=False, default=False)
    semantic_ingress_extended: bool = field(init=False, default=False)
    control_admission_requested: bool = field(init=False, default=False)
    control_intent_created: bool = field(init=False, default=False)
    qualification_state: str = field(
        init=False, default="blocked_missing_physical_producer_payload"
    )
    actionable: bool = field(init=False, default=False)
    execution_permitted: bool = field(init=False, default=False)
    direct_actuation: bool = field(init=False, default=False)
    review_only: bool = field(init=False, default=True)
    machine_protection_final_veto: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        semantic_registry = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
        profile = semantic_registry.resolve(_CONFIGURATION)
        expected = (
            _DEVICE_PROJECT,
            _PRODUCER_PROJECT,
            "scpn-fusion-core.torax-runtime-review-envelope.v1",
            "scpn_phase_orchestrator.reactor_semantics."
            "coupled_transport_handoff_from_fusion_bytes",
            "scpn-phase-orchestrator.reactor-semantic-handoff.v1",
            "spo.reactor.conventional_tokamak.coupled_transport.nonphase_review.v1",
            "1.0.0",
        )
        actual = (
            profile.device_project,
            profile.producer_project,
            profile.source_schema,
            profile.adapter_api,
            profile.handoff_schema,
            profile.semantic_profile,
            profile.semantic_profile_version,
        )
        if (
            actual != expected
            or profile.ingress_state.value != "verified_review_adapter"
        ):
            _refuse(
                ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
                "conventional-tokamak semantic adapter binding changed",
            )
        adapter = ConventionalTokamakAdapterBinding(
            ingress_state=profile.ingress_state.value,
            producer_project=cast(str, profile.producer_project),
            source_schema=cast(str, profile.source_schema),
            adapter_api=cast(str, profile.adapter_api),
            handoff_schema=cast(str, profile.handoff_schema),
            semantic_profile=cast(str, profile.semantic_profile),
            semantic_profile_version=cast(str, profile.semantic_profile_version),
        )
        observability_registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        candidates = tuple(
            _candidate_requirement(item)
            for item in observability_registry.for_configuration(_CONFIGURATION)
        )
        derived: dict[str, object] = {
            "current_adapter": adapter,
            "semantic_profile_registry_version": semantic_registry.version,
            "semantic_profile_registry_sha256": semantic_registry.digest,
            "observability_registry_version": observability_registry.version,
            "observability_registry_sha256": observability_registry.digest,
            "candidate_requirements": candidates,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)
        identity = {
            "adapter": _adapter_record(adapter),
            "candidate_ids": [item.candidate_id for item in candidates],
            "configuration": _CONFIGURATION,
            "observability_registry_sha256": observability_registry.digest,
            "producer_evidence_state_policies": [
                item.to_record() for item in PRODUCER_EVIDENCE_STATE_POLICIES
            ],
            "requirement_ids": [item.requirement_id.value for item in _REQUIREMENTS],
            "semantic_profile_registry_sha256": semantic_registry.digest,
        }
        object.__setattr__(self, "request_id", _sha256(_canonical(identity)))


def conventional_tokamak_physical_payload_request() -> (
    ConventionalTokamakPhysicalPayloadRequest
):
    """Build the current exact L1 producer prerequisite request."""
    return ConventionalTokamakPhysicalPayloadRequest()


def conventional_tokamak_physical_payload_request_to_record(
    request: ConventionalTokamakPhysicalPayloadRequest,
) -> dict[str, object]:
    """Return the complete deterministic request payload."""
    return {
        "actionable": request.actionable,
        "candidate_requirements": [
            _candidate_record(item) for item in request.candidate_requirements
        ],
        "configuration": request.configuration,
        "control_admission_requested": request.control_admission_requested,
        "control_intent_created": request.control_intent_created,
        "current_adapter": _adapter_record(request.current_adapter),
        "device_project": request.device_project,
        "direct_actuation": request.direct_actuation,
        "execution_permitted": request.execution_permitted,
        "intake_lane": request.intake_lane,
        "machine_protection_final_veto": request.machine_protection_final_veto,
        "observability_registry_sha256": request.observability_registry_sha256,
        "observability_registry_version": request.observability_registry_version,
        "observation_admitted": request.observation_admitted,
        "phase_inference_eligible": request.phase_inference_eligible,
        "phase_inference_performed": request.phase_inference_performed,
        "physical_payload_schema_allocated": request.physical_payload_schema_allocated,
        "physical_source_present": request.physical_source_present,
        "producer_evidence_state_contract_present": (
            request.producer_evidence_state_contract_present
        ),
        "producer_evidence_state_contract_required": (
            request.producer_evidence_state_contract_required
        ),
        "producer_evidence_state_policies": [
            item.to_record() for item in request.producer_evidence_state_policies
        ],
        "qualification_state": request.qualification_state,
        "request_id": request.request_id,
        "requested_owner_project": request.requested_owner_project,
        "requirements": [_requirement_record(item) for item in request.requirements],
        "review_only": request.review_only,
        "quality_state_may_substitute_for_evidence_state": (
            request.quality_state_may_substitute_for_evidence_state
        ),
        "selected_candidate_id": request.selected_candidate_id,
        "semantic_ingress_extended": request.semantic_ingress_extended,
        "semantic_profile_registry_sha256": (request.semantic_profile_registry_sha256),
        "semantic_profile_registry_version": (
            request.semantic_profile_registry_version
        ),
    }


_PAYLOAD_KEYS: Final = {
    "actionable",
    "candidate_requirements",
    "configuration",
    "control_admission_requested",
    "control_intent_created",
    "current_adapter",
    "device_project",
    "direct_actuation",
    "execution_permitted",
    "intake_lane",
    "machine_protection_final_veto",
    "observability_registry_sha256",
    "observability_registry_version",
    "observation_admitted",
    "phase_inference_eligible",
    "phase_inference_performed",
    "physical_payload_schema_allocated",
    "physical_source_present",
    "producer_evidence_state_contract_present",
    "producer_evidence_state_contract_required",
    "producer_evidence_state_policies",
    "qualification_state",
    "request_id",
    "requested_owner_project",
    "requirements",
    "review_only",
    "quality_state_may_substitute_for_evidence_state",
    "selected_candidate_id",
    "semantic_ingress_extended",
    "semantic_profile_registry_sha256",
    "semantic_profile_registry_version",
}
_OUTER_KEYS: Final = {"payload", "payload_sha256", "schema", "schema_version"}


def conventional_tokamak_physical_payload_request_from_record(
    record: object,
) -> ConventionalTokamakPhysicalPayloadRequest:
    """Rebuild the static request and compare every registry-derived field."""
    payload = _object(record, _PAYLOAD_KEYS, "request payload")
    request = ConventionalTokamakPhysicalPayloadRequest()
    if conventional_tokamak_physical_payload_request_to_record(request) != payload:
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "stored request differs from current registry-derived contract",
        )
    return request


def conventional_tokamak_physical_payload_request_to_bytes(
    request: ConventionalTokamakPhysicalPayloadRequest,
) -> bytes:
    """Serialize the request as unique canonical digest-sealed JSON."""
    payload = conventional_tokamak_physical_payload_request_to_record(request)
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA,
            "schema_version": CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION,
        }
    )


def conventional_tokamak_physical_payload_request_from_bytes(
    data: bytes,
    *,
    expected_sha256: str | None = None,
) -> ConventionalTokamakPhysicalPayloadRequest:
    """Decode canonical bytes, verify their digest, and replay all bindings."""
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.INVALID_INPUT,
            "expected_sha256 must be lowercase SHA-256 text",
        )
    if expected_sha256 is not None and _sha256(data) != expected_sha256:
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "request envelope digest mismatch",
        )
    document = _decode_document(data)
    if (
        document["schema"] != CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA
        or document["schema_version"]
        != CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION
    ):
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.UNSUPPORTED_SCHEMA,
            "unsupported request schema or version",
        )
    payload = _object(document["payload"], _PAYLOAD_KEYS, "request payload")
    if document["payload_sha256"] != _sha256(_canonical(payload)):
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "request payload digest mismatch",
        )
    return conventional_tokamak_physical_payload_request_from_record(payload)


def conventional_tokamak_physical_payload_request_digest(
    request: ConventionalTokamakPhysicalPayloadRequest,
) -> str:
    """Return the SHA-256 of the complete canonical request envelope."""
    return _sha256(conventional_tokamak_physical_payload_request_to_bytes(request))


def _candidate_requirement(
    profile: ReactorSignalCandidateProfile,
) -> ConventionalTokamakPhysicalCandidateRequirement:
    return ConventionalTokamakPhysicalCandidateRequirement(
        candidate_id=profile.candidate_id,
        phenomenon=profile.phenomenon,
        observability_class=profile.observability_class.value,
        admissible_carriers=tuple(item.value for item in profile.admissible_carriers),
        required_evidence=profile.required_evidence,
        unmet_evidence=profile.unmet_evidence.value,
        reference_required=profile.reference_required,
        observation_operator_required=profile.observation_operator_required,
        repeated_cycle_required=profile.repeated_cycle_required,
    )


def _candidate_record(
    item: ConventionalTokamakPhysicalCandidateRequirement,
) -> dict[str, object]:
    return {
        "admissible_carriers": list(item.admissible_carriers),
        "candidate_id": item.candidate_id,
        "evidence_claimed": item.evidence_claimed,
        "observation_operator_required": item.observation_operator_required,
        "observability_class": item.observability_class,
        "phenomenon": item.phenomenon,
        "reference_required": item.reference_required,
        "repeated_cycle_required": item.repeated_cycle_required,
        "required_evidence": list(item.required_evidence),
        "unmet_evidence": item.unmet_evidence,
    }


def _requirement_record(
    item: ConventionalTokamakPhysicalPayloadRequirement,
) -> dict[str, object]:
    return {
        "acceptance_condition": item.acceptance_condition,
        "evidence_subject": item.evidence_subject,
        "immutable_artifact_binding_required": (
            item.immutable_artifact_binding_required
        ),
        "missing": item.missing,
        "requirement_id": item.requirement_id.value,
    }


def _adapter_record(item: ConventionalTokamakAdapterBinding) -> dict[str, object]:
    return {
        "adapter_api": item.adapter_api,
        "evidence_state": item.evidence_state,
        "handoff_schema": item.handoff_schema,
        "ingress_state": item.ingress_state,
        "physical_source_present": item.physical_source_present,
        "producer_project": item.producer_project,
        "reusable_as_physical_evidence": item.reusable_as_physical_evidence,
        "semantic_profile": item.semantic_profile,
        "semantic_profile_version": item.semantic_profile_version,
        "source_kind": item.source_kind,
        "source_schema": item.source_schema,
    }


def _decode_document(data: bytes) -> dict[str, object]:
    if (
        not isinstance(data, bytes)
        or not data
        or len(data) > MAX_CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_BYTES
    ):
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.INVALID_INPUT,
            "request byte input invalid",
        )
    try:
        value = cast(
            object,
            json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_reject_duplicates,
                parse_constant=_reject_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.INVALID_JSON,
            f"request JSON invalid: {exc}",
        )
    document = _object(value, _OUTER_KEYS, "request document")
    if _canonical(document) != data:
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.NONCANONICAL_BYTES,
            "request is not unique canonical JSON",
        )
    return document


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} must be an object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            ConventionalTokamakPhysicalPayloadRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} keys differ from contract",
        )
    return result


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _refuse(
                ConventionalTokamakPhysicalPayloadRequestRefusalCode.DUPLICATE_JSON_KEY,
                f"duplicate key {key}",
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _refuse(
        ConventionalTokamakPhysicalPayloadRequestRefusalCode.INVALID_JSON,
        f"nonfinite constant {value}",
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _refuse(
    code: ConventionalTokamakPhysicalPayloadRequestRefusalCode,
    detail: str,
) -> NoReturn:
    raise ConventionalTokamakPhysicalPayloadRequestRefusalError(code, detail)


__all__ = [
    "CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA",
    "CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION",
    "MAX_CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_BYTES",
    "ConventionalTokamakAdapterBinding",
    "ConventionalTokamakPhysicalCandidateRequirement",
    "ConventionalTokamakPhysicalPayloadRequest",
    "ConventionalTokamakPhysicalPayloadRequestRefusalCode",
    "ConventionalTokamakPhysicalPayloadRequestRefusalError",
    "ConventionalTokamakPhysicalPayloadRequirement",
    "ConventionalTokamakPhysicalPayloadRequirementId",
    "conventional_tokamak_physical_payload_request",
    "conventional_tokamak_physical_payload_request_digest",
    "conventional_tokamak_physical_payload_request_from_bytes",
    "conventional_tokamak_physical_payload_request_from_record",
    "conventional_tokamak_physical_payload_request_to_bytes",
    "conventional_tokamak_physical_payload_request_to_record",
]
