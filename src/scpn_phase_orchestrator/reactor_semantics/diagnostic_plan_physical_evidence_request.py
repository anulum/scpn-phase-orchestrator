# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — diagnostic-plan physical-evidence request
"""Derive a physical-evidence request from an accepted diagnostic-plan review.

The request preserves exact producer-plan custody while keeping every plan
channel synthetic. It identifies the physical evidence a producer must supply
for one reactor configuration before SPO may admit an observation or consider
phase qualification. Numerical-only candidates remain model coordinates and
cannot become physical selection targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, cast

from .diagnostic_plan_review import (
    DeviceDiagnosticClockReview,
    DeviceDiagnosticPlanReview,
    DeviceDiagnosticSignalReview,
    device_diagnostic_plan_review_digest,
    device_diagnostic_plan_review_from_bytes,
    device_diagnostic_plan_review_to_bytes,
)
from .observability_profiles import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    ObservabilityClass,
    ReactorSignalCandidateProfile,
)
from .producer_evidence_state import (
    PRODUCER_EVIDENCE_STATE_POLICIES,
    ProducerEvidenceStatePolicy,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY

DEVICE_PHYSICAL_EVIDENCE_REQUEST_SCHEMA: Final = (
    "scpn-phase-orchestrator.device-physical-evidence-request.v1"
)
DEVICE_PHYSICAL_EVIDENCE_REQUEST_VERSION: Final = "1.0.0"
MAX_DEVICE_PHYSICAL_EVIDENCE_REQUEST_BYTES: Final = 16 * 1024 * 1024


class DevicePhysicalEvidenceRequirementId(StrEnum):
    """Stable identifiers for producer-owned physical-evidence obligations."""

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
class DevicePhysicalEvidenceRequirement:
    """One absent producer prerequisite and its acceptance condition."""

    requirement_id: DevicePhysicalEvidenceRequirementId
    evidence_subject: str
    acceptance_condition: str
    immutable_artifact_binding_required: bool = True
    missing: bool = True


@dataclass(frozen=True, slots=True)
class DevicePhysicalCandidateRequirement:
    """One configuration-specific candidate retained without evidence promotion."""

    candidate_id: str
    phenomenon: str
    observability_class: str
    plan_disposition: str
    channel_identifiers: tuple[str, ...]
    declared_carriers: tuple[str, ...]
    clock_identifiers: tuple[str, ...]
    evidence_slots: tuple[str, ...]
    required_physical_evidence: tuple[str, ...]
    unmet_evidence: str
    reference_required: bool
    observation_operator_required: bool
    repeated_cycle_required: bool
    physical_selection_eligible: bool
    plan_revision_required: bool
    synthetic_declaration_only: bool = True
    physical_sample_present: bool = False
    evidence_claimed: bool = False
    observation_claimed: bool = False


@dataclass(frozen=True, slots=True)
class DevicePhysicalClockRequirement:
    """One plan clock and the missing physical-correlation boundary."""

    plan_clock_identifier: str
    plan_clock_kind: str
    epoch: str
    resolution_s: float
    uncertainty_s: float
    compatibility: str
    physical_correlation_required: bool
    eligible_for_physical_reference: bool
    mapping_evidence_claimed: bool = False


_REQUIREMENTS: Final = (
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PHYSICAL_SAMPLE_IDENTITY,
        "configuration-specific physical samples and immutable observation identity",
        "Supply sampled physical values with facility, shot or experiment, diagnostic "
        "channel, interval, units, array shape, missing-data semantics, source "
        "revision, package identity and content digest. Synthetic plan values and "
        "simulation output do not qualify.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.CONFIGURATION_SPECIFIC_DIAGNOSTIC_IDENTITY,
        "diagnostic identity bound to the requested reactor configuration",
        "Name the facility, device, exact reactor configuration revision, diagnostic "
        "system, channel inventory, geometry and frame. Evidence from another "
        "configuration or family cannot be inherited.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PHENOMENON_IDENTITY,
        "controlled physical phenomenon and semantic carrier",
        "Select one physically eligible registered candidate or propose a versioned "
        "registry change, then bind its meaning, carrier, sign, units, frame and "
        "non-applicability conditions. A numerical-only candidate cannot be selected "
        "as a physical phenomenon.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PHYSICAL_REFERENCE_IDENTITY,
        "measured or reconstructed physical reference",
        "Identify the physical reference signal, state or event, convention, source, "
        "uncertainty and validity. A plan clock, synthetic trigger or solver "
        "coordinate cannot substitute for a physical reference.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PHYSICAL_CLOCK_EPOCH_CORRELATION,
        "diagnostic acquisition clock correlated to facility and event epochs",
        "Supply clock identifiers, epochs, offset and drift model, correlation method, "
        "resolution, uncertainty, validity interval and immutable provenance for "
        "every sample-to-reference time mapping.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.OBSERVATION_OPERATOR_OR_CALIBRATION,
        "validated measurement operator or calibration lineage",
        "Bind raw channels to physical quantities and the selected candidate through "
        "versioned calibration or an observation operator, geometry, transfer "
        "response, units, coverage, uncertainty and validation evidence.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.UNCERTAINTY,
        "measurement, timing, reference and operator uncertainty",
        "Quantify uncertainty with method, units, confidence or coverage, "
        "correlations and propagation rules for the final observable and any "
        "derived cyclic phase.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.VALIDITY,
        "sample, interval, channel, calibration and model validity",
        "Declare exact validity domains and invalidation rules, including "
        "out-of-distribution, missing-channel and stale-calibration conditions.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PRODUCER_EVIDENCE_STATE_SEMANTICS,
        "producer-owned reason that current plant truth is not classifiable",
        "Supply distinct unknown, out_of_distribution, low_observability and stale "
        "states with criteria, precedence, interval semantics and immutable evidence "
        "bindings. Quality is orthogonal. Every state maps to U0 validity and forces "
        "an unclassified UNKNOWN physical regime.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.QUALITY,
        "provider and derived quality semantics",
        "Supply versioned sample, channel and interval quality flags plus a "
        "fail-closed mapping to accepted, degraded, rejected and unknown states.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.PROVENANCE_AND_REPRODUCIBILITY,
        "immutable source, package and transformation provenance",
        "Bind a clean immutable source revision or complete digest-bound dirty "
        "snapshot, producer package digest, environment, commands, transformations, "
        "licences and generated artefact digests sufficient for reproduction.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.OBSERVABILITY_GATE,
        "predeclared candidate-specific observability decision",
        "Declare channels, interval, statistic, band, threshold, uncertainty rule, "
        "false-positive control and minimum evidence before evaluating the target, "
        "then return an explicit fail-closed gate result.",
    ),
    DevicePhysicalEvidenceRequirement(
        DevicePhysicalEvidenceRequirementId.INDEPENDENT_VALIDATION,
        "independent validation of diagnostic mapping and extracted meaning",
        "Supply separately owned held-out, multi-shot, cross-diagnostic or replicated "
        "validation with circular same-source evaluation excluded.",
    ),
)


class DevicePhysicalEvidenceRequestRefusalCode(StrEnum):
    """Stable reason categories for request-envelope refusal."""

    INVALID_INPUT = "invalid_input"
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NONCANONICAL_BYTES = "noncanonical_bytes"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    SOURCE_REVIEW_MISMATCH = "source_review_mismatch"
    CONFIGURATION_MISMATCH = "configuration_mismatch"
    REGISTRY_BINDING_MISMATCH = "registry_binding_mismatch"
    REQUEST_CONTRACT_MISMATCH = "request_contract_mismatch"


class DevicePhysicalEvidenceRequestRefusalError(ValueError):
    """Raised when bytes cannot reconstruct an exact physical-evidence request."""

    def __init__(
        self,
        code: DevicePhysicalEvidenceRequestRefusalCode,
        detail: str,
    ) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


DevicePhysicalEvidenceRequestRefusal = DevicePhysicalEvidenceRequestRefusalError


@dataclass(frozen=True, slots=True)
class DevicePhysicalEvidenceRequest:
    """Configuration-specific request derived from one accepted design review."""

    configuration: str
    source_review_json: str
    request_id: str = field(init=False)
    requested_owner_project: str = field(init=False)
    device_project: str = field(init=False)
    source_review_id: str = field(init=False)
    source_review_sha256: str = field(init=False)
    source_revision: str = field(init=False)
    source_artifact_sha256: str = field(init=False)
    producer_package_revision: str = field(init=False)
    plan_identifier: str = field(init=False)
    source_manifest_sha256: str = field(init=False)
    source_envelope_sha256: str = field(init=False)
    source_plan_sha256: str = field(init=False)
    source_reactor_registry_version: str = field(init=False)
    source_reactor_registry_digest: str = field(init=False)
    source_observability_registry_version: str = field(init=False)
    source_observability_registry_digest: str = field(init=False)
    reactor_registry_version: str = field(init=False)
    reactor_registry_digest: str = field(init=False)
    observability_registry_version: str = field(init=False)
    observability_registry_digest: str = field(init=False)
    semantic_profile_registry_version: str = field(init=False)
    semantic_profile_registry_digest: str = field(init=False)
    candidate_requirements: tuple[DevicePhysicalCandidateRequirement, ...] = field(
        init=False
    )
    clock_requirements: tuple[DevicePhysicalClockRequirement, ...] = field(init=False)
    requirements: tuple[DevicePhysicalEvidenceRequirement, ...] = field(
        init=False, default=_REQUIREMENTS
    )
    producer_evidence_state_policies: tuple[ProducerEvidenceStatePolicy, ...] = field(
        init=False, default=PRODUCER_EVIDENCE_STATE_POLICIES
    )
    diagnostic_plan_accepted: bool = field(init=False, default=True)
    diagnostic_plan_is_physical_evidence: bool = field(init=False, default=False)
    physical_payload_schema_allocated: bool = field(init=False, default=False)
    physical_source_present: bool = field(init=False, default=False)
    selected_candidate_id: None = field(init=False, default=None)
    producer_evidence_state_contract_required: bool = field(init=False, default=True)
    producer_evidence_state_contract_present: bool = field(init=False, default=False)
    quality_state_may_substitute_for_evidence_state: bool = field(
        init=False, default=False
    )
    observation_admitted: bool = field(init=False, default=False)
    phase_inference_eligible: bool = field(init=False, default=False)
    phase_inference_performed: bool = field(init=False, default=False)
    semantic_ingress_declared: bool = field(init=False, default=False)
    control_admission_requested: bool = field(init=False, default=False)
    control_intent_created: bool = field(init=False, default=False)
    qualification_state: str = field(
        init=False, default="blocked_missing_physical_producer_evidence"
    )
    actionable: bool = field(init=False, default=False)
    execution_permitted: bool = field(init=False, default=False)
    direct_actuation: bool = field(init=False, default=False)
    authority: str = field(init=False, default="review_only")
    machine_protection_final_veto: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, str) or not self.configuration:
            _refuse(
                DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
                "configuration must be non-empty text",
            )
        if not isinstance(self.source_review_json, str) or not self.source_review_json:
            _refuse(
                DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
                "source review JSON must be non-empty text",
            )
        try:
            review = device_diagnostic_plan_review_from_bytes(
                self.source_review_json.encode("utf-8")
            )
        except (UnicodeEncodeError, ValueError) as exc:
            _refuse(
                DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
                f"embedded diagnostic-plan review refused: {exc}",
            )
        _validate_review(review, self.configuration)
        observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        profiles = observability.for_configuration(self.configuration)
        candidates = tuple(
            _candidate_requirement(profile, review) for profile in profiles
        )
        used_clock_ids = {
            identifier
            for candidate in candidates
            for identifier in candidate.clock_identifiers
        }
        clocks = tuple(
            _clock_requirement(clock)
            for clock in review.clock_reviews
            if clock.plan_clock_identifier in used_clock_ids
        )
        source_review_sha256 = device_diagnostic_plan_review_digest(review)
        values: dict[str, object] = {
            "requested_owner_project": review.source_project,
            "device_project": review.source_project,
            "source_review_id": review.review_id,
            "source_review_sha256": source_review_sha256,
            "source_revision": review.source_revision,
            "source_artifact_sha256": review.source_artifact_sha256,
            "producer_package_revision": review.producer_package_revision,
            "plan_identifier": review.plan_identifier,
            "source_manifest_sha256": review.source_manifest_sha256,
            "source_envelope_sha256": review.source_envelope_sha256,
            "source_plan_sha256": review.source_plan_sha256,
            "source_reactor_registry_version": review.source_reactor_registry_version,
            "source_reactor_registry_digest": review.source_reactor_registry_digest,
            "source_observability_registry_version": (
                review.source_observability_registry_version
            ),
            "source_observability_registry_digest": (
                review.source_observability_registry_digest
            ),
            "reactor_registry_version": DEFAULT_REACTOR_REGISTRY.version,
            "reactor_registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
            "observability_registry_version": observability.version,
            "observability_registry_digest": observability.digest,
            "semantic_profile_registry_version": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version
            ),
            "semantic_profile_registry_digest": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
            ),
            "candidate_requirements": candidates,
            "clock_requirements": clocks,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        identity = {
            "candidate_requirements": [_candidate_record(item) for item in candidates],
            "clock_requirements": [_clock_record(item) for item in clocks],
            "configuration": self.configuration,
            "observability_registry_digest": observability.digest,
            "producer_evidence_state_policies": [
                item.to_record() for item in PRODUCER_EVIDENCE_STATE_POLICIES
            ],
            "reactor_registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
            "requirement_ids": [item.requirement_id.value for item in _REQUIREMENTS],
            "semantic_profile_registry_digest": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
            ),
            "source_review_sha256": source_review_sha256,
        }
        object.__setattr__(self, "request_id", _sha256(_canonical(identity)))


def device_physical_evidence_request_from_plan_review(
    review: DeviceDiagnosticPlanReview,
    *,
    configuration: str,
) -> DevicePhysicalEvidenceRequest:
    """Build one physical-evidence request from an accepted plan review.

    Parameters
    ----------
    review : DeviceDiagnosticPlanReview
        Revalidated synthetic design review.
    configuration : str
        Exact configuration owned by the reviewed producer declaration.

    Returns
    -------
    DevicePhysicalEvidenceRequest
        Self-contained request with no physical or control authority.
    """
    if not isinstance(review, DeviceDiagnosticPlanReview):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
            "review must be DeviceDiagnosticPlanReview",
        )
    _validate_review(review, configuration)
    for profile in DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.for_configuration(
        configuration
    ):
        _candidate_requirement(profile, review)
    return DevicePhysicalEvidenceRequest(
        configuration=configuration,
        source_review_json=device_diagnostic_plan_review_to_bytes(review).decode(
            "utf-8"
        ),
    )


def device_physical_evidence_request_to_record(
    request: DevicePhysicalEvidenceRequest,
) -> dict[str, object]:
    """Return the complete deterministic request payload."""
    return {
        "actionable": request.actionable,
        "authority": request.authority,
        "candidate_requirements": [
            _candidate_record(item) for item in request.candidate_requirements
        ],
        "clock_requirements": [
            _clock_record(item) for item in request.clock_requirements
        ],
        "configuration": request.configuration,
        "control_admission_requested": request.control_admission_requested,
        "control_intent_created": request.control_intent_created,
        "device_project": request.device_project,
        "diagnostic_plan_accepted": request.diagnostic_plan_accepted,
        "diagnostic_plan_is_physical_evidence": (
            request.diagnostic_plan_is_physical_evidence
        ),
        "direct_actuation": request.direct_actuation,
        "execution_permitted": request.execution_permitted,
        "machine_protection_final_veto": request.machine_protection_final_veto,
        "observability_registry_digest": request.observability_registry_digest,
        "observability_registry_version": request.observability_registry_version,
        "observation_admitted": request.observation_admitted,
        "phase_inference_eligible": request.phase_inference_eligible,
        "phase_inference_performed": request.phase_inference_performed,
        "physical_payload_schema_allocated": request.physical_payload_schema_allocated,
        "physical_source_present": request.physical_source_present,
        "plan_identifier": request.plan_identifier,
        "producer_evidence_state_contract_present": (
            request.producer_evidence_state_contract_present
        ),
        "producer_evidence_state_contract_required": (
            request.producer_evidence_state_contract_required
        ),
        "producer_evidence_state_policies": [
            item.to_record() for item in request.producer_evidence_state_policies
        ],
        "producer_package_revision": request.producer_package_revision,
        "qualification_state": request.qualification_state,
        "quality_state_may_substitute_for_evidence_state": (
            request.quality_state_may_substitute_for_evidence_state
        ),
        "reactor_registry_digest": request.reactor_registry_digest,
        "reactor_registry_version": request.reactor_registry_version,
        "request_id": request.request_id,
        "requested_owner_project": request.requested_owner_project,
        "requirements": [_requirement_record(item) for item in request.requirements],
        "selected_candidate_id": request.selected_candidate_id,
        "semantic_ingress_declared": request.semantic_ingress_declared,
        "semantic_profile_registry_digest": (request.semantic_profile_registry_digest),
        "semantic_profile_registry_version": (
            request.semantic_profile_registry_version
        ),
        "source_artifact_sha256": request.source_artifact_sha256,
        "source_envelope_sha256": request.source_envelope_sha256,
        "source_manifest_sha256": request.source_manifest_sha256,
        "source_observability_registry_digest": (
            request.source_observability_registry_digest
        ),
        "source_observability_registry_version": (
            request.source_observability_registry_version
        ),
        "source_plan_sha256": request.source_plan_sha256,
        "source_reactor_registry_digest": request.source_reactor_registry_digest,
        "source_reactor_registry_version": request.source_reactor_registry_version,
        "source_review_id": request.source_review_id,
        "source_review_json": request.source_review_json,
        "source_review_sha256": request.source_review_sha256,
        "source_revision": request.source_revision,
    }


_PAYLOAD_KEYS: Final = {
    "actionable",
    "authority",
    "candidate_requirements",
    "clock_requirements",
    "configuration",
    "control_admission_requested",
    "control_intent_created",
    "device_project",
    "diagnostic_plan_accepted",
    "diagnostic_plan_is_physical_evidence",
    "direct_actuation",
    "execution_permitted",
    "machine_protection_final_veto",
    "observability_registry_digest",
    "observability_registry_version",
    "observation_admitted",
    "phase_inference_eligible",
    "phase_inference_performed",
    "physical_payload_schema_allocated",
    "physical_source_present",
    "plan_identifier",
    "producer_evidence_state_contract_present",
    "producer_evidence_state_contract_required",
    "producer_evidence_state_policies",
    "producer_package_revision",
    "qualification_state",
    "quality_state_may_substitute_for_evidence_state",
    "reactor_registry_digest",
    "reactor_registry_version",
    "request_id",
    "requested_owner_project",
    "requirements",
    "selected_candidate_id",
    "semantic_ingress_declared",
    "semantic_profile_registry_digest",
    "semantic_profile_registry_version",
    "source_artifact_sha256",
    "source_envelope_sha256",
    "source_manifest_sha256",
    "source_observability_registry_digest",
    "source_observability_registry_version",
    "source_plan_sha256",
    "source_reactor_registry_digest",
    "source_reactor_registry_version",
    "source_review_id",
    "source_review_json",
    "source_review_sha256",
    "source_revision",
}
_OUTER_KEYS: Final = {"payload", "payload_sha256", "schema", "schema_version"}


def device_physical_evidence_request_from_record(
    record: object,
) -> DevicePhysicalEvidenceRequest:
    """Rebuild a request and replay every source and registry-derived field."""
    payload = _object(record, _PAYLOAD_KEYS, "request payload")
    configuration = payload["configuration"]
    source_review_json = payload["source_review_json"]
    if not isinstance(configuration, str) or not isinstance(source_review_json, str):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "configuration and source review must be text",
        )
    request = DevicePhysicalEvidenceRequest(
        configuration=configuration,
        source_review_json=source_review_json,
    )
    if device_physical_evidence_request_to_record(request) != payload:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "stored request differs from its source-review-derived contract",
        )
    return request


def device_physical_evidence_request_to_bytes(
    request: DevicePhysicalEvidenceRequest,
) -> bytes:
    """Serialise a request as unique canonical digest-sealed JSON."""
    payload = device_physical_evidence_request_to_record(request)
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": DEVICE_PHYSICAL_EVIDENCE_REQUEST_SCHEMA,
            "schema_version": DEVICE_PHYSICAL_EVIDENCE_REQUEST_VERSION,
        }
    )


def device_physical_evidence_request_from_bytes(
    data: bytes,
    *,
    expected_sha256: str | None = None,
) -> DevicePhysicalEvidenceRequest:
    """Decode canonical bytes, verify their digest, and replay all bindings."""
    if not isinstance(data, bytes):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
            "request input must be bytes",
        )
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
            "expected_sha256 must be lowercase SHA-256 text",
        )
    if expected_sha256 is not None and _sha256(data) != expected_sha256:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "request envelope digest mismatch",
        )
    document = _decode_document(data)
    if (
        document["schema"] != DEVICE_PHYSICAL_EVIDENCE_REQUEST_SCHEMA
        or document["schema_version"] != DEVICE_PHYSICAL_EVIDENCE_REQUEST_VERSION
    ):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.UNSUPPORTED_SCHEMA,
            "unsupported request schema or version",
        )
    payload = _object(document["payload"], _PAYLOAD_KEYS, "request payload")
    if document["payload_sha256"] != _sha256(_canonical(payload)):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "request payload digest mismatch",
        )
    return device_physical_evidence_request_from_record(payload)


def device_physical_evidence_request_digest(
    request: DevicePhysicalEvidenceRequest,
) -> str:
    """Return the SHA-256 of the complete canonical request envelope."""
    return _sha256(device_physical_evidence_request_to_bytes(request))


def _validate_review(review: DeviceDiagnosticPlanReview, configuration: str) -> None:
    """Validate review identity and authority for the selected configuration."""
    if (
        not review.accepted_as_design_declaration
        or review.evidence_claimed
        or review.observation_claimed
        or review.measurement_claimed
        or review.facility_binding_claimed
        or review.classification_performed
        or review.semantic_ingress_declared
        or review.control_intent_created
        or review.authority != "review_only"
        or review.actionable
    ):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
            "source review is not an accepted non-evidentiary design declaration",
        )
    if configuration not in review.configurations:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
            "configuration is not owned by the reviewed producer plan",
        )
    try:
        device_profile = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(
            configuration
        )
    except ValueError as exc:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
            f"configuration is not registered: {exc}",
        )
    if device_profile.device_project != review.source_project:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
            "review source project no longer owns the configuration",
        )
    expected_registry = (
        DEFAULT_REACTOR_REGISTRY.version,
        DEFAULT_REACTOR_REGISTRY.digest,
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version,
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest,
        DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version,
        DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest,
    )
    actual_registry = (
        review.reactor_registry_version,
        review.reactor_registry_digest,
        review.observability_registry_version,
        review.observability_registry_digest,
        review.semantic_profile_registry_version,
        review.semantic_profile_registry_digest,
    )
    if actual_registry != expected_registry:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REGISTRY_BINDING_MISMATCH,
            "source review current-registry bindings differ from installed authority",
        )


def _candidate_requirement(
    profile: ReactorSignalCandidateProfile,
    review: DeviceDiagnosticPlanReview,
) -> DevicePhysicalCandidateRequirement:
    """Build a physical-evidence requirement from one diagnostic profile."""
    signals = tuple(
        signal
        for signal in review.signal_reviews
        if signal.candidate_id == profile.candidate_id
    )
    deferred = profile.candidate_id in review.deferred_candidate_ids
    if deferred == bool(signals):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
            "candidate must be either planned or deferred",
        )
    if any(not _signal_matches_profile(signal, profile) for signal in signals):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REGISTRY_BINDING_MISMATCH,
            f"reviewed signal semantics drifted for {profile.candidate_id}",
        )
    physical = profile.observability_class not in {
        ObservabilityClass.NUMERICAL_ONLY,
        ObservabilityClass.UNOBSERVABLE,
    }
    return DevicePhysicalCandidateRequirement(
        candidate_id=profile.candidate_id,
        phenomenon=profile.phenomenon,
        observability_class=profile.observability_class.value,
        plan_disposition="deferred" if deferred else "planned",
        channel_identifiers=tuple(
            sorted(signal.channel_identifier for signal in signals)
        ),
        declared_carriers=tuple(sorted({signal.carrier.value for signal in signals})),
        clock_identifiers=tuple(
            sorted({signal.clock_identifier for signal in signals})
        ),
        evidence_slots=tuple(
            sorted({slot for signal in signals for slot in signal.evidence_slots})
        ),
        required_physical_evidence=profile.required_evidence,
        unmet_evidence=profile.unmet_evidence.value,
        reference_required=profile.reference_required,
        observation_operator_required=profile.observation_operator_required,
        repeated_cycle_required=profile.repeated_cycle_required,
        physical_selection_eligible=physical and not deferred,
        plan_revision_required=deferred,
    )


def _signal_matches_profile(
    signal: DeviceDiagnosticSignalReview,
    profile: ReactorSignalCandidateProfile,
) -> bool:
    """Report whether a reviewed signal matches the diagnostic profile."""
    return (
        signal.synthetic
        and not signal.evidence_claimed
        and not signal.observation_claimed
        and signal.observability_class is profile.observability_class
        and signal.carrier in profile.admissible_carriers
        and set(signal.evidence_slots) == set(profile.required_evidence)
    )


def _clock_requirement(
    clock: DeviceDiagnosticClockReview,
) -> DevicePhysicalClockRequirement:
    """Build one physical clock requirement from its review."""
    physical = clock.plan_clock_kind != "simulation"
    return DevicePhysicalClockRequirement(
        plan_clock_identifier=clock.plan_clock_identifier,
        plan_clock_kind=clock.plan_clock_kind,
        epoch=clock.epoch,
        resolution_s=clock.resolution_s,
        uncertainty_s=clock.uncertainty_s,
        compatibility=clock.compatibility.value,
        physical_correlation_required=physical,
        eligible_for_physical_reference=physical,
        mapping_evidence_claimed=clock.mapping_evidence_claimed,
    )


def _candidate_record(
    item: DevicePhysicalCandidateRequirement,
) -> dict[str, object]:
    """Serialize one candidate requirement into its canonical record."""
    return {
        "candidate_id": item.candidate_id,
        "channel_identifiers": list(item.channel_identifiers),
        "clock_identifiers": list(item.clock_identifiers),
        "declared_carriers": list(item.declared_carriers),
        "evidence_claimed": item.evidence_claimed,
        "evidence_slots": list(item.evidence_slots),
        "observation_claimed": item.observation_claimed,
        "observation_operator_required": item.observation_operator_required,
        "observability_class": item.observability_class,
        "phenomenon": item.phenomenon,
        "physical_sample_present": item.physical_sample_present,
        "physical_selection_eligible": item.physical_selection_eligible,
        "plan_disposition": item.plan_disposition,
        "plan_revision_required": item.plan_revision_required,
        "reference_required": item.reference_required,
        "repeated_cycle_required": item.repeated_cycle_required,
        "required_physical_evidence": list(item.required_physical_evidence),
        "synthetic_declaration_only": item.synthetic_declaration_only,
        "unmet_evidence": item.unmet_evidence,
    }


def _clock_record(item: DevicePhysicalClockRequirement) -> dict[str, object]:
    """Serialize one reviewed clock into its canonical record."""
    return {
        "compatibility": item.compatibility,
        "eligible_for_physical_reference": item.eligible_for_physical_reference,
        "epoch": item.epoch,
        "mapping_evidence_claimed": item.mapping_evidence_claimed,
        "physical_correlation_required": item.physical_correlation_required,
        "plan_clock_identifier": item.plan_clock_identifier,
        "plan_clock_kind": item.plan_clock_kind,
        "resolution_s": item.resolution_s,
        "uncertainty_s": item.uncertainty_s,
    }


def _requirement_record(
    item: DevicePhysicalEvidenceRequirement,
) -> dict[str, object]:
    """Serialize one physical-evidence requirement into its canonical record."""
    return {
        "acceptance_condition": item.acceptance_condition,
        "evidence_subject": item.evidence_subject,
        "immutable_artifact_binding_required": (
            item.immutable_artifact_binding_required
        ),
        "missing": item.missing,
        "requirement_id": item.requirement_id.value,
    }


def _decode_document(data: bytes) -> dict[str, object]:
    """Decode and validate one canonical JSON document."""
    if not data or len(data) > MAX_DEVICE_PHYSICAL_EVIDENCE_REQUEST_BYTES:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
            "request byte length is outside the accepted range",
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
            DevicePhysicalEvidenceRequestRefusalCode.INVALID_JSON,
            f"request JSON invalid: {exc}",
        )
    document = _object(value, _OUTER_KEYS, "request document")
    if _canonical(document) != data:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.NONCANONICAL_BYTES,
            "request is not unique canonical JSON",
        )
    return document


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    """Require an object with exactly the expected keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} must be an object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} keys differ from contract",
        )
    return result


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys while decoding JSON."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _refuse(
                DevicePhysicalEvidenceRequestRefusalCode.DUPLICATE_JSON_KEY,
                f"duplicate key {key}",
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    """Reject non-finite numeric constants while decoding JSON."""
    _refuse(
        DevicePhysicalEvidenceRequestRefusalCode.INVALID_JSON,
        f"nonfinite constant {value}",
    )


def _canonical(value: object) -> bytes:
    """Encode a value as byte-canonical JSON."""
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
    """Return the SHA-256 digest of the supplied bytes."""
    return hashlib.sha256(value).hexdigest()


def _refuse(
    code: DevicePhysicalEvidenceRequestRefusalCode,
    detail: str,
) -> NoReturn:
    """Raise the typed refusal for this contract."""
    raise DevicePhysicalEvidenceRequestRefusalError(code, detail)


__all__ = [
    "DEVICE_PHYSICAL_EVIDENCE_REQUEST_SCHEMA",
    "DEVICE_PHYSICAL_EVIDENCE_REQUEST_VERSION",
    "MAX_DEVICE_PHYSICAL_EVIDENCE_REQUEST_BYTES",
    "DevicePhysicalCandidateRequirement",
    "DevicePhysicalClockRequirement",
    "DevicePhysicalEvidenceRequest",
    "DevicePhysicalEvidenceRequestRefusal",
    "DevicePhysicalEvidenceRequestRefusalCode",
    "DevicePhysicalEvidenceRequestRefusalError",
    "DevicePhysicalEvidenceRequirement",
    "DevicePhysicalEvidenceRequirementId",
    "device_physical_evidence_request_digest",
    "device_physical_evidence_request_from_bytes",
    "device_physical_evidence_request_from_plan_review",
    "device_physical_evidence_request_from_record",
    "device_physical_evidence_request_to_bytes",
    "device_physical_evidence_request_to_record",
]
