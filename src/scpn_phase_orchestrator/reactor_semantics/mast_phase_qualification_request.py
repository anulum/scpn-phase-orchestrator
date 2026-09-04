# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FAIR-MAST phase qualification request
"""Build an exact producer request from reviewed FAIR-MAST source custody.

The request names what SCPN-FUSION-CORE must supply before SPO may qualify a
physical observation or infer phase. It carries no evidence for those missing
requirements and cannot grant semantic ingress or CONTROL authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, cast

from .mast_magnetic_review import (
    MastMagneticSourceReview,
    mast_magnetic_source_review_digest,
    mast_magnetic_source_review_from_bytes,
    mast_magnetic_source_review_to_bytes,
)
from .observability_profiles import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    ReactorSignalCandidateProfile,
)
from .producer_evidence_state import (
    PRODUCER_EVIDENCE_STATE_POLICIES,
    ProducerEvidenceStatePolicy,
)

MAST_PHASE_QUALIFICATION_REQUEST_SCHEMA: Final = (
    "scpn-phase-orchestrator.mast-phase-qualification-request.v1"
)
MAST_PHASE_QUALIFICATION_REQUEST_VERSION: Final = "1.1.0"
MAX_MAST_PHASE_QUALIFICATION_REQUEST_BYTES: Final = 20 * 1024 * 1024

_PRODUCER_PROJECT: Final = "SCPN-FUSION-CORE"
_DEVICE_PROJECT: Final = "SCPN-TOKAMAK-CORE"
_CONFIGURATION: Final = "spherical_tokamak"
_FACILITY: Final = "MAST"
_SOURCE_ARCHIVE: Final = "FAIR-MAST"


class MastPhaseQualificationRequirementId(StrEnum):
    """Stable identifiers for missing physical-qualification evidence."""

    PHENOMENON_IDENTITY = "phenomenon_identity"
    REPRODUCIBLE_SOURCE_INGESTION_STATE = "reproducible_source_ingestion_state"
    CALIBRATION_LINEAGE = "calibration_lineage"
    PHYSICAL_GEOMETRY_AND_FRAME_JOIN = "physical_geometry_and_frame_join"
    MODAL_OBSERVATION_OPERATOR_AND_HARMONIC_BASIS = (
        "modal_observation_operator_and_harmonic_basis"
    )
    PROVIDER_QUALITY = "provider_quality"
    UNCERTAINTY = "uncertainty"
    VALIDITY = "validity"
    PRODUCER_EVIDENCE_STATE_SEMANTICS = "producer_evidence_state_semantics"
    INSTRUMENT_FACILITY_CLOCK_CORRELATION = "instrument_facility_clock_correlation"
    RESOLVED_EVENT_IDENTITY = "resolved_event_identity"
    OBSERVABILITY_THRESHOLD = "observability_threshold"
    INDEPENDENT_MULTI_SHOT_OR_CLASSIFIER_EVIDENCE = (
        "independent_multi_shot_or_classifier_evidence"
    )


@dataclass(frozen=True, slots=True)
class MastPhaseQualificationRequirement:
    """One producer-owned prerequisite with an explicit acceptance condition."""

    requirement_id: MastPhaseQualificationRequirementId
    evidence_subject: str
    acceptance_condition: str
    immutable_artifact_binding_required: bool = True
    missing: bool = True


@dataclass(frozen=True, slots=True)
class MastPhaseCandidateRequirement:
    """One registered spherical-tokamak phenomenon candidate, not a claim."""

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


_REQUIREMENTS: Final = (
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.PHENOMENON_IDENTITY,
        "controlled physical phenomenon and semantic carrier",
        "Select one registered candidate or propose a versioned registry change; "
        "bind its physical meaning, carrier, diagnostic reference, sign, units, "
        "frame, and non-applicability conditions without inferring it from shot ID.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.REPRODUCIBLE_SOURCE_INGESTION_STATE,
        "reproducible FAIR-MAST ingestion source state",
        "Supply either a clean immutable ingestion commit or a complete digest-bound "
        "dirty diff and source snapshot, with environment, command, mapping, license, "
        "and generated artifact digests sufficient for independent reproduction.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.CALIBRATION_LINEAGE,
        "measurement calibration lineage",
        "Bind every applied scale and background transform to calibration records, "
        "revision and digest, channel coverage, units, validity, and traceable "
        "provider or laboratory provenance.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.PHYSICAL_GEOMETRY_AND_FRAME_JOIN,
        "channel geometry and physical reference frame",
        "Join every used archive channel to physical position, orientation, probe or "
        "loop identity, coordinate frame, transformation chain, geometry revision, "
        "and geometry uncertainty.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.MODAL_OBSERVATION_OPERATOR_AND_HARMONIC_BASIS,
        "magnetic measurement to mode observation operator",
        "Supply a validated observation or transfer operator with poloidal and "
        "toroidal harmonic conventions, channel sensitivity, frequency response, "
        "aliasing limits, reference convention, and operator-validation evidence.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.PROVIDER_QUALITY,
        "provider quality and rejection semantics",
        "Supply provider-defined sample, channel, and interval quality flags plus a "
        "versioned mapping to accepted, degraded, rejected, and unknown states.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.UNCERTAINTY,
        "measurement, timing, geometry, and operator uncertainty",
        "Quantify uncertainty for calibrated values, clock correlation, geometry, "
        "and the observation operator with method, confidence or coverage, units, "
        "correlation representation, and propagation rule.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.VALIDITY,
        "shot, interval, channel, frequency, and model validity",
        "Declare exact validity windows and domains for calibration, channels, "
        "clock mapping, geometry, and operator, including invalidation and "
        "out-of-domain refusal rules.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.PRODUCER_EVIDENCE_STATE_SEMANTICS,
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
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.INSTRUMENT_FACILITY_CLOCK_CORRELATION,
        "instrument acquisition clock to facility reference",
        "Bind acquisition time to an identified facility epoch with offset, drift, "
        "monotonicity, correlation method, interval validity, uncertainty, and "
        "source provenance; a reproduced archive grid is insufficient.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.RESOLVED_EVENT_IDENTITY,
        "facility event instance and interval",
        "Bind shot 27707 to a versioned event identifier, event type, start and end "
        "on the qualified clock, event source, uncertainty, and relation to the "
        "selected physical phenomenon.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.OBSERVABILITY_THRESHOLD,
        "predeclared phenomenon-specific observability gate",
        "Declare channels, interval, band, statistic, threshold, uncertainty rule, "
        "false-positive control, minimum evidence, and fail-closed result before "
        "evaluating the target evidence.",
    ),
    MastPhaseQualificationRequirement(
        MastPhaseQualificationRequirementId.INDEPENDENT_MULTI_SHOT_OR_CLASSIFIER_EVIDENCE,
        "independent validation of phenomenon and phase extraction",
        "Supply independently reviewed multi-shot evidence or a versioned classifier "
        "with non-overlapping train and evaluation custody, class definition, "
        "calibration, confusion evidence, uncertainty, and same-shot circularity "
        "excluded.",
    ),
)


class MastPhaseQualificationRequestRefusalCode(StrEnum):
    """Stable refusal categories for request-envelope intake."""

    INVALID_INPUT = "invalid_input"
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NONCANONICAL_BYTES = "noncanonical_bytes"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    SOURCE_REVIEW_MISMATCH = "source_review_mismatch"
    REQUEST_CONTRACT_MISMATCH = "request_contract_mismatch"


class MastPhaseQualificationRequestRefusalError(ValueError):
    """Raised when bytes cannot reconstruct the exact qualification request."""

    def __init__(
        self, code: MastPhaseQualificationRequestRefusalCode, detail: str
    ) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MastPhaseQualificationRequest:
    """Digest-bound request for evidence absent from a FAIR-MAST source review."""

    source_review_json: str
    request_id: str = field(init=False)
    requested_owner_project: str = field(init=False, default=_PRODUCER_PROJECT)
    device_project: str = field(init=False, default=_DEVICE_PROJECT)
    configuration: str = field(init=False, default=_CONFIGURATION)
    facility: str = field(init=False, default=_FACILITY)
    source_archive: str = field(init=False, default=_SOURCE_ARCHIVE)
    shot_id: int = field(init=False)
    source_review_id: str = field(init=False)
    source_review_sha256: str = field(init=False)
    source_revision: str = field(init=False)
    source_artifact_sha256: str = field(init=False)
    source_archive_sha256: str = field(init=False)
    source_qualification_sha256: str = field(init=False)
    archive_payload_sha256: str = field(init=False)
    qualification_payload_sha256: str = field(init=False)
    source_ingestion_revision: str = field(init=False)
    source_ingestion_tree_state: str = field(init=False)
    source_review_unresolved_fields: tuple[str, ...] = field(init=False)
    observability_registry_version: str = field(init=False)
    observability_registry_sha256: str = field(init=False)
    candidate_requirements: tuple[MastPhaseCandidateRequirement, ...] = field(
        init=False
    )
    selected_candidate_id: None = field(init=False, default=None)
    phenomenon_identity_state: str = field(
        init=False, default="unresolved_producer_evidence_required"
    )
    requirements: tuple[MastPhaseQualificationRequirement, ...] = field(
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
    qualification_state: str = field(
        init=False, default="blocked_missing_producer_evidence"
    )
    observation_admitted: bool = field(init=False, default=False)
    phase_inference_eligible: bool = field(init=False, default=False)
    phase_inference_performed: bool = field(init=False, default=False)
    semantic_ingress_declared: bool = field(init=False, default=False)
    control_admission_requested: bool = field(init=False, default=False)
    control_intent_created: bool = field(init=False, default=False)
    actionable: bool = field(init=False, default=False)
    execution_permitted: bool = field(init=False, default=False)
    direct_actuation: bool = field(init=False, default=False)
    review_only: bool = field(init=False, default=True)
    machine_protection_final_veto: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if not isinstance(self.source_review_json, str) or not self.source_review_json:
            _refuse(
                MastPhaseQualificationRequestRefusalCode.INVALID_INPUT,
                "source review JSON must be non-empty text",
            )
        try:
            review = mast_magnetic_source_review_from_bytes(
                self.source_review_json.encode("utf-8")
            )
        except (UnicodeEncodeError, ValueError) as exc:
            _refuse(
                MastPhaseQualificationRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
                f"embedded source review refused: {exc}",
            )
        if (
            not review.accepted_as_physical_source_review
            or not review.physical_source_recorded
            or review.observation_admitted
            or review.qualified_phase_evidence
            or review.semantic_ingress_declared
            or not review.review_only
        ):
            _refuse(
                MastPhaseQualificationRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
                "source review is not the expected unqualified physical custody",
            )
        source_digest = mast_magnetic_source_review_digest(review)
        registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        candidates = tuple(
            _candidate_requirement(candidate)
            for candidate in registry.for_configuration(_CONFIGURATION)
        )
        derived: dict[str, object] = {
            "shot_id": review.shot_id,
            "source_review_id": review.review_id,
            "source_review_sha256": source_digest,
            "source_revision": review.source_revision,
            "source_artifact_sha256": review.source_artifact_sha256,
            "source_archive_sha256": review.source_archive_sha256,
            "source_qualification_sha256": review.source_qualification_sha256,
            "archive_payload_sha256": review.archive_payload_sha256,
            "qualification_payload_sha256": review.qualification_payload_sha256,
            "source_ingestion_revision": review.source_ingestion_revision,
            "source_ingestion_tree_state": review.source_ingestion_tree_state,
            "source_review_unresolved_fields": (review.unresolved_qualification_fields),
            "observability_registry_version": registry.version,
            "observability_registry_sha256": registry.digest,
            "candidate_requirements": candidates,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)
        identity = {
            "candidate_ids": [item.candidate_id for item in candidates],
            "observability_registry_sha256": registry.digest,
            "producer_evidence_state_policies": [
                item.to_record() for item in PRODUCER_EVIDENCE_STATE_POLICIES
            ],
            "requirement_ids": [item.requirement_id.value for item in _REQUIREMENTS],
            "source_review_sha256": source_digest,
        }
        object.__setattr__(self, "request_id", _sha256(_canonical(identity)))


def mast_phase_qualification_request_from_source_review(
    review: MastMagneticSourceReview,
) -> MastPhaseQualificationRequest:
    """Build the exact pending-evidence request from a physical-source review.

    Parameters
    ----------
    review : MastMagneticSourceReview
        Revalidated physical-source custody with no phase qualification.

    Returns
    -------
    MastPhaseQualificationRequest
        Self-contained review-only producer request.
    """
    return MastPhaseQualificationRequest(
        source_review_json=mast_magnetic_source_review_to_bytes(review).decode("utf-8")
    )


def mast_phase_qualification_request_to_record(
    request: MastPhaseQualificationRequest,
) -> dict[str, object]:
    """Return the deterministic request payload.

    Parameters
    ----------
    request : MastPhaseQualificationRequest
        Validated qualification request.

    Returns
    -------
    dict[str, object]
        Complete JSON-compatible request payload.
    """
    return {
        "actionable": request.actionable,
        "archive_payload_sha256": request.archive_payload_sha256,
        "candidate_requirements": [
            _candidate_record(item) for item in request.candidate_requirements
        ],
        "configuration": request.configuration,
        "control_admission_requested": request.control_admission_requested,
        "control_intent_created": request.control_intent_created,
        "device_project": request.device_project,
        "direct_actuation": request.direct_actuation,
        "execution_permitted": request.execution_permitted,
        "facility": request.facility,
        "machine_protection_final_veto": request.machine_protection_final_veto,
        "observation_admitted": request.observation_admitted,
        "observability_registry_sha256": request.observability_registry_sha256,
        "observability_registry_version": request.observability_registry_version,
        "phase_inference_eligible": request.phase_inference_eligible,
        "phase_inference_performed": request.phase_inference_performed,
        "phenomenon_identity_state": request.phenomenon_identity_state,
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
        "qualification_payload_sha256": request.qualification_payload_sha256,
        "request_id": request.request_id,
        "requested_owner_project": request.requested_owner_project,
        "requirements": [_requirement_record(item) for item in request.requirements],
        "review_only": request.review_only,
        "quality_state_may_substitute_for_evidence_state": (
            request.quality_state_may_substitute_for_evidence_state
        ),
        "selected_candidate_id": request.selected_candidate_id,
        "semantic_ingress_declared": request.semantic_ingress_declared,
        "shot_id": request.shot_id,
        "source_archive": request.source_archive,
        "source_archive_sha256": request.source_archive_sha256,
        "source_artifact_sha256": request.source_artifact_sha256,
        "source_ingestion_revision": request.source_ingestion_revision,
        "source_ingestion_tree_state": request.source_ingestion_tree_state,
        "source_qualification_sha256": request.source_qualification_sha256,
        "source_review_id": request.source_review_id,
        "source_review_json": request.source_review_json,
        "source_review_sha256": request.source_review_sha256,
        "source_review_unresolved_fields": list(
            request.source_review_unresolved_fields
        ),
        "source_revision": request.source_revision,
    }


_PAYLOAD_KEYS: Final = {
    "actionable",
    "archive_payload_sha256",
    "candidate_requirements",
    "configuration",
    "control_admission_requested",
    "control_intent_created",
    "device_project",
    "direct_actuation",
    "execution_permitted",
    "facility",
    "machine_protection_final_veto",
    "observation_admitted",
    "observability_registry_sha256",
    "observability_registry_version",
    "phase_inference_eligible",
    "phase_inference_performed",
    "phenomenon_identity_state",
    "producer_evidence_state_contract_present",
    "producer_evidence_state_contract_required",
    "producer_evidence_state_policies",
    "qualification_state",
    "qualification_payload_sha256",
    "request_id",
    "requested_owner_project",
    "requirements",
    "review_only",
    "quality_state_may_substitute_for_evidence_state",
    "selected_candidate_id",
    "semantic_ingress_declared",
    "shot_id",
    "source_archive",
    "source_archive_sha256",
    "source_artifact_sha256",
    "source_ingestion_revision",
    "source_ingestion_tree_state",
    "source_qualification_sha256",
    "source_review_id",
    "source_review_json",
    "source_review_sha256",
    "source_review_unresolved_fields",
    "source_revision",
}
_OUTER_KEYS: Final = {"payload", "payload_sha256", "schema", "schema_version"}


def mast_phase_qualification_request_from_record(
    record: object,
) -> MastPhaseQualificationRequest:
    """Rebuild a request and replay its embedded source-review invariants.

    Parameters
    ----------
    record : object
        Candidate request payload.

    Returns
    -------
    MastPhaseQualificationRequest
        Reconstructed request after exact comparison.
    """
    payload = _object(record, _PAYLOAD_KEYS, "request payload")
    source_review_json = payload["source_review_json"]
    if not isinstance(source_review_json, str):
        _refuse(
            MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "source_review_json must be text",
        )
    request = MastPhaseQualificationRequest(source_review_json=source_review_json)
    if mast_phase_qualification_request_to_record(request) != payload:
        _refuse(
            MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "stored request differs from reconstructed source and registry",
        )
    return request


def mast_phase_qualification_request_to_bytes(
    request: MastPhaseQualificationRequest,
) -> bytes:
    """Serialize a request as unique canonical digest-sealed JSON bytes.

    Parameters
    ----------
    request : MastPhaseQualificationRequest
        Validated qualification request.

    Returns
    -------
    bytes
        Canonical UTF-8 envelope with one trailing newline.
    """
    payload = mast_phase_qualification_request_to_record(request)
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": MAST_PHASE_QUALIFICATION_REQUEST_SCHEMA,
            "schema_version": MAST_PHASE_QUALIFICATION_REQUEST_VERSION,
        }
    )


def mast_phase_qualification_request_from_bytes(
    data: bytes,
) -> MastPhaseQualificationRequest:
    """Decode canonical request bytes and replay every derived field.

    Parameters
    ----------
    data : bytes
        Candidate canonical request envelope.

    Returns
    -------
    MastPhaseQualificationRequest
        Fully reconstructed request.
    """
    document = _decode_document(data)
    if (
        document["schema"] != MAST_PHASE_QUALIFICATION_REQUEST_SCHEMA
        or document["schema_version"] != MAST_PHASE_QUALIFICATION_REQUEST_VERSION
    ):
        _refuse(
            MastPhaseQualificationRequestRefusalCode.UNSUPPORTED_SCHEMA,
            "unsupported request schema or version",
        )
    payload = _object(document["payload"], _PAYLOAD_KEYS, "request payload")
    expected_digest = _sha256(_canonical(payload))
    if document["payload_sha256"] != expected_digest:
        _refuse(
            MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            "request payload digest mismatch",
        )
    return mast_phase_qualification_request_from_record(payload)


def mast_phase_qualification_request_digest(
    request: MastPhaseQualificationRequest,
) -> str:
    """Return the SHA-256 of the complete canonical request envelope.

    Parameters
    ----------
    request : MastPhaseQualificationRequest
        Validated qualification request.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """
    return _sha256(mast_phase_qualification_request_to_bytes(request))


def _candidate_requirement(
    profile: ReactorSignalCandidateProfile,
) -> MastPhaseCandidateRequirement:
    """Build a physical-evidence requirement from one diagnostic profile."""
    return MastPhaseCandidateRequirement(
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


def _candidate_record(item: MastPhaseCandidateRequirement) -> dict[str, object]:
    """Serialize one candidate requirement into its canonical record."""
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
    item: MastPhaseQualificationRequirement,
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
    if (
        not isinstance(data, bytes)
        or not data
        or len(data) > MAX_MAST_PHASE_QUALIFICATION_REQUEST_BYTES
    ):
        _refuse(
            MastPhaseQualificationRequestRefusalCode.INVALID_INPUT,
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
            MastPhaseQualificationRequestRefusalCode.INVALID_JSON,
            f"request JSON invalid: {exc}",
        )
    document = _object(value, _OUTER_KEYS, "request document")
    if _canonical(document) != data:
        _refuse(
            MastPhaseQualificationRequestRefusalCode.NONCANONICAL_BYTES,
            "request is not unique canonical JSON",
        )
    return document


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    """Require an object with exactly the expected keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _refuse(
            MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} must be an object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH,
            f"{name} keys differ from contract",
        )
    return result


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys while decoding JSON."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _refuse(
                MastPhaseQualificationRequestRefusalCode.DUPLICATE_JSON_KEY,
                f"duplicate key {key}",
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    """Reject non-finite numeric constants while decoding JSON."""
    _refuse(
        MastPhaseQualificationRequestRefusalCode.INVALID_JSON,
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


def _refuse(code: MastPhaseQualificationRequestRefusalCode, detail: str) -> NoReturn:
    """Raise the typed refusal for this contract."""
    raise MastPhaseQualificationRequestRefusalError(code, detail)


__all__ = [
    "MAST_PHASE_QUALIFICATION_REQUEST_SCHEMA",
    "MAST_PHASE_QUALIFICATION_REQUEST_VERSION",
    "MAX_MAST_PHASE_QUALIFICATION_REQUEST_BYTES",
    "MastPhaseCandidateRequirement",
    "MastPhaseQualificationRequest",
    "MastPhaseQualificationRequestRefusalCode",
    "MastPhaseQualificationRequestRefusalError",
    "MastPhaseQualificationRequirement",
    "MastPhaseQualificationRequirementId",
    "mast_phase_qualification_request_digest",
    "mast_phase_qualification_request_from_bytes",
    "mast_phase_qualification_request_from_record",
    "mast_phase_qualification_request_from_source_review",
    "mast_phase_qualification_request_to_bytes",
    "mast_phase_qualification_request_to_record",
]
