# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Device diagnostic-plan design review
"""Fail-closed review of portable device diagnostic-plan declarations.

The module consumes producer-owned bytes without importing a device package.
Acceptance means only that a synthetic design declaration is internally
consistent with the installed SPO registries. It creates no observation,
semantic ingress, classifier result, control intent, or actuator authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, TypeGuard

from .diagnostic_plan_depth import (
    DiagnosticPlanDepthError,
    DiagnosticPlanDepthRefusalKind,
    validate_diagnostic_plan_depth,
)
from .observability_profiles import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    ObservabilityClass,
    ReactorSignalCandidateProfile,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
from .vocabulary import ClockKind, SemanticCarrier

DEVICE_DIAGNOSTIC_PLAN_REVIEW_SCHEMA: Final = (
    "scpn-phase-orchestrator.device-diagnostic-plan-review.v1"
)
DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION: Final = "1.0.0"
MAX_DEVICE_DIAGNOSTIC_SOURCE_BYTES: Final = 2 * 1024 * 1024
MAX_DEVICE_DIAGNOSTIC_PLAN_REVIEW_BYTES: Final = 8 * 1024 * 1024

_ENVELOPE_SCHEMA: Final = "scpn.reactor-diagnostic-plan-envelope.v1"
_ENVELOPE_VERSION_1_1: Final = "1.1.0"
_ENVELOPE_VERSION_1_2: Final = "1.2.0"
_ENVELOPE_VERSIONS: Final = frozenset({_ENVELOPE_VERSION_1_1, _ENVELOPE_VERSION_1_2})
_MANIFEST_SCHEMA: Final = "scpn.reactor-domain.v1"
_MANIFEST_VERSION: Final = "1.0.0"
_CAPABILITY: Final = "diagnostic_clock_semantics"
_MATURITY: Final = "computational_prototype"
_NON_CLAIMS: Final = (
    "no control action is proposed or authorised",
    "no physical observation is described or claimed",
)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.]*$")

_BINDING_KEYS = {"catalogue_digest_sha256", "catalogue_version"} | {
    "reactor_registry_digest_sha256",
    "reactor_registry_version",
}
_ENVELOPE_KEYS = (
    {"actionable", "authority", "binding", "capability"}
    | {"configurations", "evidence_maturity", "manifest_sha256", "non_claims"}
    | {"plan_identifier", "plan_sha256", "producer_revision", "project"}
    | {"schema", "schema_version", "synthetic"}
)
_PLAN_KEYS_1_1 = {
    "binding",
    "channels",
    "clock_relations",
    "clocks",
    "deferrals",
    "frames",
    "identifier",
}
_PLAN_KEYS_1_2 = _PLAN_KEYS_1_1 | {"clock_topology", "frame_transformations"}
_PLAN_KEYS_BY_VERSION = {
    _ENVELOPE_VERSION_1_1: _PLAN_KEYS_1_1,
    _ENVELOPE_VERSION_1_2: _PLAN_KEYS_1_2,
}
_CLOCK_KEYS = {"epoch", "identifier", "kind", "resolution_s", "uncertainty_s"}
_CHANNEL_KEYS_1_1 = (
    {"acquisition_duration_s", "acquisition_start_s", "candidate_id"}
    | {"carrier", "clock_identifier", "element_count", "evidence_bindings"}
    | {"identifier", "max_signal_frequency_hz", "sample_rate_hz"}
    | {"synthetic", "timing_uncertainty_s"}
)
_CHANNEL_KEYS_1_2 = _CHANNEL_KEYS_1_1 | {"signals"}
_CHANNEL_KEYS_BY_VERSION = {
    _ENVELOPE_VERSION_1_1: _CHANNEL_KEYS_1_1,
    _ENVELOPE_VERSION_1_2: _CHANNEL_KEYS_1_2,
}
_FRAME_KEYS = {"description", "identifier", "kind"}
_DEFERRAL_KEYS = {"candidate_id", "reason"}
_RELATION_KEYS = {
    "child_identifier",
    "evidence_claimed",
    "mapping_state",
    "max_offset_s",
} | {"method", "parent_identifier", "uncertainty_s"}
_MANIFEST_KEYS = (
    {"capabilities", "claims", "configurations", "confinement_family"}
    | {"control_adapter", "device_family", "evidence_maturity", "excluded_domains"}
    | {"fusion_solver_seams", "license", "machine_protection", "non_claims"}
    | {"owned_domains", "project", "research_group", "schema", "schema_version"}
    | {"spo_registry", "spo_semantic_profile", "studio_integration"}
)


class DeviceDiagnosticPlanRefusalCode(StrEnum):
    """Stable reason categories for fail-closed intake refusal."""

    INVALID_INPUT_TYPE = "invalid_input_type"
    INVALID_INPUT_SIZE = "invalid_input_size"
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NONCANONICAL_SOURCE_BYTES = "noncanonical_source_bytes"
    UNSUPPORTED_SOURCE_SCHEMA = "unsupported_source_schema"
    INVALID_SOURCE_IDENTITY = "invalid_source_identity"
    SOURCE_DIGEST_MISMATCH = "source_digest_mismatch"
    MANIFEST_CONTRACT_MISMATCH = "manifest_contract_mismatch"
    PROJECT_ASSIGNMENT_MISMATCH = "project_assignment_mismatch"
    REGISTRY_BINDING_MISMATCH = "registry_binding_mismatch"
    PLAN_STRUCTURE_MISMATCH = "plan_structure_mismatch"
    CANDIDATE_COVERAGE_MISMATCH = "candidate_coverage_mismatch"
    CARRIER_EVIDENCE_MISMATCH = "carrier_evidence_mismatch"
    CLOCK_SEMANTICS_MISMATCH = "clock_semantics_mismatch"
    AUTHORITY_ESCALATION = "authority_escalation"


class DeviceDiagnosticPlanRefusalError(ValueError):
    """Raised when producer bytes cannot form a safe design review."""

    def __init__(self, code: DeviceDiagnosticPlanRefusalCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


DeviceDiagnosticPlanRefusal = DeviceDiagnosticPlanRefusalError


class DiagnosticClockCompatibility(StrEnum):
    """Relationship between a producer clock and the SPO clock vocabulary."""

    SYNTHETIC_COMPATIBLE = "synthetic_compatible"
    EVENT_RELATIVE_COMPATIBLE = "event_relative_compatible"
    UNMAPPED = "unmapped"


@dataclass(frozen=True, slots=True)
class DeviceDiagnosticClockReview:
    """One reviewed producer clock without a physical mapping claim."""

    plan_clock_identifier: str
    plan_clock_kind: str
    epoch: str
    resolution_s: float
    uncertainty_s: float
    spo_clock_kind_candidate: ClockKind | None
    compatibility: DiagnosticClockCompatibility
    mapping_evidence_claimed: bool = False

    def __post_init__(self) -> None:
        if self.mapping_evidence_claimed is not False:
            _refuse(
                DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
                "clock review cannot claim mapping evidence",
            )


@dataclass(frozen=True, slots=True)
class DeviceDiagnosticSignalReview:
    """One typed synthetic signal declaration accepted for design review."""

    channel_identifier: str
    candidate_id: str
    observability_class: ObservabilityClass
    carrier: SemanticCarrier
    clock_identifier: str
    evidence_slots: tuple[str, ...]
    synthetic: bool = True
    evidence_claimed: bool = False
    observation_claimed: bool = False

    def __post_init__(self) -> None:
        if (
            self.synthetic is not True
            or self.evidence_claimed is not False
            or self.observation_claimed is not False
        ):
            _authority_refusal("signal review cannot claim physical evidence")


@dataclass(frozen=True, slots=True)
class _ValidatedSources:
    project: str
    producer_revision: str
    configurations: tuple[str, ...]
    capability: str
    maturity: str
    envelope_schema: str
    envelope_version: str
    plan_identifier: str
    manifest_sha256: str
    envelope_sha256: str
    plan_sha256: str
    planned: tuple[str, ...]
    deferred: tuple[str, ...]
    frames: tuple[str, ...]
    clocks: tuple[DeviceDiagnosticClockReview, ...]
    signals: tuple[DeviceDiagnosticSignalReview, ...]


@dataclass(frozen=True, slots=True)
class DeviceDiagnosticPlanReview:
    """Portable review of one exact producer declaration set."""

    source_revision: str
    source_artifact_sha256: str
    source_manifest_json: str
    source_envelope_json: str
    source_plan_json: str
    review_id: str = field(init=False)
    source_project: str = field(init=False)
    producer_package_revision: str = field(init=False)
    configurations: tuple[str, ...] = field(init=False)
    capability: str = field(init=False)
    evidence_maturity: str = field(init=False)
    source_envelope_schema: str = field(init=False)
    source_envelope_schema_version: str = field(init=False)
    plan_identifier: str = field(init=False)
    source_manifest_sha256: str = field(init=False)
    source_envelope_sha256: str = field(init=False)
    source_plan_sha256: str = field(init=False)
    planned_candidate_ids: tuple[str, ...] = field(init=False)
    deferred_candidate_ids: tuple[str, ...] = field(init=False)
    frame_ids: tuple[str, ...] = field(init=False)
    clock_reviews: tuple[DeviceDiagnosticClockReview, ...] = field(init=False)
    signal_reviews: tuple[DeviceDiagnosticSignalReview, ...] = field(init=False)
    reactor_registry_version: str = field(init=False)
    reactor_registry_digest: str = field(init=False)
    observability_registry_version: str = field(init=False)
    observability_registry_digest: str = field(init=False)
    semantic_profile_registry_version: str = field(init=False)
    semantic_profile_registry_digest: str = field(init=False)
    assignment_map_sha256: str = field(init=False)
    accepted_as_design_declaration: bool = field(init=False, default=True)
    evidence_claimed: bool = field(init=False, default=False)
    observation_claimed: bool = field(init=False, default=False)
    measurement_claimed: bool = field(init=False, default=False)
    facility_binding_claimed: bool = field(init=False, default=False)
    classification_performed: bool = field(init=False, default=False)
    semantic_ingress_declared: bool = field(init=False, default=False)
    control_intent_created: bool = field(init=False, default=False)
    authority: str = field(init=False, default="review_only")
    actionable: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        _require_source_identity(self.source_revision, self.source_artifact_sha256)
        validated = _validate_sources(
            self.source_manifest_json.encode("utf-8"),
            self.source_envelope_json.encode("utf-8"),
            self.source_plan_json.encode("utf-8"),
        )
        values: dict[str, object] = {
            "source_project": validated.project,
            "producer_package_revision": validated.producer_revision,
            "configurations": validated.configurations,
            "capability": validated.capability,
            "evidence_maturity": validated.maturity,
            "source_envelope_schema": validated.envelope_schema,
            "source_envelope_schema_version": validated.envelope_version,
            "plan_identifier": validated.plan_identifier,
            "source_manifest_sha256": validated.manifest_sha256,
            "source_envelope_sha256": validated.envelope_sha256,
            "source_plan_sha256": validated.plan_sha256,
            "planned_candidate_ids": validated.planned,
            "deferred_candidate_ids": validated.deferred,
            "frame_ids": validated.frames,
            "clock_reviews": validated.clocks,
            "signal_reviews": validated.signals,
            "reactor_registry_version": DEFAULT_REACTOR_REGISTRY.version,
            "reactor_registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
            "observability_registry_version": (
                DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version
            ),
            "observability_registry_digest": (
                DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
            ),
            "semantic_profile_registry_version": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version
            ),
            "semantic_profile_registry_digest": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
            ),
            "assignment_map_sha256": (
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.assignment_map_sha256
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        identity = {
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_envelope_sha256": validated.envelope_sha256,
            "source_manifest_sha256": validated.manifest_sha256,
            "source_plan_sha256": validated.plan_sha256,
            "source_project": validated.project,
            "source_revision": self.source_revision,
        }
        object.__setattr__(self, "review_id", _sha256(_canonical(identity)))


def device_diagnostic_plan_review_from_producer_bytes(
    *,
    source_revision: str,
    source_artifact_sha256: str,
    manifest_bytes: bytes,
    envelope_bytes: bytes,
    plan_bytes: bytes,
) -> DeviceDiagnosticPlanReview:
    """Review exact producer bytes against the installed SPO registries.

    Parameters
    ----------
    source_revision, source_artifact_sha256 : str
        Exact producer Git revision and installed artefact SHA-256.
    manifest_bytes, envelope_bytes, plan_bytes : bytes
        Exact canonical producer documents.

    Returns
    -------
    DeviceDiagnosticPlanReview
        A design-only, non-evidence, non-actuating portable review.

    Raises
    ------
    DeviceDiagnosticPlanRefusal
        If identity, bytes, registry bindings, or semantics do not match.
    """
    _require_source_identity(source_revision, source_artifact_sha256)
    _decode_source(manifest_bytes, "manifest", pretty=True)
    _decode_source(envelope_bytes, "envelope", pretty=False)
    _decode_source(plan_bytes, "plan", pretty=False)
    return DeviceDiagnosticPlanReview(
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        source_manifest_json=manifest_bytes.decode("utf-8"),
        source_envelope_json=envelope_bytes.decode("utf-8"),
        source_plan_json=plan_bytes.decode("utf-8"),
    )


def device_diagnostic_plan_review_to_record(
    review: DeviceDiagnosticPlanReview,
) -> dict[str, object]:
    """Return the complete deterministic review payload record.

    Parameters
    ----------
    review : DeviceDiagnosticPlanReview
        Validated design review.

    Returns
    -------
    dict[str, object]
        Complete deterministic payload.
    """
    return {
        "accepted_as_design_declaration": review.accepted_as_design_declaration,
        "actionable": review.actionable,
        "assignment_map_sha256": review.assignment_map_sha256,
        "authority": review.authority,
        "capability": review.capability,
        "classification_performed": review.classification_performed,
        "clock_reviews": [_clock_record(item) for item in review.clock_reviews],
        "configurations": list(review.configurations),
        "control_intent_created": review.control_intent_created,
        "deferred_candidate_ids": list(review.deferred_candidate_ids),
        "evidence_claimed": review.evidence_claimed,
        "evidence_maturity": review.evidence_maturity,
        "facility_binding_claimed": review.facility_binding_claimed,
        "frame_ids": list(review.frame_ids),
        "measurement_claimed": review.measurement_claimed,
        "observation_claimed": review.observation_claimed,
        "observability_registry_digest": review.observability_registry_digest,
        "observability_registry_version": review.observability_registry_version,
        "plan_identifier": review.plan_identifier,
        "planned_candidate_ids": list(review.planned_candidate_ids),
        "producer_package_revision": review.producer_package_revision,
        "reactor_registry_digest": review.reactor_registry_digest,
        "reactor_registry_version": review.reactor_registry_version,
        "review_id": review.review_id,
        "semantic_ingress_declared": review.semantic_ingress_declared,
        "semantic_profile_registry_digest": review.semantic_profile_registry_digest,
        "semantic_profile_registry_version": review.semantic_profile_registry_version,
        "signal_reviews": [_signal_record(item) for item in review.signal_reviews],
        "source_artifact_sha256": review.source_artifact_sha256,
        "source_envelope_json": review.source_envelope_json,
        "source_envelope_schema": review.source_envelope_schema,
        "source_envelope_schema_version": review.source_envelope_schema_version,
        "source_envelope_sha256": review.source_envelope_sha256,
        "source_manifest_json": review.source_manifest_json,
        "source_manifest_sha256": review.source_manifest_sha256,
        "source_plan_json": review.source_plan_json,
        "source_plan_sha256": review.source_plan_sha256,
        "source_project": review.source_project,
        "source_revision": review.source_revision,
    }


_REVIEW_KEYS = (
    {"accepted_as_design_declaration", "actionable", "assignment_map_sha256"}
    | {"authority", "capability", "classification_performed", "clock_reviews"}
    | {"configurations", "control_intent_created", "deferred_candidate_ids"}
    | {"evidence_claimed", "evidence_maturity", "facility_binding_claimed"}
    | {"frame_ids", "measurement_claimed", "observation_claimed"}
    | {"observability_registry_digest", "observability_registry_version"}
    | {"plan_identifier", "planned_candidate_ids", "producer_package_revision"}
    | {"reactor_registry_digest", "reactor_registry_version", "review_id"}
    | {"semantic_ingress_declared", "semantic_profile_registry_digest"}
    | {"semantic_profile_registry_version", "signal_reviews"}
    | {"source_artifact_sha256", "source_envelope_json", "source_envelope_schema"}
    | {"source_envelope_schema_version", "source_envelope_sha256"}
    | {"source_manifest_json", "source_manifest_sha256", "source_plan_json"}
    | {"source_plan_sha256", "source_project", "source_revision"}
)


def device_diagnostic_plan_review_from_record(
    record: object,
) -> DeviceDiagnosticPlanReview:
    """Rebuild a review record and revalidate all embedded source bytes.

    Parameters
    ----------
    record : object
        Candidate payload record.

    Returns
    -------
    DeviceDiagnosticPlanReview
        Reconstructed, revalidated review.
    """
    payload = _object(record, _REVIEW_KEYS, "review payload")
    review = DeviceDiagnosticPlanReview(
        source_revision=_text(payload, "source_revision"),
        source_artifact_sha256=_text(payload, "source_artifact_sha256"),
        source_manifest_json=_text(payload, "source_manifest_json"),
        source_envelope_json=_text(payload, "source_envelope_json"),
        source_plan_json=_text(payload, "source_plan_json"),
    )
    if device_diagnostic_plan_review_to_record(review) != payload:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH,
            "stored review fields do not match reconstructed sources",
        )
    return review


def device_diagnostic_plan_review_to_bytes(
    review: DeviceDiagnosticPlanReview,
) -> bytes:
    """Serialize a review in a digest-sealed canonical envelope.

    Parameters
    ----------
    review : DeviceDiagnosticPlanReview
        Validated design review.

    Returns
    -------
    bytes
        Canonical digest-sealed envelope.
    """
    payload = device_diagnostic_plan_review_to_record(review)
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": DEVICE_DIAGNOSTIC_PLAN_REVIEW_SCHEMA,
            "schema_version": DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION,
        }
    )


def device_diagnostic_plan_review_from_bytes(data: bytes) -> DeviceDiagnosticPlanReview:
    """Decode a canonical review envelope and verify its custody chain.

    Parameters
    ----------
    data : bytes
        Canonical digest-sealed envelope.

    Returns
    -------
    DeviceDiagnosticPlanReview
        Reconstructed, revalidated review.
    """
    record = _decode_json(
        data, "review", maximum=MAX_DEVICE_DIAGNOSTIC_PLAN_REVIEW_BYTES
    )
    if _canonical(record) != data:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.NONCANONICAL_SOURCE_BYTES,
            "review is not canonical JSON",
        )
    outer = _object(
        record,
        {"payload", "payload_sha256", "schema", "schema_version"},
        "review envelope",
    )
    if (
        outer["schema"] != DEVICE_DIAGNOSTIC_PLAN_REVIEW_SCHEMA
        or outer["schema_version"] != DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA,
            "unsupported review schema or version",
        )
    digest = _text(outer, "payload_sha256")
    if digest != _sha256(_canonical(outer["payload"])):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH,
            "review payload digest mismatch",
        )
    return device_diagnostic_plan_review_from_record(outer["payload"])


def device_diagnostic_plan_review_digest(review: DeviceDiagnosticPlanReview) -> str:
    """Return SHA-256 of the complete canonical review envelope.

    Parameters
    ----------
    review : DeviceDiagnosticPlanReview
        Validated design review.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    return _sha256(device_diagnostic_plan_review_to_bytes(review))


def _validate_sources(
    manifest_bytes: bytes, envelope_bytes: bytes, plan_bytes: bytes
) -> _ValidatedSources:
    manifest = _validate_manifest(
        _decode_source(manifest_bytes, "manifest", pretty=True)
    )
    envelope = _validate_envelope(
        _decode_source(envelope_bytes, "envelope", pretty=False)
    )
    envelope_version = _text(envelope, "schema_version")
    plan = _object(
        _decode_source(plan_bytes, "plan", pretty=False),
        _PLAN_KEYS_BY_VERSION[envelope_version],
        "source plan",
    )
    manifest_digest = _sha256(manifest_bytes)
    envelope_digest = _sha256(envelope_bytes)
    plan_digest = _sha256(plan_bytes)
    if (
        envelope["manifest_sha256"] != manifest_digest
        or envelope["plan_sha256"] != plan_digest
        or envelope["plan_identifier"] != plan["identifier"]
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH,
            "envelope does not pin the supplied manifest and plan",
        )
    project = _text(envelope, "project")
    configurations = _strings(envelope, "configurations")
    _validate_assignment(project, configurations)
    _validate_manifest_alignment(manifest, envelope, configurations)
    _validate_binding(envelope["binding"], "envelope binding")
    _validate_binding(plan["binding"], "plan binding")
    planned, deferred, frames, clocks, signals = _validate_plan(
        plan, configurations, envelope_version=envelope_version
    )
    return _ValidatedSources(
        project=project,
        producer_revision=_text(envelope, "producer_revision"),
        configurations=configurations,
        capability=_text(envelope, "capability"),
        maturity=_text(envelope, "evidence_maturity"),
        envelope_schema=_text(envelope, "schema"),
        envelope_version=envelope_version,
        plan_identifier=_text(plan, "identifier"),
        manifest_sha256=manifest_digest,
        envelope_sha256=envelope_digest,
        plan_sha256=plan_digest,
        planned=planned,
        deferred=deferred,
        frames=frames,
        clocks=clocks,
        signals=signals,
    )


def _validate_envelope(value: object) -> dict[str, object]:
    envelope = _object(value, _ENVELOPE_KEYS, "source envelope")
    schema_version = envelope["schema_version"]
    if (
        envelope["schema"] != _ENVELOPE_SCHEMA
        or not isinstance(schema_version, str)
        or schema_version not in _ENVELOPE_VERSIONS
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA,
            "unsupported diagnostic-plan envelope",
        )
    if (
        envelope["capability"] != _CAPABILITY
        or envelope["evidence_maturity"] != _MATURITY
    ):
        _manifest_refusal("unsupported capability or evidence maturity")
    if (
        envelope["synthetic"] is not True
        or envelope["authority"] != "review_only"
        or envelope["actionable"] is not False
        or _strings(envelope, "non_claims") != _NON_CLAIMS
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
            "envelope must remain synthetic, review-only, and non-actuating",
        )
    for name in ("manifest_sha256", "plan_sha256"):
        if _SHA256.fullmatch(_text(envelope, name)) is None:
            _identity_refusal(f"{name} is not a SHA-256 digest")
    if not _text(envelope, "producer_revision"):
        _identity_refusal("producer package revision is empty")
    return envelope


def _validate_manifest(value: object) -> dict[str, object]:
    manifest = _object(value, _MANIFEST_KEYS, "source manifest")
    if (
        manifest["schema"] != _MANIFEST_SCHEMA
        or manifest["schema_version"] != _MANIFEST_VERSION
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA,
            "unsupported reactor-domain manifest",
        )
    if manifest["claims"] != []:
        _authority_refusal("reactor-domain manifest contains claims")
    control = _object(
        manifest["control_adapter"],
        {
            "contract_version",
            "direct_actuation",
            "identifier",
            "implementation",
            "specification",
        },
        "control_adapter",
    )
    semantic = _object(
        manifest["spo_semantic_profile"],
        {"actionable", "control_intent_contract", "mode"},
        "spo_semantic_profile",
    )
    protection = _object(
        manifest["machine_protection"],
        {"final_veto", "statement"},
        "machine_protection",
    )
    if (
        control["direct_actuation"] is not False
        or semantic["actionable"] is not False
        or semantic["control_intent_contract"] is not None
        or semantic["mode"] != "review_only"
        or protection["final_veto"] != "independent"
    ):
        _authority_refusal("manifest weakens control or protection boundaries")
    exclusions = _object_array(manifest, "excluded_domains", {"domain", "owner"})
    actual = {(item["domain"], item["owner"]) for item in exclusions}
    required = {
        ("typed_signal_semantics_and_comparability", "SCPN-PHASE-ORCHESTRATOR"),
        ("control_admission_and_action_formation", "SCPN-CONTROL"),
        ("machine_protection_final_veto", "independent_machine_protection"),
    }
    if not required <= actual:
        _manifest_refusal("required ownership exclusions are missing")
    return manifest


def _validate_manifest_alignment(
    manifest: dict[str, object],
    envelope: dict[str, object],
    configurations: tuple[str, ...],
) -> None:
    if (
        manifest["project"] != envelope["project"]
        or _strings(manifest, "configurations") != configurations
        or manifest["evidence_maturity"] != envelope["evidence_maturity"]
    ):
        _manifest_refusal("manifest and envelope identities differ")
    capabilities = _object_array(
        manifest,
        "capabilities",
        {"evidence_maturity", "evidence_pointer", "identifier"},
    )
    matching = [
        item for item in capabilities if item["identifier"] == envelope["capability"]
    ]
    if (
        len(matching) != 1
        or matching[0]["evidence_maturity"] != envelope["evidence_maturity"]
    ):
        _manifest_refusal("diagnostic capability is absent or inconsistent")
    pin = _object(
        manifest["spo_registry"],
        {"digest_sha256", "source_path", "version"},
        "spo_registry",
    )
    if (
        pin["version"] != DEFAULT_REACTOR_REGISTRY.version
        or pin["digest_sha256"] != DEFAULT_REACTOR_REGISTRY.digest
    ):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.REGISTRY_BINDING_MISMATCH,
            "manifest registry pin differs from installed SPO",
        )


def _validate_assignment(project: str, configurations: tuple[str, ...]) -> None:
    _sorted_identifiers(configurations, "configurations")
    if not configurations:
        _assignment_refusal("configurations are empty")
    try:
        profiles = tuple(
            DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(item)
            for item in configurations
        )
    except (KeyError, ValueError) as exc:
        _assignment_refusal(f"unregistered configuration: {exc}")
    if any(profile.device_project != project for profile in profiles):
        _assignment_refusal("project does not own every declared configuration")


def _validate_binding(value: object, name: str) -> None:
    binding = _object(value, _BINDING_KEYS, name)
    expected = {
        "catalogue_digest_sha256": (
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
        ),
        "catalogue_version": DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version,
        "reactor_registry_digest_sha256": DEFAULT_REACTOR_REGISTRY.digest,
        "reactor_registry_version": DEFAULT_REACTOR_REGISTRY.version,
    }
    if binding != expected:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.REGISTRY_BINDING_MISMATCH,
            f"{name} differs from installed SPO",
        )


def _validate_plan(
    plan: dict[str, object],
    configurations: tuple[str, ...],
    *,
    envelope_version: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[DeviceDiagnosticClockReview, ...],
    tuple[DeviceDiagnosticSignalReview, ...],
]:
    if _IDENTIFIER.fullmatch(_text(plan, "identifier")) is None:
        _plan_refusal("plan identifier is malformed")
    clocks = _validate_clocks(_object_array(plan, "clocks", _CLOCK_KEYS))
    clock_index = {item.plan_clock_identifier: item for item in clocks}
    frames = _object_array(plan, "frames", _FRAME_KEYS)
    frame_ids = tuple(_text(item, "identifier") for item in frames)
    _sorted_identifiers(frame_ids, "frames")
    if any(
        not _text(item, "kind") or not _text(item, "description") for item in frames
    ):
        _plan_refusal("frame kind and description must be non-empty")
    profiles = {
        profile.candidate_id: profile
        for configuration in configurations
        for profile in DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.for_configuration(
            configuration
        )
    }
    channels = _object_array(
        plan, "channels", _CHANNEL_KEYS_BY_VERSION[envelope_version]
    )
    channel_ids = tuple(_text(item, "identifier") for item in channels)
    _sorted_identifiers(channel_ids, "channels")
    planned_ids: list[str] = []
    signal_reviews: list[DeviceDiagnosticSignalReview] = []
    for channel in channels:
        candidate_id = _text(channel, "candidate_id")
        profile = profiles.get(candidate_id)
        if profile is None:
            _candidate_refusal(f"inapplicable candidate {candidate_id!r}")
        planned_ids.append(candidate_id)
        signal_reviews.append(
            _validate_channel(channel, profile, clock_index, set(frame_ids))
        )
    deferrals = _object_array(plan, "deferrals", _DEFERRAL_KEYS)
    deferred = tuple(_text(item, "candidate_id") for item in deferrals)
    _sorted_identifiers(deferred, "deferrals")
    if any(not _text(item, "reason") for item in deferrals):
        _candidate_refusal("deferral reason must be non-empty")
    planned = tuple(sorted(set(planned_ids)))
    if set(planned) & set(deferred):
        _candidate_refusal("candidate is both planned and deferred")
    if set(planned) | set(deferred) != set(profiles):
        _candidate_refusal("candidate coverage differs from the SPO catalogue")
    relation_records = _object_array(plan, "clock_relations", _RELATION_KEYS)
    _validate_relations(relation_records, clock_index)
    if envelope_version == _ENVELOPE_VERSION_1_2:
        try:
            validate_diagnostic_plan_depth(
                plan,
                candidate_classes={
                    identifier: profile.observability_class.value
                    for identifier, profile in profiles.items()
                },
                clock_kinds={
                    identifier: review.plan_clock_kind
                    for identifier, review in clock_index.items()
                },
                frame_kinds={
                    _text(frame, "identifier"): _text(frame, "kind") for frame in frames
                },
            )
        except DiagnosticPlanDepthError as exc:
            _depth_refusal(exc)
    return planned, deferred, frame_ids, clocks, tuple(signal_reviews)


def _validate_clocks(
    records: list[dict[str, object]],
) -> tuple[DeviceDiagnosticClockReview, ...]:
    mapping = {
        "simulation": (
            ClockKind.SIMULATION_MONOTONIC,
            DiagnosticClockCompatibility.SYNTHETIC_COMPATIBLE,
        ),
        "shot_event_epoch": (
            ClockKind.SHOT_RELATIVE,
            DiagnosticClockCompatibility.EVENT_RELATIVE_COMPATIBLE,
        ),
        "facility_monotonic": (None, DiagnosticClockCompatibility.UNMAPPED),
    }
    reviews: list[DeviceDiagnosticClockReview] = []
    for record in records:
        kind = _text(record, "kind")
        try:
            spo_kind, compatibility = mapping[kind]
        except KeyError:
            _clock_refusal(f"unsupported producer clock kind {kind!r}")
        epoch = _text(record, "epoch")
        if not epoch:
            _clock_refusal("clock epoch is empty")
        reviews.append(
            DeviceDiagnosticClockReview(
                plan_clock_identifier=_text(record, "identifier"),
                plan_clock_kind=kind,
                epoch=epoch,
                resolution_s=_number(record, "resolution_s", positive=True),
                uncertainty_s=_number(record, "uncertainty_s", nonnegative=True),
                spo_clock_kind_candidate=spo_kind,
                compatibility=compatibility,
            )
        )
    _sorted_identifiers(tuple(item.plan_clock_identifier for item in reviews), "clocks")
    return tuple(reviews)


def _validate_channel(
    channel: dict[str, object],
    profile: ReactorSignalCandidateProfile,
    clocks: dict[str, DeviceDiagnosticClockReview],
    frame_ids: set[str],
) -> DeviceDiagnosticSignalReview:
    try:
        carrier = SemanticCarrier(_text(channel, "carrier"))
    except ValueError:
        _carrier_refusal("unknown semantic carrier")
    if carrier not in profile.admissible_carriers:
        _carrier_refusal("carrier is inadmissible for the candidate")
    evidence = channel["evidence_bindings"]
    if not isinstance(evidence, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in evidence.items()
    ):
        _carrier_refusal("evidence bindings must be non-empty string pairs")
    if set(evidence) != set(profile.required_evidence):
        _carrier_refusal("evidence slots differ from the SPO catalogue")
    clock_id = _text(channel, "clock_identifier")
    try:
        clock = clocks[clock_id]
    except KeyError:
        _clock_refusal("channel references an undeclared clock")
    class_name = profile.observability_class.value
    compatible = {
        "direct_cyclic": {"facility_monotonic"},
        "derived_cyclic": {"facility_monotonic"},
        "event_relative": {"shot_event_epoch"},
        "noncyclic_feature": {"facility_monotonic", "shot_event_epoch"},
        "numerical_only": {"simulation"},
    }
    if clock.plan_clock_kind not in compatible.get(class_name, set()):
        _clock_refusal("clock kind is incompatible with observability class")
    clock_slot = "simulation_clock" if class_name == "numerical_only" else "clock_epoch"
    if evidence.get(clock_slot) != clock_id:
        _clock_refusal("evidence does not bind the channel clock")
    sample_rate = _number(channel, "sample_rate_hz", positive=True)
    maximum_frequency = _number(channel, "max_signal_frequency_hz", nonnegative=True)
    _number(channel, "acquisition_start_s")
    _number(channel, "acquisition_duration_s", positive=True)
    count = channel["element_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        _plan_refusal("element_count must be a positive integer")
    if channel["synthetic"] is not True:
        _authority_refusal("diagnostic-plan channels must remain synthetic")
    timing = channel["timing_uncertainty_s"]
    if class_name == "derived_cyclic" and (
        maximum_frequency <= 0.0 or sample_rate < 2.0 * maximum_frequency
    ):
        _carrier_refusal("cyclic channel violates positive-band or Nyquist rules")
    if class_name == "event_relative":
        if not _is_number(timing) or float(timing) <= 0.0:
            _clock_refusal("event-relative timing uncertainty is invalid")
        if clock.resolution_s > float(timing):
            _clock_refusal("clock resolution exceeds timing uncertainty")
    elif timing is not None:
        _clock_refusal("only event-relative channels declare timing uncertainty")
    frame = evidence.get("coordinate_frame")
    if frame is not None and frame not in frame_ids:
        _carrier_refusal("coordinate-frame evidence names an undeclared frame")
    return DeviceDiagnosticSignalReview(
        channel_identifier=_text(channel, "identifier"),
        candidate_id=profile.candidate_id,
        observability_class=profile.observability_class,
        carrier=carrier,
        clock_identifier=clock_id,
        evidence_slots=tuple(sorted(evidence)),
    )


def _validate_relations(
    records: list[dict[str, object]],
    clocks: dict[str, DeviceDiagnosticClockReview],
) -> None:
    keys: list[tuple[str, str]] = []
    related_children: set[str] = set()
    for relation in records:
        child = _text(relation, "child_identifier")
        parent = _text(relation, "parent_identifier")
        keys.append((child, parent))
        if child == parent or child not in clocks or parent not in clocks:
            _clock_refusal("clock relation endpoints are invalid")
        if "simulation" in {
            clocks[child].plan_clock_kind,
            clocks[parent].plan_clock_kind,
        }:
            _clock_refusal("simulation clock cannot join a physical relation")
        if (
            relation["mapping_state"] != "unmapped"
            or relation["evidence_claimed"] is not False
        ):
            _authority_refusal("clock relation claims mapping evidence")
        _number(relation, "max_offset_s", nonnegative=True)
        _number(relation, "uncertainty_s", nonnegative=True)
        if not _text(relation, "method"):
            _clock_refusal("clock relation method is empty")
        related_children.add(child)
    if tuple(sorted(set(keys))) != tuple(keys):
        _clock_refusal("clock relations must be unique and sorted")
    has_facility = any(
        item.plan_clock_kind == "facility_monotonic" for item in clocks.values()
    )
    shot_ids = {
        item.plan_clock_identifier
        for item in clocks.values()
        if item.plan_clock_kind == "shot_event_epoch"
    }
    if has_facility and not shot_ids <= related_children:
        _clock_refusal("shot clocks lack a facility-clock bound")


def _decode_source(data: bytes, name: str, *, pretty: bool) -> object:
    record = _decode_json(data, name, maximum=MAX_DEVICE_DIAGNOSTIC_SOURCE_BYTES)
    expected = _pretty(record) if pretty else _canonical(record)
    if expected != data:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.NONCANONICAL_SOURCE_BYTES,
            f"{name} bytes are noncanonical",
        )
    return record


def _decode_json(data: bytes, name: str, *, maximum: int) -> object:
    if not isinstance(data, bytes):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_TYPE,
            f"{name} must be bytes",
        )
    if not data or len(data) > maximum:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_SIZE,
            f"{name} byte length is outside 1..{maximum}",
        )

    def reject_constant(literal: str) -> NoReturn:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.INVALID_JSON,
            f"{name} contains non-finite literal {literal!r}",
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _refuse(
                    DeviceDiagnosticPlanRefusalCode.DUPLICATE_JSON_KEY,
                    f"{name} contains duplicate key {key!r}",
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse(
            DeviceDiagnosticPlanRefusalCode.INVALID_JSON,
            f"{name} is not strict UTF-8 JSON: {exc}",
        )


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _plan_refusal(f"{name} must be an object")
    if set(value) != keys:
        _plan_refusal(
            f"{name} key mismatch: missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}"
        )
    return value


def _object_array(
    parent: dict[str, object], name: str, keys: set[str]
) -> list[dict[str, object]]:
    value = parent[name]
    if not isinstance(value, list):
        _plan_refusal(f"{name} must be an array")
    return [_object(item, keys, f"{name}[]") for item in value]


def _text(parent: dict[str, object], name: str) -> str:
    value = parent.get(name)
    if not isinstance(value, str):
        _plan_refusal(f"{name} must be a string")
    return value


def _strings(parent: dict[str, object], name: str) -> tuple[str, ...]:
    value = parent.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _plan_refusal(f"{name} must be an array of strings")
    return tuple(value)


def _is_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _number(
    parent: dict[str, object],
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = parent.get(name)
    if not _is_number(value):
        _plan_refusal(f"{name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        _plan_refusal(f"{name} must be positive")
    if nonnegative and result < 0.0:
        _plan_refusal(f"{name} must be non-negative")
    return result


def _sorted_identifiers(values: tuple[str, ...], name: str) -> None:
    if tuple(sorted(set(values))) != values:
        _plan_refusal(f"{name} must be unique and sorted")
    if any(_IDENTIFIER.fullmatch(value) is None for value in values):
        _plan_refusal(f"{name} contain malformed identifiers")


def _require_source_identity(source_revision: str, artifact_digest: str) -> None:
    if not isinstance(source_revision, str) or not isinstance(artifact_digest, str):
        _refuse(
            DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_TYPE,
            "source revision and artifact digest must be strings",
        )
    if (
        _GIT_SHA.fullmatch(source_revision) is None
        or _SHA256.fullmatch(artifact_digest) is None
    ):
        _identity_refusal(
            "source identity requires a lowercase Git SHA and artifact SHA-256"
        )


def _clock_record(review: DeviceDiagnosticClockReview) -> dict[str, object]:
    return {
        "compatibility": review.compatibility.value,
        "epoch": review.epoch,
        "mapping_evidence_claimed": review.mapping_evidence_claimed,
        "plan_clock_identifier": review.plan_clock_identifier,
        "plan_clock_kind": review.plan_clock_kind,
        "resolution_s": review.resolution_s,
        "spo_clock_kind_candidate": (
            None
            if review.spo_clock_kind_candidate is None
            else review.spo_clock_kind_candidate.value
        ),
        "uncertainty_s": review.uncertainty_s,
    }


def _signal_record(review: DeviceDiagnosticSignalReview) -> dict[str, object]:
    return {
        "candidate_id": review.candidate_id,
        "carrier": review.carrier.value,
        "channel_identifier": review.channel_identifier,
        "clock_identifier": review.clock_identifier,
        "evidence_claimed": review.evidence_claimed,
        "evidence_slots": list(review.evidence_slots),
        "observability_class": review.observability_class.value,
        "observation_claimed": review.observation_claimed,
        "synthetic": review.synthetic,
    }


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
    ).encode()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _depth_refusal(error: DiagnosticPlanDepthError) -> NoReturn:
    if error.kind is DiagnosticPlanDepthRefusalKind.AUTHORITY:
        _authority_refusal(error.detail)
    if error.kind is DiagnosticPlanDepthRefusalKind.CARRIER:
        _carrier_refusal(error.detail)
    if error.kind is DiagnosticPlanDepthRefusalKind.CLOCK:
        _clock_refusal(error.detail)
    _plan_refusal(error.detail)


def _refuse(code: DeviceDiagnosticPlanRefusalCode, detail: str) -> NoReturn:
    raise DeviceDiagnosticPlanRefusal(code, detail)


def _identity_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.INVALID_SOURCE_IDENTITY, detail)


def _manifest_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.MANIFEST_CONTRACT_MISMATCH, detail)


def _assignment_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH, detail)


def _plan_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH, detail)


def _candidate_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.CANDIDATE_COVERAGE_MISMATCH, detail)


def _carrier_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH, detail)


def _clock_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH, detail)


def _authority_refusal(detail: str) -> NoReturn:
    _refuse(DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION, detail)


__all__ = [
    "DEVICE_DIAGNOSTIC_PLAN_REVIEW_SCHEMA",
    "DEVICE_DIAGNOSTIC_PLAN_REVIEW_VERSION",
    "MAX_DEVICE_DIAGNOSTIC_PLAN_REVIEW_BYTES",
    "MAX_DEVICE_DIAGNOSTIC_SOURCE_BYTES",
    "DeviceDiagnosticClockReview",
    "DeviceDiagnosticPlanRefusal",
    "DeviceDiagnosticPlanRefusalCode",
    "DeviceDiagnosticPlanReview",
    "DeviceDiagnosticSignalReview",
    "DiagnosticClockCompatibility",
    "device_diagnostic_plan_review_digest",
    "device_diagnostic_plan_review_from_bytes",
    "device_diagnostic_plan_review_from_producer_bytes",
    "device_diagnostic_plan_review_from_record",
    "device_diagnostic_plan_review_to_bytes",
    "device_diagnostic_plan_review_to_record",
]
