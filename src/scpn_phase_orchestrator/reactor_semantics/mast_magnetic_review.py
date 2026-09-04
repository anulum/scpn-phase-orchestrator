# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FAIR-MAST magnetic source review
"""Review complete FAIR-MAST magnetic source and qualification bytes.

The adapter preserves a physical-source custody chain without promoting the
source to an SPO phase observation.  In particular, a complete archive and a
producer qualification do not supply calibration lineage, transfer functions,
provider quality flags, uncertainty, an instrument-clock relation, or a
facility event identity.  Consequently this module cannot infer phase, declare
semantic ingress, classify a regime, or create a control object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, TypeGuard, cast

from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
from .vocabulary import ClockKind

MAST_MAGNETIC_SOURCE_REVIEW_SCHEMA: Final = (
    "scpn-phase-orchestrator.mast-magnetic-source-review.v1"
)
MAST_MAGNETIC_SOURCE_REVIEW_VERSION: Final = "1.0.0"
MAX_MAST_MAGNETIC_SOURCE_BYTES: Final = 8 * 1024 * 1024
MAX_MAST_MAGNETIC_SOURCE_REVIEW_BYTES: Final = 16 * 1024 * 1024

_ARCHIVE_SCHEMA: Final = "scpn-fusion-core.mast-complete-magnetic-archive-envelope.v1"
_ARCHIVE_VERSION: Final = "1.0.0"
_QUALIFICATION_SCHEMA: Final = (
    "scpn-fusion-core.mast-magnetic-diagnostic-qualification.v1"
)
_QUALIFICATION_VERSION: Final = "1.0.0"
_PRODUCER_PROJECT: Final = "SCPN-FUSION-CORE"
_DEVICE_PROJECT: Final = "SCPN-TOKAMAK-CORE"
_FACILITY: Final = "MAST"
_CONFIGURATION: Final = "spherical_tokamak"
_SOURCE_ARCHIVE: Final = "FAIR-MAST"

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_OUTER_KEYS = {"payload", "payload_sha256", "schema", "schema_version"}
_ARCHIVE_KEYS = {
    "archive",
    "arrays",
    "authority",
    "clocks",
    "completeness",
    "event_id",
    "event_identity_state",
    "facility",
    "observation_id",
    "producer_artifacts",
    "producer_project",
    "provenance",
    "qualification",
    "reactor_configuration",
    "shot_id",
    "source_archive",
    "source_ingestion_revision",
    "source_ingestion_tree_state",
}
_QUALIFICATION_KEYS = {
    "archive_envelope_sha256",
    "archive_observation_id",
    "array_inventory",
    "authority",
    "channel_geometry_evidence",
    "clock_evidence",
    "completeness",
    "event_identity",
    "external_limitations",
    "facility",
    "ingestion_mapping",
    "measurement_evidence",
    "producer_project",
    "qualification_summary",
    "reactor_configuration",
    "shot_id",
    "source_archive",
}
_ARCHIVE_ARRAY_KEYS = {
    "archive_path",
    "attributes",
    "clock_dimensions",
    "data_object_paths",
    "data_type",
    "decoded_content_sha256",
    "decoded_nonfinite_count",
    "decoded_value_count",
    "dimension_names",
    "metadata_object_path",
    "metadata_object_sha256",
    "name",
    "shape",
    "zarr_metadata",
}
_ARCHIVE_CLOCK_KEYS = {
    "clock_kind_candidate",
    "clock_qualification",
    "finite",
    "first_value_s",
    "last_value_s",
    "mapping_evidence_claimed",
    "maximum_interval_s",
    "mean_interval_s",
    "minimum_interval_s",
    "name",
    "sample_count",
    "strictly_increasing",
    "units",
}
_QUALIFICATION_ARRAY_KEYS = {
    "clock_dimensions",
    "dimension_names",
    "name",
    "role",
    "shape",
}
_QUALIFICATION_CLOCK_KEYS = {
    "archive_grid_reproduced",
    "dropna",
    "first_value_s",
    "grid_origin",
    "interpolation_method",
    "last_value_s",
    "name",
    "sample_count",
    "source_clock_relation_claimed",
    "start_s",
    "step_s",
}
_MEASUREMENT_KEYS = {
    "applied_background_sample_range",
    "applied_scale",
    "archive_channel_ids",
    "array_name",
    "calibration_lineage_state",
    "channel_quality",
    "clock_name",
    "configured_source_channels",
    "empirical_quality",
    "imas_quantity_path",
    "observation_operator_state",
    "provider_quality_flags_supplied",
    "source_name",
    "source_shot_max",
    "source_shot_min",
    "source_valid_for_shot",
    "target_units",
    "uncertainty_supplied",
    "units",
}
_GEOMETRY_KEYS = {
    "archive_channel_id",
    "geometry_channel_id",
    "geometry_coordinate",
    "identifier_match_method",
    "measurement_array",
    "physical_mapping_claimed",
}
_QUALITY_KEYS = {
    "finite_count",
    "infinite_count",
    "minimum_positive_level_spacing_hex",
    "nan_count",
    "nan_fraction",
    "sample_count",
    "unique_finite_value_count",
    "zero_count",
}
_CLOCK_NAMES: Final = ("time", "time_mirnov", "time_omaha", "time_saddle")
_MEASUREMENT_NAMES: Final = (
    "b_field_pol_probe_cc_field",
    "b_field_pol_probe_ccbv_field",
    "b_field_pol_probe_obr_field",
    "b_field_pol_probe_obv_field",
    "b_field_pol_probe_omv_voltage",
    "b_field_tor_probe_cc_field",
    "b_field_tor_probe_omaha_voltage",
    "b_field_tor_probe_saddle_field",
    "b_field_tor_probe_saddle_voltage",
    "flux_loop_flux",
    "ip",
)
_ROLE_COUNTS: Final = {
    "channel_coordinate": 21,
    "clock": 4,
    "geometry": 35,
    "measurement": 11,
    "shot_identity": 1,
}
_ARCHIVE_AUTHORITY: Final = {
    "actionable": False,
    "classification_performed": False,
    "direct_actuation": False,
    "execution_permitted": False,
    "review_only": True,
}
_QUALIFICATION_AUTHORITY: Final = {
    **_ARCHIVE_AUTHORITY,
    "phase_inference_performed": False,
}
_ARCHIVE_QUALIFICATION: Final = {
    "calibration_state": "unresolved",
    "channel_geometry_mapping_state": "unresolved",
    "classification_eligible": False,
    "diagnostic_semantics_state": "source_preserved_unqualified",
    "event_clock_state": "unresolved",
    "observation_operator_state": "not_supplied",
    "phase_eligible": False,
    "quality_state": "unknown",
    "raw_samples_present": True,
    "semantic_ingress_eligible": False,
    "source_clock_relationship_state": "unresolved",
    "source_kind": "physical_archive",
    "synthetic": False,
    "uncertainty_state": "unknown",
    "validity_state": "unknown",
}
_QUALIFICATION_SUMMARY: Final = {
    "calibration_state": "applied_transforms_recorded_lineage_unavailable",
    "channel_geometry_mapping_state": "identifier_correspondence_only",
    "event_identity_state": "shot_only_event_unresolved",
    "observation_operator_state": "quantity_paths_only_transfer_functions_unavailable",
    "provider_quality_state": "not_supplied",
    "source_clock_relationship_state": (
        "derived_archive_grids_no_instrument_clock_relation"
    ),
    "uncertainty_state": "not_supplied",
    "validity_state": "source_shot_ranges_only",
}
_UNRESOLVED_FIELDS: Final = tuple(sorted(_QUALIFICATION_SUMMARY))


class MastMagneticSourceRefusalCode(StrEnum):
    """Stable categories for fail-closed source-review refusal."""

    INVALID_INPUT = "invalid_input"
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    NONCANONICAL_BYTES = "noncanonical_bytes"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    SOURCE_DIGEST_MISMATCH = "source_digest_mismatch"
    ARCHIVE_CONTRACT_MISMATCH = "archive_contract_mismatch"
    QUALIFICATION_CONTRACT_MISMATCH = "qualification_contract_mismatch"
    CROSS_SOURCE_MISMATCH = "cross_source_mismatch"
    REGISTRY_ASSIGNMENT_MISMATCH = "registry_assignment_mismatch"
    AUTHORITY_ESCALATION = "authority_escalation"


class MastMagneticSourceRefusalError(ValueError):
    """Raised when FAIR-MAST bytes cannot form a safe SPO review."""

    def __init__(self, code: MastMagneticSourceRefusalCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


MastMagneticSourceRefusal = MastMagneticSourceRefusalError


@dataclass(frozen=True, slots=True)
class MastMagneticClockReview:
    """One derived archive grid, explicitly not an instrument-clock mapping."""

    name: str
    sample_count: int
    first_value_s: float
    last_value_s: float
    step_s: float
    spo_clock_kind_candidate: ClockKind = ClockKind.SHOT_RELATIVE
    archive_grid_reproduced: bool = True
    instrument_clock_relation_claimed: bool = False
    mapping_evidence_claimed: bool = False


@dataclass(frozen=True, slots=True)
class MastMagneticMeasurementReview:
    """One producer-qualified measurement family without phase eligibility."""

    array_name: str
    clock_name: str
    units: str
    channel_count: int
    source_valid_for_shot: bool
    applied_transform_recorded: bool = True
    calibration_lineage_available: bool = False
    observation_operator_available: bool = False
    provider_quality_flags_supplied: bool = False
    uncertainty_supplied: bool = False
    phase_eligible: bool = False


@dataclass(frozen=True, slots=True)
class _ValidatedSources:
    """Hold producer artifacts after cross-source validation."""

    shot_id: int
    observation_id: str
    archive_sha256: str
    qualification_sha256: str
    archive_payload_sha256: str
    qualification_payload_sha256: str
    source_ingestion_revision: str
    source_ingestion_tree_state: str
    array_count: int
    measurement_count: int
    channel_count: int
    clock_reviews: tuple[MastMagneticClockReview, ...]
    measurement_reviews: tuple[MastMagneticMeasurementReview, ...]


@dataclass(frozen=True, slots=True)
class MastMagneticSourceReview:
    """Digest-sealed review of complete magnetic source and qualification bytes."""

    source_revision: str
    source_artifact_sha256: str
    source_archive_json: str
    source_qualification_json: str
    review_id: str = field(init=False)
    source_project: str = field(init=False, default=_PRODUCER_PROJECT)
    device_project: str = field(init=False, default=_DEVICE_PROJECT)
    facility: str = field(init=False, default=_FACILITY)
    configuration: str = field(init=False, default=_CONFIGURATION)
    source_archive: str = field(init=False, default=_SOURCE_ARCHIVE)
    shot_id: int = field(init=False)
    observation_id: str = field(init=False)
    source_archive_sha256: str = field(init=False)
    source_qualification_sha256: str = field(init=False)
    archive_payload_sha256: str = field(init=False)
    qualification_payload_sha256: str = field(init=False)
    source_ingestion_revision: str = field(init=False)
    source_ingestion_tree_state: str = field(init=False)
    array_count: int = field(init=False)
    measurement_count: int = field(init=False)
    channel_count: int = field(init=False)
    clock_reviews: tuple[MastMagneticClockReview, ...] = field(init=False)
    measurement_reviews: tuple[MastMagneticMeasurementReview, ...] = field(init=False)
    unresolved_qualification_fields: tuple[str, ...] = field(
        init=False, default=_UNRESOLVED_FIELDS
    )
    semantic_ingress_state: str = field(init=False, default="not_declared")
    accepted_as_physical_source_review: bool = field(init=False, default=True)
    physical_source_recorded: bool = field(init=False, default=True)
    observation_admitted: bool = field(init=False, default=False)
    qualified_phase_evidence: bool = field(init=False, default=False)
    phase_inference_performed: bool = field(init=False, default=False)
    semantic_ingress_declared: bool = field(init=False, default=False)
    classification_performed: bool = field(init=False, default=False)
    control_intent_created: bool = field(init=False, default=False)
    actionable: bool = field(init=False, default=False)
    execution_permitted: bool = field(init=False, default=False)
    direct_actuation: bool = field(init=False, default=False)
    review_only: bool = field(init=False, default=True)
    machine_protection_final_veto: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        _require_source_identity(self.source_revision, self.source_artifact_sha256)
        validated = _validate_sources(
            self.source_archive_json.encode("utf-8"),
            self.source_qualification_json.encode("utf-8"),
        )
        for name in (
            "shot_id",
            "observation_id",
            "archive_payload_sha256",
            "qualification_payload_sha256",
            "source_ingestion_revision",
            "source_ingestion_tree_state",
            "array_count",
            "measurement_count",
            "channel_count",
            "clock_reviews",
            "measurement_reviews",
        ):
            object.__setattr__(self, name, getattr(validated, name))
        object.__setattr__(self, "source_archive_sha256", validated.archive_sha256)
        object.__setattr__(
            self, "source_qualification_sha256", validated.qualification_sha256
        )
        identity = {
            "source_archive_sha256": validated.archive_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_qualification_sha256": validated.qualification_sha256,
            "source_revision": self.source_revision,
        }
        object.__setattr__(self, "review_id", _sha256(_canonical(identity)))


def mast_magnetic_source_review_from_producer_bytes(
    *,
    source_revision: str,
    source_artifact_sha256: str,
    archive_bytes: bytes,
    qualification_bytes: bytes,
) -> MastMagneticSourceReview:
    """Review exact installed-producer bytes without importing producer code.

    Parameters
    ----------
    source_revision : str
        Full Git SHA of the producer source used to build the installed wheel.
    source_artifact_sha256 : str
        SHA-256 of that exact producer wheel.
    archive_bytes : bytes
        Canonical complete FAIR-MAST magnetic archive envelope.
    qualification_bytes : bytes
        Canonical diagnostic-qualification document bound to the archive.

    Returns
    -------
    MastMagneticSourceReview
        Revalidated, review-only physical-source custody record.
    """
    _require_source_identity(source_revision, source_artifact_sha256)
    _decode_document(archive_bytes, "archive")
    _decode_document(qualification_bytes, "qualification")
    return MastMagneticSourceReview(
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        source_archive_json=archive_bytes.decode("utf-8"),
        source_qualification_json=qualification_bytes.decode("utf-8"),
    )


def mast_magnetic_source_review_to_record(
    review: MastMagneticSourceReview,
) -> dict[str, object]:
    """Return the complete deterministic review payload.

    Parameters
    ----------
    review : MastMagneticSourceReview
        Validated FAIR-MAST source review.

    Returns
    -------
    dict[str, object]
        Complete payload including both exact source documents.
    """
    return {
        "accepted_as_physical_source_review": review.accepted_as_physical_source_review,
        "actionable": review.actionable,
        "archive_payload_sha256": review.archive_payload_sha256,
        "array_count": review.array_count,
        "channel_count": review.channel_count,
        "classification_performed": review.classification_performed,
        "clock_reviews": [_clock_record(item) for item in review.clock_reviews],
        "configuration": review.configuration,
        "control_intent_created": review.control_intent_created,
        "device_project": review.device_project,
        "direct_actuation": review.direct_actuation,
        "execution_permitted": review.execution_permitted,
        "facility": review.facility,
        "machine_protection_final_veto": review.machine_protection_final_veto,
        "measurement_count": review.measurement_count,
        "measurement_reviews": [
            _measurement_record(item) for item in review.measurement_reviews
        ],
        "observation_admitted": review.observation_admitted,
        "observation_id": review.observation_id,
        "phase_inference_performed": review.phase_inference_performed,
        "physical_source_recorded": review.physical_source_recorded,
        "qualification_payload_sha256": review.qualification_payload_sha256,
        "qualified_phase_evidence": review.qualified_phase_evidence,
        "review_id": review.review_id,
        "review_only": review.review_only,
        "semantic_ingress_declared": review.semantic_ingress_declared,
        "semantic_ingress_state": review.semantic_ingress_state,
        "shot_id": review.shot_id,
        "source_archive": review.source_archive,
        "source_archive_json": review.source_archive_json,
        "source_archive_sha256": review.source_archive_sha256,
        "source_artifact_sha256": review.source_artifact_sha256,
        "source_ingestion_revision": review.source_ingestion_revision,
        "source_ingestion_tree_state": review.source_ingestion_tree_state,
        "source_project": review.source_project,
        "source_qualification_json": review.source_qualification_json,
        "source_qualification_sha256": review.source_qualification_sha256,
        "source_revision": review.source_revision,
        "unresolved_qualification_fields": list(review.unresolved_qualification_fields),
    }


_REVIEW_KEYS = {
    "accepted_as_physical_source_review",
    "actionable",
    "archive_payload_sha256",
    "array_count",
    "channel_count",
    "classification_performed",
    "clock_reviews",
    "configuration",
    "control_intent_created",
    "device_project",
    "direct_actuation",
    "execution_permitted",
    "facility",
    "machine_protection_final_veto",
    "measurement_count",
    "measurement_reviews",
    "observation_admitted",
    "observation_id",
    "phase_inference_performed",
    "physical_source_recorded",
    "qualification_payload_sha256",
    "qualified_phase_evidence",
    "review_id",
    "review_only",
    "semantic_ingress_declared",
    "semantic_ingress_state",
    "shot_id",
    "source_archive",
    "source_archive_json",
    "source_archive_sha256",
    "source_artifact_sha256",
    "source_ingestion_revision",
    "source_ingestion_tree_state",
    "source_project",
    "source_qualification_json",
    "source_qualification_sha256",
    "source_revision",
    "unresolved_qualification_fields",
}


def mast_magnetic_source_review_from_record(record: object) -> MastMagneticSourceReview:
    """Rebuild a review and replay every embedded source invariant.

    Parameters
    ----------
    record : object
        Candidate complete review payload.

    Returns
    -------
    MastMagneticSourceReview
        Reconstructed review after exact source and derived-field replay.
    """
    payload = _object(record, _REVIEW_KEYS, "review payload")
    review = MastMagneticSourceReview(
        source_revision=_text(payload, "source_revision"),
        source_artifact_sha256=_text(payload, "source_artifact_sha256"),
        source_archive_json=_text(payload, "source_archive_json"),
        source_qualification_json=_text(payload, "source_qualification_json"),
    )
    if mast_magnetic_source_review_to_record(review) != payload:
        _refuse(
            MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH,
            "stored review fields differ from reconstructed source bytes",
        )
    return review


def mast_magnetic_source_review_to_bytes(review: MastMagneticSourceReview) -> bytes:
    """Serialize a review in a canonical digest-sealed envelope.

    Parameters
    ----------
    review : MastMagneticSourceReview
        Validated FAIR-MAST source review.

    Returns
    -------
    bytes
        Unique canonical UTF-8 review envelope with one trailing newline.
    """
    payload = mast_magnetic_source_review_to_record(review)
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": MAST_MAGNETIC_SOURCE_REVIEW_SCHEMA,
            "schema_version": MAST_MAGNETIC_SOURCE_REVIEW_VERSION,
        }
    )


def mast_magnetic_source_review_from_bytes(data: bytes) -> MastMagneticSourceReview:
    """Decode canonical review bytes and replay the complete custody chain.

    Parameters
    ----------
    data : bytes
        Candidate canonical review envelope.

    Returns
    -------
    MastMagneticSourceReview
        Reconstructed review after envelope, digest, and source replay.
    """
    document = _decode_document(
        data, "review", maximum=MAX_MAST_MAGNETIC_SOURCE_REVIEW_BYTES
    )
    if (
        document["schema"] != MAST_MAGNETIC_SOURCE_REVIEW_SCHEMA
        or document["schema_version"] != MAST_MAGNETIC_SOURCE_REVIEW_VERSION
    ):
        _refuse(MastMagneticSourceRefusalCode.UNSUPPORTED_SCHEMA, "review schema drift")
    payload = _object(document["payload"], _REVIEW_KEYS, "review payload")
    if _text(document, "payload_sha256") != _sha256(_canonical(payload)):
        _refuse(
            MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH,
            "review payload digest mismatch",
        )
    return mast_magnetic_source_review_from_record(payload)


def mast_magnetic_source_review_digest(review: MastMagneticSourceReview) -> str:
    """Return SHA-256 of the complete canonical review envelope.

    Parameters
    ----------
    review : MastMagneticSourceReview
        Validated FAIR-MAST source review.

    Returns
    -------
    str
        Lowercase SHA-256 hexadecimal digest.
    """
    return _sha256(mast_magnetic_source_review_to_bytes(review))


def _validate_sources(
    archive_bytes: bytes, qualification_bytes: bytes
) -> _ValidatedSources:
    """Decode and cross-check the pinned producer source artifacts."""
    archive = _decode_document(archive_bytes, "archive")
    qualification = _decode_document(qualification_bytes, "qualification")
    _require_schema(archive, _ARCHIVE_SCHEMA, _ARCHIVE_VERSION, "archive")
    _require_schema(
        qualification, _QUALIFICATION_SCHEMA, _QUALIFICATION_VERSION, "qualification"
    )
    archive_payload = _object(archive["payload"], _ARCHIVE_KEYS, "archive payload")
    qualification_payload = _object(
        qualification["payload"], _QUALIFICATION_KEYS, "qualification payload"
    )
    archive_payload_digest = _validate_payload_digest(
        archive, archive_payload, "archive"
    )
    qualification_payload_digest = _validate_payload_digest(
        qualification, qualification_payload, "qualification"
    )
    archive_values = _validate_archive(archive_payload)
    qualification_values = _validate_qualification(
        qualification_payload, archive_payload, archive_values
    )
    return _ValidatedSources(
        shot_id=archive_values.shot_id,
        observation_id=archive_values.observation_id,
        archive_sha256=_sha256(archive_bytes),
        qualification_sha256=_sha256(qualification_bytes),
        archive_payload_sha256=archive_payload_digest,
        qualification_payload_sha256=qualification_payload_digest,
        source_ingestion_revision=archive_values.source_revision,
        source_ingestion_tree_state=archive_values.source_tree_state,
        array_count=len(archive_values.arrays),
        measurement_count=len(qualification_values.measurements),
        channel_count=qualification_values.channel_count,
        clock_reviews=qualification_values.clocks,
        measurement_reviews=qualification_values.measurements,
    )


@dataclass(frozen=True, slots=True)
class _ArchiveValues:
    """Hold validated MAST archive values for cross-document checks."""

    shot_id: int
    observation_id: str
    source_revision: str
    source_tree_state: str
    arrays: dict[str, dict[str, object]]
    clocks: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class _QualificationValues:
    """Hold validated MAST qualification values for review checks."""

    clocks: tuple[MastMagneticClockReview, ...]
    measurements: tuple[MastMagneticMeasurementReview, ...]
    channel_count: int


def _validate_archive(payload: dict[str, object]) -> _ArchiveValues:
    """Validate the MAST archive document and return bound values."""
    _require_equal(payload["producer_project"], _PRODUCER_PROJECT, "producer")
    _require_equal(payload["facility"], _FACILITY, "facility")
    _require_equal(payload["reactor_configuration"], _CONFIGURATION, "configuration")
    _require_equal(payload["source_archive"], _SOURCE_ARCHIVE, "source archive")
    _require_equal(payload["authority"], _ARCHIVE_AUTHORITY, "archive authority")
    _require_equal(payload["qualification"], _ARCHIVE_QUALIFICATION, "qualification")
    _validate_registry_boundary()
    shot_id = _positive_int(payload["shot_id"], "shot_id")
    observation_id = _text(payload, "observation_id")
    if not observation_id.startswith(f"mast-{shot_id}-complete-magnetics-"):
        _archive_refusal("observation identity does not bind the shot")
    if payload["event_id"] is not None or payload["event_identity_state"] != (
        "unresolved_facility_mapping"
    ):
        _authority_refusal("archive cannot claim a resolved facility event")
    source_revision = _matching_text(
        payload["source_ingestion_revision"], _GIT_SHA, "source ingestion revision"
    )
    source_tree_state = _text(payload, "source_ingestion_tree_state")
    if source_tree_state not in {"clean", "dirty"}:
        _archive_refusal("unsupported ingestion tree state")
    completeness = _object(
        payload["completeness"],
        {
            "array_count",
            "arrays_complete",
            "clock_count",
            "objects_complete",
            "source_decoded",
        },
        "archive completeness",
    )
    if completeness != {
        "array_count": 72,
        "arrays_complete": True,
        "clock_count": 4,
        "objects_complete": True,
        "source_decoded": True,
    }:
        _archive_refusal("archive is not the complete v1 magnetic inventory")
    arrays = _validate_archive_arrays(_array(payload["arrays"], "archive arrays"))
    clocks = _validate_archive_clocks(_array(payload["clocks"], "archive clocks"))
    if len(arrays) != 72 or len(clocks) != 4:
        _archive_refusal("archive inventory cardinality drifted")
    return _ArchiveValues(
        shot_id,
        observation_id,
        source_revision,
        source_tree_state,
        arrays,
        clocks,
    )


def _validate_archive_arrays(values: list[object]) -> dict[str, dict[str, object]]:
    """Validate archive array geometry and numeric evidence."""
    arrays: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(values):
        item = _object(raw, _ARCHIVE_ARRAY_KEYS, f"archive array {index}")
        name = _text(item, "name")
        if name in arrays:
            _archive_refusal("archive array names must be unique")
        _matching_text(item["decoded_content_sha256"], _SHA256, "decoded digest")
        _matching_text(item["metadata_object_sha256"], _SHA256, "metadata digest")
        _nonnegative_int(item["decoded_nonfinite_count"], "nonfinite count")
        _nonnegative_int(item["decoded_value_count"], "value count")
        _shape(item["shape"], "archive shape")
        _strings(item["dimension_names"], "archive dimensions")
        _strings(item["clock_dimensions"], "archive clock dimensions")
        arrays[name] = item
    return arrays


def _validate_archive_clocks(values: list[object]) -> dict[str, dict[str, object]]:
    """Validate archive clock declarations."""
    clocks: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(values):
        item = _object(raw, _ARCHIVE_CLOCK_KEYS, f"archive clock {index}")
        name = _text(item, "name")
        if name in clocks:
            _archive_refusal("archive clock names must be unique")
        if (
            item["clock_kind_candidate"] != "shot_relative"
            or item["clock_qualification"] != "unresolved"
            or item["mapping_evidence_claimed"] is not False
            or item["finite"] is not True
            or item["strictly_increasing"] is not True
            or item["units"] != "s"
        ):
            _authority_refusal("archive clock meaning or authority drifted")
        _positive_int(item["sample_count"], "clock sample count")
        for key in (
            "first_value_s",
            "last_value_s",
            "minimum_interval_s",
            "mean_interval_s",
            "maximum_interval_s",
        ):
            _finite_number(item[key], key)
        clocks[name] = item
    if tuple(sorted(clocks)) != tuple(sorted(_CLOCK_NAMES)):
        _archive_refusal("archive clock inventory drifted")
    return clocks


def _validate_qualification(
    payload: dict[str, object],
    archive_payload: dict[str, object],
    archive_values: _ArchiveValues,
) -> _QualificationValues:
    """Validate qualification evidence against the archive."""
    _require_equal(payload["producer_project"], _PRODUCER_PROJECT, "producer")
    _require_equal(payload["facility"], _FACILITY, "facility")
    _require_equal(payload["reactor_configuration"], _CONFIGURATION, "configuration")
    _require_equal(payload["source_archive"], _SOURCE_ARCHIVE, "source archive")
    _require_equal(
        payload["authority"], _QUALIFICATION_AUTHORITY, "qualification authority"
    )
    _require_equal(payload["qualification_summary"], _QUALIFICATION_SUMMARY, "summary")
    if payload["shot_id"] != archive_values.shot_id:
        _cross_refusal("qualification shot differs from archive")
    if payload["archive_observation_id"] != archive_values.observation_id:
        _cross_refusal("qualification observation differs from archive")
    if payload["archive_envelope_sha256"] != _sha256(
        _canonical_document(archive_payload, _ARCHIVE_SCHEMA, _ARCHIVE_VERSION)
    ):
        _cross_refusal("qualification archive digest mismatch")
    event = _object(
        payload["event_identity"],
        {"event_id", "event_time_epoch", "shot_id", "state"},
        "event identity",
    )
    if event != {
        "event_id": None,
        "event_time_epoch": None,
        "shot_id": archive_values.shot_id,
        "state": "shot_only_event_unresolved",
    }:
        _authority_refusal("qualification cannot claim resolved event identity")
    mapping = _object(
        payload["ingestion_mapping"],
        {
            "dataset_license_name",
            "dataset_license_url",
            "mapping_path",
            "mapping_sha256",
            "mapping_url",
            "source_revision",
            "source_tree_state",
        },
        "ingestion mapping",
    )
    if (
        mapping["source_revision"] != archive_values.source_revision
        or mapping["source_tree_state"] != archive_values.source_tree_state
    ):
        _cross_refusal("qualification ingestion identity differs from archive")
    _matching_text(mapping["mapping_sha256"], _SHA256, "mapping digest")
    qualification_arrays = _validate_qualification_arrays(
        _array(payload["array_inventory"], "qualification arrays"),
        archive_values.arrays,
    )
    archive_clocks = archive_values.clocks
    clock_reviews = _validate_qualification_clocks(
        _array(payload["clock_evidence"], "qualification clocks"), archive_clocks
    )
    measurements = _validate_measurements(
        _array(payload["measurement_evidence"], "measurements"),
        qualification_arrays,
        {item.name for item in clock_reviews},
    )
    geometry = _validate_geometry(
        _array(payload["channel_geometry_evidence"], "geometry evidence"),
        {item.array_name for item in measurements},
    )
    completeness = _object(
        payload["completeness"],
        {
            "archive_array_count",
            "archive_arrays_classified",
            "channel_record_count",
            "clock_count",
            "measurement_count",
            "measurements_analysed",
        },
        "qualification completeness",
    )
    channel_count = _positive_int(completeness["channel_record_count"], "channel count")
    if completeness != {
        "archive_array_count": 72,
        "archive_arrays_classified": True,
        "channel_record_count": channel_count,
        "clock_count": 4,
        "measurement_count": 11,
        "measurements_analysed": True,
    } or channel_count != len(geometry):
        _qualification_refusal("qualification completeness is inconsistent")
    limitations = _array(payload["external_limitations"], "external limitations")
    if len(limitations) != 1:
        _qualification_refusal("exactly one scoped external limitation is required")
    limitation = _object(
        limitations[0],
        {"applicability_to_shot", "issue", "reported_shot_id", "scope", "url"},
        "external limitation",
    )
    if limitation["applicability_to_shot"] != "not_assumed":
        _authority_refusal("external limitation applicability cannot be inferred")
    return _QualificationValues(clock_reviews, measurements, channel_count)


def _validate_qualification_arrays(
    values: list[object], archive: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    """Validate qualification arrays against archive geometry."""
    if len(values) != 72:
        _qualification_refusal("qualification must classify all 72 arrays")
    arrays: dict[str, dict[str, object]] = {}
    counts: dict[str, int] = {}
    for index, raw in enumerate(values):
        item = _object(raw, _QUALIFICATION_ARRAY_KEYS, f"qualification array {index}")
        name = _text(item, "name")
        role = _text(item, "role")
        if name in arrays or role not in _ROLE_COUNTS:
            _qualification_refusal("array identity or role is unsupported")
        source = archive.get(name)
        if source is None:
            _cross_refusal("qualification names an array absent from archive")
        if (
            _shape(item["shape"], "qualification shape")
            != _shape(source["shape"], "archive shape")
            or _strings(item["dimension_names"], "qualification dimensions")
            != _strings(source["dimension_names"], "archive dimensions")
            or _strings(item["clock_dimensions"], "qualification clock dimensions")
            != _strings(source["clock_dimensions"], "archive clock dimensions")
        ):
            _cross_refusal("qualification array geometry differs from archive")
        arrays[name] = item
        counts[role] = counts.get(role, 0) + 1
    if counts != _ROLE_COUNTS or set(arrays) != set(archive):
        _qualification_refusal("complete array role inventory drifted")
    return arrays


def _validate_qualification_clocks(
    values: list[object], archive: dict[str, dict[str, object]]
) -> tuple[MastMagneticClockReview, ...]:
    """Validate qualification clocks against archive clocks."""
    reviews: list[MastMagneticClockReview] = []
    for index, raw in enumerate(values):
        item = _object(raw, _QUALIFICATION_CLOCK_KEYS, f"qualification clock {index}")
        name = _text(item, "name")
        source = archive.get(name)
        if source is None:
            _cross_refusal("qualification clock is absent from archive")
        if (
            item["archive_grid_reproduced"] is not True
            or item["dropna"] is not True
            or item["grid_origin"] != "level2_interpolation"
            or item["interpolation_method"] != "zero"
            or item["source_clock_relation_claimed"] is not False
        ):
            _authority_refusal("qualification clock meaning or authority drifted")
        sample_count = _positive_int(item["sample_count"], "clock sample count")
        first = _finite_number(item["first_value_s"], "first clock value")
        last = _finite_number(item["last_value_s"], "last clock value")
        start = _finite_number(item["start_s"], "clock start")
        step = _positive_number(item["step_s"], "clock step")
        if (
            sample_count != source["sample_count"]
            or first != source["first_value_s"]
            or last != source["last_value_s"]
            or start != first
            or not math.isclose(
                step,
                _finite_number(source["mean_interval_s"], "mean interval"),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            _cross_refusal("qualification clock grid differs from archive")
        reviews.append(MastMagneticClockReview(name, sample_count, first, last, step))
    reviews.sort(key=lambda item: item.name)
    if tuple(item.name for item in reviews) != tuple(sorted(_CLOCK_NAMES)):
        _qualification_refusal("qualification clock inventory drifted")
    return tuple(reviews)


def _validate_measurements(
    values: list[object],
    arrays: dict[str, dict[str, object]],
    clocks: set[str],
) -> tuple[MastMagneticMeasurementReview, ...]:
    """Validate measurements against arrays and clock definitions."""
    reviews: list[MastMagneticMeasurementReview] = []
    for index, raw in enumerate(values):
        item = _object(raw, _MEASUREMENT_KEYS, f"measurement {index}")
        name = _text(item, "array_name")
        clock = _text(item, "clock_name")
        if arrays.get(name, {}).get("role") != "measurement" or clock not in clocks:
            _qualification_refusal("measurement array or clock binding is invalid")
        if (
            item["calibration_lineage_state"] != "not_supplied"
            or item["observation_operator_state"]
            != "imas_quantity_path_only_transfer_function_not_supplied"
            or item["provider_quality_flags_supplied"] is not False
            or item["uncertainty_supplied"] is not False
            or item["source_valid_for_shot"] is not True
        ):
            _authority_refusal("measurement qualification was over-promoted")
        channel_ids = _strings(item["archive_channel_ids"], "measurement channels")
        if len(channel_ids) != len(set(channel_ids)):
            _qualification_refusal("measurement channels must be unique")
        quality = _array(item["channel_quality"], "channel quality")
        if len(quality) != len(channel_ids):
            _qualification_refusal(
                "channel quality does not cover measurement channels"
            )
        _validate_quality(item["empirical_quality"], "empirical quality")
        for quality_index, raw_quality in enumerate(quality):
            row = _object(
                raw_quality,
                {"archive_channel_id", "quality"},
                f"channel quality {quality_index}",
            )
            if row["archive_channel_id"] not in channel_ids:
                _qualification_refusal("channel quality identity is unbound")
            _validate_quality(row["quality"], "channel quality values")
        reviews.append(
            MastMagneticMeasurementReview(
                array_name=name,
                clock_name=clock,
                units=_text(item, "units"),
                channel_count=len(channel_ids),
                source_valid_for_shot=True,
            )
        )
    reviews.sort(key=lambda item: item.array_name)
    if tuple(item.array_name for item in reviews) != tuple(sorted(_MEASUREMENT_NAMES)):
        _qualification_refusal("measurement inventory drifted")
    return tuple(reviews)


def _validate_quality(value: object, name: str) -> None:
    """Validate one measurement-quality value."""
    quality = _object(value, _QUALITY_KEYS, name)
    for key in (
        "finite_count",
        "infinite_count",
        "nan_count",
        "sample_count",
        "unique_finite_value_count",
        "zero_count",
    ):
        _nonnegative_int(quality[key], f"{name} {key}")
    fraction = _finite_number(quality["nan_fraction"], f"{name} nan fraction")
    if not 0.0 <= fraction <= 1.0:
        _qualification_refusal("quality NaN fraction is outside [0,1]")
    spacing = quality["minimum_positive_level_spacing_hex"]
    if spacing is not None and not isinstance(spacing, str):
        _qualification_refusal("quality spacing must be hexadecimal text or null")


def _validate_geometry(
    values: list[object], measurements: set[str]
) -> list[dict[str, object]]:
    """Validate cross-measurement geometry invariants."""
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(values):
        item = _object(raw, _GEOMETRY_KEYS, f"geometry row {index}")
        if item["measurement_array"] not in measurements:
            _qualification_refusal("geometry row references an unknown measurement")
        if item["physical_mapping_claimed"] is not False:
            _authority_refusal(
                "identifier correspondence cannot claim physical mapping"
            )
        rows.append(item)
    return rows


def _validate_registry_boundary() -> None:
    """Validate the governed MAST registry boundary."""
    profile = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(_CONFIGURATION)
    if (
        profile.device_project != _DEVICE_PROJECT
        or profile.producer_project is not None
        or profile.ingress_state.value != "not_declared"
        or profile.actionable is not False
        or profile.machine_protection_final_veto is not True
    ):
        _refuse(
            MastMagneticSourceRefusalCode.REGISTRY_ASSIGNMENT_MISMATCH,
            "spherical-tokamak semantic profile no longer has the review-only boundary",
        )


def _decode_document(
    data: bytes, name: str, *, maximum: int = MAX_MAST_MAGNETIC_SOURCE_BYTES
) -> dict[str, object]:
    """Decode and validate one canonical JSON document."""
    if not isinstance(data, bytes) or not data or len(data) > maximum:
        _refuse(
            MastMagneticSourceRefusalCode.INVALID_INPUT, f"{name} byte input invalid"
        )
    try:
        text = data.decode("utf-8")
        value = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_reject_duplicates,
                parse_constant=_reject_constant,
            ),
        )
    except UnicodeDecodeError as exc:
        _refuse(
            MastMagneticSourceRefusalCode.INVALID_JSON, f"{name} is not UTF-8: {exc}"
        )
    except json.JSONDecodeError as exc:
        _refuse(
            MastMagneticSourceRefusalCode.INVALID_JSON, f"{name} JSON invalid: {exc}"
        )
    document = _object(value, _OUTER_KEYS, f"{name} document")
    if _canonical(document) != data:
        _refuse(
            MastMagneticSourceRefusalCode.NONCANONICAL_BYTES,
            f"{name} is not unique canonical JSON",
        )
    return document


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys while decoding JSON."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _refuse(
                MastMagneticSourceRefusalCode.DUPLICATE_JSON_KEY, f"duplicate key {key}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    """Reject non-finite numeric constants while decoding JSON."""
    _refuse(MastMagneticSourceRefusalCode.INVALID_JSON, f"nonfinite constant {value}")


def _require_schema(
    document: dict[str, object], schema: str, version: str, name: str
) -> None:
    """Require the expected document schema and version."""
    if document["schema"] != schema or document["schema_version"] != version:
        _refuse(
            MastMagneticSourceRefusalCode.UNSUPPORTED_SCHEMA,
            f"unsupported {name} schema or version",
        )


def _validate_payload_digest(
    document: dict[str, object], payload: dict[str, object], name: str
) -> str:
    """Validate a document digest against its canonical payload."""
    digest = _matching_text(
        document["payload_sha256"], _SHA256, f"{name} payload digest"
    )
    if digest != _sha256(_canonical(payload)):
        _refuse(
            MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH,
            f"{name} payload digest mismatch",
        )
    return digest


def _canonical_document(payload: dict[str, object], schema: str, version: str) -> bytes:
    """Encode a schema-bound document as byte-canonical JSON."""
    return _canonical(
        {
            "payload": payload,
            "payload_sha256": _sha256(_canonical(payload)),
            "schema": schema,
            "schema_version": version,
        }
    )


def _require_source_identity(revision: str, digest: str) -> None:
    """Require an immutable source revision and artifact digest."""
    if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
        _refuse(
            MastMagneticSourceRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "source revision must be a complete Git SHA",
        )
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _refuse(
            MastMagneticSourceRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "source artifact digest must be lowercase SHA-256",
        )


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    """Require an object with exactly the expected keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _qualification_refusal(f"{name} must be an object")
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _qualification_refusal(f"{name} keys differ from contract")
    return result


def _array(value: object, name: str) -> list[object]:
    """Require an array value."""
    if not isinstance(value, list):
        _qualification_refusal(f"{name} must be an array")
    return cast(list[object], value)


def _text(parent: dict[str, object], key: str) -> str:
    """Require a text field from the supplied record."""
    value = parent[key]
    if not isinstance(value, str) or not value:
        _qualification_refusal(f"{key} must be non-empty text")
    return value


def _matching_text(value: object, pattern: re.Pattern[str], name: str) -> str:
    """Require text matching the supplied pattern."""
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _qualification_refusal(f"{name} has invalid syntax")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    """Require a sequence of non-empty text values."""
    raw = _array(value, name)
    if not all(isinstance(item, str) and item for item in raw):
        _qualification_refusal(f"{name} must contain non-empty strings")
    return tuple(cast(list[str], raw))


def _shape(value: object, name: str) -> tuple[int, ...]:
    """Require a positive integer array shape."""
    raw = _array(value, name)
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in raw
    ):
        _qualification_refusal(f"{name} must contain non-negative integers")
    return tuple(cast(list[int], raw))


def _is_number(value: object) -> TypeGuard[int | float]:
    """Report whether a value is a finite real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, name: str) -> float:
    """Require a finite real number."""
    if not _is_number(value) or not math.isfinite(float(value)):
        _qualification_refusal(f"{name} must be finite")
    return float(value)


def _positive_number(value: object, name: str) -> float:
    """Require a strictly positive finite number."""
    result = _finite_number(value, name)
    if result <= 0.0:
        _qualification_refusal(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str) -> int:
    """Require a strictly positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _qualification_refusal(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    """Require a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _qualification_refusal(f"{name} must be a non-negative integer")
    return value


def _require_equal(value: object, expected: object, name: str) -> None:
    """Require exact equality with the governed value."""
    if value != expected:
        _qualification_refusal(f"{name} differs from contract")


def _clock_record(review: MastMagneticClockReview) -> dict[str, object]:
    """Serialize one reviewed clock into its canonical record."""
    return {
        "archive_grid_reproduced": review.archive_grid_reproduced,
        "first_value_s": review.first_value_s,
        "instrument_clock_relation_claimed": review.instrument_clock_relation_claimed,
        "last_value_s": review.last_value_s,
        "mapping_evidence_claimed": review.mapping_evidence_claimed,
        "name": review.name,
        "sample_count": review.sample_count,
        "spo_clock_kind_candidate": review.spo_clock_kind_candidate.value,
        "step_s": review.step_s,
    }


def _measurement_record(review: MastMagneticMeasurementReview) -> dict[str, object]:
    """Serialize one reviewed measurement into its canonical record."""
    return {
        "applied_transform_recorded": review.applied_transform_recorded,
        "array_name": review.array_name,
        "calibration_lineage_available": review.calibration_lineage_available,
        "channel_count": review.channel_count,
        "clock_name": review.clock_name,
        "observation_operator_available": review.observation_operator_available,
        "phase_eligible": review.phase_eligible,
        "provider_quality_flags_supplied": review.provider_quality_flags_supplied,
        "source_valid_for_shot": review.source_valid_for_shot,
        "uncertainty_supplied": review.uncertainty_supplied,
        "units": review.units,
    }


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


def _refuse(code: MastMagneticSourceRefusalCode, detail: str) -> NoReturn:
    """Raise the typed refusal for this contract."""
    raise MastMagneticSourceRefusalError(code, detail)


def _archive_refusal(detail: str) -> NoReturn:
    """Raise a MAST archive refusal."""
    _refuse(MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH, detail)


def _qualification_refusal(detail: str) -> NoReturn:
    """Raise a MAST qualification refusal."""
    _refuse(MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH, detail)


def _cross_refusal(detail: str) -> NoReturn:
    """Raise a cross-document consistency refusal."""
    _refuse(MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH, detail)


def _authority_refusal(detail: str) -> NoReturn:
    """Raise an authority-boundary refusal."""
    _refuse(MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION, detail)


__all__ = [
    "MAST_MAGNETIC_SOURCE_REVIEW_SCHEMA",
    "MAST_MAGNETIC_SOURCE_REVIEW_VERSION",
    "MAX_MAST_MAGNETIC_SOURCE_BYTES",
    "MAX_MAST_MAGNETIC_SOURCE_REVIEW_BYTES",
    "MastMagneticClockReview",
    "MastMagneticMeasurementReview",
    "MastMagneticSourceRefusal",
    "MastMagneticSourceRefusalError",
    "MastMagneticSourceRefusalCode",
    "MastMagneticSourceReview",
    "mast_magnetic_source_review_digest",
    "mast_magnetic_source_review_from_bytes",
    "mast_magnetic_source_review_from_producer_bytes",
    "mast_magnetic_source_review_from_record",
    "mast_magnetic_source_review_to_bytes",
    "mast_magnetic_source_review_to_record",
]
