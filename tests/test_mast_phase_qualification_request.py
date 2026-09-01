# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FAIR-MAST phase qualification request tests
"""Exercise the physical-source review to producer-prerequisite boundary."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

FIXTURES = Path("tests/fixtures/mast_magnetic_source_review")
SOURCE_REVISION = "c30fb3932b47a812dc26d5846761030cdd0bc94c"
SOURCE_WHEEL_SHA256 = "a709b8aeecbd9483254bc3df1b29b87bf9df59ada92255af41631d861db430c9"
EXPECTED_REQUIREMENTS = (
    "phenomenon_identity",
    "reproducible_source_ingestion_state",
    "calibration_lineage",
    "physical_geometry_and_frame_join",
    "modal_observation_operator_and_harmonic_basis",
    "provider_quality",
    "uncertainty",
    "validity",
    "instrument_facility_clock_correlation",
    "resolved_event_identity",
    "observability_threshold",
    "independent_multi_shot_or_classifier_evidence",
)
EXPECTED_CANDIDATES = (
    "closed.equilibrium_profiles",
    "closed.recurrent_transient",
    "closed.resolved_mhd_mode",
    "model.synthetic_oscillator_coordinate",
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


def _review() -> rs.MastMagneticSourceReview:
    return rs.mast_magnetic_source_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=SOURCE_WHEEL_SHA256,
        archive_bytes=(FIXTURES / "MAGNETIC_ARCHIVE_ENVELOPE.json").read_bytes(),
        qualification_bytes=(
            FIXTURES / "MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
        ).read_bytes(),
    )


def _request() -> rs.MastPhaseQualificationRequest:
    return rs.mast_phase_qualification_request_from_source_review(_review())


def _reseal(document: dict[str, Any]) -> bytes:
    document["payload_sha256"] = hashlib.sha256(
        _canonical(document["payload"])
    ).hexdigest()
    return _canonical(document)


def test_request_binds_exact_source_review_and_registry_candidates() -> None:
    review = _review()
    request = rs.mast_phase_qualification_request_from_source_review(review)

    assert request.requested_owner_project == "SCPN-FUSION-CORE"
    assert request.device_project == "SCPN-TOKAMAK-CORE"
    assert request.configuration == "spherical_tokamak"
    assert request.facility == "MAST"
    assert request.source_archive == "FAIR-MAST"
    assert request.shot_id == 27707
    assert request.source_review_id == review.review_id
    assert request.source_review_sha256 == rs.mast_magnetic_source_review_digest(review)
    assert request.source_revision == SOURCE_REVISION
    assert request.source_artifact_sha256 == SOURCE_WHEEL_SHA256
    assert request.source_archive_sha256 == review.source_archive_sha256
    assert request.source_qualification_sha256 == review.source_qualification_sha256
    assert request.archive_payload_sha256 == review.archive_payload_sha256
    assert request.qualification_payload_sha256 == review.qualification_payload_sha256
    assert request.source_ingestion_revision == review.source_ingestion_revision
    assert request.source_ingestion_tree_state == "dirty"
    assert request.source_review_unresolved_fields == (
        review.unresolved_qualification_fields
    )
    assert request.observability_registry_version == "1.0.0"
    assert (
        request.observability_registry_sha256
        == rs.DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
    )
    assert tuple(item.candidate_id for item in request.candidate_requirements) == (
        EXPECTED_CANDIDATES
    )
    assert all(not item.evidence_claimed for item in request.candidate_requirements)
    assert request.selected_candidate_id is None
    assert request.phenomenon_identity_state == (
        "unresolved_producer_evidence_required"
    )


def test_request_names_every_missing_qualification_requirement() -> None:
    request = _request()

    assert tuple(item.requirement_id.value for item in request.requirements) == (
        EXPECTED_REQUIREMENTS
    )
    assert all(
        item.missing
        and item.immutable_artifact_binding_required
        and item.evidence_subject
        and item.acceptance_condition
        for item in request.requirements
    )


def test_request_never_promotes_phase_semantic_or_control_authority() -> None:
    request = _request()

    assert request.qualification_state == "blocked_missing_producer_evidence"
    assert not request.observation_admitted
    assert not request.phase_inference_eligible
    assert not request.phase_inference_performed
    assert not request.semantic_ingress_declared
    assert not request.control_admission_requested
    assert not request.control_intent_created
    assert not request.actionable
    assert not request.execution_permitted
    assert not request.direct_actuation
    assert request.review_only
    assert request.machine_protection_final_veto


def test_request_is_byte_canonical_digest_sealed_and_round_trips() -> None:
    request = _request()
    record = rs.mast_phase_qualification_request_to_record(request)
    encoded = rs.mast_phase_qualification_request_to_bytes(request)

    assert encoded.endswith(b"\n")
    assert encoded == _canonical(json.loads(encoded))
    assert rs.mast_phase_qualification_request_from_record(record) == request
    assert rs.mast_phase_qualification_request_from_bytes(encoded) == request
    assert rs.mast_phase_qualification_request_digest(request) == (
        hashlib.sha256(encoded).hexdigest()
    )
    assert len(request.request_id) == 64


def test_portable_request_matches_its_published_json_schema() -> None:
    schema = json.loads(
        Path("docs/specs/mast_phase_qualification_request.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        json.loads(rs.mast_phase_qualification_request_to_bytes(_request()))
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qualification_state", "qualified"),
        ("selected_candidate_id", "closed.resolved_mhd_mode"),
        ("observation_admitted", True),
        ("phase_inference_eligible", True),
        ("semantic_ingress_declared", True),
        ("control_admission_requested", True),
        ("actionable", True),
        ("execution_permitted", True),
        ("direct_actuation", True),
        ("review_only", False),
        ("machine_protection_final_veto", False),
    ],
)
def test_reconstruction_refuses_stored_authority_or_scientific_promotion(
    field: str, value: object
) -> None:
    record = rs.mast_phase_qualification_request_to_record(_request())
    record[field] = value

    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_record(record)

    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )


def test_reconstruction_refuses_missing_requirement_or_candidate() -> None:
    for field in ("requirements", "candidate_requirements"):
        record = rs.mast_phase_qualification_request_to_record(_request())
        cast(list[object], record[field]).pop()
        with pytest.raises(rs.MastPhaseQualificationRequestRefusalError):
            rs.mast_phase_qualification_request_from_record(record)


def test_envelope_refuses_digest_schema_and_noncanonical_drift() -> None:
    document = json.loads(rs.mast_phase_qualification_request_to_bytes(_request()))

    bad_digest = deepcopy(document)
    bad_digest["payload_sha256"] = "0" * 64
    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_bytes(_canonical(bad_digest))
    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )

    bad_schema = deepcopy(document)
    bad_schema["schema_version"] = "2.0.0"
    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_bytes(_canonical(bad_schema))
    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.UNSUPPORTED_SCHEMA
    )

    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_bytes(
            b" " + rs.mast_phase_qualification_request_to_bytes(_request())
        )
    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.NONCANONICAL_BYTES
    )


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (cast(bytes, "text"), "invalid_input"),
        (b"", "invalid_input"),
        (b"\xff", "invalid_json"),
        (b"{", "invalid_json"),
        (b'{"payload":1,"payload":2}\n', "duplicate_json_key"),
    ],
)
def test_request_byte_boundary_refuses_invalid_inputs(data: bytes, code: str) -> None:
    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_bytes(data)
    assert caught.value.code.value == code


def test_request_byte_boundary_refuses_oversized_input() -> None:
    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_bytes(
            b"x" * (rs.MAX_MAST_PHASE_QUALIFICATION_REQUEST_BYTES + 1)
        )
    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.INVALID_INPUT
    )


def test_embedded_source_review_drift_is_refused() -> None:
    record = rs.mast_phase_qualification_request_to_record(_request())
    source = json.loads(cast(str, record["source_review_json"]))
    source["payload"]["observation_admitted"] = True
    source["payload_sha256"] = hashlib.sha256(_canonical(source["payload"])).hexdigest()
    record["source_review_json"] = _canonical(source).decode("utf-8")

    with pytest.raises(rs.MastPhaseQualificationRequestRefusalError) as caught:
        rs.mast_phase_qualification_request_from_record(record)
    assert caught.value.code is (
        rs.MastPhaseQualificationRequestRefusalCode.SOURCE_REVIEW_MISMATCH
    )
