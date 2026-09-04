# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FRC-compression MIF physical request tests

"""Canonical replay and fail-closed tests for the L1 producer request."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from scpn_phase_orchestrator import reactor_semantics as rs

REQUEST_ID = "c7002da6adee357f85b16bae94f1feee97804e58d15d8e57923821f39886e925"
REQUEST_SHA256 = "fe47d835b83ba0838222f5218967ef81a815add916b0ea81729cb491cc2eec41"
SEMANTIC_REGISTRY_SHA256 = (
    "270ed1ecbabe09cc45b078504c575ce8a77f0f6416378640140d2dc281951063"
)
OBSERVABILITY_REGISTRY_SHA256 = (
    "0aaf9bc7234113bedb98de51f2acd124a21da579e4d1ab1234e5b30ebc7880e0"
)
SCHEMA = Path("docs/specs/frc_compression_mif_physical_payload_request.schema.json")


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


def _document() -> dict[str, object]:
    request = rs.frc_compression_mif_physical_payload_request()
    return json.loads(rs.frc_compression_mif_physical_payload_request_to_bytes(request))


def _reseal(document: dict[str, object]) -> bytes:
    payload = document["payload"]
    document["payload_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return _canonical(document)


def test_request_binds_exact_l1_adapter_and_registered_candidates() -> None:
    request = rs.frc_compression_mif_physical_payload_request()

    assert request.request_id == REQUEST_ID
    assert request.requested_owner_project == "SCPN-MIF-CORE"
    assert request.device_project == "SCPN-MIF-CORE"
    assert request.configuration == "frc_compression_mif"
    assert request.intake_lane == "L1_extend_exercised_review_adapter"
    assert request.semantic_profile_registry_sha256 == SEMANTIC_REGISTRY_SHA256
    assert request.observability_registry_sha256 == OBSERVABILITY_REGISTRY_SHA256
    assert request.current_adapter == rs.FRCCompressionMIFAdapterBinding(
        ingress_state="verified_review_adapter",
        producer_project="SCPN-MIF-CORE",
        source_schema="scpn-mif-core.merge-compression-observation.v1",
        adapter_api=(
            "scpn_phase_orchestrator.reactor_semantics."
            "mif_merge_compression_handoff_from_mif_bytes"
        ),
        handoff_schema="scpn-phase-orchestrator.mif-merge-compression-handoff.v1",
        semantic_profile="spo.reactor.frc_compression_mif.merge_compression_review.v1",
        semantic_profile_version="1.0.0",
    )
    assert request.current_adapter.source_kind == "simulation"
    assert request.current_adapter.physical_source_present is False
    assert request.current_adapter.reusable_as_physical_evidence is False
    assert [item.candidate_id for item in request.candidate_requirements] == [
        "magneto_inertial.driver_arrival",
        "magneto_inertial.resolved_asymmetry_mode",
        "magneto_inertial.translation_and_compression",
        "model.synthetic_oscillator_coordinate",
    ]
    assert all(not item.evidence_claimed for item in request.candidate_requirements)


def test_request_names_every_missing_physical_evidence_obligation() -> None:
    request = rs.frc_compression_mif_physical_payload_request()

    assert [item.requirement_id.value for item in request.requirements] == [
        "physical_sample_identity",
        "configuration_specific_diagnostic_identity",
        "phenomenon_identity",
        "physical_reference_identity",
        "physical_clock_epoch_correlation",
        "observation_operator_or_calibration",
        "uncertainty",
        "validity",
        "plant_truth_state_semantics",
        "quality",
        "provenance_and_reproducibility",
        "observability_gate",
        "independent_validation",
    ]
    assert all(item.missing for item in request.requirements)
    assert all(
        item.immutable_artifact_binding_required for item in request.requirements
    )


def test_request_requires_distinct_producer_owned_plant_truth_states() -> None:
    request = rs.frc_compression_mif_physical_payload_request()

    assert request.required_distinct_plant_truth_states == (
        *(item.value for item in rs.ProducerEvidenceDisposition),
    )
    assert request.plant_truth_state_contract_required is True
    assert request.plant_truth_state_contract_present is False
    assert request.quality_state_may_substitute_for_plant_truth_state is False

    obligation = next(
        item
        for item in request.requirements
        if item.requirement_id.value == "plant_truth_state_semantics"
    )
    for required_state in request.required_distinct_plant_truth_states:
        assert required_state in obligation.acceptance_condition
    assert "orthogonal" in obligation.acceptance_condition
    assert "unclassified UNKNOWN regime" in obligation.acceptance_condition
    assert "none is a physical reactor-regime label" in obligation.acceptance_condition
    assert obligation.missing is True


def test_request_authority_is_exhaustively_fail_closed() -> None:
    request = rs.frc_compression_mif_physical_payload_request()

    assert request.selected_candidate_id is None
    assert request.physical_payload_schema_allocated is False
    assert request.physical_source_present is False
    assert request.plant_truth_state_contract_present is False
    assert request.observation_admitted is False
    assert request.phase_inference_eligible is False
    assert request.phase_inference_performed is False
    assert request.semantic_ingress_extended is False
    assert request.control_admission_requested is False
    assert request.control_intent_created is False
    assert request.qualification_state == "blocked_missing_physical_producer_payload"
    assert request.actionable is False
    assert request.execution_permitted is False
    assert request.direct_actuation is False
    assert request.review_only is True
    assert request.machine_protection_final_veto is True


def test_request_bytes_are_canonical_sealed_replayable_and_schema_valid() -> None:
    request = rs.frc_compression_mif_physical_payload_request()
    encoded = rs.frc_compression_mif_physical_payload_request_to_bytes(request)

    assert encoded.endswith(b"\n")
    assert len(encoded) == 9807
    assert hashlib.sha256(encoded).hexdigest() == REQUEST_SHA256
    assert rs.frc_compression_mif_physical_payload_request_digest(request) == (
        REQUEST_SHA256
    )
    assert (
        rs.frc_compression_mif_physical_payload_request_from_bytes(
            encoded, expected_sha256=REQUEST_SHA256
        )
        == request
    )
    document = json.loads(encoded)
    assert (
        document["payload_sha256"]
        == hashlib.sha256(_canonical(document["payload"])).hexdigest()
    )
    jsonschema.validate(document, json.loads(SCHEMA.read_text()))


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (
            lambda payload: payload.__setitem__("selected_candidate_id", "x"),
            "stored request differs",
        ),
        (
            lambda payload: payload["current_adapter"].__setitem__(
                "reusable_as_physical_evidence", True
            ),
            "stored request differs",
        ),
        (
            lambda payload: payload["requirements"][0].__setitem__("missing", False),
            "stored request differs",
        ),
    ],
)
def test_resealed_semantic_tampering_is_refused(mutate, detail: str) -> None:
    document = deepcopy(_document())
    mutate(document["payload"])

    with pytest.raises(
        rs.FRCCompressionMIFPhysicalPayloadRequestRefusalError,
        match=detail,
    ):
        rs.frc_compression_mif_physical_payload_request_from_bytes(_reseal(document))


def test_noncanonical_duplicate_unsupported_and_wrong_digest_inputs_are_refused() -> (
    None
):
    encoded = rs.frc_compression_mif_physical_payload_request_to_bytes(
        rs.frc_compression_mif_physical_payload_request()
    )
    document = _document()
    document["schema_version"] = "9.9.9"
    unsupported = _canonical(document)

    with pytest.raises(
        rs.FRCCompressionMIFPhysicalPayloadRequestRefusalError,
        match="noncanonical_bytes",
    ):
        rs.frc_compression_mif_physical_payload_request_from_bytes(
            json.dumps(json.loads(encoded), indent=2).encode()
        )
    with pytest.raises(
        rs.FRCCompressionMIFPhysicalPayloadRequestRefusalError,
        match="duplicate_json_key",
    ):
        rs.frc_compression_mif_physical_payload_request_from_bytes(
            b'{"payload":{},"payload":{},"payload_sha256":"x",'
            b'"schema":"x","schema_version":"x"}\n'
        )
    with pytest.raises(
        rs.FRCCompressionMIFPhysicalPayloadRequestRefusalError,
        match="unsupported_schema",
    ):
        rs.frc_compression_mif_physical_payload_request_from_bytes(unsupported)
    with pytest.raises(
        rs.FRCCompressionMIFPhysicalPayloadRequestRefusalError,
        match="envelope digest mismatch",
    ):
        rs.frc_compression_mif_physical_payload_request_from_bytes(
            encoded, expected_sha256="0" * 64
        )
