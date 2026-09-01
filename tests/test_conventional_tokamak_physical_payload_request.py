# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Conventional-tokamak physical request tests

"""Canonical replay and fail-closed tests for the L1 producer request."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

from scpn_phase_orchestrator import reactor_semantics as rs

REQUEST_ID = "2102de134b9326a1c5eb280d3ac1bffd487ba55342a03f664550bf5afd4d24fd"
REQUEST_SHA256 = "4fa48cecbb4bc39bec49fd0411e71ac9219c4e033044878fc90f5a23823a0ad5"
SEMANTIC_REGISTRY_SHA256 = (
    "6ac7f3863e1a5f50af297c572ec0b80b60820a23de1a769fda6bb0a831243ec3"
)
OBSERVABILITY_REGISTRY_SHA256 = (
    "d70c0de696534e5a77066ef8420cf7ca17bc4d7321984b0ac83523dbc1dce609"
)
SCHEMA = Path("docs/specs/conventional_tokamak_physical_payload_request.schema.json")


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
    request = rs.conventional_tokamak_physical_payload_request()
    return json.loads(
        rs.conventional_tokamak_physical_payload_request_to_bytes(request)
    )


def _reseal(document: dict[str, object]) -> bytes:
    payload = document["payload"]
    document["payload_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return _canonical(document)


def test_request_binds_exact_l1_adapter_and_registered_candidates() -> None:
    request = rs.conventional_tokamak_physical_payload_request()

    assert request.request_id == REQUEST_ID
    assert request.requested_owner_project == "SCPN-FUSION-CORE"
    assert request.device_project == "SCPN-TOKAMAK-CORE"
    assert request.configuration == "conventional_tokamak"
    assert request.intake_lane == "L1_extend_exercised_review_adapter"
    assert request.semantic_profile_registry_sha256 == SEMANTIC_REGISTRY_SHA256
    assert request.observability_registry_sha256 == OBSERVABILITY_REGISTRY_SHA256
    assert request.current_adapter == rs.ConventionalTokamakAdapterBinding(
        ingress_state="verified_review_adapter",
        producer_project="SCPN-FUSION-CORE",
        source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
        adapter_api=(
            "scpn_phase_orchestrator.reactor_semantics."
            "coupled_transport_handoff_from_fusion_bytes"
        ),
        handoff_schema="scpn-phase-orchestrator.reactor-semantic-handoff.v1",
        semantic_profile=(
            "spo.reactor.conventional_tokamak.coupled_transport.nonphase_review.v1"
        ),
        semantic_profile_version="1.0.0",
    )
    assert request.current_adapter.source_kind == "simulation"
    assert request.current_adapter.physical_source_present is False
    assert request.current_adapter.reusable_as_physical_evidence is False
    assert [item.candidate_id for item in request.candidate_requirements] == [
        "closed.equilibrium_profiles",
        "closed.recurrent_transient",
        "closed.resolved_mhd_mode",
        "model.synthetic_oscillator_coordinate",
    ]
    assert all(not item.evidence_claimed for item in request.candidate_requirements)


def test_request_names_every_missing_physical_evidence_obligation() -> None:
    request = rs.conventional_tokamak_physical_payload_request()

    assert [item.requirement_id.value for item in request.requirements] == [
        "physical_sample_identity",
        "configuration_specific_diagnostic_identity",
        "phenomenon_identity",
        "physical_reference_identity",
        "physical_clock_epoch_correlation",
        "observation_operator_or_calibration",
        "uncertainty",
        "validity",
        "quality",
        "provenance_and_reproducibility",
        "observability_gate",
        "independent_validation",
    ]
    assert all(item.missing for item in request.requirements)
    assert all(
        item.immutable_artifact_binding_required for item in request.requirements
    )


def test_request_authority_is_exhaustively_fail_closed() -> None:
    request = rs.conventional_tokamak_physical_payload_request()

    assert request.selected_candidate_id is None
    assert request.physical_payload_schema_allocated is False
    assert request.physical_source_present is False
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
    request = rs.conventional_tokamak_physical_payload_request()
    encoded = rs.conventional_tokamak_physical_payload_request_to_bytes(request)

    assert encoded.endswith(b"\n")
    assert len(encoded) == 8648
    assert hashlib.sha256(encoded).hexdigest() == REQUEST_SHA256
    assert rs.conventional_tokamak_physical_payload_request_digest(request) == (
        REQUEST_SHA256
    )
    assert (
        rs.conventional_tokamak_physical_payload_request_from_bytes(
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
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match=detail,
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(_reseal(document))


def test_noncanonical_duplicate_unsupported_and_wrong_digest_inputs_are_refused() -> (
    None
):
    encoded = rs.conventional_tokamak_physical_payload_request_to_bytes(
        rs.conventional_tokamak_physical_payload_request()
    )
    document = _document()
    document["schema_version"] = "9.9.9"
    unsupported = _canonical(document)

    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="noncanonical_bytes",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(
            json.dumps(json.loads(encoded), indent=2).encode()
        )
    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="duplicate_json_key",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(
            b'{"payload":{},"payload":{},"payload_sha256":"x",'
            b'"schema":"x","schema_version":"x"}\n'
        )
    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="unsupported_schema",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(unsupported)
    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="envelope digest mismatch",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(
            encoded, expected_sha256="0" * 64
        )
