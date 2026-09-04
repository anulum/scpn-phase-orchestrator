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
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import jsonschema
import pytest

from scpn_phase_orchestrator import reactor_semantics as rs

REQUEST_ID = "fec4e93971190c7183410f200c60a9ef0ffcfeaf01fa69f9fc3514e9e352603c"
REQUEST_SHA256 = "a506c0ad7c37ee53719b3f2194906b39585e4293e1c5f9d25245f987c0b08945"
SEMANTIC_REGISTRY_SHA256 = (
    "270ed1ecbabe09cc45b078504c575ce8a77f0f6416378640140d2dc281951063"
)
OBSERVABILITY_REGISTRY_SHA256 = (
    "0aaf9bc7234113bedb98de51f2acd124a21da579e4d1ab1234e5b30ebc7880e0"
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


def test_registry_contract_drift_is_refused_through_public_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module(
        "scpn_phase_orchestrator.reactor_semantics."
        "conventional_tokamak_physical_payload_request"
    )
    registry = rs.DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
    profiles = dict(registry.profiles)
    profiles["conventional_tokamak"] = replace(
        profiles["conventional_tokamak"], semantic_profile_version="9.9.9"
    )
    drifted = rs.ReactorSemanticProfileRegistry(
        version=registry.version,
        reactor_registry_version=registry.reactor_registry_version,
        reactor_registry_digest=registry.reactor_registry_digest,
        assignment_map_sha256=registry.assignment_map_sha256,
        profiles=profiles,
    )
    monkeypatch.setattr(module, "DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY", drifted)

    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="semantic adapter binding changed",
    ):
        rs.conventional_tokamak_physical_payload_request()


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
        "producer_evidence_state_semantics",
        "quality",
        "provenance_and_reproducibility",
        "observability_gate",
        "independent_validation",
    ]
    assert all(item.missing for item in request.requirements)
    assert all(
        item.immutable_artifact_binding_required for item in request.requirements
    )


def test_request_pins_distinct_producer_evidence_states_and_regime_abstention() -> None:
    request = rs.conventional_tokamak_physical_payload_request()

    assert request.producer_evidence_state_contract_required is True
    assert request.producer_evidence_state_contract_present is False
    assert request.quality_state_may_substitute_for_evidence_state is False
    assert request.producer_evidence_state_policies == (
        rs.PRODUCER_EVIDENCE_STATE_POLICIES
    )
    assert tuple(
        (policy.disposition.value, policy.validity_state.value)
        for policy in request.producer_evidence_state_policies
    ) == (
        ("unknown", "unknown"),
        ("out_of_distribution", "out_of_distribution"),
        ("low_observability", "unobservable"),
        ("stale", "stale"),
    )
    assert all(
        policy.regime_state.value == "unknown"
        and not policy.physical_regime_classified
        and not policy.quality_may_substitute
        for policy in request.producer_evidence_state_policies
    )

    requirement = next(
        item
        for item in request.requirements
        if item.requirement_id.value == "producer_evidence_state_semantics"
    )
    for marker in (
        "unknown",
        "out_of_distribution",
        "low_observability",
        "stale",
        "quality",
        "U0 validity",
        "UNKNOWN physical regime",
    ):
        assert marker in requirement.acceptance_condition


def test_request_authority_is_exhaustively_fail_closed() -> None:
    request = rs.conventional_tokamak_physical_payload_request()

    assert request.selected_candidate_id is None
    assert request.physical_payload_schema_allocated is False
    assert request.physical_source_present is False
    assert request.producer_evidence_state_contract_present is False
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
    assert len(encoded) == 10767
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


def test_expected_digest_requires_lowercase_sha256_text() -> None:
    encoded = rs.conventional_tokamak_physical_payload_request_to_bytes(
        rs.conventional_tokamak_physical_payload_request()
    )

    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="expected_sha256 must be lowercase SHA-256 text",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(
            encoded, expected_sha256="A" * 64
        )


def test_payload_digest_mismatch_is_refused() -> None:
    document = _document()
    document["payload_sha256"] = "0" * 64

    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match="request payload digest mismatch",
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(
            _canonical(document)
        )


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
        (
            lambda payload: payload.__setitem__(
                "producer_evidence_state_contract_present", True
            ),
            "stored request differs",
        ),
        (
            lambda payload: payload.__setitem__(
                "quality_state_may_substitute_for_evidence_state", True
            ),
            "stored request differs",
        ),
        (
            lambda payload: payload["producer_evidence_state_policies"][2].__setitem__(
                "validity_state", "unknown"
            ),
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


@pytest.mark.parametrize(
    ("data", "detail"),
    (
        (b"", "request byte input invalid"),
        (b"\xff", "request JSON invalid"),
        (b'{"payload":NaN}\n', "nonfinite constant NaN"),
    ),
)
def test_request_byte_decoder_refuses_invalid_transport(
    data: bytes, detail: str
) -> None:
    with pytest.raises(
        rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
        match=detail,
    ):
        rs.conventional_tokamak_physical_payload_request_from_bytes(data)


def test_request_record_boundary_requires_exact_object_shape() -> None:
    for record, detail in (
        (None, "request payload must be an object"),
        ({}, "request payload keys differ from contract"),
    ):
        with pytest.raises(
            rs.ConventionalTokamakPhysicalPayloadRequestRefusalError,
            match=detail,
        ):
            rs.conventional_tokamak_physical_payload_request_from_record(record)
