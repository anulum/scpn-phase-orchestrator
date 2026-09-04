# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — diagnostic-plan physical-evidence request tests
"""Exercise accepted producer-plan review to physical-evidence request."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

FIXTURES = Path("tests/fixtures/icf_laser_diagnostic_plan")
MATERIALISED_REQUEST = Path(
    "docs/reference/data/laser_icf_direct_drive_physical_evidence_request.v1.json"
)
SCHEMA = Path("docs/specs/device_physical_evidence_request.schema.json")
SOURCE_REVISION = "a0ad63207c9aff5f00273e2c37fa580feb9d6c38"
SOURCE_ARTIFACT_SHA256 = (
    "e480e4a2c67b45dc04ef04cebd44ddcb3a6a3efc24739f54606d25cd0661bde3"
)
CONFIGURATION = "laser_icf_direct_drive"


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


def _review() -> rs.DeviceDiagnosticPlanReview:
    fixture = json.loads((FIXTURES / "plan_envelope_fixture.json").read_bytes())
    manifest = json.loads((FIXTURES / "reactor-domain.json").read_bytes())
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    plan_bytes = _canonical(fixture["plan"])
    envelope = fixture["envelope"]
    envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["plan_identifier"] = fixture["plan"]["identifier"]
    envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    return rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=SOURCE_ARTIFACT_SHA256,
        manifest_bytes=manifest_bytes,
        envelope_bytes=_canonical(envelope),
        plan_bytes=plan_bytes,
    )


def _request() -> rs.DevicePhysicalEvidenceRequest:
    return rs.device_physical_evidence_request_from_plan_review(
        _review(), configuration=CONFIGURATION
    )


def _reseal(document: dict[str, Any]) -> bytes:
    document["payload_sha256"] = hashlib.sha256(
        _canonical(document["payload"])
    ).hexdigest()
    return _canonical(document)


def test_request_binds_exact_plan_review_and_configuration() -> None:
    review = _review()
    request = rs.device_physical_evidence_request_from_plan_review(
        review, configuration=CONFIGURATION
    )

    assert request.requested_owner_project == "SCPN-ICF-LASER-CORE"
    assert request.device_project == "SCPN-ICF-LASER-CORE"
    assert request.configuration == CONFIGURATION
    assert request.source_review_id == review.review_id
    assert request.source_review_sha256 == rs.device_diagnostic_plan_review_digest(
        review
    )
    assert request.source_revision == SOURCE_REVISION
    assert request.source_artifact_sha256 == SOURCE_ARTIFACT_SHA256
    assert request.producer_package_revision == "0.1.0.dev0"
    assert request.plan_identifier == "icf_laser_reference_plan"
    assert request.source_manifest_sha256 == review.source_manifest_sha256
    assert request.source_envelope_sha256 == review.source_envelope_sha256
    assert request.source_plan_sha256 == review.source_plan_sha256
    assert request.source_reactor_registry_version == "1.0.0"
    assert request.source_reactor_registry_digest == (
        "786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090"
    )
    assert request.reactor_registry_version == "1.1.0"
    assert request.reactor_registry_digest == rs.DEFAULT_REACTOR_REGISTRY.digest
    assert request.observability_registry_version == "1.1.0"
    assert (
        request.observability_registry_digest
        == rs.DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
    )
    assert (
        request.semantic_profile_registry_digest
        == rs.DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
    )


def test_candidate_meanings_preserve_physical_and_numerical_boundaries() -> None:
    by_id = {item.candidate_id: item for item in _request().candidate_requirements}

    assert tuple(by_id) == (
        "inertial.driver_timing",
        "inertial.implosion_trajectory",
        "inertial.resolved_asymmetry_mode",
        "inertial.shot_outcome",
        "model.synthetic_oscillator_coordinate",
    )
    assert (
        by_id["inertial.driver_timing"].observability_class,
        by_id["inertial.driver_timing"].declared_carriers,
        by_id["inertial.driver_timing"].channel_identifiers,
    ) == ("event_relative", ("event_cycle",), ("ch_beam_timing_train",))
    assert by_id["inertial.driver_timing"].repeated_cycle_required
    assert by_id["inertial.driver_timing"].physical_selection_eligible

    assert (
        by_id["inertial.resolved_asymmetry_mode"].observability_class,
        by_id["inertial.resolved_asymmetry_mode"].declared_carriers,
    ) == ("derived_cyclic", ("complex_mode",))
    assert by_id["inertial.resolved_asymmetry_mode"].observation_operator_required
    assert by_id["inertial.resolved_asymmetry_mode"].physical_selection_eligible

    for candidate_id in (
        "inertial.implosion_trajectory",
        "inertial.shot_outcome",
    ):
        assert by_id[candidate_id].observability_class == "noncyclic_feature"
        assert by_id[candidate_id].physical_selection_eligible

    numerical = by_id["model.synthetic_oscillator_coordinate"]
    assert numerical.observability_class == "numerical_only"
    assert numerical.declared_carriers == ("numerical_phase",)
    assert not numerical.physical_selection_eligible
    assert not numerical.plan_revision_required
    assert numerical.synthetic_declaration_only
    assert not numerical.physical_sample_present
    assert not numerical.evidence_claimed
    assert not numerical.observation_claimed


def test_plan_clocks_remain_uncorrelated_design_declarations() -> None:
    clocks = {
        item.plan_clock_identifier: item for item in _request().clock_requirements
    }

    assert tuple(clocks) == ("clk_facility", "clk_shot", "clk_sim")
    assert clocks["clk_facility"].compatibility == "unmapped"
    assert clocks["clk_facility"].physical_correlation_required
    assert clocks["clk_facility"].eligible_for_physical_reference
    assert clocks["clk_shot"].compatibility == "event_relative_compatible"
    assert clocks["clk_shot"].physical_correlation_required
    assert clocks["clk_shot"].eligible_for_physical_reference
    assert clocks["clk_sim"].compatibility == "synthetic_compatible"
    assert not clocks["clk_sim"].physical_correlation_required
    assert not clocks["clk_sim"].eligible_for_physical_reference
    assert not any(item.mapping_evidence_claimed for item in clocks.values())


def test_request_names_all_missing_physical_evidence_obligations() -> None:
    request = _request()

    assert tuple(item.requirement_id.value for item in request.requirements) == (
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
    )
    assert all(
        item.missing
        and item.immutable_artifact_binding_required
        and item.evidence_subject
        and item.acceptance_condition
        for item in request.requirements
    )


def test_request_pins_evidence_dispositions_without_regime_promotion() -> None:
    request = _request()

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
    assert request.producer_evidence_state_contract_required
    assert not request.producer_evidence_state_contract_present
    assert not request.quality_state_may_substitute_for_evidence_state


def test_request_authority_is_exhaustively_fail_closed() -> None:
    request = _request()

    assert request.diagnostic_plan_accepted
    assert not request.diagnostic_plan_is_physical_evidence
    assert not request.physical_payload_schema_allocated
    assert not request.physical_source_present
    assert request.selected_candidate_id is None
    assert not request.observation_admitted
    assert not request.phase_inference_eligible
    assert not request.phase_inference_performed
    assert not request.semantic_ingress_declared
    assert not request.control_admission_requested
    assert not request.control_intent_created
    assert request.qualification_state == ("blocked_missing_physical_producer_evidence")
    assert not request.actionable
    assert not request.execution_permitted
    assert not request.direct_actuation
    assert request.authority == "review_only"
    assert request.machine_protection_final_veto


def test_request_is_canonical_sealed_replayable_and_schema_valid() -> None:
    request = _request()
    encoded = rs.device_physical_evidence_request_to_bytes(request)
    schema = json.loads(SCHEMA.read_bytes())

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))
    assert encoded == _canonical(json.loads(encoded))
    assert (
        rs.device_physical_evidence_request_digest(request)
        == hashlib.sha256(encoded).hexdigest()
    )
    assert (
        rs.device_physical_evidence_request_from_bytes(
            encoded, expected_sha256=hashlib.sha256(encoded).hexdigest()
        )
        == request
    )
    assert (
        rs.device_physical_evidence_request_from_record(
            rs.device_physical_evidence_request_to_record(request)
        )
        == request
    )


def test_materialised_direct_drive_request_matches_runtime_exactly() -> None:
    encoded = rs.device_physical_evidence_request_to_bytes(_request())

    assert MATERIALISED_REQUEST.read_bytes() == encoded
    assert (
        rs.device_physical_evidence_request_from_bytes(
            MATERIALISED_REQUEST.read_bytes()
        )
        == _request()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selected_candidate_id", "inertial.driver_timing"),
        ("diagnostic_plan_is_physical_evidence", True),
        ("physical_source_present", True),
        ("phase_inference_eligible", True),
        ("semantic_ingress_declared", True),
        ("control_admission_requested", True),
        ("actionable", True),
        ("direct_actuation", True),
        ("machine_protection_final_veto", False),
        ("producer_evidence_state_contract_required", False),
    ),
)
def test_record_reconstruction_refuses_authority_or_scientific_promotion(
    field: str,
    value: object,
) -> None:
    record = rs.device_physical_evidence_request_to_record(_request())
    record[field] = value

    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_record(record)
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )


def test_candidate_reclassification_and_numerical_promotion_are_refused() -> None:
    for field, value in (
        ("observability_class", "direct_cyclic"),
        ("physical_selection_eligible", True),
        ("synthetic_declaration_only", False),
    ):
        record = rs.device_physical_evidence_request_to_record(_request())
        candidates = cast(list[dict[str, object]], record["candidate_requirements"])
        numerical = next(
            item
            for item in candidates
            if item["candidate_id"] == "model.synthetic_oscillator_coordinate"
        )
        numerical[field] = value
        with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
            rs.device_physical_evidence_request_from_record(record)


def test_embedded_review_authority_drift_is_refused() -> None:
    record = rs.device_physical_evidence_request_to_record(_request())
    review_document = json.loads(cast(str, record["source_review_json"]))
    review_document["payload"]["evidence_claimed"] = True
    review_document["payload_sha256"] = hashlib.sha256(
        _canonical(review_document["payload"])
    ).hexdigest()
    record["source_review_json"] = _canonical(review_document).decode("utf-8")

    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_record(record)
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH
    )


@pytest.mark.parametrize(
    "configuration",
    ("ion_beam_icf", "scpn.reactor_systems:lattice_confinement_fusion", ""),
)
def test_unowned_or_invalid_configuration_is_refused(configuration: str) -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_plan_review(
            _review(), configuration=configuration
        )
    assert caught.value.code in {
        rs.DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT,
        rs.DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
    }


def test_public_factory_requires_a_real_review() -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_plan_review(
            cast(rs.DeviceDiagnosticPlanReview, object()),
            configuration=CONFIGURATION,
        )
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT
    )


@pytest.mark.parametrize("source_review_json", ("", cast(str, object())))
def test_public_request_type_refuses_invalid_embedded_review_text(
    source_review_json: str,
) -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
        rs.DevicePhysicalEvidenceRequest(
            configuration=CONFIGURATION,
            source_review_json=source_review_json,
        )


@pytest.mark.parametrize("configuration", ("", cast(str, object())))
def test_public_request_type_refuses_invalid_configuration(
    configuration: str,
) -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
        rs.DevicePhysicalEvidenceRequest(
            configuration=configuration,
            source_review_json=rs.device_diagnostic_plan_review_to_bytes(
                _review()
            ).decode("utf-8"),
        )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        (
            "accepted_as_design_declaration",
            False,
            rs.DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
        ),
        (
            "configurations",
            ("not_registered",),
            rs.DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
        ),
        (
            "source_project",
            "SCPN-ICF-BEAM-CORE",
            rs.DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH,
        ),
        (
            "reactor_registry_version",
            "0.0.0",
            rs.DevicePhysicalEvidenceRequestRefusalCode.REGISTRY_BINDING_MISMATCH,
        ),
        (
            "deferred_candidate_ids",
            ("inertial.driver_timing",),
            rs.DevicePhysicalEvidenceRequestRefusalCode.SOURCE_REVIEW_MISMATCH,
        ),
    ),
)
def test_public_factory_refuses_corrupted_or_stale_review_objects(
    field: str,
    value: object,
    code: rs.DevicePhysicalEvidenceRequestRefusalCode,
) -> None:
    review = _review()
    object.__setattr__(review, field, value)
    configuration = "not_registered" if field == "configurations" else CONFIGURATION

    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_plan_review(
            review, configuration=configuration
        )
    assert caught.value.code is code


def test_public_factory_refuses_corrupted_candidate_semantics() -> None:
    review = _review()
    signal = replace(
        review.signal_reviews[0],
        carrier=rs.SemanticCarrier.BOUNDED_FEATURE,
    )
    object.__setattr__(review, "signal_reviews", (signal, *review.signal_reviews[1:]))

    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_plan_review(
            review, configuration=CONFIGURATION
        )
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.REGISTRY_BINDING_MISMATCH
    )


@pytest.mark.parametrize(
    "record",
    (
        None,
        {},
        {1: "not-a-string-key"},
    ),
)
def test_record_boundary_refuses_nonobjects_or_wrong_keys(record: object) -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
        rs.device_physical_evidence_request_from_record(record)


@pytest.mark.parametrize("field", ("configuration", "source_review_json"))
def test_record_boundary_refuses_nontext_core_fields(field: str) -> None:
    record = rs.device_physical_evidence_request_to_record(_request())
    record[field] = None

    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_record(record)
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )


def test_byte_boundary_refuses_malformed_noncanonical_and_resealed_drift() -> None:
    encoded = rs.device_physical_evidence_request_to_bytes(_request())
    document = json.loads(encoded)

    cases = (
        cast(bytes, "text"),
        b"",
        b"\xff",
        b"{",
        b'{"payload":{},"payload":{}}\n',
        b'{"payload":NaN}\n',
        b" " + encoded,
    )
    for value in cases:
        with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
            rs.device_physical_evidence_request_from_bytes(value)

    unsupported = deepcopy(document)
    unsupported["schema_version"] = "9.9.9"
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_bytes(_canonical(unsupported))
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.UNSUPPORTED_SCHEMA
    )

    promoted = deepcopy(document)
    promoted["payload"]["observation_admitted"] = True
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_bytes(_reseal(promoted))
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )

    wrong_payload_seal = deepcopy(document)
    wrong_payload_seal["payload_sha256"] = "0" * 64
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_bytes(_canonical(wrong_payload_seal))
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.REQUEST_CONTRACT_MISMATCH
    )


def test_byte_boundary_refuses_wrong_or_malformed_expected_digest() -> None:
    encoded = rs.device_physical_evidence_request_to_bytes(_request())

    for digest in ("0" * 64, "A" * 64, "short"):
        with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError):
            rs.device_physical_evidence_request_from_bytes(
                encoded, expected_sha256=digest
            )


def test_byte_boundary_refuses_oversized_input_before_decode() -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_bytes(
            b"x" * (rs.MAX_DEVICE_PHYSICAL_EVIDENCE_REQUEST_BYTES + 1)
        )
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.INVALID_INPUT
    )
