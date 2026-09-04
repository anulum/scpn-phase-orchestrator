# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Device diagnostic-plan review tests
"""Exercise the public producer-bytes to design-review boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

FIXTURES = Path("tests/fixtures/tokamak_diagnostic_plan")
BEAM_TARGET_FIXTURES = Path("tests/fixtures/beam_target_diagnostic_plan")
SOURCE_REVISION = "ea2159d1607ebe0cce3059c3a6a8500968cc6f42"
ARTIFACT_SHA256 = "a" * 64


def _compact(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = json.loads((FIXTURES / "plan_envelope_fixture.json").read_text())
    manifest = json.loads((FIXTURES / "reactor-domain.json").read_text())
    return manifest, fixture["envelope"], fixture["plan"]


def _source_bytes(
    manifest: dict[str, Any],
    envelope: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[bytes, bytes, bytes]:
    manifest_bytes = _pretty(manifest)
    plan_bytes = _compact(plan)
    envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    envelope["plan_identifier"] = plan["identifier"]
    return manifest_bytes, _compact(envelope), plan_bytes


def _review(
    manifest: dict[str, Any] | None = None,
    envelope: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> rs.DeviceDiagnosticPlanReview:
    defaults = _records()
    source = _source_bytes(
        deepcopy(defaults[0] if manifest is None else manifest),
        deepcopy(defaults[1] if envelope is None else envelope),
        deepcopy(defaults[2] if plan is None else plan),
    )
    return rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=ARTIFACT_SHA256,
        manifest_bytes=source[0],
        envelope_bytes=source[1],
        plan_bytes=source[2],
    )


def _assert_refusal(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    manifest: dict[str, Any],
    envelope: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest, envelope, plan)
    assert caught.value.code is code


def _assert_plan_mutation(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest, envelope, plan = _records()
    mutate(plan)
    _assert_refusal(code, manifest, envelope, plan)


def test_exact_tokamak_fixture_produces_full_design_review() -> None:
    review = _review()

    assert review.source_project == "SCPN-TOKAMAK-CORE"
    assert review.source_revision == SOURCE_REVISION
    assert review.source_artifact_sha256 == ARTIFACT_SHA256
    assert review.producer_package_revision == "0.1.0.dev0"
    assert review.configurations == ("conventional_tokamak", "spherical_tokamak")
    assert review.source_reactor_registry_version == "1.0.0"
    assert review.source_reactor_registry_digest == (
        "786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090"
    )
    assert review.source_observability_registry_version == "1.0.0"
    assert review.source_observability_registry_digest == (
        "d70c0de696534e5a77066ef8420cf7ca17bc4d7321984b0ac83523dbc1dce609"
    )
    assert review.reactor_registry_version == "1.1.0"
    assert review.reactor_registry_digest == rs.DEFAULT_REACTOR_REGISTRY.digest
    assert review.planned_candidate_ids == (
        "closed.equilibrium_profiles",
        "closed.recurrent_transient",
        "closed.resolved_mhd_mode",
        "model.synthetic_oscillator_coordinate",
    )
    assert review.deferred_candidate_ids == ()
    assert review.frame_ids == ("frm_flux", "frm_machine")
    assert {
        (item.observability_class.value, item.carrier.value)
        for item in review.signal_reviews
    } == {
        ("derived_cyclic", "complex_mode"),
        ("event_relative", "event_cycle"),
        ("noncyclic_feature", "bounded_feature"),
        ("numerical_only", "numerical_phase"),
    }
    assert all(
        item.synthetic
        and not item.evidence_claimed
        and not item.observation_claimed
        and item.evidence_slots
        for item in review.signal_reviews
    )


def test_exact_beam_target_fixture_accepts_direct_cyclic_facility_clock() -> None:
    fixture_bytes = (BEAM_TARGET_FIXTURES / "plan_envelope_fixture.json").read_bytes()
    manifest_bytes = (BEAM_TARGET_FIXTURES / "reactor-domain.json").read_bytes()
    fixture = json.loads(fixture_bytes)

    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == "29b45eb736ad4e2594ee2f9d00dfdf24517cfa67e66d62d7c4fb5669140d1c67"
    )
    assert (
        hashlib.sha256(fixture_bytes).hexdigest()
        == "2a26c05646ace804fe87cf15e530d13784d45977ae51af48e29af8add920c766"
    )
    review = rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision="a09ff304e74a3acc14b167820f4a5f6fd619a8c2",
        source_artifact_sha256=(
            "a10e34420c991ed751d1294bb93980c270e53f34d479f683d169c7945eec7cfb"
        ),
        manifest_bytes=manifest_bytes,
        envelope_bytes=_compact(fixture["envelope"]),
        plan_bytes=_compact(fixture["plan"]),
    )
    direct = next(
        item
        for item in review.signal_reviews
        if item.observability_class is rs.ObservabilityClass.DIRECT_CYCLIC
    )

    assert direct.carrier is rs.SemanticCarrier.CYCLIC_PHASE
    assert direct.clock_identifier == "clk_facility"
    assert direct.synthetic
    assert not direct.evidence_claimed
    assert not direct.observation_claimed
    assert review.accepted_as_design_declaration
    assert not review.observation_claimed
    assert not review.classification_performed
    assert not review.semantic_ingress_declared
    assert not review.control_intent_created
    assert not review.actionable


def test_clock_meanings_remain_nonisomorphic_and_unmapped() -> None:
    by_kind = {item.plan_clock_kind: item for item in _review().clock_reviews}

    assert by_kind["facility_monotonic"].spo_clock_kind_candidate is None
    assert by_kind["facility_monotonic"].compatibility is (
        rs.DiagnosticClockCompatibility.UNMAPPED
    )
    assert (
        by_kind["shot_event_epoch"].spo_clock_kind_candidate
        is rs.ClockKind.SHOT_RELATIVE
    )
    assert by_kind["simulation"].spo_clock_kind_candidate is (
        rs.ClockKind.SIMULATION_MONOTONIC
    )
    assert not any(item.mapping_evidence_claimed for item in by_kind.values())


def test_review_is_digest_sealed_and_round_trips() -> None:
    review = _review()
    encoded = rs.device_diagnostic_plan_review_to_bytes(review)

    assert rs.device_diagnostic_plan_review_from_bytes(encoded) == review
    assert (
        rs.device_diagnostic_plan_review_digest(review)
        == hashlib.sha256(encoded).hexdigest()
    )
    assert (
        rs.device_diagnostic_plan_review_from_record(
            rs.device_diagnostic_plan_review_to_record(review)
        )
        == review
    )
    assert len(review.review_id) == 64


def test_portable_review_matches_its_published_json_schema() -> None:
    schema = json.loads(
        Path("docs/specs/device_diagnostic_plan_review.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        json.loads(rs.device_diagnostic_plan_review_to_bytes(_review()))
    )


def test_review_has_no_evidence_ingress_classification_or_control_authority() -> None:
    review = _review()

    assert review.accepted_as_design_declaration
    assert not review.evidence_claimed
    assert not review.observation_claimed
    assert not review.measurement_claimed
    assert not review.facility_binding_claimed
    assert not review.classification_performed
    assert not review.semantic_ingress_declared
    assert not review.control_intent_created
    assert review.authority == "review_only"
    assert not review.actionable


@pytest.mark.parametrize("source_revision", ["", "A" * 40, "0" * 39])
def test_source_revision_must_be_an_exact_lowercase_git_sha(
    source_revision: str,
) -> None:
    manifest, envelope, plan = _records()
    source = _source_bytes(manifest, envelope, plan)

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_producer_bytes(
            source_revision=source_revision,
            source_artifact_sha256=ARTIFACT_SHA256,
            manifest_bytes=source[0],
            envelope_bytes=source[1],
            plan_bytes=source[2],
        )
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.INVALID_SOURCE_IDENTITY
    )


def test_source_input_types_sizes_json_and_canonical_bytes_are_refused() -> None:
    manifest, envelope, plan = _records()
    source = _source_bytes(manifest, envelope, plan)
    cases: tuple[tuple[object, rs.DeviceDiagnosticPlanRefusalCode], ...] = (
        ("not bytes", rs.DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_TYPE),
        (b"", rs.DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_SIZE),
        (b"\xff", rs.DeviceDiagnosticPlanRefusalCode.INVALID_JSON),
        (b"{}", rs.DeviceDiagnosticPlanRefusalCode.NONCANONICAL_SOURCE_BYTES),
        (b'{"x":1,"x":2}\n', rs.DeviceDiagnosticPlanRefusalCode.DUPLICATE_JSON_KEY),
        (b'{"x":NaN}\n', rs.DeviceDiagnosticPlanRefusalCode.INVALID_JSON),
    )
    for bad_plan, expected in cases:
        with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
            rs.device_diagnostic_plan_review_from_producer_bytes(
                source_revision=SOURCE_REVISION,
                source_artifact_sha256=ARTIFACT_SHA256,
                manifest_bytes=source[0],
                envelope_bytes=source[1],
                plan_bytes=cast(bytes, bad_plan),
            )
        assert caught.value.code is expected


def test_oversized_source_is_refused_before_json_decode() -> None:
    manifest, envelope, plan = _records()
    source = _source_bytes(manifest, envelope, plan)
    oversized = b"x" * (rs.MAX_DEVICE_DIAGNOSTIC_SOURCE_BYTES + 1)

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=ARTIFACT_SHA256,
            manifest_bytes=source[0],
            envelope_bytes=source[1],
            plan_bytes=oversized,
        )
    assert caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_SIZE


def test_envelope_schema_registry_and_authority_drift_are_refused() -> None:
    manifest, envelope, plan = _records()
    envelope["schema_version"] = "2.0.0"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    envelope["binding"]["catalogue_version"] = "9.0.0"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.REGISTRY_BINDING_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    envelope["actionable"] = True
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
        manifest,
        envelope,
        plan,
    )


def test_manifest_assignment_registry_and_safety_drift_are_refused() -> None:
    manifest, envelope, plan = _records()
    manifest["project"] = envelope["project"] = "SCPN-NOT-THE-OWNER"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["spo_registry"]["version"] = "9.0.0"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.REGISTRY_BINDING_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["control_adapter"]["direct_actuation"] = True
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
        manifest,
        envelope,
        plan,
    )


def test_candidate_carrier_evidence_and_clock_drift_are_refused() -> None:
    manifest, envelope, plan = _records()
    plan["channels"].pop()
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CANDIDATE_COVERAGE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["carrier"] = "bounded_feature"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["evidence_bindings"].pop("validity")
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["clock_identifier"] = "clk_facility"
    plan["channels"][0]["evidence_bindings"]["clock_epoch"] = "clk_facility"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        manifest,
        envelope,
        plan,
    )


def test_frame_relation_numeric_and_synthetic_drift_are_refused() -> None:
    manifest, envelope, plan = _records()
    plan["frames"].pop(0)
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["clock_relations"][0]["evidence_claimed"] = True
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["element_count"] = True
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["synthetic"] = False
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
        manifest,
        envelope,
        plan,
    )


def test_output_tamper_unknown_fields_and_authority_objects_are_refused() -> None:
    review = _review()
    outer = json.loads(rs.device_diagnostic_plan_review_to_bytes(review))
    outer["payload"]["actionable"] = True
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_bytes(_compact(outer))
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH
    )

    record = rs.device_diagnostic_plan_review_to_record(review)
    record["unknown"] = False
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_record(record)
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH
    )


def test_remaining_deferral_relation_and_text_guards_are_refused() -> None:
    manifest, envelope, plan = _records()
    plan["channels"].pop()
    plan["deferrals"] = [
        {
            "candidate_id": "model.synthetic_oscillator_coordinate",
            "reason": "",
        }
    ]
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CANDIDATE_COVERAGE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["clock_relations"].append(deepcopy(plan["clock_relations"][0]))
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    plan["identifier"] = 1
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    review = _review()
    clock = review.clock_reviews[0]
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal):
        rs.DeviceDiagnosticClockReview(
            plan_clock_identifier=clock.plan_clock_identifier,
            plan_clock_kind=clock.plan_clock_kind,
            epoch=clock.epoch,
            resolution_s=clock.resolution_s,
            uncertainty_s=clock.uncertainty_s,
            spo_clock_kind_candidate=clock.spo_clock_kind_candidate,
            compatibility=clock.compatibility,
            mapping_evidence_claimed=True,
        )

    signal = review.signal_reviews[0]
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal):
        rs.DeviceDiagnosticSignalReview(
            channel_identifier=signal.channel_identifier,
            candidate_id=signal.candidate_id,
            observability_class=signal.observability_class,
            carrier=signal.carrier,
            clock_identifier=signal.clock_identifier,
            evidence_slots=signal.evidence_slots,
            evidence_claimed=True,
        )


def test_review_record_and_envelope_metadata_tamper_are_refused() -> None:
    review = _review()
    record = rs.device_diagnostic_plan_review_to_record(review)
    record["review_id"] = "0" * 64
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_record(record)
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH
    )

    encoded = rs.device_diagnostic_plan_review_to_bytes(review)
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_bytes(encoded[:-1] + b" \n")
    assert caught.value.code is (
        rs.DeviceDiagnosticPlanRefusalCode.NONCANONICAL_SOURCE_BYTES
    )

    outer = json.loads(encoded)
    outer["schema_version"] = "2.0.0"
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_bytes(_compact(outer))
    assert (
        caught.value.code
        is rs.DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA
    )


def test_source_digest_is_checked_before_semantic_interpretation() -> None:
    manifest, envelope, plan = _records()
    manifest_bytes, envelope_bytes, _ = _source_bytes(manifest, envelope, plan)
    plan["identifier"] = "another_plan"

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=ARTIFACT_SHA256,
            manifest_bytes=manifest_bytes,
            envelope_bytes=envelope_bytes,
            plan_bytes=_compact(plan),
        )
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.SOURCE_DIGEST_MISMATCH
    )


def test_additional_envelope_contract_refusals() -> None:
    for field, value, code in (
        (
            "capability",
            "device_configuration_model",
            rs.DeviceDiagnosticPlanRefusalCode.MANIFEST_CONTRACT_MISMATCH,
        ),
        (
            "manifest_sha256",
            "bad",
            rs.DeviceDiagnosticPlanRefusalCode.INVALID_SOURCE_IDENTITY,
        ),
        (
            "producer_revision",
            "",
            rs.DeviceDiagnosticPlanRefusalCode.INVALID_SOURCE_IDENTITY,
        ),
    ):
        manifest, envelope, plan = _records()
        envelope[field] = value
        if field == "manifest_sha256":
            plan_bytes = _compact(plan)
            with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
                rs.device_diagnostic_plan_review_from_producer_bytes(
                    source_revision=SOURCE_REVISION,
                    source_artifact_sha256=ARTIFACT_SHA256,
                    manifest_bytes=_pretty(manifest),
                    envelope_bytes=_compact(envelope),
                    plan_bytes=plan_bytes,
                )
            assert caught.value.code is code
        else:
            _assert_refusal(code, manifest, envelope, plan)


def test_additional_manifest_contract_refusals() -> None:
    manifest, envelope, plan = _records()
    manifest["schema_version"] = "2.0.0"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.UNSUPPORTED_SOURCE_SCHEMA,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["claims"] = ["machine ready"]
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["excluded_domains"] = manifest["excluded_domains"][3:]
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.MANIFEST_CONTRACT_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["configurations"] = list(reversed(manifest["configurations"]))
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.MANIFEST_CONTRACT_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["capabilities"][1]["identifier"] = "not_the_capability"
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.MANIFEST_CONTRACT_MISMATCH,
        manifest,
        envelope,
        plan,
    )


def test_configuration_assignment_refusals() -> None:
    manifest, envelope, plan = _records()
    manifest["configurations"] = envelope["configurations"] = []
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH,
        manifest,
        envelope,
        plan,
    )

    manifest, envelope, plan = _records()
    manifest["configurations"] = envelope["configurations"] = ["unknown_machine"]
    _assert_refusal(
        rs.DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH,
        manifest,
        envelope,
        plan,
    )


@pytest.mark.parametrize(
    ("code", "mutate"),
    (
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan.__setitem__("identifier", "Bad"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH,
            lambda plan: plan["frames"][0].__setitem__("description", ""),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CANDIDATE_COVERAGE_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__(
                "candidate_id", "closed.not_applicable"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CANDIDATE_COVERAGE_MISMATCH,
            lambda plan: plan["deferrals"].append(
                {"candidate_id": "closed.resolved_mhd_mode", "reason": "defer"}
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__("carrier", "not_a_carrier"),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][0]["evidence_bindings"].__setitem__(
                "validity", ""
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__(
                "clock_identifier", "not_declared"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["channels"][0]["evidence_bindings"].__setitem__(
                "clock_epoch", "clk_facility"
            ),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
            lambda plan: plan["channels"][2].__setitem__("sample_rate_hz", 10.0),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["channels"][0].__setitem__("timing_uncertainty_s", 0.0),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["clocks"][1].__setitem__("resolution_s", 1.0),
        ),
        (
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
            lambda plan: plan["channels"][1].__setitem__("timing_uncertainty_s", 0.1),
        ),
    ),
)
def test_deep_channel_and_plan_refusals(
    code: rs.DeviceDiagnosticPlanRefusalCode,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _assert_plan_mutation(code, mutate)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda plan: plan["clocks"][0].__setitem__("kind", "wall_clock"),
        lambda plan: plan["clocks"][0].__setitem__("epoch", ""),
        lambda plan: plan["clock_relations"][0].__setitem__(
            "parent_identifier", "clk_shot"
        ),
        lambda plan: plan["clock_relations"][0].__setitem__(
            "parent_identifier", "clk_sim"
        ),
        lambda plan: plan["clock_relations"][0].__setitem__("method", ""),
        lambda plan: plan.__setitem__("clock_relations", []),
    ),
)
def test_deep_clock_relation_refusals(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _assert_plan_mutation(
        rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH, mutate
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda plan: plan.__setitem__("binding", []),
        lambda plan: plan.__setitem__("frames", {}),
        lambda plan: plan["frames"].__setitem__(0, []),
        lambda plan: plan["channels"][0].__setitem__("sample_rate_hz", "fast"),
        lambda plan: plan["channels"][0].__setitem__("sample_rate_hz", 0.0),
        lambda plan: plan["clocks"][0].__setitem__("uncertainty_s", -1.0),
        lambda plan: plan.__setitem__("frames", list(reversed(plan["frames"]))),
        lambda plan: plan["frames"][0].__setitem__("identifier", "Bad"),
    ),
)
def test_structural_numeric_and_order_refusals(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _assert_plan_mutation(
        rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH, mutate
    )


def test_identity_type_and_configuration_value_types_are_refused() -> None:
    manifest, envelope, plan = _records()
    source = _source_bytes(manifest, envelope, plan)
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        rs.device_diagnostic_plan_review_from_producer_bytes(
            source_revision=cast(str, 1),
            source_artifact_sha256=ARTIFACT_SHA256,
            manifest_bytes=source[0],
            envelope_bytes=source[1],
            plan_bytes=source[2],
        )
    assert caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.INVALID_INPUT_TYPE

    manifest, envelope, plan = _records()
    envelope["configurations"] = [1]
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest, envelope, plan)
    assert (
        caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.PLAN_STRUCTURE_MISMATCH
    )
