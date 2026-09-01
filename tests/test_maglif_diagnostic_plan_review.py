# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MagLIF diagnostic-plan intake tests
"""Exercise the exact MagLIF producer-bytes to SPO design-review boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

FIXTURES = Path("tests/fixtures/maglif_diagnostic_plan")
SOURCE_REVISION = "f39139790fd932871de1f18da96c0e428a436d18"
MANIFEST_SHA256 = "7f27182b3f555a084476434215f49a3c185a0706e2a23cc20a02a702e0233199"
FIXTURE_SHA256 = "81ff0f7bb2390736a91dec078b3032da7efc963369301f6c92e7123776621c57"
ARTIFACT_SHA256 = "c8f6455e89b439f4d8ecdf4352efd417222a54854aaa4d798b7359c435a102b9"
REVIEW_ID = "521ef22044d3430ca395c53e32cef7c3feeab757706dfbd19a023becb7cf694f"
REVIEW_SHA256 = "c8baac8822ea0c54525450b72f2ceb560e823cef485bb205fa4e2e84adeaba8e"


def _compact(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _records() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = json.loads((FIXTURES / "plan_envelope_fixture.json").read_bytes())
    manifest = json.loads((FIXTURES / "reactor-domain.json").read_bytes())
    return manifest, document["envelope"], document["plan"]


def _source_bytes(
    manifest: dict[str, Any],
    envelope: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[bytes, bytes, bytes]:
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    plan_bytes = _compact(plan)
    envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["plan_identifier"] = plan["identifier"]
    envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    return manifest_bytes, _compact(envelope), plan_bytes


def _review(
    *,
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


def test_exact_maglif_producer_objects_are_digest_pinned() -> None:
    manifest_bytes = (FIXTURES / "reactor-domain.json").read_bytes()
    fixture_bytes = (FIXTURES / "plan_envelope_fixture.json").read_bytes()

    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256


def test_exact_maglif_fixture_produces_one_design_review() -> None:
    review = _review()

    assert review.source_project == "SCPN-MIF-MAGLIF-CORE"
    assert review.source_revision == SOURCE_REVISION
    assert review.source_artifact_sha256 == ARTIFACT_SHA256
    assert review.source_manifest_sha256 == MANIFEST_SHA256
    assert review.producer_package_revision == "0.1.0.dev0"
    assert review.configurations == ("maglif",)
    assert review.plan_identifier == "maglif_reference_plan"
    assert review.frame_ids == ("frm_liner_axis",)
    assert review.planned_candidate_ids == (
        "magneto_inertial.driver_arrival",
        "magneto_inertial.resolved_asymmetry_mode",
        "magneto_inertial.translation_and_compression",
        "model.synthetic_oscillator_coordinate",
    )
    assert review.deferred_candidate_ids == ()
    assert review.review_id == REVIEW_ID


def test_maglif_feature_event_mode_and_numerical_meanings_remain_distinct() -> None:
    review = _review()
    meanings = {
        item.channel_identifier: (
            item.candidate_id,
            item.observability_class.value,
            item.carrier.value,
            item.clock_identifier,
        )
        for item in review.signal_reviews
    }

    assert meanings == {
        "ch_compression_trajectory_set": (
            "magneto_inertial.translation_and_compression",
            "noncyclic_feature",
            "bounded_feature",
            "clk_shot",
        ),
        "ch_liner_arrival_train": (
            "magneto_inertial.driver_arrival",
            "event_relative",
            "event_cycle",
            "clk_shot",
        ),
        "ch_liner_asymmetry_set": (
            "magneto_inertial.resolved_asymmetry_mode",
            "derived_cyclic",
            "complex_mode",
            "clk_facility",
        ),
        "ch_synthetic_oscillator": (
            "model.synthetic_oscillator_coordinate",
            "numerical_only",
            "numerical_phase",
            "clk_sim",
        ),
    }
    assert all(
        signal.synthetic
        and not signal.evidence_claimed
        and not signal.observation_claimed
        for signal in review.signal_reviews
    )


def test_maglif_clock_classes_do_not_claim_facility_correlation() -> None:
    clocks = {item.plan_clock_identifier: item for item in _review().clock_reviews}

    assert clocks["clk_facility"].compatibility is (
        rs.DiagnosticClockCompatibility.UNMAPPED
    )
    assert clocks["clk_facility"].spo_clock_kind_candidate is None
    assert clocks["clk_shot"].compatibility is (
        rs.DiagnosticClockCompatibility.EVENT_RELATIVE_COMPATIBLE
    )
    assert clocks["clk_shot"].spo_clock_kind_candidate is rs.ClockKind.SHOT_RELATIVE
    assert clocks["clk_sim"].compatibility is (
        rs.DiagnosticClockCompatibility.SYNTHETIC_COMPATIBLE
    )
    assert clocks["clk_sim"].spo_clock_kind_candidate is (
        rs.ClockKind.SIMULATION_MONOTONIC
    )
    assert not any(item.mapping_evidence_claimed for item in clocks.values())


def test_maglif_review_is_canonical_schema_valid_and_round_trips() -> None:
    review = _review()
    encoded = rs.device_diagnostic_plan_review_to_bytes(review)
    schema = json.loads(
        Path("docs/specs/device_diagnostic_plan_review.schema.json").read_bytes()
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))
    assert hashlib.sha256(encoded).hexdigest() == REVIEW_SHA256
    assert rs.device_diagnostic_plan_review_from_bytes(encoded) == review
    assert (
        rs.device_diagnostic_plan_review_from_record(
            rs.device_diagnostic_plan_review_to_record(review)
        )
        == review
    )


def test_maglif_design_review_does_not_inherit_frc_mif_evidence() -> None:
    review = _review()
    profile = rs.DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve("maglif")

    assert review.accepted_as_design_declaration
    assert not review.evidence_claimed
    assert not review.measurement_claimed
    assert not review.observation_claimed
    assert not review.facility_binding_claimed
    assert not review.classification_performed
    assert not review.semantic_ingress_declared
    assert not review.control_intent_created
    assert review.authority == "review_only"
    assert not review.actionable
    assert profile.ingress_state is rs.SemanticIngressState.NOT_DECLARED
    assert profile.producer_project is None
    assert "scpn_mif_maglif_core" not in sys.modules


@pytest.mark.parametrize(
    ("channel_index", "field", "value", "code"),
    (
        (
            0,
            "timing_uncertainty_s",
            5e-10,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            1,
            "timing_uncertainty_s",
            None,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            1,
            "timing_uncertainty_s",
            5e-11,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            2,
            "timing_uncertainty_s",
            1e-11,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            2,
            "clock_identifier",
            "clk_shot",
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            3,
            "carrier",
            "cyclic_phase",
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
        ),
    ),
)
def test_maglif_timing_clock_and_carrier_drift_is_refused(
    channel_index: int,
    field: str,
    value: object,
    code: rs.DeviceDiagnosticPlanRefusalCode,
) -> None:
    manifest, envelope, plan = _records()
    plan["channels"][channel_index][field] = value

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest=manifest, envelope=envelope, plan=plan)
    assert caught.value.code is code


def test_maglif_assignment_and_authority_escalation_is_refused() -> None:
    manifest, envelope, plan = _records()
    envelope["configurations"] = ["frc_compression_mif"]
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest=manifest, envelope=envelope, plan=plan)
    assert caught.value.code is (
        rs.DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH
    )

    manifest, envelope, plan = _records()
    plan["channels"][0]["synthetic"] = False
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest=manifest, envelope=envelope, plan=plan)
    assert caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION

    manifest, envelope, plan = _records()
    plan["clock_relations"][0]["evidence_claimed"] = True
    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest=manifest, envelope=envelope, plan=plan)
    assert caught.value.code is rs.DeviceDiagnosticPlanRefusalCode.AUTHORITY_ESCALATION
