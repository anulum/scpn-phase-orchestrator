# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — dense-plasma-focus diagnostic-plan intake tests
"""Exercise the exact DPF producer-bytes to SPO design-review boundary."""

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

FIXTURES = Path("tests/fixtures/dense_plasma_focus_diagnostic_plan")
SOURCE_REVISION = "d964bc33ed7bf3c4a6ee7f61220daff66ba8e89c"
MANIFEST_SHA256 = "d872b68fad10d71cf7a43277446231e41b5a6a46f57939a84b7bd193a96aee8a"
FIXTURE_SHA256 = "21abf14f74639c78a973e3d97c6f0a1bbc50affee5b42a8502c6ea60bcf26b27"
ARTIFACT_SHA256 = "212bf5b8189686905511dd3c6281caa33e0e1b2c8ffcb6ce6afd74a43db3cd2c"
REVIEW_ID = "5ee59adbc190cd7877bf07a144cc50e44a9e227069c81edc2ca324051e02d8e4"
REVIEW_SHA256 = "38a186887c964dbe8a326ac432374212dcb021e958d2edde2bca4e3c74a5f914"


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


def test_exact_producer_objects_are_digest_pinned() -> None:
    manifest_bytes = (FIXTURES / "reactor-domain.json").read_bytes()
    fixture_bytes = (FIXTURES / "plan_envelope_fixture.json").read_bytes()

    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256
    assert hashlib.sha256(fixture_bytes).hexdigest() == FIXTURE_SHA256


def test_exact_dpf_fixture_produces_one_design_review() -> None:
    review = _review()

    assert review.source_project == "SCPN-DENSE-PLASMA-FOCUS-CORE"
    assert review.source_revision == SOURCE_REVISION
    assert review.source_artifact_sha256 == ARTIFACT_SHA256
    assert review.source_manifest_sha256 == MANIFEST_SHA256
    assert review.producer_package_revision == "0.1.0.dev0"
    assert review.configurations == ("dense_plasma_focus",)
    assert review.plan_identifier == "dpf_reference_plan"
    assert review.frame_ids == ("frm_electrode_axis",)
    assert review.planned_candidate_ids == (
        "model.synthetic_oscillator_coordinate",
        "self_magnetic.drive_waveform",
        "self_magnetic.resolved_instability_mode",
    )
    assert review.deferred_candidate_ids == ()
    assert review.review_id == REVIEW_ID


def test_dpf_event_mode_and_numerical_meanings_remain_distinct() -> None:
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
        "ch_discharge_current_train": (
            "self_magnetic.drive_waveform",
            "event_relative",
            "event_cycle",
            "clk_shot",
        ),
        "ch_neck_mode_array": (
            "self_magnetic.resolved_instability_mode",
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


def test_dpf_clock_classes_do_not_claim_facility_correlation() -> None:
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


def test_dpf_review_is_canonical_schema_valid_and_round_trips() -> None:
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


def test_dpf_design_review_has_no_semantic_or_control_authority() -> None:
    review = _review()
    profile = rs.DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve("dense_plasma_focus")

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
    assert "scpn_dense_plasma_focus_core" not in sys.modules


@pytest.mark.parametrize(
    ("channel_index", "field", "value", "code"),
    (
        (
            0,
            "timing_uncertainty_s",
            None,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            0,
            "timing_uncertainty_s",
            5e-11,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            1,
            "timing_uncertainty_s",
            5e-9,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            1,
            "clock_identifier",
            "clk_shot",
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            2,
            "carrier",
            "cyclic_phase",
            rs.DeviceDiagnosticPlanRefusalCode.CARRIER_EVIDENCE_MISMATCH,
        ),
    ),
)
def test_dpf_timing_clock_and_carrier_drift_is_refused(
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


def test_dpf_assignment_and_authority_escalation_is_refused() -> None:
    manifest, envelope, plan = _records()
    envelope["configurations"] = ["z_pinch"]
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
