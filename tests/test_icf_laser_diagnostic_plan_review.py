# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — laser ICF diagnostic-plan intake tests
"""Exercise exact laser-ICF producer bytes at SPO's design-review boundary."""

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

FIXTURES = Path("tests/fixtures/icf_laser_diagnostic_plan")
SOURCE_REVISION = "a0ad63207c9aff5f00273e2c37fa580feb9d6c38"
MANIFEST_SHA256 = "0d0d56b5494a8aed24edb98499bf8d809dfbef6d25e02448f0336e3593f2b6c8"
FIXTURE_SHA256 = "7e8cc1e3b72a15c5df1f4a890571284d5f4c872f9ec73cbda6e567a42b93bceb"
ARTIFACT_SHA256 = "e480e4a2c67b45dc04ef04cebd44ddcb3a6a3efc24739f54606d25cd0661bde3"
REVIEW_ID = "0dac2e7bf5043eab60f5979b1fbf73a5331928816b2a7152c6ad41b27151d083"
REVIEW_SHA256 = "f1979a1bd847157117add24bec7c271754aecf0a619b24c41c705bb09387e50a"


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


def test_exact_icf_laser_producer_objects_are_digest_pinned() -> None:
    assert (
        hashlib.sha256((FIXTURES / "reactor-domain.json").read_bytes()).hexdigest()
        == MANIFEST_SHA256
    )
    assert (
        hashlib.sha256(
            (FIXTURES / "plan_envelope_fixture.json").read_bytes()
        ).hexdigest()
        == FIXTURE_SHA256
    )


def test_exact_icf_laser_fixture_produces_one_design_review() -> None:
    review = _review()

    assert review.source_project == "SCPN-ICF-LASER-CORE"
    assert review.source_revision == SOURCE_REVISION
    assert review.source_artifact_sha256 == ARTIFACT_SHA256
    assert review.source_manifest_sha256 == MANIFEST_SHA256
    assert review.producer_package_revision == "0.1.0.dev0"
    assert review.configurations == (
        "laser_icf_direct_drive",
        "laser_icf_fast_or_shock_ignition",
        "laser_icf_indirect_drive",
    )
    assert review.plan_identifier == "icf_laser_reference_plan"
    assert review.frame_ids == ("frm_beamline", "frm_chamber")
    assert review.planned_candidate_ids == (
        "inertial.driver_timing",
        "inertial.implosion_trajectory",
        "inertial.resolved_asymmetry_mode",
        "inertial.shot_outcome",
        "model.synthetic_oscillator_coordinate",
    )
    assert review.deferred_candidate_ids == ()
    assert review.review_id == REVIEW_ID


def test_icf_laser_signal_meanings_remain_distinct() -> None:
    meanings = {
        item.channel_identifier: (
            item.candidate_id,
            item.observability_class.value,
            item.carrier.value,
            item.clock_identifier,
        )
        for item in _review().signal_reviews
    }

    assert meanings == {
        "ch_asymmetry_mode_set": (
            "inertial.resolved_asymmetry_mode",
            "derived_cyclic",
            "complex_mode",
            "clk_facility",
        ),
        "ch_beam_timing_train": (
            "inertial.driver_timing",
            "event_relative",
            "event_cycle",
            "clk_shot",
        ),
        "ch_shot_outcome_set": (
            "inertial.shot_outcome",
            "noncyclic_feature",
            "bounded_feature",
            "clk_shot",
        ),
        "ch_synthetic_oscillator": (
            "model.synthetic_oscillator_coordinate",
            "numerical_only",
            "numerical_phase",
            "clk_sim",
        ),
        "ch_trajectory_radiography": (
            "inertial.implosion_trajectory",
            "noncyclic_feature",
            "bounded_feature",
            "clk_shot",
        ),
    }


def test_icf_laser_clocks_do_not_claim_facility_correlation() -> None:
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


def test_icf_laser_review_is_canonical_schema_valid_and_round_trips() -> None:
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


def test_icf_laser_review_does_not_equate_or_inherit_icf_evidence() -> None:
    review = _review()
    profiles = tuple(
        rs.DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(configuration)
        for configuration in review.configurations
    )

    assert review.accepted_as_design_declaration
    assert all(
        signal.synthetic
        and not signal.evidence_claimed
        and not signal.observation_claimed
        for signal in review.signal_reviews
    )
    assert not review.evidence_claimed
    assert not review.measurement_claimed
    assert not review.observation_claimed
    assert not review.facility_binding_claimed
    assert not review.classification_performed
    assert not review.semantic_ingress_declared
    assert not review.control_intent_created
    assert review.authority == "review_only"
    assert not review.actionable
    assert all(
        profile.ingress_state is rs.SemanticIngressState.NOT_DECLARED
        and profile.producer_project is None
        for profile in profiles
    )
    assert len({profile.configuration for profile in profiles}) == 3
    assert "scpn_icf_laser_core" not in sys.modules


@pytest.mark.parametrize(
    ("channel_index", "field", "value", "code"),
    (
        (
            0,
            "timing_uncertainty_s",
            1e-12,
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
            5e-13,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            2,
            "timing_uncertainty_s",
            1e-12,
            rs.DeviceDiagnosticPlanRefusalCode.CLOCK_SEMANTICS_MISMATCH,
        ),
        (
            0,
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
def test_icf_laser_timing_clock_and_carrier_drift_is_refused(
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


def test_icf_laser_tighter_valid_timing_bound_stays_non_evidentiary() -> None:
    manifest, envelope, plan = _records()
    plan["channels"][1]["timing_uncertainty_s"] = 5e-12

    review = _review(manifest=manifest, envelope=envelope, plan=plan)
    signal = next(
        item
        for item in review.signal_reviews
        if item.channel_identifier == "ch_beam_timing_train"
    )

    assert signal.observability_class is rs.ObservabilityClass.EVENT_RELATIVE
    assert signal.carrier is rs.SemanticCarrier.EVENT_CYCLE
    assert not signal.evidence_claimed
    assert not signal.observation_claimed
    assert not review.evidence_claimed


@pytest.mark.parametrize(
    "configuration",
    ("ion_beam_icf", "projectile_or_impact_icf", "beam_target"),
)
def test_icf_laser_sibling_configuration_assignment_is_refused(
    configuration: str,
) -> None:
    manifest, envelope, plan = _records()
    envelope["configurations"] = [configuration]

    with pytest.raises(rs.DeviceDiagnosticPlanRefusal) as caught:
        _review(manifest=manifest, envelope=envelope, plan=plan)
    assert caught.value.code is (
        rs.DeviceDiagnosticPlanRefusalCode.PROJECT_ASSIGNMENT_MISMATCH
    )


def test_icf_laser_authority_escalation_is_refused() -> None:
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
