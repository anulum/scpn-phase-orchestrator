# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — laser-ICF fast/shock evidence-request tests
"""Exercise exact ICF-laser custody without inheriting direct-drive evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/icf_laser_diagnostic_plan"
MATERIALISED_REQUEST = (
    REPO / "docs/reference/data/"
    "laser_icf_fast_or_shock_ignition_physical_evidence_request.v1.json"
)
DIRECT_DRIVE_REQUEST = (
    REPO
    / "docs/reference/data/laser_icf_direct_drive_physical_evidence_request.v1.json"
)
SCHEMA = REPO / "docs/specs/device_physical_evidence_request.schema.json"
MATERIALISER = REPO / "tools/materialize_device_physical_evidence_request.py"
SOURCE_REVISION = "a0ad63207c9aff5f00273e2c37fa580feb9d6c38"
SOURCE_ARTIFACT_SHA256 = (
    "e480e4a2c67b45dc04ef04cebd44ddcb3a6a3efc24739f54606d25cd0661bde3"
)
CONFIGURATION = "laser_icf_fast_or_shock_ignition"
REQUEST_ID = "b3c6dc4c666b2af38833f8f506ea79ba6efe838c6fa4d0d48ea68e20f1f57691"
REQUEST_SHA256 = "d4cdd8b0ea88397807457e24aff511ab3dc262266b02295d4540cc8fb7d3103d"
SOURCE_REVIEW_ID = "0dac2e7bf5043eab60f5979b1fbf73a5331928816b2a7152c6ad41b27151d083"
SOURCE_REVIEW_SHA256 = (
    "5cb5824bd6058a148d8ab71ead7a0d35939a30b8ddd8d40c1f68cad3caaf0467"
)


def _compact(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _review() -> rs.DeviceDiagnosticPlanReview:
    document: dict[str, Any] = json.loads(
        (FIXTURES / "plan_envelope_fixture.json").read_bytes()
    )
    manifest: dict[str, Any] = json.loads(
        (FIXTURES / "reactor-domain.json").read_bytes()
    )
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    plan_bytes = _compact(document["plan"])
    envelope = deepcopy(document["envelope"])
    envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["plan_identifier"] = document["plan"]["identifier"]
    envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    return rs.device_diagnostic_plan_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=SOURCE_ARTIFACT_SHA256,
        manifest_bytes=manifest_bytes,
        envelope_bytes=_compact(envelope),
        plan_bytes=plan_bytes,
    )


def _request() -> rs.DevicePhysicalEvidenceRequest:
    return rs.device_physical_evidence_request_from_plan_review(
        _review(), configuration=CONFIGURATION
    )


def _cli(output: Path, *, check: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(MATERIALISER),
        "--fixture-dir",
        str(FIXTURES),
        "--configuration",
        CONFIGURATION,
        "--source-revision",
        SOURCE_REVISION,
        "--source-artifact-sha256",
        SOURCE_ARTIFACT_SHA256,
        "--output",
        str(output),
    ]
    if check:
        command.append("--check")
    return command


def test_request_binds_exact_icf_laser_custody_and_fast_shock_configuration() -> None:
    request = _request()

    assert request.configuration == CONFIGURATION
    assert request.device_project == "SCPN-ICF-LASER-CORE"
    assert request.requested_owner_project == "SCPN-ICF-LASER-CORE"
    assert request.source_revision == SOURCE_REVISION
    assert request.source_artifact_sha256 == SOURCE_ARTIFACT_SHA256
    assert request.source_review_id == SOURCE_REVIEW_ID
    assert request.source_review_sha256 == SOURCE_REVIEW_SHA256
    assert request.request_id == REQUEST_ID


def test_request_preserves_beam_timing_asymmetry_features_and_model_phase() -> None:
    candidates = {item.candidate_id: item for item in _request().candidate_requirements}

    timing = candidates["inertial.driver_timing"]
    assert timing.observability_class == "event_relative"
    assert timing.channel_identifiers == ("ch_beam_timing_train",)
    assert timing.declared_carriers == ("event_cycle",)
    asymmetry = candidates["inertial.resolved_asymmetry_mode"]
    assert asymmetry.observability_class == "derived_cyclic"
    assert asymmetry.observation_operator_required
    trajectory = candidates["inertial.implosion_trajectory"]
    outcome = candidates["inertial.shot_outcome"]
    assert trajectory.observability_class == "noncyclic_feature"
    assert outcome.observability_class == "noncyclic_feature"
    numerical = candidates["model.synthetic_oscillator_coordinate"]
    assert numerical.observability_class == "numerical_only"
    assert numerical.declared_carriers == ("numerical_phase",)
    assert not numerical.physical_selection_eligible


def test_shared_review_does_not_transfer_direct_drive_identity_or_evidence() -> None:
    fast_shock_request = _request()
    direct_drive_request = rs.device_physical_evidence_request_from_bytes(
        DIRECT_DRIVE_REQUEST.read_bytes()
    )

    assert fast_shock_request.source_review_id == direct_drive_request.source_review_id
    assert (
        fast_shock_request.source_review_sha256
        == direct_drive_request.source_review_sha256
    )
    assert fast_shock_request.configuration != direct_drive_request.configuration
    assert fast_shock_request.request_id != direct_drive_request.request_id
    assert fast_shock_request.request_id == REQUEST_ID
    assert all(item.missing for item in fast_shock_request.requirements)
    assert not fast_shock_request.physical_source_present
    assert fast_shock_request.selected_candidate_id is None
    assert not fast_shock_request.observation_admitted
    assert not fast_shock_request.phase_inference_eligible
    assert not fast_shock_request.semantic_ingress_declared
    assert not fast_shock_request.control_admission_requested
    assert not fast_shock_request.control_intent_created
    assert not fast_shock_request.actionable
    assert not fast_shock_request.execution_permitted
    assert not fast_shock_request.direct_actuation
    assert fast_shock_request.authority == "review_only"
    assert fast_shock_request.machine_protection_final_veto


def test_materialised_request_is_exact_schema_valid_and_replayable() -> None:
    encoded = rs.device_physical_evidence_request_to_bytes(_request())
    schema = json.loads(SCHEMA.read_bytes())

    assert MATERIALISED_REQUEST.read_bytes() == encoded
    assert hashlib.sha256(encoded).hexdigest() == REQUEST_SHA256
    assert rs.device_physical_evidence_request_digest(_request()) == REQUEST_SHA256
    assert (
        rs.device_physical_evidence_request_from_bytes(
            encoded, expected_sha256=REQUEST_SHA256
        )
        == _request()
    )
    Draft202012Validator(schema).validate(json.loads(encoded))


def test_materialiser_replays_from_foreign_cwd_and_check_mode_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "request.json"
    written = subprocess.run(
        _cli(output),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    receipt = json.loads(written.stdout)

    assert output.read_bytes() == MATERIALISED_REQUEST.read_bytes()
    assert receipt["configuration"] == CONFIGURATION
    assert receipt["request_id"] == REQUEST_ID
    assert receipt["envelope_sha256"] == REQUEST_SHA256
    subprocess.run(
        _cli(output, check=True),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    output.write_bytes(b"stale\n")
    stale = subprocess.run(
        _cli(output, check=True),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert stale.returncode != 0
    assert "stale or missing materialised request" in stale.stderr
