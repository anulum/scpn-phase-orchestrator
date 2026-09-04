# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — projectile/impact-ICF physical-evidence request tests
"""Exercise exact ICF-impact custody through the configuration request boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/icf_impact_diagnostic_plan"
MATERIALISED_REQUEST = (
    REPO
    / "docs/reference/data/projectile_or_impact_icf_physical_evidence_request.v1.json"
)
SCHEMA = REPO / "docs/specs/device_physical_evidence_request.schema.json"
MATERIALISER = REPO / "tools/materialize_device_physical_evidence_request.py"
SOURCE_REVISION = "397f1f2a5fb2af1ad620174a43720eb1c7b5de5f"
SOURCE_ARTIFACT_SHA256 = (
    "4485dffa189017837f8234791b9b160fa12b9da3ab2ea71e9a2128f55f7ca162"
)
CONFIGURATION = "projectile_or_impact_icf"
REQUEST_ID = "27a576dd67b149069bd4eefa1ef343c570a0084688acd3370721e6a34023ac62"
REQUEST_SHA256 = "ccdda701953cdec025d3b7f63f026bbaf92efed54ecebddfbb510eb83eab64e1"
SOURCE_REVIEW_ID = "eeefac32254f871dc94ce655353b60327f2aa1e7dde566bd92c89c86cb8eaa84"
SOURCE_REVIEW_SHA256 = (
    "5035b44a327b916f662125cc452777fb30bc43c6ee37642d354d2c46c2ff60e3"
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


def test_request_binds_exact_icf_impact_custody_and_configuration() -> None:
    request = _request()

    assert request.configuration == CONFIGURATION
    assert request.device_project == "SCPN-ICF-IMPACT-CORE"
    assert request.requested_owner_project == "SCPN-ICF-IMPACT-CORE"
    assert request.source_revision == SOURCE_REVISION
    assert request.source_artifact_sha256 == SOURCE_ARTIFACT_SHA256
    assert request.source_review_id == SOURCE_REVIEW_ID
    assert request.source_review_sha256 == SOURCE_REVIEW_SHA256
    assert request.request_id == REQUEST_ID


def test_request_preserves_impact_timing_features_and_model_phase() -> None:
    candidates = {item.candidate_id: item for item in _request().candidate_requirements}

    timing = candidates["inertial.driver_timing"]
    assert timing.observability_class == "event_relative"
    assert timing.declared_carriers == ("event_cycle",)
    assert timing.channel_identifiers == ("ch_impact_timing_train",)
    asymmetry = candidates["inertial.resolved_asymmetry_mode"]
    assert asymmetry.observability_class == "derived_cyclic"
    assert asymmetry.declared_carriers == ("complex_mode",)
    assert asymmetry.observation_operator_required
    trajectory = candidates["inertial.implosion_trajectory"]
    outcome = candidates["inertial.shot_outcome"]
    assert trajectory.observability_class == "noncyclic_feature"
    assert trajectory.declared_carriers == ("bounded_feature",)
    assert outcome.observability_class == "noncyclic_feature"
    assert outcome.declared_carriers == ("bounded_feature",)
    numerical = candidates["model.synthetic_oscillator_coordinate"]
    assert numerical.observability_class == "numerical_only"
    assert numerical.declared_carriers == ("numerical_phase",)
    assert not numerical.physical_selection_eligible


@pytest.mark.parametrize(
    "other_configuration",
    (
        "laser_icf_direct_drive",
        "ion_beam_icf",
        "pulsed_electron_beam_icf",
        "beam_target",
    ),
)
def test_impact_plan_cannot_transfer_request_to_other_icf_configurations(
    other_configuration: str,
) -> None:
    with pytest.raises(rs.DevicePhysicalEvidenceRequestRefusalError) as caught:
        rs.device_physical_evidence_request_from_plan_review(
            _review(), configuration=other_configuration
        )
    assert caught.value.code is (
        rs.DevicePhysicalEvidenceRequestRefusalCode.CONFIGURATION_MISMATCH
    )


def test_request_keeps_all_physical_and_control_gates_closed() -> None:
    request = _request()

    assert len(request.requirements) == 13
    assert all(item.missing for item in request.requirements)
    assert not request.physical_source_present
    assert request.selected_candidate_id is None
    assert not request.observation_admitted
    assert not request.phase_inference_eligible
    assert not request.semantic_ingress_declared
    assert not request.control_admission_requested
    assert not request.control_intent_created
    assert not request.actionable
    assert not request.execution_permitted
    assert not request.direct_actuation
    assert request.authority == "review_only"
    assert request.machine_protection_final_veto


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


def test_materialiser_reproduces_impact_request_from_a_foreign_working_directory(
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
