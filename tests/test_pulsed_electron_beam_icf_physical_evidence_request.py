# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — pulsed-electron-beam-ICF evidence-request tests
"""Exercise exact ICF-beam custody without inheriting ion-beam evidence."""

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
FIXTURES = REPO / "tests/fixtures/icf_beam_diagnostic_plan"
MATERIALISED_REQUEST = (
    REPO / "docs/reference/data/"
    "pulsed_electron_beam_icf_physical_evidence_request.v1.json"
)
ION_BEAM_REQUEST = (
    REPO / "docs/reference/data/ion_beam_icf_physical_evidence_request.v1.json"
)
SCHEMA = REPO / "docs/specs/device_physical_evidence_request.schema.json"
MATERIALISER = REPO / "tools/materialize_device_physical_evidence_request.py"
SOURCE_REVISION = "3ee15a5bf56b38614770351caa1112a4832d5254"
SOURCE_ARTIFACT_SHA256 = (
    "2c644ff2ddd1b2bd1d5af96f610b8af59c7e504f7b3ea2eed26903cb4ddd65eb"
)
CONFIGURATION = "pulsed_electron_beam_icf"
REQUEST_ID = "bbe0825d5aeb893089a10bb6ec6d94decf76dbc2b5f93b735ff704885f63c2e7"
REQUEST_SHA256 = "9461ddbc89f623bb0f6d2584e6734eef66e5c9abc1c94f4b18ca131acc9fa15a"
SOURCE_REVIEW_ID = "5da4be074476c8b3bd4a16c199d5f9f359e11f4e1fa36554765a1c880bf41719"
SOURCE_REVIEW_SHA256 = (
    "6200379b8ec7284f05c2f271a0a3fda72c1e0efe3fbfaa97aef49a01a7700b3d"
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


def test_request_binds_exact_icf_beam_custody_and_electron_configuration() -> None:
    request = _request()

    assert request.configuration == CONFIGURATION
    assert request.device_project == "SCPN-ICF-BEAM-CORE"
    assert request.requested_owner_project == "SCPN-ICF-BEAM-CORE"
    assert request.source_revision == SOURCE_REVISION
    assert request.source_artifact_sha256 == SOURCE_ARTIFACT_SHA256
    assert request.source_review_id == SOURCE_REVIEW_ID
    assert request.source_review_sha256 == SOURCE_REVIEW_SHA256
    assert request.request_id == REQUEST_ID


def test_request_preserves_beam_timing_asymmetry_features_and_model_phase() -> None:
    candidates = {item.candidate_id: item for item in _request().candidate_requirements}

    timing = candidates["inertial.driver_timing"]
    assert timing.observability_class == "event_relative"
    assert timing.channel_identifiers == ("ch_bunch_timing_train",)
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


def test_shared_review_does_not_transfer_ion_request_identity_or_evidence() -> None:
    electron_request = _request()
    ion_request = rs.device_physical_evidence_request_from_bytes(
        ION_BEAM_REQUEST.read_bytes()
    )

    assert electron_request.source_review_id == ion_request.source_review_id
    assert electron_request.source_review_sha256 == ion_request.source_review_sha256
    assert electron_request.configuration != ion_request.configuration
    assert electron_request.request_id != ion_request.request_id
    assert electron_request.request_id == REQUEST_ID
    assert all(item.missing for item in electron_request.requirements)
    assert not electron_request.physical_source_present
    assert electron_request.selected_candidate_id is None
    assert not electron_request.observation_admitted
    assert not electron_request.phase_inference_eligible
    assert not electron_request.semantic_ingress_declared
    assert not electron_request.control_admission_requested
    assert not electron_request.control_intent_created
    assert not electron_request.actionable
    assert not electron_request.execution_permitted
    assert not electron_request.direct_actuation
    assert electron_request.authority == "review_only"
    assert electron_request.machine_protection_final_veto


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
