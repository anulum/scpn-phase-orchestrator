# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor producer-evidence priority register guards

"""Keep producer-intake priority exact, multi-axis, and fail-closed."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA,
    CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION,
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGISTRY,
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
    FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST_SCHEMA,
    FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST_VERSION,
    conventional_tokamak_physical_payload_request,
    conventional_tokamak_physical_payload_request_digest,
    frc_compression_mif_physical_payload_request,
    frc_compression_mif_physical_payload_request_digest,
)

REGISTER = Path(
    "docs/reference/data/reactor_producer_evidence_priority_register.v1.json"
)
SCHEMA = Path("docs/specs/reactor_producer_evidence_priority_register.schema.json")
SOURCE_PATHS = {
    "configuration_evidence_coverage": Path(
        "docs/reference/data/reactor_configuration_evidence_coverage.v1.json"
    ),
    "diagnostic_plan_portfolio_status": Path(
        "docs/reference/data/reactor_diagnostic_plan_portfolio_status.v1.json"
    ),
    "signal_occurrence_ledger": Path(
        "docs/reference/data/reactor_signal_occurrence_ledger.v1.json"
    ),
    "technology_diagnostic_atlas": Path(
        "docs/reference/data/reactor_technology_diagnostic_atlas.v1.json"
    ),
}
LANES = (
    "L0_qualify_existing_physical_source",
    "L1_extend_exercised_review_adapter",
    "L2_build_from_accepted_plan",
    "L3_repair_refused_plan_before_intake",
)
REQUIRED_EVIDENCE = (
    "physical_sample",
    "phenomenon_identity",
    "reference",
    "clock_epoch",
    "observation_operator_or_calibration",
    "uncertainty",
    "validity",
    "quality",
    "provenance",
    "observability_gate",
)
MAST_L0_REQUIREMENTS = (
    "phenomenon_identity",
    "reproducible_source_ingestion_state",
    "calibration_lineage",
    "physical_geometry_and_frame_join",
    "modal_observation_operator_and_harmonic_basis",
    "provider_quality",
    "uncertainty",
    "validity",
    "instrument_facility_clock_correlation",
    "resolved_event_identity",
    "observability_threshold",
    "independent_multi_shot_or_classifier_evidence",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _payload() -> dict[str, object]:
    payload = _load(REGISTER)["payload"]
    assert isinstance(payload, dict)
    return payload


def _rows() -> list[dict[str, object]]:
    rows = _payload()["configurations"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return cast(list[dict[str, object]], rows)


def test_priority_register_matches_schema_and_payload_seal() -> None:
    schema = _load(SCHEMA)
    register = _load(REGISTER)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(register)
    assert (
        register["payload_sha256"]
        == hashlib.sha256(_canonical(register["payload"])).hexdigest()
    )


def test_priority_register_binds_exact_current_source_artifacts() -> None:
    payload = _payload()
    bindings = payload["source_bindings"]
    assert isinstance(bindings, dict)

    for name, path in SOURCE_PATHS.items():
        source = _load(path)
        binding = bindings[name]
        assert isinstance(binding, dict)
        assert binding == {
            "path": path.as_posix(),
            "schema": source["schema"],
            "schema_version": source["schema_version"],
            "payload_sha256": source["payload_sha256"],
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_priority_register_joins_every_exact_configuration_and_candidate() -> None:
    payload = _payload()
    rows = _rows()
    reactor_records = DEFAULT_REACTOR_REGISTRY.to_record()["configurations"]
    assert isinstance(reactor_records, list)

    assert [row["configuration"] for row in rows] == [
        record["identifier"] for record in reactor_records
    ]
    assert len({row["device_project"] for row in rows}) == 21

    scope = payload["project_scope"]
    assert isinstance(scope, dict)
    plan_payload = _load(SOURCE_PATHS["diagnostic_plan_portfolio_status"])["payload"]
    assert isinstance(plan_payload, dict)
    producers = plan_payload["producers"]
    assert isinstance(producers, list)
    assert all(isinstance(producer, dict) for producer in producers)
    plan_projects = {producer["project"] for producer in producers}
    registry_projects = {row["device_project"] for row in rows}
    upstream_projects = plan_projects | {"SCPN-FUSION-CORE", "SCPN-MIF-CORE"}

    assert set(scope["diagnostic_plan_portfolio_projects"]) == plan_projects
    assert set(scope["registry_device_owner_projects"]) == registry_projects
    assert set(scope["upstream_reactor_projects"]) == upstream_projects
    assert len(upstream_projects) == 22
    assert scope["control_project"] == "SCPN-CONTROL"
    assert scope["orchestration_authority_project"] == ("SCPN-PHASE-ORCHESTRATOR")

    for row, record in zip(rows, reactor_records, strict=True):
        identifier = cast(str, row["configuration"])
        profile = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(identifier)
        candidates = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.for_configuration(
            identifier
        )
        assert row["confinement_family"] == record["confinement_family"]
        assert row["topology"] == record["topology"]
        assert row["device_project"] == profile.device_project
        assert row["observability_candidate_ids"] == sorted(
            candidate.candidate_id for candidate in candidates
        )


def test_priority_register_joins_exact_current_plan_status() -> None:
    rows = _rows()
    plan_payload = _load(SOURCE_PATHS["diagnostic_plan_portfolio_status"])["payload"]
    assert isinstance(plan_payload, dict)
    producers = plan_payload["producers"]
    assert isinstance(producers, list)
    assert all(isinstance(producer, dict) for producer in producers)
    producers_by_project = {
        cast(str, producer["project"]): cast(dict[str, object], producer)
        for producer in producers
    }

    joined_configurations = 0
    for row in rows:
        project = cast(str, row["device_project"])
        diagnostic_plan = row["diagnostic_plan"]
        readiness = row["readiness_axes"]
        assert isinstance(diagnostic_plan, dict)
        assert isinstance(readiness, dict)
        producer = producers_by_project.get(project)
        if producer is None:
            assert row["configuration"] == "frc_compression_mif"
            assert diagnostic_plan == {
                "structural_status": "not_in_portfolio",
                "custody_state": "not_in_portfolio",
                "observed_revision": None,
                "fixture_sha256": None,
                "missing_required_members": [],
                "affected_channel_ids": [],
            }
            assert readiness["structural_plan_accepted"] is False
            assert readiness["exact_plan_fixture_custody"] is False
            continue

        joined_configurations += 1
        assert diagnostic_plan == {
            "structural_status": producer["structural_status"],
            "custody_state": producer["custody_state"],
            "observed_revision": producer["observed_revision"],
            "fixture_sha256": producer["fixture_sha256"],
            "missing_required_members": producer["missing_required_members"],
            "affected_channel_ids": producer["affected_channel_ids"],
        }
        assert readiness["structural_plan_accepted"] == (
            producer["structural_status"] == "accepted"
        )
        assert readiness["exact_plan_fixture_custody"] == (
            producer["custody_state"] == "exact_fixture_custody"
        )

    assert joined_configurations == 31
    assert all(
        row["readiness_axes"]["exact_plan_fixture_custody"] is False for row in rows
    )


def test_priority_lanes_follow_custody_precedence_not_external_rank() -> None:
    payload = _payload()
    rows = _rows()
    by_configuration = {row["configuration"]: row for row in rows}

    assert tuple(payload["lane_order"]) == LANES
    assert Counter(row["intake_lane"] for row in rows) == {
        LANES[0]: 1,
        LANES[1]: 2,
        LANES[2]: 29,
    }
    assert all(row["intake_lane"] != LANES[3] for row in rows)
    assert all(row["priority_score"] is None for row in rows)
    assert all(
        row["next_gate"] == "supply_physical_sample_envelope"
        for row in rows
        if row["intake_lane"] == LANES[2]
    )

    assert by_configuration["spherical_tokamak"]["intake_lane"] == LANES[0]
    assert by_configuration["conventional_tokamak"]["intake_lane"] == LANES[1]
    assert by_configuration["frc_compression_mif"]["intake_lane"] == LANES[1]
    assert by_configuration["dense_plasma_focus"]["intake_lane"] == LANES[2]
    assert by_configuration["field_reversed_configuration"]["intake_lane"] == (LANES[2])

    # External rank does not split configurations whose exact plans are both
    # structurally accepted.
    assert by_configuration["beam_target"]["intake_lane"] == LANES[2]
    assert by_configuration["colliding_beam"]["intake_lane"] == LANES[2]
    beam_context = by_configuration["beam_target"]["external_context"]
    colliding_context = by_configuration["colliding_beam"]["external_context"]
    assert isinstance(beam_context, dict)
    assert isinstance(colliding_context, dict)
    assert beam_context["evidence_rank"].startswith("E5_")
    assert colliding_context["evidence_rank"].startswith("E1_")


def test_priority_register_requests_evidence_without_granting_authority() -> None:
    payload = _payload()
    rows = _rows()
    by_configuration = {row["configuration"]: row for row in rows}

    assert payload["authority"] == "review_only"
    assert payload["actionable"] is False
    assert payload["direct_actuation_authorized"] is False
    assert payload["machine_protection_final_veto"] is True
    assert payload["counts"] == {
        "built_in_configurations": 32,
        "built_in_confinement_families": 8,
        "registry_device_owner_projects": 21,
        "upstream_reactor_projects": 22,
        "diagnostic_plan_portfolio_projects": 20,
        "control_projects": 1,
        "upstream_plus_control_projects": 23,
        LANES[0]: 1,
        LANES[1]: 2,
        LANES[2]: 29,
        LANES[3]: 0,
        "qualified_physical_observations": 0,
        "qualified_physical_phases": 0,
        "control_admitted": 0,
    }
    assert tuple(payload["required_physical_evidence"]) == REQUIRED_EVIDENCE

    assert (
        by_configuration["spherical_tokamak"]["producer_request"][
            "requested_owner_project"
        ]
        == "SCPN-FUSION-CORE"
    )
    mast_evidence = by_configuration["spherical_tokamak"]["current_spo_evidence"]
    mast_readiness = by_configuration["spherical_tokamak"]["readiness_axes"]
    mast_request = by_configuration["spherical_tokamak"]["producer_request"]
    assert isinstance(mast_evidence, dict)
    assert isinstance(mast_readiness, dict)
    assert isinstance(mast_request, dict)
    assert mast_evidence["producer_project"] == "SCPN-FUSION-CORE"
    assert mast_evidence["source_schema"] == (
        "scpn-fusion-core.mast-complete-magnetic-archive-envelope.v1"
    )
    assert mast_evidence["adapter_api"] == (
        "scpn_phase_orchestrator.reactor_semantics."
        "mast_magnetic_source_review_from_producer_bytes"
    )
    assert mast_evidence["portable_review_adapter_present"] is True
    assert mast_readiness["portable_review_adapter_present"] is True
    assert tuple(mast_request["lane_blockers"]) == MAST_L0_REQUIREMENTS
    conventional_request = by_configuration["conventional_tokamak"]["producer_request"]
    assert isinstance(conventional_request, dict)
    materialized = conventional_request["materialized_request"]
    assert isinstance(materialized, dict)
    runtime_request = conventional_tokamak_physical_payload_request()
    assert materialized == {
        "api": (
            "scpn_phase_orchestrator.reactor_semantics."
            "conventional_tokamak_physical_payload_request"
        ),
        "envelope_sha256": conventional_tokamak_physical_payload_request_digest(
            runtime_request
        ),
        "request_id": runtime_request.request_id,
        "schema": CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_SCHEMA,
        "schema_version": CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST_VERSION,
    }
    mif_request = by_configuration["frc_compression_mif"]["producer_request"]
    assert isinstance(mif_request, dict)
    assert mif_request["requested_owner_project"] == "SCPN-MIF-CORE"
    mif_materialized = mif_request["materialized_request"]
    assert isinstance(mif_materialized, dict)
    runtime_mif_request = frc_compression_mif_physical_payload_request()
    assert mif_materialized == {
        "api": (
            "scpn_phase_orchestrator.reactor_semantics."
            "frc_compression_mif_physical_payload_request"
        ),
        "envelope_sha256": frc_compression_mif_physical_payload_request_digest(
            runtime_mif_request
        ),
        "request_id": runtime_mif_request.request_id,
        "schema": FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST_SCHEMA,
        "schema_version": FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST_VERSION,
    }

    for row in rows:
        context = row["external_context"]
        current = row["current_spo_evidence"]
        readiness = row["readiness_axes"]
        request = row["producer_request"]
        assert isinstance(context, dict)
        assert isinstance(current, dict)
        assert isinstance(readiness, dict)
        assert isinstance(request, dict)
        assert context["technology_is_priority_score"] is False
        assert context["technology_is_signal_evidence"] is False
        assert context["technology_is_control_evidence"] is False
        assert current["qualified_observation"] is False
        assert current["qualified_physical_phase"] is False
        assert readiness["complete_physical_evidence"] is False
        assert readiness["control_admission"] is False
        assert tuple(request["required_evidence"]) == REQUIRED_EVIDENCE
        assert request["canonical_bytes_required"] is True
        assert request["independent_validation_required"] is True
        assert row["authority"] == "review_only"
        assert row["actionable"] is False
        assert row["direct_actuation_authorized"] is False
        assert row["machine_protection_final_veto"] is True
