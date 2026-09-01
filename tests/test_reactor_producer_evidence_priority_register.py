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
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGISTRY,
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
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


def test_priority_lanes_follow_custody_precedence_not_external_rank() -> None:
    payload = _payload()
    rows = _rows()
    by_configuration = {row["configuration"]: row for row in rows}

    assert tuple(payload["lane_order"]) == LANES
    assert Counter(row["intake_lane"] for row in rows) == {
        LANES[0]: 1,
        LANES[1]: 2,
        LANES[2]: 13,
        LANES[3]: 16,
    }
    assert all(row["priority_score"] is None for row in rows)

    assert by_configuration["spherical_tokamak"]["intake_lane"] == LANES[0]
    assert by_configuration["conventional_tokamak"]["intake_lane"] == LANES[1]
    assert by_configuration["frc_compression_mif"]["intake_lane"] == LANES[1]
    assert by_configuration["dense_plasma_focus"]["intake_lane"] == LANES[2]
    assert by_configuration["field_reversed_configuration"]["intake_lane"] == (LANES[3])

    # E5 context cannot bypass a refused plan, and E1 does not demote the same
    # device project into a different readiness lane.
    assert by_configuration["beam_target"]["intake_lane"] == LANES[3]
    assert by_configuration["colliding_beam"]["intake_lane"] == LANES[3]
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
        LANES[2]: 13,
        LANES[3]: 16,
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
    assert (
        by_configuration["frc_compression_mif"]["producer_request"][
            "requested_owner_project"
        ]
        == "SCPN-MIF-CORE"
    )

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
