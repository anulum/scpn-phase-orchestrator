# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reactor technology and diagnostic atlas guards

"""Verify the evidence-ranked reactor technology and diagnostic atlas."""

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

ATLAS_PATH = Path("docs/reference/data/reactor_technology_diagnostic_atlas.v1.json")
SCHEMA_PATH = Path("docs/specs/reactor_technology_diagnostic_atlas.schema.json")

RANK_ORDER = (
    "E5_integrated_fusion_observation",
    "E4_integrated_plasma_experiment",
    "E3_component_or_driver_experiment",
    "E2_engineering_or_facility_development",
    "E1_concept_or_simulation",
    "E0_no_qualifying_source",
)
QUALIFICATION_GAPS = (
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
    """Load one repository-owned JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> bytes:
    """Return the byte-canonical JSON form used for payload seals."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload() -> dict[str, object]:
    """Return the atlas payload with a checked object boundary."""
    payload = _load(ATLAS_PATH)["payload"]
    assert isinstance(payload, dict)
    return payload


def _rows() -> list[dict[str, object]]:
    """Return atlas rows with checked object boundaries."""
    rows = _payload()["configurations"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    return cast(list[dict[str, object]], rows)


def test_atlas_matches_strict_schema_and_payload_seal() -> None:
    """Reject structural drift and non-canonical payload digests."""
    atlas = _load(ATLAS_PATH)
    schema = _load(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(atlas)
    assert (
        atlas["payload_sha256"]
        == hashlib.sha256(_canonical(atlas["payload"])).hexdigest()
    )


def test_atlas_is_an_exact_join_of_public_reactor_registries() -> None:
    """Bind every row to exact reactor, project, and candidate identities."""
    payload = _payload()
    rows = _rows()
    bindings = payload["source_bindings"]
    assert isinstance(bindings, dict)
    reactor_records = DEFAULT_REACTOR_REGISTRY.to_record()["configurations"]
    assert isinstance(reactor_records, list)

    assert bindings["reactor_configuration_registry"] == {
        "version": DEFAULT_REACTOR_REGISTRY.version,
        "digest_sha256": DEFAULT_REACTOR_REGISTRY.digest,
        "source_path": "src/scpn_phase_orchestrator/reactor_semantics/registry.py",
        "source_sha256": hashlib.sha256(
            Path(
                "src/scpn_phase_orchestrator/reactor_semantics/registry.py"
            ).read_bytes()
        ).hexdigest(),
    }
    assert bindings["reactor_observability_registry"] == {
        "version": DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version,
        "digest_sha256": DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest,
        "source_path": (
            "src/scpn_phase_orchestrator/reactor_semantics/observability_profiles.py"
        ),
        "source_sha256": hashlib.sha256(
            Path(
                "src/scpn_phase_orchestrator/reactor_semantics/observability_profiles.py"
            ).read_bytes()
        ).hexdigest(),
    }
    assert [row["configuration"] for row in rows] == [
        record["identifier"] for record in reactor_records
    ]

    for row, record in zip(rows, reactor_records, strict=True):
        identifier = record["identifier"]
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


def test_primary_source_graph_is_closed_and_configuration_specific() -> None:
    """Require every claim to resolve to a source for the exact configuration."""
    payload = _payload()
    rows = _rows()
    sources = payload["sources"]
    assert isinstance(sources, list)
    assert all(isinstance(source, dict) for source in sources)
    source_by_id = {source["id"]: source for source in sources}

    assert len(source_by_id) == len(sources)
    assert list(source_by_id) == sorted(source_by_id)
    covered_configurations = {
        configuration
        for source in sources
        for configuration in source["configurations"]
    }
    assert covered_configurations == {row["configuration"] for row in rows}
    assert payload["counts"]["primary_sources"] == len(sources)

    for source in sources:
        assert source["url"].startswith("https://")
        assert source["year"] <= 2026
        assert source["configurations"] == sorted(source["configurations"])

    for row in rows:
        identifier = row["configuration"]
        source_ids = row["source_ids"]
        assert source_ids == sorted(source_ids)
        for source_id in source_ids:
            assert identifier in source_by_id[source_id]["configurations"]
        for system in row["reference_systems"]:
            assert set(system["source_ids"]) <= set(source_ids)
        for claim in row["capability_claims"]:
            assert set(claim["source_ids"]) <= set(source_ids)
            assert all(
                identifier in source_by_id[source_id]["configurations"]
                for source_id in claim["source_ids"]
            )


def test_rank_semantics_do_not_promote_component_or_concept_evidence() -> None:
    """Keep fusion observations, plasma experiments, and components distinct."""
    rows = _rows()
    payload = _payload()

    assert tuple(payload["rank_order"]) == RANK_ORDER
    ranks = Counter(row["external_evidence_rank"] for row in rows)
    assert ranks == {
        "E5_integrated_fusion_observation": 7,
        "E4_integrated_plasma_experiment": 18,
        "E3_component_or_driver_experiment": 5,
        "E1_concept_or_simulation": 2,
    }

    for row in rows:
        rank = row["external_evidence_rank"]
        claims = row["capability_claims"]
        capabilities = [claim["capability_id"] for claim in claims]
        assert len(capabilities) == len(set(capabilities))
        if rank == "E5_integrated_fusion_observation":
            assert any(
                claim["capability_id"] == "fusion_product_measurement"
                and claim["status"] == "observed_integrated"
                for claim in claims
            )
            assert "integrated_fusion_output" not in row["missing_capabilities"]
        else:
            assert "integrated_fusion_output" in row["missing_capabilities"]
        if rank == "E3_component_or_driver_experiment":
            assert all(claim["status"] != "observed_integrated" for claim in claims)
        if rank == "E1_concept_or_simulation":
            assert all(claim["status"] == "concept_only" for claim in claims)


def test_external_technology_evidence_never_qualifies_spo_signals() -> None:
    """Fail closed across phase, observation, CONTROL, and actuation boundaries."""
    rows = _rows()
    payload = _payload()

    assert payload["authority"] == "review_only"
    assert payload["actionable"] is False
    assert payload["direct_actuation_authorized"] is False
    assert payload["machine_protection_final_veto"] is True
    assert payload["counts"] == {
        "built_in_configurations": 32,
        "built_in_confinement_families": 8,
        "primary_sources": 34,
        "physical_observations_qualified": 0,
        "physical_phases_qualified": 0,
        "control_admitted": 0,
    }

    for row in rows:
        qualification = row["spo_qualification"]
        assert qualification == {
            "admission_state": "refused_no_producer_evidence",
            "missing_evidence": list(QUALIFICATION_GAPS),
            "physical_observation_qualified": False,
            "physical_phase_qualified": False,
            "control_admitted": False,
        }
        assert row["technology_is_control_evidence"] is False
        assert row["authority"] == "review_only"
        assert row["actionable"] is False
        assert row["direct_actuation_authorized"] is False
        assert row["machine_protection_final_veto"] is True
        assert row["missing_capabilities"] == sorted(row["missing_capabilities"])
        assert "public_machine_readable_calibrated_data" in row["missing_capabilities"]
        assert "spo_producer_adapter" in row["missing_capabilities"]
        assert "control_admission" in row["missing_capabilities"]
