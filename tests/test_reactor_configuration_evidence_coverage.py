# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reactor-configuration evidence coverage guards

"""Verify exhaustive, fail-closed evidence coverage for all reactor types."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGISTRY,
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
)

COVERAGE_PATH = Path(
    "docs/reference/data/reactor_configuration_evidence_coverage.v1.json"
)
SCHEMA_PATH = Path("docs/specs/reactor_configuration_evidence_coverage.schema.json")
OCCURRENCE_PATH = Path("docs/reference/data/reactor_signal_occurrence_ledger.v1.json")

EXPECTED_SOURCE_MAP = {
    "conventional_tokamak": (
        "verified_review_adapter_simulation",
        "exact_contract_identity",
        ("FUS-008", "SPO-006", "CTRL-001"),
    ),
    "field_reversed_configuration": (
        "local_model_only",
        "explicit_source_domain",
        ("FUS-004", "FUS-005"),
    ),
    "frc_compression_mif": (
        "verified_review_adapter_simulation",
        "exact_contract_identity",
        ("MIF-001", "SPO-007", "CTRL-002"),
    ),
    "spherical_tokamak": (
        "physical_source_unqualified",
        "exact_source_review",
        ("FUS-009", "FUS-010", "SPO-009"),
    ),
    "stellarator": (
        "synthetic_replay_only",
        "explicit_source_domain",
        ("FUS-011",),
    ),
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rows() -> list[dict[str, object]]:
    payload = _load(COVERAGE_PATH)["payload"]
    assert isinstance(payload, dict)
    rows = payload["configurations"]
    assert isinstance(rows, list)
    return rows


def test_coverage_matches_strict_schema_and_payload_seal() -> None:
    coverage = _load(COVERAGE_PATH)
    schema = _load(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(coverage)
    assert (
        coverage["payload_sha256"]
        == hashlib.sha256(_canonical(coverage["payload"])).hexdigest()
    )


def test_coverage_is_exact_join_of_all_three_public_registries() -> None:
    coverage = _load(COVERAGE_PATH)
    payload = coverage["payload"]
    assert isinstance(payload, dict)
    bindings = payload["source_bindings"]
    assert isinstance(bindings, dict)
    rows = _rows()
    reactor_records = DEFAULT_REACTOR_REGISTRY.to_record()["configurations"]

    assert bindings["reactor_configuration_registry"]["version"] == (
        DEFAULT_REACTOR_REGISTRY.version
    )
    assert bindings["reactor_configuration_registry"]["digest_sha256"] == (
        DEFAULT_REACTOR_REGISTRY.digest
    )
    assert bindings["reactor_observability_registry"]["digest_sha256"] == (
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
    )
    assert bindings["reactor_semantic_profile_registry"]["digest_sha256"] == (
        DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
    )
    assert [row["configuration"] for row in rows] == [
        item["identifier"] for item in reactor_records
    ]
    observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY

    for row, configuration in zip(rows, reactor_records, strict=True):
        identifier = configuration["identifier"]
        profile = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve(
            identifier
        ).to_record()
        assert row["confinement_family"] == configuration["confinement_family"]
        assert row["topology"] == configuration["topology"]
        assert row["device_project"] == profile["device_project"]
        assert row["semantic_ingress_state"] == profile["ingress_state"]
        assert row["semantic_profile"] == profile["semantic_profile"]
        assert row["producer_project"] == profile["producer_project"]
        assert row["adapter_api"] == profile["adapter_api"]
        assert row["observability_candidate_ids"] == sorted(
            candidate.candidate_id
            for candidate in observability.for_configuration(identifier)
        )


def test_coverage_counts_distinguish_source_evidence_from_semantic_ingress() -> None:
    rows = _rows()
    payload = _load(COVERAGE_PATH)["payload"]
    assert isinstance(payload, dict)

    counts = {
        "built_in_configurations": len(rows),
        "built_in_confinement_families": len(
            {row["confinement_family"] for row in rows}
        ),
        "evidence_source_present": sum(
            row["evidence_source_present"] is True for row in rows
        ),
        "evidence_source_absent": sum(
            row["evidence_source_present"] is False for row in rows
        ),
        "verified_review_adapters": sum(
            row["portable_review_adapter_present"] is True for row in rows
        ),
        "semantic_producerless": sum(
            row["semantic_producerless"] is True for row in rows
        ),
        "physical_source_unqualified": sum(
            row["evidence_state"] == "physical_source_unqualified" for row in rows
        ),
        "qualified_observations": sum(
            row["qualified_observation"] is True for row in rows
        ),
        "qualified_physical_phases": sum(
            row["qualified_physical_phase"] is True for row in rows
        ),
    }
    assert payload["coverage"] == counts
    assert Counter(row["evidence_state"] for row in rows) == {
        "producerless": 27,
        "verified_review_adapter_simulation": 2,
        "local_model_only": 1,
        "physical_source_unqualified": 1,
        "synthetic_replay_only": 1,
    }


def test_only_exact_evidence_bindings_reference_occurrence_ledger_rows() -> None:
    rows = _rows()
    occurrence = _load(OCCURRENCE_PATH)
    occurrence_payload = occurrence["payload"]
    assert isinstance(occurrence_payload, dict)
    occurrence_rows = occurrence_payload["occurrences"]
    assert isinstance(occurrence_rows, list)
    occurrence_ids = {row["occurrence_id"] for row in occurrence_rows}
    bindings = _load(COVERAGE_PATH)["payload"]["source_bindings"]

    assert (
        bindings["reactor_signal_occurrence_ledger"]["payload_sha256"]
        == (occurrence["payload_sha256"])
    )
    for row in rows:
        identifier = row["configuration"]
        if identifier in EXPECTED_SOURCE_MAP:
            expected_state, expected_binding, expected_occurrences = (
                EXPECTED_SOURCE_MAP[identifier]
            )
            assert row["evidence_state"] == expected_state
            assert row["evidence_binding"] == expected_binding
            assert tuple(row["occurrence_ids"]) == expected_occurrences
            assert set(row["occurrence_ids"]) <= occurrence_ids
            assert row["evidence_source_present"] is True
            assert row["evidence_producerless"] is False
            assert row["evidence_producerless_reason"] is None
        else:
            assert row["evidence_state"] == "producerless"
            assert row["evidence_binding"] == "none"
            assert row["evidence_source_projects"] == []
            assert row["occurrence_ids"] == []
            assert row["evidence_source_present"] is False
            assert row["evidence_producerless"] is True
            assert row["evidence_producerless_reason"]


def test_only_two_exact_portable_adapters_are_present() -> None:
    rows = {row["configuration"]: row for row in _rows()}
    adapter_configs = {
        identifier
        for identifier, row in rows.items()
        if row["portable_review_adapter_present"]
    }

    assert adapter_configs == {"conventional_tokamak", "frc_compression_mif"}
    assert rows["conventional_tokamak"]["source_schema"] == (
        "scpn-fusion-core.torax-runtime-review-envelope.v1"
    )
    assert rows["conventional_tokamak"]["handoff_schema"] == (
        "scpn-phase-orchestrator.reactor-semantic-handoff.v1"
    )
    assert rows["frc_compression_mif"]["source_schema"] == (
        "scpn-mif-core.merge-compression-observation.v1"
    )
    assert rows["frc_compression_mif"]["handoff_schema"] == (
        "scpn-phase-orchestrator.mif-merge-compression-handoff.v1"
    )

    for identifier, row in rows.items():
        if identifier in adapter_configs:
            assert row["semantic_producerless"] is False
            assert row["semantic_producerless_reason"] is None
        else:
            assert row["semantic_producerless"] is True
            assert row["semantic_producerless_reason"]


def test_no_row_inherits_observation_phase_or_actuation_authority() -> None:
    rows = _rows()
    physical = [row for row in rows if row["physical_source_present"]]

    assert [row["configuration"] for row in physical] == ["spherical_tokamak"]
    for row in rows:
        assert row["authority"] == "review_only"
        assert row["actionable"] is False
        assert row["direct_actuation_authorized"] is False
        assert row["machine_protection_final_veto"] is True
        assert row["qualified_observation"] is False
        assert row["qualified_physical_phase"] is False
