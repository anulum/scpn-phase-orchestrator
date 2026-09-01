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
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
)

COVERAGE_PATH = Path(
    "docs/reference/data/reactor_configuration_evidence_coverage.v1.json"
)
SCHEMA_PATH = Path("docs/specs/reactor_configuration_evidence_coverage.schema.json")
OCCURRENCE_PATH = Path("docs/reference/data/reactor_signal_occurrence_ledger.v1.json")
MIF_FIXTURE_PATH = Path(
    "tests/fixtures/mif_merge_compression/mif_merge_compression_observation_v1.json"
)

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


def test_frc_mif_review_chain_receipt_seals_all_three_non_actuating_stages() -> None:
    rows = {row["configuration"]: row for row in _rows()}
    receipt_rows = {
        identifier: row["review_chain_receipts"]
        for identifier, row in rows.items()
        if "review_chain_receipts" in row
    }
    receipt_ids = [
        receipt["receipt_id"]
        for configuration, receipts in receipt_rows.items()
        for receipt in receipts
        if receipt["configuration"] == configuration
    ]

    assert set(receipt_rows) == {"frc_compression_mif"}
    assert len(receipt_ids) == sum(len(receipts) for receipts in receipt_rows.values())
    assert len(receipt_ids) == len(set(receipt_ids))
    assert len(receipt_rows["frc_compression_mif"]) == 1
    receipt = receipt_rows["frc_compression_mif"][0]
    assert receipt["receipt_id"] == "frc_compression_mif.simulation_review.v1"
    assert receipt["configuration"] == "frc_compression_mif"
    assert receipt["evidence_class"] == "simulation"

    source_bytes = MIF_FIXTURE_PATH.read_bytes()
    assert len(source_bytes) == receipt["producer"]["byte_length"] == 2_475
    assert (
        hashlib.sha256(source_bytes).hexdigest()
        == receipt["producer"]["envelope_sha256"]
    )
    assert receipt["producer"]["envelope_sha256"] == (
        "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
    )
    handoff = mif_merge_compression_handoff_from_mif_bytes(
        source_bytes,
        expected_sha256=receipt["producer"]["envelope_sha256"],
    )
    handoff_bytes = mif_merge_compression_handoff_to_bytes(handoff)
    assert len(handoff_bytes) == receipt["semantic_handoff"]["byte_length"] == 101_652
    assert (
        hashlib.sha256(handoff_bytes).hexdigest()
        == receipt["semantic_handoff"]["envelope_sha256"]
    )
    assert receipt["semantic_handoff"]["envelope_sha256"] == (
        "c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2"
    )

    assert receipt["semantic_handoff"]["package_version"] == "1.3.1"
    assert receipt["semantic_handoff"]["package_source_revision"] == (
        "c2a7581d58819060806c6f173da941c822103695"
    )
    assert receipt["semantic_handoff"]["package_wheel_sha256"] == (
        "c2d7c0a5c0ad47f420fee02e54ccc28122bf8d128eb3b80ca51ba5f034320274"
    )
    assert receipt["control_review"] == {
        "project": "SCPN-CONTROL",
        "package_version": "0.23.0",
        "receiver_api": (
            "scpn_control.reactor_semantic_admission.admit_mif_reactor_semantic_handoff"
        ),
        "schema": "scpn-control.reactor-semantic-admission.v1",
        "schema_version": "1.0.0",
        "decision": "admitted_for_review",
        "byte_length": 964,
        "decision_digest": (
            "d1900dacb70893d080bd6c6902a00a68e08920d39457a4240ce89f0db0bac8c9"
        ),
        "envelope_sha256": (
            "50be73641cc6b4f59cc95403c6421d9442e6a19219a0ecce160cd1646385da75"
        ),
        "review_only": True,
        "actionable": False,
    }
    assert {source["project"] for source in receipt["verification_sources"]} == {
        "SCPN-MIF-CORE",
        "SCPN-PHASE-ORCHESTRATOR",
        "SCPN-CONTROL",
    }
    assert len(receipt["verification_sources"]) == 8
    assert [
        (source["project"], source["path"])
        for source in receipt["verification_sources"]
    ] == sorted(
        (source["project"], source["path"])
        for source in receipt["verification_sources"]
    )
    assert receipt["host_independent"] is True
    assert receipt["sibling_source_execution"] is False
    assert receipt["physical_source_present"] is False
    assert receipt["physical_observation_admitted"] is False
    assert receipt["qualified_physical_phase"] is False
    assert receipt["control_action_created"] is False
    assert receipt["authority"] == "review_only"
    assert receipt["actionable"] is False


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
