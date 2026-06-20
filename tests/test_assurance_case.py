# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case bundle tests

"""Catalogue integrity, hashing, evidence validation, and bundle assembly tests."""

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)
from scpn_phase_orchestrator.assurance.case import (
    ADDRESSED,
    ASSURANCE_CASE_SCHEMA,
    CONFORMANCE_STATUSES,
    DEFAULT_EVIDENCE_CLAUSE_MAP,
    NOT_ADDRESSED,
    PARTIALLY_ADDRESSED,
    AssuranceCaseBundle,
    ClauseConformance,
    build_assurance_case_bundle,
)
from scpn_phase_orchestrator.assurance.evidence import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    EVIDENCE_CATEGORIES,
    FORMAL_VERIFICATION,
    REPLAY_DETERMINISM,
    TWIN_CONFIDENCE,
    EvidenceItem,
    build_evidence_item,
)
from scpn_phase_orchestrator.assurance.standards import (
    REGULATORY_DISCLAIMER,
    SUPPORTED_STANDARDS,
    RegulatoryClause,
    clause_catalogue,
    clause_for_key,
)


# --------------------------------------------------------------------------- #
# _hashing
# --------------------------------------------------------------------------- #
def test_canonical_hash_is_order_independent_and_deterministic() -> None:
    left = canonical_record_hash({"a": 1, "b": [2, 3], "c": "x"})
    right = canonical_record_hash({"c": "x", "b": [2, 3], "a": 1})
    assert left == right
    assert len(left) == 64


def test_canonical_hash_changes_with_content() -> None:
    assert canonical_record_hash({"a": 1}) != canonical_record_hash({"a": 2})


def test_require_sha256_accepts_valid_digest() -> None:
    digest = canonical_record_hash({"a": 1})
    assert require_sha256(digest, "field") == digest


@pytest.mark.parametrize(
    "bad",
    ["", "abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65],
)
def test_require_sha256_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="SHA-256|hexadecimal"):
        require_sha256(bad, "field")


def test_require_sha256_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        require_sha256(12345, "field")


# --------------------------------------------------------------------------- #
# standards
# --------------------------------------------------------------------------- #
def test_catalogue_has_unique_keys_across_all_standards() -> None:
    catalogue = clause_catalogue()
    keys = [clause.key for clause in catalogue]
    assert len(keys) == len(set(keys))
    assert {clause.standard for clause in catalogue} == set(SUPPORTED_STANDARDS)


def test_clause_for_key_round_trips_and_rejects_unknown() -> None:
    for clause in clause_catalogue():
        assert clause_for_key(clause.key) is clause
    with pytest.raises(KeyError, match="unknown clause key"):
        clause_for_key("Nonexistent::Clause 1")


def test_regulatory_clause_record_is_json_safe() -> None:
    clause = clause_catalogue()[0]
    record = clause.to_audit_record()
    assert set(record) == {"standard", "clause_id", "title", "provenance"}
    json.dumps(record)


def test_disclaimer_disclaims_certification() -> None:
    assert "not constitute" in REGULATORY_DISCLAIMER
    assert "certification" in REGULATORY_DISCLAIMER


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
def test_build_evidence_item_computes_content_hash() -> None:
    item = build_evidence_item("e1", AUDIT_LOGGING, "summary", {"ok": True})
    assert item.content_hash == canonical_record_hash({"ok": True})
    json.dumps(item.to_audit_record())


def test_evidence_item_accepts_matching_explicit_hash() -> None:
    record = {"ok": True}
    digest = canonical_record_hash(record)
    item = EvidenceItem("e1", AUDIT_LOGGING, "s", record, content_hash=digest)
    assert item.content_hash == digest


def test_evidence_item_rejects_mismatching_hash() -> None:
    other = canonical_record_hash({"different": 1})
    with pytest.raises(ValueError, match="content_hash does not match"):
        EvidenceItem("e1", AUDIT_LOGGING, "s", {"ok": True}, content_hash=other)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"evidence_id": "  "}, "evidence_id"),
        ({"category": "bogus"}, "category"),
        ({"summary": "   "}, "summary"),
    ],
)
def test_evidence_item_field_validation(kwargs: dict[str, str], match: str) -> None:
    base = {
        "evidence_id": "e1",
        "category": AUDIT_LOGGING,
        "summary": "s",
        "record": {"ok": True},
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        EvidenceItem(**base)  # type: ignore[arg-type]


def test_evidence_item_rejects_non_mapping_record() -> None:
    with pytest.raises(ValueError, match="record must be a mapping"):
        EvidenceItem("e1", AUDIT_LOGGING, "s", ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_evidence_categories_match_default_map_keys() -> None:
    assert set(DEFAULT_EVIDENCE_CLAUSE_MAP) == EVIDENCE_CATEGORIES


# --------------------------------------------------------------------------- #
# DEFAULT_EVIDENCE_CLAUSE_MAP integrity
# --------------------------------------------------------------------------- #
def test_default_map_references_only_known_clauses_and_statuses() -> None:
    known = {clause.key for clause in clause_catalogue()}
    for category, contributions in DEFAULT_EVIDENCE_CLAUSE_MAP.items():
        assert category in EVIDENCE_CATEGORIES
        for clause_key, coverage, rationale in contributions:
            assert clause_key in known, clause_key
            assert coverage in {ADDRESSED, PARTIALLY_ADDRESSED}
            assert rationale.strip()


# --------------------------------------------------------------------------- #
# ClauseConformance
# --------------------------------------------------------------------------- #
def test_clause_conformance_record_is_json_safe() -> None:
    clause = clause_catalogue()[0]
    entry = ClauseConformance(clause, ADDRESSED, ("e1",), "because")
    json.dumps(entry.to_audit_record())
    assert entry.status in CONFORMANCE_STATUSES


@pytest.mark.parametrize(
    "status,evidence_ids,rationale,match",
    [
        ("bogus", ("e1",), "r", "status must be"),
        (NOT_ADDRESSED, ("e1",), "r", "no evidence_ids"),
        (ADDRESSED, (), "r", "at least one evidence_id"),
        (ADDRESSED, ("e1",), "   ", "rationale"),
    ],
)
def test_clause_conformance_validation(
    status: str, evidence_ids: tuple[str, ...], rationale: str, match: str
) -> None:
    clause = clause_catalogue()[0]
    with pytest.raises(ValueError, match=match):
        ClauseConformance(clause, status, evidence_ids, rationale)


# --------------------------------------------------------------------------- #
# build_assurance_case_bundle + AssuranceCaseBundle
# --------------------------------------------------------------------------- #
def _sample_evidence() -> list[EvidenceItem]:
    return [
        build_evidence_item("audit", AUDIT_LOGGING, "signed audit log", {"ok": True}),
        build_evidence_item("twin", TWIN_CONFIDENCE, "twin healthy", {"cov": 0.9}),
        build_evidence_item("cptc", CONFORMAL_GATE, "gate calibrated", {"a": 0.1}),
    ]


def test_bundle_covers_every_catalogue_clause_once() -> None:
    bundle = build_assurance_case_bundle("sys", _sample_evidence())
    catalogue_keys = [clause.key for clause in clause_catalogue()]
    bundle_keys = [entry.clause.key for entry in bundle.conformance]
    assert sorted(bundle_keys) == sorted(catalogue_keys)
    assert bundle.schema == ASSURANCE_CASE_SCHEMA
    assert bundle.actuation_permitted is False


def test_bundle_hash_is_deterministic_and_order_independent() -> None:
    evidence = _sample_evidence()
    first = build_assurance_case_bundle("sys", evidence)
    reversed_bundle = build_assurance_case_bundle("sys", list(reversed(evidence)))
    assert first.bundle_hash == reversed_bundle.bundle_hash


def test_bundle_addresses_audit_record_keeping_and_flags_gaps() -> None:
    bundle = build_assurance_case_bundle("sys", _sample_evidence())
    by_key = {entry.clause.key: entry for entry in bundle.conformance}
    record_keeping = by_key["EU AI Act 2024/1689::Article 12"]
    assert record_keeping.status == ADDRESSED
    assert "audit" in record_keeping.evidence_ids
    # A governance clause with no technical evidence stays not_addressed.
    transparency = by_key["EU AI Act 2024/1689::Article 13"]
    assert transparency.status == NOT_ADDRESSED
    assert transparency.evidence_ids == ()


def test_partial_status_when_no_full_evidence_present() -> None:
    # replay alone only partially addresses ISO Clause 9 (performance evaluation).
    replay_only = [
        build_evidence_item("replay", REPLAY_DETERMINISM, "deterministic", {"ok": True})
    ]
    bundle = build_assurance_case_bundle("sys", replay_only)
    by_key = {entry.clause.key: entry for entry in bundle.conformance}
    assert by_key["ISO/IEC 42001:2023::Clause 9"].status == PARTIALLY_ADDRESSED


def test_coverage_summary_totals_match_catalogue() -> None:
    bundle = build_assurance_case_bundle("sys", _sample_evidence())
    summary = bundle.coverage_summary()
    assert set(summary) == set(SUPPORTED_STANDARDS)
    total = sum(bucket["total"] for bucket in summary.values())
    assert total == len(clause_catalogue())
    for bucket in summary.values():
        assert (
            bucket[ADDRESSED] + bucket[PARTIALLY_ADDRESSED] + bucket[NOT_ADDRESSED]
            == bucket["total"]
        )


def test_bundle_to_audit_record_is_json_safe_and_carries_disclaimer() -> None:
    bundle = build_assurance_case_bundle("sys", _sample_evidence())
    record = bundle.to_audit_record()
    json.dumps(record)
    assert record["disclaimer"] == REGULATORY_DISCLAIMER
    assert record["actuation_permitted"] is False
    assert record["bundle_hash"] == bundle.bundle_hash


def test_build_rejects_duplicate_evidence_ids() -> None:
    dup = [
        build_evidence_item("x", AUDIT_LOGGING, "a", {"v": 1}),
        build_evidence_item("x", TWIN_CONFIDENCE, "b", {"v": 2}),
    ]
    with pytest.raises(ValueError, match="unique"):
        build_assurance_case_bundle("sys", dup)


def test_formal_evidence_addresses_ul_safety_case() -> None:
    formal = [
        build_evidence_item("formal", FORMAL_VERIFICATION, "package", {"hash": "x"})
    ]
    bundle = build_assurance_case_bundle("sys", formal)
    by_key = {entry.clause.key: entry for entry in bundle.conformance}
    assert by_key["ANSI/UL 4600::safety-case"].status == ADDRESSED


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("system_name", "  ", "system_name"),
        ("version", "  ", "version"),
        ("schema", "wrong", "schema must be"),
        ("actuation_permitted", True, "review-only"),
    ],
)
def test_bundle_field_validation(field: str, value: object, match: str) -> None:
    base = build_assurance_case_bundle("sys", _sample_evidence())
    kwargs: dict[str, object] = {
        "system_name": base.system_name,
        "version": base.version,
        "evidence": base.evidence,
        "conformance": base.conformance,
        "standards_covered": base.standards_covered,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        AssuranceCaseBundle(**kwargs)  # type: ignore[arg-type]


def test_bundle_rejects_conformance_referencing_unknown_evidence() -> None:
    clause = clause_catalogue()[0]
    bogus = ClauseConformance(clause, ADDRESSED, ("ghost",), "r")
    with pytest.raises(ValueError, match="unknown evidence_ids"):
        AssuranceCaseBundle(
            system_name="sys",
            version="1.0.0",
            evidence=(),
            conformance=(bogus,),
            standards_covered=tuple(SUPPORTED_STANDARDS),
        )


def test_bundle_rejects_mismatching_bundle_hash() -> None:
    base = build_assurance_case_bundle("sys", _sample_evidence())
    wrong = canonical_record_hash({"not": "the seed"})
    with pytest.raises(ValueError, match="bundle_hash does not match"):
        AssuranceCaseBundle(
            system_name=base.system_name,
            version=base.version,
            evidence=base.evidence,
            conformance=base.conformance,
            standards_covered=base.standards_covered,
            bundle_hash=wrong,
        )


def test_bundle_rejects_duplicate_evidence_ids_in_constructor() -> None:
    item = build_evidence_item("dup", AUDIT_LOGGING, "a", {"v": 1})
    other = build_evidence_item("dup", TWIN_CONFIDENCE, "b", {"v": 2})
    with pytest.raises(ValueError, match="unique"):
        AssuranceCaseBundle(
            system_name="sys",
            version="1.0.0",
            evidence=(item, other),
            conformance=(),
            standards_covered=tuple(SUPPORTED_STANDARDS),
        )


def test_regulatory_clause_is_hashable_and_frozen() -> None:
    clause = clause_catalogue()[0]
    assert isinstance(hash(clause), int)
    with pytest.raises(AttributeError):
        clause.title = "mutated"  # type: ignore[misc]
    assert isinstance(clause, RegulatoryClause)
