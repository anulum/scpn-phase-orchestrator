# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO panel evidence-coverage data tests

"""Tests for the STUDIO panel evidence-coverage producer.

Assert the panel payload restates the assurance clause map faithfully: every
category's clauses resolve to catalogued standards/titles, the per-category and
summary counts are internally consistent, the coverage statuses stay honest
(only ``addressed`` / ``partially_addressed``), the category order is guarded
against drift, and the rendered JSON round-trips.
"""

from __future__ import annotations

import json

import pytest

# Full dotted import so the module-linkage guard
# (tools/check_test_module_linkage.py) sees the module's own import path.
import scpn_phase_orchestrator.studio.panel_data as panel_data
from scpn_phase_orchestrator.assurance.case import (
    ADDRESSED,
    CONFORMANCE_STATUSES,
    DEFAULT_EVIDENCE_CLAUSE_MAP,
    NOT_ADDRESSED,
    PARTIALLY_ADDRESSED,
)
from scpn_phase_orchestrator.assurance.evidence import EVIDENCE_CATEGORIES
from scpn_phase_orchestrator.assurance.standards import (
    REGULATORY_DISCLAIMER,
    clause_catalogue,
)
from scpn_phase_orchestrator.studio.panel_data import (
    PANEL_DATA_SCHEMA,
    STUDIO_ID,
    build_evidence_coverage_panel,
    render_panel_data_json,
)

_HONEST_STATUSES = {ADDRESSED, PARTIALLY_ADDRESSED}


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _str(value: object) -> str:
    assert isinstance(value, str)
    return value


def _categories(payload: dict[str, object]) -> list[dict[str, object]]:
    return [_dict(category) for category in _list(payload["categories"])]


def _clauses(category: dict[str, object]) -> list[dict[str, object]]:
    return [_dict(clause) for clause in _list(category["clauses"])]


def test_top_level_shape_and_identity() -> None:
    payload = build_evidence_coverage_panel()
    assert payload["schema"] == PANEL_DATA_SCHEMA == "spo.studio.evidence-coverage.v1"
    assert payload["studio"] == STUDIO_ID == "scpn-phase-orchestrator"
    assert payload["disclaimer"] == REGULATORY_DISCLAIMER
    assert set(payload) == {"schema", "studio", "disclaimer", "categories", "summary"}


def test_categories_cover_every_evidence_category_in_a_fixed_order() -> None:
    payload = build_evidence_coverage_panel()
    names = [_str(category["category"]) for category in _categories(payload)]
    # Every producer-defined evidence category appears exactly once.
    assert set(names) == set(EVIDENCE_CATEGORIES)
    assert len(names) == len(EVIDENCE_CATEGORIES)
    # The order is the assurance-lifecycle narrative, deterministic across calls.
    assert names == [
        "audit_logging",
        "replay_determinism",
        "formal_verification",
        "twin_confidence",
        "conformal_gate",
        "control_envelope",
    ]
    assert build_evidence_coverage_panel()["categories"] == payload["categories"]


def test_each_clause_resolves_to_a_catalogued_standard_and_title() -> None:
    payload = build_evidence_coverage_panel()
    catalogue = {clause.key: clause for clause in clause_catalogue()}
    for category in _categories(payload):
        clauses = _clauses(category)
        assert clauses, "every mapped category contributes at least one clause"
        for clause in clauses:
            key = f"{_str(clause['standard'])}::{_str(clause['clause_id'])}"
            assert key in catalogue
            assert clause["title"] == catalogue[key].title
            assert clause["status"] in _HONEST_STATUSES
            assert _str(clause["rationale"]).strip()


def test_per_category_counts_match_their_clauses() -> None:
    payload = build_evidence_coverage_panel()
    for category in _categories(payload):
        clauses = _clauses(category)
        addressed = sum(1 for clause in clauses if clause["status"] == ADDRESSED)
        partial = sum(
            1 for clause in clauses if clause["status"] == PARTIALLY_ADDRESSED
        )
        assert _int(category["clause_count"]) == len(clauses)
        assert _int(category["addressed_count"]) == addressed
        assert _int(category["partially_addressed_count"]) == partial
        assert addressed + partial == len(clauses)


def test_summary_totals_are_consistent_with_the_categories() -> None:
    payload = build_evidence_coverage_panel()
    categories = _categories(payload)
    summary = _dict(payload["summary"])
    assert _int(summary["category_count"]) == len(categories)
    assert _int(summary["clause_mapping_count"]) == sum(
        _int(category["clause_count"]) for category in categories
    )
    assert _int(summary["addressed_count"]) == sum(
        _int(category["addressed_count"]) for category in categories
    )
    assert _int(summary["partially_addressed_count"]) == sum(
        _int(category["partially_addressed_count"]) for category in categories
    )
    assert summary["standards_covered"] == [
        "ANSI/UL 4600",
        "EU AI Act 2024/1689",
        "ISO/IEC 42001:2023",
    ]


def test_statuses_are_a_subset_of_the_declared_conformance_statuses() -> None:
    payload = build_evidence_coverage_panel()
    seen = {
        clause["status"]
        for category in _categories(payload)
        for clause in _clauses(category)
    }
    assert seen <= CONFORMANCE_STATUSES
    assert seen == _HONEST_STATUSES  # the map carries no not_addressed entries


def test_render_json_round_trips_and_ends_with_a_newline() -> None:
    rendered = render_panel_data_json()
    assert rendered.endswith("\n")
    assert json.loads(rendered) == build_evidence_coverage_panel()
    # Non-ASCII (e.g. the disclaimer's punctuation) is kept verbatim, not escaped.
    assert "\\u" not in rendered


def test_category_order_drift_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dropping a category from the order must raise, never silently omit it.
    monkeypatch.setattr(
        panel_data,
        "_CATEGORY_ORDER",
        panel_data._CATEGORY_ORDER[:-1],
    )
    with pytest.raises(ValueError, match="drifted from the assurance clause map"):
        build_evidence_coverage_panel()


def test_not_addressed_status_is_excluded_from_both_tallies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # White-box: a future not_addressed contribution is listed but counted in
    # neither the addressed nor the partially_addressed tally.
    single = {
        "audit_logging": (
            ("EU AI Act 2024/1689::Article 12", NOT_ADDRESSED, "placeholder"),
        )
    }
    monkeypatch.setattr(panel_data, "DEFAULT_EVIDENCE_CLAUSE_MAP", single)
    monkeypatch.setattr(panel_data, "_CATEGORY_ORDER", ("audit_logging",))
    payload = build_evidence_coverage_panel()
    category = _categories(payload)[0]
    assert _int(category["clause_count"]) == 1
    assert _int(category["addressed_count"]) == 0
    assert _int(category["partially_addressed_count"]) == 0
    assert payload["summary"] == {
        "category_count": 1,
        "clause_mapping_count": 1,
        "addressed_count": 0,
        "partially_addressed_count": 0,
        "standards_covered": ["EU AI Act 2024/1689"],
    }


def test_default_map_is_unmutated_by_the_producer() -> None:
    # The producer only reads the shared clause map; it must not mutate it.
    before = dict(DEFAULT_EVIDENCE_CLAUSE_MAP)
    build_evidence_coverage_panel()
    assert before == DEFAULT_EVIDENCE_CLAUSE_MAP
