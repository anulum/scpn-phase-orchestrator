# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO panel evidence-coverage data

"""Assemble the honest data the SPO STUDIO federation panel renders.

The federated studio panel (``studio-web/``) is a pure renderer: it never
computes and never upgrades a claim. Its one committed data surface is the
**evidence-coverage map** — for each of SPO's six assurance evidence categories
(:data:`~scpn_phase_orchestrator.assurance.evidence.EVIDENCE_CATEGORIES`), the
regulatory clauses that category contributes to and whether the contribution is
``addressed`` or ``partially_addressed``. The map is the same one the
assurance-case bundle uses
(:data:`~scpn_phase_orchestrator.assurance.case.DEFAULT_EVIDENCE_CLAUSE_MAP`),
so the panel restates the repository's own honest self-description rather than a
hand-authored copy that could drift.

The panel bundle is inlined into the JavaScript remote at build time as
``studio-web/src/panel/evidence_coverage.json``; :func:`render_panel_data_json`
produces that file and a drift-guard test asserts the committed snapshot equals
the live producer output, keeping the two in lock-step.
"""

from __future__ import annotations

import json

from scpn_phase_orchestrator.assurance.case import (
    ADDRESSED,
    DEFAULT_EVIDENCE_CLAUSE_MAP,
    PARTIALLY_ADDRESSED,
)
from scpn_phase_orchestrator.assurance.evidence import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    CONTROL_ENVELOPE,
    FORMAL_VERIFICATION,
    REPLAY_DETERMINISM,
    TWIN_CONFIDENCE,
)
from scpn_phase_orchestrator.assurance.standards import (
    REGULATORY_DISCLAIMER,
    clause_for_key,
)

__all__ = [
    "PANEL_DATA_SCHEMA",
    "STUDIO_ID",
    "build_evidence_coverage_panel",
    "render_panel_data_json",
]

STUDIO_ID = "scpn-phase-orchestrator"
"""The stable federation id, matching the STUDIO capability manifest."""

PANEL_DATA_SCHEMA = "spo.studio.evidence-coverage.v1"
"""Schema identifier for the panel evidence-coverage payload."""

# The evidence categories in their assurance-lifecycle narrative order: what the
# run records (audit), that the record replays (replay), the models it satisfies
# (formal), the live divergence it tracks (twin), the admission it gates
# (conformal), and the closed loop it bounds (control). A drift from the clause
# map's key set is a fail-closed error, mirroring the assurance-case guard.
_CATEGORY_ORDER: tuple[str, ...] = (
    AUDIT_LOGGING,
    REPLAY_DETERMINISM,
    FORMAL_VERIFICATION,
    TWIN_CONFIDENCE,
    CONFORMAL_GATE,
    CONTROL_ENVELOPE,
)


def build_evidence_coverage_panel() -> dict[str, object]:
    """Return the JSON-serialisable evidence-coverage panel payload.

    For every assurance evidence category, the payload lists the regulatory
    clauses the category contributes to — resolved to their catalogued standard,
    identifier and official title — with the ``addressed`` /
    ``partially_addressed`` status and rationale taken verbatim from
    :data:`~scpn_phase_orchestrator.assurance.case.DEFAULT_EVIDENCE_CLAUSE_MAP`.
    A summary block totals the categories, clause mappings, and coverage status
    counts, and names the standards spanned.

    Returns
    -------
    dict[str, object]
        The panel payload: ``schema``, ``studio``, ``disclaimer``, ordered
        ``categories`` (each with its ``clauses`` and per-category counts), and a
        ``summary``.

    Raises
    ------
    ValueError
        If :data:`_CATEGORY_ORDER` has drifted from the clause-map key set, so a
        renamed or added category can never silently drop from the panel.
    """
    if set(_CATEGORY_ORDER) != set(DEFAULT_EVIDENCE_CLAUSE_MAP):
        raise ValueError(
            "the studio panel category order has drifted from the assurance "
            "clause map; align _CATEGORY_ORDER with DEFAULT_EVIDENCE_CLAUSE_MAP"
        )

    categories: list[dict[str, object]] = []
    standards_covered: set[str] = set()
    total_addressed = 0
    total_partial = 0
    total_mappings = 0

    for category in _CATEGORY_ORDER:
        clause_records: list[dict[str, str]] = []
        addressed = 0
        partial = 0
        for key, status, rationale in DEFAULT_EVIDENCE_CLAUSE_MAP[category]:
            clause = clause_for_key(key)
            clause_records.append(
                {
                    "standard": clause.standard,
                    "clause_id": clause.clause_id,
                    "title": clause.title,
                    "status": status,
                    "rationale": rationale,
                }
            )
            standards_covered.add(clause.standard)
            if status == ADDRESSED:
                addressed += 1
            elif status == PARTIALLY_ADDRESSED:
                partial += 1
        total_addressed += addressed
        total_partial += partial
        total_mappings += len(clause_records)
        categories.append(
            {
                "category": category,
                "clause_count": len(clause_records),
                "addressed_count": addressed,
                "partially_addressed_count": partial,
                "clauses": clause_records,
            }
        )

    summary: dict[str, object] = {
        "category_count": len(categories),
        "clause_mapping_count": total_mappings,
        "addressed_count": total_addressed,
        "partially_addressed_count": total_partial,
        "standards_covered": sorted(standards_covered),
    }
    return {
        "schema": PANEL_DATA_SCHEMA,
        "studio": STUDIO_ID,
        "disclaimer": REGULATORY_DISCLAIMER,
        "categories": categories,
        "summary": summary,
    }


def render_panel_data_json() -> str:
    """Return the panel payload as a canonical, human-diffable JSON document.

    The document is indented two spaces, keeps non-ASCII characters verbatim, and
    ends with a trailing newline, matching the committed
    ``studio-web/src/panel/evidence_coverage.json`` snapshot.

    Returns
    -------
    str
        The rendered JSON document with a trailing newline.
    """
    return (
        json.dumps(build_evidence_coverage_panel(), indent=2, ensure_ascii=False) + "\n"
    )
