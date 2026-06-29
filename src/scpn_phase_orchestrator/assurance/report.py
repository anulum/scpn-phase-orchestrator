# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case conformity report renderer

"""Render an assurance-case bundle as a human-readable conformity report.

The certification evidence package seals machine-readable JSON (the bundle, the
hash test vectors, and a manifest). A regulatory assessor, however, reads a
document: this module renders the same sealed evidence as a deterministic
Markdown conformity report — a per-standard, clause-by-clause table of
conformance status, contributing evidence, and rationale, prefixed by the
coverage rollup and the regulatory disclaimer and anchored to the bundle hash for
traceability.

The report is review-only and adds no new claims: every status, evidence
identifier, and rationale is read verbatim from the
:class:`~scpn_phase_orchestrator.assurance.case.AssuranceCaseBundle`. Rendering is
deterministic — standards, clauses, and evidence are emitted in a stable sort
order so the report digest is reproducible.
"""

from __future__ import annotations

from scpn_phase_orchestrator.assurance.case import (
    ADDRESSED,
    NOT_ADDRESSED,
    PARTIALLY_ADDRESSED,
    AssuranceCaseBundle,
)

CONFORMITY_REPORT_SCHEMA = "scpn_conformity_report_v1"

_STATUS_LABELS: dict[str, str] = {
    ADDRESSED: "Addressed",
    PARTIALLY_ADDRESSED: "Partially addressed",
    NOT_ADDRESSED: "Not addressed",
}

_COVERAGE_COLUMNS: tuple[str, ...] = (ADDRESSED, PARTIALLY_ADDRESSED, NOT_ADDRESSED)


def _cell(text: str) -> str:
    """Return ``text`` made safe for a single Markdown table cell.

    Pipes are escaped and newlines collapsed so a multi-line rationale cannot
    break the table layout.

    Parameters
    ----------
    text:
        The raw cell text.

    Returns
    -------
    str
        The escaped, single-line cell text.
    """
    return " ".join(text.split()).replace("\\", "\\\\").replace("|", "\\|")


def _status_label(status: str) -> str:
    """Return the human-readable label for a conformance ``status``."""
    return _STATUS_LABELS.get(status, status)


def _header_block(bundle: AssuranceCaseBundle) -> list[str]:
    """Return the title and traceability metadata lines."""
    standards = ", ".join(sorted(bundle.standards_covered))
    return [
        f"# Conformity Evidence Report — {bundle.system_name}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Report schema | {CONFORMITY_REPORT_SCHEMA} |",
        f"| Bundle schema | {bundle.schema} |",
        f"| Bundle version | {_cell(bundle.version)} |",
        f"| Bundle hash | `{bundle.bundle_hash}` |",
        f"| Standards covered | {_cell(standards)} |",
        "| Actuation permitted | No (review-only) |",
        "",
        f"> {bundle.disclaimer}",
        "",
    ]


def _coverage_block(bundle: AssuranceCaseBundle) -> list[str]:
    """Return the per-standard coverage summary table."""
    lines = [
        "## Coverage summary",
        "",
        "| Standard | Addressed | Partially addressed | Not addressed | Total |",
        "| --- | --- | --- | --- | --- |",
    ]
    summary = bundle.coverage_summary()
    for standard in sorted(summary):
        counts = summary[standard]
        lines.append(
            f"| {_cell(standard)} | {counts[ADDRESSED]} | "
            f"{counts[PARTIALLY_ADDRESSED]} | {counts[NOT_ADDRESSED]} | "
            f"{counts['total']} |"
        )
    lines.append("")
    return lines


def _clause_blocks(bundle: AssuranceCaseBundle) -> list[str]:
    """Return one clause table per standard, in stable order."""
    by_standard: dict[str, list[str]] = {}
    for entry in sorted(
        bundle.conformance,
        key=lambda row: (row.clause.standard, row.clause.clause_id),
    ):
        rows = by_standard.setdefault(entry.clause.standard, [])
        evidence = ", ".join(entry.evidence_ids) if entry.evidence_ids else "—"
        rows.append(
            f"| {_cell(entry.clause.clause_id)} | {_cell(entry.clause.title)} | "
            f"{_status_label(entry.status)} | {_cell(evidence)} | "
            f"{_cell(entry.rationale)} |"
        )
    lines: list[str] = []
    for standard in sorted(by_standard):
        lines.append(f"## {standard}")
        lines.append("")
        lines.append("| Clause | Title | Status | Evidence | Rationale |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.extend(by_standard[standard])
        lines.append("")
    return lines


def _evidence_block(bundle: AssuranceCaseBundle) -> list[str]:
    """Return the evidence inventory table."""
    lines = [
        "## Evidence inventory",
        "",
        "| Evidence ID | Category | Summary | Content hash |",
        "| --- | --- | --- | --- |",
    ]
    for item in sorted(bundle.evidence, key=lambda row: row.evidence_id):
        lines.append(
            f"| {_cell(item.evidence_id)} | {_cell(item.category)} | "
            f"{_cell(item.summary)} | `{item.content_hash}` |"
        )
    lines.append("")
    return lines


def render_conformity_report(bundle: AssuranceCaseBundle) -> str:
    """Render an assurance-case bundle as a Markdown conformity report.

    The report restates the bundle verbatim — coverage rollup, per-standard
    clause conformance (status, contributing evidence, rationale), and the
    evidence inventory — under the regulatory disclaimer and anchored to the
    bundle hash. It adds no claim not already present in ``bundle`` and is
    review-only. Rendering is deterministic for a given bundle.

    Parameters
    ----------
    bundle:
        The hash-sealed assurance-case bundle to render.

    Returns
    -------
    str
        The Markdown conformity report, terminated by a single newline.
    """
    lines: list[str] = []
    lines.extend(_header_block(bundle))
    lines.extend(_coverage_block(bundle))
    lines.extend(_clause_blocks(bundle))
    lines.extend(_evidence_block(bundle))
    lines.append("---")
    lines.append("")
    lines.append(
        "Review-only conformity evidence report. This document is a technical "
        "evidence aid, not a conformity assessment or certification of "
        "compliance. Traceability anchor: bundle hash "
        f"`{bundle.bundle_hash}`."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CONFORMITY_REPORT_SCHEMA",
    "render_conformity_report",
]
