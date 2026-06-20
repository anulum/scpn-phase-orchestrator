# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — regulatory clause catalogue

"""Reference catalogue of regulatory clauses for assurance-case mapping.

The catalogue records the clause identifiers and official titles of the three
standards an SPO deployment is most often measured against:

* Regulation (EU) 2024/1689 (the EU AI Act) — high-risk requirements;
* ISO/IEC 42001:2023 — AI management system clauses and Annex A controls;
* ANSI/UL 4600 — claim-based safety case for autonomous products.

Each :class:`RegulatoryClause` carries a ``provenance`` note identifying the
source the identifier and title were taken from. The catalogue is a structured
reference aid for assembling evidence; it is **not** a legal interpretation, and
clause text must be confirmed against the official standard before any external
submission. See :data:`REGULATORY_DISCLAIMER`.
"""

from __future__ import annotations

from dataclasses import dataclass

EU_AI_ACT = "EU AI Act 2024/1689"
ISO_IEC_42001 = "ISO/IEC 42001:2023"
UL_4600 = "ANSI/UL 4600"

_EU_PROVENANCE = (
    "Regulation (EU) 2024/1689, Chapter III Section 2; "
    "titles per EUR-Lex CELEX:32024R1689"
)
_ISO_PROVENANCE = (
    "ISO/IEC 42001:2023 management-system clauses 4-10 and Annex A reference "
    "controls (topic headings A.2-A.10)"
)
_UL_PROVENANCE = "ANSI/UL 4600 claim-based safety-case topic structure"

REGULATORY_DISCLAIMER = (
    "This assurance-case bundle is a technical evidence-mapping artifact. It "
    "links SPO runtime evidence to published clause identifiers to support "
    "review; it does not constitute legal advice, a conformity assessment, or a "
    "certification of compliance. Clause text must be confirmed against the "
    "official standard, and conformance determined by a qualified assessor."
)


@dataclass(frozen=True, slots=True)
class RegulatoryClause:
    """A single referenceable clause of a regulatory standard.

    Parameters
    ----------
    standard:
        Human-readable standard name (e.g. ``"EU AI Act 2024/1689"``).
    clause_id:
        Stable clause identifier within the standard (e.g. ``"Article 12"``,
        ``"Clause 9"``, ``"A.6"``, ``"data-integrity"``).
    title:
        Official clause title.
    provenance:
        Note identifying the source of the identifier and title.
    """

    standard: str
    clause_id: str
    title: str
    provenance: str

    @property
    def key(self) -> str:
        """Return a globally unique ``standard::clause_id`` key.

        Returns
        -------
        str
            The composite key.
        """
        return f"{self.standard}::{self.clause_id}"

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe clause record.

        Returns
        -------
        dict[str, object]
            A JSON-safe clause record.
        """
        return {
            "standard": self.standard,
            "clause_id": self.clause_id,
            "title": self.title,
            "provenance": self.provenance,
        }


def _eu(clause_id: str, title: str) -> RegulatoryClause:
    return RegulatoryClause(EU_AI_ACT, clause_id, title, _EU_PROVENANCE)


def _iso(clause_id: str, title: str) -> RegulatoryClause:
    return RegulatoryClause(ISO_IEC_42001, clause_id, title, _ISO_PROVENANCE)


def _ul(clause_id: str, title: str) -> RegulatoryClause:
    return RegulatoryClause(UL_4600, clause_id, title, _UL_PROVENANCE)


EU_AI_ACT_CLAUSES: tuple[RegulatoryClause, ...] = (
    _eu("Article 9", "Risk Management System"),
    _eu("Article 10", "Data and Data Governance"),
    _eu("Article 11", "Technical Documentation"),
    _eu("Article 12", "Record-Keeping"),
    _eu("Article 13", "Transparency and Provision of Information to Deployers"),
    _eu("Article 14", "Human Oversight"),
    _eu("Article 15", "Accuracy, Robustness and Cybersecurity"),
    _eu("Article 17", "Quality Management System"),
    _eu("Article 72", "Post-Market Monitoring by Providers"),
)

ISO_IEC_42001_CLAUSES: tuple[RegulatoryClause, ...] = (
    _iso("Clause 6", "Planning"),
    _iso("Clause 8", "Operation"),
    _iso("Clause 9", "Performance evaluation"),
    _iso("Clause 10", "Improvement"),
    _iso("A.5", "Assessing Impacts of AI Systems"),
    _iso("A.6", "AI System Life Cycle"),
    _iso("A.7", "Data for AI Systems"),
    _iso("A.8", "Information for Interested Parties of AI Systems"),
)

UL_4600_CLAUSES: tuple[RegulatoryClause, ...] = (
    _ul("safety-case", "Safety case construction"),
    _ul("risk-analysis", "Risk analysis"),
    _ul("autonomy-validation", "Autonomy validation"),
    _ul("data-integrity", "Data integrity"),
    _ul("tool-qualification", "Tool qualification"),
    _ul("metrics-conformance", "Metrics and conformance"),
)

_ALL_CLAUSES: tuple[RegulatoryClause, ...] = (
    EU_AI_ACT_CLAUSES + ISO_IEC_42001_CLAUSES + UL_4600_CLAUSES
)
_CLAUSE_BY_KEY: dict[str, RegulatoryClause] = {c.key: c for c in _ALL_CLAUSES}

SUPPORTED_STANDARDS: tuple[str, ...] = (EU_AI_ACT, ISO_IEC_42001, UL_4600)


def clause_catalogue() -> tuple[RegulatoryClause, ...]:
    """Return every catalogued clause across all supported standards.

    Returns
    -------
    tuple[RegulatoryClause, ...]
        The full clause catalogue.
    """
    return _ALL_CLAUSES


def clause_for_key(key: str) -> RegulatoryClause:
    """Return the clause registered under ``key``.

    Parameters
    ----------
    key:
        A ``standard::clause_id`` key.

    Returns
    -------
    RegulatoryClause
        The matching clause.

    Raises
    ------
    KeyError
        If no clause is registered under ``key``.
    """
    try:
        return _CLAUSE_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown clause key: {key}") from exc
