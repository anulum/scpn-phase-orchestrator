# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case bundle assembly

"""Assemble SPO runtime evidence into a hash-sealed assurance-case bundle.

The bundle links each catalogued regulatory clause
(:mod:`scpn_phase_orchestrator.assurance.standards`) to the SPO evidence that
addresses it, records the conformance status and rationale, and seals the whole
into a deterministic hash. The bundle is review-only: ``actuation_permitted`` is
always ``False`` and the bundle carries the
:data:`~scpn_phase_orchestrator.assurance.standards.REGULATORY_DISCLAIMER`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)
from scpn_phase_orchestrator.assurance.evidence import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    FORMAL_VERIFICATION,
    REPLAY_DETERMINISM,
    TWIN_CONFIDENCE,
    EvidenceItem,
)
from scpn_phase_orchestrator.assurance.standards import (
    REGULATORY_DISCLAIMER,
    RegulatoryClause,
    clause_catalogue,
)

ADDRESSED = "addressed"
PARTIALLY_ADDRESSED = "partially_addressed"
NOT_ADDRESSED = "not_addressed"
CONFORMANCE_STATUSES: frozenset[str] = frozenset(
    {ADDRESSED, PARTIALLY_ADDRESSED, NOT_ADDRESSED}
)

ASSURANCE_CASE_SCHEMA = "scpn_assurance_case_bundle_v1"

_EU = "EU AI Act 2024/1689"
_ISO = "ISO/IEC 42001:2023"
_UL = "ANSI/UL 4600"


def _k(standard: str, clause_id: str) -> str:
    """Return the canonical key for an assurance clause."""
    return f"{standard}::{clause_id}"


# Verified evidence -> clause contribution map. Each entry states whether the
# evidence category fully ("addressed") or partially ("partially_addressed")
# satisfies the clause, with a rationale tying the SPO surface to the clause.
DEFAULT_EVIDENCE_CLAUSE_MAP: dict[str, tuple[tuple[str, str, str], ...]] = {
    AUDIT_LOGGING: (
        (
            _k(_EU, "Article 12"),
            ADDRESSED,
            "Hash-chained, optionally HMAC-signed audit log captures events over "
            "the system lifetime.",
        ),
        (
            _k(_EU, "Article 15"),
            PARTIALLY_ADDRESSED,
            "Tamper-evident hash chain and signing support the integrity facet of "
            "robustness and cybersecurity.",
        ),
        (
            _k(_ISO, "Clause 8"),
            PARTIALLY_ADDRESSED,
            "Operational records of controlled execution.",
        ),
        (
            _k(_UL, "data-integrity"),
            ADDRESSED,
            "Append-only chained log with payload digests evidences data integrity.",
        ),
        (
            _k(_UL, "safety-case"),
            PARTIALLY_ADDRESSED,
            "Audit trail forms part of the supporting evidence body.",
        ),
    ),
    REPLAY_DETERMINISM: (
        (
            _k(_EU, "Article 15"),
            ADDRESSED,
            "Deterministic re-execution demonstrates reproducibility and "
            "robustness of recorded behaviour.",
        ),
        (
            _k(_EU, "Article 12"),
            PARTIALLY_ADDRESSED,
            "Replay verifies the integrity of the recorded audit log.",
        ),
        (
            _k(_ISO, "Clause 9"),
            PARTIALLY_ADDRESSED,
            "Reproducibility check contributes to performance evaluation.",
        ),
        (
            _k(_UL, "autonomy-validation"),
            PARTIALLY_ADDRESSED,
            "Deterministic replay validates recorded autonomous decisions.",
        ),
        (
            _k(_UL, "tool-qualification"),
            PARTIALLY_ADDRESSED,
            "Deterministic engine behaviour supports tool qualification.",
        ),
    ),
    FORMAL_VERIFICATION: (
        (
            _k(_EU, "Article 15"),
            ADDRESSED,
            "Formal safety properties and the fail-closed runtime certificate "
            "evidence robustness and accuracy bounds.",
        ),
        (
            _k(_EU, "Article 9"),
            PARTIALLY_ADDRESSED,
            "Property-based checks are a risk-control measure.",
        ),
        (
            _k(_ISO, "Clause 6"),
            PARTIALLY_ADDRESSED,
            "Formal property planning supports risk treatment.",
        ),
        (
            _k(_ISO, "A.6"),
            PARTIALLY_ADDRESSED,
            "Verification artefacts are produced across the AI system life cycle.",
        ),
        (
            _k(_UL, "safety-case"),
            ADDRESSED,
            "Structured formal argument with attached evidence is a safety case.",
        ),
        (
            _k(_UL, "risk-analysis"),
            ADDRESSED,
            "Formal safety properties encode the analysed hazards.",
        ),
    ),
    TWIN_CONFIDENCE: (
        (
            _k(_EU, "Article 9"),
            ADDRESSED,
            "Continuous twin divergence scoring against a calibrated baseline is a "
            "live risk-management control.",
        ),
        (
            _k(_EU, "Article 72"),
            ADDRESSED,
            "Online drift monitoring supports post-market monitoring duties.",
        ),
        (
            _k(_EU, "Article 15"),
            PARTIALLY_ADDRESSED,
            "Divergence bands monitor robustness against the digital twin.",
        ),
        (
            _k(_ISO, "Clause 9"),
            ADDRESSED,
            "Quantitative confidence metrics drive performance evaluation.",
        ),
        (
            _k(_ISO, "A.8"),
            PARTIALLY_ADDRESSED,
            "Confidence summaries inform interested parties of operating state.",
        ),
        (
            _k(_UL, "metrics-conformance"),
            ADDRESSED,
            "Confidence scores are conformance metrics for operation.",
        ),
    ),
    CONFORMAL_GATE: (
        (
            _k(_EU, "Article 9"),
            ADDRESSED,
            "Coverage-valid admission gate bounds residual risk distribution-free.",
        ),
        (
            _k(_EU, "Article 14"),
            PARTIALLY_ADDRESSED,
            "Flagged ticks and fail-closed admission route decisions to human "
            "oversight.",
        ),
        (
            _k(_EU, "Article 72"),
            PARTIALLY_ADDRESSED,
            "Empirical coverage tracking supports post-market monitoring.",
        ),
        (
            _k(_ISO, "Clause 6"),
            PARTIALLY_ADDRESSED,
            "Target miscoverage encodes the planned risk acceptance criterion.",
        ),
        (
            _k(_UL, "risk-analysis"),
            ADDRESSED,
            "Conformal threshold realises the analysed risk-acceptance bound.",
        ),
        (
            _k(_UL, "metrics-conformance"),
            PARTIALLY_ADDRESSED,
            "Empirical coverage is a conformance metric.",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class ClauseConformance:
    """Conformance status of one clause against the bundle's evidence.

    Parameters
    ----------
    clause:
        The catalogued regulatory clause.
    status:
        One of :data:`CONFORMANCE_STATUSES`.
    evidence_ids:
        Evidence identifiers addressing the clause (empty iff ``not_addressed``).
    rationale:
        Explanation linking the evidence to the clause.
    """

    clause: RegulatoryClause
    status: str
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.status not in CONFORMANCE_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(CONFORMANCE_STATUSES)}, "
                f"got {self.status!r}"
            )
        if self.status == NOT_ADDRESSED and self.evidence_ids:
            raise ValueError("not_addressed clauses must carry no evidence_ids")
        if self.status != NOT_ADDRESSED and not self.evidence_ids:
            raise ValueError(
                f"{self.status} clauses must carry at least one evidence_id"
            )
        if not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe conformance record.

        Returns
        -------
        dict[str, object]
            A JSON-safe conformance record.
        """
        return {
            "clause": self.clause.to_audit_record(),
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class AssuranceCaseBundle:
    """A hash-sealed, review-only assurance-case evidence bundle.

    Parameters
    ----------
    system_name:
        Name of the system the bundle describes.
    version:
        Bundle schema instance version (semantic, e.g. ``"1.0.0"``).
    evidence:
        The collected evidence items (unique ``evidence_id``).
    conformance:
        Per-clause conformance, one entry per catalogued clause.
    standards_covered:
        Standards the clause catalogue spans.
    bundle_hash:
        SHA-256 over the canonical bundle seed; recomputed and checked.
    schema:
        Schema identifier; fixed to :data:`ASSURANCE_CASE_SCHEMA`.
    disclaimer:
        Regulatory disclaimer; fixed to ``REGULATORY_DISCLAIMER``.
    actuation_permitted:
        Always ``False`` — the bundle is review-only.
    """

    system_name: str
    version: str
    evidence: tuple[EvidenceItem, ...]
    conformance: tuple[ClauseConformance, ...]
    standards_covered: tuple[str, ...]
    bundle_hash: str = field(default="")
    schema: str = ASSURANCE_CASE_SCHEMA
    disclaimer: str = REGULATORY_DISCLAIMER
    actuation_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.system_name.strip():
            raise ValueError("system_name must be a non-empty string")
        if not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if self.schema != ASSURANCE_CASE_SCHEMA:
            raise ValueError(f"schema must be {ASSURANCE_CASE_SCHEMA!r}")
        if self.actuation_permitted:
            raise ValueError("assurance bundles are review-only; actuation_permitted")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        known = set(ids)
        for entry in self.conformance:
            missing = set(entry.evidence_ids) - known
            if missing:
                raise ValueError(
                    f"conformance references unknown evidence_ids: {sorted(missing)}"
                )
        computed = canonical_record_hash(self._seed())
        if not self.bundle_hash:
            object.__setattr__(self, "bundle_hash", computed)
        else:
            require_sha256(self.bundle_hash, "bundle_hash")
            if self.bundle_hash != computed:
                raise ValueError("bundle_hash does not match the canonical bundle seed")

    def _seed(self) -> dict[str, object]:
        """Return the deterministic seed for the assurance case."""
        return {
            "schema": self.schema,
            "version": self.version,
            "system_name": self.system_name,
            "standards_covered": sorted(self.standards_covered),
            "disclaimer": self.disclaimer,
            "actuation_permitted": self.actuation_permitted,
            "evidence": sorted(item.content_hash for item in self.evidence),
            "conformance": sorted(
                canonical_record_hash(entry.to_audit_record())
                for entry in self.conformance
            ),
        }

    def coverage_summary(self) -> dict[str, dict[str, int]]:
        """Return per-standard counts of clause conformance statuses.

        Returns
        -------
        dict[str, dict[str, int]]
            Maps each standard to ``addressed`` / ``partially_addressed`` /
            ``not_addressed`` / ``total`` counts.
        """
        summary: dict[str, dict[str, int]] = {}
        for entry in self.conformance:
            bucket = summary.setdefault(
                entry.clause.standard,
                {ADDRESSED: 0, PARTIALLY_ADDRESSED: 0, NOT_ADDRESSED: 0, "total": 0},
            )
            bucket[entry.status] += 1
            bucket["total"] += 1
        return summary

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe bundle record.

        Returns
        -------
        dict[str, object]
            A JSON-safe bundle record.
        """
        return {
            "schema": self.schema,
            "version": self.version,
            "system_name": self.system_name,
            "standards_covered": sorted(self.standards_covered),
            "disclaimer": self.disclaimer,
            "actuation_permitted": self.actuation_permitted,
            "bundle_hash": self.bundle_hash,
            "coverage_summary": self.coverage_summary(),
            "evidence": [item.to_audit_record() for item in self.evidence],
            "conformance": [entry.to_audit_record() for entry in self.conformance],
        }


def _conformance_for_clause(
    clause: RegulatoryClause,
    category_evidence: Mapping[str, list[str]],
) -> ClauseConformance:
    """Return the conformance status for a clause."""
    addressed_ids: list[str] = []
    rationales: list[str] = []
    best_status = NOT_ADDRESSED
    for category in sorted(category_evidence):
        evidence_ids = category_evidence[category]
        for clause_key, coverage, rationale in DEFAULT_EVIDENCE_CLAUSE_MAP[category]:
            if clause_key != clause.key:
                continue
            addressed_ids.extend(evidence_ids)
            rationales.append(rationale)
            if coverage == ADDRESSED:
                best_status = ADDRESSED
            elif best_status != ADDRESSED:
                best_status = PARTIALLY_ADDRESSED
    if best_status == NOT_ADDRESSED:
        return ClauseConformance(
            clause=clause,
            status=NOT_ADDRESSED,
            evidence_ids=(),
            rationale=(
                "No technical evidence in this bundle addresses this clause; "
                "organisational or process evidence is required."
            ),
        )
    return ClauseConformance(
        clause=clause,
        status=best_status,
        evidence_ids=tuple(sorted(set(addressed_ids))),
        rationale=" ".join(dict.fromkeys(rationales)),
    )


def build_assurance_case_bundle(
    system_name: str,
    evidence: Sequence[EvidenceItem],
    *,
    version: str = "1.0.0",
) -> AssuranceCaseBundle:
    """Assemble an assurance-case bundle from collected evidence.

    Each catalogued clause is mapped to the evidence addressing it via
    :data:`DEFAULT_EVIDENCE_CLAUSE_MAP`; clauses with no contributing evidence
    are recorded as ``not_addressed`` so gaps are explicit.

    Parameters
    ----------
    system_name:
        Name of the system the bundle describes.
    evidence:
        The evidence items to include (unique ``evidence_id``).
    version:
        Bundle instance version.

    Returns
    -------
    AssuranceCaseBundle
        The sealed bundle.

    Raises
    ------
    ValueError
        If ``evidence_id`` values are not unique.
    """
    ids = [item.evidence_id for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence_id values must be unique")
    category_evidence: dict[str, list[str]] = {}
    for item in evidence:
        category_evidence.setdefault(item.category, []).append(item.evidence_id)
    conformance = tuple(
        _conformance_for_clause(clause, category_evidence)
        for clause in clause_catalogue()
    )
    standards_covered = tuple(
        sorted({clause.standard for clause in clause_catalogue()})
    )
    return AssuranceCaseBundle(
        system_name=system_name,
        version=version,
        evidence=tuple(evidence),
        conformance=conformance,
        standards_covered=standards_covered,
    )


__all__ = [
    "ADDRESSED",
    "ASSURANCE_CASE_SCHEMA",
    "AssuranceCaseBundle",
    "CONFORMANCE_STATUSES",
    "ClauseConformance",
    "DEFAULT_EVIDENCE_CLAUSE_MAP",
    "NOT_ADDRESSED",
    "PARTIALLY_ADDRESSED",
    "build_assurance_case_bundle",
]
