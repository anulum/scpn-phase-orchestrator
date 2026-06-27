# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case evidence bundle

"""Assurance-case evidence bundles for regulated SPO deployment.

This package composes existing SPO runtime evidence (audit-chain integrity,
replay determinism, formal verification, twin-confidence, and the conformal
admission gate) into a single hash-sealed bundle, and maps it to the published
clauses of the EU AI Act, ISO/IEC 42001, and ANSI/UL 4600. The bundle is
review-only and carries a disclaimer that it is a technical evidence-mapping aid,
not a legal conformity assessment.
"""

from __future__ import annotations

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
from scpn_phase_orchestrator.assurance.certification import (
    CERTIFICATION_EVIDENCE_PACKAGE_DISCLAIMER,
    CERTIFICATION_EVIDENCE_PACKAGE_SCHEMA,
    CertificationEvidencePackage,
    build_certification_evidence_package,
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
    EU_AI_ACT,
    EU_AI_ACT_CLAUSES,
    ISO_IEC_42001,
    ISO_IEC_42001_CLAUSES,
    REGULATORY_DISCLAIMER,
    SUPPORTED_STANDARDS,
    UL_4600,
    UL_4600_CLAUSES,
    RegulatoryClause,
    clause_catalogue,
    clause_for_key,
)

__all__ = [
    "ADDRESSED",
    "ASSURANCE_CASE_SCHEMA",
    "AUDIT_LOGGING",
    "AssuranceCaseBundle",
    "CONFORMAL_GATE",
    "CONFORMANCE_STATUSES",
    "CERTIFICATION_EVIDENCE_PACKAGE_DISCLAIMER",
    "CERTIFICATION_EVIDENCE_PACKAGE_SCHEMA",
    "CertificationEvidencePackage",
    "ClauseConformance",
    "DEFAULT_EVIDENCE_CLAUSE_MAP",
    "EU_AI_ACT",
    "EU_AI_ACT_CLAUSES",
    "EVIDENCE_CATEGORIES",
    "EvidenceItem",
    "FORMAL_VERIFICATION",
    "ISO_IEC_42001",
    "ISO_IEC_42001_CLAUSES",
    "NOT_ADDRESSED",
    "PARTIALLY_ADDRESSED",
    "REGULATORY_DISCLAIMER",
    "REPLAY_DETERMINISM",
    "RegulatoryClause",
    "SUPPORTED_STANDARDS",
    "TWIN_CONFIDENCE",
    "UL_4600",
    "UL_4600_CLAUSES",
    "build_assurance_case_bundle",
    "build_certification_evidence_package",
    "build_evidence_item",
    "clause_catalogue",
    "clause_for_key",
]
