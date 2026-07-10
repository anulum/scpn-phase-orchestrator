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
from scpn_phase_orchestrator.assurance.dsse import (
    DSSE_PAYLOAD_TYPE,
    DsseEnvelope,
    DsseSignature,
    sign_provenance_statement,
    verify_dsse_envelope,
)
from scpn_phase_orchestrator.assurance.envelope import (
    SIGNED_CERTIFICATION_ENVELOPE_SCHEMA,
    SignedCertificationEnvelope,
    build_signed_certification_envelope,
    verify_signed_certification_envelope,
)
from scpn_phase_orchestrator.assurance.evidence import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    CONTROL_ENVELOPE,
    EVIDENCE_CATEGORIES,
    FORMAL_VERIFICATION,
    REPLAY_DETERMINISM,
    TWIN_CONFIDENCE,
    EvidenceItem,
    build_evidence_item,
)
from scpn_phase_orchestrator.assurance.formal_evidence import (
    build_formal_verification_evidence,
)
from scpn_phase_orchestrator.assurance.power_grid_prc_bundle import (
    DVOC_DAMPING_ROLE,
    IBR_RIDE_THROUGH_ROLE,
    PMU_RINGDOWN_ROLE,
    POWER_GRID_PRC_AUDIT_BUNDLE_DISCLAIMER,
    POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA,
    POWER_GRID_PRC_CLAIM_BOUNDARY,
    POWER_GRID_PRC_REQUIRED_ROLES,
    PowerGridPRCArtifact,
    PowerGridPRCAuditBundle,
    PowerGridPRCInputArtifact,
    build_power_grid_prc_audit_bundle,
)
from scpn_phase_orchestrator.assurance.provenance import (
    IN_TOTO_STATEMENT_TYPE,
    SLSA_PROVENANCE_PREDICATE_TYPE,
    ArtifactSubject,
    BuildDefinition,
    ResourceDescriptor,
    RunDetails,
    SlsaProvenanceStatement,
    build_slsa_provenance_statement,
    provenance_statement_hash,
    pypi_resolved_dependency,
)
from scpn_phase_orchestrator.assurance.report import (
    CONFORMITY_REPORT_SCHEMA,
    render_conformity_report,
    render_conformity_report_pdf,
)
from scpn_phase_orchestrator.assurance.run_evidence import build_run_evidence
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
from scpn_phase_orchestrator.assurance.twin_confidence_evidence import (
    build_twin_confidence_evidence,
)

__all__ = [
    "ADDRESSED",
    "ASSURANCE_CASE_SCHEMA",
    "AUDIT_LOGGING",
    "ArtifactSubject",
    "AssuranceCaseBundle",
    "BuildDefinition",
    "DSSE_PAYLOAD_TYPE",
    "DsseEnvelope",
    "DsseSignature",
    "IN_TOTO_STATEMENT_TYPE",
    "DVOC_DAMPING_ROLE",
    "ResourceDescriptor",
    "RunDetails",
    "SLSA_PROVENANCE_PREDICATE_TYPE",
    "SlsaProvenanceStatement",
    "build_slsa_provenance_statement",
    "provenance_statement_hash",
    "pypi_resolved_dependency",
    "sign_provenance_statement",
    "verify_dsse_envelope",
    "CONFORMAL_GATE",
    "CONTROL_ENVELOPE",
    "CONFORMANCE_STATUSES",
    "CONFORMITY_REPORT_SCHEMA",
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
    "IBR_RIDE_THROUGH_ROLE",
    "ISO_IEC_42001",
    "ISO_IEC_42001_CLAUSES",
    "NOT_ADDRESSED",
    "PARTIALLY_ADDRESSED",
    "PMU_RINGDOWN_ROLE",
    "POWER_GRID_PRC_AUDIT_BUNDLE_DISCLAIMER",
    "POWER_GRID_PRC_AUDIT_BUNDLE_SCHEMA",
    "POWER_GRID_PRC_CLAIM_BOUNDARY",
    "POWER_GRID_PRC_REQUIRED_ROLES",
    "PowerGridPRCArtifact",
    "PowerGridPRCAuditBundle",
    "PowerGridPRCInputArtifact",
    "REGULATORY_DISCLAIMER",
    "REPLAY_DETERMINISM",
    "RegulatoryClause",
    "SIGNED_CERTIFICATION_ENVELOPE_SCHEMA",
    "SUPPORTED_STANDARDS",
    "SignedCertificationEnvelope",
    "TWIN_CONFIDENCE",
    "UL_4600",
    "UL_4600_CLAUSES",
    "build_assurance_case_bundle",
    "build_certification_evidence_package",
    "build_evidence_item",
    "build_formal_verification_evidence",
    "build_power_grid_prc_audit_bundle",
    "build_run_evidence",
    "build_signed_certification_envelope",
    "build_twin_confidence_evidence",
    "clause_catalogue",
    "clause_for_key",
    "render_conformity_report",
    "render_conformity_report_pdf",
    "verify_signed_certification_envelope",
]
