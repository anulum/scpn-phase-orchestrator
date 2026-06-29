# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — certification evidence package assembly

"""Assemble review packages from assurance-case evidence bundles.

The package writer is deliberately narrow: it wraps the existing
assurance-case bundle with deterministic hash test vectors and a manifest that
seals every emitted file. The output is a technical evidence package for human
review, not a certification claim or live actuation gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.case import (
    AssuranceCaseBundle,
    build_assurance_case_bundle,
)
from scpn_phase_orchestrator.assurance.evidence import EvidenceItem
from scpn_phase_orchestrator.assurance.report import render_conformity_report
from scpn_phase_orchestrator.assurance.standards import REGULATORY_DISCLAIMER

CERTIFICATION_EVIDENCE_PACKAGE_SCHEMA = "scpn_certification_evidence_package_v1"
CERTIFICATION_EVIDENCE_PACKAGE_DISCLAIMER = (
    "This is a technical evidence-mapping package for reviewer triage. It "
    "contains deterministic SPO assurance evidence and hash test vectors, but "
    "it is not legal advice, a conformity assessment, or a certification of "
    "compliance."
)


def _dump_json(record: Mapping[str, object]) -> str:
    """Return deterministic pretty JSON for a package file."""
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _sha256_text(payload: str) -> str:
    """Return the SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CertificationEvidencePackage:
    """A deterministic standards-shaped review package.

    Parameters
    ----------
    assurance_bundle:
        The underlying hash-sealed assurance-case bundle.
    test_vectors:
        JSON-safe deterministic vectors that let reviewers recompute evidence
        hashes and clause-conformance rationale hashes.
    manifest:
        JSON-safe package manifest containing file digests and package digest.
    file_contents:
        Mapping from relative package paths to deterministic JSON file content.
    """

    assurance_bundle: AssuranceCaseBundle
    test_vectors: Mapping[str, object]
    manifest: Mapping[str, object]
    file_contents: Mapping[str, str]

    def to_files(self) -> dict[str, str]:
        """Return package files keyed by relative path.

        Returns
        -------
        dict[str, str]
            The package files, including ``manifest.json``.
        """
        return dict(self.file_contents)


def _build_evidence_hash_vectors(
    evidence: Sequence[EvidenceItem],
) -> list[dict[str, object]]:
    """Return content-hash vectors for each evidence item."""
    return [
        {
            "evidence_id": item.evidence_id,
            "category": item.category,
            "summary": item.summary,
            "record": dict(item.record),
            "expected_content_hash": item.content_hash,
        }
        for item in sorted(evidence, key=lambda row: row.evidence_id)
    ]


def _build_clause_vectors(bundle: AssuranceCaseBundle) -> list[dict[str, object]]:
    """Return deterministic vectors for clause-conformance review."""
    return [
        {
            "standard": entry.clause.standard,
            "clause_id": entry.clause.clause_id,
            "status": entry.status,
            "evidence_ids": list(entry.evidence_ids),
            "expected_rationale_hash": canonical_record_hash(
                {"rationale": entry.rationale}
            ),
        }
        for entry in sorted(
            bundle.conformance,
            key=lambda row: (row.clause.standard, row.clause.clause_id),
        )
    ]


def _build_test_vectors(bundle: AssuranceCaseBundle) -> dict[str, object]:
    """Return JSON-safe test vectors for the assurance bundle."""
    return {
        "schema": "scpn_certification_evidence_test_vectors_v1",
        "assurance_bundle_hash": bundle.bundle_hash,
        "evidence_hash_vectors": _build_evidence_hash_vectors(bundle.evidence),
        "clause_conformance_vectors": _build_clause_vectors(bundle),
    }


def build_certification_evidence_package(
    system_name: str,
    evidence: Sequence[EvidenceItem],
    *,
    version: str = "1.0.0",
) -> CertificationEvidencePackage:
    """Build a deterministic review package from assurance evidence.

    Parameters
    ----------
    system_name:
        Name of the reviewed system.
    evidence:
        Assurance evidence items to include.
    version:
        Package schema instance version.

    Returns
    -------
    CertificationEvidencePackage
        The assembled package with deterministic JSON file contents.
    """
    bundle = build_assurance_case_bundle(system_name, evidence, version=version)
    bundle_payload = _dump_json(bundle.to_audit_record())
    test_vectors = _build_test_vectors(bundle)
    vector_payload = _dump_json(test_vectors)
    report_payload = render_conformity_report(bundle)
    file_rows = [
        {
            "path": "assurance_bundle.json",
            "sha256": _sha256_text(bundle_payload),
            "bytes": len(bundle_payload.encode("utf-8")),
        },
        {
            "path": "conformity_report.md",
            "sha256": _sha256_text(report_payload),
            "bytes": len(report_payload.encode("utf-8")),
        },
        {
            "path": "test_vectors.json",
            "sha256": _sha256_text(vector_payload),
            "bytes": len(vector_payload.encode("utf-8")),
        },
    ]
    package_seed: dict[str, object] = {
        "schema": CERTIFICATION_EVIDENCE_PACKAGE_SCHEMA,
        "version": version,
        "system_name": system_name,
        "assurance_bundle_hash": bundle.bundle_hash,
        "files": file_rows,
    }
    manifest: dict[str, object] = {
        **package_seed,
        "standards_covered": list(bundle.standards_covered),
        "coverage_summary": bundle.coverage_summary(),
        "disclaimer": CERTIFICATION_EVIDENCE_PACKAGE_DISCLAIMER,
        "assurance_disclaimer": REGULATORY_DISCLAIMER,
        "package_hash": canonical_record_hash(package_seed),
    }
    return CertificationEvidencePackage(
        assurance_bundle=bundle,
        test_vectors=test_vectors,
        manifest=manifest,
        file_contents={
            "assurance_bundle.json": bundle_payload,
            "conformity_report.md": report_payload,
            "test_vectors.json": vector_payload,
            "manifest.json": _dump_json(manifest),
        },
    )


__all__ = [
    "CERTIFICATION_EVIDENCE_PACKAGE_DISCLAIMER",
    "CERTIFICATION_EVIDENCE_PACKAGE_SCHEMA",
    "CertificationEvidencePackage",
    "build_certification_evidence_package",
]
