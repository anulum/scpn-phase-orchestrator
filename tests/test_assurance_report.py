# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance conformity report renderer tests

"""Tests for the Markdown conformity report renderer.

The report must restate the sealed bundle verbatim (coverage rollup, per-standard
clause conformance, evidence inventory) under the disclaimer and anchored to the
bundle hash, deterministically and without introducing any new claim.
"""

from __future__ import annotations

import hashlib

import scpn_phase_orchestrator.assurance.report as _report
from scpn_phase_orchestrator.assurance import (
    AUDIT_LOGGING,
    EU_AI_ACT,
    ISO_IEC_42001,
    TWIN_CONFIDENCE,
    UL_4600,
    AssuranceCaseBundle,
    build_assurance_case_bundle,
    build_certification_evidence_package,
    build_evidence_item,
    render_conformity_report,
    render_conformity_report_pdf,
)
from scpn_phase_orchestrator.assurance.report import (
    CONFORMITY_REPORT_SCHEMA,
    _status_label,
)

assert _report is not None


def _bundle() -> AssuranceCaseBundle:
    """Return a bundle with audit + twin evidence (one summary carries a pipe)."""
    evidence = [
        build_evidence_item(
            evidence_id="audit-chain-integrity",
            category=AUDIT_LOGGING,
            summary="Audit-chain integrity over run-a | run-b",
            record={"integrity_ok": True, "verified_records": 12},
        ),
        build_evidence_item(
            evidence_id="twin-divergence",
            category=TWIN_CONFIDENCE,
            summary="Twin divergence within calibrated band",
            record={"js_divergence": 0.02, "wasserstein": 0.1},
        ),
    ]
    return build_assurance_case_bundle("Grid Supervisor", evidence)


def test_report_carries_traceability_header_and_disclaimer() -> None:
    report = render_conformity_report(_bundle())

    assert report.startswith("# Conformity Evidence Report — Grid Supervisor")
    assert CONFORMITY_REPORT_SCHEMA in report
    assert f"`{_bundle().bundle_hash}`" in report
    assert "| Actuation permitted | No (review-only) |" in report
    assert _bundle().disclaimer in report
    assert report.endswith("\n")


def test_report_renders_every_standard_section() -> None:
    report = render_conformity_report(_bundle())

    assert f"## {EU_AI_ACT}" in report
    assert f"## {ISO_IEC_42001}" in report
    assert f"## {UL_4600}" in report
    assert "## Coverage summary" in report
    assert "## Evidence inventory" in report
    assert "| Clause | Title | Status | Evidence | Rationale |" in report


def test_report_shows_status_labels_and_addressed_evidence() -> None:
    report = render_conformity_report(_bundle())

    # Article 12 (record-keeping) is addressed by the audit evidence.
    assert "Addressed" in report
    assert "Partially addressed" in report
    assert "audit-chain-integrity" in report
    assert "| `" in report  # evidence inventory content-hash cells


def test_report_marks_unaddressed_clauses_with_a_dash() -> None:
    # No formal-verification or conformal-gate evidence is supplied, so some
    # clauses are not addressed and must render an em dash in the evidence cell.
    report = render_conformity_report(_bundle())

    assert "Not addressed" in report
    assert "| — |" in report


def test_report_escapes_pipes_in_cell_text() -> None:
    report = render_conformity_report(_bundle())

    # The piped summary must be escaped so it cannot break the table.
    assert "run-a \\| run-b" in report
    assert "run-a | run-b" not in report


def test_report_is_deterministic() -> None:
    bundle = _bundle()
    assert render_conformity_report(bundle) == render_conformity_report(bundle)


def test_status_label_passes_through_an_unknown_status() -> None:
    assert _status_label("addressed") == "Addressed"
    assert _status_label("speculative") == "speculative"


def test_pdf_report_is_a_deterministic_pdf() -> None:
    bundle = _bundle()
    pdf = render_conformity_report_pdf(bundle)

    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert b"CONFORMITY EVIDENCE REPORT" in pdf
    assert render_conformity_report_pdf(bundle) == pdf


def test_certification_package_seals_the_conformity_report() -> None:
    evidence = [
        build_evidence_item(
            evidence_id="audit-chain-integrity",
            category=AUDIT_LOGGING,
            summary="Audit-chain integrity",
            record={"integrity_ok": True},
        ),
    ]
    package = build_certification_evidence_package("Grid Supervisor", evidence)
    files = package.to_files()

    report = files["conformity_report.md"].decode("utf-8")
    assert report == render_conformity_report(package.assurance_bundle)
    assert files["conformity_report.pdf"] == render_conformity_report_pdf(
        package.assurance_bundle
    )

    rows = {row["path"]: row for row in package.manifest["files"]}
    assert set(rows) == {
        "assurance_bundle.json",
        "conformity_report.md",
        "conformity_report.pdf",
        "test_vectors.json",
    }
    assert (
        rows["conformity_report.md"]["sha256"]
        == hashlib.sha256(report.encode("utf-8")).hexdigest()
    )
    assert (
        rows["conformity_report.pdf"]["sha256"]
        == hashlib.sha256(files["conformity_report.pdf"]).hexdigest()
    )
