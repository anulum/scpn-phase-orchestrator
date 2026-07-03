# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Poincare monitor API reference documentation tests

from __future__ import annotations

from pathlib import Path

POINCARE_REFERENCE = Path("docs/reference/api/monitor_poincare.md")


def test_poincare_api_reference_meets_depth_baseline() -> None:
    doc = POINCARE_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_poincare_api_reference_documents_monitor_contracts() -> None:
    doc = POINCARE_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "PoincareResult",
        "poincare_section",
        "phase_poincare",
        "return_times",
        "ACTIVE_BACKEND",
        "AVAILABLE_BACKENDS",
        "Rust, Mojo, Julia, and Go",
        "sample-index units",
        "strictly increasing",
        "boolean aliases",
        "numeric-string aliases",
        "complex aliases",
        "backend-output",
        "Python fallback",
        "direction IDs",
        "UPDEEngine",
        "observational",
    )

    for phrase in required_phrases:
        assert phrase in doc
