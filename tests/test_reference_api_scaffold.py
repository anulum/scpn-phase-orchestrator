# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Scaffold API reference documentation tests

from __future__ import annotations

from pathlib import Path

SCAFFOLD_REFERENCE = Path("docs/reference/api/scaffold.md")


def test_scaffold_api_reference_meets_depth_baseline() -> None:
    doc = SCAFFOLD_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_scaffold_api_reference_documents_review_contracts() -> None:
    doc = SCAFFOLD_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "LLMScaffoldConfig",
        "LLMScaffoldProposal",
        "StaticJSONScaffoldProvider",
        "LocalHTTPScaffoldProvider",
        "propose_domainpack_from_description",
        "configured_llm_scaffold_provider",
        "strict JSON object",
        "review-only",
        "binding_spec.yaml",
        "llm_scaffold_audit.json",
        "prompt-override",
        "load_binding_spec",
        "validate_binding_spec",
        "spo scaffold",
    )

    for phrase in required_phrases:
        assert phrase in doc
