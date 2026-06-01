# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Python facade API reference documentation tests

from __future__ import annotations

from pathlib import Path

API_REFERENCE = Path("docs/reference/api/api.md")


def test_python_facade_api_reference_meets_depth_baseline() -> None:
    doc = API_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_python_facade_api_reference_documents_execution_contracts() -> None:
    doc = API_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "Orchestrator.from_yaml",
        "Orchestrator.run",
        "OrchestratorState",
        "to_record",
        "scpn import alias",
        "research-tier",
        "Kuramoto binding specs",
        "no hardware actuation",
        "no network IO",
        "BindingSpec",
        "CouplingBuilder",
        "UPDEEngine",
        "compute_order_parameter",
        "validate_binding_spec",
        "deterministic local simulation",
    )

    for phrase in required_phrases:
        assert phrase in doc
