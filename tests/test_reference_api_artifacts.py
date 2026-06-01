# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Artifact API reference documentation tests

from __future__ import annotations

from pathlib import Path

ARTIFACTS_REFERENCE = Path("docs/reference/api/artifacts.md")


def test_artifacts_api_reference_meets_depth_baseline() -> None:
    doc = ARTIFACTS_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_artifacts_api_reference_documents_safety_contracts() -> None:
    doc = ARTIFACTS_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "QPUDataArtifact",
        "artifact_sha256",
        "K_nm",
        "theta0",
        "publication-safe",
        "finite JSON",
        "zero diagonal",
        "symmetric",
        "non-negative",
        "SCPN Quantum Control",
        "read_qpu_data_artifact",
        "write_qpu_data_artifact",
    )

    for phrase in required_phrases:
        assert phrase in doc
