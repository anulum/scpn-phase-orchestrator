# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Coupling inference API reference documentation tests

from __future__ import annotations

from pathlib import Path

COUPLING_INFER_REFERENCE = Path("docs/reference/api/coupling_infer.md")


def test_coupling_infer_api_reference_meets_depth_baseline() -> None:
    doc = COUPLING_INFER_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_coupling_infer_api_reference_documents_inference_contracts() -> None:
    doc = COUPLING_INFER_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "CouplingInferenceConfig",
        "CouplingInferenceResult",
        "auto_coupling_estimation",
        "infer_coupling_from_timeseries",
        "auto-coupling-estimation",
        "source-to-target",
        "target-by-source",
        "to_upde_knm",
        "to_audit_record",
        "transfer entropy",
        "threshold_absolute takes precedence",
        "boolean aliases",
        "complex values",
        "non-finite samples",
        "backend score matrices",
        "NotImplementedError",
        "review-only",
    )

    for phrase in required_phrases:
        assert phrase in doc
