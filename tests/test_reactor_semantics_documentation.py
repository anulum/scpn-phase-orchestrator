# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor-semantic documentation guard

"""Keep the public U0 ownership and epistemic boundary explicit."""

from __future__ import annotations

from pathlib import Path

REFERENCE = Path("docs/reference/api/reactor_semantics.md")


def test_reactor_semantics_reference_preserves_u0_boundaries() -> None:
    text = " ".join(REFERENCE.read_text(encoding="utf-8").split())
    required = (
        "SCPN-FUSION-CORE",
        "SCPN-MIF-CORE",
        "SCPN-CONTROL",
        "review_only",
        "nine non-exclusive design slices",
        "Eight carriers are not interchangeable",
        "cannot acquire an angle merely by normalization",
        "Legacy `FusionCoreBridge.observables_to_phases()`",
        "not a valid U0 semantic producer",
        "observability threshold",
        "picosecond offset",
        "validate_observable_sequence",
        "duplicate JSON keys",
        "schema `1.0.0`",
        "reactor_semantics_u0.schema.json",
        "ReactorSemanticHandoff",
        "handoff_from_bytes()",
        "zero phase confidence and observability",
        "actionable=false",
        "reactor_semantic_handoff.schema.json",
        "Device diagnostic-plan design review",
        "device_diagnostic_plan_review_from_producer_bytes()",
        "device_diagnostic_plan_review.schema.json",
        "simulation-monotonic evidence to wall time implicitly",
        "FAIR-MAST magnetic physical-source review",
        "mast_magnetic_source_review_from_producer_bytes()",
        "72 arrays, 11 measurement families, 132 qualified channel records",
        "not instrument clocks",
        "observation_admitted=false",
        "qualified_phase_evidence=false",
        "mast_magnetic_source_review.schema.json",
    )

    for marker in required:
        assert marker in text
