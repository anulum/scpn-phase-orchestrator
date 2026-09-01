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
OCCURRENCE_LEDGER = Path("docs/reference/reactor_signal_occurrence_ledger.md")
CONFIGURATION_COVERAGE = Path(
    "docs/reference/reactor_configuration_evidence_coverage.md"
)


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
        "tokamak, dense- plasma-focus, MagLIF, mechanical-or-liquid-liner MIF, "
        "plasma-jet MIF, and laser-ICF device plans",
        "discharge-current `event_cycle`",
        "neck-mode `complex_mode`",
        "simulation-clocked `numerical_phase`",
        "compression-trajectory `bounded_feature`",
        "liner-arrival `event_cycle`",
        "resolved-asymmetry `complex_mode`",
        "The two liner-MIF reviews keep compression-trajectory",
        "Plasma-jet MIF separately keeps convergence trajectory",
        "jet-arrival `event_cycle`",
        "merge-asymmetry `complex_mode`",
        "Laser ICF separately keeps beam timing `event_cycle`",
        "shot outcome and implosion trajectory as distinct `bounded_feature` rows",
        "one shared laser-ICF plan does not equate direct drive, indirect drive",
        "all seven device-plan-only semantic-ingress profiles remain `not_declared`",
        "Passing a design review does not add source evidence",
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


def test_reactor_signal_occurrence_ledger_preserves_epistemic_boundaries() -> None:
    text = " ".join(OCCURRENCE_LEDGER.read_text(encoding="utf-8").split())
    required = (
        "39 stable public or cross-project occurrence groups",
        "An occurrence is evidence that a concept or value exists in source",
        "It is not by itself evidence that the concept was measured in a reactor",
        "physical_observation_admitted=false",
        "physical_phase_eligible=false",
        "owns MIF facts only",
        "admitted_for_review` is not an actuation decision",
        "no path that turns an SPO semantic or regime-assessment record",
        "The physical-observation gap remains open",
        "reactor_signal_occurrence_ledger.v1.json",
        "reactor_signal_occurrence_ledger.schema.json",
    )

    for marker in required:
        assert marker in text


def test_reactor_configuration_coverage_is_exhaustive_and_fail_closed() -> None:
    text = " ".join(CONFIGURATION_COVERAGE.read_text(encoding="utf-8").split())
    required = (
        "All **32 built-in configurations** across **8 confinement families**",
        "**2 configurations** have an exercised, byte-canonical, review-only "
        "producer adapter",
        "**27 configurations** have no configuration-specific source occurrence",
        "**30 configurations** have no portable producer-to-SPO semantic ingress "
        "profile",
        "**0 configurations** have a qualified physical observation",
        "No row inherits evidence from another configuration",
        "`spherical_tokamak` has an exact physical-source review",
        "`semantic_ingress_state=not_declared`",
        "Design declarations are not evidence ingress",
        "dense- plasma-focus, MagLIF, mechanical-or-liquid-liner MIF, plasma-jet "
        "MIF, and laser-ICF producer fixtures",
        "the three laser-ICF configurations do not change their seven matrix rows",
        "Neither MagLIF, mechanical-or-liquid-liner MIF, nor plasma-jet MIF "
        "inherits the verified `frc_compression_mif` adapter",
        "the three MIF design reviews also do not provide evidence for one another",
        "Direct-drive, indirect-drive, and fast/shock-ignition laser ICF do not "
        "provide evidence for one another",
        "projectile, or impact ICF",
        "`machine_protection_final_veto=true`",
        "reactor_configuration_evidence_coverage.v1.json",
        "reactor_configuration_evidence_coverage.schema.json",
    )

    for marker in required:
        assert marker in text
