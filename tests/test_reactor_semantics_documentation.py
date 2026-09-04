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
PLAN_PORTFOLIO_STATUS = Path(
    "docs/reference/reactor_diagnostic_plan_portfolio_status.md"
)
TECHNOLOGY_DIAGNOSTIC_ATLAS = Path(
    "docs/reference/reactor_technology_diagnostic_atlas.md"
)
PRODUCER_EVIDENCE_PRIORITY_REGISTER = Path(
    "docs/reference/reactor_producer_evidence_priority_register.md"
)
DEVICE_PHYSICAL_EVIDENCE_REQUEST = Path(
    "docs/reference/device_physical_evidence_request.md"
)
CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST = Path(
    "docs/reference/conventional_tokamak_physical_payload_request.md"
)
FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST = Path(
    "docs/reference/frc_compression_mif_physical_payload_request.md"
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
        "`ProducerEvidenceDisposition`",
        "`low_observability` | `unobservable`",
        "not physical regime labels",
        "`physical_regime_classified=false`",
        "`quality_may_substitute=false`",
        "a merely small score above that gate does not qualify",
        "versioned validated applicability domain",
        "declared freshness or validity window",
        "ReactorSemanticHandoff",
        "handoff_from_bytes()",
        "zero phase confidence and observability",
        "actionable=false",
        "reactor_semantic_handoff.schema.json",
        "Device diagnostic-plan design review",
        "device_diagnostic_plan_review_from_producer_bytes()",
        "device_diagnostic_plan_review.schema.json",
        "Accepted-plan to physical-evidence request",
        "device_physical_evidence_request_from_plan_review()",
        "The first materialised instance targets only `laser_icf_direct_drive`",
        "The second instance targets only `ion_beam_icf`",
        "host-independent materialisation CLI reads only exact local fixture bytes",
        "The third instance targets only `projectile_or_impact_icf`",
        "public factory refuses laser, ion-beam, pulsed-electron- beam",
        "The fourth instance targets only `pulsed_electron_beam_icf`",
        "without transferring a physical sample, phase, validity, regime",
        "source-bound gaps or provisional candidates",
        "device_physical_evidence_request.schema.json",
        "versions `1.1.0` and `1.2.0` to separate exact plan and channel shapes",
        "Signal quantity and unit text cannot override the registered candidate",
        "embedded byte-for-byte in the review output",
        "tokamak, dense- plasma-focus, theta-pinch, Z-pinch, MagLIF, "
        "mechanical-or-liquid-liner MIF, plasma-jet MIF, laser-ICF, ICF-beam, "
        "and ICF-impact device plans",
        "discharge-current `event_cycle`",
        "neck-mode `complex_mode`",
        "simulation-clocked `numerical_phase`",
        "theta-pinch review separately keeps a shot-relative bank-waveform",
        "rotation-probe `complex_mode`",
        "Z-pinch review separately keeps a shot-relative current-and-voltage",
        "pinch-mode `complex_mode`",
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
        "Beam ICF separately keeps bunch timing `event_cycle`",
        "The ICF-beam plan does not equate its ion-beam and pulsed-electron-beam "
        "configurations",
        "Impact ICF separately keeps impact timing `event_cycle`",
        "The ICF-impact plan does not inherit either plan's evidence",
        "all thirteen device-plan-only semantic-ingress profiles remain `not_declared`",
        "One shared Z-pinch plan does not equate the `z_pinch` and "
        "`sheared_flow_z_pinch` configurations",
        "Passing a design review does not add source evidence",
        "reactor diagnostic-plan portfolio status",
        "all 22 Reactor Systems device projects",
        "Seven exact producer objects are accepted",
        "thirteen are refused",
        "lattice and muon architecture projects have no declared diagnostic plan",
        "simulation-monotonic evidence to wall time implicitly",
        "FAIR-MAST magnetic physical-source review",
        "mast_magnetic_source_review_from_producer_bytes()",
        "72 arrays, 11 measurement families, 132 qualified channel records",
        "not instrument clocks",
        "observation_admitted=false",
        "qualified_phase_evidence=false",
        "mast_magnetic_source_review.schema.json",
        "FAIR-MAST phase-qualification prerequisite request",
        "mast_phase_qualification_request_from_source_review()",
        "thirteen missing evidence obligations",
        "low_observability",
        "Provider quality is orthogonal",
        "blocked_missing_producer_evidence",
        "mast_phase_qualification_request.schema.json",
        "Conventional-tokamak L1 physical-payload request",
        "conventional_tokamak_physical_payload_request()",
        "reusable_as_physical_evidence=false",
        "allocate a new configuration-specific canonical physical payload",
        "conventional_tokamak_physical_payload_request.schema.json",
        "FRC-compression MIF L1 physical-payload request",
        "frc_compression_mif_physical_payload_request()",
        "simulation-monotonic clock",
        "four unselected FRC-compression-MIF candidates",
        "frc_compression_mif_physical_payload_request.schema.json",
    )

    for marker in required:
        assert marker in text


def test_reactor_diagnostic_plan_portfolio_status_is_fail_closed() -> None:
    text = " ".join(PLAN_PORTFOLIO_STATUS.read_text(encoding="utf-8").split())
    required = (
        "**22 device projects** were examined",
        "**7 producer objects** are structurally accepted",
        "**13 are refused**",
        "**2 architecture-only projects** have no declared diagnostic plan",
        "**0 current fixtures** have byte-identical SPO custody",
        "**7 fixtures** are digest-pinned public producer objects",
        "**148/154 hosted workflows** completed successfully",
        "**0 fixtures** constitute a qualified physical observation",
        "must not be purged while unresolved",
        "reactor_diagnostic_plan_portfolio_status.v1.json",
        "reactor_diagnostic_plan_portfolio_status.schema.json",
    )

    for marker in required:
        assert marker in text


def test_reactor_technology_diagnostic_atlas_is_exhaustive_and_fail_closed() -> None:
    text = " ".join(TECHNOLOGY_DIAGNOSTIC_ATLAS.read_text(encoding="utf-8").split())
    required = (
        "all **34 registered configurations** across **9 confinement families**",
        "**37 primary sources**",
        "broader than tokamaks",
        "technology-readiness levels",
        "related topologies never inherit evidence",
        "`admission_state=refused_no_producer_evidence`",
        "a paper or facility page is not a producer payload",
        "SCPN-FUSION-CORE and SCPN-MIF-CORE remain owners",
        "zero admitted physical observations",
        "zero qualified physical phases",
        "zero CONTROL admissions",
        "do not establish net energy gain",
        "`machine_protection_final_veto=true`",
        "reactor_technology_diagnostic_atlas.v1.json",
        "reactor_technology_diagnostic_atlas.schema.json",
    )

    for marker in required:
        assert marker in text


def test_reactor_producer_evidence_priority_is_non_scalar_and_fail_closed() -> None:
    text = " ".join(
        PRODUCER_EVIDENCE_PRIORITY_REGISTER.read_text(encoding="utf-8").split()
    )
    required = (
        "all **34 registered configurations** across **9 confinement families**",
        "**24 upstream reactor projects**",
        "**23 distinct `device_project` owners**",
        "**22 Reactor Systems device repositories**",
        "No opaque or additive priority score is emitted",
        "rows within one lane are deliberately unordered",
        "External `E5` through `E0` evidence ranks remain context only",
        "`spherical_tokamak` is the only L0 row",
        "reviewed FAIR-MAST physical-source bytes",
        "controlled phenomenon identity",
        "reproducible source-ingestion state",
        "mast_phase_qualification_request_from_source_review()",
        "conventional_tokamak_physical_payload_request()",
        "frc_compression_mif_physical_payload_request()",
        "forbidding reuse as physical evidence",
        "Both current adapters are simulation-only",
        "The two namespaced extensions",
        "Sixteen configurations map to thirteen refused producer objects",
        "zero complete physical evidence chains",
        "zero qualified physical observations",
        "zero qualified physical phases",
        "zero CONTROL admissions",
        "`machine_protection_final_veto=true`",
        "`629c04b00cce05d835e8d4dd1d0cb8ee586cb725a363a49aebed3294da61615d`",
        "direct-drive laser-ICF",
        "`device_physical_evidence_request_from_plan_review()`",
        "specific to `ion_beam_icf`",
        "fourth materialised L3 boundary is specific to `pulsed_electron_beam_icf`",
        "specific to `projectile_or_impact_icf`",
        "No laser, beam, or generic target configuration inherits",
        "reactor_producer_evidence_priority_register.v1.json",
        "reactor_producer_evidence_priority_register.schema.json",
    )

    for marker in required:
        assert marker in text


def test_device_physical_evidence_request_is_exact_and_fail_closed() -> None:
    text = " ".join(
        DEVICE_PHYSICAL_EVIDENCE_REQUEST.read_text(encoding="utf-8").split()
    )
    required = (
        "common L3 boundary",
        "scoped only to `laser_icf_direct_drive`",
        "second materialised request uses exact `SCPN-ICF-BEAM-CORE` fixture bytes",
        "selects only `ion_beam_icf`",
        "produces a distinct request ID",
        "host-independent",
        "imports no producer module",
        "third materialised request selects only `projectile_or_impact_icf`",
        "public factory refuses each as a configuration mismatch",
        "fourth materialised request selects only `pulsed_electron_beam_icf`",
        "Shared plan-review custody is not shared physical evidence",
        "event-relative `event_cycle`",
        "`derived_cyclic`",
        "`noncyclic_feature`",
        "`numerical_only`",
        "All thirteen prerequisites are explicitly missing",
        "producer evidence-state contract",
        "New peer discoveries enter the atlas as explicit gaps or provisional",
        "A peer report alone is not physical evidence",
        "`selected_candidate_id=null`",
        "`control_admission_requested=false`",
        "`direct_actuation=false`",
        "`machine_protection_final_veto=true`",
        "`3f273e5ef1fb68e7a928913a7f7a8c9b5e6055a7649c722598911fa39458111a`",
        "`f42a9817dcef628caefab5ba5681853327bae9b21ba72459eb9588e14c2ed6a9`",
        "`b381e5d5dc8aaff311da8f7d0453ed458f154f3930dd1a3297df07d366d93854`",
        "`c36256af2280a5caf786953c0c1e293b552f128acb02704123ea8073c5153b9b`",
        "`27a576dd67b149069bd4eefa1ef343c570a0084688acd3370721e6a34023ac62`",
        "`ccdda701953cdec025d3b7f63f026bbaf92efed54ecebddfbb510eb83eab64e1`",
        "`bbe0825d5aeb893089a10bb6ec6d94decf76dbc2b5f93b735ff704885f63c2e7`",
        "`9461ddbc89f623bb0f6d2584e6734eef66e5c9abc1c94f4b18ca131acc9fa15a`",
        "device_physical_evidence_request.schema.json",
    )

    for marker in required:
        assert marker in text


def test_conventional_tokamak_physical_request_is_exact_and_fail_closed() -> None:
    text = " ".join(
        CONVENTIONAL_TOKAMAK_PHYSICAL_PAYLOAD_REQUEST.read_text(
            encoding="utf-8"
        ).split()
    )
    required = (
        "`L1_extend_exercised_review_adapter`",
        "`conventional_tokamak_physical_payload_request()`",
        "`source_kind=simulation`",
        "`reusable_as_physical_evidence=false`",
        "all thirteen prerequisites",
        "`low_observability`",
        "`quality_state_may_substitute_for_evidence_state=false`",
        "`fec4e93971190c7183410f200c60a9ef0ffcfeaf01fa69f9fc3514e9e352603c`",
        "`a506c0ad7c37ee53719b3f2194906b39585e4293e1c5f9d25245f987c0b08945`",
        "conventional-tokamak-specific diagnostic",
        "independent validation without same-shot circularity",
        "New peer discoveries enter the atlas as gaps or provisional candidates",
        "`selected_candidate_id=null`",
        "`control_admission_requested=false`",
        "`direct_actuation=false`",
        "conventional_tokamak_physical_payload_request.schema.json",
    )

    for marker in required:
        assert marker in text


def test_frc_compression_mif_physical_request_is_exact_and_fail_closed() -> None:
    text = " ".join(
        FRC_COMPRESSION_MIF_PHYSICAL_PAYLOAD_REQUEST.read_text(encoding="utf-8").split()
    )
    required = (
        "`L1_extend_exercised_review_adapter`",
        "`frc_compression_mif_physical_payload_request()`",
        "SCPN-MIF-CORE merge-compression adapter",
        "`source_kind=simulation`",
        "`reusable_as_physical_evidence=false`",
        "all thirteen prerequisites",
        "FRC-compression-MIF-specific diagnostic",
        "`unknown`, `out_of_distribution`, `low_observability`, and `stale`",
        "does not close the gap",
        "`plant_truth_state_contract_present=false`",
        "driver arrival",
        "simulation-only synthetic oscillator coordinate",
        "New peer discoveries enter the atlas as gaps or provisional candidates",
        "`selected_candidate_id=null`",
        "`control_admission_requested=false`",
        "`direct_actuation=false`",
        "frc_compression_mif_physical_payload_request.schema.json",
    )

    for marker in required:
        assert marker in text


def test_reactor_signal_occurrence_ledger_preserves_epistemic_boundaries() -> None:
    text = " ".join(OCCURRENCE_LEDGER.read_text(encoding="utf-8").split())
    required = (
        "43 stable public or cross-project occurrence groups",
        "An occurrence is evidence that a concept or value exists in source",
        "It is not by itself evidence that the concept was measured in a reactor",
        "physical_observation_admitted=false",
        "physical_phase_eligible=false",
        "White Rabbit TAI seconds",
        "`phase_locked` place is a protocol state",
        "`STATE-01`",
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
        "All **34 registered configurations** across **9 confinement families**",
        "**2 configurations** have an exercised, byte-canonical, review-only "
        "producer adapter",
        "**29 configurations** have no configuration-specific source occurrence",
        "**32 configurations** have no portable producer-to-SPO semantic ingress "
        "profile",
        "**0 configurations** have a qualified physical observation",
        "No row inherits evidence from another configuration",
        "`spherical_tokamak` has an exact physical-source review",
        "`semantic_ingress_state=not_declared`",
        "Design declarations are not evidence ingress",
        "dense- plasma-focus, theta-pinch, Z-pinch, MagLIF, "
        "mechanical-or-liquid-liner MIF, plasma-jet MIF, laser-ICF, ICF-beam, "
        "and ICF-impact producer fixtures",
        "the two ICF-beam configurations, and the ICF-impact configuration do "
        "not change their thirteen matrix rows",
        "Neither MagLIF, mechanical-or-liquid-liner MIF, nor plasma-jet MIF "
        "inherits the verified `frc_compression_mif` adapter",
        "the three MIF design reviews also do not provide evidence for one another",
        "Direct-drive, indirect-drive, and fast/shock-ignition laser ICF do not "
        "provide evidence for one another",
        "ion/electron-beam, projectile, or impact ICF evidence",
        "Ion-beam and pulsed-electron-beam ICF likewise do not provide evidence",
        "generic beam-target configurations",
        "Projectile-or-impact ICF does not inherit evidence from any laser or "
        "beam-driven ICF design review",
        "Theta pinch likewise inherits no evidence from dense plasma focus, "
        "z-pinch, or any other self-magnetic configuration",
        "The shared Z-pinch design review does not equate `z_pinch` with "
        "`sheared_flow_z_pinch`",
        "`machine_protection_final_veto=true`",
        "`7d56f34fdb5c0863813c954d5ad38bb0c1f1dd129ebfc5d93635dfdc47daf5f2`",
        "`8210dc2310a7031ccad1a1675677e3e92007a2dd82e696c39d25202d2f9f022f`",
        "reactor_configuration_evidence_coverage.v1.json",
        "reactor_configuration_evidence_coverage.schema.json",
    )

    for marker in required:
        assert marker in text
