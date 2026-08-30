# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-reactor regime and mode ontology tests

"""Public fail-closed tests for cross-family regime and mode meanings."""

from __future__ import annotations

from dataclasses import replace

import pytest

import scpn_phase_orchestrator.reactor_semantics as semantics
from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGIME_MODE_ONTOLOGY_VERSION,
    AxisApplicability,
    EvidenceClass,
    ModeDomain,
    ReactorModeBinding,
    ReactorModeDefinition,
    ReactorRegimeAxisAssignment,
    ReactorRegimeModeOntologyRegistry,
    SemanticCarrier,
)


def _physical_binding(**changes: object) -> ReactorModeBinding:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_mode(
        "physical.closed.resolved_mhd_mode"
    )
    values: dict[str, object] = {
        "definition": definition,
        "configuration": "conventional_tokamak",
        "carrier": SemanticCarrier.COMPLEX_MODE,
        "evidence_class": EvidenceClass.EXPERIMENTAL,
        "mode_identity": "tokamak.mhd.n1",
        "harmonic_coordinates": (2, 1),
        "observation_operator_id": "operator.magnetic_array.v1",
        "reference_frame": "tokamak.flux_coordinates.v1",
        "reference_signal_id": "diagnostic.magnetic_reference",
        "orientation": "positive_toroidal",
        "phase_origin": "reference_signal_zero_crossing",
        "wrap_convention": "zero_to_two_pi",
        "observability_threshold": 0.7,
        "validity_id": "validity.mode_window",
        "quality_id": "quality.mode_fit",
        "provenance_id": "provenance.mode_fit",
    }
    values.update(changes)
    return ReactorModeBinding(**values)  # type: ignore[arg-type]


def _numerical_binding(**changes: object) -> ReactorModeBinding:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_mode(
        "numerical.model.synthetic_oscillator_coordinate"
    )
    values: dict[str, object] = {
        "definition": definition,
        "configuration": "frc_compression_mif",
        "carrier": SemanticCarrier.NUMERICAL_PHASE,
        "evidence_class": EvidenceClass.SIMULATION,
        "mode_identity": "mif.oscillator.theta_0",
        "harmonic_coordinates": None,
        "observation_operator_id": None,
        "reference_frame": None,
        "reference_signal_id": None,
        "orientation": None,
        "phase_origin": None,
        "wrap_convention": None,
        "observability_threshold": None,
        "validity_id": "validity.solver_step",
        "quality_id": "quality.solver_state",
        "provenance_id": "provenance.mif_model",
    }
    values.update(changes)
    return ReactorModeBinding(**values)  # type: ignore[arg-type]


def test_default_ontology_covers_all_reactors_and_public_facade() -> None:
    registry = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY

    assert registry.version == REACTOR_REGIME_MODE_ONTOLOGY_VERSION
    assert semantics.DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY is registry
    assert len(registry.axes) == 8
    assert len(registry.modes) == 7
    assert len(registry.digest) == 64
    assert registry.to_record()["authority"] == "review_only"
    assert registry.to_record()["actionable"] is False
    assert registry.observability_registry_digest == (
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
    )
    for configuration in DEFAULT_REACTOR_REGISTRY.configurations:
        modes = registry.modes_for_configuration(configuration)
        assert modes
        assert any(mode.domain is ModeDomain.NUMERICAL for mode in modes)
        assert all(configuration in mode.configurations for mode in modes)


def test_axis_applicability_is_distinct_from_classification() -> None:
    registry = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY

    assert (
        registry.applicability_for("plant_readiness", "beam_target")
        is AxisApplicability.APPLICABLE
    )
    assert (
        registry.applicability_for("stability_or_symmetry", "stellarator")
        is AxisApplicability.APPLICABLE
    )
    assert (
        registry.applicability_for("stability_or_symmetry", "beam_target")
        is AxisApplicability.NOT_APPLICABLE
    )
    assert (
        registry.applicability_for("driver_synchronization", "frc")
        is AxisApplicability.NOT_APPLICABLE
    )
    assert (
        registry.applicability_for("exhaust_or_boundary", "simple_magnetic_mirror")
        is AxisApplicability.APPLICABLE
    )
    with pytest.raises(ValueError, match="unknown regime axis"):
        registry.resolve_axis("nonexistent_axis")
    with pytest.raises(ValueError, match="unknown mode definition"):
        registry.resolve_mode("physical.generic_mode")


def test_axis_assignment_projects_unknown_and_non_applicable_without_nominal() -> None:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(
        "stability_or_symmetry"
    )
    applicable = ReactorRegimeAxisAssignment(
        definition=definition,
        applicability=AxisApplicability.APPLICABLE,
        label="coherent_mode",
        confidence=0.8,
        evidence_ids=("evidence.mode_fit",),
        applicability_basis=("context.stellarator",),
    )
    unavailable = ReactorRegimeAxisAssignment(
        definition=definition,
        applicability=AxisApplicability.NOT_APPLICABLE,
        label=None,
        confidence=0.0,
        evidence_ids=(),
        applicability_basis=("context.beam_target",),
    )
    unknown = ReactorRegimeAxisAssignment(
        definition=definition,
        applicability=AxisApplicability.UNKNOWN,
        label=None,
        confidence=0.0,
        evidence_ids=(),
        applicability_basis=(),
    )

    assert applicable.to_regime_axis().label == "coherent_mode"
    assert unavailable.to_regime_axis().label == "not_applicable"
    assert unknown.to_regime_axis().label == "unknown"
    assert "nominal" not in {
        applicable.to_regime_axis().label,
        unavailable.to_regime_axis().label,
        unknown.to_regime_axis().label,
    }
    assert unavailable.to_record()["applicability_basis"] == ["context.beam_target"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"applicability": "applicable"}, "AxisApplicability"),
        ({"label": None}, "classified label"),
        ({"label": "unknown"}, "classified label"),
        ({"label": "invented"}, "not defined"),
        ({"evidence_ids": ()}, "classification evidence"),
        ({"evidence_ids": ("evidence.x", "evidence.x")}, "must be unique"),
        ({"authority": "control"}, "review-only"),
        ({"actionable": True}, "review-only"),
    ],
)
def test_applicable_axis_assignment_refuses_invalid_claims(
    changes: dict[str, object], message: str
) -> None:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(
        "stability_or_symmetry"
    )
    values: dict[str, object] = {
        "definition": definition,
        "applicability": AxisApplicability.APPLICABLE,
        "label": "coherent_mode",
        "confidence": 0.8,
        "evidence_ids": ("evidence.mode_fit",),
        "applicability_basis": ("context.stellarator",),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        ReactorRegimeAxisAssignment(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("applicability", "changes", "message"),
    [
        (AxisApplicability.NOT_APPLICABLE, {"label": "unresolved"}, "forbids"),
        (AxisApplicability.NOT_APPLICABLE, {"confidence": 0.1}, "must be zero"),
        (AxisApplicability.NOT_APPLICABLE, {"applicability_basis": ()}, "basis"),
        (AxisApplicability.UNKNOWN, {"label": "unresolved"}, "forbids"),
        (AxisApplicability.UNKNOWN, {"confidence": 0.1}, "must be zero"),
    ],
)
def test_nonclassified_axis_assignment_fails_closed(
    applicability: AxisApplicability,
    changes: dict[str, object],
    message: str,
) -> None:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(
        "stability_or_symmetry"
    )
    values: dict[str, object] = {
        "definition": definition,
        "applicability": applicability,
        "label": None,
        "confidence": 0.0,
        "evidence_ids": (),
        "applicability_basis": ("context.beam_target",),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        ReactorRegimeAxisAssignment(**values)  # type: ignore[arg-type]


def test_physical_and_numerical_bindings_remain_epistemically_distinct() -> None:
    physical = _physical_binding()
    numerical = _numerical_binding()

    assert physical.configuration == "conventional_tokamak"
    assert physical.to_record()["domain"] == "physical"
    assert physical.to_record()["harmonic_coordinates"] == [2, 1]
    assert numerical.to_record()["domain"] == "numerical"
    assert numerical.to_record()["harmonic_coordinates"] is None
    assert numerical.to_record()["candidate_id"] == (
        "model.synthetic_oscillator_coordinate"
    )
    alias = _physical_binding(configuration="rfx")
    assert alias.configuration == "reversed_field_pinch"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"configuration": "beam_target"}, "not defined"),
        ({"carrier": SemanticCarrier.NUMERICAL_PHASE}, "carrier"),
        ({"evidence_class": EvidenceClass.CONCEPT}, "evidence"),
        ({"mode_identity": "not an identifier"}, "mode_identity"),
        ({"observation_operator_id": None}, "missing fields"),
        ({"reference_frame": None}, "missing fields"),
        ({"reference_signal_id": None}, "missing fields"),
        ({"orientation": None}, "missing fields"),
        ({"phase_origin": None}, "missing fields"),
        ({"wrap_convention": None}, "missing fields"),
        ({"harmonic_coordinates": None}, "harmonic coordinates"),
        ({"harmonic_coordinates": (0, 1)}, "positive integers"),
        ({"harmonic_coordinates": (True, 1)}, "positive integers"),
        ({"observability_threshold": None}, "observability threshold"),
        ({"observability_threshold": 2.0}, r"in \[0, 1\]"),
        ({"validity_id": "bad id"}, "validity_id"),
        ({"authority": "control"}, "review-only"),
        ({"actionable": True}, "review-only"),
    ],
)
def test_physical_mode_binding_refuses_incomplete_or_mismatched_evidence(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _physical_binding(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"carrier": SemanticCarrier.CYCLIC_PHASE}, "carrier"),
        ({"evidence_class": EvidenceClass.OBSERVED}, "evidence"),
        ({"harmonic_coordinates": (1, 1)}, "no physical harmonic"),
    ],
)
def test_numerical_binding_cannot_claim_physical_mode_evidence(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _numerical_binding(**changes)


def test_axis_definition_refuses_open_vocabulary_and_authority_escalation() -> None:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis("plant_readiness")
    cases = (
        ({"labels": ()}, "non-empty"),
        ({"labels": ("commissioning", "commissioning")}, "unique"),
        ({"labels": ("commissioning",)}, "include unknown"),
        ({"candidate_ids": ("x", "x")}, "candidate identifiers"),
        ({"required_evidence": ()}, "evidence requirements"),
        ({"required_evidence": ("validity", "validity")}, "evidence requirements"),
        ({"applicability_policy": "universal"}, "AxisApplicabilityPolicy"),
        ({"authority": "action"}, "review-only"),
        ({"actionable": True}, "review-only"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(definition, **changes)  # type: ignore[arg-type]
    assert definition.to_record()["axis_id"] == "plant_readiness"


def test_mode_definition_refuses_domain_and_vocabulary_mismatch() -> None:
    physical = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_mode(
        "physical.closed.resolved_mhd_mode"
    )
    numerical = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_mode(
        "numerical.model.synthetic_oscillator_coordinate"
    )
    cases = (
        (physical, {"domain": "physical"}, "ModeDomain"),
        (physical, {"configurations": ()}, "mode configurations"),
        (physical, {"configurations": ("stellarator", "stellarator")}, "unique"),
        (physical, {"configurations": ("unknown_reactor",)}, "unknown reactor"),
        (physical, {"admissible_carriers": ()}, "mode carriers"),
        (
            physical,
            {"admissible_carriers": (SemanticCarrier.COMPLEX_MODE,) * 2},
            "mode carriers",
        ),
        (physical, {"admissible_evidence": ()}, "evidence classes"),
        (physical, {"required_semantic_fields": ()}, "semantic fields"),
        (physical, {"harmonic_basis": None}, "harmonic basis"),
        (
            physical,
            {"admissible_carriers": (SemanticCarrier.NUMERICAL_PHASE,)},
            "cannot admit numerical",
        ),
        (
            numerical,
            {"admissible_carriers": (SemanticCarrier.CYCLIC_PHASE,)},
            "only numerical_phase",
        ),
        (
            numerical,
            {"admissible_evidence": (EvidenceClass.CONCEPT,)},
            "only simulation",
        ),
        (physical, {"authority": "action"}, "review-only"),
        (physical, {"actionable": True}, "review-only"),
    )
    for definition, changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(definition, **changes)  # type: ignore[arg-type]
    assert physical.to_record()["harmonic_basis"] == ("device_coordinate_mode_numbers")


def test_registry_refuses_drift_bad_keys_bad_candidates_and_bad_coverage() -> None:
    registry = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
    with pytest.raises(ValueError, match="exactly eight"):
        replace(registry, axes={})
    wrong_axes = dict(registry.axes)
    axis = wrong_axes.pop("plant_readiness")
    wrong_axes["wrong_key"] = axis
    with pytest.raises(ValueError, match="axis key"):
        replace(registry, axes=wrong_axes)
    unknown_candidate_axes = dict(registry.axes)
    unknown_candidate_axes["plant_readiness"] = replace(
        axis,
        candidate_ids=("model.nonexistent",),
    )
    with pytest.raises(ValueError, match="unknown signal candidate"):
        replace(registry, axes=unknown_candidate_axes)
    with pytest.raises(ValueError, match="mode definitions"):
        replace(registry, modes={})
    wrong_modes = dict(registry.modes)
    mode = wrong_modes.pop("physical.closed.resolved_mhd_mode")
    wrong_modes["wrong_key"] = mode
    with pytest.raises(ValueError, match="mode key"):
        replace(registry, modes=wrong_modes)
    mismatched = dict(registry.modes)
    mismatched[mode.mode_id] = replace(
        mode,
        candidate_id="open.resolved_interchange_mode",
    )
    with pytest.raises(ValueError, match="candidate applicability"):
        replace(registry, modes=mismatched)
    wrong_carriers = dict(registry.modes)
    wrong_carriers[mode.mode_id] = replace(
        mode,
        admissible_carriers=(SemanticCarrier.COMPLEX_MODE,),
    )
    with pytest.raises(ValueError, match="must equal"):
        replace(registry, modes=wrong_carriers)
    wrong_domain = dict(registry.modes)
    wrong_domain[mode.mode_id] = replace(
        mode,
        domain=ModeDomain.NUMERICAL,
        admissible_carriers=(SemanticCarrier.NUMERICAL_PHASE,),
        admissible_evidence=(EvidenceClass.SIMULATION,),
        harmonic_basis=None,
    )
    with pytest.raises(ValueError, match="mode carriers must equal"):
        replace(registry, modes=wrong_domain)
    no_numerical = {
        key: item
        for key, item in registry.modes.items()
        if item.domain is not ModeDomain.NUMERICAL
    }
    with pytest.raises(ValueError, match="all-configuration coverage"):
        replace(registry, modes=no_numerical)
    with pytest.raises(ValueError, match="exact SPO reactor"):
        replace(registry, reactor_registry_digest="0" * 64)
    with pytest.raises(ValueError, match="exact observability"):
        replace(registry, observability_registry_digest="0" * 64)


def test_registry_rejects_observability_class_domain_conflict() -> None:
    registry = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
    modes = dict(registry.modes)
    original = modes["physical.closed.resolved_mhd_mode"]
    candidate = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(
        "closed.equilibrium_profiles"
    )
    modes[original.mode_id] = ReactorModeDefinition(
        mode_id=original.mode_id,
        meaning=original.meaning,
        domain=ModeDomain.PHYSICAL,
        candidate_id=candidate.candidate_id,
        configurations=candidate.configurations,
        admissible_carriers=candidate.admissible_carriers,
        admissible_evidence=original.admissible_evidence,
        harmonic_basis=original.harmonic_basis,
        required_semantic_fields=original.required_semantic_fields,
    )
    with pytest.raises(ValueError, match="observability class"):
        ReactorRegimeModeOntologyRegistry(
            version=registry.version,
            reactor_registry_version=registry.reactor_registry_version,
            reactor_registry_digest=registry.reactor_registry_digest,
            observability_registry_version=registry.observability_registry_version,
            observability_registry_digest=registry.observability_registry_digest,
            axes=registry.axes,
            modes=modes,
        )
