# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic reference portfolio

"""Non-actuating U0 reference records spanning nine reactor families."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .contracts import (
    ObservableDescriptor,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeAxis,
    RegimeEstimate,
)
from .evidence import (
    CalibrationReference,
    ClockReference,
    ProvenanceRecord,
    QualityAssessment,
    Uncertainty,
    ValidityWindow,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .vocabulary import (
    ClockKind,
    ConversionKind,
    DriverKind,
    EvidenceClass,
    OperatingCadence,
    QualityState,
    ReactionKind,
    RegimeState,
    SemanticCarrier,
    ValidityState,
)


@dataclass(frozen=True, slots=True)
class ReactorReferenceSlice:
    """One multi-contract semantic fixture for a reactor family."""

    slice_id: str
    family: str
    context: ReactorContext
    observable: ObservableDescriptor
    semantics: tuple[PhaseSemanticRecord, ...]
    regime: RegimeEstimate


@dataclass(frozen=True, slots=True)
class _SliceDefinition:
    """Compact definition used to assemble a reference slice."""

    slice_id: str
    family: str
    configuration: str
    drivers: tuple[DriverKind, ...]
    cadence: OperatingCadence
    reaction: ReactionKind
    conversion: ConversionKind
    carriers: tuple[SemanticCarrier, ...]


_DEFINITIONS = (
    _SliceDefinition(
        "A1",
        "axisymmetric_toroidal_magnetic",
        "conventional_tokamak",
        (DriverKind.EXTERNAL_MAGNETIC_COILS, DriverKind.PLASMA_CURRENT),
        OperatingCadence.LONG_PULSE,
        ReactionKind.DEUTERIUM_TRITIUM,
        ConversionKind.THERMAL_BLANKET,
        (SemanticCarrier.COMPLEX_MODE, SemanticCarrier.CYCLIC_PHASE),
    ),
    _SliceDefinition(
        "N1",
        "non_axisymmetric_toroidal_magnetic",
        "stellarator",
        (
            DriverKind.EXTERNAL_MAGNETIC_COILS,
            DriverKind.RADIOFREQUENCY_OR_MICROWAVE,
        ),
        OperatingCadence.QUASI_STEADY,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.EXPERIMENTAL_NO_POWER_CONVERSION,
        (SemanticCarrier.FIELD_PHASE,),
    ),
    _SliceDefinition(
        "C1",
        "compact_toroid",
        "field_reversed_configuration",
        (DriverKind.EXTERNAL_MAGNETIC_COILS, DriverKind.NEUTRAL_BEAM),
        OperatingCadence.PULSED_SHOT,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.EXPERIMENTAL_NO_POWER_CONVERSION,
        (
            SemanticCarrier.COMPLEX_MODE,
            SemanticCarrier.PROTOCOL_PHASE,
            SemanticCarrier.NUMERICAL_PHASE,
        ),
    ),
    _SliceDefinition(
        "O1",
        "open_magnetic",
        "tandem_mirror",
        (DriverKind.EXTERNAL_MAGNETIC_COILS, DriverKind.NEUTRAL_BEAM),
        OperatingCadence.QUASI_STEADY,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.DIRECT_CONVERSION,
        (SemanticCarrier.BOUNDED_FEATURE, SemanticCarrier.CATEGORICAL_STATE),
    ),
    _SliceDefinition(
        "P1",
        "self_pinched_pulsed_magnetic",
        "z_pinch",
        (DriverKind.PULSED_POWER,),
        OperatingCadence.PULSED_SHOT,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.EXPERIMENTAL_NO_POWER_CONVERSION,
        (SemanticCarrier.EVENT_CYCLE, SemanticCarrier.COMPLEX_MODE),
    ),
    _SliceDefinition(
        "I1",
        "inertial",
        "laser_icf_indirect_drive",
        (DriverKind.LASER,),
        OperatingCadence.REPETITIVE_TARGET,
        ReactionKind.DEUTERIUM_TRITIUM,
        ConversionKind.THERMAL_BLANKET,
        (SemanticCarrier.EVENT_CYCLE, SemanticCarrier.PROTOCOL_PHASE),
    ),
    _SliceDefinition(
        "H1",
        "magneto_inertial",
        "maglif",
        (
            DriverKind.PULSED_POWER,
            DriverKind.LASER,
            DriverKind.SOLID_OR_LIQUID_LINER,
        ),
        OperatingCadence.PULSED_SHOT,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.EXPERIMENTAL_NO_POWER_CONVERSION,
        (
            SemanticCarrier.COMPLEX_MODE,
            SemanticCarrier.EVENT_CYCLE,
            SemanticCarrier.PROTOCOL_PHASE,
        ),
    ),
    _SliceDefinition(
        "E1",
        "electrostatic_or_beam_target",
        "gridded_iec",
        (DriverKind.ELECTROSTATIC_POTENTIAL,),
        OperatingCadence.QUASI_STEADY,
        ReactionKind.DEUTERIUM_DEUTERIUM,
        ConversionKind.EXPERIMENTAL_NO_POWER_CONVERSION,
        (SemanticCarrier.BOUNDED_FEATURE, SemanticCarrier.CATEGORICAL_STATE),
    ),
    _SliceDefinition(
        "X1",
        "nuclear_or_energy_hybrid",
        "fusion_fission_hybrid",
        (DriverKind.COMBINED,),
        OperatingCadence.LONG_PULSE,
        ReactionKind.DEUTERIUM_TRITIUM,
        ConversionKind.SUBCRITICAL_FISSION_BLANKET,
        (SemanticCarrier.CATEGORICAL_STATE, SemanticCarrier.PROTOCOL_PHASE),
    ),
)


def build_reactor_reference_portfolio() -> tuple[ReactorReferenceSlice, ...]:
    """Return deterministic non-actuating records for all U0 family slices.

    These records verify semantic breadth. They are scaffold evidence and make
    no experimental-validation or reactor-readiness claim.

    Returns
    -------
    tuple[ReactorReferenceSlice, ...]
        One deterministic review-only reference record per defined family slice.
    """
    return tuple(_build_slice(definition) for definition in _DEFINITIONS)


def _build_slice(definition: _SliceDefinition) -> ReactorReferenceSlice:
    """Build one reference slice from its compact definition."""
    configuration = DEFAULT_REACTOR_REGISTRY.resolve(definition.configuration)
    context = ReactorContext(
        context_id=f"u0.{definition.slice_id.lower()}.context",
        configuration=configuration.identifier,
        confinement_family=configuration.confinement_family,
        topology=configuration.topology,
        coordinate_frame=f"u0.{definition.slice_id.lower()}.frame",
        drivers=definition.drivers,
        cadence=definition.cadence,
        reaction=definition.reaction,
        conversion=definition.conversion,
        facility=f"u0.{definition.slice_id.lower()}.reference",
        event_id=(
            f"integrator.{definition.slice_id.lower()}.event"
            if definition.cadence
            in {
                OperatingCadence.PULSED_SHOT,
                OperatingCadence.REPETITIVE_TARGET,
                OperatingCadence.SINGLE_EXPERIMENT,
            }
            else None
        ),
        configuration_version="1.0.0",
        operating_point={
            "purpose": "semantic_contract_fixture",
            "production_actuation": False,
        },
        evidence_class=EvidenceClass.SCAFFOLD,
        registry_version=DEFAULT_REACTOR_REGISTRY.version,
        registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
    ).validate_registry()
    observable = _observable(definition, context)
    semantics = tuple(
        _semantic(definition, context, observable, carrier, index)
        for index, carrier in enumerate(definition.carriers)
    )
    regime = _regime(definition, context, observable)
    return ReactorReferenceSlice(
        slice_id=definition.slice_id,
        family=definition.family,
        context=context,
        observable=observable,
        semantics=semantics,
        regime=regime,
    )


def _observable(
    definition: _SliceDefinition,
    context: ReactorContext,
) -> ObservableDescriptor:
    """Build one calibrated reference observable."""
    stem = definition.slice_id.lower()
    return ObservableDescriptor(
        observable_id=f"u0.{stem}.observable",
        reactor_context=context,
        physical_quantity="reference_signal",
        units="1",
        coordinate_frame=context.coordinate_frame,
        spatial_support="declared_reference_support",
        diagnostic="u0_reference_diagnostic",
        channel=f"{definition.slice_id}_channel",
        value=0.25,
        clock=ClockReference(
            domain=f"u0.{stem}.clock",
            kind=ClockKind.SIMULATION_MONOTONIC,
            epoch=f"u0.{stem}.fixture_start",
            timestamp_ns=1_000_000,
            sample_rate_hz=1_000.0,
            latency_s=0.0001,
        ),
        calibration=CalibrationReference(
            calibration_id=f"u0.{stem}.calibration",
            transfer_function_id=f"u0.{stem}.transfer",
            calibrated_at_ns=900_000,
        ),
        uncertainty=Uncertainty(
            standard_deviation=0.01,
            confidence_level=0.95,
            lower_bound=0.20,
            upper_bound=0.30,
        ),
        quality=QualityAssessment(QualityState.VALID),
        validity=ValidityWindow(
            ValidityState.VALID,
            valid_from_ns=900_000,
            valid_until_ns=1_100_000,
        ),
        provenance=ProvenanceRecord(
            source_project="SCPN-PHASE-ORCHESTRATOR",
            component="reactor_semantics.reference_portfolio",
            symbol=definition.slice_id,
            artifact_uri=f"u0://reference/{definition.slice_id}",
            sha256=hashlib.sha256(definition.slice_id.encode("ascii")).hexdigest(),
            attributes=(
                ("authority", "scaffold"),
                ("runtime", "deterministic_python_fixture"),
            ),
        ),
    )


def _semantic(
    definition: _SliceDefinition,
    context: ReactorContext,
    observable: ObservableDescriptor,
    carrier: SemanticCarrier,
    index: int,
) -> PhaseSemanticRecord:
    """Build one carrier-specific semantic record."""
    is_phase = carrier in {
        SemanticCarrier.CYCLIC_PHASE,
        SemanticCarrier.COMPLEX_MODE,
        SemanticCarrier.FIELD_PHASE,
        SemanticCarrier.EVENT_CYCLE,
        SemanticCarrier.NUMERICAL_PHASE,
    }
    phase_rad = 0.25 + 0.1 * index if is_phase else None
    is_complex = carrier is SemanticCarrier.COMPLEX_MODE
    is_event = carrier is SemanticCarrier.EVENT_CYCLE
    is_numerical = carrier is SemanticCarrier.NUMERICAL_PHASE
    return PhaseSemanticRecord(
        phase_id=f"u0.{definition.slice_id.lower()}.{carrier.value}",
        reactor_context_id=context.context_id,
        observable_ids=(observable.observable_id,),
        carrier_type=carrier,
        phenomenon=f"{definition.family}_{carrier.value}",
        phase_rad=phase_rad,
        amplitude=0.5 if is_complex else None,
        frequency_hz=1_000.0 if is_phase else None,
        bandwidth_hz=10.0 if is_phase else None,
        mode_identity=f"{definition.slice_id}_mode" if is_complex else None,
        mode_harmonic=(2, 1) if is_complex else None,
        phase_origin="declared_reference_zero" if is_phase else None,
        orientation="positive" if is_phase else None,
        reference_frame=context.coordinate_frame,
        clock_domain=observable.clock.domain,
        clock_kind=observable.clock.kind,
        clock_epoch=observable.clock.epoch,
        wrap_convention="zero_to_two_pi" if is_phase else None,
        reference_signal=observable.observable_id if is_phase or is_event else None,
        extractor=f"u0.{carrier.value}.extractor",
        extractor_version="1.0.0",
        observation_operator=(
            None if is_numerical else f"u0.{carrier.value}.observation_operator"
        ),
        uncertainty=Uncertainty(
            standard_deviation=0.02,
            confidence_level=0.95,
            circular_std_rad=0.02 if is_phase else None,
        ),
        confidence=0.95,
        observability=0.95,
        observability_threshold=0.05,
        validity=observable.validity,
        quality=observable.quality,
        evidence_class=EvidenceClass.SCAFFOLD,
    )


def _regime(
    definition: _SliceDefinition,
    context: ReactorContext,
    observable: ObservableDescriptor,
) -> RegimeEstimate:
    """Build one family-qualified review-only regime estimate."""
    stem = definition.slice_id.lower()
    return RegimeEstimate(
        regime_id=f"u0.{stem}.regime",
        reactor_context_id=context.context_id,
        axes=(
            RegimeAxis("lifecycle", "reference", 0.95),
            RegimeAxis("estimator", "valid", 0.95),
            RegimeAxis("family_physics", f"{stem}_reference", 0.90),
        ),
        state=RegimeState.NOMINAL,
        evidence_ids=(observable.observable_id,),
        classifier=f"u0.{stem}.classifier",
        classifier_version="1.0.0",
        threshold_provenance=(f"u0.{stem}.thresholds",),
        confidence=0.90,
        hysteresis=0.05,
        dwell_time_s=0.001,
        transition_reason="reference fixture initialization",
        safety_effect="review only; no production actuation",
        validity=observable.validity,
    )


__all__ = [
    "ReactorReferenceSlice",
    "build_reactor_reference_portfolio",
]
