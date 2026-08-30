# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-reactor regime and mode ontology

"""Closed meanings for reactor regime axes and physical or numerical modes.

The ontology validates names and evidence bindings.  It does not infer a
regime, extract a mode, admit evidence for control, or actuate a device.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .contracts import RegimeAxis
from .observability_profiles import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    ObservabilityClass,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .vocabulary import (
    EvidenceClass,
    SemanticCarrier,
    probability,
    require_enum,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
)

REACTOR_REGIME_MODE_ONTOLOGY_VERSION = "1.0.0"


class AxisApplicability(StrEnum):
    """Whether one regime axis has meaning in a particular context."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class AxisApplicabilityPolicy(StrEnum):
    """How static reactor context constrains an axis."""

    UNIVERSAL = "universal"
    CONTEXT_DEPENDENT = "context_dependent"


class ModeDomain(StrEnum):
    """Epistemically distinct physical and model-owned mode domains."""

    PHYSICAL = "physical"
    NUMERICAL = "numerical"


@dataclass(frozen=True, slots=True)
class ReactorRegimeAxisDefinition:
    """Versioned meaning and closed labels for one compositional regime axis."""

    axis_id: str
    meaning: str
    labels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    applicability_policy: AxisApplicabilityPolicy
    required_evidence: tuple[str, ...]
    authority: str = "review_only"
    actionable: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.axis_id, field="axis_id")
        require_text(self.meaning, field="axis meaning")
        labels = tuple(
            require_identifier(label, field="axis label") for label in self.labels
        )
        if not labels or tuple(sorted(set(labels))) != labels:
            raise ValueError("axis labels must be non-empty, unique, and sorted")
        if "unknown" not in labels:
            raise ValueError("axis labels must include unknown")
        candidates = tuple(
            require_identifier(item, field="candidate_id")
            for item in self.candidate_ids
        )
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("axis candidate identifiers must be unique and sorted")
        requirements = tuple(
            require_identifier(item, field="required_evidence")
            for item in self.required_evidence
        )
        if not requirements or len(set(requirements)) != len(requirements):
            raise ValueError("axis evidence requirements must be non-empty and unique")
        require_enum(
            self.applicability_policy,
            AxisApplicabilityPolicy,
            field="applicability_policy",
        )
        if self.authority != "review_only" or self.actionable is not False:
            raise ValueError("regime axes must remain review-only")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "required_evidence", requirements)

    def to_record(self) -> dict[str, object]:
        """Return a deterministic definition record."""
        return {
            "actionable": self.actionable,
            "applicability_policy": self.applicability_policy.value,
            "authority": self.authority,
            "axis_id": self.axis_id,
            "candidate_ids": list(self.candidate_ids),
            "labels": list(self.labels),
            "meaning": self.meaning,
            "required_evidence": list(self.required_evidence),
        }


@dataclass(frozen=True, slots=True)
class ReactorRegimeAxisAssignment:
    """Fail-closed runtime assignment against one axis definition."""

    definition: ReactorRegimeAxisDefinition
    applicability: AxisApplicability
    label: str | None
    confidence: float
    evidence_ids: tuple[str, ...]
    applicability_basis: tuple[str, ...]
    authority: str = "review_only"
    actionable: bool = False

    def __post_init__(self) -> None:
        applicability = require_enum(
            self.applicability,
            AxisApplicability,
            field="axis applicability",
        )
        confidence = probability(self.confidence, field="axis confidence")
        evidence = tuple(
            require_identifier(item, field="evidence_id") for item in self.evidence_ids
        )
        basis = tuple(
            require_identifier(item, field="applicability_basis")
            for item in self.applicability_basis
        )
        if len(set(evidence)) != len(evidence) or len(set(basis)) != len(basis):
            raise ValueError("assignment evidence and basis identifiers must be unique")
        if self.authority != "review_only" or self.actionable is not False:
            raise ValueError("regime assignments must remain review-only")
        if applicability is AxisApplicability.APPLICABLE:
            if self.label is None or self.label == "unknown":
                raise ValueError("applicable axis requires a classified label")
            label = require_identifier(self.label, field="axis label")
            if label not in self.definition.labels:
                raise ValueError("axis label is not defined by the ontology")
            if not evidence:
                raise ValueError("applicable axis requires classification evidence")
        else:
            if self.label is not None:
                raise ValueError("non-classified axis forbids a physics label")
            if confidence != 0.0:
                raise ValueError("non-classified axis confidence must be zero")
            if applicability is AxisApplicability.NOT_APPLICABLE and not basis:
                raise ValueError("not-applicable axis requires an applicability basis")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "applicability_basis", basis)

    def to_regime_axis(self) -> RegimeAxis:
        """Project into U0 without collapsing absence into a nominal label."""
        label = self.label if self.label is not None else self.applicability.value
        return RegimeAxis(
            name=self.definition.axis_id,
            label=label,
            confidence=self.confidence,
        )

    def to_record(self) -> dict[str, object]:
        """Return a deterministic assignment record."""
        return {
            "actionable": self.actionable,
            "applicability": self.applicability.value,
            "applicability_basis": list(self.applicability_basis),
            "authority": self.authority,
            "axis_id": self.definition.axis_id,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class ReactorModeDefinition:
    """Namespaced physical or numerical mode-family definition."""

    mode_id: str
    meaning: str
    domain: ModeDomain
    candidate_id: str
    configurations: tuple[str, ...]
    admissible_carriers: tuple[SemanticCarrier, ...]
    admissible_evidence: tuple[EvidenceClass, ...]
    harmonic_basis: str | None
    required_semantic_fields: tuple[str, ...]
    authority: str = "review_only"
    actionable: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.mode_id, field="mode_id")
        require_text(self.meaning, field="mode meaning")
        domain = require_enum(self.domain, ModeDomain, field="mode domain")
        require_identifier(self.candidate_id, field="candidate_id")
        configurations = tuple(
            require_identifier(item, field="configuration")
            for item in self.configurations
        )
        if not configurations or tuple(sorted(set(configurations))) != configurations:
            raise ValueError(
                "mode configurations must be non-empty, unique, and sorted"
            )
        for configuration in configurations:
            if configuration not in DEFAULT_REACTOR_REGISTRY.configurations:
                raise ValueError(f"unknown reactor configuration: {configuration}")
        carriers = tuple(
            require_enum(item, SemanticCarrier, field="admissible_carrier")
            for item in self.admissible_carriers
        )
        evidence = tuple(
            require_enum(item, EvidenceClass, field="admissible_evidence")
            for item in self.admissible_evidence
        )
        if not carriers or len(set(carriers)) != len(carriers):
            raise ValueError("mode carriers must be non-empty and unique")
        if not evidence or len(set(evidence)) != len(evidence):
            raise ValueError("mode evidence classes must be non-empty and unique")
        fields = tuple(
            require_identifier(item, field="required_semantic_field")
            for item in self.required_semantic_fields
        )
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("required semantic fields must be non-empty and unique")
        if domain is ModeDomain.PHYSICAL:
            if self.harmonic_basis is None:
                raise ValueError("physical mode requires a harmonic basis")
            require_text(self.harmonic_basis, field="harmonic_basis")
            if SemanticCarrier.NUMERICAL_PHASE in carriers:
                raise ValueError("physical mode cannot admit numerical phase")
        else:
            if carriers != (SemanticCarrier.NUMERICAL_PHASE,):
                raise ValueError("numerical mode admits only numerical_phase")
            if evidence != (EvidenceClass.SIMULATION,):
                raise ValueError("numerical mode admits only simulation evidence")
        if self.authority != "review_only" or self.actionable is not False:
            raise ValueError("mode definitions must remain review-only")
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "admissible_carriers", carriers)
        object.__setattr__(self, "admissible_evidence", evidence)
        object.__setattr__(self, "required_semantic_fields", fields)

    def to_record(self) -> dict[str, object]:
        """Return a deterministic mode-family definition."""
        return {
            "actionable": self.actionable,
            "admissible_carriers": [item.value for item in self.admissible_carriers],
            "admissible_evidence": [item.value for item in self.admissible_evidence],
            "authority": self.authority,
            "candidate_id": self.candidate_id,
            "configurations": list(self.configurations),
            "domain": self.domain.value,
            "harmonic_basis": self.harmonic_basis,
            "meaning": self.meaning,
            "mode_id": self.mode_id,
            "required_semantic_fields": list(self.required_semantic_fields),
        }


@dataclass(frozen=True, slots=True)
class ReactorModeBinding:
    """Evidence-bearing identity for one extracted physical or numerical mode."""

    definition: ReactorModeDefinition
    configuration: str
    carrier: SemanticCarrier
    evidence_class: EvidenceClass
    mode_identity: str
    harmonic_coordinates: tuple[int, int] | None
    observation_operator_id: str | None
    reference_frame: str | None
    reference_signal_id: str | None
    orientation: str | None
    phase_origin: str | None
    wrap_convention: str | None
    observability_threshold: float | None
    validity_id: str
    quality_id: str
    provenance_id: str
    authority: str = "review_only"
    actionable: bool = False

    def __post_init__(self) -> None:
        configuration = DEFAULT_REACTOR_REGISTRY.resolve(self.configuration).identifier
        if configuration not in self.definition.configurations:
            raise ValueError("mode is not defined for this reactor configuration")
        carrier = require_enum(self.carrier, SemanticCarrier, field="mode carrier")
        evidence = require_enum(
            self.evidence_class,
            EvidenceClass,
            field="mode evidence_class",
        )
        if carrier not in self.definition.admissible_carriers:
            raise ValueError("mode carrier is not admitted by its definition")
        if evidence not in self.definition.admissible_evidence:
            raise ValueError("mode evidence is not admitted by its definition")
        require_identifier(self.mode_identity, field="mode_identity")
        for field_name in ("validity_id", "quality_id", "provenance_id"):
            require_identifier(getattr(self, field_name), field=field_name)
        if self.authority != "review_only" or self.actionable is not False:
            raise ValueError("mode bindings must remain review-only")
        if self.definition.domain is ModeDomain.PHYSICAL:
            required = {
                "observation_operator_id": self.observation_operator_id,
                "reference_frame": self.reference_frame,
                "reference_signal_id": self.reference_signal_id,
                "orientation": self.orientation,
                "phase_origin": self.phase_origin,
                "wrap_convention": self.wrap_convention,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"physical mode binding missing fields: {missing}")
            for name, value in required.items():
                require_identifier(value, field=name)
            if self.harmonic_coordinates is None:
                raise ValueError("physical mode binding requires harmonic coordinates")
            first, second = self.harmonic_coordinates
            if (
                isinstance(first, bool)
                or isinstance(second, bool)
                or not isinstance(first, int)
                or not isinstance(second, int)
                or first <= 0
                or second <= 0
            ):
                raise ValueError("mode harmonic coordinates must be positive integers")
            if self.observability_threshold is None:
                raise ValueError(
                    "physical mode binding requires observability threshold"
                )
            probability(
                self.observability_threshold,
                field="observability_threshold",
            )
        else:
            if self.harmonic_coordinates is not None:
                raise ValueError(
                    "numerical oscillator coordinate has no physical harmonic"
                )
        object.__setattr__(self, "configuration", configuration)

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic mode binding."""
        harmonic = (
            list(self.harmonic_coordinates)
            if self.harmonic_coordinates is not None
            else None
        )
        return {
            "actionable": self.actionable,
            "authority": self.authority,
            "candidate_id": self.definition.candidate_id,
            "carrier": self.carrier.value,
            "configuration": self.configuration,
            "domain": self.definition.domain.value,
            "evidence_class": self.evidence_class.value,
            "harmonic_coordinates": harmonic,
            "mode_id": self.definition.mode_id,
            "mode_identity": self.mode_identity,
            "observability_threshold": self.observability_threshold,
            "observation_operator_id": self.observation_operator_id,
            "orientation": self.orientation,
            "phase_origin": self.phase_origin,
            "provenance_id": self.provenance_id,
            "quality_id": self.quality_id,
            "reference_frame": self.reference_frame,
            "reference_signal_id": self.reference_signal_id,
            "validity_id": self.validity_id,
            "wrap_convention": self.wrap_convention,
        }


@dataclass(frozen=True, slots=True)
class ReactorRegimeModeOntologyRegistry:
    """Immutable ontology bound to exact reactor and observability registries."""

    version: str
    reactor_registry_version: str
    reactor_registry_digest: str
    observability_registry_version: str
    observability_registry_digest: str
    axes: Mapping[str, ReactorRegimeAxisDefinition]
    modes: Mapping[str, ReactorModeDefinition]

    def __post_init__(self) -> None:
        version = require_semver(self.version, field="ontology version")
        reactor_version = require_semver(
            self.reactor_registry_version,
            field="reactor registry version",
        )
        reactor_digest = require_sha256(
            self.reactor_registry_digest,
            field="reactor registry digest",
        )
        observability_version = require_semver(
            self.observability_registry_version,
            field="observability registry version",
        )
        observability_digest = require_sha256(
            self.observability_registry_digest,
            field="observability registry digest",
        )
        if (
            reactor_version != DEFAULT_REACTOR_REGISTRY.version
            or reactor_digest != DEFAULT_REACTOR_REGISTRY.digest
        ):
            raise ValueError("ontology must bind the exact SPO reactor registry")
        observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        if (
            observability_version != observability.version
            or observability_digest != observability.digest
        ):
            raise ValueError("ontology must bind the exact observability registry")
        axes = dict(self.axes)
        modes = dict(self.modes)
        if len(axes) != 8:
            raise ValueError("ontology requires exactly eight regime axes")
        for axis_id, axis in axes.items():
            if axis_id != axis.axis_id:
                raise ValueError("axis key must equal axis_id")
            self._validate_candidates(axis.candidate_ids)
        if not modes:
            raise ValueError("ontology requires mode definitions")
        for mode_id, mode in modes.items():
            if mode_id != mode.mode_id:
                raise ValueError("mode key must equal mode_id")
            candidate = observability.resolve(mode.candidate_id)
            if not set(mode.configurations).issubset(candidate.configurations):
                raise ValueError("mode configurations exceed candidate applicability")
            if set(mode.admissible_carriers) != set(candidate.admissible_carriers):
                raise ValueError("mode carriers must equal observability candidates")
            expected_class = (
                ObservabilityClass.NUMERICAL_ONLY
                if mode.domain is ModeDomain.NUMERICAL
                else ObservabilityClass.DERIVED_CYCLIC
            )
            if candidate.observability_class is not expected_class:
                raise ValueError("mode domain conflicts with observability class")
        covered = {
            configuration
            for mode in modes.values()
            for configuration in mode.configurations
        }
        if covered != set(DEFAULT_REACTOR_REGISTRY.configurations):
            raise ValueError("mode definitions require all-configuration coverage")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "reactor_registry_version", reactor_version)
        object.__setattr__(self, "reactor_registry_digest", reactor_digest)
        object.__setattr__(
            self, "observability_registry_version", observability_version
        )
        object.__setattr__(self, "observability_registry_digest", observability_digest)
        object.__setattr__(self, "axes", MappingProxyType(axes))
        object.__setattr__(self, "modes", MappingProxyType(modes))

    @staticmethod
    def _validate_candidates(candidate_ids: tuple[str, ...]) -> None:
        observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        for candidate_id in candidate_ids:
            observability.resolve(candidate_id)

    def resolve_axis(self, axis_id: str) -> ReactorRegimeAxisDefinition:
        """Resolve one exact cross-family axis identifier."""
        key = require_identifier(axis_id, field="axis_id")
        try:
            return self.axes[key]
        except KeyError as exc:
            raise ValueError(f"unknown regime axis: {key}") from exc

    def resolve_mode(self, mode_id: str) -> ReactorModeDefinition:
        """Resolve one exact namespaced mode-family identifier."""
        key = require_identifier(mode_id, field="mode_id")
        try:
            return self.modes[key]
        except KeyError as exc:
            raise ValueError(f"unknown mode definition: {key}") from exc

    def modes_for_configuration(
        self, configuration: str
    ) -> tuple[ReactorModeDefinition, ...]:
        """Return physical definitions plus the explicit numerical fallback."""
        canonical = DEFAULT_REACTOR_REGISTRY.resolve(configuration).identifier
        return tuple(
            mode for mode in self.modes.values() if canonical in mode.configurations
        )

    def applicability_for(
        self,
        axis_id: str,
        configuration: str,
    ) -> AxisApplicability:
        """Return static semantic applicability, never a measured regime state."""
        axis = self.resolve_axis(axis_id)
        canonical = DEFAULT_REACTOR_REGISTRY.resolve(configuration).identifier
        if axis.applicability_policy is AxisApplicabilityPolicy.UNIVERSAL:
            return AxisApplicability.APPLICABLE
        observability = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
        if any(
            canonical in observability.resolve(candidate_id).configurations
            for candidate_id in axis.candidate_ids
        ):
            return AxisApplicability.APPLICABLE
        return AxisApplicability.NOT_APPLICABLE

    def to_record(self) -> dict[str, object]:
        """Return the canonical JSON-compatible ontology record."""
        return {
            "actionable": False,
            "authority": "review_only",
            "axes": [self.axes[key].to_record() for key in sorted(self.axes)],
            "modes": [self.modes[key].to_record() for key in sorted(self.modes)],
            "observability_registry_digest": self.observability_registry_digest,
            "observability_registry_version": self.observability_registry_version,
            "reactor_registry_digest": self.reactor_registry_digest,
            "reactor_registry_version": self.reactor_registry_version,
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        """Return SHA-256 of the canonical ontology record."""
        payload = json.dumps(
            self.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _axis(
    axis_id: str,
    meaning: str,
    labels: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    policy: AxisApplicabilityPolicy,
    required_evidence: tuple[str, ...],
) -> ReactorRegimeAxisDefinition:
    return ReactorRegimeAxisDefinition(
        axis_id=axis_id,
        meaning=meaning,
        labels=tuple(sorted(labels)),
        candidate_ids=tuple(sorted(candidate_ids)),
        applicability_policy=policy,
        required_evidence=required_evidence,
    )


_PHYSICAL_FIELDS = (
    "harmonic_coordinates",
    "mode_identity",
    "observability_threshold",
    "observation_operator_id",
    "orientation",
    "phase_origin",
    "provenance_id",
    "quality_id",
    "reference_frame",
    "reference_signal_id",
    "validity_id",
    "wrap_convention",
)
_PHYSICAL_EVIDENCE = (
    EvidenceClass.OBSERVED,
    EvidenceClass.EXPERIMENTAL,
    EvidenceClass.SIMULATION,
)


def _mode(
    mode_id: str,
    meaning: str,
    candidate_id: str,
    harmonic_basis: str,
) -> ReactorModeDefinition:
    candidate = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(candidate_id)
    return ReactorModeDefinition(
        mode_id=mode_id,
        meaning=meaning,
        domain=ModeDomain.PHYSICAL,
        candidate_id=candidate_id,
        configurations=candidate.configurations,
        admissible_carriers=candidate.admissible_carriers,
        admissible_evidence=_PHYSICAL_EVIDENCE,
        harmonic_basis=harmonic_basis,
        required_semantic_fields=_PHYSICAL_FIELDS,
    )


_AXES = (
    _axis(
        "plant_readiness",
        "device or experiment readiness declared by its plant-truth owner",
        ("unknown", "unavailable", "commissioning", "experimental", "operational"),
        (),
        AxisApplicabilityPolicy.UNIVERSAL,
        ("owner_declaration", "validity", "provenance"),
    ),
    _axis(
        "diagnostic_observability",
        "ability of declared diagnostics and operators to support interpretation",
        ("unknown", "absent", "partial", "qualified"),
        tuple(DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.candidates),
        AxisApplicabilityPolicy.UNIVERSAL,
        ("diagnostic_inventory", "quality", "validity", "provenance"),
    ),
    _axis(
        "confinement_or_assembly",
        "formation, confinement, assembly, compression, or loss of the fusion system",
        ("unknown", "forming", "established", "degrading", "lost"),
        (
            "beam.target_outcome",
            "closed.equilibrium_profiles",
            "hybrid.source_blanket_response",
            "iec.steady_state",
            "inertial.implosion_trajectory",
            "magneto_inertial.translation_and_compression",
            "open.equilibrium_and_loss",
            "self_magnetic.drive_waveform",
        ),
        AxisApplicabilityPolicy.UNIVERSAL,
        ("classifier", "threshold_provenance", "validity", "evidence_ids"),
    ),
    _axis(
        "stability_or_symmetry",
        "resolved stability, symmetry, or coherent-mode condition",
        (
            "unknown",
            "unresolved",
            "symmetric_or_quiescent",
            "coherent_mode",
            "disrupted",
        ),
        (
            "closed.resolved_mhd_mode",
            "iec.resolved_bunching",
            "inertial.resolved_asymmetry_mode",
            "magneto_inertial.resolved_asymmetry_mode",
            "open.resolved_interchange_mode",
            "self_magnetic.resolved_instability_mode",
        ),
        AxisApplicabilityPolicy.CONTEXT_DEPENDENT,
        ("observation_operator", "threshold_provenance", "quality", "validity"),
    ),
    _axis(
        "driver_synchronization",
        "timing or phase relation among declared drivers, targets, or arrivals",
        ("unknown", "asynchronous", "aligning", "synchronized", "desynchronized"),
        (
            "beam.rf_bunch_phase",
            "inertial.driver_timing",
            "magneto_inertial.driver_arrival",
            "open.drive_reference",
            "self_magnetic.drive_waveform",
        ),
        AxisApplicabilityPolicy.CONTEXT_DEPENDENT,
        ("clock_epoch", "reference_signal", "timing_uncertainty", "validity"),
    ),
    _axis(
        "power_or_burn",
        "evidence-supported fusion power, burn, yield, or source condition",
        ("unknown", "subthreshold", "rising", "sustained", "declining"),
        (
            "beam.target_outcome",
            "closed.equilibrium_profiles",
            "hybrid.source_blanket_response",
            "iec.steady_state",
            "inertial.shot_outcome",
        ),
        AxisApplicabilityPolicy.UNIVERSAL,
        ("reaction_model", "measurement_or_model", "uncertainty", "validity"),
    ),
    _axis(
        "exhaust_or_boundary",
        "exhaust, end loss, wall, blanket, or energy-conversion boundary condition",
        ("unknown", "uncontrolled", "conditioned", "regulated", "saturated"),
        (
            "closed.equilibrium_profiles",
            "hybrid.source_blanket_response",
            "open.equilibrium_and_loss",
        ),
        AxisApplicabilityPolicy.CONTEXT_DEPENDENT,
        ("boundary_definition", "measurement_or_model", "quality", "validity"),
    ),
    _axis(
        "evidence_maturity",
        "epistemic maturity of the evidence supporting the complete estimate",
        (
            "unknown",
            "scaffold",
            "concept",
            "developing",
            "simulation",
            "experimental",
            "observed",
        ),
        (),
        AxisApplicabilityPolicy.UNIVERSAL,
        ("evidence_class", "provenance", "validity"),
    ),
)

_MODES = (
    _mode(
        "physical.closed.resolved_mhd_mode",
        "resolved closed-field MHD mode",
        "closed.resolved_mhd_mode",
        "device_coordinate_mode_numbers",
    ),
    _mode(
        "physical.open.resolved_interchange_mode",
        "resolved open-field interchange or flute mode",
        "open.resolved_interchange_mode",
        "device_coordinate_azimuthal_axial_mode_numbers",
    ),
    _mode(
        "physical.self_magnetic.resolved_instability_mode",
        "resolved pinch sausage, kink, or related instability mode",
        "self_magnetic.resolved_instability_mode",
        "cylindrical_azimuthal_axial_mode_numbers",
    ),
    _mode(
        "physical.inertial.resolved_asymmetry_mode",
        "resolved implosion asymmetry mode",
        "inertial.resolved_asymmetry_mode",
        "spherical_harmonic_degree_order",
    ),
    _mode(
        "physical.magneto_inertial.resolved_asymmetry_mode",
        "resolved liner, jet, or compressed-target asymmetry mode",
        "magneto_inertial.resolved_asymmetry_mode",
        "declared_geometry_harmonic_coordinates",
    ),
    _mode(
        "physical.iec.resolved_bunching_mode",
        "resolved electrostatic density or potential bunching mode",
        "iec.resolved_bunching",
        "device_coordinate_mode_numbers",
    ),
)

_NUMERICAL_CANDIDATE = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(
    "model.synthetic_oscillator_coordinate"
)
_NUMERICAL_MODE = ReactorModeDefinition(
    mode_id="numerical.model.synthetic_oscillator_coordinate",
    meaning="model-owned oscillator coordinate without physical-mode equivalence",
    domain=ModeDomain.NUMERICAL,
    candidate_id=_NUMERICAL_CANDIDATE.candidate_id,
    configurations=_NUMERICAL_CANDIDATE.configurations,
    admissible_carriers=_NUMERICAL_CANDIDATE.admissible_carriers,
    admissible_evidence=(EvidenceClass.SIMULATION,),
    harmonic_basis=None,
    required_semantic_fields=(
        "mode_identity",
        "provenance_id",
        "quality_id",
        "validity_id",
    ),
)

DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY = ReactorRegimeModeOntologyRegistry(
    version=REACTOR_REGIME_MODE_ONTOLOGY_VERSION,
    reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
    reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
    observability_registry_version=(
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version
    ),
    observability_registry_digest=(
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
    ),
    axes={axis.axis_id: axis for axis in _AXES},
    modes={mode.mode_id: mode for mode in (*_MODES, _NUMERICAL_MODE)},
)
