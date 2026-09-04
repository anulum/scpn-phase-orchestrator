# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-reactor observability profiles

"""Machine-readable phase-meaning requirements across reactor concepts.

The catalogue records candidate phenomena and the evidence required to assign
semantic carriers. It is a design constraint, not evidence that a diagnostic,
producer, phase, or reactor capability exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .registry import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_V1_0_0,
    resolve_reactor_registry_release,
)
from .vocabulary import (
    SemanticCarrier,
    require_enum,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
)

REACTOR_OBSERVABILITY_PROFILE_REGISTRY_VERSION = "1.1.0"
REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0_VERSION = "1.0.0"


class ObservabilityClass(StrEnum):
    """Epistemic route by which a candidate could acquire meaning."""

    DIRECT_CYCLIC = "direct_cyclic"
    DERIVED_CYCLIC = "derived_cyclic"
    EVENT_RELATIVE = "event_relative"
    NONCYCLIC_FEATURE = "noncyclic_feature"
    NUMERICAL_ONLY = "numerical_only"
    UNOBSERVABLE = "unobservable"


class UnmetEvidenceDisposition(StrEnum):
    """Fail-closed result when a candidate lacks its required evidence."""

    UNOBSERVABLE_PHASE = "unobservable_phase"
    EVENT_TIMESTAMPS_OR_PROTOCOL = "event_timestamps_or_protocol"
    PRESERVE_NONCYCLIC = "preserve_noncyclic"
    RETAIN_SIMULATION_CLASS = "retain_simulation_class"
    NO_USABLE_PHASE = "no_usable_phase"


_ALLOWED_CARRIERS = {
    ObservabilityClass.DIRECT_CYCLIC: frozenset(
        {SemanticCarrier.CYCLIC_PHASE, SemanticCarrier.FIELD_PHASE}
    ),
    ObservabilityClass.DERIVED_CYCLIC: frozenset(
        {
            SemanticCarrier.COMPLEX_MODE,
            SemanticCarrier.CYCLIC_PHASE,
            SemanticCarrier.FIELD_PHASE,
        }
    ),
    ObservabilityClass.EVENT_RELATIVE: frozenset(
        {SemanticCarrier.EVENT_CYCLE, SemanticCarrier.PROTOCOL_PHASE}
    ),
    ObservabilityClass.NONCYCLIC_FEATURE: frozenset(
        {SemanticCarrier.BOUNDED_FEATURE, SemanticCarrier.CATEGORICAL_STATE}
    ),
    ObservabilityClass.NUMERICAL_ONLY: frozenset({SemanticCarrier.NUMERICAL_PHASE}),
    ObservabilityClass.UNOBSERVABLE: frozenset(),
}
_REQUIRED_DISPOSITION = {
    ObservabilityClass.DIRECT_CYCLIC: UnmetEvidenceDisposition.UNOBSERVABLE_PHASE,
    ObservabilityClass.DERIVED_CYCLIC: UnmetEvidenceDisposition.UNOBSERVABLE_PHASE,
    ObservabilityClass.EVENT_RELATIVE: (
        UnmetEvidenceDisposition.EVENT_TIMESTAMPS_OR_PROTOCOL
    ),
    ObservabilityClass.NONCYCLIC_FEATURE: (UnmetEvidenceDisposition.PRESERVE_NONCYCLIC),
    ObservabilityClass.NUMERICAL_ONLY: (
        UnmetEvidenceDisposition.RETAIN_SIMULATION_CLASS
    ),
    ObservabilityClass.UNOBSERVABLE: UnmetEvidenceDisposition.NO_USABLE_PHASE,
}


@dataclass(frozen=True, slots=True)
class ReactorSignalCandidateProfile:
    """Evidence requirements for one candidate phenomenon.

    Applicability means only that the phenomenon is meaningful to investigate
    for those configurations. It does not assert implementation, measurement,
    observability, experimental validation, or readiness.
    """

    candidate_id: str
    phenomenon: str
    configurations: tuple[str, ...]
    observability_class: ObservabilityClass
    admissible_carriers: tuple[SemanticCarrier, ...]
    required_evidence: tuple[str, ...]
    unmet_evidence: UnmetEvidenceDisposition
    reference_required: bool
    observation_operator_required: bool
    repeated_cycle_required: bool
    authority: str = "review_only"
    actionable: bool = False
    evidence_claimed: bool = False

    def __post_init__(self) -> None:
        """Validate applicability, carrier, evidence, and authority invariants."""
        require_identifier(self.candidate_id, field="candidate_id")
        require_text(self.phenomenon, field="phenomenon")
        observability_class = require_enum(
            self.observability_class,
            ObservabilityClass,
            field="observability_class",
        )
        unmet_evidence = require_enum(
            self.unmet_evidence,
            UnmetEvidenceDisposition,
            field="unmet_evidence",
        )
        if not self.configurations:
            raise ValueError("signal candidate requires reactor configurations")
        if tuple(sorted(set(self.configurations))) != self.configurations:
            raise ValueError(
                "signal candidate configurations must be unique and sorted"
            )
        for configuration in self.configurations:
            if configuration not in DEFAULT_REACTOR_REGISTRY.configurations:
                raise ValueError(f"unknown reactor configuration: {configuration}")
        carriers = tuple(
            require_enum(item, SemanticCarrier, field="admissible_carrier")
            for item in self.admissible_carriers
        )
        if len(set(carriers)) != len(carriers):
            raise ValueError("admissible carriers must be unique")
        if set(carriers) != set(_ALLOWED_CARRIERS[observability_class]):
            raise ValueError("admissible carriers do not match observability class")
        if not self.required_evidence:
            raise ValueError("signal candidate requires evidence requirements")
        for requirement in self.required_evidence:
            require_identifier(requirement, field="required_evidence")
        if len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("evidence requirements must be unique")
        if unmet_evidence is not _REQUIRED_DISPOSITION[observability_class]:
            raise ValueError("unmet-evidence disposition does not match class")
        if self.observation_operator_required is not (
            observability_class is ObservabilityClass.DERIVED_CYCLIC
        ):
            raise ValueError(
                "derived cyclic candidates require an observation operator"
            )
        if self.reference_required is not (
            observability_class
            in {
                ObservabilityClass.DIRECT_CYCLIC,
                ObservabilityClass.DERIVED_CYCLIC,
                ObservabilityClass.EVENT_RELATIVE,
            }
        ):
            raise ValueError("reference requirement does not match observability class")
        if self.repeated_cycle_required is not (
            observability_class is ObservabilityClass.EVENT_RELATIVE
        ):
            raise ValueError("only event-relative candidates use repetition gating")
        if (
            self.authority != "review_only"
            or self.actionable is not False
            or self.evidence_claimed is not False
        ):
            raise ValueError("candidate profiles are non-actuating design requirements")

    def to_record(self) -> dict[str, object]:
        """Return a complete deterministic candidate record.

        Returns
        -------
        dict[str, object]
            JSON-compatible candidate profile fields.
        """
        return {
            "actionable": self.actionable,
            "admissible_carriers": sorted(
                item.value for item in self.admissible_carriers
            ),
            "authority": self.authority,
            "candidate_id": self.candidate_id,
            "configurations": list(self.configurations),
            "evidence_claimed": self.evidence_claimed,
            "observation_operator_required": self.observation_operator_required,
            "observability_class": self.observability_class.value,
            "phenomenon": self.phenomenon,
            "reference_required": self.reference_required,
            "repeated_cycle_required": self.repeated_cycle_required,
            "required_evidence": list(self.required_evidence),
            "unmet_evidence": self.unmet_evidence.value,
        }


@dataclass(frozen=True, slots=True)
class ReactorObservabilityProfileRegistry:
    """Immutable candidate catalogue covering every built-in configuration."""

    version: str
    reactor_registry_version: str
    reactor_registry_digest: str
    candidates: Mapping[str, ReactorSignalCandidateProfile]

    def __post_init__(self) -> None:
        """Validate registry identity, coverage, and reactor-registry binding."""
        version = require_semver(self.version, field="observability registry version")
        reactor_version = require_semver(
            self.reactor_registry_version,
            field="reactor registry version",
        )
        reactor_digest = require_sha256(
            self.reactor_registry_digest,
            field="reactor registry digest",
        )
        candidates = dict(self.candidates)
        if not candidates:
            raise ValueError("observability registry requires candidates")
        for candidate_id, candidate in candidates.items():
            if candidate_id != candidate.candidate_id:
                raise ValueError("candidate key must equal candidate_id")
        try:
            reactor_registry = resolve_reactor_registry_release(
                reactor_version,
                reactor_digest,
            )
        except ValueError as exc:
            raise ValueError(
                "observability profiles require a recognised exact reactor registry"
            ) from exc
        covered = {
            configuration
            for candidate in candidates.values()
            for configuration in candidate.configurations
        }
        expected = set(reactor_registry.configurations)
        if covered != expected:
            missing = sorted(expected - covered)
            extra = sorted(covered - expected)
            raise ValueError(
                "observability configuration coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "reactor_registry_version", reactor_version)
        object.__setattr__(self, "reactor_registry_digest", reactor_digest)
        object.__setattr__(self, "candidates", MappingProxyType(candidates))

    def for_configuration(
        self, configuration: str
    ) -> tuple[ReactorSignalCandidateProfile, ...]:
        """Return sorted candidates applicable to a configuration or alias.

        Parameters
        ----------
        configuration : str
            Canonical reactor configuration identifier or registered alias.

        Returns
        -------
        tuple[ReactorSignalCandidateProfile, ...]
            Applicable candidates ordered by candidate identifier.
        """
        reactor_registry = resolve_reactor_registry_release(
            self.reactor_registry_version,
            self.reactor_registry_digest,
        )
        canonical = reactor_registry.resolve(configuration).identifier
        return tuple(
            self.candidates[key]
            for key in sorted(self.candidates)
            if canonical in self.candidates[key].configurations
        )

    def resolve(self, candidate_id: str) -> ReactorSignalCandidateProfile:
        """Resolve one exact candidate identifier.

        Parameters
        ----------
        candidate_id : str
            Exact observability candidate identifier.

        Returns
        -------
        ReactorSignalCandidateProfile
            Registered candidate profile.

        Raises
        ------
        ValueError
            If the identifier is invalid or is not registered.
        """
        key = require_identifier(candidate_id, field="candidate_id")
        try:
            return self.candidates[key]
        except KeyError as exc:
            raise ValueError(f"unknown signal candidate: {key}") from exc

    def to_record(self) -> dict[str, object]:
        """Return the canonical registry record.

        Returns
        -------
        dict[str, object]
            JSON-compatible registry identity and candidate records.
        """
        return {
            "candidates": [
                self.candidates[key].to_record() for key in sorted(self.candidates)
            ],
            "evidence_claimed": False,
            "reactor_registry_digest": self.reactor_registry_digest,
            "reactor_registry_version": self.reactor_registry_version,
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the canonical registry record."""
        payload = json.dumps(
            self.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_COMMON_DIRECT = (
    "calibration",
    "clock_epoch",
    "diagnostic_reference",
    "provenance",
    "quality",
    "uncertainty",
    "validity",
)
_COMMON_DERIVED = (
    "calibration",
    "clock_epoch",
    "mode_identity",
    "observability_threshold",
    "observation_operator",
    "operator_validation",
    "provenance",
    "quality",
    "reference_signal",
    "uncertainty",
    "validity",
)
_COMMON_EVENT = (
    "clock_epoch",
    "event_reference",
    "provenance",
    "repetition_evidence",
    "timing_uncertainty",
    "validity",
)
_COMMON_NONCYCLIC = (
    "calibration",
    "clock_epoch",
    "coordinate_frame",
    "provenance",
    "quality",
    "uncertainty",
    "units",
    "validity",
)
_COMMON_NUMERICAL = (
    "initial_condition",
    "model_revision",
    "provenance",
    "simulation_clock",
    "solver_validity",
)

_CLOSED = tuple(
    sorted(
        {
            "conventional_tokamak",
            "field_reversed_configuration",
            "heliotron",
            "reversed_field_pinch",
            "spheromak",
            "spherical_tokamak",
            "stellarator",
            "torsatron",
        }
    )
)
_OPEN = tuple(
    sorted(
        {
            "cusp",
            "gas_dynamic_mirror",
            "levitated_dipole",
            "polywell",
            "simple_magnetic_mirror",
            "tandem_mirror",
        }
    )
)
_SELF_MAGNETIC = tuple(
    sorted(
        {
            "dense_plasma_focus",
            "sheared_flow_z_pinch",
            "theta_pinch",
            "z_pinch",
        }
    )
)
_INERTIAL = tuple(
    sorted(
        {
            "ion_beam_icf",
            "laser_icf_direct_drive",
            "laser_icf_fast_or_shock_ignition",
            "laser_icf_indirect_drive",
            "projectile_or_impact_icf",
            "pulsed_electron_beam_icf",
        }
    )
)
_MAGNETO_INERTIAL = tuple(
    sorted(
        {
            "frc_compression_mif",
            "maglif",
            "mechanical_or_liquid_liner_mif",
            "plasma_jet_mif",
        }
    )
)
_IEC = ("gridded_iec", "polywell")
_BEAM = ("beam_target", "colliding_beam")
_ALL_V1_0_0 = tuple(sorted(REACTOR_REGISTRY_V1_0_0.configurations))
_ALL = tuple(sorted(DEFAULT_REACTOR_REGISTRY.configurations))


def _candidate(
    candidate_id: str,
    phenomenon: str,
    configurations: tuple[str, ...],
    observability_class: ObservabilityClass,
) -> ReactorSignalCandidateProfile:
    """Build a candidate with evidence rules derived from its class."""
    carriers = tuple(sorted(_ALLOWED_CARRIERS[observability_class], key=str))
    evidence = {
        ObservabilityClass.DIRECT_CYCLIC: _COMMON_DIRECT,
        ObservabilityClass.DERIVED_CYCLIC: _COMMON_DERIVED,
        ObservabilityClass.EVENT_RELATIVE: _COMMON_EVENT,
        ObservabilityClass.NONCYCLIC_FEATURE: _COMMON_NONCYCLIC,
        ObservabilityClass.NUMERICAL_ONLY: _COMMON_NUMERICAL,
        ObservabilityClass.UNOBSERVABLE: ("observability_failure_reason",),
    }[observability_class]
    return ReactorSignalCandidateProfile(
        candidate_id=candidate_id,
        phenomenon=phenomenon,
        configurations=configurations,
        observability_class=observability_class,
        admissible_carriers=carriers,
        required_evidence=evidence,
        unmet_evidence=_REQUIRED_DISPOSITION[observability_class],
        reference_required=observability_class
        in {
            ObservabilityClass.DIRECT_CYCLIC,
            ObservabilityClass.DERIVED_CYCLIC,
            ObservabilityClass.EVENT_RELATIVE,
        },
        observation_operator_required=(
            observability_class is ObservabilityClass.DERIVED_CYCLIC
        ),
        repeated_cycle_required=(
            observability_class is ObservabilityClass.EVENT_RELATIVE
        ),
    )


_COMMON_CANDIDATES = (
    _candidate(
        "closed.equilibrium_profiles",
        "magnetic equilibrium, geometry, and transport profiles",
        _CLOSED,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "closed.resolved_mhd_mode",
        "spatially and temporally resolved MHD mode",
        _CLOSED,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "closed.recurrent_transient",
        "recurrent edge or core transient sequence",
        ("conventional_tokamak", "spherical_tokamak"),
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "open.drive_reference",
        "RF, microwave, or beam drive phase against a facility reference",
        _OPEN,
        ObservabilityClass.DIRECT_CYCLIC,
    ),
    _candidate(
        "open.equilibrium_and_loss",
        "equilibrium, loss-cone, end-loss, and steady-state metrics",
        _OPEN,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "open.resolved_interchange_mode",
        "resolved interchange, flute, or related instability mode",
        _OPEN,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "self_magnetic.drive_waveform",
        "pulsed-power current and voltage event progression",
        _SELF_MAGNETIC,
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "self_magnetic.resolved_instability_mode",
        "resolved sausage, kink, or other pinch mode",
        _SELF_MAGNETIC,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "inertial.driver_timing",
        "laser, beam, projectile, or igniter timing within a shot",
        _INERTIAL,
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "inertial.implosion_trajectory",
        "implosion radius, velocity, convergence, and stagnation trajectory",
        _INERTIAL,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "inertial.resolved_asymmetry_mode",
        "resolved nonzero-order implosion asymmetry",
        _INERTIAL,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "inertial.shot_outcome",
        "yield, ion temperature, areal density, and burn outcome",
        _INERTIAL,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "magneto_inertial.driver_arrival",
        "driver, plasma, liner, or target arrival within a compression event",
        _MAGNETO_INERTIAL,
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "magneto_inertial.resolved_asymmetry_mode",
        "resolved liner, jet, or target asymmetry mode",
        _MAGNETO_INERTIAL,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "magneto_inertial.translation_and_compression",
        "translation, merge, radius, velocity, and compression trajectory",
        _MAGNETO_INERTIAL,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "iec.resolved_bunching",
        "resolved electrostatic density or potential oscillation",
        _IEC,
        ObservabilityClass.DERIVED_CYCLIC,
    ),
    _candidate(
        "iec.steady_state",
        "potential, density, current, and fusion-yield state",
        _IEC,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "beam.rf_bunch_phase",
        "accelerator cavity and bunch phase",
        _BEAM,
        ObservabilityClass.DIRECT_CYCLIC,
    ),
    _candidate(
        "beam.target_outcome",
        "beam-target interaction and fusion-yield outcome",
        _BEAM,
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "hybrid.source_blanket_response",
        "fusion source and delayed blanket neutronic or thermal response",
        ("fusion_fission_hybrid",),
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
)

_EXTENSION_CANDIDATES = (
    _candidate(
        "lattice.external_driver_timing",
        "external-driver exposure timing within a declared material run",
        ("scpn.reactor_systems:lattice_confinement_fusion",),
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "lattice.material_and_nuclear_response",
        (
            "target loading, material state, calibrated nuclear signatures, "
            "and calorimetric outcome"
        ),
        ("scpn.reactor_systems:lattice_confinement_fusion",),
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
    _candidate(
        "muon.beam_and_target_timing",
        "muon delivery and target interaction timing within an experimental run",
        ("scpn.reactor_systems:muon_catalysed_fusion",),
        ObservabilityClass.EVENT_RELATIVE,
    ),
    _candidate(
        "muon.catalysis_kinetics_and_outcome",
        (
            "muon stopping, molecular formation, fusion, sticking, decay, "
            "and neutron-yield outcome"
        ),
        ("scpn.reactor_systems:muon_catalysed_fusion",),
        ObservabilityClass.NONCYCLIC_FEATURE,
    ),
)

_NUMERICAL_CANDIDATE_V1_0_0 = _candidate(
    "model.synthetic_oscillator_coordinate",
    "model-owned synthetic oscillator coordinate",
    _ALL_V1_0_0,
    ObservabilityClass.NUMERICAL_ONLY,
)
_NUMERICAL_CANDIDATE = _candidate(
    "model.synthetic_oscillator_coordinate",
    "model-owned synthetic oscillator coordinate",
    _ALL,
    ObservabilityClass.NUMERICAL_ONLY,
)

_CANDIDATES_V1_0_0 = (*_COMMON_CANDIDATES, _NUMERICAL_CANDIDATE_V1_0_0)
_CANDIDATES = (*_COMMON_CANDIDATES, *_EXTENSION_CANDIDATES, _NUMERICAL_CANDIDATE)

REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0 = ReactorObservabilityProfileRegistry(
    version=REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0_VERSION,
    reactor_registry_version=REACTOR_REGISTRY_V1_0_0.version,
    reactor_registry_digest=REACTOR_REGISTRY_V1_0_0.digest,
    candidates={candidate.candidate_id: candidate for candidate in _CANDIDATES_V1_0_0},
)

DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY = ReactorObservabilityProfileRegistry(
    version=REACTOR_OBSERVABILITY_PROFILE_REGISTRY_VERSION,
    reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
    reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
    candidates={candidate.candidate_id: candidate for candidate in _CANDIDATES},
)

_REACTOR_OBSERVABILITY_PROFILE_REGISTRY_RELEASES = MappingProxyType(
    {
        (
            REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0.version,
            REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0.digest,
        ): REACTOR_OBSERVABILITY_PROFILE_REGISTRY_V1_0_0,
        (
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version,
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest,
        ): DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    }
)


def resolve_reactor_observability_profile_registry_release(
    version: str,
    digest: str,
) -> ReactorObservabilityProfileRegistry:
    """Resolve one exact immutable observability-catalogue release.

    Parameters
    ----------
    version : str
        Semantic version of the requested registry.
    digest : str
        Lowercase SHA-256 digest of the requested registry.

    Returns
    -------
    ReactorObservabilityProfileRegistry
        Exact immutable registry matching both identifiers.

    Raises
    ------
    ValueError
        If either identifier is invalid or the release is unknown.
    """
    key = (
        require_semver(version, field="observability registry version"),
        require_sha256(digest, field="observability registry digest"),
    )
    try:
        return _REACTOR_OBSERVABILITY_PROFILE_REGISTRY_RELEASES[key]
    except KeyError as exc:
        raise ValueError("unrecognised observability registry release") from exc
