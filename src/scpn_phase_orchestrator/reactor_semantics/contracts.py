# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic contracts

"""U0 reactor context, observation, phase, relationship, and regime contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

from .evidence import (
    CalibrationReference,
    ClockReference,
    ProvenanceRecord,
    QualityAssessment,
    Uncertainty,
    ValidityWindow,
)
from .registry import DEFAULT_REACTOR_REGISTRY, ReactorConfigurationRegistry
from .vocabulary import (
    ACTION_OWNER,
    REVIEW_ONLY_AUTHORITY,
    SEMANTIC_OWNER,
    U0_SCHEMA_VERSION,
    ClockKind,
    ConfinementFamily,
    ConversionKind,
    DriverKind,
    EvidenceClass,
    OperatingCadence,
    PhaseRelationType,
    QualityState,
    ReactionKind,
    RegimeState,
    RelationInterpretation,
    SemanticCarrier,
    ValidityState,
    finite_real,
    non_negative_real,
    probability,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
    require_u0_schema,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]

_PHASE_CARRIERS = frozenset(
    {
        SemanticCarrier.CYCLIC_PHASE,
        SemanticCarrier.COMPLEX_MODE,
        SemanticCarrier.FIELD_PHASE,
        SemanticCarrier.EVENT_CYCLE,
        SemanticCarrier.NUMERICAL_PHASE,
    }
)
_NON_PHASE_CARRIERS = frozenset(
    {
        SemanticCarrier.BOUNDED_FEATURE,
        SemanticCarrier.CATEGORICAL_STATE,
        SemanticCarrier.PROTOCOL_PHASE,
    }
)
_USABLE_VALIDITY = frozenset({ValidityState.VALID, ValidityState.DEGRADED})


@dataclass(frozen=True, slots=True)
class ReactorContext:
    """Faceted reactor identity with no privileged configuration family."""

    context_id: str
    configuration: str
    confinement_family: ConfinementFamily
    topology: str
    coordinate_frame: str
    drivers: tuple[DriverKind, ...]
    cadence: OperatingCadence
    reaction: ReactionKind
    conversion: ConversionKind
    facility: str
    event_id: str | None
    configuration_version: str
    operating_point: Mapping[str, JsonValue]
    evidence_class: EvidenceClass
    registry_version: str
    registry_digest: str
    schema_version: str = U0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_u0_schema(self.schema_version),
        )
        object.__setattr__(
            self,
            "context_id",
            require_identifier(self.context_id, field="context_id"),
        )
        object.__setattr__(
            self,
            "configuration",
            require_identifier(self.configuration, field="configuration"),
        )
        object.__setattr__(
            self,
            "confinement_family",
            require_enum(
                self.confinement_family,
                ConfinementFamily,
                field="confinement_family",
            ),
        )
        object.__setattr__(
            self, "topology", require_text(self.topology, field="topology")
        )
        object.__setattr__(
            self,
            "coordinate_frame",
            require_identifier(self.coordinate_frame, field="coordinate_frame"),
        )
        if not self.drivers:
            raise ValueError("reactor context requires at least one driver")
        object.__setattr__(
            self,
            "drivers",
            tuple(
                require_enum(driver, DriverKind, field="driver")
                for driver in self.drivers
            ),
        )
        if len(set(self.drivers)) != len(self.drivers):
            raise ValueError("reactor context drivers must be unique")
        object.__setattr__(
            self,
            "facility",
            require_identifier(self.facility, field="facility"),
        )
        object.__setattr__(
            self,
            "cadence",
            require_enum(self.cadence, OperatingCadence, field="cadence"),
        )
        object.__setattr__(
            self,
            "reaction",
            require_enum(self.reaction, ReactionKind, field="reaction"),
        )
        object.__setattr__(
            self,
            "conversion",
            require_enum(self.conversion, ConversionKind, field="conversion"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            require_enum(self.evidence_class, EvidenceClass, field="evidence_class"),
        )
        if self.event_id is not None:
            object.__setattr__(
                self,
                "event_id",
                require_identifier(self.event_id, field="event_id"),
            )
        if (
            self.cadence
            in {
                OperatingCadence.PULSED_SHOT,
                OperatingCadence.REPETITIVE_TARGET,
                OperatingCadence.SINGLE_EXPERIMENT,
            }
            and self.event_id is None
        ):
            raise ValueError("pulsed or experimental context requires event_id")
        object.__setattr__(
            self,
            "configuration_version",
            require_semver(self.configuration_version, field="configuration_version"),
        )
        object.__setattr__(
            self,
            "registry_version",
            require_semver(self.registry_version, field="registry_version"),
        )
        object.__setattr__(
            self,
            "registry_digest",
            require_sha256(self.registry_digest, field="registry_digest"),
        )
        object.__setattr__(
            self,
            "operating_point",
            _freeze_json_mapping(self.operating_point, field="operating_point"),
        )

    def validate_registry(
        self,
        registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
    ) -> ReactorContext:
        """Validate configuration identity and family against a registry.

        Parameters
        ----------
        registry : ReactorConfigurationRegistry
            Registry that owns the expected configuration identity.

        Returns
        -------
        ReactorContext
            This context after successful validation.

        Raises
        ------
        ValueError
            If registry identity, family, or topology differs.
        """
        if self.registry_version != registry.version:
            raise ValueError(
                "reactor context registry_version does not match the active registry"
            )
        if self.registry_digest != registry.digest:
            raise ValueError(
                "reactor context registry_digest does not match the active registry"
            )
        configuration = registry.resolve(self.configuration)
        if configuration.confinement_family is not self.confinement_family:
            raise ValueError(
                "reactor context confinement_family contradicts its configuration"
            )
        if configuration.topology != self.topology:
            raise ValueError("reactor context topology contradicts its configuration")
        return self

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible context record.

        Returns
        -------
        dict[str, object]
            Complete reactor-context fields for serialization.
        """
        return {
            "cadence": self.cadence.value,
            "configuration": self.configuration,
            "configuration_version": self.configuration_version,
            "confinement_family": self.confinement_family.value,
            "context_id": self.context_id,
            "conversion": self.conversion.value,
            "coordinate_frame": self.coordinate_frame,
            "drivers": [driver.value for driver in self.drivers],
            "evidence_class": self.evidence_class.value,
            "event_id": self.event_id,
            "facility": self.facility,
            "operating_point": _thaw_json(self.operating_point),
            "reaction": self.reaction.value,
            "registry_version": self.registry_version,
            "registry_digest": self.registry_digest,
            "schema_version": self.schema_version,
            "topology": self.topology,
        }

    @classmethod
    def from_record(
        cls,
        payload: object,
        *,
        registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
    ) -> ReactorContext:
        """Construct and registry-validate a context from serialized input.

        Parameters
        ----------
        payload : object
            Candidate reactor-context mapping.
        registry : ReactorConfigurationRegistry
            Registry used to validate the decoded configuration.

        Returns
        -------
        ReactorContext
            Validated reactor context.

        Raises
        ------
        ValueError
            If list, operating-point, or registry fields are invalid.
        """
        record = require_exact_keys(
            payload,
            required=frozenset(
                {
                    "cadence",
                    "configuration",
                    "configuration_version",
                    "confinement_family",
                    "context_id",
                    "conversion",
                    "coordinate_frame",
                    "drivers",
                    "evidence_class",
                    "event_id",
                    "facility",
                    "operating_point",
                    "reaction",
                    "registry_version",
                    "registry_digest",
                    "schema_version",
                    "topology",
                }
            ),
            field="reactor_context",
        )
        drivers = record["drivers"]
        operating_point = record["operating_point"]
        if not isinstance(drivers, list):
            raise ValueError("reactor_context.drivers must be a list")
        if not isinstance(operating_point, dict):
            raise ValueError("reactor_context.operating_point must be an object")
        context = cls(
            context_id=cast(str, record["context_id"]),
            configuration=cast(str, record["configuration"]),
            confinement_family=ConfinementFamily(
                cast(str, record["confinement_family"])
            ),
            topology=cast(str, record["topology"]),
            coordinate_frame=cast(str, record["coordinate_frame"]),
            drivers=tuple(DriverKind(driver) for driver in drivers),
            cadence=OperatingCadence(cast(str, record["cadence"])),
            reaction=ReactionKind(cast(str, record["reaction"])),
            conversion=ConversionKind(cast(str, record["conversion"])),
            facility=cast(str, record["facility"]),
            configuration_version=cast(str, record["configuration_version"]),
            operating_point=operating_point,
            evidence_class=EvidenceClass(cast(str, record["evidence_class"])),
            event_id=_optional_text(record["event_id"], field="event_id"),
            registry_version=cast(str, record["registry_version"]),
            registry_digest=cast(str, record["registry_digest"]),
            schema_version=cast(str, record["schema_version"]),
        )
        return context.validate_registry(registry)


@dataclass(frozen=True, slots=True)
class ObservableDescriptor:
    """Calibrated, timestamped, provenance-bearing reactor observable."""

    observable_id: str
    reactor_context: ReactorContext
    physical_quantity: str
    units: str
    coordinate_frame: str
    spatial_support: str
    diagnostic: str
    channel: str
    value: JsonValue
    clock: ClockReference
    calibration: CalibrationReference
    uncertainty: Uncertainty
    quality: QualityAssessment
    validity: ValidityWindow
    provenance: ProvenanceRecord
    schema_version: str = U0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_u0_schema(self.schema_version),
        )
        object.__setattr__(
            self,
            "observable_id",
            require_identifier(self.observable_id, field="observable_id"),
        )
        for name in (
            "physical_quantity",
            "units",
            "spatial_support",
            "diagnostic",
            "channel",
        ):
            object.__setattr__(
                self,
                name,
                require_text(getattr(self, name), field=name),
            )
        object.__setattr__(
            self,
            "coordinate_frame",
            require_identifier(self.coordinate_frame, field="coordinate_frame"),
        )
        object.__setattr__(self, "value", _freeze_json(self.value, field="value"))
        if self.provenance.source_project not in {
            SEMANTIC_OWNER,
            "SCPN-FUSION-CORE",
            "SCPN-MIF-CORE",
            "SCPN-CONTROL",
        }:
            raise ValueError("observable provenance has an unrecognized source owner")
        if self.calibration.calibrated_at_ns > self.clock.timestamp_ns:
            raise ValueError("observable calibration cannot postdate its sample")
        if (
            not self.validity.valid_from_ns
            <= self.clock.timestamp_ns
            <= self.validity.valid_until_ns
        ):
            raise ValueError("observable timestamp lies outside its validity window")
        if (
            self.quality.state in {QualityState.UNKNOWN, QualityState.INVALID}
            and self.validity.state in _USABLE_VALIDITY
        ):
            raise ValueError("unknown or invalid quality cannot have usable validity")

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible observable record.

        Returns
        -------
        dict[str, object]
            Complete observable fields for serialization.
        """
        return {
            "calibration": self.calibration.to_record(),
            "channel": self.channel,
            "clock": self.clock.to_record(),
            "coordinate_frame": self.coordinate_frame,
            "diagnostic": self.diagnostic,
            "observable_id": self.observable_id,
            "physical_quantity": self.physical_quantity,
            "provenance": self.provenance.to_record(),
            "quality": self.quality.to_record(),
            "reactor_context": self.reactor_context.to_record(),
            "schema_version": self.schema_version,
            "spatial_support": self.spatial_support,
            "uncertainty": self.uncertainty.to_record(),
            "units": self.units,
            "validity": self.validity.to_record(),
            "value": _thaw_json(self.value),
        }

    @classmethod
    def from_record(
        cls,
        payload: object,
        *,
        registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
    ) -> ObservableDescriptor:
        """Construct an observable from strict serialized input.

        Parameters
        ----------
        payload : object
            Candidate observable mapping.
        registry : ReactorConfigurationRegistry
            Registry used to validate the embedded reactor context.

        Returns
        -------
        ObservableDescriptor
            Validated observable descriptor.
        """
        record = require_exact_keys(
            payload,
            required=frozenset(
                {
                    "calibration",
                    "channel",
                    "clock",
                    "coordinate_frame",
                    "diagnostic",
                    "observable_id",
                    "physical_quantity",
                    "provenance",
                    "quality",
                    "reactor_context",
                    "schema_version",
                    "spatial_support",
                    "uncertainty",
                    "units",
                    "validity",
                    "value",
                }
            ),
            field="observable_descriptor",
        )
        return cls(
            observable_id=cast(str, record["observable_id"]),
            reactor_context=ReactorContext.from_record(
                record["reactor_context"],
                registry=registry,
            ),
            physical_quantity=cast(str, record["physical_quantity"]),
            units=cast(str, record["units"]),
            coordinate_frame=cast(str, record["coordinate_frame"]),
            spatial_support=cast(str, record["spatial_support"]),
            diagnostic=cast(str, record["diagnostic"]),
            channel=cast(str, record["channel"]),
            value=_freeze_json(record["value"], field="value"),
            clock=ClockReference.from_record(record["clock"]),
            calibration=CalibrationReference.from_record(record["calibration"]),
            uncertainty=Uncertainty.from_record(record["uncertainty"]),
            quality=QualityAssessment.from_record(record["quality"]),
            validity=ValidityWindow.from_record(record["validity"]),
            provenance=ProvenanceRecord.from_record(record["provenance"]),
            schema_version=cast(str, record["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PhaseSemanticRecord:
    """Typed phase meaning derived from one or more observables."""

    phase_id: str
    reactor_context_id: str
    observable_ids: tuple[str, ...]
    carrier_type: SemanticCarrier
    phenomenon: str
    phase_rad: float | None
    amplitude: float | None
    frequency_hz: float | None
    bandwidth_hz: float | None
    mode_identity: str | None
    mode_harmonic: tuple[int, int] | None
    phase_origin: str | None
    orientation: str | None
    reference_frame: str
    clock_domain: str
    clock_kind: ClockKind
    clock_epoch: str
    wrap_convention: str | None
    reference_signal: str | None
    extractor: str
    extractor_version: str
    observation_operator: str | None
    uncertainty: Uncertainty
    confidence: float
    observability: float
    observability_threshold: float
    validity: ValidityWindow
    quality: QualityAssessment
    evidence_class: EvidenceClass
    schema_version: str = U0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_u0_schema(self.schema_version),
        )
        object.__setattr__(
            self,
            "phase_id",
            require_identifier(self.phase_id, field="phase_id"),
        )
        object.__setattr__(
            self,
            "reactor_context_id",
            require_identifier(self.reactor_context_id, field="reactor_context_id"),
        )
        object.__setattr__(
            self,
            "carrier_type",
            require_enum(self.carrier_type, SemanticCarrier, field="carrier_type"),
        )
        object.__setattr__(
            self,
            "clock_kind",
            require_enum(self.clock_kind, ClockKind, field="clock_kind"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            require_enum(self.evidence_class, EvidenceClass, field="evidence_class"),
        )
        if not self.observable_ids:
            raise ValueError("phase record requires at least one observable")
        object.__setattr__(
            self,
            "observable_ids",
            tuple(
                require_identifier(item, field="observable_id")
                for item in self.observable_ids
            ),
        )
        if len(set(self.observable_ids)) != len(self.observable_ids):
            raise ValueError("phase observable_ids must be unique")
        object.__setattr__(
            self,
            "phenomenon",
            require_text(self.phenomenon, field="phenomenon"),
        )
        object.__setattr__(
            self,
            "reference_frame",
            require_identifier(self.reference_frame, field="reference_frame"),
        )
        object.__setattr__(
            self,
            "clock_domain",
            require_identifier(self.clock_domain, field="clock_domain"),
        )
        object.__setattr__(
            self,
            "clock_epoch",
            require_identifier(self.clock_epoch, field="clock_epoch"),
        )
        object.__setattr__(
            self,
            "extractor",
            require_identifier(self.extractor, field="extractor"),
        )
        object.__setattr__(
            self,
            "extractor_version",
            require_semver(self.extractor_version, field="extractor_version"),
        )
        object.__setattr__(
            self,
            "confidence",
            probability(self.confidence, field="confidence"),
        )
        object.__setattr__(
            self,
            "observability",
            probability(self.observability, field="observability"),
        )
        threshold = probability(
            self.observability_threshold,
            field="observability_threshold",
        )
        if threshold == 0.0:
            raise ValueError("observability_threshold must be positive")
        object.__setattr__(self, "observability_threshold", threshold)
        if self.observability < threshold:
            if self.validity.state is not ValidityState.UNOBSERVABLE:
                raise ValueError(
                    "below-threshold observability requires UNOBSERVABLE validity"
                )
            if self.phase_rad is not None:
                raise ValueError(
                    "below-threshold observability cannot publish phase_rad"
                )
        _validate_optional_semantics(self)
        _validate_carrier_semantics(self)

    @property
    def is_usable(self) -> bool:
        """Return whether phase comparison is permitted."""
        return (
            self.validity.state in _USABLE_VALIDITY
            and self.quality.state in {QualityState.VALID, QualityState.DEGRADED}
            and self.confidence > 0.0
            and self.observability >= self.observability_threshold
        )

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible phase record.

        Returns
        -------
        dict[str, object]
            Complete phase-semantic fields for serialization.
        """
        return {
            "amplitude": self.amplitude,
            "bandwidth_hz": self.bandwidth_hz,
            "carrier_type": self.carrier_type.value,
            "clock_domain": self.clock_domain,
            "clock_epoch": self.clock_epoch,
            "clock_kind": self.clock_kind.value,
            "confidence": self.confidence,
            "evidence_class": self.evidence_class.value,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "frequency_hz": self.frequency_hz,
            "mode_harmonic": (
                list(self.mode_harmonic) if self.mode_harmonic is not None else None
            ),
            "mode_identity": self.mode_identity,
            "observability": self.observability,
            "observability_threshold": self.observability_threshold,
            "observable_ids": list(self.observable_ids),
            "observation_operator": self.observation_operator,
            "orientation": self.orientation,
            "phase_id": self.phase_id,
            "phase_origin": self.phase_origin,
            "phase_rad": self.phase_rad,
            "phenomenon": self.phenomenon,
            "quality": self.quality.to_record(),
            "reactor_context_id": self.reactor_context_id,
            "reference_frame": self.reference_frame,
            "reference_signal": self.reference_signal,
            "schema_version": self.schema_version,
            "uncertainty": self.uncertainty.to_record(),
            "validity": self.validity.to_record(),
            "wrap_convention": self.wrap_convention,
        }

    @classmethod
    def from_record(cls, payload: object) -> PhaseSemanticRecord:
        """Construct a phase semantic record from strict serialized input.

        Parameters
        ----------
        payload : object
            Candidate phase-semantic mapping.

        Returns
        -------
        PhaseSemanticRecord
            Validated phase-semantic record.

        Raises
        ------
        ValueError
            If the observable identifier collection is not a list.
        """
        required = frozenset(
            {
                "amplitude",
                "bandwidth_hz",
                "carrier_type",
                "clock_domain",
                "clock_epoch",
                "clock_kind",
                "confidence",
                "evidence_class",
                "extractor",
                "extractor_version",
                "frequency_hz",
                "mode_harmonic",
                "mode_identity",
                "observability",
                "observability_threshold",
                "observable_ids",
                "observation_operator",
                "orientation",
                "phase_id",
                "phase_origin",
                "phase_rad",
                "phenomenon",
                "quality",
                "reactor_context_id",
                "reference_frame",
                "reference_signal",
                "schema_version",
                "uncertainty",
                "validity",
                "wrap_convention",
            }
        )
        record = require_exact_keys(payload, required=required, field="phase_record")
        observable_ids = record["observable_ids"]
        if not isinstance(observable_ids, list):
            raise ValueError("phase_record.observable_ids must be a list")
        harmonic = _optional_harmonic(record["mode_harmonic"])
        return cls(
            phase_id=cast(str, record["phase_id"]),
            reactor_context_id=cast(str, record["reactor_context_id"]),
            observable_ids=tuple(observable_ids),
            carrier_type=SemanticCarrier(cast(str, record["carrier_type"])),
            phenomenon=cast(str, record["phenomenon"]),
            phase_rad=_optional_number(record["phase_rad"], field="phase_rad"),
            amplitude=_optional_number(record["amplitude"], field="amplitude"),
            frequency_hz=_optional_number(record["frequency_hz"], field="frequency_hz"),
            bandwidth_hz=_optional_number(record["bandwidth_hz"], field="bandwidth_hz"),
            mode_identity=_optional_text(
                record["mode_identity"], field="mode_identity"
            ),
            mode_harmonic=harmonic,
            phase_origin=_optional_text(record["phase_origin"], field="phase_origin"),
            orientation=_optional_text(record["orientation"], field="orientation"),
            reference_frame=cast(str, record["reference_frame"]),
            clock_domain=cast(str, record["clock_domain"]),
            clock_kind=ClockKind(cast(str, record["clock_kind"])),
            clock_epoch=cast(str, record["clock_epoch"]),
            wrap_convention=_optional_text(
                record["wrap_convention"],
                field="wrap_convention",
            ),
            reference_signal=_optional_text(
                record["reference_signal"],
                field="reference_signal",
            ),
            extractor=cast(str, record["extractor"]),
            extractor_version=cast(str, record["extractor_version"]),
            observation_operator=_optional_text(
                record["observation_operator"],
                field="observation_operator",
            ),
            uncertainty=Uncertainty.from_record(record["uncertainty"]),
            confidence=cast(float, record["confidence"]),
            observability=cast(float, record["observability"]),
            observability_threshold=cast(
                float,
                record["observability_threshold"],
            ),
            validity=ValidityWindow.from_record(record["validity"]),
            quality=QualityAssessment.from_record(record["quality"]),
            evidence_class=EvidenceClass(cast(str, record["evidence_class"])),
            schema_version=cast(str, record["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PhaseRelation:
    """Validated relationship between two compatible phase records."""

    relation_id: str
    source_phase_id: str
    target_phase_id: str
    relation_type: PhaseRelationType
    interpretation: RelationInterpretation
    reference_transform: str | None
    clock_transform_id: str | None
    harmonic_ratio: tuple[int, int]
    lag_s: float
    causal_direction: str | None
    identification_method: str
    evidence_class: EvidenceClass
    schema_version: str = U0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_u0_schema(self.schema_version),
        )
        for field_name in ("relation_id", "source_phase_id", "target_phase_id"):
            object.__setattr__(
                self,
                field_name,
                require_identifier(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "relation_type",
            require_enum(self.relation_type, PhaseRelationType, field="relation_type"),
        )
        object.__setattr__(
            self,
            "interpretation",
            require_enum(
                self.interpretation,
                RelationInterpretation,
                field="interpretation",
            ),
        )
        object.__setattr__(
            self,
            "evidence_class",
            require_enum(self.evidence_class, EvidenceClass, field="evidence_class"),
        )
        if self.source_phase_id == self.target_phase_id:
            raise ValueError("phase relation requires distinct source and target")
        if len(self.harmonic_ratio) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.harmonic_ratio
        ):
            raise ValueError("harmonic_ratio must contain two positive integers")
        object.__setattr__(self, "lag_s", finite_real(self.lag_s, field="lag_s"))
        object.__setattr__(
            self,
            "identification_method",
            require_text(self.identification_method, field="identification_method"),
        )
        for field_name in (
            "reference_transform",
            "clock_transform_id",
            "causal_direction",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    require_identifier(value, field=field_name),
                )

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible relation record.

        Returns
        -------
        dict[str, object]
            Complete phase-relation fields for serialization.
        """
        return {
            "causal_direction": self.causal_direction,
            "clock_transform_id": self.clock_transform_id,
            "evidence_class": self.evidence_class.value,
            "harmonic_ratio": list(self.harmonic_ratio),
            "identification_method": self.identification_method,
            "interpretation": self.interpretation.value,
            "lag_s": self.lag_s,
            "reference_transform": self.reference_transform,
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "schema_version": self.schema_version,
            "source_phase_id": self.source_phase_id,
            "target_phase_id": self.target_phase_id,
        }

    @classmethod
    def from_record(cls, payload: object) -> PhaseRelation:
        """Construct a relation from strict serialized input.

        Parameters
        ----------
        payload : object
            Candidate phase-relation mapping.

        Returns
        -------
        PhaseRelation
            Validated phase relation.
        """
        required = frozenset(
            {
                "causal_direction",
                "clock_transform_id",
                "evidence_class",
                "harmonic_ratio",
                "identification_method",
                "interpretation",
                "lag_s",
                "reference_transform",
                "relation_id",
                "relation_type",
                "schema_version",
                "source_phase_id",
                "target_phase_id",
            }
        )
        record = require_exact_keys(payload, required=required, field="phase_relation")
        harmonic = _required_harmonic(record["harmonic_ratio"])
        return cls(
            relation_id=cast(str, record["relation_id"]),
            source_phase_id=cast(str, record["source_phase_id"]),
            target_phase_id=cast(str, record["target_phase_id"]),
            relation_type=PhaseRelationType(cast(str, record["relation_type"])),
            interpretation=RelationInterpretation(cast(str, record["interpretation"])),
            reference_transform=_optional_text(
                record["reference_transform"],
                field="reference_transform",
            ),
            clock_transform_id=_optional_text(
                record["clock_transform_id"],
                field="clock_transform_id",
            ),
            harmonic_ratio=harmonic,
            lag_s=cast(float, record["lag_s"]),
            causal_direction=_optional_text(
                record["causal_direction"],
                field="causal_direction",
            ),
            identification_method=cast(str, record["identification_method"]),
            evidence_class=EvidenceClass(cast(str, record["evidence_class"])),
            schema_version=cast(str, record["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class RegimeAxis:
    """One independently estimated physical or operational regime axis."""

    name: str
    label: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", require_identifier(self.name, field="axis name")
        )
        object.__setattr__(
            self, "label", require_identifier(self.label, field="axis label")
        )
        object.__setattr__(
            self,
            "confidence",
            probability(self.confidence, field="axis confidence"),
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible regime-axis record.

        Returns
        -------
        dict[str, object]
            Complete regime-axis fields for serialization.
        """
        return {
            "confidence": self.confidence,
            "label": self.label,
            "name": self.name,
        }

    @classmethod
    def from_record(cls, payload: object) -> RegimeAxis:
        """Construct a regime axis from strict serialized input.

        Parameters
        ----------
        payload : object
            Candidate regime-axis mapping.

        Returns
        -------
        RegimeAxis
            Validated regime axis.
        """
        record = require_exact_keys(
            payload,
            required=frozenset({"confidence", "label", "name"}),
            field="regime_axis",
        )
        return cls(
            name=cast(str, record["name"]),
            label=cast(str, record["label"]),
            confidence=cast(float, record["confidence"]),
        )


@dataclass(frozen=True, slots=True)
class RegimeEstimate:
    """Compositional non-actuating reactor regime estimate."""

    regime_id: str
    reactor_context_id: str
    axes: tuple[RegimeAxis, ...]
    state: RegimeState
    evidence_ids: tuple[str, ...]
    classifier: str
    classifier_version: str
    threshold_provenance: tuple[str, ...]
    confidence: float
    hysteresis: float
    dwell_time_s: float
    transition_reason: str
    safety_effect: str
    validity: ValidityWindow
    semantic_owner: str = SEMANTIC_OWNER
    action_owner: str = ACTION_OWNER
    authority: str = REVIEW_ONLY_AUTHORITY
    schema_version: str = U0_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_u0_schema(self.schema_version),
        )
        for field_name in ("regime_id", "reactor_context_id", "classifier"):
            object.__setattr__(
                self,
                field_name,
                require_identifier(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "state",
            require_enum(self.state, RegimeState, field="regime state"),
        )
        object.__setattr__(
            self,
            "classifier_version",
            require_semver(self.classifier_version, field="classifier_version"),
        )
        if not self.axes:
            raise ValueError("regime estimate requires at least one axis")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("regime axis names must be unique")
        if not self.evidence_ids:
            raise ValueError("regime estimate requires evidence_ids")
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(
                require_identifier(item, field="evidence_id")
                for item in self.evidence_ids
            ),
        )
        object.__setattr__(
            self,
            "threshold_provenance",
            tuple(
                require_identifier(item, field="threshold provenance")
                for item in self.threshold_provenance
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            probability(self.confidence, field="regime confidence"),
        )
        object.__setattr__(
            self,
            "hysteresis",
            non_negative_real(self.hysteresis, field="hysteresis"),
        )
        object.__setattr__(
            self,
            "dwell_time_s",
            non_negative_real(self.dwell_time_s, field="dwell_time_s"),
        )
        object.__setattr__(
            self,
            "transition_reason",
            require_text(self.transition_reason, field="transition_reason"),
        )
        object.__setattr__(
            self,
            "safety_effect",
            require_text(self.safety_effect, field="safety_effect"),
        )
        if self.semantic_owner != SEMANTIC_OWNER:
            raise ValueError("SPO must remain the semantic owner")
        if self.action_owner != ACTION_OWNER:
            raise ValueError("SCPN-CONTROL must remain the action owner")
        if self.authority != REVIEW_ONLY_AUTHORITY:
            raise ValueError("regime estimates must remain review-only")
        if (
            self.validity.state not in _USABLE_VALIDITY
            and self.state is not RegimeState.UNKNOWN
        ):
            raise ValueError("non-usable validity requires UNKNOWN regime state")

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible regime record.

        Returns
        -------
        dict[str, object]
            Complete regime-estimate fields for serialization.
        """
        return {
            "action_owner": self.action_owner,
            "authority": self.authority,
            "axes": [axis.to_record() for axis in self.axes],
            "classifier": self.classifier,
            "classifier_version": self.classifier_version,
            "confidence": self.confidence,
            "dwell_time_s": self.dwell_time_s,
            "evidence_ids": list(self.evidence_ids),
            "hysteresis": self.hysteresis,
            "reactor_context_id": self.reactor_context_id,
            "regime_id": self.regime_id,
            "safety_effect": self.safety_effect,
            "schema_version": self.schema_version,
            "semantic_owner": self.semantic_owner,
            "state": self.state.value,
            "threshold_provenance": list(self.threshold_provenance),
            "transition_reason": self.transition_reason,
            "validity": self.validity.to_record(),
        }

    @classmethod
    def from_record(cls, payload: object) -> RegimeEstimate:
        """Construct a compositional regime estimate from strict input.

        Parameters
        ----------
        payload : object
            Candidate regime-estimate mapping.

        Returns
        -------
        RegimeEstimate
            Validated compositional regime estimate.

        Raises
        ------
        ValueError
            If axes, evidence identifiers, or threshold provenance are not lists.
        """
        required = frozenset(
            {
                "action_owner",
                "authority",
                "axes",
                "classifier",
                "classifier_version",
                "confidence",
                "dwell_time_s",
                "evidence_ids",
                "hysteresis",
                "reactor_context_id",
                "regime_id",
                "safety_effect",
                "schema_version",
                "semantic_owner",
                "state",
                "threshold_provenance",
                "transition_reason",
                "validity",
            }
        )
        record = require_exact_keys(payload, required=required, field="regime_estimate")
        axes = record["axes"]
        evidence_ids = record["evidence_ids"]
        threshold_provenance = record["threshold_provenance"]
        if not isinstance(axes, list):
            raise ValueError("regime_estimate.axes must be a list")
        if not isinstance(evidence_ids, list):
            raise ValueError("regime_estimate.evidence_ids must be a list")
        if not isinstance(threshold_provenance, list):
            raise ValueError("regime_estimate.threshold_provenance must be a list")
        return cls(
            regime_id=cast(str, record["regime_id"]),
            reactor_context_id=cast(str, record["reactor_context_id"]),
            axes=tuple(RegimeAxis.from_record(axis) for axis in axes),
            state=RegimeState(cast(str, record["state"])),
            evidence_ids=tuple(evidence_ids),
            classifier=cast(str, record["classifier"]),
            classifier_version=cast(str, record["classifier_version"]),
            threshold_provenance=tuple(threshold_provenance),
            confidence=cast(float, record["confidence"]),
            hysteresis=cast(float, record["hysteresis"]),
            dwell_time_s=cast(float, record["dwell_time_s"]),
            transition_reason=cast(str, record["transition_reason"]),
            safety_effect=cast(str, record["safety_effect"]),
            validity=ValidityWindow.from_record(record["validity"]),
            semantic_owner=cast(str, record["semantic_owner"]),
            action_owner=cast(str, record["action_owner"]),
            authority=cast(str, record["authority"]),
            schema_version=cast(str, record["schema_version"]),
        )


def build_phase_relation(
    source: PhaseSemanticRecord,
    target: PhaseSemanticRecord,
    *,
    relation_id: str,
    relation_type: PhaseRelationType,
    interpretation: RelationInterpretation,
    identification_method: str,
    evidence_class: EvidenceClass,
    reference_transform: str | None = None,
    clock_transform_id: str | None = None,
    harmonic_ratio: tuple[int, int] = (1, 1),
    lag_s: float = 0.0,
    causal_direction: str | None = None,
) -> PhaseRelation:
    """Build a relation only when phase records are semantically comparable.

    Different frames, clock domains, or harmonics require explicit transforms.
    Non-phase carriers and unusable records are rejected.

    Parameters
    ----------
    source : PhaseSemanticRecord
        Source phase record.
    target : PhaseSemanticRecord
        Target phase record.
    relation_id : str
        Identifier for the relation.
    relation_type : PhaseRelationType
        Declared relationship type.
    interpretation : RelationInterpretation
        Operational interpretation of the relation.
    identification_method : str
        Method used to identify the relation.
    evidence_class : EvidenceClass
        Evidence maturity supporting the relation.
    reference_transform : str or None
        Explicit transform between reference frames, when required.
    clock_transform_id : str or None
        Explicit transform between clock identities, when required.
    harmonic_ratio : tuple[int, int]
        Positive source-to-target harmonic ratio.
    lag_s : float
        Signed relation lag in seconds.
    causal_direction : str or None
        Optional declared causal direction.

    Returns
    -------
    PhaseRelation
        Validated relation between the supplied phase records.

    Raises
    ------
    ValueError
        If records are unusable or their semantic identities are incompatible.
    """
    if source.carrier_type not in _PHASE_CARRIERS:
        raise ValueError("source carrier is not phase-comparable")
    if target.carrier_type not in _PHASE_CARRIERS:
        raise ValueError("target carrier is not phase-comparable")
    if not source.is_usable or not target.is_usable:
        raise ValueError("phase comparison requires usable source and target")
    if source.reactor_context_id != target.reactor_context_id:
        raise ValueError("cross-context phase comparison is not implicit")
    if source.reference_frame != target.reference_frame and reference_transform is None:
        raise ValueError("incompatible reference frames require a declared transform")
    source_clock = (source.clock_domain, source.clock_kind, source.clock_epoch)
    target_clock = (target.clock_domain, target.clock_kind, target.clock_epoch)
    if source_clock != target_clock and clock_transform_id is None:
        raise ValueError("incompatible clock domains require a declared transform")
    if source.mode_harmonic != target.mode_harmonic:
        if relation_type is not PhaseRelationType.HARMONIC:
            raise ValueError("different harmonics require a harmonic relation")
        if harmonic_ratio == (1, 1):
            raise ValueError("different harmonics require an explicit ratio")
    return PhaseRelation(
        relation_id=relation_id,
        source_phase_id=source.phase_id,
        target_phase_id=target.phase_id,
        relation_type=relation_type,
        interpretation=interpretation,
        reference_transform=reference_transform,
        clock_transform_id=clock_transform_id,
        harmonic_ratio=harmonic_ratio,
        lag_s=lag_s,
        causal_direction=causal_direction,
        identification_method=identification_method,
        evidence_class=evidence_class,
    )


def validate_observable_sequence(
    observables: tuple[ObservableDescriptor, ...],
) -> tuple[ObservableDescriptor, ...]:
    """Return one usable, strictly monotonic observable stream or fail closed.

    Parameters
    ----------
    observables : tuple[ObservableDescriptor, ...]
        Candidate samples from one observable stream.

    Returns
    -------
    tuple[ObservableDescriptor, ...]
        The unchanged validated sequence.

    Raises
    ------
    ValueError
        If the sequence is empty, mixed, non-monotonic, or unusable.
    """
    if not observables:
        raise ValueError("observable sequence must not be empty")
    first = observables[0]
    expected = (
        first.observable_id,
        first.reactor_context.context_id,
        first.coordinate_frame,
        first.clock.domain,
        first.clock.kind,
        first.clock.epoch,
    )
    previous_timestamp = -1
    for observable in observables:
        current = (
            observable.observable_id,
            observable.reactor_context.context_id,
            observable.coordinate_frame,
            observable.clock.domain,
            observable.clock.kind,
            observable.clock.epoch,
        )
        if current != expected:
            raise ValueError(
                "observable sequence mixes stream, context, frame, or clock"
            )
        if observable.clock.timestamp_ns <= previous_timestamp:
            raise ValueError("observable sequence clock must be strictly monotonic")
        if (
            observable.validity.state not in _USABLE_VALIDITY
            or observable.quality.state
            not in {QualityState.VALID, QualityState.DEGRADED}
        ):
            raise ValueError("observable sequence contains unusable evidence")
        previous_timestamp = observable.clock.timestamp_ns
    return observables


def _validate_optional_semantics(record: PhaseSemanticRecord) -> None:
    """Normalize and validate optional phase fields."""
    for field_name in (
        "mode_identity",
        "phase_origin",
        "orientation",
        "wrap_convention",
        "reference_signal",
        "observation_operator",
    ):
        value = getattr(record, field_name)
        if value is not None:
            object.__setattr__(
                record,
                field_name,
                require_text(value, field=field_name),
            )
    for field_name in ("amplitude", "frequency_hz", "bandwidth_hz"):
        value = getattr(record, field_name)
        if value is not None:
            object.__setattr__(
                record,
                field_name,
                non_negative_real(value, field=field_name),
            )
    if record.phase_rad is not None:
        phase = finite_real(record.phase_rad, field="phase_rad")
        if not 0.0 <= phase < 2.0 * math.pi:
            raise ValueError("phase_rad must use the [0, 2*pi) convention")
        object.__setattr__(record, "phase_rad", phase)
    if record.mode_harmonic is not None:
        _required_harmonic(record.mode_harmonic)


def _validate_carrier_semantics(record: PhaseSemanticRecord) -> None:
    """Enforce phase-carrier and validity invariants."""
    usable = record.validity.state in _USABLE_VALIDITY
    if not usable and record.phase_rad is not None:
        raise ValueError("non-usable phase record cannot publish phase_rad")
    if record.quality.state in {QualityState.UNKNOWN, QualityState.INVALID} and usable:
        raise ValueError("unknown or invalid quality cannot publish usable phase")
    if record.carrier_type in _NON_PHASE_CARRIERS and record.phase_rad is not None:
        raise ValueError("non-phase carrier cannot publish phase_rad")
    if record.carrier_type in _PHASE_CARRIERS and usable and record.phase_rad is None:
        raise ValueError("usable phase carrier requires phase_rad")
    if record.carrier_type is SemanticCarrier.COMPLEX_MODE:
        if record.amplitude is None or record.mode_identity is None:
            raise ValueError("complex_mode requires amplitude and mode_identity")
        if record.amplitude == 0.0 and usable:
            raise ValueError("zero-amplitude complex mode is unobservable")
    if (
        record.carrier_type is SemanticCarrier.EVENT_CYCLE
        and record.reference_signal is None
        and record.phase_rad is not None
    ):
        raise ValueError("event_cycle phase requires a reference_signal")
    if record.carrier_type is SemanticCarrier.NUMERICAL_PHASE and (
        record.evidence_class
        in {
            EvidenceClass.OBSERVED,
            EvidenceClass.EXPERIMENTAL,
        }
    ):
        raise ValueError(
            "numerical_phase cannot claim observed or experimental evidence"
        )
    if record.phase_rad is not None:
        required = {
            "phase_origin": record.phase_origin,
            "orientation": record.orientation,
            "wrap_convention": record.wrap_convention,
            "reference_signal": record.reference_signal,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("phase_rad requires " + ", ".join(sorted(missing)))


def _freeze_json(value: object, *, field: str) -> JsonValue:
    """Validate and deeply freeze a JSON value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, field=field)
    raise ValueError(f"{field} must contain only JSON-compatible values")


def _freeze_json_mapping(
    value: Mapping[str, object],
    *,
    field: str,
) -> Mapping[str, JsonValue]:
    """Validate and deeply freeze a string-keyed JSON mapping."""
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{field} keys must be non-empty strings")
    frozen = {
        key: _freeze_json(item, field=f"{field}.{key}")
        for key, item in sorted(value.items())
    }
    return MappingProxyType(frozen)


def _thaw_json(value: JsonValue) -> object:
    """Return mutable JSON primitives for serialization."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _optional_number(value: object, *, field: str) -> float | None:
    """Return a finite optional real."""
    if value is None:
        return None
    return finite_real(value, field=field)


def _optional_text(value: object, *, field: str) -> str | None:
    """Return optional non-empty text."""
    if value is None:
        return None
    return require_text(value, field=field)


def _optional_harmonic(value: object) -> tuple[int, int] | None:
    """Return an optional two-index mode harmonic."""
    if value is None:
        return None
    return _required_harmonic(value)


def _required_harmonic(value: object) -> tuple[int, int]:
    """Return a two-integer mode harmonic."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("mode harmonic must contain two integers")
    first, second = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("mode harmonic must contain two integers")
    return (first, second)


__all__ = [
    "JsonValue",
    "ObservableDescriptor",
    "PhaseRelation",
    "PhaseSemanticRecord",
    "ReactorContext",
    "RegimeAxis",
    "RegimeEstimate",
    "build_phase_relation",
    "validate_observable_sequence",
]
