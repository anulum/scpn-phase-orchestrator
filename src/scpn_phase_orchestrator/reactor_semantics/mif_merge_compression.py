# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — MIF merge-compression semantic adapter
"""Strict MIF merge-compression evidence to review-only U0 semantics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final, TypeVar, cast

from .contracts import (
    JsonValue,
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
from .handoff import MAX_SOURCE_ENVELOPE_BYTES
from .registry import DEFAULT_REACTOR_REGISTRY, ReactorConfigurationRegistry
from .serialization import contract_from_record, contract_to_record
from .vocabulary import (
    ACTION_OWNER,
    REVIEW_ONLY_AUTHORITY,
    U0_SCHEMA_VERSION,
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
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
)

MIF_MERGE_COMPRESSION_SOURCE_SCHEMA: Final = (
    "scpn-mif-core.merge-compression-observation.v1"
)
MIF_MERGE_COMPRESSION_SOURCE_VERSION: Final = "1.0.0"
MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA: Final = (
    "scpn-phase-orchestrator.mif-merge-compression-handoff.v1"
)
MIF_MERGE_COMPRESSION_HANDOFF_VERSION: Final = "1.0.0"
MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES: Final = 128 * 1024 * 1024

_MIF_PROJECT = "SCPN-MIF-CORE"
_CONFIGURATION = "frc_compression_mif"
_CALIBRATION_ID = "mif.merge_compression.model_declared_units.v1"
_TRANSFER_ID = "mif.merge_compression.identity_projection.v1"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_MAX_SAFE_INTEGER = 2**53 - 1
_SOURCE_ROOT_KEYS = frozenset(
    {
        "event_id",
        "payload",
        "payload_sha256",
        "schema",
        "schema_version",
        "source_project",
        "source_revision",
    }
)
_PAYLOAD_KEYS = frozenset(
    {
        "authority",
        "clock",
        "evidence",
        "kinematics",
        "merge_window",
        "reactor",
        "trigger",
    }
)
_HANDOFF_PAYLOAD_KEYS = frozenset(
    {
        "actionable",
        "authority",
        "event_id",
        "observables",
        "phase_relations",
        "reactor_context",
        "regime",
        "registry_digest",
        "registry_version",
        "semantics",
        "source_envelope_json",
        "source_envelope_sha256",
        "source_project",
        "source_revision",
        "source_schema",
        "u0_schema_version",
    }
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class MIFMergeCompressionHandoff:
    """Immutable, non-actuating MIF merge-compression semantic handoff."""

    source_revision: str
    source_envelope_json: str
    event_id: str
    context: ReactorContext
    observables: tuple[ObservableDescriptor, ...]
    semantics: tuple[PhaseSemanticRecord, ...]
    regime: RegimeEstimate
    source_schema: str = MIF_MERGE_COMPRESSION_SOURCE_SCHEMA
    source_project: str = _MIF_PROJECT
    phase_relations: tuple[()] = ()
    authority: str = REVIEW_ONLY_AUTHORITY
    actionable: bool = False
    schema: str = MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA
    schema_version: str = MIF_MERGE_COMPRESSION_HANDOFF_VERSION

    def __post_init__(self) -> None:
        """Validate source custody and the complete U0 contract graph."""
        if self.schema != MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA:
            raise ValueError("unsupported MIF merge-compression handoff schema")
        if require_semver(self.schema_version, field="handoff schema_version") != (
            MIF_MERGE_COMPRESSION_HANDOFF_VERSION
        ):
            raise ValueError("unsupported MIF merge-compression handoff version")
        if self.source_schema != MIF_MERGE_COMPRESSION_SOURCE_SCHEMA:
            raise ValueError("unsupported MIF merge-compression source schema")
        if self.source_project != _MIF_PROJECT:
            raise ValueError("MIF handoff source owner must be SCPN-MIF-CORE")
        if _HEX_40.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a 40-character Git revision")
        object.__setattr__(
            self, "event_id", require_identifier(self.event_id, field="event_id")
        )
        source_bytes = self.source_envelope_json.encode("utf-8")
        source, _, _ = _decode_source(source_bytes, expected_sha256=None)
        if source["source_revision"] != self.source_revision:
            raise ValueError(
                "handoff source revision does not match embedded MIF bytes"
            )
        if source["event_id"] != self.event_id:
            raise ValueError("handoff event_id does not match embedded MIF bytes")
        if self.authority != REVIEW_ONLY_AUTHORITY or self.actionable is not False:
            raise ValueError("MIF handoff must remain review-only and non-actionable")
        self._validate_contract_graph()

    @property
    def source_envelope_sha256(self) -> str:
        """Return SHA-256 of the exact embedded MIF bytes."""
        return hashlib.sha256(self.source_envelope_json.encode("utf-8")).hexdigest()

    def _validate_contract_graph(self) -> None:
        """Validate context, clock, semantic coverage, and regime consistency."""
        self.context.validate_registry()
        if self.context.configuration != _CONFIGURATION:
            raise ValueError("MIF handoff requires frc_compression_mif context")
        if self.context.event_id != self.event_id:
            raise ValueError("handoff and context event_id must match")
        if not self.observables:
            raise ValueError("MIF handoff requires observables")
        observable_ids = tuple(item.observable_id for item in self.observables)
        if len(set(observable_ids)) != len(observable_ids):
            raise ValueError("MIF handoff observable_ids must be unique")
        clocks: set[tuple[str, ClockKind, str, int]] = set()
        for observable in self.observables:
            if observable.reactor_context != self.context:
                raise ValueError("MIF observable context must match the handoff")
            if observable.provenance.source_project != _MIF_PROJECT:
                raise ValueError("MIF observables must retain MIF provenance")
            clocks.add(
                (
                    observable.clock.domain,
                    observable.clock.kind,
                    observable.clock.epoch,
                    observable.clock.timestamp_ns,
                )
            )
        if len(clocks) != 1:
            raise ValueError("MIF observables must share one exact clock sample")
        clock_domain, clock_kind, clock_epoch, _ = next(iter(clocks))
        if clock_kind is not ClockKind.SIMULATION_MONOTONIC:
            raise ValueError("MIF v1 requires a simulation-monotonic clock")
        if len(self.semantics) != len(self.observables):
            raise ValueError("MIF handoff requires one semantic per observable")
        semantic_observables: list[str] = []
        semantic_ids: set[str] = set()
        for semantic in self.semantics:
            if semantic.phase_id in semantic_ids:
                raise ValueError("MIF semantic identifiers must be unique")
            semantic_ids.add(semantic.phase_id)
            if semantic.reactor_context_id != self.context.context_id:
                raise ValueError("MIF semantic context must match the handoff")
            if len(semantic.observable_ids) != 1:
                raise ValueError("each MIF semantic must name one observable")
            semantic_observables.extend(semantic.observable_ids)
            if (
                semantic.clock_domain,
                semantic.clock_kind,
                semantic.clock_epoch,
            ) != (clock_domain, clock_kind, clock_epoch):
                raise ValueError("MIF semantic and observable clocks must match")
            if semantic.carrier_type is SemanticCarrier.NUMERICAL_PHASE:
                _validate_numerical_semantic(semantic)
            elif semantic.carrier_type in {
                SemanticCarrier.BOUNDED_FEATURE,
                SemanticCarrier.CATEGORICAL_STATE,
            }:
                _validate_nonphase_semantic(semantic)
            else:
                raise ValueError(
                    "MIF v1 contains only numerical phase or nonphase data"
                )
        if set(semantic_observables) != set(observable_ids):
            raise ValueError("MIF semantics must cover every observable exactly once")
        if self.phase_relations:
            raise ValueError("MIF v1 cannot assert phase relations")
        if self.regime.reactor_context_id != self.context.context_id:
            raise ValueError("MIF regime context must match the handoff")
        if (
            self.regime.state is not RegimeState.UNKNOWN
            or self.regime.confidence != 0.0
        ):
            raise ValueError("MIF v1 regime must remain UNKNOWN with zero confidence")
        if self.regime.action_owner != ACTION_OWNER:
            raise ValueError("MIF regime action owner must remain SCPN-CONTROL")
        if self.regime.authority != REVIEW_ONLY_AUTHORITY:
            raise ValueError("MIF regime authority must remain review_only")
        if set(self.regime.evidence_ids) != set(observable_ids):
            raise ValueError("MIF regime evidence must cover every observable")


@dataclass(frozen=True, slots=True)
class _ObservableSpec:
    """Static metadata and value for one projected MIF observable."""

    observable_id: str
    channel: str
    physical_quantity: str
    units: str
    spatial_support: str
    value: JsonValue
    carrier: SemanticCarrier


def mif_merge_compression_handoff_from_mif_bytes(
    source_envelope: bytes,
    *,
    expected_sha256: str | None = None,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> MIFMergeCompressionHandoff:
    """Validate canonical MIF bytes and assign strict U0 semantic carriers.

    Parameters
    ----------
    source_envelope : bytes
        Canonical producer envelope to validate and project.
    expected_sha256 : str | None
        Optional expected SHA-256 digest of ``source_envelope``.
    registry : ReactorConfigurationRegistry
        Reactor registry against which the projected context is validated.

    Returns
    -------
    MIFMergeCompressionHandoff
        Validated review-only merge-and-compression handoff.
    """
    source, source_json, source_digest = _decode_source(
        source_envelope, expected_sha256=expected_sha256
    )
    body = cast(Mapping[str, object], source["payload"])
    reactor = cast(Mapping[str, object], body["reactor"])
    clock_raw = cast(Mapping[str, object], body["clock"])
    evidence = cast(Mapping[str, object], body["evidence"])
    kinematics = cast(Mapping[str, object], body["kinematics"])
    merge_window = cast(Mapping[str, object], body["merge_window"])
    trigger = cast(Mapping[str, object], body["trigger"])
    event_id = cast(str, source["event_id"])
    source_revision = cast(str, source["source_revision"])
    configuration = registry.resolve(_CONFIGURATION)
    context = ReactorContext(
        context_id=f"spo.mif.frc_compression.{source_digest[:24]}",
        configuration=configuration.identifier,
        confinement_family=configuration.confinement_family,
        topology=configuration.topology,
        coordinate_frame=cast(str, reactor["coordinate_frame"]),
        drivers=tuple(DriverKind(item) for item in cast(list[str], reactor["drivers"])),
        cadence=OperatingCadence.PULSED_SHOT,
        reaction=ReactionKind(cast(str, reactor["reaction"])),
        conversion=ConversionKind(cast(str, reactor["conversion"])),
        facility=cast(str, reactor["facility"]),
        event_id=event_id,
        configuration_version="1.0.0",
        operating_point={
            "merge_candidate_lock": cast(bool, merge_window["candidate_lock"]),
            "merge_lock_achieved": cast(bool, merge_window["lock_achieved"]),
            "model_phase_count": len(cast(list[str], kinematics["phases_rad"])),
            "trigger_decision": cast(str, trigger["decision"]),
        },
        evidence_class=EvidenceClass.SIMULATION,
        registry_version=registry.version,
        registry_digest=registry.digest,
    ).validate_registry(registry)
    clock = ClockReference(
        domain=cast(str, clock_raw["domain"]),
        kind=ClockKind.SIMULATION_MONOTONIC,
        epoch=cast(str, clock_raw["epoch"]),
        timestamp_ns=cast(int, clock_raw["timestamp_ns"]),
        sample_rate_hz=float(cast(str, clock_raw["sample_rate_hz"])),
        latency_s=float(cast(str, clock_raw["latency_s"])),
        picosecond_offset=cast(int, clock_raw["picosecond_offset"]),
        synchronized_to=None,
    )
    specs = _observable_specs(body)
    observables = tuple(
        _build_observable(
            spec,
            context=context,
            clock=clock,
            source=source,
            evidence=evidence,
            source_digest=source_digest,
        )
        for spec in specs
    )
    semantics = tuple(
        _build_semantic(item, spec=spec)
        for item, spec in zip(observables, specs, strict=True)
    )
    regime = RegimeEstimate(
        regime_id="spo.mif.merge_compression.regime.unknown",
        reactor_context_id=context.context_id,
        axes=(
            RegimeAxis("merge_lock", "unclassified", 0.0),
            RegimeAxis("trigger_state", "unclassified", 0.0),
        ),
        state=RegimeState.UNKNOWN,
        evidence_ids=tuple(item.observable_id for item in observables),
        classifier="spo.mif.merge_compression.no_classifier",
        classifier_version="1.0.0",
        threshold_provenance=("producer_has_no_versioned_regime_classifier",),
        confidence=0.0,
        hysteresis=0.0,
        dwell_time_s=0.0,
        transition_reason=(
            "MIF source supplies merge and trigger facts, not a reactor regime "
            "classifier"
        ),
        safety_effect="review only; no control consequence",
        validity=ValidityWindow(
            ValidityState.UNKNOWN,
            valid_from_ns=clock.timestamp_ns,
            valid_until_ns=clock.timestamp_ns,
            reasons=("no versioned MIF regime classifier",),
        ),
    )
    return MIFMergeCompressionHandoff(
        source_revision=source_revision,
        source_envelope_json=source_json,
        event_id=event_id,
        context=context,
        observables=observables,
        semantics=semantics,
        regime=regime,
    )


def mif_merge_compression_handoff_to_record(
    handoff: MIFMergeCompressionHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> dict[str, object]:
    """Return the digest-sealed portable MIF handoff record.

    Parameters
    ----------
    handoff : MIFMergeCompressionHandoff
        Validated handoff to encode as a record.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate and encode nested contracts.

    Returns
    -------
    dict[str, object]
        Portable envelope containing the payload and its SHA-256 digest.
    """
    handoff.context.validate_registry(registry)
    payload: dict[str, object] = {
        "actionable": handoff.actionable,
        "authority": handoff.authority,
        "event_id": handoff.event_id,
        "observables": [
            contract_to_record(item, registry=registry) for item in handoff.observables
        ],
        "phase_relations": [],
        "reactor_context": contract_to_record(handoff.context, registry=registry),
        "regime": contract_to_record(handoff.regime, registry=registry),
        "registry_digest": registry.digest,
        "registry_version": registry.version,
        "semantics": [
            contract_to_record(item, registry=registry) for item in handoff.semantics
        ],
        "source_envelope_json": handoff.source_envelope_json,
        "source_envelope_sha256": handoff.source_envelope_sha256,
        "source_project": handoff.source_project,
        "source_revision": handoff.source_revision,
        "source_schema": handoff.source_schema,
        "u0_schema_version": U0_SCHEMA_VERSION,
    }
    return {
        "payload": payload,
        "payload_sha256": _digest_record(payload),
        "schema": handoff.schema,
        "schema_version": handoff.schema_version,
    }


def mif_merge_compression_handoff_from_record(
    raw: object,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> MIFMergeCompressionHandoff:
    """Decode a strict MIF handoff record and verify its digest chain.

    Parameters
    ----------
    raw : object
        Candidate portable MIF handoff record.
    registry : ReactorConfigurationRegistry
        Reactor registry required by the encoded identity binding.

    Returns
    -------
    MIFMergeCompressionHandoff
        Validated handoff reconstructed from the record.

    Raises
    ------
    ValueError
        If schema, version, digest, registry, U0, or contract invariants fail.
    """
    envelope = require_exact_keys(
        raw,
        required=frozenset({"payload", "payload_sha256", "schema", "schema_version"}),
        field="MIF handoff envelope",
    )
    if envelope["schema"] != MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA:
        raise ValueError("unsupported MIF merge-compression handoff schema")
    if envelope["schema_version"] != MIF_MERGE_COMPRESSION_HANDOFF_VERSION:
        raise ValueError("unsupported MIF merge-compression handoff version")
    payload = require_exact_keys(
        envelope["payload"], required=_HANDOFF_PAYLOAD_KEYS, field="MIF handoff payload"
    )
    if require_sha256(
        envelope["payload_sha256"], field="payload_sha256"
    ) != _digest_record(payload):
        raise ValueError("MIF handoff payload digest mismatch")
    if (
        payload["registry_version"] != registry.version
        or payload["registry_digest"] != registry.digest
    ):
        raise ValueError("MIF handoff registry identity mismatch")
    if payload["u0_schema_version"] != U0_SCHEMA_VERSION:
        raise ValueError("MIF handoff U0 schema mismatch")
    source_json = cast(str, payload["source_envelope_json"])
    source_digest = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
    if (
        require_sha256(
            payload["source_envelope_sha256"], field="source_envelope_sha256"
        )
        != source_digest
    ):
        raise ValueError("embedded MIF source envelope digest mismatch")
    observables_raw = _list(payload["observables"], "observables")
    semantics_raw = _list(payload["semantics"], "semantics")
    relations = _list(payload["phase_relations"], "phase_relations")
    if relations:
        raise ValueError("MIF v1 phase_relations must be empty")
    context = _contract_type(
        contract_from_record(payload["reactor_context"], registry=registry),
        ReactorContext,
    )
    regime = _contract_type(
        contract_from_record(payload["regime"], registry=registry), RegimeEstimate
    )
    observables = tuple(
        _contract_type(
            contract_from_record(item, registry=registry), ObservableDescriptor
        )
        for item in observables_raw
    )
    semantics = tuple(
        _contract_type(
            contract_from_record(item, registry=registry), PhaseSemanticRecord
        )
        for item in semantics_raw
    )
    return MIFMergeCompressionHandoff(
        source_revision=cast(str, payload["source_revision"]),
        source_envelope_json=source_json,
        event_id=cast(str, payload["event_id"]),
        context=context,
        observables=observables,
        semantics=semantics,
        regime=regime,
        source_schema=cast(str, payload["source_schema"]),
        source_project=cast(str, payload["source_project"]),
        authority=cast(str, payload["authority"]),
        actionable=cast(bool, payload["actionable"]),
    )


def mif_merge_compression_handoff_to_bytes(
    handoff: MIFMergeCompressionHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> bytes:
    """Serialize a MIF handoff to unique canonical UTF-8 bytes.

    Parameters
    ----------
    handoff : MIFMergeCompressionHandoff
        Validated handoff to serialize.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate and encode nested contracts.

    Returns
    -------
    bytes
        Canonical compact JSON representation of the handoff.
    """
    return _handoff_json(
        mif_merge_compression_handoff_to_record(handoff, registry=registry)
    ).encode("utf-8")


def mif_merge_compression_handoff_from_bytes(
    payload: bytes,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> MIFMergeCompressionHandoff:
    """Decode only the unique canonical MIF handoff byte representation.

    Parameters
    ----------
    payload : bytes
        Candidate canonical MIF handoff bytes.
    registry : ReactorConfigurationRegistry
        Reactor registry required by the encoded identity binding.

    Returns
    -------
    MIFMergeCompressionHandoff
        Validated handoff reconstructed from the bytes.

    Raises
    ------
    ValueError
        If bytes are empty, oversized, malformed, noncanonical, or invalid.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("MIF handoff must be non-empty bytes")
    if len(payload) > MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES:
        raise ValueError("MIF handoff exceeds the maximum byte size")
    try:
        text = payload.decode("utf-8", errors="strict")
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MIF handoff must be strict UTF-8 JSON") from exc
    if _handoff_json(raw) != text:
        raise ValueError("MIF handoff must use the unique canonical JSON encoding")
    return mif_merge_compression_handoff_from_record(raw, registry=registry)


def mif_merge_compression_handoff_digest(
    handoff: MIFMergeCompressionHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> str:
    """Return SHA-256 of the canonical MIF handoff bytes.

    Parameters
    ----------
    handoff : MIFMergeCompressionHandoff
        Validated handoff whose canonical bytes are hashed.
    registry : ReactorConfigurationRegistry
        Reactor registry used when producing the canonical bytes.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(
        mif_merge_compression_handoff_to_bytes(handoff, registry=registry)
    ).hexdigest()


def _observable_specs(body: Mapping[str, object]) -> tuple[_ObservableSpec, ...]:
    """Enumerate phase, kinematic, merge, and trigger observables."""
    kinematics = cast(Mapping[str, object], body["kinematics"])
    merge = cast(Mapping[str, object], body["merge_window"])
    trigger = cast(Mapping[str, object], body["trigger"])
    specs: list[_ObservableSpec] = []
    for index, value in enumerate(cast(list[str], kinematics["phases_rad"])):
        specs.append(
            _ObservableSpec(
                f"mif.merge_compression.phase.{index}",
                f"kinematics.phases_rad[{index}]",
                "model oscillator phase",
                "rad",
                f"oscillator:{index}",
                value,
                SemanticCarrier.NUMERICAL_PHASE,
            )
        )
    for name, quantity, unit in (
        ("positions_m", "model axial position", "m"),
        ("velocities_m_s", "model axial velocity", "m/s"),
    ):
        for index, value in enumerate(cast(list[str], kinematics[name])):
            specs.append(
                _ObservableSpec(
                    f"mif.merge_compression.{name}.{index}",
                    f"kinematics.{name}[{index}]",
                    quantity,
                    unit,
                    f"oscillator:{index}",
                    value,
                    SemanticCarrier.BOUNDED_FEATURE,
                )
            )
    fixed = (
        (
            "kinematics",
            "phase_lock_error_rad",
            "maximum circular phase separation",
            "rad",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "kinematics",
            "order_parameter",
            "Kuramoto coherence order parameter",
            "1",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "kinematics",
            "reference_point_m",
            "model axial reference point",
            "m",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "kinematics",
            "reference_error_m",
            "maximum axial reference error",
            "m",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "kinematics",
            "separation_m",
            "model axial separation",
            "m",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "kinematics",
            "local_error_estimate",
            "mixed numerical integrator error estimate",
            "1",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "merge_window",
            "phase_tolerance_rad",
            "merge classifier phase tolerance",
            "rad",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "merge_window",
            "spatial_tolerance_m",
            "merge classifier spatial tolerance",
            "m",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "merge_window",
            "consecutive_samples",
            "merge classifier required streak",
            "sample",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "merge_window",
            "streak",
            "merge classifier current streak",
            "sample",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "merge_window",
            "candidate_lock",
            "merge candidate state",
            "1",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "merge_window",
            "lock_achieved",
            "merge lock state",
            "1",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "trigger",
            "armed",
            "software trigger armed state",
            "1",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "trigger",
            "bank_feasible",
            "software bank feasibility state",
            "1",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "trigger",
            "decision",
            "software trigger decision",
            "1",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "trigger",
            "safety_slack_m",
            "sampled kinematic safety slack",
            "m",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "trigger",
            "sample_index",
            "software trigger sample index",
            "sample",
            SemanticCarrier.BOUNDED_FEATURE,
        ),
        (
            "trigger",
            "first_fire_timestamp_ns",
            "software first-fire event timestamp",
            "ns",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
        (
            "trigger",
            "first_violation_index",
            "software first-violation sample index",
            "sample",
            SemanticCarrier.CATEGORICAL_STATE,
        ),
    )
    containers = {"kinematics": kinematics, "merge_window": merge, "trigger": trigger}
    for group, name, quantity, unit, carrier in fixed:
        specs.append(
            _ObservableSpec(
                f"mif.merge_compression.{group}.{name}",
                f"{group}.{name}",
                quantity,
                unit,
                "event_sample",
                cast(JsonValue, containers[group][name]),
                carrier,
            )
        )
    return tuple(specs)


def _build_observable(
    spec: _ObservableSpec,
    *,
    context: ReactorContext,
    clock: ClockReference,
    source: Mapping[str, object],
    evidence: Mapping[str, object],
    source_digest: str,
) -> ObservableDescriptor:
    """Build an evidence-bearing observable from one validated MIF atom."""
    quality_name = cast(str, evidence["quality"])
    quality_state = QualityState(quality_name)
    flags = tuple(cast(list[str], evidence["quality_flags"]))
    validity_state = (
        ValidityState.VALID
        if quality_state is QualityState.VALID
        else ValidityState.DEGRADED
        if quality_state is QualityState.DEGRADED
        else ValidityState.INVALID
    )
    validity_reasons = () if validity_state is ValidityState.VALID else tuple(flags)
    payload_digest = cast(str, source["payload_sha256"])
    input_digests = ",".join(cast(list[str], evidence["input_sha256"]))
    return ObservableDescriptor(
        observable_id=spec.observable_id,
        reactor_context=context,
        physical_quantity=spec.physical_quantity,
        units=spec.units,
        coordinate_frame=context.coordinate_frame,
        spatial_support=spec.spatial_support,
        diagnostic="mif_merge_compression_model",
        channel=spec.channel,
        value=spec.value,
        clock=clock,
        calibration=CalibrationReference(
            calibration_id=_CALIBRATION_ID,
            transfer_function_id=_TRANSFER_ID,
            calibrated_at_ns=cast(int, evidence["calibrated_at_ns"]),
        ),
        uncertainty=Uncertainty(standard_deviation=0.0, confidence_level=0.0),
        quality=QualityAssessment(quality_state, flags=flags),
        validity=ValidityWindow(
            validity_state,
            valid_from_ns=cast(int, evidence["valid_from_ns"]),
            valid_until_ns=cast(int, evidence["valid_until_ns"]),
            reasons=validity_reasons,
        ),
        provenance=ProvenanceRecord(
            source_project=_MIF_PROJECT,
            component=cast(str, evidence["component"]),
            symbol=cast(str, evidence["symbol"]),
            artifact_uri=f"artifact:sha256:{source_digest}",
            sha256=source_digest,
            attributes=(
                ("backend", cast(str, evidence["backend"])),
                ("backend_version", cast(str, evidence["backend_version"])),
                ("event_id", cast(str, source["event_id"])),
                ("input_sha256", input_digests),
                ("payload_sha256", payload_digest),
                ("producer_revision", cast(str, source["source_revision"])),
                (
                    "uncertainty_basis",
                    "serialized_model_state_not_physical_uncertainty",
                ),
            ),
        ),
    )


def _build_semantic(
    observable: ObservableDescriptor, *, spec: _ObservableSpec
) -> PhaseSemanticRecord:
    """Assign numerical-phase or explicit nonphase semantics to an observable."""
    numerical = spec.carrier is SemanticCarrier.NUMERICAL_PHASE
    usable = numerical and observable.validity.state in {
        ValidityState.VALID,
        ValidityState.DEGRADED,
    }
    timestamp = observable.clock.timestamp_ns
    if numerical:
        validity = (
            observable.validity
            if usable
            else ValidityWindow(
                ValidityState.UNOBSERVABLE,
                timestamp,
                timestamp,
                ("producer evidence is not usable",),
            )
        )
        quality = (
            observable.quality
            if usable
            else QualityAssessment(
                QualityState.UNKNOWN, ("unusable_numerical_model_state",)
            )
        )
        phase = float(cast(str, observable.value)) if usable else None
    else:
        validity = ValidityWindow(
            ValidityState.UNOBSERVABLE,
            timestamp,
            timestamp,
            ("producer atom is not an angular phase",),
        )
        quality = QualityAssessment(QualityState.UNKNOWN, ("nonphase_mif_evidence",))
        phase = None
    return PhaseSemanticRecord(
        phase_id=f"spo.{observable.observable_id}.{spec.carrier.value}",
        reactor_context_id=observable.reactor_context.context_id,
        observable_ids=(observable.observable_id,),
        carrier_type=spec.carrier,
        phenomenon=observable.physical_quantity,
        phase_rad=phase,
        amplitude=None,
        frequency_hz=None,
        bandwidth_hz=None,
        mode_identity=None,
        mode_harmonic=None,
        phase_origin="event_start" if numerical else None,
        orientation="positive_model_evolution" if numerical else None,
        reference_frame=observable.coordinate_frame,
        clock_domain=observable.clock.domain,
        clock_kind=observable.clock.kind,
        clock_epoch=observable.clock.epoch,
        wrap_convention="[0,2pi)" if numerical else None,
        reference_signal="model_phase_origin_at_event_start" if numerical else None,
        extractor="spo.mif.merge_compression.identity_model_phase"
        if numerical
        else "spo.mif.merge_compression.nonphase_classification",
        extractor_version="1.0.0",
        observation_operator="identity_on_serialized_model_phase"
        if numerical
        else None,
        uncertainty=Uncertainty(0.0, 0.0, circular_std_rad=0.0 if numerical else None),
        confidence=1.0 if usable else 0.0,
        observability=1.0 if usable else 0.0,
        observability_threshold=1.0,
        validity=validity,
        quality=quality,
        evidence_class=EvidenceClass.SIMULATION,
    )


def _validate_numerical_semantic(semantic: PhaseSemanticRecord) -> None:
    """Require the fixed simulation evidence and numerical phase references."""
    if semantic.evidence_class is not EvidenceClass.SIMULATION:
        raise ValueError("MIF numerical phases must retain simulation evidence")
    required = (
        semantic.phase_origin == "event_start",
        semantic.orientation == "positive_model_evolution",
        semantic.wrap_convention == "[0,2pi)",
        semantic.reference_signal == "model_phase_origin_at_event_start",
        semantic.observation_operator == "identity_on_serialized_model_phase",
    )
    if not all(required):
        raise ValueError("MIF numerical phase reference semantics drifted")


def _validate_nonphase_semantic(semantic: PhaseSemanticRecord) -> None:
    """Require nonphase evidence to omit angular claims."""
    if any(
        value is not None
        for value in (
            semantic.phase_rad,
            semantic.amplitude,
            semantic.frequency_hz,
            semantic.bandwidth_hz,
            semantic.mode_identity,
            semantic.mode_harmonic,
            semantic.phase_origin,
            semantic.orientation,
            semantic.wrap_convention,
            semantic.reference_signal,
            semantic.observation_operator,
        )
    ):
        raise ValueError("MIF nonphase evidence cannot carry phase fields")
    if semantic.observability != 0.0 or semantic.confidence != 0.0:
        raise ValueError("MIF nonphase evidence has zero phase observability")
    if semantic.validity.state is not ValidityState.UNOBSERVABLE:
        raise ValueError("MIF nonphase semantics must be UNOBSERVABLE as phase")


def _decode_source(
    payload: bytes, *, expected_sha256: str | None
) -> tuple[Mapping[str, object], str, str]:
    """Decode, authenticate, and validate a canonical MIF source envelope."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("MIF source envelope must be non-empty bytes")
    if len(payload) > MAX_SOURCE_ENVELOPE_BYTES:
        raise ValueError("MIF source envelope exceeds the maximum byte size")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != require_sha256(
        expected_sha256, field="expected_sha256"
    ):
        raise ValueError("MIF source envelope byte digest mismatch")
    try:
        text = payload.decode("utf-8", errors="strict")
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MIF source envelope must be strict UTF-8 JSON") from exc
    source = _object(raw, "MIF source envelope")
    _exact(source, _SOURCE_ROOT_KEYS, "MIF source envelope")
    _assert_jcs_safe(source)
    if _source_json(source) != text:
        raise ValueError("MIF source envelope must use unique canonical JSON bytes")
    if (
        source["schema"] != MIF_MERGE_COMPRESSION_SOURCE_SCHEMA
        or source["schema_version"] != MIF_MERGE_COMPRESSION_SOURCE_VERSION
    ):
        raise ValueError("unsupported MIF merge-compression source schema")
    if source["source_project"] != _MIF_PROJECT:
        raise ValueError("MIF source owner mismatch")
    revision = source["source_revision"]
    if not isinstance(revision, str) or _HEX_40.fullmatch(revision) is None:
        raise ValueError("invalid MIF source revision")
    require_identifier(source["event_id"], field="event_id")
    body = _object(source["payload"], "payload")
    _exact(body, _PAYLOAD_KEYS, "payload")
    if require_sha256(
        source["payload_sha256"], field="payload_sha256"
    ) != _digest_record(body, trailing_newline=True):
        raise ValueError("MIF source payload digest mismatch")
    _validate_source_payload(body)
    return source, text, digest


def _validate_source_payload(body: Mapping[str, object]) -> None:
    """Validate source authority, reactor identity, evidence, and event state."""
    authority = _object(body["authority"], "authority")
    _exact(authority, frozenset({"actionable", "review_only"}), "authority")
    if authority != {"actionable": False, "review_only": True}:
        raise ValueError("MIF source authority must be review-only")
    reactor = _object(body["reactor"], "reactor")
    _exact(
        reactor,
        frozenset(
            {
                "cadence",
                "configuration",
                "conversion",
                "coordinate_frame",
                "drivers",
                "facility",
                "reaction",
            }
        ),
        "reactor",
    )
    if (
        reactor["configuration"] != _CONFIGURATION
        or reactor["cadence"] != "pulsed_shot"
    ):
        raise ValueError("MIF source reactor profile mismatch")
    for field in ("coordinate_frame", "facility"):
        require_identifier(reactor[field], field=field)
    ReactionKind(cast(str, reactor["reaction"]))
    ConversionKind(cast(str, reactor["conversion"]))
    drivers = _strings(reactor["drivers"], "reactor.drivers")
    if (
        drivers != tuple(sorted(set(drivers)))
        or not drivers
        or set(drivers) - {"external_magnetic_coils", "pulsed_power"}
    ):
        raise ValueError("MIF source drivers are unsupported or noncanonical")
    clock = _validate_source_clock(body["clock"])
    evidence = _validate_source_evidence(
        body["evidence"], timestamp_ns=cast(int, clock["timestamp_ns"])
    )
    _validate_source_kinematics(body["kinematics"])
    _validate_source_merge_and_trigger(
        body, timestamp_ns=cast(int, clock["timestamp_ns"])
    )
    if evidence["class"] != "simulation":
        raise ValueError("MIF v1 cannot claim physical observed evidence")


def _validate_source_clock(value: object) -> Mapping[str, object]:
    """Validate the simulation clock and its exact sample cadence."""
    clock = _object(value, "clock")
    _exact(
        clock,
        frozenset(
            {
                "domain",
                "epoch",
                "kind",
                "latency_s",
                "picosecond_offset",
                "sample_period_ns",
                "sample_rate_hz",
                "synchronized_to",
                "timestamp_ns",
            }
        ),
        "clock",
    )
    if clock["kind"] != "simulation_monotonic" or clock["synchronized_to"] is not None:
        raise ValueError("MIF v1 clock profile mismatch")
    require_identifier(clock["domain"], field="clock.domain")
    require_identifier(clock["epoch"], field="clock.epoch")
    timestamp = _safe_int(clock["timestamp_ns"], "timestamp_ns", minimum=0)
    period = _safe_int(clock["sample_period_ns"], "sample_period_ns", minimum=1)
    _safe_int(clock["picosecond_offset"], "picosecond_offset", minimum=0, maximum=999)
    if _decimal(clock["latency_s"], "latency_s") < 0:
        raise ValueError("MIF latency must be non-negative")
    if _decimal(clock["sample_rate_hz"], "sample_rate_hz") != Decimal.from_float(
        1_000_000_000.0 / period
    ):
        raise ValueError("MIF sample rate does not match sample period")
    return {**clock, "timestamp_ns": timestamp}


def _validate_source_evidence(
    value: object, *, timestamp_ns: int
) -> Mapping[str, object]:
    """Validate calibration, validity, quality, and input digest evidence."""
    evidence = _object(value, "evidence")
    _exact(
        evidence,
        frozenset(
            {
                "backend",
                "backend_version",
                "calibrated_at_ns",
                "calibration_id",
                "class",
                "component",
                "input_sha256",
                "quality",
                "quality_flags",
                "symbol",
                "transfer_function_id",
                "valid_from_ns",
                "valid_until_ns",
            }
        ),
        "evidence",
    )
    for field in ("backend", "component", "symbol"):
        require_identifier(evidence[field], field=f"evidence.{field}")
    require_semver(evidence["backend_version"], field="backend_version")
    if (
        evidence["calibration_id"] != _CALIBRATION_ID
        or evidence["transfer_function_id"] != _TRANSFER_ID
    ):
        raise ValueError("MIF calibration identity drifted")
    calibrated = _safe_int(evidence["calibrated_at_ns"], "calibrated_at_ns", minimum=0)
    valid_from = _safe_int(evidence["valid_from_ns"], "valid_from_ns", minimum=0)
    valid_until = _safe_int(evidence["valid_until_ns"], "valid_until_ns", minimum=0)
    if calibrated > timestamp_ns or not valid_from <= timestamp_ns <= valid_until:
        raise ValueError("MIF source evidence is not valid at the sample")
    digests = _strings(evidence["input_sha256"], "input_sha256")
    if not digests or digests != tuple(sorted(set(digests))):
        raise ValueError("MIF input digests must be sorted and unique")
    for item in digests:
        require_sha256(item, field="input_sha256")
    flags = _strings(evidence["quality_flags"], "quality_flags")
    if flags != tuple(sorted(set(flags))):
        raise ValueError("MIF quality flags must be sorted and unique")
    quality = evidence["quality"]
    if quality not in {"valid", "degraded", "invalid"} or (
        (quality == "valid") != (not flags)
    ):
        raise ValueError("MIF source quality and flags are inconsistent")
    return evidence


def _validate_source_kinematics(value: object) -> None:
    """Validate oscillator vectors and recompute derived kinematic values."""
    kinematics = _object(value, "kinematics")
    _exact(
        kinematics,
        frozenset(
            {
                "local_error_estimate",
                "order_parameter",
                "phase_lock_error_rad",
                "phases_rad",
                "positions_m",
                "reference_error_m",
                "reference_point_m",
                "separation_m",
                "velocities_m_s",
            }
        ),
        "kinematics",
    )
    phases = _decimals(kinematics["phases_rad"], "phases_rad")
    positions = _decimals(kinematics["positions_m"], "positions_m")
    velocities = _decimals(kinematics["velocities_m_s"], "velocities_m_s")
    if (
        len(phases) < 2
        or len(positions) != len(phases)
        or len(velocities) != len(phases)
    ):
        raise ValueError("MIF kinematic vectors must have one entry per oscillator")
    if any(not 0.0 <= float(item) < 2.0 * math.pi for item in phases):
        raise ValueError("MIF numerical phases must use [0,2pi)")
    reference = _decimal(kinematics["reference_point_m"], "reference_point_m")
    nonnegative = (
        "local_error_estimate",
        "order_parameter",
        "phase_lock_error_rad",
        "reference_error_m",
        "separation_m",
    )
    values = {name: _decimal(kinematics[name], name) for name in nonnegative}
    if any(item < 0 for item in values.values()) or values["order_parameter"] > 1:
        raise ValueError("MIF derived kinematics are outside their domains")
    expected_separation = _float_decimal(
        max(map(float, positions)) - min(map(float, positions))
    )
    expected_reference = _float_decimal(
        max(abs(float(item - reference)) for item in positions)
    )
    phase_values = tuple(map(float, phases))
    expected_lock = _float_decimal(_phase_lock_error(phase_values))
    if (
        kinematics["separation_m"] != expected_separation
        or kinematics["reference_error_m"] != expected_reference
        or kinematics["phase_lock_error_rad"] != expected_lock
    ):
        raise ValueError("MIF derived kinematic values do not recompute")


def _validate_source_merge_and_trigger(
    body: Mapping[str, object], *, timestamp_ns: int
) -> None:
    """Recompute merge predicates and validate trigger prerequisites."""
    kinematics = _object(body["kinematics"], "kinematics")
    merge = _object(body["merge_window"], "merge_window")
    _exact(
        merge,
        frozenset(
            {
                "candidate_lock",
                "consecutive_samples",
                "lock_achieved",
                "phase_tolerance_rad",
                "spatial_tolerance_m",
                "streak",
            }
        ),
        "merge_window",
    )
    consecutive = _safe_int(
        merge["consecutive_samples"], "consecutive_samples", minimum=1
    )
    streak = _safe_int(merge["streak"], "streak", minimum=0)
    if not isinstance(merge["candidate_lock"], bool) or not isinstance(
        merge["lock_achieved"], bool
    ):
        raise ValueError("MIF merge decisions must be booleans")
    phase_tolerance = _decimal(merge["phase_tolerance_rad"], "phase_tolerance_rad")
    spatial_tolerance = _decimal(merge["spatial_tolerance_m"], "spatial_tolerance_m")
    if phase_tolerance <= 0 or spatial_tolerance <= 0:
        raise ValueError("MIF merge tolerances must be positive")
    candidate = (
        _decimal(kinematics["phase_lock_error_rad"], "phase_lock_error_rad")
        <= phase_tolerance
        and _decimal(kinematics["reference_error_m"], "reference_error_m")
        <= spatial_tolerance
    )
    if merge["candidate_lock"] is not candidate or merge["lock_achieved"] != (
        streak >= consecutive
    ):
        raise ValueError("MIF merge predicates are inconsistent")
    trigger = _object(body["trigger"], "trigger")
    _exact(
        trigger,
        frozenset(
            {
                "armed",
                "bank_feasible",
                "decision",
                "first_fire_timestamp_ns",
                "first_violation_index",
                "safety_slack_m",
                "sample_index",
            }
        ),
        "trigger",
    )
    if not isinstance(trigger["armed"], bool) or not isinstance(
        trigger["bank_feasible"], bool
    ):
        raise ValueError("MIF trigger gates must be booleans")
    _safe_int(trigger["sample_index"], "sample_index", minimum=0)
    _optional_int(
        trigger["first_fire_timestamp_ns"], "first_fire_timestamp_ns", minimum=0
    )
    _optional_int(trigger["first_violation_index"], "first_violation_index", minimum=0)
    decision = trigger["decision"]
    if decision not in {
        "hold_no_lock",
        "fire",
        "abort_unsafe",
        "abort_bank_infeasible",
    }:
        raise ValueError("unknown MIF trigger decision")
    safety_slack = _decimal(trigger["safety_slack_m"], "safety_slack_m")
    if decision == "fire":
        if (
            trigger["first_fire_timestamp_ns"] != timestamp_ns
            or not trigger["armed"]
            or not trigger["bank_feasible"]
            or not merge["lock_achieved"]
            or safety_slack < 0
        ):
            raise ValueError("MIF fire decision lacks its declared prerequisites")
    elif trigger["first_fire_timestamp_ns"] is not None:
        raise ValueError("non-fire MIF decision cannot carry a fire timestamp")


def _phase_lock_error(phases: tuple[float, ...]) -> float:
    """Return the maximum pairwise circular phase separation."""
    maximum = 0.0
    for index, left in enumerate(phases):
        for right in phases[index + 1 :]:
            maximum = max(
                maximum, abs(((right - left + math.pi) % (2.0 * math.pi)) - math.pi)
            )
    return maximum


def _source_json(value: object) -> str:
    """Return canonical newline-terminated JSON for a JCS-safe source value."""
    _assert_jcs_safe(value)
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _handoff_json(value: object) -> str:
    """Return canonical compact JSON for a handoff value."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest_record(value: object, *, trailing_newline: bool = False) -> str:
    """Return SHA-256 of a canonical source or handoff record."""
    text = _source_json(value) if trailing_newline else _handoff_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _float_decimal(value: float) -> str:
    """Return the exact decimal expansion of a binary float."""
    return str(Decimal.from_float(float(value)))


def _decimal(value: object, field: str) -> Decimal:
    """Parse a finite exact decimal string."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} contains an invalid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _decimals(value: object, field: str) -> tuple[Decimal, ...]:
    """Parse a non-empty list of finite exact decimal strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty decimal-string list")
    return tuple(_decimal(item, field) for item in value)


def _safe_int(
    value: object,
    field: str,
    *,
    minimum: int = -_MAX_SAFE_INTEGER,
    maximum: int = _MAX_SAFE_INTEGER,
) -> int:
    """Return a non-boolean integer within the declared JCS-safe bounds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field} must be a JCS-safe integer")
    return value


def _optional_int(value: object, field: str, *, minimum: int) -> int | None:
    """Validate an optional JCS-safe integer with a lower bound."""
    return None if value is None else _safe_int(value, field, minimum=minimum)


def _assert_jcs_safe(value: object, *, path: str = "$") -> None:
    """Recursively reject floats and out-of-range integers before encoding."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, float):
        raise ValueError(f"JSON floats are forbidden at {path}")
    if isinstance(value, int):
        _safe_int(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_jcs_safe(item, path=f"{path}[{index}]")
        return
    mapping = _object(value, path)
    for key, item in mapping.items():
        _assert_jcs_safe(item, path=f"{path}.{key}")


def _object(value: object, field: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the value."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, field: str) -> list[object]:
    """Return a list or reject the value."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    """Return a tuple converted from a list containing only strings."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _exact(value: Mapping[str, object], keys: frozenset[str], field: str) -> None:
    """Require an exact object field set."""
    if set(value) != keys:
        raise ValueError(f"{field} fields differ")


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _contract_type(value: object, expected: type[_T]) -> _T:
    """Return a contract of the expected runtime type."""
    if not isinstance(value, expected):
        raise ValueError(f"expected {expected.__name__} contract")
    return value


__all__ = [
    "MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES",
    "MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA",
    "MIF_MERGE_COMPRESSION_HANDOFF_VERSION",
    "MIF_MERGE_COMPRESSION_SOURCE_SCHEMA",
    "MIF_MERGE_COMPRESSION_SOURCE_VERSION",
    "MIFMergeCompressionHandoff",
    "mif_merge_compression_handoff_digest",
    "mif_merge_compression_handoff_from_bytes",
    "mif_merge_compression_handoff_from_mif_bytes",
    "mif_merge_compression_handoff_from_record",
    "mif_merge_compression_handoff_to_bytes",
    "mif_merge_compression_handoff_to_record",
]
