# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Review-only reactor semantic handoff

"""Digest-sealed, non-actuating handoff of reactor semantic evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar, cast

from .contracts import (
    ObservableDescriptor,
    PhaseRelation,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeEstimate,
)
from .registry import (
    DEFAULT_REACTOR_REGISTRY,
    ReactorConfigurationRegistry,
    resolve_reactor_registry_release,
)
from .serialization import contract_from_record, contract_to_record
from .vocabulary import (
    ACTION_OWNER,
    REVIEW_ONLY_AUTHORITY,
    U0_SCHEMA_VERSION,
    ClockKind,
    QualityState,
    RegimeState,
    SemanticCarrier,
    ValidityState,
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
)

HANDOFF_SCHEMA = "scpn-phase-orchestrator.reactor-semantic-handoff.v1"
HANDOFF_SCHEMA_VERSION = "1.0.0"
MAX_SOURCE_ENVELOPE_BYTES = 64 * 1024 * 1024
MAX_HANDOFF_JSON_BYTES = 128 * 1024 * 1024
_FUSION_OWNER = "SCPN-FUSION-CORE"
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ContractType = TypeVar("_ContractType")


@dataclass(frozen=True, slots=True)
class ReactorSemanticHandoff:
    """One immutable review bundle spanning producer and U0 evidence.

    Parameters
    ----------
    source_schema : str
        Exact owner-allocated FUSION producer schema identifier.
    source_revision : str
        Exact 40-character FUSION Git revision.
    source_envelope_json : str
        Byte-canonical producer envelope. The handoff embeds these bytes so a
        downstream consumer can independently recompute the provenance digest.
    event_id : str
        Opaque identity supplied to the producer for this simulation event.
    context : ReactorContext
        Registry-validated U0 context.
    observables : tuple[ObservableDescriptor, ...]
        Declared transport profiles and budgets.
    semantics : tuple[PhaseSemanticRecord, ...]
        Nonphase bounded-feature interpretations of the observables.
    phase_relations : tuple[PhaseRelation, ...]
        Empty for the coupled-transport exchange.
    regime : RegimeEstimate
        UNKNOWN, review-only regime result.
    """

    source_schema: str
    source_revision: str
    source_envelope_json: str
    event_id: str
    context: ReactorContext
    observables: tuple[ObservableDescriptor, ...]
    semantics: tuple[PhaseSemanticRecord, ...]
    phase_relations: tuple[PhaseRelation, ...]
    regime: RegimeEstimate
    source_project: str = _FUSION_OWNER
    authority: str = REVIEW_ONLY_AUTHORITY
    actionable: bool = False
    schema: str = HANDOFF_SCHEMA
    schema_version: str = HANDOFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != HANDOFF_SCHEMA:
            raise ValueError("unsupported reactor semantic handoff schema")
        if require_semver(self.schema_version, field="handoff schema_version") != (
            HANDOFF_SCHEMA_VERSION
        ):
            raise ValueError("unsupported reactor semantic handoff version")
        if self.source_project != _FUSION_OWNER:
            raise ValueError("coupled-transport source owner must be SCPN-FUSION-CORE")
        object.__setattr__(
            self,
            "source_schema",
            require_identifier(self.source_schema, field="source_schema"),
        )
        if not self.source_schema.startswith("scpn-fusion-core."):
            raise ValueError("source_schema must be allocated by SCPN-FUSION-CORE")
        revision = require_text(self.source_revision, field="source_revision")
        if _GIT_REVISION_RE.fullmatch(revision) is None:
            raise ValueError("source_revision must be a 40-character Git revision")
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(
            self,
            "event_id",
            require_identifier(self.event_id, field="event_id"),
        )
        _validate_canonical_source_json(self.source_envelope_json)
        _validate_source_crosslinks(
            self.source_envelope_json,
            source_schema=self.source_schema,
            source_revision=self.source_revision,
            event_id=self.event_id,
        )
        if self.authority != REVIEW_ONLY_AUTHORITY:
            raise ValueError("reactor semantic handoff authority must be review_only")
        if self.actionable is not False:
            raise ValueError("reactor semantic handoff must not be actionable")
        self._validate_contract_graph()

    @property
    def source_envelope_sha256(self) -> str:
        """Return the SHA-256 digest of the embedded producer bytes."""
        return hashlib.sha256(self.source_envelope_json.encode("utf-8")).hexdigest()

    def _validate_contract_graph(self) -> None:
        """Enforce ownership, identity, clock, and nonphase invariants."""
        registry = resolve_reactor_registry_release(
            self.context.registry_version,
            self.context.registry_digest,
        )
        self.context.validate_registry(registry)
        if self.context.event_id != self.event_id:
            raise ValueError("handoff and reactor context event_id must match")
        if not self.observables:
            raise ValueError("reactor semantic handoff requires observables")
        observable_ids = tuple(item.observable_id for item in self.observables)
        if len(set(observable_ids)) != len(observable_ids):
            raise ValueError("handoff observable_ids must be unique")
        clocks: set[tuple[str, ClockKind, str]] = set()
        for observable in self.observables:
            if observable.reactor_context != self.context:
                raise ValueError("handoff observable context must match the handoff")
            if observable.provenance.source_project != _FUSION_OWNER:
                raise ValueError("handoff observables must retain FUSION provenance")
            clocks.add(
                (
                    observable.clock.domain,
                    observable.clock.kind,
                    observable.clock.epoch,
                )
            )
        if len(clocks) != 1:
            raise ValueError(
                "handoff observables must share one clock domain and epoch"
            )
        clock_domain, clock_kind, clock_epoch = next(iter(clocks))
        if clock_kind is not ClockKind.SIMULATION_MONOTONIC:
            raise ValueError("coupled transport requires a simulation-monotonic clock")

        if len(self.semantics) != len(self.observables):
            raise ValueError("handoff requires one semantic record per observable")
        semantic_ids = tuple(item.phase_id for item in self.semantics)
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("handoff semantic identifiers must be unique")
        semantic_observables: list[str] = []
        for semantic in self.semantics:
            if semantic.reactor_context_id != self.context.context_id:
                raise ValueError("semantic context must match the handoff")
            if len(semantic.observable_ids) != 1:
                raise ValueError("each bounded feature must name one observable")
            semantic_observables.extend(semantic.observable_ids)
            if semantic.carrier_type is not SemanticCarrier.BOUNDED_FEATURE:
                raise ValueError("coupled transport permits only bounded_feature")
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
                raise ValueError("bounded transport evidence cannot carry phase fields")
            if semantic.clock_domain != clock_domain:
                raise ValueError("semantic and observable clock domains must match")
            if semantic.clock_kind is not clock_kind:
                raise ValueError("semantic and observable clock kinds must match")
            if semantic.clock_epoch != clock_epoch:
                raise ValueError("semantic and observable clock epochs must match")
            if semantic.observability != 0.0 or semantic.confidence != 0.0:
                raise ValueError("noncyclic transport has zero phase observability")
            if semantic.validity.state is not ValidityState.UNOBSERVABLE:
                raise ValueError("noncyclic transport semantics must be unobservable")
            if semantic.quality.state is not QualityState.UNKNOWN or (
                "noncyclic_transport_evidence" not in semantic.quality.flags
            ):
                raise ValueError(
                    "noncyclic transport requires explicit unknown phase quality"
                )
        if set(semantic_observables) != set(observable_ids):
            raise ValueError(
                "semantic records must cover every observable exactly once"
            )
        if self.phase_relations:
            raise ValueError("coupled transport handoff cannot contain phase relations")
        if self.regime.reactor_context_id != self.context.context_id:
            raise ValueError("regime context must match the handoff")
        if self.regime.state is not RegimeState.UNKNOWN:
            raise ValueError("coupled transport cannot infer a reactor regime")
        if self.regime.confidence != 0.0:
            raise ValueError("UNKNOWN coupled-transport regime confidence must be zero")
        if self.regime.action_owner != ACTION_OWNER:
            raise ValueError("regime action owner must remain SCPN-CONTROL")
        if self.regime.authority != REVIEW_ONLY_AUTHORITY:
            raise ValueError("regime authority must remain review_only")
        if set(self.regime.evidence_ids) != set(observable_ids):
            raise ValueError("regime evidence_ids must cover the handoff observables")


def handoff_to_record(
    handoff: ReactorSemanticHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> dict[str, object]:
    """Return the digest-sealed portable handoff record.

    Parameters
    ----------
    handoff : ReactorSemanticHandoff
        Validated semantic handoff to serialize.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate embedded contracts.

    Returns
    -------
    dict[str, object]
        Envelope containing the handoff payload and payload digest.
    """
    handoff.context.validate_registry(registry)
    payload: dict[str, object] = {
        "actionable": handoff.actionable,
        "authority": handoff.authority,
        "event_id": handoff.event_id,
        "observables": [
            contract_to_record(item, registry=registry) for item in handoff.observables
        ],
        "phase_relations": [
            contract_to_record(item, registry=registry)
            for item in handoff.phase_relations
        ],
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
        "payload_sha256": _canonical_digest(payload),
        "schema": handoff.schema,
        "schema_version": handoff.schema_version,
    }


def handoff_from_record(
    raw: object,
    *,
    registry: ReactorConfigurationRegistry | None = None,
) -> ReactorSemanticHandoff:
    """Decode a strict handoff record and verify its complete digest chain.

    Parameters
    ----------
    raw : object
        Candidate serialized handoff envelope.
    registry : ReactorConfigurationRegistry or None
        Explicit reactor registry required by embedded U0 contracts. When
        omitted, resolve only the exact allowlisted release declared by the
        digest-sealed payload.

    Returns
    -------
    ReactorSemanticHandoff
        Validated review-only semantic handoff.

    Raises
    ------
    ValueError
        If schema, digest, registry, source, or contract-graph checks fail.
    """
    envelope = require_exact_keys(
        raw,
        required=frozenset({"payload", "payload_sha256", "schema", "schema_version"}),
        field="reactor semantic handoff envelope",
    )
    if envelope["schema"] != HANDOFF_SCHEMA:
        raise ValueError("unsupported reactor semantic handoff schema")
    if envelope["schema_version"] != HANDOFF_SCHEMA_VERSION:
        raise ValueError("unsupported reactor semantic handoff version")
    payload = require_exact_keys(
        envelope["payload"],
        required=frozenset(
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
        ),
        field="reactor semantic handoff payload",
    )
    supplied_payload_digest = require_sha256(
        envelope["payload_sha256"], field="payload_sha256"
    )
    if supplied_payload_digest != _canonical_digest(payload):
        raise ValueError("reactor semantic handoff payload digest mismatch")
    if registry is None:
        registry = resolve_reactor_registry_release(
            cast(str, payload["registry_version"]),
            cast(str, payload["registry_digest"]),
        )
    if payload["registry_version"] != registry.version:
        raise ValueError("reactor semantic handoff registry version mismatch")
    if payload["registry_digest"] != registry.digest:
        raise ValueError("reactor semantic handoff registry digest mismatch")
    if payload["u0_schema_version"] != U0_SCHEMA_VERSION:
        raise ValueError("reactor semantic handoff U0 schema mismatch")
    source_json = cast(str, payload["source_envelope_json"])
    _validate_canonical_source_json(source_json)
    source_digest = require_sha256(
        payload["source_envelope_sha256"], field="source_envelope_sha256"
    )
    if source_digest != hashlib.sha256(source_json.encode("utf-8")).hexdigest():
        raise ValueError("embedded FUSION source envelope digest mismatch")

    observables_raw = _require_list(payload["observables"], field="observables")
    semantics_raw = _require_list(payload["semantics"], field="semantics")
    relations_raw = _require_list(payload["phase_relations"], field="phase_relations")
    context = contract_from_record(payload["reactor_context"], registry=registry)
    regime = contract_from_record(payload["regime"], registry=registry)
    observables = tuple(
        _require_contract_type(
            contract_from_record(item, registry=registry), ObservableDescriptor
        )
        for item in observables_raw
    )
    semantics = tuple(
        _require_contract_type(
            contract_from_record(item, registry=registry), PhaseSemanticRecord
        )
        for item in semantics_raw
    )
    relations = tuple(
        _require_contract_type(
            contract_from_record(item, registry=registry), PhaseRelation
        )
        for item in relations_raw
    )
    return ReactorSemanticHandoff(
        source_schema=cast(str, payload["source_schema"]),
        source_revision=cast(str, payload["source_revision"]),
        source_envelope_json=source_json,
        event_id=cast(str, payload["event_id"]),
        context=_require_contract_type(context, ReactorContext),
        observables=observables,
        semantics=semantics,
        phase_relations=relations,
        regime=_require_contract_type(regime, RegimeEstimate),
        source_project=cast(str, payload["source_project"]),
        authority=cast(str, payload["authority"]),
        actionable=cast(bool, payload["actionable"]),
        schema=envelope["schema"],
        schema_version=envelope["schema_version"],
    )


def handoff_to_json(
    handoff: ReactorSemanticHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> str:
    """Serialize a handoff to byte-stable canonical JSON.

    Parameters
    ----------
    handoff : ReactorSemanticHandoff
        Validated semantic handoff to serialize.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate embedded contracts.

    Returns
    -------
    str
        Canonical compact JSON text.
    """
    return _canonical_json(handoff_to_record(handoff, registry=registry))


def handoff_to_bytes(
    handoff: ReactorSemanticHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> bytes:
    """Serialize a handoff to its unique canonical UTF-8 bytes.

    Parameters
    ----------
    handoff : ReactorSemanticHandoff
        Validated semantic handoff to encode.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate embedded contracts.

    Returns
    -------
    bytes
        Canonical compact JSON bytes.
    """
    return handoff_to_json(handoff, registry=registry).encode("utf-8")


def handoff_from_json(
    payload: str,
    *,
    registry: ReactorConfigurationRegistry | None = None,
) -> ReactorSemanticHandoff:
    """Deserialize handoff JSON while refusing duplicate object keys.

    Parameters
    ----------
    payload : str
        Candidate JSON handoff text.
    registry : ReactorConfigurationRegistry or None
        Explicit reactor registry required by embedded U0 contracts. When
        omitted, resolve only the exact allowlisted release declared by the
        digest-sealed payload.

    Returns
    -------
    ReactorSemanticHandoff
        Validated review-only semantic handoff.

    Raises
    ------
    ValueError
        If JSON is empty, oversized, malformed, duplicated, or semantically invalid.
    """
    if not isinstance(payload, str) or not payload:
        raise ValueError("handoff JSON must be a non-empty string")
    if len(payload.encode("utf-8")) > MAX_HANDOFF_JSON_BYTES:
        raise ValueError("handoff JSON exceeds the maximum byte size")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("handoff JSON is invalid") from exc
    return handoff_from_record(raw, registry=registry)


def handoff_from_bytes(
    payload: bytes,
    *,
    registry: ReactorConfigurationRegistry | None = None,
) -> ReactorSemanticHandoff:
    """Decode the unique canonical UTF-8 handoff representation.

    This is the cross-project admission surface. Unlike ``handoff_from_json``,
    it rejects alternate whitespace, key ordering, a trailing newline, and any
    other byte representation of the same JSON value.

    Parameters
    ----------
    payload : bytes
        Candidate canonical handoff bytes.
    registry : ReactorConfigurationRegistry or None
        Explicit reactor registry required by embedded U0 contracts. When
        omitted, resolve only the exact allowlisted release declared by the
        digest-sealed payload.

    Returns
    -------
    ReactorSemanticHandoff
        Validated handoff reconstructed from the canonical bytes.

    Raises
    ------
    ValueError
        If the input is empty, oversized, malformed, or not canonical JSON.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("handoff bytes must be non-empty bytes")
    if len(payload) > MAX_HANDOFF_JSON_BYTES:
        raise ValueError("handoff bytes exceed the maximum byte size")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("handoff bytes must be strict UTF-8") from exc
    handoff = handoff_from_json(text, registry=registry)
    resolved_registry = registry or resolve_reactor_registry_release(
        handoff.context.registry_version,
        handoff.context.registry_digest,
    )
    if handoff_to_bytes(handoff, registry=resolved_registry) != payload:
        raise ValueError("handoff bytes must use canonical JSON")
    return handoff


def handoff_digest(
    handoff: ReactorSemanticHandoff,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> str:
    """Return the SHA-256 digest of canonical handoff JSON.

    Parameters
    ----------
    handoff : ReactorSemanticHandoff
        Validated handoff whose canonical bytes are hashed.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate embedded contracts.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(handoff_to_bytes(handoff, registry=registry)).hexdigest()


def canonicalize_source_envelope(payload: str) -> str:
    """Canonicalize strict producer JSON for embedding in a handoff.

    Parameters
    ----------
    payload : str
        Candidate producer-envelope JSON text.

    Returns
    -------
    str
        Canonical compact JSON object text.

    Raises
    ------
    ValueError
        If the input is empty, oversized, malformed, duplicated, or not an object.
    """
    if not isinstance(payload, str) or not payload:
        raise ValueError("source envelope JSON must be a non-empty string")
    if len(payload.encode("utf-8")) > MAX_SOURCE_ENVELOPE_BYTES:
        raise ValueError("source envelope JSON exceeds the maximum byte size")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("source envelope JSON is invalid") from exc
    if not isinstance(raw, dict):
        raise ValueError("source envelope JSON must contain an object")
    return _canonical_json(raw)


def _validate_canonical_source_json(payload: object) -> None:
    """Require canonical, finite, duplicate-free producer JSON bytes."""
    if not isinstance(payload, str) or not payload:
        raise ValueError("source_envelope_json must be a non-empty string")
    canonical = canonicalize_source_envelope(payload)
    if canonical != payload:
        raise ValueError("source_envelope_json must use canonical JSON")


def _validate_source_crosslinks(
    payload: str,
    *,
    source_schema: str,
    source_revision: str,
    event_id: str,
) -> None:
    """Bind handoff identity fields to the embedded producer envelope."""
    source = cast(
        dict[str, object],
        json.loads(payload, object_pairs_hook=_unique_object),
    )
    expected = {
        "schema": source_schema,
        "source_revision": source_revision,
        "event_id": event_id,
    }
    for field, value in expected.items():
        if field not in source:
            raise ValueError(f"embedded FUSION source envelope lacks {field}")
        if source[field] != value:
            raise ValueError(
                f"embedded FUSION source {field} does not match the handoff"
            )


def _canonical_json(payload: object) -> str:
    """Serialize a JSON value deterministically and reject non-finite numbers."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_digest(payload: object) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while refusing duplicate keys."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key: {key}")
        record[key] = value
    return record


def _require_list(value: object, *, field: str) -> list[object]:
    """Return a JSON list without coercion."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _require_contract_type(
    value: object, expected: type[_ContractType]
) -> _ContractType:
    """Return a decoded contract only when it has the expected public type."""
    if not isinstance(value, expected):
        raise ValueError(f"expected {expected.__name__} contract")
    return value


__all__ = [
    "HANDOFF_SCHEMA",
    "HANDOFF_SCHEMA_VERSION",
    "MAX_HANDOFF_JSON_BYTES",
    "MAX_SOURCE_ENVELOPE_BYTES",
    "ReactorSemanticHandoff",
    "canonicalize_source_envelope",
    "handoff_digest",
    "handoff_from_bytes",
    "handoff_from_json",
    "handoff_from_record",
    "handoff_to_json",
    "handoff_to_bytes",
    "handoff_to_record",
]
