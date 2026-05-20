# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract

"""Transport-neutral digital-twin contracts derived from bindings.

This module turns a validated `BindingSpec` into deterministic contract hashes,
adapter manifests, sync capabilities, and envelope validation records for
simulators, services, and hardware twins. REST, gRPC, Kafka, JSONL, hardware,
and in-memory helpers validate decoded payloads only; they do not open sockets,
spawn servers, or apply live control actions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path

from scpn_phase_orchestrator.binding.channel_algebra import (
    ChannelAlgebraReport,
    build_channel_algebra_report,
)
from scpn_phase_orchestrator.binding.types import BindingSpec

__all__ = [
    "DigitalTwinBindingContract",
    "DigitalTwinAdapterManifest",
    "DigitalTwinAdapterCompatibility",
    "DigitalTwinLayerContract",
    "DigitalTwinSyncCapability",
    "DigitalTwinSyncEnvelope",
    "DigitalTwinSyncGrpcAdapter",
    "DigitalTwinSyncGrpcResponse",
    "DigitalTwinSyncHardwareAdapter",
    "DigitalTwinSyncHardwareResponse",
    "DigitalTwinSyncJsonlReport",
    "DigitalTwinSyncKafkaAdapter",
    "DigitalTwinSyncKafkaResponse",
    "DigitalTwinSyncMemoryAdapter",
    "DigitalTwinOperatorEvidence",
    "DigitalTwinSyncRestAdapter",
    "DigitalTwinSyncRestResponse",
    "DigitalTwinTransportValidation",
    "build_digital_twin_adapter_manifest",
    "build_digital_twin_binding_contract",
    "build_digital_twin_operator_evidence",
    "build_digital_twin_sync_envelope",
    "read_digital_twin_sync_jsonl",
    "validate_digital_twin_sync_envelope",
    "write_digital_twin_sync_jsonl",
]

_DEFAULT_CONTRACT_VERSION = "spo-digital-twin-binding/v1"
_DEFAULT_SYNC_CAPABILITIES = (
    "state_snapshot",
    "phase_observation",
    "control_action_proposal",
    "audit_replay",
)
_VALID_ADAPTER_TRANSPORTS = frozenset(
    {"memory", "jsonl", "rest", "grpc", "kafka", "hardware"}
)


@dataclass(frozen=True)
class DigitalTwinAdapterManifest:
    """Reviewable manifest for a concrete digital-twin transport adapter."""

    name: str
    transport: str
    sync_capabilities: tuple[str, ...]
    supports_replay: bool
    requires_auth: bool
    notes: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "adapter name")
        _require_non_empty(self.transport, "adapter transport")
        if self.transport not in _VALID_ADAPTER_TRANSPORTS:
            raise ValueError(
                f"adapter transport must be one of {sorted(_VALID_ADAPTER_TRANSPORTS)}"
            )
        if not self.sync_capabilities:
            raise ValueError("adapter sync_capabilities must not be empty")
        for capability in self.sync_capabilities:
            _require_non_empty(capability, "adapter sync capability")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe adapter manifest."""
        return {
            "name": self.name,
            "transport": self.transport,
            "sync_capabilities": list(self.sync_capabilities),
            "supports_replay": self.supports_replay,
            "requires_auth": self.requires_auth,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DigitalTwinAdapterCompatibility:
    """Compatibility result for an adapter manifest and binding contract."""

    compatible: bool
    reasons: tuple[str, ...]
    manifest: DigitalTwinAdapterManifest
    contract_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe adapter compatibility report."""
        return {
            "compatible": self.compatible,
            "reasons": list(self.reasons),
            "manifest": self.manifest.to_audit_record(),
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True)
class DigitalTwinLayerContract:
    """Reduced layer contract exposed to simulators and hardware twins."""

    name: str
    index: int
    oscillator_count: int
    oscillator_ids: tuple[str, ...]
    family: str | None = None

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe layer contract."""
        return {
            "name": self.name,
            "index": self.index,
            "oscillator_count": self.oscillator_count,
            "oscillator_ids": list(self.oscillator_ids),
            "family": self.family,
        }


@dataclass(frozen=True)
class DigitalTwinSyncCapability:
    """Named live-sync capability declared by the binding contract."""

    name: str
    direction: str
    payload: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe sync capability contract."""
        return {
            "name": self.name,
            "direction": self.direction,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class DigitalTwinBindingContract:
    """Versioned bidirectional contract derived from a binding spec."""

    contract_version: str
    binding_name: str
    binding_version: str
    safety_tier: str
    sample_period_s: float
    control_period_s: float
    layers: tuple[DigitalTwinLayerContract, ...]
    actuators: tuple[dict[str, object], ...]
    channel_algebra: ChannelAlgebraReport
    sync_capabilities: tuple[DigitalTwinSyncCapability, ...]
    contract_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe digital-twin contract."""
        return {
            "contract_version": self.contract_version,
            "binding": {
                "name": self.binding_name,
                "version": self.binding_version,
                "safety_tier": self.safety_tier,
            },
            "timing": {
                "sample_period_s": self.sample_period_s,
                "control_period_s": self.control_period_s,
            },
            "layers": [layer.to_audit_record() for layer in self.layers],
            "actuators": list(self.actuators),
            "channel_algebra": self.channel_algebra.to_audit_record(),
            "sync_capabilities": [
                capability.to_audit_record() for capability in self.sync_capabilities
            ],
            "contract_hash": self.contract_hash,
        }

    def to_json(self) -> str:
        """Serialise the contract with deterministic key ordering."""
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DigitalTwinSyncEnvelope:
    """Transport-neutral live-sync payload envelope for digital twins."""

    contract_hash: str
    capability: str
    direction: str
    sequence: int
    payload: dict[str, object]

    def __post_init__(self) -> None:
        _require_non_empty(self.contract_hash, "contract_hash")
        _require_non_empty(self.capability, "capability")
        _require_non_empty(self.direction, "direction")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe sync envelope."""
        return {
            "contract_hash": self.contract_hash,
            "capability": self.capability,
            "direction": self.direction,
            "sequence": self.sequence,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        """Serialise the envelope with deterministic key ordering."""
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DigitalTwinTransportValidation:
    """Validation result for one digital-twin sync envelope."""

    accepted: bool
    reason: str
    envelope: DigitalTwinSyncEnvelope

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe validation record."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "envelope": self.envelope.to_audit_record(),
        }


@dataclass(frozen=True)
class DigitalTwinSyncJsonlReport:
    """JSONL file-adapter replay report for digital-twin sync envelopes."""

    path: str
    written: int
    accepted: tuple[DigitalTwinTransportValidation, ...]
    rejected: tuple[dict[str, object], ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe file-adapter report."""
        return {
            "path": self.path,
            "written": self.written,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "accepted": [validation.to_audit_record() for validation in self.accepted],
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True)
class DigitalTwinOperatorEvidence:
    """Transport-neutral operator summary for live or replayed twin sync."""

    contract_hash: str
    accepted_count: int
    rejected_count: int
    adapter_count: int
    unhealthy_adapter_count: int
    latest_sequence: int | None
    capability_counts: dict[str, int]
    direction_counts: dict[str, int]
    max_abs_twin_residual: float | None
    mismatch_reasons: tuple[str, ...]
    status: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe operator evidence record."""
        return {
            "contract_hash": self.contract_hash,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "adapter_count": self.adapter_count,
            "unhealthy_adapter_count": self.unhealthy_adapter_count,
            "latest_sequence": self.latest_sequence,
            "capability_counts": dict(sorted(self.capability_counts.items())),
            "direction_counts": dict(sorted(self.direction_counts.items())),
            "max_abs_twin_residual": self.max_abs_twin_residual,
            "mismatch_reasons": list(self.mismatch_reasons),
            "status": self.status,
        }


@dataclass
class DigitalTwinSyncMemoryAdapter:
    """In-memory reference adapter for validated digital-twin sync payloads."""

    contract: DigitalTwinBindingContract
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
    ) -> DigitalTwinSyncMemoryAdapter:
        """Create an empty adapter for a digital-twin binding contract."""
        return cls(contract=contract, _queue=[])

    def submit(
        self,
        envelope: DigitalTwinSyncEnvelope,
    ) -> DigitalTwinTransportValidation:
        """Validate and queue one envelope when accepted."""
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if validation.accepted:
            self._queue.append(envelope)
        return validation

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return queued envelopes in submission order and clear the queue."""
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return adapter state without exposing any network surface."""
        return {
            "contract_hash": self.contract.contract_hash,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


@dataclass(frozen=True)
class DigitalTwinSyncGrpcResponse:
    """gRPC-style response for a digital-twin sync boundary."""

    status_code: str
    accepted: bool
    reason: str
    message: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe gRPC adapter response."""
        return {
            "status_code": self.status_code,
            "accepted": self.accepted,
            "reason": self.reason,
            "message": dict(self.message),
        }


@dataclass
class DigitalTwinSyncGrpcAdapter:
    """Dependency-free gRPC boundary for digital-twin sync payloads.

    The adapter does not start a gRPC server or import generated protobuf
    classes. A real servicer can pass decoded protobuf fields into
    :meth:`handle_unary`; this boundary then applies the same contract checks
    as other transports before queuing accepted envelopes.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        name: str = "grpc-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = False,
    ) -> DigitalTwinSyncGrpcAdapter:
        """Create a gRPC adapter boundary for a digital-twin contract."""
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="grpc",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free gRPC boundary",
        )
        return cls(contract=contract, compatibility=compatibility, _queue=[])

    def handle_unary(
        self,
        request: Mapping[str, object],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncGrpcResponse:
        """Validate one unary gRPC request and queue accepted envelopes."""
        if not self.compatibility.compatible:
            return _grpc_response(
                "FAILED_PRECONDITION",
                False,
                "adapter_incompatible",
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            metadata,
        ):
            return _grpc_response(
                "UNAUTHENTICATED",
                False,
                "auth_required",
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(request))
        if envelope is None:
            return _grpc_response(
                "INVALID_ARGUMENT",
                False,
                "invalid_envelope",
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _grpc_response(
                "FAILED_PRECONDITION",
                False,
                validation.reason,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _grpc_response(
            "OK",
            True,
            "accepted",
            {
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted gRPC envelopes in arrival order and clear the queue."""
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return gRPC adapter state without exposing payload contents."""
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


@dataclass(frozen=True)
class DigitalTwinSyncKafkaResponse:
    """Broker-style response for a Kafka digital-twin sync boundary."""

    accepted: bool
    reason: str
    retryable: bool
    message: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe Kafka adapter response."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "retryable": self.retryable,
            "message": dict(self.message),
        }


@dataclass
class DigitalTwinSyncKafkaAdapter:
    """Dependency-free Kafka boundary for digital-twin sync payloads.

    The adapter expects a broker consumer to pass a decoded message dictionary.
    It does not import Kafka clients, open sockets, or commit offsets. Accepted
    envelopes are queued for caller-controlled runtime handoff.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    topic: str
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        topic: str = "spo.digital_twin.sync",
        name: str = "kafka-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = True,
    ) -> DigitalTwinSyncKafkaAdapter:
        """Create a Kafka message-boundary adapter for a digital-twin contract."""
        _require_non_empty(topic, "kafka topic")
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="kafka",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free Kafka boundary",
        )
        return cls(
            contract=contract,
            compatibility=compatibility,
            topic=topic,
            _queue=[],
        )

    def handle_message(
        self,
        message: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncKafkaResponse:
        """Validate one decoded Kafka message and queue accepted envelopes."""
        message_topic = message.get("topic", self.topic)
        if not isinstance(message_topic, str) or message_topic != self.topic:
            return _kafka_response(
                False,
                "topic_mismatch",
                False,
                {"expected_topic": self.topic, "observed_topic": message_topic},
            )
        if not self.compatibility.compatible:
            return _kafka_response(
                False,
                "adapter_incompatible",
                True,
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _kafka_response(
                False,
                "auth_required",
                True,
                {"contract_hash": self.contract.contract_hash},
            )
        value = message.get("value")
        if not isinstance(value, Mapping):
            return _kafka_response(
                False,
                "invalid_message_value",
                False,
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(value))
        if envelope is None:
            return _kafka_response(
                False,
                "invalid_envelope",
                False,
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _kafka_response(
                False,
                validation.reason,
                False,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _kafka_response(
            True,
            "accepted",
            False,
            {
                "topic": self.topic,
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted Kafka envelopes in arrival order and clear the queue."""
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return Kafka adapter state without exposing payload contents."""
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "topic": self.topic,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


@dataclass(frozen=True)
class DigitalTwinSyncHardwareResponse:
    """No-I/O response for a hardware digital-twin sync boundary."""

    accepted: bool
    reason: str
    hardware_write_permitted: bool
    frame: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe hardware adapter response."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "hardware_write_permitted": self.hardware_write_permitted,
            "frame": dict(self.frame),
        }


@dataclass
class DigitalTwinSyncHardwareAdapter:
    """No-I/O hardware boundary for digital-twin sync payloads.

    The adapter validates decoded frames from a hardware integration layer. It
    never opens device files, writes registers, toggles GPIO, or applies
    actuation; accepted envelopes are only queued for caller-controlled review.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    device_ids: tuple[str, ...]
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        device_ids: Sequence[str],
        name: str = "hardware-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = True,
    ) -> DigitalTwinSyncHardwareAdapter:
        """Create a no-I/O hardware boundary for a digital-twin contract."""
        if not device_ids:
            raise ValueError("hardware device_ids must not be empty")
        checked_device_ids = tuple(device_ids)
        for device_id in checked_device_ids:
            _require_non_empty(device_id, "hardware device_id")
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="hardware",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="no-I/O hardware boundary",
        )
        return cls(
            contract=contract,
            compatibility=compatibility,
            device_ids=checked_device_ids,
            _queue=[],
        )

    def handle_frame(
        self,
        frame: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncHardwareResponse:
        """Validate one decoded hardware frame and queue accepted envelopes."""
        device_id = frame.get("device_id")
        if not isinstance(device_id, str) or device_id not in self.device_ids:
            return _hardware_response(
                False,
                "device_not_registered",
                {"device_id": device_id, "registered_devices": list(self.device_ids)},
            )
        if frame.get("safety_interlock") is not True:
            return _hardware_response(
                False,
                "safety_interlock_required",
                {"device_id": device_id},
            )
        if not self.compatibility.compatible:
            return _hardware_response(
                False,
                "adapter_incompatible",
                {
                    "device_id": device_id,
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _hardware_response(
                False,
                "auth_required",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        value = frame.get("value")
        if not isinstance(value, Mapping):
            return _hardware_response(
                False,
                "invalid_frame_value",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(value))
        if envelope is None:
            return _hardware_response(
                False,
                "invalid_envelope",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _hardware_response(
                False,
                validation.reason,
                {
                    "device_id": device_id,
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _hardware_response(
            True,
            "accepted",
            {
                "device_id": device_id,
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted hardware envelopes in arrival order and clear the queue."""
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return hardware adapter state without exposing payload contents."""
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "device_ids": list(self.device_ids),
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
            "hardware_write_permitted": False,
        }


@dataclass(frozen=True)
class DigitalTwinSyncRestResponse:
    """HTTP-style response for a REST digital-twin sync boundary."""

    status_code: int
    accepted: bool
    reason: str
    body: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe REST adapter response."""
        return {
            "status_code": self.status_code,
            "accepted": self.accepted,
            "reason": self.reason,
            "body": dict(self.body),
        }


@dataclass
class DigitalTwinSyncRestAdapter:
    """Dependency-free REST boundary for digital-twin sync payloads.

    The adapter deliberately does not open sockets. Web frameworks can call
    :meth:`handle_post` from a route handler after parsing request JSON and
    headers; the adapter then enforces manifest compatibility, authentication
    posture, envelope shape, and contract validation before queuing payloads.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        name: str = "rest-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = False,
    ) -> DigitalTwinSyncRestAdapter:
        """Create a REST adapter boundary for a digital-twin contract."""
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="rest",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free REST boundary",
        )
        return cls(contract=contract, compatibility=compatibility, _queue=[])

    def handle_post(
        self,
        body: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncRestResponse:
        """Validate one HTTP POST body and queue accepted sync envelopes."""
        if not self.compatibility.compatible:
            return _rest_response(
                503,
                False,
                "adapter_incompatible",
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _rest_response(
                401,
                False,
                "auth_required",
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(body))
        if envelope is None:
            return _rest_response(
                400,
                False,
                "invalid_envelope",
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _rest_response(
                422,
                False,
                validation.reason,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _rest_response(
            202,
            True,
            "accepted",
            {
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted REST envelopes in arrival order and clear the queue."""
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return REST adapter state without exposing payload contents."""
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


def build_digital_twin_binding_contract(
    spec: BindingSpec,
    *,
    contract_version: str = _DEFAULT_CONTRACT_VERSION,
    sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
) -> DigitalTwinBindingContract:
    """Build a versioned live-sync contract from a validated binding spec.

    The contract is read-only and transport-neutral. It describes what a
    simulator, service twin, or hardware twin may exchange with SPO without
    opening network connections or applying actuation.
    """
    _require_non_empty(contract_version, "contract_version")
    if not sync_capabilities:
        raise ValueError("sync_capabilities must contain at least one capability")
    capabilities = tuple(
        _capability_from_name(capability) for capability in sync_capabilities
    )
    layers = tuple(
        DigitalTwinLayerContract(
            name=layer.name,
            index=layer.index,
            oscillator_count=len(layer.oscillator_ids),
            oscillator_ids=tuple(layer.oscillator_ids),
            family=layer.family,
        )
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )
    actuator_records: list[dict[str, object]] = [
        {
            "name": actuator.name,
            "knob": actuator.knob,
            "scope": actuator.scope,
            "limits": list(actuator.limits),
        }
        for actuator in sorted(spec.actuators, key=lambda item: item.name)
    ]
    actuators = tuple(actuator_records)
    channel_algebra = build_channel_algebra_report(spec)
    base_record: dict[str, object] = {
        "contract_version": contract_version,
        "binding": {
            "name": spec.name,
            "version": spec.version,
            "safety_tier": spec.safety_tier,
        },
        "timing": {
            "sample_period_s": spec.sample_period_s,
            "control_period_s": spec.control_period_s,
        },
        "layers": [layer.to_audit_record() for layer in layers],
        "actuators": list(actuators),
        "channel_algebra": channel_algebra.to_audit_record(),
        "sync_capabilities": [
            capability.to_audit_record() for capability in capabilities
        ],
    }
    contract_hash = _record_hash(base_record)
    return DigitalTwinBindingContract(
        contract_version=contract_version,
        binding_name=spec.name,
        binding_version=spec.version,
        safety_tier=spec.safety_tier,
        sample_period_s=spec.sample_period_s,
        control_period_s=spec.control_period_s,
        layers=layers,
        actuators=actuators,
        channel_algebra=channel_algebra,
        sync_capabilities=capabilities,
        contract_hash=contract_hash,
    )


def build_digital_twin_adapter_manifest(
    contract: DigitalTwinBindingContract,
    *,
    name: str,
    transport: str,
    sync_capabilities: Sequence[str],
    supports_replay: bool,
    requires_auth: bool,
    notes: str = "",
) -> DigitalTwinAdapterCompatibility:
    """Build and validate a transport-adapter manifest against a contract."""
    manifest = DigitalTwinAdapterManifest(
        name=name,
        transport=transport,
        sync_capabilities=tuple(sync_capabilities),
        supports_replay=supports_replay,
        requires_auth=requires_auth,
        notes=notes,
    )
    declared = {capability.name for capability in contract.sync_capabilities}
    reasons: list[str] = []
    missing = sorted(set(manifest.sync_capabilities) - declared)
    if missing:
        reasons.append(f"capability_not_declared:{','.join(missing)}")
    if (
        manifest.transport in {"rest", "grpc", "kafka", "hardware"}
        and not requires_auth
    ):
        reasons.append("live_transport_requires_auth")
    if manifest.transport in {"jsonl", "memory"} and not supports_replay:
        reasons.append("offline_transport_requires_replay")
    return DigitalTwinAdapterCompatibility(
        compatible=not reasons,
        reasons=tuple(reasons),
        manifest=manifest,
        contract_hash=contract.contract_hash,
    )


def build_digital_twin_sync_envelope(
    contract: DigitalTwinBindingContract,
    *,
    capability: str,
    direction: str,
    sequence: int,
    payload: dict[str, object],
) -> DigitalTwinSyncEnvelope:
    """Build a transport-neutral sync payload envelope for a contract.

    This helper does not send data. It creates the deterministic envelope that
    REST, gRPC, Kafka, file, or hardware adapters can validate before handing a
    payload to the runtime.
    """
    return DigitalTwinSyncEnvelope(
        contract_hash=contract.contract_hash,
        capability=capability,
        direction=direction,
        sequence=sequence,
        payload=payload,
    )


def write_digital_twin_sync_jsonl(
    path: str | Path,
    envelopes: Sequence[DigitalTwinSyncEnvelope],
) -> DigitalTwinSyncJsonlReport:
    """Write sync envelopes to deterministic JSONL for offline replay."""
    target = Path(path)
    lines = [envelope.to_json() for envelope in envelopes]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return DigitalTwinSyncJsonlReport(
        path=str(target),
        written=len(lines),
        accepted=(),
        rejected=(),
    )


def build_digital_twin_operator_evidence(
    contract: DigitalTwinBindingContract,
    validations: Sequence[DigitalTwinTransportValidation],
    *,
    rejected: Sequence[Mapping[str, object]] = (),
    adapter_records: Sequence[Mapping[str, object]] = (),
    residual_warning_threshold: float = 0.05,
    residual_critical_threshold: float = 0.2,
) -> DigitalTwinOperatorEvidence:
    """Summarise live or replayed digital-twin sync evidence for operators.

    Accepted validations may come from REST, gRPC, Kafka, hardware, memory, or
    JSONL replay paths. Rejected JSONL lines and adapter audit records are
    folded into the same deterministic summary so dashboards can display live
    and replayed health with the same fields.
    """
    warning_threshold = _validated_residual_threshold(
        residual_warning_threshold,
        "residual_warning_threshold",
    )
    critical_threshold = _validated_residual_threshold(
        residual_critical_threshold,
        "residual_critical_threshold",
    )
    if warning_threshold > critical_threshold:
        raise ValueError(
            "residual_warning_threshold must be <= residual_critical_threshold"
        )

    accepted: list[DigitalTwinTransportValidation] = []
    mismatch_reasons: list[str] = []
    capability_counts = {
        capability.name: 0 for capability in contract.sync_capabilities
    }
    direction_counts: dict[str, int] = {}
    latest_sequence: int | None = None
    residuals: list[float] = []

    for validation in validations:
        envelope = validation.envelope
        if envelope.contract_hash != contract.contract_hash:
            mismatch_reasons.append("contract_hash_mismatch")
            continue
        if not validation.accepted:
            mismatch_reasons.append(validation.reason)
            continue
        accepted.append(validation)
        capability_counts[envelope.capability] = (
            capability_counts.get(envelope.capability, 0) + 1
        )
        direction_counts[envelope.direction] = (
            direction_counts.get(
                envelope.direction,
                0,
            )
            + 1
        )
        latest_sequence = (
            envelope.sequence
            if latest_sequence is None
            else max(latest_sequence, envelope.sequence)
        )
        residual = _extract_twin_residual(envelope.payload)
        if residual is not None:
            residuals.append(abs(residual))

    for rejection in rejected:
        reason = rejection.get("reason")
        if isinstance(reason, str) and reason:
            mismatch_reasons.append(reason)
        else:
            mismatch_reasons.append("rejected")

    unhealthy_adapter_count = sum(
        1 for record in adapter_records if record.get("compatible") is False
    )
    max_abs_residual = max(residuals) if residuals else None
    rejected_count = len(validations) - len(accepted) + len(rejected)
    status = _operator_status(
        rejected_count=rejected_count,
        unhealthy_adapter_count=unhealthy_adapter_count,
        max_abs_residual=max_abs_residual,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    return DigitalTwinOperatorEvidence(
        contract_hash=contract.contract_hash,
        accepted_count=len(accepted),
        rejected_count=rejected_count,
        adapter_count=len(adapter_records),
        unhealthy_adapter_count=unhealthy_adapter_count,
        latest_sequence=latest_sequence,
        capability_counts=capability_counts,
        direction_counts=direction_counts,
        max_abs_twin_residual=max_abs_residual,
        mismatch_reasons=tuple(sorted(mismatch_reasons)),
        status=status,
    )


def read_digital_twin_sync_jsonl(
    contract: DigitalTwinBindingContract,
    path: str | Path,
) -> DigitalTwinSyncJsonlReport:
    """Read JSONL sync envelopes and validate them against a contract."""
    source = Path(path)
    accepted: list[DigitalTwinTransportValidation] = []
    rejected: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_record = json.loads(raw_line)
        except json.JSONDecodeError:
            rejected.append(_jsonl_rejection(line_number, "malformed_json"))
            continue
        envelope = _envelope_from_record(raw_record)
        if envelope is None:
            rejected.append(_jsonl_rejection(line_number, "invalid_envelope"))
            continue
        validation = validate_digital_twin_sync_envelope(contract, envelope)
        if validation.accepted:
            accepted.append(validation)
        else:
            rejected.append(_jsonl_rejection(line_number, validation.reason))
    return DigitalTwinSyncJsonlReport(
        path=str(source),
        written=0,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


def validate_digital_twin_sync_envelope(
    contract: DigitalTwinBindingContract,
    envelope: DigitalTwinSyncEnvelope,
) -> DigitalTwinTransportValidation:
    """Validate a digital-twin sync envelope against a binding contract."""
    if envelope.contract_hash != contract.contract_hash:
        return _transport_validation(False, "contract_hash_mismatch", envelope)
    capability = _find_capability(contract, envelope.capability)
    if capability is None:
        return _transport_validation(False, "capability_not_declared", envelope)
    if not _direction_allowed(
        declared=capability.direction,
        observed=envelope.direction,
    ):
        return _transport_validation(False, "direction_not_allowed", envelope)
    if not envelope.payload:
        return _transport_validation(False, "payload_empty", envelope)
    return _transport_validation(True, "accepted", envelope)


def _capability_from_name(name: str) -> DigitalTwinSyncCapability:
    _require_non_empty(name, "sync capability")
    payloads = {
        "state_snapshot": ("twin_to_spo", "state_vector"),
        "phase_observation": ("twin_to_spo", "phase_channel_summary"),
        "control_action_proposal": ("spo_to_twin", "control_action"),
        "audit_replay": ("bidirectional", "audit_record"),
    }
    direction, payload = payloads.get(name, ("bidirectional", "json_object"))
    return DigitalTwinSyncCapability(name=name, direction=direction, payload=payload)


def _find_capability(
    contract: DigitalTwinBindingContract,
    name: str,
) -> DigitalTwinSyncCapability | None:
    for capability in contract.sync_capabilities:
        if capability.name == name:
            return capability
    return None


def _envelope_from_record(record: object) -> DigitalTwinSyncEnvelope | None:
    if not isinstance(record, dict):
        return None
    contract_hash = record.get("contract_hash")
    capability = record.get("capability")
    direction = record.get("direction")
    sequence = record.get("sequence")
    payload = record.get("payload")
    if not isinstance(contract_hash, str):
        return None
    if not isinstance(capability, str):
        return None
    if not isinstance(direction, str):
        return None
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return DigitalTwinSyncEnvelope(
            contract_hash=contract_hash,
            capability=capability,
            direction=direction,
            sequence=sequence,
            payload=dict(payload),
        )
    except ValueError:
        return None


def _jsonl_rejection(line_number: int, reason: str) -> dict[str, object]:
    return {
        "line_number": line_number,
        "reason": reason,
    }


def _direction_allowed(*, declared: str, observed: str) -> bool:
    return declared == "bidirectional" or declared == observed


def _transport_validation(
    accepted: bool,
    reason: str,
    envelope: DigitalTwinSyncEnvelope,
) -> DigitalTwinTransportValidation:
    return DigitalTwinTransportValidation(
        accepted=accepted,
        reason=reason,
        envelope=envelope,
    )


def _rest_response(
    status_code: int,
    accepted: bool,
    reason: str,
    body: dict[str, object],
) -> DigitalTwinSyncRestResponse:
    return DigitalTwinSyncRestResponse(
        status_code=status_code,
        accepted=accepted,
        reason=reason,
        body=body,
    )


def _grpc_response(
    status_code: str,
    accepted: bool,
    reason: str,
    message: dict[str, object],
) -> DigitalTwinSyncGrpcResponse:
    return DigitalTwinSyncGrpcResponse(
        status_code=status_code,
        accepted=accepted,
        reason=reason,
        message=message,
    )


def _kafka_response(
    accepted: bool,
    reason: str,
    retryable: bool,
    message: dict[str, object],
) -> DigitalTwinSyncKafkaResponse:
    return DigitalTwinSyncKafkaResponse(
        accepted=accepted,
        reason=reason,
        retryable=retryable,
        message=message,
    )


def _hardware_response(
    accepted: bool,
    reason: str,
    frame: dict[str, object],
) -> DigitalTwinSyncHardwareResponse:
    return DigitalTwinSyncHardwareResponse(
        accepted=accepted,
        reason=reason,
        hardware_write_permitted=False,
        frame=frame,
    )


def _has_authorization(headers: Mapping[str, str] | None) -> bool:
    if headers is None:
        return False
    normalised = {key.lower(): value for key, value in headers.items()}
    token = normalised.get("authorization")
    return isinstance(token, str) and bool(token.strip())


def _extract_twin_residual(payload: Mapping[str, object]) -> float | None:
    for key in (
        "TwinResidual",
        "twin_residual",
        "twin_residual_norm",
        "residual",
        "residual_norm",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite real value")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{key} must be a finite real value")
        return result
    return None


def _operator_status(
    *,
    rejected_count: int,
    unhealthy_adapter_count: int,
    max_abs_residual: float | None,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    if max_abs_residual is not None and max_abs_residual > critical_threshold:
        return "critical"
    if rejected_count or unhealthy_adapter_count:
        return "degraded"
    if max_abs_residual is not None and max_abs_residual > warning_threshold:
        return "warning"
    return "healthy"


def _validated_residual_threshold(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite non-negative real value")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative real value")
    return result


def _record_hash(record: dict[str, object]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
