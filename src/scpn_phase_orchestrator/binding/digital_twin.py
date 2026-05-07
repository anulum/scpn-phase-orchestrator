# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.binding.channel_algebra import (
    ChannelAlgebraReport,
    build_channel_algebra_report,
)
from scpn_phase_orchestrator.binding.types import BindingSpec

__all__ = [
    "DigitalTwinBindingContract",
    "DigitalTwinLayerContract",
    "DigitalTwinSyncCapability",
    "DigitalTwinSyncEnvelope",
    "DigitalTwinTransportValidation",
    "build_digital_twin_binding_contract",
    "build_digital_twin_sync_envelope",
    "validate_digital_twin_sync_envelope",
]

_DEFAULT_CONTRACT_VERSION = "spo-digital-twin-binding/v1"
_DEFAULT_SYNC_CAPABILITIES = (
    "state_snapshot",
    "phase_observation",
    "control_action_proposal",
    "audit_replay",
)


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


def _record_hash(record: dict[str, object]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
