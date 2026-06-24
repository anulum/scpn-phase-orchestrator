# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract and adapter manifests

"""Binding contract hashing, layer contracts, sync capabilities, and manifests."""

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

from ._shared import _require_non_empty

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
        """Return a JSON-safe adapter manifest.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinAdapterManifest
            fields.
        """
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
        """Return a JSON-safe adapter compatibility report.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the
            DigitalTwinAdapterCompatibility fields.
        """
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
        """Return a JSON-safe layer contract.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinLayerContract
            fields.
        """
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
        """Return a JSON-safe sync capability contract.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncCapability
            fields.
        """
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
        """Return a deterministic JSON-safe digital-twin contract.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinBindingContract
            fields.
        """
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
        """Serialise the contract with deterministic key ordering.

        Returns
        -------
        str
            The contract serialised as a JSON string with deterministically sorted keys.
        """
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


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

    Parameters
    ----------
    spec : BindingSpec
        The validated binding specification.
    contract_version : str, optional
        Semantic version label for the emitted contract.
    sync_capabilities : Sequence[str], optional
        Capabilities the contract advertises.

    Returns
    -------
    DigitalTwinBindingContract
        A read-only, transport-neutral live-sync contract.

    Raises
    ------
    ValueError
        If the spec cannot form a valid live-sync contract.
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
    """Build and validate a transport-adapter manifest against a contract.

    Parameters
    ----------
    contract : DigitalTwinBindingContract
        The contract the adapter must satisfy.
    name : str
        Adapter name.
    transport : str
        Transport identifier (e.g. ``rest``, ``grpc``, ``kafka``).
    sync_capabilities : Sequence[str]
        Capabilities the adapter implements.
    supports_replay : bool
        Whether the adapter supports replay.
    requires_auth : bool
        Whether the adapter requires authentication.
    notes : str, optional
        Free-form manifest notes.

    Returns
    -------
    DigitalTwinAdapterCompatibility
        The adapter compatibility report against the contract.
    """
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


def _capability_from_name(name: str) -> DigitalTwinSyncCapability:
    """Return the capability for a name, else raise."""
    _require_non_empty(name, "sync capability")
    payloads = {
        "state_snapshot": ("twin_to_spo", "state_vector"),
        "phase_observation": ("twin_to_spo", "phase_channel_summary"),
        "control_action_proposal": ("spo_to_twin", "control_action"),
        "audit_replay": ("bidirectional", "audit_record"),
    }
    direction, payload = payloads.get(name, ("bidirectional", "json_object"))
    return DigitalTwinSyncCapability(name=name, direction=direction, payload=payload)


def _record_hash(record: dict[str, object]) -> str:
    """Return the canonical hash of a record."""
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
