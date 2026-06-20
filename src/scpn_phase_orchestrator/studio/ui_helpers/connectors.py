# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio live connector builders

"""Live connector plan, run-record, and owned-runtime builders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import cast

from scpn_phase_orchestrator.binding.digital_twin import (
    DigitalTwinBindingContract,
    DigitalTwinSyncGrpcAdapter,
    DigitalTwinSyncHardwareAdapter,
    DigitalTwinSyncKafkaAdapter,
    DigitalTwinSyncRestAdapter,
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
    build_digital_twin_sync_envelope,
)
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec

from ._shared import (
    _connector_by_transport,
    _mapping_count,
    _non_negative_int,
    _require_non_empty_text,
)
from ._state import StudioReplayResult


def build_live_connector_plan(spec: BindingSpec) -> dict[str, object]:
    """Return non-opening connector ownership guidance for Studio.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    dict[str, object]
        Non-opening connector ownership guidance for Studio.
    """
    contract = build_digital_twin_binding_contract(spec)
    connector_specs = (
        ("memory", True, False, "review offline memory connector"),
        ("jsonl", True, False, "review JSONL replay connector"),
        ("rest", False, True, "assign connector owner and auth policy"),
        ("grpc", False, True, "assign connector owner and auth policy"),
        ("kafka", False, True, "assign connector owner and auth policy"),
        ("hardware", False, True, "assign connector owner and auth policy"),
    )
    connectors: list[dict[str, object]] = []
    for transport, supports_replay, requires_auth, action in connector_specs:
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=f"studio-{transport}",
            transport=transport,
            sync_capabilities=[
                capability.name for capability in contract.sync_capabilities
            ],
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="SPO Studio connector review",
        )
        manifest = compatibility.manifest
        owner_required = transport in {"rest", "grpc", "kafka", "hardware"}
        connectors.append(
            {
                "name": manifest.name,
                "transport": manifest.transport,
                "status": "owner_required" if owner_required else "review_ready",
                "compatible": compatibility.compatible,
                "reasons": list(compatibility.reasons),
                "sync_capabilities": list(manifest.sync_capabilities),
                "supports_replay": manifest.supports_replay,
                "requires_auth": manifest.requires_auth,
                "operator_action": action,
                "network_opened": False,
                "hardware_write_permitted": False,
            }
        )
    return {
        "plan_kind": "studio_live_connector_plan",
        "project_name": spec.name,
        "contract_hash": contract.contract_hash,
        "network_opened": False,
        "actuation_permitted": False,
        "connectors": connectors,
    }


def build_live_connector_run_record(
    connector_plan: Mapping[str, object],
    *,
    transport: str,
    payload: Mapping[str, object],
    dry_run: bool = True,
) -> dict[str, object]:
    """Return a gated live-connector execution record without opening transport.

    Parameters
    ----------
    connector_plan : Mapping[str, object]
        The live-connector plan mapping.
    transport : str
        Transport identifier.
    payload : Mapping[str, object]
        The payload mapping or bytes.
    dry_run : bool
        Whether to run without opening transport.

    Returns
    -------
    dict[str, object]
        A gated live-connector execution record without opening transport.
    """
    connector = _connector_by_transport(
        connector_plan,
        _require_non_empty_text(transport, "transport"),
    )
    payload_json = _stable_json_payload(payload, "payload")
    connector_status = _require_non_empty_text(connector.get("status"), "status")
    blocked_reasons: list[str] = []
    if connector_status != "review_ready":
        blocked_reasons.append("connector owner and auth policy required")
    if not dry_run:
        blocked_reasons.append("Studio live execution uses dry-run records only")

    status = "blocked" if blocked_reasons else "accepted"
    return {
        "record_kind": "studio_live_connector_run",
        "project_name": _require_non_empty_text(
            connector_plan.get("project_name"),
            "project_name",
        ),
        "transport": connector["transport"],
        "connector_name": connector["name"],
        "status": status,
        "dry_run": bool(dry_run),
        "payload_sha256": sha256(payload_json.encode("utf-8")).hexdigest(),
        "blocked_reasons": blocked_reasons,
        "operator_action": (
            "review dry-run connector payload"
            if status == "accepted"
            else _require_non_empty_text(
                connector.get("operator_action"),
                "operator_action",
            )
        ),
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
    }


def build_owned_live_connector_runtime_record(
    result: StudioReplayResult,
    *,
    transport: str,
    owner: str,
    auth_policy: Mapping[str, object],
    payload: Mapping[str, object],
    sequence: int = 1,
    capability: str = "audit_replay",
    direction: str = "twin_to_spo",
) -> dict[str, object]:
    """Validate an owned live connector boundary without opening transport.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    transport : str
        Transport identifier.
    owner : str
        Owner of the connector boundary.
    auth_policy : Mapping[str, object]
        The connector authentication policy.
    payload : Mapping[str, object]
        The payload mapping or bytes.
    sequence : int
        Monotonic sequence number.
    capability : str
        The sync capability identifier.
    direction : str
        Sync direction (e.g. ``inbound``/``outbound``).

    Returns
    -------
    dict[str, object]
        An owned live connector boundary without opening transport.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    checked_transport = _require_non_empty_text(transport, "transport")
    checked_payload = _normalise_json_mapping(
        cast("Mapping[object, object]", payload),
        "payload",
    )
    payload_json = _stable_json_payload(checked_payload, "payload")
    blocked_reasons = _owned_runtime_blocked_reasons(
        result.connector_plan,
        checked_transport,
        owner,
        auth_policy,
    )
    base = _owned_runtime_base_record(
        result,
        transport=checked_transport,
        owner=owner,
        payload_sha256=sha256(payload_json.encode("utf-8")).hexdigest(),
        sequence=sequence,
        capability=capability,
        direction=direction,
    )
    if blocked_reasons:
        return {
            **base,
            "status": "blocked",
            "blocked_reasons": blocked_reasons,
            "response": {},
            "adapter": {},
            "queued_count": 0,
        }

    spec_path = _result_binding_spec_path(result)
    spec = load_binding_spec(spec_path)
    contract = build_digital_twin_binding_contract(spec)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability=_require_non_empty_text(capability, "capability"),
        direction=_require_non_empty_text(direction, "direction"),
        sequence=_non_negative_int(sequence, "sequence"),
        payload=checked_payload,
    )
    response, adapter_record = _run_owned_live_adapter(
        contract,
        transport=checked_transport,
        envelope_record=envelope.to_audit_record(),
    )
    return {
        **base,
        "status": "accepted" if response.get("accepted") is True else "blocked",
        "blocked_reasons": (
            [] if response.get("accepted") is True else [str(response["reason"])]
        ),
        "response": response,
        "adapter": adapter_record,
        "queued_count": _mapping_count(adapter_record, "queued_count"),
    }


def _owned_runtime_blocked_reasons(
    connector_plan: Mapping[str, object],
    transport: str,
    owner: str,
    auth_policy: Mapping[str, object],
) -> list[str]:
    blocked: list[str] = []
    connector = _connector_by_transport(connector_plan, transport)
    if transport not in {"rest", "grpc", "kafka", "hardware"}:
        blocked.append("owned runtime requires a live connector transport")
    if connector.get("compatible") is not True:
        blocked.append("connector manifest is incompatible")
    if not isinstance(owner, str) or not owner.strip():
        blocked.append("owner must be assigned")
    if not isinstance(auth_policy, Mapping):
        blocked.append("auth_policy must be a mapping")
        return blocked
    scheme = auth_policy.get("scheme")
    credential_label = auth_policy.get("credential_label")
    if not isinstance(scheme, str) or not scheme.strip():
        blocked.append("auth_policy.scheme must be assigned")
    if not isinstance(credential_label, str) or not credential_label.strip():
        blocked.append("auth_policy.credential_label must be assigned")
    return blocked


def _owned_runtime_base_record(
    result: StudioReplayResult,
    *,
    transport: str,
    owner: str,
    payload_sha256: str,
    sequence: int,
    capability: str,
    direction: str,
) -> dict[str, object]:
    return {
        "record_kind": "studio_owned_live_connector_runtime",
        "project_name": result.project_state.project_name,
        "transport": transport,
        "owner": owner.strip() if isinstance(owner, str) else "",
        "contract_hash": result.connector_plan.get("contract_hash", ""),
        "capability": capability,
        "direction": direction,
        "sequence": sequence,
        "payload_sha256": payload_sha256,
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
    }


def _result_binding_spec_path(result: StudioReplayResult) -> Path:
    source_path = result.project_state.binding.provenance.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("project binding provenance must include source_path")
    return Path(source_path)


def _run_owned_live_adapter(
    contract: DigitalTwinBindingContract,
    *,
    transport: str,
    envelope_record: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    headers = {"authorization": "Bearer studio-owned-runtime"}
    if transport == "rest":
        rest = DigitalTwinSyncRestAdapter.for_contract(contract, name="studio-rest")
        rest_response = rest.handle_post(envelope_record, headers=headers)
        return rest_response.to_audit_record(), rest.to_audit_record()
    if transport == "grpc":
        grpc = DigitalTwinSyncGrpcAdapter.for_contract(contract, name="studio-grpc")
        grpc_response = grpc.handle_unary(envelope_record, metadata=headers)
        return grpc_response.to_audit_record(), grpc.to_audit_record()
    if transport == "kafka":
        kafka = DigitalTwinSyncKafkaAdapter.for_contract(contract, name="studio-kafka")
        kafka_response = kafka.handle_message(
            {"topic": kafka.topic, "value": envelope_record},
            headers=headers,
        )
        return kafka_response.to_audit_record(), kafka.to_audit_record()
    if transport == "hardware":
        hardware = DigitalTwinSyncHardwareAdapter.for_contract(
            contract,
            name="studio-hardware",
            device_ids=("studio-review-device",),
        )
        hardware_response = hardware.handle_frame(
            {
                "device_id": "studio-review-device",
                "safety_interlock": True,
                "value": envelope_record,
            },
            headers=headers,
        )
        return hardware_response.to_audit_record(), hardware.to_audit_record()
    raise ValueError(f"connector transport {transport!r} is not a live runtime")


def _stable_json_payload(value: object, field_name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return json.dumps(
        _normalise_json_mapping(value, field_name),
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalise_json_mapping(
    value: Mapping[object, object],
    field_name: str,
) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} contains an invalid key")
        safe[key] = _normalise_json_value(item, field_name)
    return safe


def _normalise_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return _normalise_json_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_normalise_json_value(item, field_name) for item in value]
    raise ValueError(f"{field_name} contains a non-JSON-safe value")
