# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic serialization

"""Canonical serialization and strict dispatch for U0 contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import TypeAlias

from .contracts import (
    ObservableDescriptor,
    PhaseRelation,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeEstimate,
)
from .registry import DEFAULT_REACTOR_REGISTRY, ReactorConfigurationRegistry
from .vocabulary import require_exact_keys, require_u0_schema

ReactorSemanticContract: TypeAlias = (
    ReactorContext
    | ObservableDescriptor
    | PhaseSemanticRecord
    | PhaseRelation
    | RegimeEstimate
)

_CONTRACT_TYPES = {
    ReactorContext: "reactor_context",
    ObservableDescriptor: "observable_descriptor",
    PhaseSemanticRecord: "phase_semantic_record",
    PhaseRelation: "phase_relation",
    RegimeEstimate: "regime_estimate",
}


def contract_to_record(
    contract: ReactorSemanticContract,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> dict[str, object]:
    """Return a typed envelope for a U0 contract."""
    try:
        contract_type = _CONTRACT_TYPES[type(contract)]
    except KeyError as exc:
        raise TypeError("unsupported reactor semantic contract") from exc
    if isinstance(contract, ReactorContext):
        contract.validate_registry(registry)
    elif isinstance(contract, ObservableDescriptor):
        contract.reactor_context.validate_registry(registry)
    record = contract.to_record()
    return {
        "contract_type": contract_type,
        "payload": record,
        "schema_version": require_u0_schema(record["schema_version"]),
    }


def contract_from_record(
    raw: object,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> ReactorSemanticContract:
    """Load one contract and refuse unknown fields, kinds, or schema versions."""
    envelope = require_exact_keys(
        raw,
        required=frozenset({"contract_type", "payload", "schema_version"}),
        field="contract envelope",
    )
    schema_version = require_u0_schema(envelope["schema_version"])
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ValueError("contract envelope payload must be an object")
    if payload.get("schema_version") != schema_version:
        raise ValueError("envelope and payload schema versions must match")
    contract_type = envelope["contract_type"]
    if contract_type == "reactor_context":
        return ReactorContext.from_record(payload, registry=registry)
    if contract_type == "observable_descriptor":
        return ObservableDescriptor.from_record(payload, registry=registry)
    if contract_type == "phase_semantic_record":
        return PhaseSemanticRecord.from_record(payload)
    if contract_type == "phase_relation":
        return PhaseRelation.from_record(payload)
    if contract_type == "regime_estimate":
        return RegimeEstimate.from_record(payload)
    raise ValueError(f"unsupported contract_type: {contract_type!r}")


def canonical_json(
    contract: ReactorSemanticContract,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> str:
    """Serialize a contract to byte-stable canonical JSON."""
    return json.dumps(
        contract_to_record(contract, registry=registry),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def contract_from_json(
    payload: str,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> ReactorSemanticContract:
    """Deserialize canonical or ordinary JSON with duplicate-key refusal."""
    if not isinstance(payload, str) or not payload:
        raise ValueError("contract JSON must be a non-empty string")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("contract JSON is invalid") from exc
    return contract_from_record(raw, registry=registry)


def contract_digest(
    contract: ReactorSemanticContract,
    *,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> str:
    """Return the SHA-256 digest of canonical contract JSON."""
    return hashlib.sha256(
        canonical_json(contract, registry=registry).encode("utf-8")
    ).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON key: {key}")
        record[key] = value
    return record


__all__ = [
    "ReactorSemanticContract",
    "canonical_json",
    "contract_digest",
    "contract_from_json",
    "contract_from_record",
    "contract_to_record",
]
