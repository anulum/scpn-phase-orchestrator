# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated transport envelope and replay

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

__all__ = [
    "FederatedTransportEnvelope",
    "FederatedTransportReplayLedger",
    "FederatedTransportDeploymentPreflightManifest",
    "build_signed_transport_envelopes",
    "build_transport_deployment_preflight_manifest",
    "validate_transport_deployment_preflight_manifest",
    "replay_federated_transport_batch",
    "validate_federated_transport_batch",
]


_DEFAULT_SCHEMA_NAME = "federated_transport_envelope"
_DEFAULT_SCHEMA_VERSION = "0.1.0"
_ALLOWED_SCHEMA_NAMES = {_DEFAULT_SCHEMA_NAME}
_ALLOWED_SCHEMA_VERSIONS = {_DEFAULT_SCHEMA_VERSION}
_ALLOWED_TRANSPORTS = {"jsonl", "rest", "grpc", "kafka"}
_LIVE_TRANSPORTS = {"rest", "grpc", "kafka"}
_ZERO_SHA256 = "0" * 64
_ALLOWED_UPDATE_KEYS = {
    "node_id",
    "policy_delta",
    "sample_count",
    "local_loss",
    "previous_audit_hash",
    "privacy_epsilon_spent",
    "clipped_l2_norm",
    "clip_scale",
    "accepted",
    "rejection_reasons",
    "update_hash",
}
_FORBIDDEN_UPDATE_SUBSTRINGS = ("raw_time_series", "raw_data_series", "raw_data")
_ALLOWED_TRANSPORT_DECLARATION_KEYS = {
    "transport",
    "endpoint",
    "owner",
    "auth_policy",
    "tls",
    "secure_channel",
    "replay_supported",
    "local_path_evidence",
    "operator_approved",
}


@dataclass(frozen=True)
class FederatedTransportEnvelope:
    """Signed/hash-linked transport envelope around one node audit update."""

    schema_name: str
    schema_version: str
    batch_id: str
    sequence_position: int
    node_id: str
    node_sequence: int
    envelope_id: str
    parent_envelope_hash: str
    node_update_audit_record: tuple[tuple[str, object], ...]
    node_update_audit_hash: str
    envelope_signature: str
    envelope_hash: str
    transport_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe audit evidence for this transport envelope."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "sequence_position": self.sequence_position,
            "node_id": self.node_id,
            "node_sequence": self.node_sequence,
            "envelope_id": self.envelope_id,
            "parent_envelope_hash": self.parent_envelope_hash,
            "node_update_audit_record": [
                [key, value] for key, value in self.node_update_audit_record
            ],
            "node_update_audit_hash": self.node_update_audit_hash,
            "envelope_signature": self.envelope_signature,
            "envelope_hash": self.envelope_hash,
            "transport_execution_permitted": self.transport_execution_permitted,
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
        }


@dataclass(frozen=True)
class FederatedTransportReplayLedger:
    """Replay result for an ordered transport batch."""

    schema_name: str
    schema_version: str
    batch_id: str
    envelope_count: int
    envelope_ids: tuple[str, ...]
    node_last_sequences: tuple[tuple[str, int], ...]
    replay_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe replay evidence."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "envelope_count": self.envelope_count,
            "envelope_ids": list(self.envelope_ids),
            "node_last_sequences": [list(pair) for pair in self.node_last_sequences],
            "replay_hash": self.replay_hash,
        }


@dataclass(frozen=True)
class FederatedTransportDeploymentPreflightManifest:
    """Deterministic, review-only preflight manifest for transport deployment."""

    schema_name: str
    schema_version: str
    batch_id: str
    preflight_id: str
    transport: str
    transport_endpoint: str
    transport_audit_record: tuple[tuple[str, object], ...]
    transport_audit_hash: str
    replay_ledger_hash: str
    transport_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    preflight_signature: str
    preflight_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe preflight audit evidence."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "preflight_id": self.preflight_id,
            "transport": self.transport,
            "transport_endpoint": self.transport_endpoint,
            "transport_audit_record": [
                [key, value] for key, value in self.transport_audit_record
            ],
            "transport_audit_hash": self.transport_audit_hash,
            "replay_ledger_hash": self.replay_ledger_hash,
            "transport_execution_permitted": self.transport_execution_permitted,
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "preflight_signature": self.preflight_signature,
            "preflight_hash": self.preflight_hash,
        }


def build_signed_transport_envelopes(
    node_update_audit_records: Sequence[Mapping[str, object]],
    *,
    schema_name: str = _DEFAULT_SCHEMA_NAME,
    schema_version: str = _DEFAULT_SCHEMA_VERSION,
    batch_id: str | None = None,
) -> tuple[FederatedTransportEnvelope, ...]:
    """Build deterministic hash-linked envelopes from node audit records."""
    if not isinstance(node_update_audit_records, Sequence) or isinstance(
        node_update_audit_records, (str, bytes, bytearray)
    ):
        raise ValueError("node_update_audit_records must be a sequence of mappings")
    if not node_update_audit_records:
        raise ValueError("node_update_audit_records must be non-empty")
    if schema_name not in _ALLOWED_SCHEMA_NAMES:
        raise ValueError(
            f"schema_name must be one of {_sorted_repr(_ALLOWED_SCHEMA_NAMES)}"
        )
    if schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {_sorted_repr(_ALLOWED_SCHEMA_VERSIONS)}"
        )

    validated_records = tuple(
        _normalise_update_record(record) for record in node_update_audit_records
    )
    generated_batch_id = batch_id or _stable_hash(
        {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "records": [record["update_hash"] for record in validated_records],
        }
    )
    node_sequence_counters: dict[str, int] = {}
    node_parent_hashes: dict[str, str] = {}
    built_envelopes: list[FederatedTransportEnvelope] = []
    seen_envelope_ids: set[str] = set()

    for position, record in enumerate(validated_records, start=1):
        node_id = _text(record["node_id"], "node_id")
        node_sequence = node_sequence_counters.get(node_id, 0) + 1
        node_sequence_counters[node_id] = node_sequence

        parent_hash = node_parent_hashes.get(node_id, _ZERO_SHA256)
        if not _is_sha256(parent_hash):
            raise ValueError(f"invalid parent hash for node '{node_id}'")

        envelope_id = _stable_hash(
            {
                "schema_name": schema_name,
                "schema_version": schema_version,
                "batch_id": generated_batch_id,
                "sequence_position": position,
                "node_id": node_id,
                "node_sequence": node_sequence,
                "parent_hash": parent_hash,
                "node_update_hash": record["update_hash"],
            }
        )
        if envelope_id in seen_envelope_ids:
            raise ValueError(f"duplicate envelope id generated for record {position}")
        seen_envelope_ids.add(envelope_id)

        envelope_signature = _build_envelope_signature(
            schema_name=schema_name,
            schema_version=schema_version,
            batch_id=generated_batch_id,
            sequence_position=position,
            node_id=node_id,
            node_sequence=node_sequence,
            parent_hash=parent_hash,
            node_update_hash=record["update_hash"],
            node_update_record=record["payload"],
        )

        envelope_hash = _stable_hash(
            {
                "envelope_id": envelope_id,
                "envelope_signature": envelope_signature,
            }
        )

        envelope = FederatedTransportEnvelope(
            schema_name=schema_name,
            schema_version=schema_version,
            batch_id=generated_batch_id,
            sequence_position=position,
            node_id=node_id,
            node_sequence=node_sequence,
            envelope_id=envelope_id,
            parent_envelope_hash=parent_hash,
            node_update_audit_record=tuple(record["payload"]),
            node_update_audit_hash=record["update_hash"],
            envelope_signature=envelope_signature,
            envelope_hash=envelope_hash,
            transport_execution_permitted=False,
            raw_data_export_permitted=False,
            operator_review_required=True,
        )
        node_parent_hashes[node_id] = envelope_hash
        built_envelopes.append(envelope)

    return tuple(built_envelopes)


def validate_federated_transport_batch(
    envelopes: Sequence[FederatedTransportEnvelope],
) -> tuple[FederatedTransportEnvelope, ...]:
    """Validate deterministic hash-links and ordering for an ordered transport batch."""
    if not isinstance(envelopes, Sequence) or isinstance(
        envelopes, (str, bytes, bytearray)
    ):
        raise ValueError("envelopes must be a sequence of transport envelopes")
    if not envelopes:
        raise ValueError("envelopes must be non-empty")

    first = envelopes[0]
    if first.schema_name not in _ALLOWED_SCHEMA_NAMES:
        raise ValueError(
            f"schema_name must be one of {_sorted_repr(_ALLOWED_SCHEMA_NAMES)}"
        )
    if first.schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {_sorted_repr(_ALLOWED_SCHEMA_VERSIONS)}"
        )

    seen_envelope_ids: set[str] = set()
    node_last_sequence: dict[str, int] = {}
    node_last_hash: dict[str, str] = {}
    for expected_position, envelope in enumerate(envelopes, start=1):
        if not isinstance(envelope, FederatedTransportEnvelope):
            raise ValueError("envelopes must be FederatedTransportEnvelope instances")

        if envelope.schema_name != first.schema_name:
            raise ValueError("mixed schema_name in transport batch")
        if envelope.schema_version != first.schema_version:
            raise ValueError("mixed schema_version in transport batch")
        if envelope.batch_id != first.batch_id:
            raise ValueError("mixed batch_id in transport batch")
        if envelope.sequence_position != expected_position:
            raise ValueError(
                "envelopes must preserve contiguous sequence_position ordering"
            )
        if envelope.node_id == "":
            raise ValueError("node_id must be non-empty")
        if not _is_sha256(envelope.node_update_audit_hash):
            raise ValueError(
                f"invalid node_update_audit_hash for envelope {envelope.envelope_id}"
            )
        if not _is_sha256(envelope.envelope_hash):
            raise ValueError(
                f"invalid envelope_hash for envelope {envelope.envelope_id}"
            )
        if not _is_sha256(envelope.envelope_signature):
            raise ValueError(
                f"invalid envelope_signature for envelope {envelope.envelope_id}"
            )
        if envelope.envelope_id in seen_envelope_ids:
            raise ValueError(f"duplicate envelope id '{envelope.envelope_id}'")
        seen_envelope_ids.add(envelope.envelope_id)

        if envelope.transport_execution_permitted is not False:
            raise ValueError("transport_execution_permitted must be False")
        if envelope.raw_data_export_permitted is not False:
            raise ValueError("raw_data_export_permitted must be False")
        if envelope.operator_review_required is not True:
            raise ValueError("operator_review_required must be True")

        normalised_node_update = _normalise_update_record(
            dict(envelope.node_update_audit_record)
        )
        if normalised_node_update["update_hash"] != envelope.node_update_audit_hash:
            raise ValueError(
                f"node update hash mismatch for envelope '{envelope.envelope_id}'"
            )

        expected_parent = node_last_hash.get(envelope.node_id, _ZERO_SHA256)
        if envelope.parent_envelope_hash != expected_parent:
            raise ValueError(
                f"parent hash mismatch for node '{envelope.node_id}' at "
                f"position {envelope.sequence_position}"
            )

        expected_node_sequence = node_last_sequence.get(envelope.node_id, 0) + 1
        if envelope.node_sequence != expected_node_sequence:
            raise ValueError(
                f"non-monotonic node_sequence for node '{envelope.node_id}': "
                f"{envelope.node_sequence}"
            )

        expected_signature = _build_envelope_signature(
            schema_name=envelope.schema_name,
            schema_version=envelope.schema_version,
            batch_id=envelope.batch_id,
            sequence_position=envelope.sequence_position,
            node_id=envelope.node_id,
            node_sequence=envelope.node_sequence,
            parent_hash=envelope.parent_envelope_hash,
            node_update_hash=envelope.node_update_audit_hash,
            node_update_record=tuple(
                [tuple(kv) for kv in envelope.node_update_audit_record]
            ),
        )
        expected_hash = _stable_hash(
            {
                "envelope_id": envelope.envelope_id,
                "envelope_signature": expected_signature,
            }
        )
        if envelope.envelope_signature != expected_signature:
            raise ValueError(
                f"signature mismatch for envelope '{envelope.envelope_id}'"
            )
        if envelope.envelope_hash != expected_hash:
            raise ValueError(f"hash mismatch for envelope '{envelope.envelope_id}'")

        node_last_sequence[envelope.node_id] = envelope.node_sequence
        node_last_hash[envelope.node_id] = envelope.envelope_hash

    return tuple(envelopes)


def replay_federated_transport_batch(
    envelopes: Sequence[FederatedTransportEnvelope],
) -> FederatedTransportReplayLedger:
    """Replay and materialise a deterministic digest for an ordered transport batch."""
    validated = validate_federated_transport_batch(envelopes)
    first = validated[0]
    replay_hash = _stable_hash(
        {
            "schema_name": first.schema_name,
            "schema_version": first.schema_version,
            "batch_id": first.batch_id,
            "envelopes": [envelope.to_audit_record() for envelope in validated],
        }
    )
    node_last = tuple(sorted(_group_last_sequence(validated), key=lambda item: item[0]))
    return FederatedTransportReplayLedger(
        schema_name=first.schema_name,
        schema_version=first.schema_version,
        batch_id=first.batch_id,
        envelope_count=len(validated),
        envelope_ids=tuple(envelope.envelope_id for envelope in validated),
        node_last_sequences=node_last,
        replay_hash=replay_hash,
    )


def build_transport_deployment_preflight_manifest(
    transport_declaration: Mapping[str, object],
    *,
    replay_ledger: FederatedTransportReplayLedger,
    schema_name: str = _DEFAULT_SCHEMA_NAME,
    schema_version: str = _DEFAULT_SCHEMA_VERSION,
    batch_id: str | None = None,
) -> FederatedTransportDeploymentPreflightManifest:
    """Build deterministic transport preflight evidence."""
    if schema_name not in _ALLOWED_SCHEMA_NAMES:
        raise ValueError(
            f"schema_name must be one of {_sorted_repr(_ALLOWED_SCHEMA_NAMES)}"
        )
    if schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {_sorted_repr(_ALLOWED_SCHEMA_VERSIONS)}"
        )
    if not isinstance(replay_ledger, FederatedTransportReplayLedger):
        raise ValueError("replay_ledger must be a FederatedTransportReplayLedger")
    if replay_ledger.schema_name != schema_name:
        raise ValueError("replay_ledger.schema_name must match schema_name")
    if replay_ledger.schema_version != schema_version:
        raise ValueError("replay_ledger.schema_version must match schema_version")
    if replay_ledger.batch_id == "":
        raise ValueError("replay_ledger.batch_id must be non-empty")
    if not _is_sha256(replay_ledger.replay_hash):
        raise ValueError("replay_ledger.replay_hash must be a SHA-256 digest")

    normalised_declaration = _normalise_transport_declaration(
        _normalise_transport_input(transport_declaration)
    )

    resolved_batch_id = _text(batch_id or replay_ledger.batch_id, "batch_id")
    if resolved_batch_id != replay_ledger.batch_id:
        raise ValueError("batch_id must match replay_ledger.batch_id")

    transport_audit_record = tuple(
        (key, value)
        for key, value in (
            ("transport", normalised_declaration["transport"]),
            ("endpoint", normalised_declaration["endpoint"]),
            ("owner", normalised_declaration["owner"]),
            ("auth_policy", normalised_declaration["auth_policy"]),
            ("secure_channel", normalised_declaration["secure_channel"]),
            ("replay_supported", normalised_declaration["replay_supported"]),
            ("operator_approved", normalised_declaration["operator_approved"]),
            ("local_path_evidence", normalised_declaration["local_path_evidence"]),
        )
    )
    transport_audit_hash = _stable_hash(
        {
            "transport_audit_record": [list(pair) for pair in transport_audit_record],
            "replay_ledger_hash": replay_ledger.replay_hash,
        }
    )
    preflight_signature = _build_transport_preflight_signature(
        schema_name=schema_name,
        schema_version=schema_version,
        batch_id=resolved_batch_id,
        transport=normalised_declaration["transport"],
        transport_endpoint=normalised_declaration["endpoint"],
        transport_audit_record=transport_audit_record,
        replay_ledger_hash=replay_ledger.replay_hash,
        transport_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
    )
    preflight_id = _stable_hash(
        {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "batch_id": resolved_batch_id,
            "transport_audit_hash": transport_audit_hash,
            "replay_ledger_hash": replay_ledger.replay_hash,
            "transport_endpoint": normalised_declaration["endpoint"],
        }
    )
    preflight_hash = _stable_hash(
        {
            "preflight_id": preflight_id,
            "preflight_signature": preflight_signature,
            "transport_audit_hash": transport_audit_hash,
        }
    )

    manifest = FederatedTransportDeploymentPreflightManifest(
        schema_name=schema_name,
        schema_version=schema_version,
        batch_id=resolved_batch_id,
        preflight_id=preflight_id,
        transport=normalised_declaration["transport"],
        transport_endpoint=normalised_declaration["endpoint"],
        transport_audit_record=transport_audit_record,
        transport_audit_hash=transport_audit_hash,
        replay_ledger_hash=replay_ledger.replay_hash,
        transport_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        preflight_signature=preflight_signature,
        preflight_hash=preflight_hash,
    )
    return validate_transport_deployment_preflight_manifest(manifest)


def validate_transport_deployment_preflight_manifest(
    manifest: FederatedTransportDeploymentPreflightManifest,
) -> FederatedTransportDeploymentPreflightManifest:
    """Validate deterministic transport preflight manifest content and hashes."""
    if not isinstance(manifest, FederatedTransportDeploymentPreflightManifest):
        raise ValueError(
            "manifest must be a FederatedTransportDeploymentPreflightManifest"
        )

    if manifest.schema_name not in _ALLOWED_SCHEMA_NAMES:
        raise ValueError(
            f"schema_name must be one of {_sorted_repr(_ALLOWED_SCHEMA_NAMES)}"
        )
    if manifest.schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {_sorted_repr(_ALLOWED_SCHEMA_VERSIONS)}"
        )
    if manifest.batch_id == "":
        raise ValueError("batch_id must be non-empty")
    if not _is_sha256(manifest.replay_ledger_hash):
        raise ValueError("replay_ledger_hash must be a SHA-256 digest")
    if not _is_sha256(manifest.transport_audit_hash):
        raise ValueError("transport_audit_hash must be a SHA-256 digest")
    if not _is_sha256(manifest.preflight_signature):
        raise ValueError("preflight_signature must be a SHA-256 digest")
    if not _is_sha256(manifest.preflight_hash):
        raise ValueError("preflight_hash must be a SHA-256 digest")

    if manifest.transport_execution_permitted is not False:
        raise ValueError("transport_execution_permitted must be False")
    if manifest.raw_data_export_permitted is not False:
        raise ValueError("raw_data_export_permitted must be False")
    if manifest.operator_review_required is not True:
        raise ValueError("operator_review_required must be True")
    if manifest.non_actuating is not True:
        raise ValueError("non_actuating must be True")

    declaration_from_record = dict(manifest.transport_audit_record)
    if declaration_from_record.get("transport") != manifest.transport:
        raise ValueError("transport mismatch")
    if declaration_from_record.get("endpoint") != manifest.transport_endpoint:
        raise ValueError("transport_endpoint mismatch")
    _normalise_transport_declaration(declaration_from_record)
    expected_signature = _build_transport_preflight_signature(
        schema_name=manifest.schema_name,
        schema_version=manifest.schema_version,
        batch_id=manifest.batch_id,
        transport=manifest.transport,
        transport_endpoint=manifest.transport_endpoint,
        transport_audit_record=manifest.transport_audit_record,
        replay_ledger_hash=manifest.replay_ledger_hash,
        transport_execution_permitted=manifest.transport_execution_permitted,
        raw_data_export_permitted=manifest.raw_data_export_permitted,
        operator_review_required=manifest.operator_review_required,
        non_actuating=manifest.non_actuating,
    )
    if manifest.preflight_signature != expected_signature:
        raise ValueError("preflight_signature mismatch")

    expected_transport_audit_hash = _stable_hash(
        {
            "transport_audit_record": [
                list(pair) for pair in manifest.transport_audit_record
            ],
            "replay_ledger_hash": manifest.replay_ledger_hash,
        }
    )
    if manifest.transport_audit_hash != expected_transport_audit_hash:
        raise ValueError("transport_audit_hash mismatch")

    expected_preflight_id = _stable_hash(
        {
            "schema_name": manifest.schema_name,
            "schema_version": manifest.schema_version,
            "batch_id": manifest.batch_id,
            "transport_audit_hash": manifest.transport_audit_hash,
            "replay_ledger_hash": manifest.replay_ledger_hash,
            "transport_endpoint": manifest.transport_endpoint,
        }
    )
    if manifest.preflight_id != expected_preflight_id:
        raise ValueError("preflight_id mismatch")

    expected_hash = _stable_hash(
        {
            "preflight_id": manifest.preflight_id,
            "preflight_signature": manifest.preflight_signature,
            "transport_audit_hash": manifest.transport_audit_hash,
        }
    )
    if manifest.preflight_hash != expected_hash:
        raise ValueError("preflight_hash mismatch")

    return manifest


def _normalise_transport_input(
    transport_declaration: Mapping[str, object] | object,
) -> Mapping[str, object]:
    if not isinstance(transport_declaration, Mapping):
        raise ValueError("transport declaration must be a mapping")
    return transport_declaration


def _normalise_transport_declaration(
    raw: Mapping[str, object],
) -> dict[str, object]:
    for key in raw:
        if not isinstance(key, str):
            raise ValueError("transport declaration keys must be text")
        if key not in _ALLOWED_TRANSPORT_DECLARATION_KEYS:
            raise ValueError(f"unsupported transport declaration key: {key}")

    transport = _text(raw.get("transport"), "transport").lower()
    if transport not in _ALLOWED_TRANSPORTS:
        raise ValueError(
            f"transport must be one of {_sorted_repr(_ALLOWED_TRANSPORTS)}"
        )
    endpoint = _text(raw.get("endpoint"), "endpoint")

    replay_supported = _bool(raw.get("replay_supported", False), "replay_supported")
    operator_approved = _bool(raw.get("operator_approved", False), "operator_approved")
    tls = raw.get("tls")
    secure_channel = raw.get("secure_channel")
    if tls is not None and secure_channel is not None:
        tls_bool = _bool(tls, "tls")
        secure_bool = _bool(secure_channel, "secure_channel")
        if tls_bool != secure_bool:
            raise ValueError("tls and secure_channel values must match")
        secure_channel_value = tls_bool
    elif tls is not None:
        secure_channel_value = _bool(tls, "tls")
    elif secure_channel is not None:
        secure_channel_value = _bool(secure_channel, "secure_channel")
    else:
        secure_channel_value = False

    if transport in _LIVE_TRANSPORTS:
        owner = _text(raw.get("owner"), "owner")
        auth_policy = _text(raw.get("auth_policy"), "auth_policy")
        if not operator_approved:
            raise ValueError("operator approval is required for live transport")
        if not replay_supported:
            raise ValueError("replay support must be true for live transport")
        if not secure_channel_value:
            raise ValueError("live transport must use TLS or secure_channel")
        local_path_evidence = ""
    else:
        owner = ""
        auth_policy = ""
        if raw.get("owner") not in (None, ""):
            raise ValueError("owner is not permitted for jsonl transport")
        if raw.get("auth_policy") not in (None, ""):
            raise ValueError("auth_policy is not permitted for jsonl transport")
        if "tls" in raw or raw.get("secure_channel") not in (None, False):
            raise ValueError("TLS is not applicable to jsonl transport")
        if not replay_supported:
            raise ValueError("jsonl transport must declare replay support")
        local_path_evidence = _text(
            raw.get("local_path_evidence"), "local_path_evidence"
        )

    return {
        "transport": transport,
        "endpoint": endpoint,
        "owner": owner,
        "auth_policy": auth_policy,
        "secure_channel": secure_channel_value,
        "replay_supported": replay_supported,
        "local_path_evidence": local_path_evidence,
        "operator_approved": operator_approved,
    }


def _build_transport_preflight_signature(
    *,
    schema_name: str,
    schema_version: str,
    batch_id: str,
    transport: str,
    transport_endpoint: str,
    transport_audit_record: tuple[tuple[str, object], ...],
    replay_ledger_hash: str,
    transport_execution_permitted: bool,
    raw_data_export_permitted: bool,
    operator_review_required: bool,
    non_actuating: bool,
) -> str:
    return _stable_hash(
        {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "batch_id": batch_id,
            "transport": transport,
            "transport_endpoint": transport_endpoint,
            "transport_audit_record": [list(pair) for pair in transport_audit_record],
            "replay_ledger_hash": replay_ledger_hash,
            "transport_execution_permitted": transport_execution_permitted,
            "raw_data_export_permitted": raw_data_export_permitted,
            "operator_review_required": operator_review_required,
            "non_actuating": non_actuating,
        }
    )


def _normalise_update_record(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("each node update audit record must be a mapping")

    for forbidden in raw:
        if not isinstance(forbidden, str):
            raise ValueError("node update audit keys must be text")
        lower_key = forbidden.lower()
        if any(fragment in lower_key for fragment in _FORBIDDEN_UPDATE_SUBSTRINGS):
            raise ValueError(
                "raw time-series content is not permitted in transport records"
            )

    keys = set(raw.keys())
    required = {
        "node_id",
        "policy_delta",
        "sample_count",
        "local_loss",
        "previous_audit_hash",
        "privacy_epsilon_spent",
        "clipped_l2_norm",
        "clip_scale",
        "accepted",
        "rejection_reasons",
        "update_hash",
    }
    if not required.issubset(keys):
        missing = ", ".join(sorted(required - keys))
        raise ValueError(f"missing required keys: {missing}")
    if keys - required:
        unknown = ", ".join(sorted(keys - required))
        raise ValueError(f"unsupported keys in node update audit record: {unknown}")

    node_id = _text(raw["node_id"], "node_id")
    sample_count = _int(raw["sample_count"], "sample_count")
    if sample_count < 0:
        raise ValueError("sample_count must be >= 0")
    local_loss = _finite_float(raw["local_loss"], "local_loss")
    privacy_epsilon_spent = _finite_float(
        raw["privacy_epsilon_spent"], "privacy_epsilon_spent"
    )
    clipped_l2_norm = _finite_float(raw["clipped_l2_norm"], "clipped_l2_norm")
    clip_scale = _finite_float(raw["clip_scale"], "clip_scale")
    accepted = _bool(raw["accepted"], "accepted")
    rejection_reasons = _string_tuple(raw["rejection_reasons"], "rejection_reasons")
    previous_audit_hash = _sha256_text(
        raw["previous_audit_hash"], "previous_audit_hash"
    )
    update_hash = _sha256_text(raw["update_hash"], "update_hash")
    policy_delta = _normalise_policy_delta(raw["policy_delta"])

    payload = {
        "node_id": node_id,
        "policy_delta": policy_delta,
        "sample_count": sample_count,
        "local_loss": local_loss,
        "previous_audit_hash": previous_audit_hash,
        "privacy_epsilon_spent": privacy_epsilon_spent,
        "clipped_l2_norm": clipped_l2_norm,
        "clip_scale": clip_scale,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "update_hash": update_hash,
    }

    return {
        "node_id": node_id,
        "sample_count": sample_count,
        "local_loss": local_loss,
        "previous_audit_hash": previous_audit_hash,
        "privacy_epsilon_spent": privacy_epsilon_spent,
        "clipped_l2_norm": clipped_l2_norm,
        "clip_scale": clip_scale,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "policy_delta": policy_delta,
        "update_hash": update_hash,
        "payload": tuple((key, value) for key, value in payload.items()),
    }


def _normalise_policy_delta(raw: object) -> tuple[tuple[str, float], ...]:
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = []
        for item in raw:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError(
                    "policy_delta must be mapping entries or key-value pairs"
                )
            if isinstance(item[0], str):
                items.append((item[0], item[1]))
            else:
                raise ValueError("policy_delta keys must be text")
    else:
        raise ValueError("policy_delta must be a mapping")
    if not items:
        raise ValueError("policy_delta must be non-empty")
    pairs: list[tuple[str, float]] = []
    for key, value in items:
        key_text = _text(key, "policy_delta key")
        value_float = _finite_float(value, f"policy_delta[{key_text}]")
        pairs.append((key_text, float(value_float)))
    return tuple(sorted(pairs, key=lambda item: item[0]))


def _build_envelope_signature(
    *,
    schema_name: str,
    schema_version: str,
    batch_id: str,
    sequence_position: int,
    node_id: str,
    node_sequence: int,
    parent_hash: str,
    node_update_hash: str,
    node_update_record: tuple[tuple[str, object], ...]
    | tuple[tuple[str, object], tuple[str, object]]
    | Sequence[tuple[str, object]],
) -> str:
    return _stable_hash(
        {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "batch_id": batch_id,
            "sequence_position": sequence_position,
            "node_id": node_id,
            "node_sequence": node_sequence,
            "parent_hash": parent_hash,
            "node_update_hash": node_update_hash,
            "node_update_record": [list(pair) for pair in node_update_record],
            "transport_execution_permitted": False,
            "raw_data_export_permitted": False,
            "operator_review_required": True,
        }
    )


def _string_tuple(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of text")
    return tuple(_text(value, f"{name} item") for value in raw)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a text value")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _sha256_text(value: object, name: str) -> str:
    return _text(value, name) if _is_sha256(value) else _raise_sha256(name)


def _raise_sha256(name: str) -> str:
    raise ValueError(f"{name} must be a 64-character hex SHA-256 digest")


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    return value_float


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _group_last_sequence(
    envelopes: Sequence[FederatedTransportEnvelope],
) -> dict[str, int]:
    node_last: dict[str, int] = {}
    for envelope in envelopes:
        node_last[envelope.node_id] = envelope.node_sequence
    return node_last


def _sorted_repr(items: set[str] | frozenset[str]) -> str:
    return ", ".join(sorted(items))


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _stable_json(payload: object) -> str:
    return json.dumps(
        _to_json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _to_json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_ready(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("numbers in transport records must be finite")
        return number
    if isinstance(value, (str, int, bool, type(None))):
        return value
    raise ValueError("transport payload contains unsupported JSON type")
