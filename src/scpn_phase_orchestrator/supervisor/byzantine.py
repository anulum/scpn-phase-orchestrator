# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Byzantine meta-orchestrator

"""Offline Byzantine-tolerant policy proposal consensus manifests."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256

__all__ = ["build_bft_meta_orchestrator_manifest", "sign_policy_proposal"]


def sign_policy_proposal(
    node_id: str,
    payload: Mapping[str, object],
    previous_audit_hash: str,
    signing_key: str,
) -> dict[str, object]:
    """Return a deterministic signed policy proposal record.

    Parameters
    ----------
    node_id : str
        Identifier of the proposing node.
    payload : Mapping[str, object]
        The policy-proposal payload to sign.
    previous_audit_hash : str
        Hash of the previous audit record in the chain.
    signing_key : str
        HMAC signing key for the record.

    Returns
    -------
    dict[str, object]
        The deterministic signed policy-proposal record.
    """
    clean_node = _require_text(node_id, "node_id")
    clean_key = _require_text(signing_key, "signing_key")
    _require_hash(previous_audit_hash, "previous_audit_hash")
    payload_hash = _hash_payload(payload)
    signature_payload = _signature_payload(
        clean_node,
        payload_hash,
        previous_audit_hash,
    )
    signature = hmac.new(
        clean_key.encode("utf-8"),
        signature_payload.encode("utf-8"),
        sha256,
    ).hexdigest()
    return {
        "node_id": clean_node,
        "payload": _json_round_trip(payload),
        "payload_hash": payload_hash,
        "previous_audit_hash": previous_audit_hash,
        "signature": signature,
        "signature_algorithm": "hmac-sha256",
    }


def build_bft_meta_orchestrator_manifest(
    proposals: Sequence[Mapping[str, object]],
    keyring: Mapping[str, str],
    *,
    quorum: int,
) -> dict[str, object]:
    """Build a review-only three-node BFT consensus manifest.

    Parameters
    ----------
    proposals : Sequence[Mapping[str, object]]
        Signed policy proposals from the participating nodes.
    keyring : Mapping[str, str]
        Mapping of node id to its verification key.
    quorum : int
        Number of agreeing nodes required for consensus.

    Returns
    -------
    dict[str, object]
        The review-only BFT consensus manifest.

    Raises
    ------
    ValueError
        If the proposals fail signature or quorum checks.
    """
    if quorum < 1:
        raise ValueError("quorum must be >= 1")
    if isinstance(proposals, Mapping) or not proposals:
        raise ValueError("proposals must be a non-empty sequence")
    if isinstance(keyring, Sequence) or not keyring:
        raise ValueError("keyring must be a non-empty mapping")

    verified: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    blocked_reasons: list[str] = []
    seen_nodes: set[str] = set()
    for proposal in proposals:
        record = _verify_proposal(proposal, keyring)
        node_id = str(record["node_id"])
        if node_id in seen_nodes:
            record["valid"] = False
            record["reason"] = f"{node_id} duplicate proposal"
        seen_nodes.add(node_id)
        if record["valid"] is True:
            verified.append(record)
        else:
            rejected.append(record)
            blocked_reasons.append(str(record["reason"]))

    accepted_group = _accepted_quorum_group(verified, quorum)
    if accepted_group is None:
        accepted_group = []
        blocked_reasons.append("valid quorum not reached")
    accepted_node_ids = sorted(str(record["node_id"]) for record in accepted_group)
    accepted_node_set = set(accepted_node_ids)
    non_winning = (
        [
            str(record["node_id"])
            for record in verified
            if str(record["node_id"]) not in accepted_node_set
        ]
        if accepted_node_ids
        else []
    )
    rejected_node_ids = sorted(
        [*(str(record["node_id"]) for record in rejected), *non_winning]
    )
    consensus_hash = str(accepted_group[0]["payload_hash"]) if accepted_group else ""
    audit_chain_hash = _audit_chain_hash(accepted_group)
    manifest: dict[str, object] = {
        "manifest_kind": "bft_meta_orchestrator_manifest",
        "schema_version": 1,
        "status": "accepted" if accepted_group else "blocked",
        "quorum": quorum,
        "node_count": len(proposals),
        "accepted_node_ids": accepted_node_ids,
        "rejected_node_ids": rejected_node_ids,
        "consensus_hash": consensus_hash,
        "audit_chain_hash": audit_chain_hash,
        "blocked_reasons": _dedupe(blocked_reasons),
        "actuation_permitted": False,
        "network_opened": False,
        "operator_commands": [
            "review bft_meta_orchestrator_manifest.json",
            "apply accepted policy only through the normal supervisor review gate",
        ],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def _verify_proposal(
    proposal: Mapping[str, object],
    keyring: Mapping[str, str],
) -> dict[str, object]:
    if not isinstance(proposal, Mapping):
        raise ValueError("proposal must be a mapping")
    node_id = _require_text(proposal.get("node_id"), "node_id")
    payload = proposal.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    payload_hash = _require_hash(proposal.get("payload_hash"), "payload_hash")
    previous_audit_hash = _require_hash(
        proposal.get("previous_audit_hash"),
        "previous_audit_hash",
    )
    signature = _require_hash(proposal.get("signature"), "signature")
    signing_key = keyring.get(node_id)
    if not isinstance(signing_key, str) or not signing_key:
        return {
            "node_id": node_id,
            "payload_hash": payload_hash,
            "previous_audit_hash": previous_audit_hash,
            "valid": False,
            "reason": f"{node_id} missing signing key",
        }
    expected_payload_hash = _hash_payload(payload)
    if expected_payload_hash != payload_hash:
        return {
            "node_id": node_id,
            "payload_hash": payload_hash,
            "previous_audit_hash": previous_audit_hash,
            "valid": False,
            "reason": f"{node_id} payload hash mismatch",
        }
    expected_signature = hmac.new(
        signing_key.encode("utf-8"),
        _signature_payload(node_id, payload_hash, previous_audit_hash).encode("utf-8"),
        sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        return {
            "node_id": node_id,
            "payload_hash": payload_hash,
            "previous_audit_hash": previous_audit_hash,
            "valid": False,
            "reason": f"{node_id} signature verification failed",
        }
    return {
        "node_id": node_id,
        "payload_hash": payload_hash,
        "previous_audit_hash": previous_audit_hash,
        "valid": True,
        "reason": "",
    }


def _accepted_quorum_group(
    verified: Sequence[Mapping[str, object]],
    quorum: int,
) -> list[Mapping[str, object]] | None:
    by_hash: dict[str, list[Mapping[str, object]]] = {}
    for record in verified:
        by_hash.setdefault(str(record["payload_hash"]), []).append(record)
    groups = sorted(
        by_hash.values(),
        key=lambda group: (-len(group), str(group[0]["payload_hash"])),
    )
    for group in groups:
        if len(group) >= quorum:
            return sorted(group, key=lambda record: str(record["node_id"]))
    return None


def _audit_chain_hash(records: Sequence[Mapping[str, object]]) -> str:
    if not records:
        return ""
    payload = [
        {
            "node_id": record["node_id"],
            "payload_hash": record["payload_hash"],
            "previous_audit_hash": record["previous_audit_hash"],
        }
        for record in records
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_payload(payload: Mapping[str, object]) -> str:
    return sha256(
        json.dumps(
            _json_round_trip(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _signature_payload(
    node_id: str, payload_hash: str, previous_audit_hash: str
) -> str:
    return json.dumps(
        {
            "node_id": node_id,
            "payload_hash": payload_hash,
            "previous_audit_hash": previous_audit_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_round_trip(payload: Mapping[str, object]) -> dict[str, object]:
    loaded = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    if not isinstance(loaded, dict):
        raise ValueError("payload must encode as a JSON object")
    return {str(key): value for key, value in loaded.items()}


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex string") from exc
    return value


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
