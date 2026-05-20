# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline secure aggregation manifest

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

__all__ = [
    "SecureAggregationConfig",
    "SecureNodeCommitment",
    "FederatedSecureAggregationManifest",
    "build_federated_secure_aggregation_manifest",
]


@dataclass(frozen=True)
class SecureAggregationConfig:
    """Offline policy for deterministic manifest-only secure aggregation."""

    clipping_norm: float = 1.0
    min_node_count: int = 3
    epsilon: float = 3.0
    delta: float = 1e-6

    def __post_init__(self) -> None:
        _positive_float(self.clipping_norm, "clipping_norm")
        _positive_int(self.min_node_count, "min_node_count")
        _positive_float(self.epsilon, "epsilon")
        _positive_float(self.delta, "delta")
        if self.delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")


@dataclass(frozen=True)
class SecureNodeCommitment:
    """Validated masked node commitment used in secure aggregation review."""

    node_id: str
    masked_policy_delta: tuple[tuple[str, float], ...]
    sample_count: int
    share_commitment: str
    share_commitment_hash: str
    share_hash: str
    masked_delta_hash: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    update_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record for this node commitment."""
        return {
            "node_id": self.node_id,
            "masked_policy_delta": list(self.masked_policy_delta),
            "sample_count": self.sample_count,
            "share_commitment": self.share_commitment,
            "share_commitment_hash": self.share_commitment_hash,
            "share_hash": self.share_hash,
            "masked_delta_hash": self.masked_delta_hash,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "update_hash": self.update_hash,
        }


@dataclass(frozen=True)
class FederatedSecureAggregationManifest:
    """Manifest for deterministic offline secure aggregation review only."""

    schema_name: str
    schema_version: str
    config: SecureAggregationConfig
    required_policy_keys: tuple[str, ...]
    node_commitments: tuple[SecureNodeCommitment, ...]
    accepted_node_count: int
    rejected_node_count: int
    total_sample_count: int
    aggregate_masked_delta: tuple[tuple[str, float], ...]
    aggregate_masked_delta_hash: str
    secure_aggregation_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    quorum_met: bool
    claim_boundary: str
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe aggregate audit record."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "clipping_norm": self.config.clipping_norm,
            "min_node_count": self.config.min_node_count,
            "epsilon": self.config.epsilon,
            "delta": self.config.delta,
            "required_policy_keys": list(self.required_policy_keys),
            "node_commitments": [
                node.to_audit_record() for node in self.node_commitments
            ],
            "accepted_node_count": self.accepted_node_count,
            "rejected_node_count": self.rejected_node_count,
            "total_sample_count": self.total_sample_count,
            "aggregate_masked_delta": list(self.aggregate_masked_delta),
            "aggregate_masked_delta_hash": self.aggregate_masked_delta_hash,
            "secure_aggregation_execution_permitted": (
                self.secure_aggregation_execution_permitted
            ),
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "quorum_met": self.quorum_met,
            "claim_boundary": self.claim_boundary,
            "report_hash": self.report_hash,
        }


def build_federated_secure_aggregation_manifest(
    node_commitments: Sequence[Mapping[str, object]],
    *,
    required_policy_keys: Sequence[str] | None = None,
    clipping_norm: float = 1.0,
    min_node_count: int = 3,
    epsilon: float = 3.0,
    delta: float = 1e-6,
) -> FederatedSecureAggregationManifest:
    """Build a deterministic secure aggregation manifest."""

    config = SecureAggregationConfig(
        clipping_norm=clipping_norm,
        min_node_count=min_node_count,
        epsilon=epsilon,
        delta=delta,
    )
    if not isinstance(node_commitments, Sequence) or isinstance(
        node_commitments, (str, bytes, bytearray)
    ):
        raise ValueError("node_commitments must be a sequence of mappings")
    if not node_commitments:
        raise ValueError("node_commitments must be non-empty")

    keys = _resolve_required_policy_keys(required_policy_keys, node_commitments)
    seen_node_ids: set[str] = set()
    validated = tuple(
        _validate_node_commitment(
            raw,
            required_policy_keys=keys,
            config=config,
            seen_node_ids=seen_node_ids,
        )
        for raw in node_commitments
    )
    validated = tuple(sorted(validated, key=lambda entry: entry.node_id))
    accepted = tuple(node for node in validated if node.accepted)
    rejected = tuple(node for node in validated if not node.accepted)

    if len(accepted) < config.min_node_count:
        raise ValueError(
            "quorum_not_met: accepted node commitments below minimum node count"
        )

    aggregate_masked_delta, total_sample_count = _weighted_masked_average(
        accepted, keys
    )
    aggregate_masked_delta_hash = _stable_hash(
        {
            "aggregate_masked_delta": list(aggregate_masked_delta),
            "accepted_node_ids": sorted(node.node_id for node in accepted),
            "required_policy_keys": list(keys),
        }
    )

    report = FederatedSecureAggregationManifest(
        schema_name="federated_secure_aggregation_manifest",
        schema_version="0.1.0",
        config=config,
        required_policy_keys=keys,
        node_commitments=validated,
        accepted_node_count=len(accepted),
        rejected_node_count=len(rejected),
        total_sample_count=total_sample_count,
        aggregate_masked_delta=aggregate_masked_delta,
        aggregate_masked_delta_hash=aggregate_masked_delta_hash,
        secure_aggregation_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        quorum_met=True,
        claim_boundary="offline_review_only_no_live_transport",
        report_hash="",
    )
    return FederatedSecureAggregationManifest(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        config=report.config,
        required_policy_keys=report.required_policy_keys,
        node_commitments=report.node_commitments,
        accepted_node_count=report.accepted_node_count,
        rejected_node_count=report.rejected_node_count,
        total_sample_count=report.total_sample_count,
        aggregate_masked_delta=report.aggregate_masked_delta,
        aggregate_masked_delta_hash=report.aggregate_masked_delta_hash,
        secure_aggregation_execution_permitted=report.secure_aggregation_execution_permitted,
        raw_data_export_permitted=report.raw_data_export_permitted,
        operator_review_required=report.operator_review_required,
        non_actuating=report.non_actuating,
        quorum_met=report.quorum_met,
        claim_boundary=report.claim_boundary,
        report_hash=_stable_hash(report.to_audit_record()),
    )


def _resolve_required_policy_keys(
    required_policy_keys: Sequence[str] | None,
    node_commitments: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if required_policy_keys is not None:
        if not isinstance(required_policy_keys, Sequence) or isinstance(
            required_policy_keys, (str, bytes, bytearray)
        ):
            raise ValueError("required_policy_keys must be a sequence of strings")
        keys = tuple(
            _non_empty_text(key, "required_policy_key") for key in required_policy_keys
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("required_policy_keys must be non-empty and unique")
        return tuple(sorted(keys))

    discovered: set[str] = set()
    for raw in node_commitments:
        if not isinstance(raw, Mapping):
            raise ValueError("each node commitment must be a mapping")
        delta = raw.get("masked_policy_delta")
        if isinstance(delta, Mapping):
            discovered.update(_non_empty_text(key, "policy key") for key in delta)
        else:
            raise ValueError("masked_policy_delta must be a non-empty mapping")
    if not discovered:
        raise ValueError("masked_policy_delta must be a non-empty mapping")
    return tuple(sorted(discovered))


def _validate_node_commitment(
    raw: Mapping[str, object],
    *,
    required_policy_keys: tuple[str, ...],
    config: SecureAggregationConfig,
    seen_node_ids: set[str],
) -> SecureNodeCommitment:
    if not isinstance(raw, Mapping):
        raise ValueError("each node commitment must be a mapping")

    aggregated_from_masked_audit_values = bool(
        raw.get("aggregated_from_masked_audit_values", False)
    )
    if (
        any(
            key in raw
            for key in (
                "raw_policy_delta",
                "policy_delta",
                "raw_time_series",
                "samples",
                "time_series",
            )
        )
        and not aggregated_from_masked_audit_values
    ):
        raise ValueError("raw deltas or time-series require masked-audit provenance")

    node_id = _non_empty_text(raw.get("node_id"), "node_id")
    if node_id in seen_node_ids:
        raise ValueError(f"node_id must be unique, duplicated: {node_id}")
    seen_node_ids.add(node_id)

    sample_count = _positive_int(raw.get("sample_count"), "sample_count")
    share_commitment = _non_empty_text(raw.get("share_commitment"), "share_commitment")
    share_commitment_hash = _sha256_hex(
        raw.get("share_commitment_hash"), "share_commitment_hash"
    )
    share_hash = _sha256_hex(raw.get("share_hash"), "share_hash")
    raw_delta_map = raw.get("masked_policy_delta")
    if not isinstance(raw_delta_map, Mapping) or not raw_delta_map:
        raise ValueError("masked_policy_delta must be a non-empty mapping")

    missing = [key for key in required_policy_keys if key not in raw_delta_map]
    if missing:
        raise ValueError(f"masked_policy_delta missing required keys: {missing}")

    values = {
        _non_empty_text(key, "policy key"): _finite_float(
            raw_delta_map[key], f"policy_delta[{key}]"
        )
        for key in required_policy_keys
    }
    reasons: list[str] = []

    l2_norm = math.sqrt(sum(value * value for value in values.values()))
    scale = 1.0 if l2_norm <= config.clipping_norm else config.clipping_norm / l2_norm
    if scale < 1.0:
        reasons.append("masked_delta_clipped")

    masked_delta = tuple(
        (key, float(values[key] * scale)) for key in required_policy_keys
    )
    expected_share_hash = _stable_hash(
        {
            "node_id": node_id,
            "masked_policy_delta": list(masked_delta),
        }
    )
    if share_hash != expected_share_hash:
        reasons.append("share_hash_mismatch")

    expected_commitment_hash = _stable_hash(
        {
            "node_id": node_id,
            "share_commitment": share_commitment,
        }
    )
    if share_commitment_hash != expected_commitment_hash:
        reasons.append("share_commitment_hash_mismatch")

    masked_delta_hash = _stable_hash(
        {"node_id": node_id, "masked_policy_delta": list(masked_delta)}
    )
    accepted = len(reasons) == 0

    payload = {
        "node_id": node_id,
        "masked_policy_delta": list(masked_delta),
        "sample_count": sample_count,
        "share_commitment": share_commitment,
        "share_commitment_hash": share_commitment_hash,
        "share_hash": share_hash,
        "masked_delta_hash": masked_delta_hash,
    }
    return SecureNodeCommitment(
        node_id=node_id,
        masked_policy_delta=masked_delta,
        sample_count=sample_count,
        share_commitment=share_commitment,
        share_commitment_hash=share_commitment_hash,
        share_hash=share_hash,
        masked_delta_hash=masked_delta_hash,
        accepted=accepted,
        rejection_reasons=tuple(reasons),
        update_hash=_stable_hash(payload),
    )


def _weighted_masked_average(
    accepted_nodes: tuple[SecureNodeCommitment, ...],
    required_policy_keys: tuple[str, ...],
) -> tuple[tuple[str, float], int]:
    total_sample_count = sum(node.sample_count for node in accepted_nodes)
    if total_sample_count <= 0:
        raise ValueError("total_sample_count must be positive for aggregation")

    aggregate: dict[str, float] = dict.fromkeys(required_policy_keys, 0.0)
    for node in accepted_nodes:
        for key, value in node.masked_policy_delta:
            aggregate[key] += value * node.sample_count
    for key in required_policy_keys:
        aggregate[key] /= float(total_sample_count)
    return tuple(
        (key, aggregate[key]) for key in required_policy_keys
    ), total_sample_count


def _sha256_hex(value: object, field: str) -> str:
    text = _non_empty_text(value, field)
    digest = _normalise_hex(text)
    if not _is_sha256_hex(digest):
        raise ValueError(
            f"{field} must be a 64-character lower-case SHA-256 hex digest"
        )
    return digest


def _normalise_hex(value: str) -> str:
    return value.lower()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive int")
    return int(value)


def _positive_float(value: object, field: str) -> float:
    number = _to_float(value, field)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive float")
    return float(number)


def _finite_float(value: object, field: str) -> float:
    number = _to_float(value, field)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite float")
    return float(number)


def _to_float(value: object, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{field} must be a real number")
    return float(value)


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
