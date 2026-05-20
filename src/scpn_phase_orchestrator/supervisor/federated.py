# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated meta-orchestrator

"""Review-only federated policy-gradient aggregation manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

__all__ = [
    "FederatedAggregationConfig",
    "FederatedNodeUpdate",
    "FederatedPolicyAggregationReport",
    "build_federated_meta_orchestrator_manifest",
]


@dataclass(frozen=True)
class FederatedAggregationConfig:
    """Privacy and acceptance bounds for one offline aggregation review."""

    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    epsilon: float = 3.0
    delta: float = 1e-6
    min_node_count: int = 3

    def __post_init__(self) -> None:
        _positive_float(self.clipping_norm, "clipping_norm")
        _non_negative_float(self.noise_multiplier, "noise_multiplier")
        _positive_float(self.epsilon, "epsilon")
        _positive_float(self.delta, "delta")
        if self.delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        _positive_int(self.min_node_count, "min_node_count")


@dataclass(frozen=True)
class FederatedNodeUpdate:
    """Validated node-local policy-gradient update without raw time-series."""

    node_id: str
    policy_delta: tuple[tuple[str, float], ...]
    sample_count: int
    local_loss: float
    previous_audit_hash: str
    privacy_epsilon_spent: float
    clipped_l2_norm: float
    clip_scale: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    update_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe node update evidence."""
        return {
            "node_id": self.node_id,
            "policy_delta": [[key, value] for key, value in self.policy_delta],
            "sample_count": self.sample_count,
            "local_loss": self.local_loss,
            "previous_audit_hash": self.previous_audit_hash,
            "privacy_epsilon_spent": self.privacy_epsilon_spent,
            "clipped_l2_norm": self.clipped_l2_norm,
            "clip_scale": self.clip_scale,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "update_hash": self.update_hash,
        }


@dataclass(frozen=True)
class FederatedPolicyAggregationReport:
    """Offline federated aggregation report with explicit safety boundaries."""

    schema_name: str
    schema_version: str
    config: FederatedAggregationConfig
    required_policy_keys: tuple[str, ...]
    node_updates: tuple[FederatedNodeUpdate, ...]
    accepted_node_count: int
    rejected_node_count: int
    total_sample_count: int
    aggregate_delta: tuple[tuple[str, float], ...]
    aggregate_hash: str
    privacy_budget_spent: float
    privacy_budget_remaining: float
    raw_time_series_received: bool
    claim_boundary: str
    operator_review_required: bool
    non_actuating: bool
    execution_disabled: bool
    live_transport_permitted: bool
    raw_data_export_permitted: bool
    actuation_permitted: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe aggregate evidence."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "clipping_norm": self.config.clipping_norm,
            "noise_multiplier": self.config.noise_multiplier,
            "epsilon": self.config.epsilon,
            "delta": self.config.delta,
            "min_node_count": self.config.min_node_count,
            "required_policy_keys": list(self.required_policy_keys),
            "node_updates": [update.to_audit_record() for update in self.node_updates],
            "accepted_node_count": self.accepted_node_count,
            "rejected_node_count": self.rejected_node_count,
            "total_sample_count": self.total_sample_count,
            "aggregate_delta": [[key, value] for key, value in self.aggregate_delta],
            "aggregate_hash": self.aggregate_hash,
            "privacy_budget_spent": self.privacy_budget_spent,
            "privacy_budget_remaining": self.privacy_budget_remaining,
            "raw_time_series_received": self.raw_time_series_received,
            "claim_boundary": self.claim_boundary,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "live_transport_permitted": self.live_transport_permitted,
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "actuation_permitted": self.actuation_permitted,
            "report_hash": self.report_hash,
        }


def build_federated_meta_orchestrator_manifest(
    node_updates: Sequence[Mapping[str, object]],
    *,
    required_policy_keys: Sequence[str] | None = None,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    epsilon: float = 3.0,
    delta: float = 1e-6,
    min_node_count: int = 3,
) -> FederatedPolicyAggregationReport:
    """Build a deterministic review manifest for federated policy aggregation."""
    config = FederatedAggregationConfig(
        clipping_norm=clipping_norm,
        noise_multiplier=noise_multiplier,
        epsilon=epsilon,
        delta=delta,
        min_node_count=min_node_count,
    )
    if not isinstance(node_updates, Sequence) or isinstance(
        node_updates, (str, bytes, bytearray)
    ):
        raise ValueError("node_updates must be a sequence of mappings")
    if not node_updates:
        raise ValueError("node_updates must be non-empty")
    keys = _required_keys(required_policy_keys, node_updates)
    updates = tuple(
        _validate_node_update(raw, required_policy_keys=keys, config=config)
        for raw in node_updates
    )
    accepted = tuple(update for update in updates if update.accepted)
    rejected = tuple(update for update in updates if not update.accepted)
    if len(accepted) < config.min_node_count:
        aggregate_delta: tuple[tuple[str, float], ...] = tuple(
            (key, 0.0) for key in keys
        )
        total_samples = 0
    else:
        aggregate_delta, total_samples = _weighted_average(accepted, keys)
    aggregate_hash = _stable_hash(
        {
            "required_policy_keys": list(keys),
            "aggregate_delta": [[key, value] for key, value in aggregate_delta],
            "accepted_node_ids": [update.node_id for update in accepted],
        }
    )
    privacy_spent = max(
        (update.privacy_epsilon_spent for update in accepted), default=0.0
    )
    report = FederatedPolicyAggregationReport(
        schema_name="federated_meta_orchestrator_policy_aggregation",
        schema_version="0.1.0",
        config=config,
        required_policy_keys=keys,
        node_updates=updates,
        accepted_node_count=len(accepted),
        rejected_node_count=len(rejected),
        total_sample_count=total_samples,
        aggregate_delta=aggregate_delta,
        aggregate_hash=aggregate_hash,
        privacy_budget_spent=privacy_spent,
        privacy_budget_remaining=max(0.0, config.epsilon - privacy_spent),
        raw_time_series_received=False,
        claim_boundary="federated_meta_orchestrator_review_not_live_transport",
        operator_review_required=True,
        non_actuating=True,
        execution_disabled=True,
        live_transport_permitted=False,
        raw_data_export_permitted=False,
        actuation_permitted=False,
        report_hash="",
    )
    return FederatedPolicyAggregationReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        config=report.config,
        required_policy_keys=report.required_policy_keys,
        node_updates=report.node_updates,
        accepted_node_count=report.accepted_node_count,
        rejected_node_count=report.rejected_node_count,
        total_sample_count=report.total_sample_count,
        aggregate_delta=report.aggregate_delta,
        aggregate_hash=report.aggregate_hash,
        privacy_budget_spent=report.privacy_budget_spent,
        privacy_budget_remaining=report.privacy_budget_remaining,
        raw_time_series_received=report.raw_time_series_received,
        claim_boundary=report.claim_boundary,
        operator_review_required=report.operator_review_required,
        non_actuating=report.non_actuating,
        execution_disabled=report.execution_disabled,
        live_transport_permitted=report.live_transport_permitted,
        raw_data_export_permitted=report.raw_data_export_permitted,
        actuation_permitted=report.actuation_permitted,
        report_hash=_stable_hash(report.to_audit_record()),
    )


def _required_keys(
    required_policy_keys: Sequence[str] | None,
    node_updates: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if required_policy_keys is not None:
        if not isinstance(required_policy_keys, Sequence) or isinstance(
            required_policy_keys, (str, bytes, bytearray)
        ):
            raise ValueError("required_policy_keys must be a sequence of strings")
        keys = tuple(_text(key, "required_policy_key") for key in required_policy_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("required_policy_keys must be non-empty and unique")
        return tuple(sorted(keys))
    discovered: set[str] = set()
    for update in node_updates:
        if not isinstance(update, Mapping):
            raise ValueError("each node update must be a mapping")
        delta = update.get("policy_delta")
        if not isinstance(delta, Mapping) or not delta:
            raise ValueError("policy_delta must be a non-empty mapping")
        discovered.update(_text(key, "policy_delta key") for key in delta)
    return tuple(sorted(discovered))


def _validate_node_update(
    raw: Mapping[str, object],
    *,
    required_policy_keys: tuple[str, ...],
    config: FederatedAggregationConfig,
) -> FederatedNodeUpdate:
    if not isinstance(raw, Mapping):
        raise ValueError("each node update must be a mapping")
    if any(key in raw for key in ("raw_time_series", "time_series", "samples")):
        raise ValueError("node update must not include raw time-series fields")
    node_id = _text(raw.get("node_id"), "node_id")
    sample_count = _positive_int(raw.get("sample_count"), "sample_count")
    local_loss = _non_negative_float(raw.get("local_loss"), "local_loss")
    previous_audit_hash = _hash(raw.get("previous_audit_hash"), "previous_audit_hash")
    epsilon_spent = _non_negative_float(
        raw.get("privacy_epsilon_spent", config.epsilon),
        "privacy_epsilon_spent",
    )
    delta = raw.get("policy_delta")
    if not isinstance(delta, Mapping) or not delta:
        raise ValueError("policy_delta must be a non-empty mapping")
    missing = [key for key in required_policy_keys if key not in delta]
    if missing:
        raise ValueError(f"policy_delta missing required keys: {missing}")
    values = {
        key: _finite_float(delta.get(key), f"policy_delta[{key}]")
        for key in required_policy_keys
    }
    l2 = math.sqrt(sum(value * value for value in values.values()))
    scale = 1.0 if l2 <= config.clipping_norm else config.clipping_norm / l2
    clipped = tuple((key, float(values[key] * scale)) for key in required_policy_keys)
    reasons: list[str] = []
    if epsilon_spent > config.epsilon:
        reasons.append("privacy_epsilon_exceeds_budget")
    if l2 > config.clipping_norm:
        reasons.append("update_clipped_to_norm")
    accepted = not reasons or reasons == ["update_clipped_to_norm"]
    payload = {
        "node_id": node_id,
        "policy_delta": [[key, value] for key, value in clipped],
        "sample_count": sample_count,
        "local_loss": local_loss,
        "previous_audit_hash": previous_audit_hash,
        "privacy_epsilon_spent": epsilon_spent,
        "clipped_l2_norm": min(l2, config.clipping_norm),
        "clip_scale": scale,
        "accepted": accepted,
        "rejection_reasons": [] if accepted else reasons,
    }
    return FederatedNodeUpdate(
        node_id=node_id,
        policy_delta=clipped,
        sample_count=sample_count,
        local_loss=local_loss,
        previous_audit_hash=previous_audit_hash,
        privacy_epsilon_spent=epsilon_spent,
        clipped_l2_norm=min(l2, config.clipping_norm),
        clip_scale=scale,
        accepted=accepted,
        rejection_reasons=() if accepted else tuple(reasons),
        update_hash=_stable_hash(payload),
    )


def _weighted_average(
    updates: Sequence[FederatedNodeUpdate],
    keys: tuple[str, ...],
) -> tuple[tuple[tuple[str, float], ...], int]:
    total = sum(update.sample_count for update in updates)
    if total <= 0:
        raise ValueError("accepted sample count must be positive")
    values = dict.fromkeys(keys, 0.0)
    for update in updates:
        weight = update.sample_count / total
        delta = dict(update.policy_delta)
        for key in keys:
            values[key] += weight * float(delta[key])
    return tuple((key, float(values[key])) for key in keys), total


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _non_negative_float(value: object, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _positive_float(value: object, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex string") from exc
    return value


def _stable_hash(payload: object) -> str:
    clean = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    if isinstance(clean, dict):
        clean.pop("report_hash", None)
        clean.pop("update_hash", None)
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
