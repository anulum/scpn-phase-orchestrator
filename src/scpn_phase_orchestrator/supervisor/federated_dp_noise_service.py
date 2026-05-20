# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated DP noise service

"""Offline differential-privacy noise service manifests for review-only use."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

__all__ = [
    "DpNoiseNodePrivacyBudget",
    "DpNoiseServiceReadiness",
    "DpNoiseServiceRequestManifest",
    "DpNoiseServiceResponseManifest",
    "build_dp_noise_service_manifest",
]


@dataclass(frozen=True)
class DpNoiseServiceReadiness:
    """Review readiness for the offline audit service."""

    ready: bool
    reason: str

    def to_audit_record(self) -> dict[str, object]:
        return {"ready": self.ready, "reason": self.reason}


@dataclass(frozen=True)
class DpNoiseNodePrivacyBudget:
    """Per-node privacy spend declared for DP budget accounting."""

    node_id: str
    epsilon_spent: float

    def to_audit_record(self) -> dict[str, object]:
        return {"node_id": self.node_id, "epsilon_spent": self.epsilon_spent}


@dataclass(frozen=True)
class DpNoiseServiceRequestManifest:
    """Validated request for offline DP-noise boundary review."""

    epsilon: float
    delta: float
    sensitivity: float
    noise_multiplier: float
    node_count: int
    seed_hash: str
    policy_keys: tuple[str, ...]
    node_budgets: tuple[DpNoiseNodePrivacyBudget, ...]

    schema_name: str = "federated_dp_noise_service"
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.epsilon, Real) or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be a finite float")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be greater than 0")
        if self.epsilon > 1_000_000.0:
            raise ValueError("epsilon exceeds max bound 1_000_000.0")

        if not isinstance(self.delta, Real) or not math.isfinite(self.delta):
            raise ValueError("delta must be a finite float")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")

        if not isinstance(self.sensitivity, Real) or not math.isfinite(
            self.sensitivity
        ):
            raise ValueError("sensitivity must be a finite float")
        if self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be greater than 0")

        if not isinstance(self.noise_multiplier, Real) or not math.isfinite(
            self.noise_multiplier
        ):
            raise ValueError("noise_multiplier must be a finite float")
        if self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be greater than 0")

        if not isinstance(self.node_count, Integral):
            raise ValueError("node_count must be an integer")
        if self.node_count <= 0:
            raise ValueError("node_count must be greater than 0")

        if not isinstance(self.seed_hash, str) or len(self.seed_hash.strip()) == 0:
            raise ValueError("seed_hash is required")
        if len(self.seed_hash) != 64:
            raise ValueError("seed_hash must be 64 hex characters")
        try:
            bytes.fromhex(self.seed_hash)
        except ValueError as exc:
            raise ValueError("seed_hash must be a hex string") from exc

        if not isinstance(self.policy_keys, tuple):
            raise ValueError("policy_keys must be a tuple of strings")
        if len(self.policy_keys) == 0:
            raise ValueError("policy_keys must be non-empty")
        keys = tuple(_text(self.policy_keys, "policy key"))
        if len(set(keys)) != len(keys):
            raise ValueError("policy_keys must be unique")
        object.__setattr__(self, "policy_keys", tuple(sorted(keys)))

        if not isinstance(self.node_budgets, tuple):
            raise ValueError("node_budgets must be a tuple of DpNoiseNodePrivacyBudget")
        if len(self.node_budgets) != self.node_count:
            raise ValueError("node_budgets length must match node_count")
        if not self.node_budgets:
            raise ValueError("node_budgets must not be empty")

        node_ids = set[str]()
        for budget in self.node_budgets:
            if not isinstance(budget, DpNoiseNodePrivacyBudget):
                raise ValueError(
                    "node_budgets must only contain DpNoiseNodePrivacyBudget"
                )
            if not isinstance(budget.node_id, str) or not budget.node_id:
                raise ValueError("node_id is required")
            if budget.node_id in node_ids:
                raise ValueError("node_id values must be unique")
            if budget.epsilon_spent < 0.0:
                raise ValueError("node epsilon spent must be non-negative")
            node_ids.add(budget.node_id)

        if self.node_count > 0:
            spent = sum(budget.epsilon_spent for budget in self.node_budgets)
            if spent > self.epsilon:
                raise ValueError("privacy budget exceeded")

    def to_audit_record(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "noise_multiplier": self.noise_multiplier,
            "node_count": self.node_count,
            "seed_hash": self.seed_hash,
            "policy_keys": list(self.policy_keys),
            "node_budgets": [budget.to_audit_record() for budget in self.node_budgets],
        }


@dataclass(frozen=True)
class DpNoiseServiceResponseManifest:
    """Offline review manifest returned by the audit boundary."""

    schema_name: str
    schema_version: str
    request_hash: str
    service_readiness: DpNoiseServiceReadiness
    epsilon: float
    delta: float
    sensitivity: float
    noise_multiplier: float
    privacy_budget_spent: float
    privacy_budget_remaining: float
    node_count: int
    policy_keys: tuple[str, ...]
    policy_noise_audit_vector: tuple[tuple[str, float], ...]
    service_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    node_budgets: tuple[DpNoiseNodePrivacyBudget, ...]
    audit_record_hash: str

    def to_audit_record(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "request_hash": self.request_hash,
            "service_readiness": self.service_readiness.to_audit_record(),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "noise_multiplier": self.noise_multiplier,
            "privacy_budget_spent": self.privacy_budget_spent,
            "privacy_budget_remaining": self.privacy_budget_remaining,
            "node_count": self.node_count,
            "policy_keys": list(self.policy_keys),
            "policy_noise_audit_vector": [
                [key, value] for key, value in self.policy_noise_audit_vector
            ],
            "service_execution_permitted": self.service_execution_permitted,
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "node_budgets": [budget.to_audit_record() for budget in self.node_budgets],
            "audit_record_hash": self.audit_record_hash,
        }


def build_dp_noise_service_manifest(
    request: DpNoiseServiceRequestManifest,
) -> DpNoiseServiceResponseManifest:
    """Build a deterministic, dependency-free offline DP-noise review manifest."""
    request_hash = _stable_hash(request.to_audit_record())
    policy_noise_audit_vector = _generate_audit_noise(
        request.seed_hash,
        request.policy_keys,
        request.sensitivity,
        request.noise_multiplier,
    )
    privacy_budget_spent = round(
        sum(budget.epsilon_spent for budget in request.node_budgets), 12
    )
    privacy_budget_remaining = round(request.epsilon - privacy_budget_spent, 12)
    response = DpNoiseServiceResponseManifest(
        schema_name=request.schema_name,
        schema_version=request.schema_version,
        request_hash=request_hash,
        service_readiness=DpNoiseServiceReadiness(
            ready=True, reason="offline_review_only_manifest_ready"
        ),
        epsilon=request.epsilon,
        delta=request.delta,
        sensitivity=request.sensitivity,
        noise_multiplier=request.noise_multiplier,
        privacy_budget_spent=privacy_budget_spent,
        privacy_budget_remaining=privacy_budget_remaining,
        node_count=request.node_count,
        policy_keys=request.policy_keys,
        policy_noise_audit_vector=policy_noise_audit_vector,
        service_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        node_budgets=request.node_budgets,
        audit_record_hash="",
    )
    return DpNoiseServiceResponseManifest(
        schema_name=response.schema_name,
        schema_version=response.schema_version,
        request_hash=response.request_hash,
        service_readiness=response.service_readiness,
        epsilon=response.epsilon,
        delta=response.delta,
        sensitivity=response.sensitivity,
        noise_multiplier=response.noise_multiplier,
        privacy_budget_spent=response.privacy_budget_spent,
        privacy_budget_remaining=response.privacy_budget_remaining,
        node_count=response.node_count,
        policy_keys=response.policy_keys,
        policy_noise_audit_vector=response.policy_noise_audit_vector,
        service_execution_permitted=response.service_execution_permitted,
        raw_data_export_permitted=response.raw_data_export_permitted,
        operator_review_required=response.operator_review_required,
        non_actuating=response.non_actuating,
        node_budgets=response.node_budgets,
        audit_record_hash=_stable_hash(response.to_audit_record()),
    )


def _generate_audit_noise(
    seed_hash: str,
    policy_keys: tuple[str, ...],
    sensitivity: float,
    noise_multiplier: float,
) -> tuple[tuple[str, float], ...]:
    vector: list[tuple[str, float]] = []
    scale = sensitivity * noise_multiplier
    for index, key in enumerate(policy_keys):
        key_seed = f"{seed_hash}:{index}:{key}:{scale}".encode()
        digest = hashlib.sha256(key_seed).digest()
        u1 = _uniform_from_digest(digest[:8])
        u2 = _uniform_from_digest(digest[8:16])
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        vector.append((key, round(z * scale, 12)))
    return tuple(vector)


def _stable_hash(payload: Mapping[str, object]) -> str:
    serialised = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _uniform_from_digest(digest: bytes) -> float:
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return (value + 1.0) / (2.0**64 + 1.0)


def _text(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    converted: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        converted.append(value)
    return tuple(converted)
