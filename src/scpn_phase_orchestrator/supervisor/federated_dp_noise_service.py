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
    "DpNoiseServiceDeploymentPreflightManifest",
    "build_dp_noise_service_manifest",
    "build_dp_noise_service_deployment_preflight_manifest",
]


@dataclass(frozen=True)
class DpNoiseServiceReadiness:
    """Review readiness for the offline audit service."""

    ready: bool
    reason: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {"ready": self.ready, "reason": self.reason}


@dataclass(frozen=True)
class DpNoiseNodePrivacyBudget:
    """Per-node privacy spend declared for DP budget accounting."""

    node_id: str
    epsilon_spent: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
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
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
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
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
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


@dataclass(frozen=True)
class DpNoiseServiceDeploymentPreflightManifest:
    """Deterministic review-only deployment preflight manifest."""

    schema_name: str
    schema_version: str
    mechanism_label: str
    privacy_accountant_owner: str
    seed_custody_label: str
    budget_issuer_label: str
    service_endpoint_label: str
    operator_approved: bool
    request_hash: str
    response_hash: str
    epsilon: float
    delta: float
    deployment_readiness: DpNoiseServiceReadiness
    service_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    audit_record_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism_label, str):
            raise ValueError("mechanism_label must be a non-empty string")
        if not isinstance(self.privacy_accountant_owner, str):
            raise ValueError("privacy_accountant_owner must be a non-empty string")
        if not isinstance(self.seed_custody_label, str):
            raise ValueError("seed_custody_label must be a non-empty string")
        if not isinstance(self.budget_issuer_label, str):
            raise ValueError("budget_issuer_label must be a non-empty string")
        if not isinstance(self.service_endpoint_label, str):
            raise ValueError("service_endpoint_label must be a non-empty string")
        if not isinstance(self.operator_approved, bool):
            raise ValueError("operator_approved must be a boolean")
        if not isinstance(self.request_hash, str) or len(self.request_hash) != 64:
            raise ValueError("request_hash must be a 64-char hex hash")
        if not isinstance(self.response_hash, str) or len(self.response_hash) != 64:
            raise ValueError("response_hash must be a 64-char hex hash")
        if not isinstance(self.epsilon, Real) or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be a finite float")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be greater than 0")
        if not isinstance(self.delta, Real) or not math.isfinite(self.delta):
            raise ValueError("delta must be a finite float")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        try:
            bytes.fromhex(self.request_hash)
        except ValueError as exc:
            raise ValueError("request_hash must be hexadecimal") from exc
        try:
            bytes.fromhex(self.response_hash)
        except ValueError as exc:
            raise ValueError("response_hash must be hexadecimal") from exc

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "mechanism_label": self.mechanism_label,
            "privacy_accountant_owner": self.privacy_accountant_owner,
            "seed_custody_label": self.seed_custody_label,
            "budget_issuer_label": self.budget_issuer_label,
            "service_endpoint_label": self.service_endpoint_label,
            "operator_approved": self.operator_approved,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "deployment_readiness": self.deployment_readiness.to_audit_record(),
            "service_execution_permitted": self.service_execution_permitted,
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "audit_record_hash": self.audit_record_hash,
        }


def build_dp_noise_service_manifest(
    request: DpNoiseServiceRequestManifest,
) -> DpNoiseServiceResponseManifest:
    """Build a deterministic, dependency-free offline DP-noise review manifest.

    Parameters
    ----------
    request : DpNoiseServiceRequestManifest
        The DP-noise service request manifest.

    Returns
    -------
    DpNoiseServiceResponseManifest
        The offline DP-noise review response manifest.
    """
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


def build_dp_noise_service_deployment_preflight_manifest(
    request_manifest: DpNoiseServiceRequestManifest,
    response_manifest: DpNoiseServiceResponseManifest,
    *,
    mechanism_label: str,
    privacy_accountant_owner: str,
    seed_custody_label: str,
    budget_issuer_label: str,
    service_endpoint_label: str,
    operator_approved: bool,
) -> DpNoiseServiceDeploymentPreflightManifest:
    """Build a deterministic DP-noise deployment preflight manifest.

    Parameters
    ----------
    request_manifest : DpNoiseServiceRequestManifest
        The DP-noise service request manifest.
    response_manifest : DpNoiseServiceResponseManifest
        The DP-noise service response manifest.
    mechanism_label : str
        Label of the privacy mechanism.
    privacy_accountant_owner : str
        Owner of the privacy accountant.
    seed_custody_label : str
        Label describing seed custody.
    budget_issuer_label : str
        Label of the privacy-budget issuer.
    service_endpoint_label : str
        Label of the service endpoint.
    operator_approved : bool
        Whether a human operator has approved the deployment.

    Returns
    -------
    DpNoiseServiceDeploymentPreflightManifest
        The DP-noise deployment preflight manifest.

    Raises
    ------
    ValueError
        If the request and response manifests are inconsistent.
    """
    if not isinstance(request_manifest, DpNoiseServiceRequestManifest):
        raise ValueError("request_manifest must be a DpNoiseServiceRequestManifest")
    if not isinstance(response_manifest, DpNoiseServiceResponseManifest):
        raise ValueError("response_manifest must be a DpNoiseServiceResponseManifest")

    request_hash = _stable_hash(request_manifest.to_audit_record())
    response_hash_record = response_manifest.to_audit_record()
    response_hash_record["audit_record_hash"] = ""
    response_hash = _stable_hash(response_hash_record)

    missing_reasons: list[str] = []

    mechanism_label = _validated_label(
        mechanism_label, "mechanism_label", missing_reasons
    )
    privacy_accountant_owner = _validated_label(
        privacy_accountant_owner,
        "privacy_accountant_owner",
        missing_reasons,
    )
    seed_custody_label = _validated_label(
        seed_custody_label, "seed_custody_label", missing_reasons
    )
    budget_issuer_label = _validated_label(
        budget_issuer_label, "budget_issuer_label", missing_reasons
    )
    service_endpoint_label = _validated_label(
        service_endpoint_label, "service_endpoint_label", missing_reasons
    )
    if not isinstance(operator_approved, bool):
        raise ValueError("operator_approved must be a boolean")
    if operator_approved is False:
        missing_reasons.append("operator approval required")

    if response_manifest.request_hash != request_hash:
        missing_reasons.append("request and response hash linkage broken")
    if response_manifest.audit_record_hash != response_hash:
        missing_reasons.append("response hash integrity check failed")

    if request_manifest.epsilon != response_manifest.epsilon:
        missing_reasons.append(
            "epsilon mismatch between request and response manifests"
        )
    if request_manifest.delta != response_manifest.delta:
        missing_reasons.append("delta mismatch between request and response manifests")

    if not math.isfinite(response_manifest.epsilon) or response_manifest.epsilon <= 0.0:
        missing_reasons.append("response epsilon must be finite and positive")
    if (
        not math.isfinite(response_manifest.delta)
        or not 0.0 < response_manifest.delta < 1.0
    ):
        missing_reasons.append("response delta must be finite in (0, 1)")

    ready = len(missing_reasons) == 0
    reason = (
        "offline_deployment_preflight_ready" if ready else "; ".join(missing_reasons)
    )
    manifest = DpNoiseServiceDeploymentPreflightManifest(
        schema_name="federated_dp_noise_service_deployment_preflight_manifest",
        schema_version="1.0.0",
        mechanism_label=mechanism_label.strip(),
        privacy_accountant_owner=privacy_accountant_owner.strip(),
        seed_custody_label=seed_custody_label.strip(),
        budget_issuer_label=budget_issuer_label.strip(),
        service_endpoint_label=service_endpoint_label.strip(),
        operator_approved=operator_approved,
        request_hash=request_hash,
        response_hash=response_hash,
        epsilon=request_manifest.epsilon,
        delta=request_manifest.delta,
        deployment_readiness=DpNoiseServiceReadiness(ready=ready, reason=reason),
        service_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        audit_record_hash="",
    )
    return DpNoiseServiceDeploymentPreflightManifest(
        schema_name=manifest.schema_name,
        schema_version=manifest.schema_version,
        mechanism_label=manifest.mechanism_label,
        privacy_accountant_owner=manifest.privacy_accountant_owner,
        seed_custody_label=manifest.seed_custody_label,
        budget_issuer_label=manifest.budget_issuer_label,
        service_endpoint_label=manifest.service_endpoint_label,
        operator_approved=manifest.operator_approved,
        request_hash=manifest.request_hash,
        response_hash=manifest.response_hash,
        epsilon=manifest.epsilon,
        delta=manifest.delta,
        deployment_readiness=manifest.deployment_readiness,
        service_execution_permitted=manifest.service_execution_permitted,
        raw_data_export_permitted=manifest.raw_data_export_permitted,
        operator_review_required=manifest.operator_review_required,
        non_actuating=manifest.non_actuating,
        audit_record_hash=_stable_hash(manifest.to_audit_record()),
    )


def _generate_audit_noise(
    seed_hash: str,
    policy_keys: tuple[str, ...],
    sensitivity: float,
    noise_multiplier: float,
) -> tuple[tuple[str, float], ...]:
    """Return deterministic audit noise for a label."""
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


def _validated_label(value: object, field_name: str, reasons: list[str]) -> str:
    """Return the validated noise label, else raise."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    text = value.strip()
    if not text:
        reasons.append(f"{field_name} is required")
    return text


def _stable_hash(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 hash of the inputs."""
    serialised = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _uniform_from_digest(digest: bytes) -> float:
    """Return a uniform deviate derived from a digest."""
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return (value + 1.0) / (2.0**64 + 1.0)


def _text(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    converted: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        converted.append(value)
    return tuple(converted)
