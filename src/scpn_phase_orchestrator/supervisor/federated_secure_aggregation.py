# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline secure aggregation manifest

"""Offline secure aggregation manifests for federated supervisor review."""

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
    "SecureNodeCustodyRecord",
    "SecureAggregationQuorumEvidence",
    "FederatedSecureAggregationManifest",
    "FederatedSecureAggregationPreflightManifest",
    "build_federated_secure_aggregation_manifest",
    "build_federated_secure_aggregation_preflight_manifest",
]


SUPPORTED_CUSTODY_ROTATION_POLICIES = ("continuous", "manual", "scheduled")


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
        """Return a JSON-safe audit record for this node commitment.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe audit record for this node commitment.
        """
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
class SecureAggregationQuorumEvidence:
    """Signed quorum metadata for review-only preflight."""

    node_id: str
    evidence_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe quorum evidence record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe quorum evidence record.
        """
        return {"node_id": self.node_id, "evidence_hash": self.evidence_hash}


@dataclass(frozen=True)
class SecureNodeCustodyRecord:
    """Custody metadata for key and share labels used in review."""

    node_id: str
    key_custody_label: str
    share_custody_label: str
    previous_key_custody_label: str
    previous_share_custody_label: str
    key_custody_continuity_hash: str
    share_custody_continuity_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe node-custody audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe node-custody audit record.
        """
        return {
            "node_id": self.node_id,
            "key_custody_label": self.key_custody_label,
            "share_custody_label": self.share_custody_label,
            "previous_key_custody_label": self.previous_key_custody_label,
            "previous_share_custody_label": self.previous_share_custody_label,
            "key_custody_continuity_hash": self.key_custody_continuity_hash,
            "share_custody_continuity_hash": self.share_custody_continuity_hash,
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
        """Return a JSON-safe aggregate audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe aggregate audit record.
        """
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


@dataclass(frozen=True)
class FederatedSecureAggregationPreflightManifest:
    """Review-only deployment preflight envelope for manifest execution."""

    schema_name: str
    schema_version: str
    secure_aggregation_schema_name: str
    secure_aggregation_schema_version: str
    secure_aggregation_report_hash: str
    accepted_node_threshold: int
    accepted_node_count: int
    quorum_evidence: tuple[SecureAggregationQuorumEvidence, ...]
    custody_rotation_policy: str
    custody_records: tuple[SecureNodeCustodyRecord, ...]
    operator_approved: bool
    operator_id: str
    service_owner: str
    secure_aggregation_execution_permitted: bool
    raw_data_export_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe preflight audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe preflight audit record.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "secure_aggregation_schema_name": self.secure_aggregation_schema_name,
            "secure_aggregation_schema_version": self.secure_aggregation_schema_version,
            "secure_aggregation_report_hash": self.secure_aggregation_report_hash,
            "accepted_node_threshold": self.accepted_node_threshold,
            "accepted_node_count": self.accepted_node_count,
            "quorum_evidence": [
                entry.to_audit_record() for entry in self.quorum_evidence
            ],
            "custody_rotation_policy": self.custody_rotation_policy,
            "custody_records": [
                record.to_audit_record() for record in self.custody_records
            ],
            "operator_approved": self.operator_approved,
            "operator_id": self.operator_id,
            "service_owner": self.service_owner,
            "secure_aggregation_execution_permitted": (
                self.secure_aggregation_execution_permitted
            ),
            "raw_data_export_permitted": self.raw_data_export_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
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
    """Build a deterministic secure aggregation manifest.

    Parameters
    ----------
    node_commitments : Sequence[Mapping[str, object]]
        Secure-aggregation node commitment records.
    required_policy_keys : Sequence[str] | None
        Policy keys every node update must carry, or ``None``.
    clipping_norm : float
        L2 clipping norm applied to each node update.
    min_node_count : int
        Minimum number of participating nodes required.
    epsilon : float
        Differential-privacy ``ε`` budget.
    delta : float
        Differential-privacy ``δ`` budget.

    Returns
    -------
    FederatedSecureAggregationManifest
        The secure-aggregation manifest.

    Raises
    ------
    ValueError
        If the node commitments or privacy parameters are invalid.
    """
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


def build_federated_secure_aggregation_preflight_manifest(
    secure_aggregation_manifest: FederatedSecureAggregationManifest,
    *,
    quorum_evidence: Sequence[Mapping[str, object]],
    custody_rotation_policy: str,
    custody_records: Sequence[Mapping[str, object]],
    accepted_node_threshold: int,
    operator_approved: bool,
    operator_id: str,
    service_owner: str,
) -> FederatedSecureAggregationPreflightManifest:
    """Build a deterministic review-only deployment preflight manifest.

    Parameters
    ----------
    secure_aggregation_manifest : FederatedSecureAggregationManifest
        The secure-aggregation manifest to preflight.
    quorum_evidence : Sequence[Mapping[str, object]]
        Per-node quorum evidence records.
    custody_rotation_policy : str
        Key-custody rotation policy label.
    custody_records : Sequence[Mapping[str, object]]
        Node custody records.
    accepted_node_threshold : int
        Minimum number of accepted nodes required.
    operator_approved : bool
        Whether a human operator has approved the deployment.
    operator_id : str
        Identifier of the approving operator.
    service_owner : str
        Owner of the aggregation service.

    Returns
    -------
    FederatedSecureAggregationPreflightManifest
        The secure-aggregation deployment preflight manifest.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the manifest or quorum evidence is invalid.
    """
    if not isinstance(secure_aggregation_manifest, FederatedSecureAggregationManifest):
        raise TypeError(
            "secure_aggregation_manifest must be a FederatedSecureAggregationManifest"
        )

    if not operator_approved:
        raise ValueError("operator approval is required for preflight")

    operator = _non_empty_text(operator_id, "operator_id")
    owner = _non_empty_text(service_owner, "service_owner")
    threshold = _positive_int(accepted_node_threshold, "accepted_node_threshold")
    rotation_policy = _non_empty_text(
        custody_rotation_policy, "custody_rotation_policy"
    )
    if rotation_policy not in SUPPORTED_CUSTODY_ROTATION_POLICIES:
        raise ValueError("unsupported custody rotation policy")

    if not secure_aggregation_manifest.quorum_met:
        raise ValueError("secure aggregation manifest quorum not met")

    report_record = secure_aggregation_manifest.to_audit_record()
    report_record["report_hash"] = ""
    expected_report_hash = _stable_hash(report_record)
    manifest_report_hash = _sha256_hex(
        secure_aggregation_manifest.report_hash, "secure_aggregation_report_hash"
    )
    if manifest_report_hash != expected_report_hash:
        raise ValueError("secure_aggregation_manifest report hash mismatch")

    accepted_nodes = tuple(
        node.node_id
        for node in secure_aggregation_manifest.node_commitments
        if node.accepted
    )
    if len(accepted_nodes) < threshold:
        raise ValueError("accepted-node threshold not met")

    evidence = _resolve_preflight_quorum_evidence(
        quorum_evidence,
        accepted_nodes=accepted_nodes,
        accepted_node_threshold=threshold,
    )
    custody = _resolve_node_custody_records(
        custody_records,
        accepted_nodes=accepted_nodes,
        custody_rotation_policy=rotation_policy,
    )

    custody_node_ids = tuple(record.node_id for record in custody)
    if len(custody_node_ids) != len(set(custody_node_ids)):
        raise ValueError("custody records must be unique per accepted node")
    if set(custody_node_ids) != set(accepted_nodes):
        raise ValueError("custody labels must cover all accepted nodes")

    preflight = FederatedSecureAggregationPreflightManifest(
        schema_name="federated_secure_aggregation_preflight_manifest",
        schema_version="0.1.0",
        secure_aggregation_schema_name=secure_aggregation_manifest.schema_name,
        secure_aggregation_schema_version=secure_aggregation_manifest.schema_version,
        secure_aggregation_report_hash=manifest_report_hash,
        accepted_node_threshold=threshold,
        accepted_node_count=len(accepted_nodes),
        quorum_evidence=tuple(sorted(evidence, key=lambda entry: entry.node_id)),
        custody_rotation_policy=rotation_policy,
        custody_records=tuple(sorted(custody, key=lambda entry: entry.node_id)),
        operator_approved=operator_approved,
        operator_id=operator,
        service_owner=owner,
        secure_aggregation_execution_permitted=False,
        raw_data_export_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        report_hash="",
    )
    return FederatedSecureAggregationPreflightManifest(
        schema_name=preflight.schema_name,
        schema_version=preflight.schema_version,
        secure_aggregation_schema_name=preflight.secure_aggregation_schema_name,
        secure_aggregation_schema_version=preflight.secure_aggregation_schema_version,
        secure_aggregation_report_hash=preflight.secure_aggregation_report_hash,
        accepted_node_threshold=preflight.accepted_node_threshold,
        accepted_node_count=preflight.accepted_node_count,
        quorum_evidence=preflight.quorum_evidence,
        custody_rotation_policy=preflight.custody_rotation_policy,
        custody_records=preflight.custody_records,
        operator_approved=preflight.operator_approved,
        operator_id=preflight.operator_id,
        service_owner=preflight.service_owner,
        secure_aggregation_execution_permitted=preflight.secure_aggregation_execution_permitted,
        raw_data_export_permitted=preflight.raw_data_export_permitted,
        operator_review_required=preflight.operator_review_required,
        non_actuating=preflight.non_actuating,
        report_hash=_stable_hash(preflight.to_audit_record()),
    )


def _resolve_required_policy_keys(
    required_policy_keys: Sequence[str] | None,
    node_commitments: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the required policy keys for aggregation."""
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
    """Validate a node's masked commitment, else raise."""
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
) -> tuple[tuple[tuple[str, float], ...], int]:
    """Return the weighted average of masked node updates."""
    total_sample_count = sum(node.sample_count for node in accepted_nodes)
    if total_sample_count <= 0:
        raise ValueError("total_sample_count must be positive for aggregation")

    aggregate: dict[str, float] = dict.fromkeys(required_policy_keys, 0.0)
    for node in accepted_nodes:
        for key, value in node.masked_policy_delta:
            aggregate[key] += value * node.sample_count
    for key in required_policy_keys:
        aggregate[key] /= float(total_sample_count)
    return (
        tuple((key, aggregate[key]) for key in required_policy_keys),
        total_sample_count,
    )


def _resolve_preflight_quorum_evidence(
    quorum_evidence: Sequence[Mapping[str, object]],
    *,
    accepted_nodes: tuple[str, ...],
    accepted_node_threshold: int,
) -> tuple[SecureAggregationQuorumEvidence, ...]:
    """Resolve the preflight quorum evidence."""
    if not isinstance(quorum_evidence, Sequence) or isinstance(
        quorum_evidence, (str, bytes, bytearray)
    ):
        raise ValueError("quorum_evidence must be a sequence of mappings")
    if not quorum_evidence:
        raise ValueError("quorum_evidence must be non-empty")

    seen_nodes: set[str] = set()
    accepted_set = set(accepted_nodes)
    resolved = tuple(
        _validate_preflight_quorum_evidence(
            raw,
            seen_nodes=seen_nodes,
            accepted_nodes=accepted_set,
        )
        for raw in quorum_evidence
    )

    if len(resolved) < accepted_node_threshold:
        raise ValueError("quorum evidence below accepted-node threshold")
    return resolved


def _validate_preflight_quorum_evidence(
    raw: Mapping[str, object],
    *,
    seen_nodes: set[str],
    accepted_nodes: set[str],
) -> SecureAggregationQuorumEvidence:
    """Validate the preflight quorum evidence, else raise."""
    if not isinstance(raw, Mapping):
        raise ValueError("each quorum evidence entry must be a mapping")

    node_id = _non_empty_text(raw.get("node_id"), "quorum_evidence.node_id")
    if node_id in seen_nodes:
        raise ValueError(f"quorum evidence duplicated node_id: {node_id}")
    if node_id not in accepted_nodes:
        raise ValueError("quorum evidence must reference accepted nodes only")
    seen_nodes.add(node_id)
    evidence_hash = _sha256_hex(
        raw.get("evidence_hash"), "quorum_evidence.evidence_hash"
    )

    return SecureAggregationQuorumEvidence(node_id=node_id, evidence_hash=evidence_hash)


def _resolve_node_custody_records(
    custody_records: Sequence[Mapping[str, object]],
    *,
    accepted_nodes: tuple[str, ...],
    custody_rotation_policy: str,
) -> tuple[SecureNodeCustodyRecord, ...]:
    """Resolve the per-node custody records."""
    if not isinstance(custody_records, Sequence) or isinstance(
        custody_records, (str, bytes, bytearray)
    ):
        raise ValueError("custody_records must be a sequence of mappings")
    if not custody_records:
        raise ValueError("custody_records must be non-empty")

    accepted_set = set(accepted_nodes)
    seen_nodes: set[str] = set()
    return tuple(
        _validate_node_custody_record(
            raw,
            seen_nodes=seen_nodes,
            accepted_nodes=accepted_set,
            custody_rotation_policy=custody_rotation_policy,
        )
        for raw in custody_records
    )


def _validate_node_custody_record(
    raw: Mapping[str, object],
    *,
    seen_nodes: set[str],
    accepted_nodes: set[str],
    custody_rotation_policy: str,
) -> SecureNodeCustodyRecord:
    """Validate a node custody record, else raise."""
    if not isinstance(raw, Mapping):
        raise ValueError("each custody record must be a mapping")

    node_id = _non_empty_text(raw.get("node_id"), "custody_records.node_id")
    if node_id in seen_nodes:
        raise ValueError(f"custody record duplicated node_id: {node_id}")
    if node_id not in accepted_nodes:
        raise ValueError("custody labels must reference accepted nodes only")
    seen_nodes.add(node_id)

    key_custody_label = _sha256_hex(
        raw.get("key_custody_label"), "custody_records.key_custody_label"
    )
    share_custody_label = _sha256_hex(
        raw.get("share_custody_label"), "custody_records.share_custody_label"
    )
    previous_key_custody_label = _sha256_hex(
        raw.get("previous_key_custody_label"),
        "custody_records.previous_key_custody_label",
    )
    previous_share_custody_label = _sha256_hex(
        raw.get("previous_share_custody_label"),
        "custody_records.previous_share_custody_label",
    )
    key_custody_continuity_hash = _sha256_hex(
        raw.get("key_custody_continuity_hash"),
        "custody_records.key_custody_continuity_hash",
    )
    share_custody_continuity_hash = _sha256_hex(
        raw.get("share_custody_continuity_hash"),
        "custody_records.share_custody_continuity_hash",
    )

    expected_key_continuity = _stable_hash(
        {
            "node_id": node_id,
            "rotation_policy": custody_rotation_policy,
            "previous_key_custody_label": previous_key_custody_label,
            "key_custody_label": key_custody_label,
        }
    )
    if key_custody_continuity_hash != expected_key_continuity:
        raise ValueError("key custody continuity hash mismatch")
    expected_share_continuity = _stable_hash(
        {
            "node_id": node_id,
            "rotation_policy": custody_rotation_policy,
            "previous_share_custody_label": previous_share_custody_label,
            "share_custody_label": share_custody_label,
        }
    )
    if share_custody_continuity_hash != expected_share_continuity:
        raise ValueError("share custody continuity hash mismatch")

    return SecureNodeCustodyRecord(
        node_id=node_id,
        key_custody_label=key_custody_label,
        share_custody_label=share_custody_label,
        previous_key_custody_label=previous_key_custody_label,
        previous_share_custody_label=previous_share_custody_label,
        key_custody_continuity_hash=key_custody_continuity_hash,
        share_custody_continuity_hash=share_custody_continuity_hash,
    )


def _sha256_hex(value: object, field: str) -> str:
    """Return the SHA-256 hex digest of the inputs."""
    text = _non_empty_text(value, field)
    digest = _normalise_hex(text)
    if not _is_sha256_hex(digest):
        raise ValueError(
            f"{field} must be a 64-character lower-case SHA-256 hex digest"
        )
    return digest


def _normalise_hex(value: str) -> str:
    """Return the normalised lowercase hex string, else raise."""
    return value.lower()


def _is_sha256_hex(value: str) -> bool:
    """Return whether ``value`` is a SHA-256 hex digest."""
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _non_empty_text(value: object, field: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive int")
    return int(value)


def _positive_float(value: object, field: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    number = _to_float(value, field)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive float")
    return float(number)


def _finite_float(value: object, field: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    number = _to_float(value, field)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite float")
    return float(number)


def _to_float(value: object, field: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{field} must be a real number")
    return float(value)


def _stable_hash(value: object) -> str:
    """Return a stable SHA-256 hash of the inputs."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
