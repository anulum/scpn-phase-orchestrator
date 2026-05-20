# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multiverse counterfactual branch-risk gate

"""Fail-closed review gate over precomputed branch rollout manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

__all__ = [
    "BranchRiskDecision",
    "MultiverseRiskReport",
    "MultiverseRiskThresholds",
    "evaluate_multiverse_branch_risk",
]


@dataclass(frozen=True)
class MultiverseRiskThresholds:
    """Guard thresholds for branch review in the multiverse gate."""

    min_mean_R: float = 0.0
    min_final_R: float = 0.0
    max_action_count: int = 64
    max_topology_edge_count: int | None = None
    max_topology_scale: float | None = None


@dataclass(frozen=True)
class BranchRiskDecision:
    """Outcome for one branch in a manifest."""

    branch_id: str
    branch_hash: str
    final_R: float
    mean_R: float
    min_R: float
    max_R: float
    action_count: int
    topology_edge_count: int | None
    topology_scale: float | None
    approved: bool
    rejection_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe branch decision record."""
        return {
            "branch_id": self.branch_id,
            "branch_hash": self.branch_hash,
            "final_R": self.final_R,
            "mean_R": self.mean_R,
            "min_R": self.min_R,
            "max_R": self.max_R,
            "action_count": self.action_count,
            "topology_edge_count": self.topology_edge_count,
            "topology_scale": self.topology_scale,
            "approved": self.approved,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class MultiverseRiskReport:
    """JSON-safe aggregate of the branch-risk review decision."""

    schema_name: str
    schema_version: str
    branch_decisions: tuple[BranchRiskDecision, ...]
    approved_count: int
    rejected_count: int
    safest_branch_id: str | None
    safest_branch_hash: str | None
    rejection_reasons: tuple[str, ...]
    claim_boundary: str
    non_actuating: bool
    execution_disabled: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe multiverse risk gate audit record."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "branch_count": len(self.branch_decisions),
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "safest_branch_id": self.safest_branch_id,
            "safest_branch_hash": self.safest_branch_hash,
            "rejection_reasons": list(self.rejection_reasons),
            "branch_decisions": [
                decision.to_audit_record() for decision in self.branch_decisions
            ],
            "claim_boundary": self.claim_boundary,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "report_hash": self.report_hash,
        }


def _is_finite_real(value: object, field: str) -> float:
    """Return a finite float or raise ValueError with field context."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real")
    value_f = float(value)
    if value_f != value_f or value_f in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite real")
    return value_f


def _is_non_negative_int(value: object, field: str) -> int:
    """Return a non-negative int or raise ValueError with field context."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite non-negative integer")
    if float(value) != int(value):
        raise ValueError(f"{field} must be a finite non-negative integer")
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{field} must be a finite non-negative integer")
    return value_i


def _normalise_thresholds(
    thresholds: MultiverseRiskThresholds | None,
) -> MultiverseRiskThresholds:
    """Validate and normalise thresholds to defaults."""
    if thresholds is None:
        return MultiverseRiskThresholds()
    _is_finite_real(thresholds.min_mean_R, "min_mean_R")
    _is_finite_real(thresholds.min_final_R, "min_final_R")
    _is_non_negative_int(thresholds.max_action_count, "max_action_count")
    if thresholds.max_topology_edge_count is not None:
        _is_non_negative_int(
            thresholds.max_topology_edge_count, "max_topology_edge_count"
        )
    if thresholds.max_topology_scale is not None:
        _is_finite_real(thresholds.max_topology_scale, "max_topology_scale")
        if thresholds.max_topology_scale < 0:
            raise ValueError("max_topology_scale must be non-negative")
    return thresholds


def _extract_manifest_branches(
    manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Extract branch records with strict mapping and sequence checks."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    branches_raw = manifest.get("branches")
    if branches_raw is None:
        branches_raw = manifest.get("branch_records")
    if branches_raw is None:
        raise ValueError("manifest must include a 'branches' or 'branch_records' key")
    if not isinstance(branches_raw, list | tuple):
        raise ValueError("manifest branches must be a list or tuple")
    if not branches_raw:
        raise ValueError("manifest branches must be non-empty")
    branches: list[Mapping[str, object]] = []
    for branch in branches_raw:
        if not isinstance(branch, Mapping):
            raise ValueError("each branch entry must be a mapping")
        branches.append(branch)
    return tuple(branches)


def _build_branch_decision(
    branch: Mapping[str, object],
    thresholds: MultiverseRiskThresholds,
) -> BranchRiskDecision:
    """Build a deterministic per-branch decision record."""
    branch_id = branch.get("branch_id")
    if not isinstance(branch_id, str) or not branch_id:
        raise ValueError("branch_id is required and must be a non-empty string")
    branch_hash = branch.get("branch_hash")
    if not isinstance(branch_hash, str) or not branch_hash:
        raise ValueError("branch_hash is required and must be a non-empty string")

    final_R = _is_finite_real(branch.get("final_R"), f"branch[{branch_id}].final_R")
    mean_R = _is_finite_real(branch.get("mean_R"), f"branch[{branch_id}].mean_R")
    min_R = _is_finite_real(branch.get("min_R"), f"branch[{branch_id}].min_R")
    max_R = _is_finite_real(branch.get("max_R"), f"branch[{branch_id}].max_R")
    action_count = _is_non_negative_int(
        branch.get("action_count"), f"branch[{branch_id}].action_count"
    )

    if min_R > max_R:
        raise ValueError(f"branch[{branch_id}] has invalid R interval")

    topology_edge_count: int | None = None
    if "topology_edge_count" in branch:
        topology_edge_count = _is_non_negative_int(
            branch.get("topology_edge_count"),
            f"branch[{branch_id}].topology_edge_count",
        )
    topology_scale: float | None = None
    if "topology_scale" in branch:
        topology_scale = _is_finite_real(
            branch.get("topology_scale"),
            f"branch[{branch_id}].topology_scale",
        )
        if topology_scale < 0:
            raise ValueError(f"branch[{branch_id}].topology_scale must be non-negative")

    if topology_edge_count is None and topology_scale is None:
        raise ValueError(
            f"branch[{branch_id}] must include topology_edge_count or topology_scale"
        )

    reasons: list[str] = []
    if mean_R < thresholds.min_mean_R:
        reasons.append("mean_R_below_minimum")
    if final_R < thresholds.min_final_R:
        reasons.append("final_R_below_minimum")
    if action_count > thresholds.max_action_count:
        reasons.append("action_count_exceeds_limit")
    if (
        topology_edge_count is not None
        and thresholds.max_topology_edge_count is not None
        and topology_edge_count > thresholds.max_topology_edge_count
    ):
        reasons.append("topology_edge_count_exceeds_limit")
    if (
        topology_scale is not None
        and thresholds.max_topology_scale is not None
        and topology_scale > thresholds.max_topology_scale
    ):
        reasons.append("topology_scale_exceeds_limit")

    return BranchRiskDecision(
        branch_id=branch_id,
        branch_hash=branch_hash,
        final_R=final_R,
        mean_R=mean_R,
        min_R=min_R,
        max_R=max_R,
        action_count=action_count,
        topology_edge_count=topology_edge_count,
        topology_scale=topology_scale,
        approved=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _build_report_hash(record: dict[str, object]) -> str:
    """Build a deterministic hash for the audit record payload."""
    payload = dict(record)
    payload.pop("report_hash", None)
    payload_blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_blob.encode("utf-8")).hexdigest()


def _select_safest_branch(
    decisions: tuple[BranchRiskDecision, ...],
) -> tuple[str | None, str | None]:
    """Select deterministic safest approved branch."""
    approved = [decision for decision in decisions if decision.approved]
    if not approved:
        return None, None

    def _safety_key(
        decision: BranchRiskDecision,
    ) -> tuple[float, float, int, float, str]:
        topo_pressure = (
            float(decision.topology_edge_count)
            if decision.topology_edge_count is not None
            else decision.topology_scale or float("inf")
        )
        return (
            decision.mean_R,
            decision.final_R,
            -decision.action_count,
            -topo_pressure,
            f"{decision.branch_id}|{decision.branch_hash}",
        )

    safest = max(approved, key=_safety_key)
    return safest.branch_id, safest.branch_hash


def evaluate_multiverse_branch_risk(
    manifest: Mapping[str, object],
    thresholds: MultiverseRiskThresholds | None = None,
) -> MultiverseRiskReport:
    """Evaluate branch risk decisions for a branch manifest without actuation."""
    thresholds = _normalise_thresholds(thresholds)
    branches = _extract_manifest_branches(manifest)

    branch_decisions = tuple(
        _build_branch_decision(branch=branch, thresholds=thresholds)
        for branch in branches
    )

    approved = tuple(decision for decision in branch_decisions if decision.approved)
    rejected = tuple(decision for decision in branch_decisions if not decision.approved)
    safest_branch_id, safest_branch_hash = _select_safest_branch(branch_decisions)

    report = MultiverseRiskReport(
        schema_name="multiverse_branch_risk_gate",
        schema_version="0.1.0",
        branch_decisions=branch_decisions,
        approved_count=len(approved),
        rejected_count=len(rejected),
        safest_branch_id=safest_branch_id,
        safest_branch_hash=safest_branch_hash,
        rejection_reasons=tuple(
            sorted(
                {
                    reason
                    for decision in rejected
                    for reason in decision.rejection_reasons
                }
            )
        ),
        claim_boundary="counterfactual_branch_risk_gate_not_live_actuation",
        non_actuating=True,
        execution_disabled=True,
        report_hash="",
    )

    record = report.to_audit_record()
    report_hash = _build_report_hash(record)
    return MultiverseRiskReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        branch_decisions=report.branch_decisions,
        approved_count=report.approved_count,
        rejected_count=report.rejected_count,
        safest_branch_id=report.safest_branch_id,
        safest_branch_hash=report.safest_branch_hash,
        rejection_reasons=report.rejection_reasons,
        claim_boundary=report.claim_boundary,
        non_actuating=report.non_actuating,
        execution_disabled=report.execution_disabled,
        report_hash=report_hash,
    )
