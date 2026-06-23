# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio multiverse counterfactual panel

"""Multiverse counterfactual branch review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _normalise_optional_text_sequence,
    _normalise_text_sequence,
    _optional_non_negative_int,
    _optional_sha256_hex,
    _positive_int,
    _require_non_empty_text,
    _require_sha256_hex,
    _required_bool,
    _unit_interval_number,
)

_MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY = "counterfactual_branch_rollout_not_live_actuation"


_MULTIVERSE_RISK_CLAIM_BOUNDARY = "counterfactual_branch_risk_gate_not_live_actuation"


_MULTIVERSE_BACKENDS = frozenset({"numpy_vectorized", "jax_vectorized"})


def build_multiverse_counterfactual_studio_panel(
    manifest: Mapping[str, object],
    risk_report: Mapping[str, object],
) -> dict[str, object]:
    """Return a Studio panel payload for multiverse branch review evidence.

    The panel joins a non-actuating rollout manifest with a non-actuating risk
    gate report. It validates both audit artefacts before rendering branch
    comparison rows, safest-branch metadata, and coherence ranges for operator
    review. The helper never emits executable actions.

    Parameters
    ----------
    manifest : Mapping[str, object]
        The manifest object.
    risk_report : Mapping[str, object]
        The multiverse risk report mapping.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for multiverse branch review evidence.
    """
    rollout = _normalise_multiverse_manifest(manifest)
    risk = _normalise_multiverse_risk_report(risk_report)
    branch_rows = _join_multiverse_branch_rows(rollout, risk)
    final_values = [cast("float", row["final_R"]) for row in branch_rows]
    mean_values = [cast("float", row["mean_R"]) for row in branch_rows]
    min_values = [cast("float", row["min_R"]) for row in branch_rows]
    max_values = [cast("float", row["max_R"]) for row in branch_rows]
    rejected_ids = [
        cast("str", row["branch_id"])
        for row in branch_rows
        if row["risk_approved"] is not True
    ]
    return {
        "panel_kind": "studio_multiverse_counterfactual_panel",
        "simulator": "multiverse_counterfactual",
        "risk_gate": "multiverse_branch_risk_gate",
        "schema_version": rollout["schema_version"],
        "risk_schema_version": risk["schema_version"],
        "backend": rollout["backend"],
        "horizon": rollout["horizon"],
        "branch_count": rollout["branch_count"],
        "approved_count": risk["approved_count"],
        "rejected_count": risk["rejected_count"],
        "safest_branch_id": risk["safest_branch_id"],
        "safest_branch_hash": risk["safest_branch_hash"],
        "rejected_branch_ids": rejected_ids,
        "rejection_reasons": risk["rejection_reasons"],
        "manifest_hash": rollout["manifest_hash"],
        "risk_report_hash": risk["report_hash"],
        "claim_boundary": _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY,
        "risk_claim_boundary": _MULTIVERSE_RISK_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "branch_rows": branch_rows,
        "coherence_range": {
            "minimum": min(min_values),
            "maximum": max(max_values),
            "final_minimum": min(final_values),
            "final_maximum": max(final_values),
            "mean_minimum": min(mean_values),
            "mean_maximum": max(mean_values),
        },
        "operator_summary": (
            "multiverse branch review: "
            f"{risk['approved_count']}/{rollout['branch_count']} branches approved"
        ),
        "operator_action": (
            "render as counterfactual review evidence only; no branch action "
            "may be applied without a separate safety-gated control workflow"
        ),
    }


def _normalise_multiverse_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate a multiverse counterfactual-rollout manifest for the panel.

    The manifest must use the rollout schema, keep the review-safe claim boundary
    and non-actuating/execution-disabled flags, name a supported backend, and have
    a branch count matching the length of its normalised branch records.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("multiverse manifest must be a mapping")
    if manifest.get("schema_name") != "multiverse_counterfactual_rollout":
        raise ValueError("schema_name must be multiverse_counterfactual_rollout")
    schema_version = _require_non_empty_text(
        manifest.get("schema_version"),
        "schema_version",
    )
    branch_count = _positive_int(
        manifest.get("branch_count"), "branch_count", minimum=1
    )
    horizon = _positive_int(manifest.get("horizon"), "horizon", minimum=1)
    backend = _require_non_empty_text(manifest.get("backend"), "backend")
    if backend not in _MULTIVERSE_BACKENDS:
        raise ValueError("backend must be a supported multiverse rollout backend")
    if manifest.get("non_actuating") is not True:
        raise ValueError("multiverse manifest must be non_actuating")
    if manifest.get("execution_disabled") is not True:
        raise ValueError("multiverse manifest execution_disabled must be true")
    if manifest.get("claim_boundary") != _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY:
        raise ValueError("multiverse manifest claim boundary is invalid")
    branch_records = _normalise_multiverse_branch_records(
        manifest.get("branch_records")
    )
    if len(branch_records) != branch_count:
        raise ValueError("branch_count must match branch_records length")
    return {
        "schema_name": "multiverse_counterfactual_rollout",
        "schema_version": schema_version,
        "branch_count": branch_count,
        "horizon": horizon,
        "backend": backend,
        "non_actuating": True,
        "execution_disabled": True,
        "claim_boundary": _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY,
        "manifest_hash": _require_sha256_hex(
            manifest.get("manifest_hash"),
            "manifest_hash",
        ),
        "branch_records": branch_records,
    }


def _normalise_multiverse_branch_records(
    records: object,
) -> tuple[dict[str, object], ...]:
    """Validate rollout branch records.

    Each must have a unique id, matching action labels, and an ordered
    ``min_R <= mean_R <= max_R`` interval that contains ``final_R``.
    """
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("branch_records must be a sequence")
    if not records:
        raise ValueError("branch_records must be non-empty")
    normalised: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("branch_records entries must be mappings")
        branch_id = _require_non_empty_text(record.get("branch_id"), "branch_id")
        if branch_id in seen_ids:
            raise ValueError("branch_records must have unique branch_id values")
        seen_ids.add(branch_id)
        action_count = _non_negative_int(record.get("action_count"), "action_count")
        action_labels = _normalise_text_sequence(
            record.get("action_labels"),
            "action_labels",
        )
        if len(action_labels) != action_count:
            raise ValueError("action_count must match action_labels length")
        final_r = _unit_interval_number(record.get("final_R"), "final_R")
        mean_r = _unit_interval_number(record.get("mean_R"), "mean_R")
        min_r = _unit_interval_number(record.get("min_R"), "min_R")
        max_r = _unit_interval_number(record.get("max_R"), "max_R")
        if min_r > mean_r + 1e-12 or mean_r > max_r + 1e-12:
            raise ValueError("R interval must satisfy min_R <= mean_R <= max_R")
        if final_r < min_r - 1e-12 or final_r > max_r + 1e-12:
            raise ValueError("R interval must contain final_R")
        normalised.append(
            {
                "branch_index": index,
                "branch_id": branch_id,
                "branch_hash": _require_sha256_hex(
                    record.get("branch_hash"),
                    "branch_hash",
                ),
                "action_count": action_count,
                "action_labels": list(action_labels),
                "topology_edge_count": _non_negative_int(
                    record.get("topology_edge_count"),
                    "topology_edge_count",
                ),
                "topology_scale": _non_negative_float(
                    record.get("topology_scale"),
                    "topology_scale",
                ),
                "final_R": final_r,
                "mean_R": mean_r,
                "min_R": min_r,
                "max_R": max_r,
                "final_psi": _non_negative_float(
                    record.get("final_psi"),
                    "final_psi",
                ),
            }
        )
    return tuple(normalised)


def _normalise_multiverse_risk_report(
    risk_report: Mapping[str, object],
) -> dict[str, object]:
    """Validate a multiverse branch-risk-gate report for the panel.

    The report must use the risk-gate schema, keep the review-safe claim boundary
    and non-actuating flags, and have a branch count and approved/rejected counts
    that agree with the per-branch decisions.
    """
    if not isinstance(risk_report, Mapping):
        raise ValueError("multiverse risk report must be a mapping")
    if risk_report.get("schema_name") != "multiverse_branch_risk_gate":
        raise ValueError("schema_name must be multiverse_branch_risk_gate")
    if risk_report.get("non_actuating") is not True:
        raise ValueError("multiverse risk report must be non_actuating")
    if risk_report.get("execution_disabled") is not True:
        raise ValueError("multiverse risk report execution_disabled must be true")
    if risk_report.get("claim_boundary") != _MULTIVERSE_RISK_CLAIM_BOUNDARY:
        raise ValueError("multiverse risk report claim boundary is invalid")
    decisions = _normalise_multiverse_risk_decisions(
        risk_report.get("branch_decisions")
    )
    branch_count = _positive_int(
        risk_report.get("branch_count"),
        "branch_count",
        minimum=1,
    )
    if branch_count != len(decisions):
        raise ValueError("branch_count must match branch_decisions length")
    approved_count = _non_negative_int(
        risk_report.get("approved_count"),
        "approved_count",
    )
    rejected_count = _non_negative_int(
        risk_report.get("rejected_count"),
        "rejected_count",
    )
    if approved_count + rejected_count != branch_count:
        raise ValueError("approved_count and rejected_count must sum to branch_count")
    if approved_count != sum(1 for decision in decisions if decision["approved"]):
        raise ValueError("approved_count must match approved branch decisions")
    if rejected_count != sum(1 for decision in decisions if not decision["approved"]):
        raise ValueError("rejected_count must match rejected branch decisions")
    return {
        "schema_name": "multiverse_branch_risk_gate",
        "schema_version": _require_non_empty_text(
            risk_report.get("schema_version"),
            "schema_version",
        ),
        "branch_count": branch_count,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "safest_branch_id": _optional_text(
            risk_report.get("safest_branch_id"),
            "safest_branch_id",
        ),
        "safest_branch_hash": _optional_sha256_hex(
            risk_report.get("safest_branch_hash"),
            "safest_branch_hash",
        ),
        "rejection_reasons": list(
            _normalise_optional_text_sequence(
                risk_report.get("rejection_reasons"),
                "rejection_reasons",
            )
        ),
        "claim_boundary": _MULTIVERSE_RISK_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "report_hash": _require_sha256_hex(
            risk_report.get("report_hash"), "report_hash"
        ),
        "branch_decisions": decisions,
    }


def _normalise_multiverse_risk_decisions(
    decisions: object,
) -> tuple[dict[str, object], ...]:
    """Validate per-branch risk decisions.

    Each must have a unique id, an ordered R interval, and an explicit approved
    flag with optional rejection reasons.
    """
    if isinstance(decisions, str | bytes) or not isinstance(decisions, Sequence):
        raise ValueError("branch_decisions must be a sequence")
    if not decisions:
        raise ValueError("branch_decisions must be non-empty")
    normalised: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("branch_decisions entries must be mappings")
        branch_id = _require_non_empty_text(decision.get("branch_id"), "branch_id")
        if branch_id in seen_ids:
            raise ValueError("branch_decisions must have unique branch_id values")
        seen_ids.add(branch_id)
        final_r = _unit_interval_number(decision.get("final_R"), "final_R")
        mean_r = _unit_interval_number(decision.get("mean_R"), "mean_R")
        min_r = _unit_interval_number(decision.get("min_R"), "min_R")
        max_r = _unit_interval_number(decision.get("max_R"), "max_R")
        if min_r > mean_r + 1e-12 or mean_r > max_r + 1e-12:
            raise ValueError("risk decision R interval is invalid")
        normalised.append(
            {
                "branch_id": branch_id,
                "branch_hash": _require_sha256_hex(
                    decision.get("branch_hash"),
                    "branch_hash",
                ),
                "final_R": final_r,
                "mean_R": mean_r,
                "min_R": min_r,
                "max_R": max_r,
                "action_count": _non_negative_int(
                    decision.get("action_count"),
                    "action_count",
                ),
                "topology_edge_count": _optional_non_negative_int(
                    decision.get("topology_edge_count"),
                    "topology_edge_count",
                ),
                "topology_scale": _optional_non_negative_float(
                    decision.get("topology_scale"),
                    "topology_scale",
                ),
                "approved": _required_bool(decision.get("approved"), "approved"),
                "rejection_reasons": list(
                    _normalise_optional_text_sequence(
                        decision.get("rejection_reasons"),
                        "rejection_reasons",
                    )
                ),
            }
        )
    return tuple(normalised)


def _join_multiverse_branch_rows(
    rollout: Mapping[str, object],
    risk: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Join rollout branches with their risk decisions into combined table rows.

    Every rollout branch must have a risk decision with a matching hash and
    metrics; raises ``ValueError`` on any mismatch or missing decision.
    """
    branch_records = cast("tuple[dict[str, object], ...]", rollout["branch_records"])
    decisions = cast("tuple[dict[str, object], ...]", risk["branch_decisions"])
    decision_by_id = {cast("str", item["branch_id"]): item for item in decisions}
    rows: list[dict[str, object]] = []
    for branch in branch_records:
        branch_id = cast("str", branch["branch_id"])
        decision = decision_by_id.get(branch_id)
        if decision is None or decision["branch_hash"] != branch["branch_hash"]:
            raise ValueError("risk decision must match every rollout branch")
        for field_name in ("final_R", "mean_R", "min_R", "max_R", "action_count"):
            if decision[field_name] != branch[field_name]:
                raise ValueError(
                    f"risk decision {field_name} must match rollout branch"
                )
        rows.append(
            {
                "branch_index": branch["branch_index"],
                "branch_id": branch_id,
                "branch_hash": branch["branch_hash"],
                "action_count": branch["action_count"],
                "action_labels": branch["action_labels"],
                "topology_edge_count": branch["topology_edge_count"],
                "topology_scale": branch["topology_scale"],
                "final_R": branch["final_R"],
                "mean_R": branch["mean_R"],
                "min_R": branch["min_R"],
                "max_R": branch["max_R"],
                "final_psi": branch["final_psi"],
                "risk_approved": decision["approved"],
                "risk_rejection_reasons": decision["rejection_reasons"],
            }
        )
    return tuple(rows)


def _optional_text(value: object, name: str) -> str | None:
    """Return ``None`` for a null value, else the validated non-empty text."""
    if value is None:
        return None
    return _require_non_empty_text(value, name)


def _optional_non_negative_float(value: object, name: str) -> float | None:
    """Return ``None`` for a null value, else the validated non-negative float."""
    if value is None:
        return None
    return _non_negative_float(value, name)
