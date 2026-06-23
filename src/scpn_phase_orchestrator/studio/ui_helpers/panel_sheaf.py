# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio sheaf-cohomology review panel

"""Sheaf-cohomology review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _normalise_float_sequence,
    _normalise_text_sequence,
    _positive_float,
    _positive_int,
    _require_non_empty_text,
)

_SHEAF_COHOMOLOGY_CLAIM_BOUNDARY = "sheaf_cohomology_review_not_live_actuation"


_SHEAF_RESULT_METHOD = "directed_cellular_sheaf_laplacian"


_SHEAF_CONTROL_METHOD = "sheaf_laplacian_gradient_descent_review"


def build_sheaf_cohomology_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    summaries: Sequence[Mapping[str, object]],
    control_proposals: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel for sheaf-cohomology review evidence.

    The panel renders already-computed sheaf-Laplacian obstruction records,
    residual triage summaries, and review-only control proposals. It validates
    cohomology dimensions, finite obstruction/energy metrics, residual rows,
    disabled execution gates, and monotone accepted projections before exposing
    evidence to Studio. No returned field is an executable control channel.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    summaries : Sequence[Mapping[str, object]]
        Summary records.
    control_proposals : Sequence[Mapping[str, object]]
        Sheaf control-proposal records.

    Returns
    -------
    dict[str, object]
        A Studio panel for sheaf-cohomology review evidence.
    """
    normalised_records = _normalise_sheaf_cohomology_records(records)
    normalised_summaries, residual_rows = _normalise_sheaf_obstruction_summaries(
        summaries
    )
    normalised_proposals = _normalise_sheaf_control_proposals(control_proposals)
    obstruction_scores = [
        cast("float", record["obstruction_score"]) for record in normalised_records
    ]
    consistency_energies = [
        cast("float", record["consistency_energy"]) for record in normalised_records
    ]
    kernel_dimensions = [
        cast("int", record["kernel_dimension"]) for record in normalised_records
    ]
    obstruction_dimensions = [
        cast("int", record["obstruction_dimension"]) for record in normalised_records
    ]
    accepted_count = sum(
        1
        for proposal in normalised_proposals
        if proposal["accepted_for_review"] is True
    )
    critical_count = sum(
        1 for summary in normalised_summaries if summary["severity"] == "critical"
    )
    return {
        "panel_kind": "studio_sheaf_cohomology_panel",
        "supervisor": "sheaf_cohomology_control",
        "claim_boundary": _SHEAF_COHOMOLOGY_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "record_count": len(normalised_records),
        "summary_count": len(normalised_summaries),
        "control_proposal_count": len(normalised_proposals),
        "accepted_control_proposal_count": accepted_count,
        "critical_summary_count": critical_count,
        "records": normalised_records,
        "summaries": normalised_summaries,
        "control_proposals": normalised_proposals,
        "top_residual_rows": residual_rows,
        "obstruction_range": {
            "minimum": min(obstruction_scores),
            "maximum": max(obstruction_scores),
        },
        "consistency_energy_range": {
            "minimum": min(consistency_energies),
            "maximum": max(consistency_energies),
        },
        "cohomology_dimension_range": {
            "kernel_minimum": min(kernel_dimensions),
            "kernel_maximum": max(kernel_dimensions),
            "obstruction_minimum": min(obstruction_dimensions),
            "obstruction_maximum": max(obstruction_dimensions),
        },
        "operator_summary": (
            "sheaf-cohomology review: "
            f"{len(normalised_records)} obstruction record(s), "
            f"{len(residual_rows)} residual edge row(s), "
            f"{accepted_count}/{len(normalised_proposals)} accepted proposal(s)"
        ),
        "operator_action": (
            "render as non-actuating sheaf-Laplacian obstruction evidence; "
            "review residual edges and cohomology-dimension changes before any "
            "separately approved operator workflow"
        ),
    }


def _normalise_sheaf_cohomology_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate sheaf-cohomology records for the panel.

    Each record must use the supported method and carry a square Laplacian shape
    and a node-square residual shape whose product matches the Laplacian, plus
    non-negative obstruction/consistency scores, kernel/obstruction dimensions,
    and tolerance.
    """
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("sheaf-cohomology records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("sheaf-cohomology record must be a mapping")
        label = f"sheaf-cohomology record {index}"
        method = _require_non_empty_text(record.get("method"), f"{label} method")
        if method != _SHEAF_RESULT_METHOD:
            raise ValueError(f"{label} method is unsupported")
        laplacian_shape = _normalise_sheaf_shape(
            record.get("laplacian_shape"),
            f"{label} laplacian_shape",
            expected_rank=2,
        )
        residual_shape = _normalise_sheaf_shape(
            record.get("residual_shape"),
            f"{label} residual_shape",
            expected_rank=3,
        )
        if laplacian_shape[0] != laplacian_shape[1]:
            raise ValueError(f"{label} laplacian_shape must be square")
        if residual_shape[0] != residual_shape[1]:
            raise ValueError(f"{label} residual_shape must be node-square")
        if residual_shape[0] * residual_shape[2] != laplacian_shape[0]:
            raise ValueError(f"{label} residual_shape must match laplacian_shape")
        normalised.append(
            {
                "method": _SHEAF_RESULT_METHOD,
                "obstruction_score": _non_negative_float(
                    record.get("obstruction_score"),
                    f"{label} obstruction_score",
                ),
                "consistency_energy": _non_negative_float(
                    record.get("consistency_energy"),
                    f"{label} consistency_energy",
                ),
                "kernel_dimension": _non_negative_int(
                    record.get("kernel_dimension"),
                    f"{label} kernel_dimension",
                ),
                "obstruction_dimension": _non_negative_int(
                    record.get("obstruction_dimension"),
                    f"{label} obstruction_dimension",
                ),
                "edge_count": _non_negative_int(
                    record.get("edge_count"),
                    f"{label} edge_count",
                ),
                "laplacian_shape": laplacian_shape,
                "residual_shape": residual_shape,
                "tolerance": _non_negative_float(
                    record.get("tolerance"),
                    f"{label} tolerance",
                ),
            }
        )
    return tuple(normalised)


def _normalise_sheaf_obstruction_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Validate sheaf-obstruction summaries and return summaries and residual rows.

    Each summary must carry a supported severity, ``warning <= critical``
    thresholds, and top residual edges; returns the normalised summaries and the
    flattened per-edge residual rows.
    """
    if (
        isinstance(summaries, Mapping)
        or not isinstance(summaries, Sequence)
        or not summaries
    ):
        raise ValueError("sheaf-obstruction summaries must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping):
            raise ValueError("sheaf-obstruction summary must be a mapping")
        label = f"sheaf-obstruction summary {index}"
        severity = _require_non_empty_text(summary.get("severity"), f"{label} severity")
        if severity not in {"nominal", "warning", "critical"}:
            raise ValueError(f"{label} severity is unsupported")
        warning_threshold = _non_negative_float(
            summary.get("warning_threshold"),
            f"{label} warning_threshold",
        )
        critical_threshold = _non_negative_float(
            summary.get("critical_threshold"),
            f"{label} critical_threshold",
        )
        if critical_threshold < warning_threshold:
            raise ValueError(f"{label} critical_threshold must be >= warning_threshold")
        top_edges = _normalise_sheaf_residual_rows(
            summary.get("top_residual_edges"),
            label,
        )
        obstruction_score = _non_negative_float(
            summary.get("obstruction_score"),
            f"{label} obstruction_score",
        )
        normalised.append(
            {
                "severity": severity,
                "obstruction_score": obstruction_score,
                "warning_threshold": warning_threshold,
                "critical_threshold": critical_threshold,
                "top_residual_edges": top_edges,
            }
        )
        for row in top_edges:
            residual_rows.append({"summary_index": index, "severity": severity, **row})
    return tuple(normalised), tuple(residual_rows)


def _normalise_sheaf_control_proposals(
    proposals: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate sheaf-control proposals (supported method) as review-only records."""
    if (
        isinstance(proposals, Mapping)
        or not isinstance(proposals, Sequence)
        or not proposals
    ):
        raise ValueError("sheaf-control proposals must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise ValueError("sheaf-control proposal must be a mapping")
        label = f"sheaf-control proposal {index}"
        method = _require_non_empty_text(proposal.get("method"), f"{label} method")
        if method != _SHEAF_CONTROL_METHOD:
            raise ValueError(f"{label} method is unsupported")
        if proposal.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if proposal.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        if proposal.get("operator_review_required") is not True:
            raise ValueError(f"{label} operator_review_required must be true")
        accepted = _strict_bool(
            proposal.get("accepted_for_review"),
            f"{label} accepted_for_review",
        )
        baseline_obstruction = _non_negative_float(
            proposal.get("baseline_obstruction_score"),
            f"{label} baseline_obstruction_score",
        )
        projected_obstruction = _non_negative_float(
            proposal.get("projected_obstruction_score"),
            f"{label} projected_obstruction_score",
        )
        baseline_energy = _non_negative_float(
            proposal.get("baseline_consistency_energy"),
            f"{label} baseline_consistency_energy",
        )
        projected_energy = _non_negative_float(
            proposal.get("projected_consistency_energy"),
            f"{label} projected_consistency_energy",
        )
        if accepted and projected_obstruction > baseline_obstruction + 1e-12:
            raise ValueError(f"{label} projected obstruction must be monotone")
        if accepted and projected_energy > baseline_energy + 1e-12:
            raise ValueError(f"{label} projected energy must be monotone")
        update_norm = _non_negative_float(
            proposal.get("update_norm"),
            f"{label} update_norm",
        )
        max_update_norm = _non_negative_float(
            proposal.get("max_update_norm"),
            f"{label} max_update_norm",
        )
        if update_norm > max_update_norm + 1e-12:
            raise ValueError(f"{label} update_norm exceeds max_update_norm")
        blocked_reasons = _normalise_text_sequence(
            proposal.get("blocked_reasons", ()),
            f"{label} blocked_reasons",
        )
        if accepted and blocked_reasons:
            raise ValueError(f"{label} accepted proposal must not be blocked")
        if not accepted and not blocked_reasons:
            raise ValueError(f"{label} rejected proposal requires blocked_reasons")
        normalised.append(
            {
                "method": _SHEAF_CONTROL_METHOD,
                "baseline_obstruction_score": baseline_obstruction,
                "projected_obstruction_score": projected_obstruction,
                "baseline_consistency_energy": baseline_energy,
                "projected_consistency_energy": projected_energy,
                "cohomology_dimensions": _normalise_sheaf_cohomology_dimensions(
                    proposal.get("cohomology_dimensions"),
                    label,
                ),
                "recommended_update_shape": _normalise_sheaf_shape(
                    proposal.get("recommended_update_shape"),
                    f"{label} recommended_update_shape",
                    expected_rank=2,
                ),
                "projected_node_state_shape": _normalise_sheaf_shape(
                    proposal.get("projected_node_state_shape"),
                    f"{label} projected_node_state_shape",
                    expected_rank=2,
                ),
                "update_norm": update_norm,
                "step_size": _positive_float(
                    proposal.get("step_size"),
                    f"{label} step_size",
                ),
                "max_update_norm": max_update_norm,
                "accepted_for_review": accepted,
                "non_actuating": True,
                "execution_disabled": True,
                "operator_review_required": True,
                "blocked_reasons": blocked_reasons,
            }
        )
    return tuple(normalised)


def _normalise_sheaf_residual_rows(
    value: object,
    label: str,
) -> tuple[dict[str, object], ...]:
    """Validate top residual-edge rows (target, source, norm, non-empty residual)."""
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise ValueError(f"{label} top_residual_edges must be a sequence")
    rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(value):
        row_label = f"{label} top_residual_edges[{index}]"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{row_label} must be a mapping")
        residual = _normalise_float_sequence(
            raw_row.get("residual"),
            f"{row_label} residual",
        )
        if not residual:
            raise ValueError(f"{row_label} residual must not be empty")
        rows.append(
            {
                "target": _non_negative_int(
                    raw_row.get("target"),
                    f"{row_label} target",
                ),
                "source": _non_negative_int(
                    raw_row.get("source"),
                    f"{row_label} source",
                ),
                "norm": _non_negative_float(
                    raw_row.get("norm"),
                    f"{row_label} norm",
                ),
                "residual": residual,
            }
        )
    return tuple(rows)


def _normalise_sheaf_cohomology_dimensions(
    value: object,
    label: str,
) -> dict[str, int]:
    """Validate the baseline/projected kernel and obstruction dimensions mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} cohomology_dimensions must be a mapping")
    return {
        key: _non_negative_int(value.get(key), f"{label} {key}")
        for key in (
            "baseline_kernel_dimension",
            "projected_kernel_dimension",
            "baseline_obstruction_dimension",
            "projected_obstruction_dimension",
        )
    }


def _normalise_sheaf_shape(
    value: object,
    label: str,
    *,
    expected_rank: int,
) -> tuple[int, ...]:
    """Return ``value`` as a positive-integer shape tuple of ``expected_rank``."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    shape = tuple(
        _positive_int(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if len(shape) != expected_rank:
        raise ValueError(f"{label} must have rank {expected_rank}")
    return shape


def _strict_bool(value: object, name: str) -> bool:
    """Return ``value`` if it is a real boolean, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value
