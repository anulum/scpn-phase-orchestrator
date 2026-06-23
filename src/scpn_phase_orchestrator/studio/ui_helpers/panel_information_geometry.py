# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio information-geometry review panel

"""Information-geometry control review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from ._shared import (
    _finite_number,
    _non_negative_float,
    _normalise_float_sequence,
    _normalise_information_geometry_gradient,
    _normalise_text_sequence,
    _positive_float,
    _require_non_empty_text,
    _require_sha256_hex,
)

_INFORMATION_GEOMETRY_CLAIM_BOUNDARY = "information_geometry_control_not_live_actuation"


_INFORMATION_GEOMETRY_BACKENDS = frozenset(
    {
        "numpy_jax_compatible_information_geometry",
        "jax_native_information_geometry",
    }
)


def build_information_geometry_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    scenarios: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for information-geometry control evidence.

    The panel renders already-computed Fisher-Rao/Wasserstein review proposals
    and deterministic scenario fixtures. It validates metric tensors, simplex
    coordinates, natural-gradient tangents, geodesic/curvature metrics, hash
    fields, and disabled-execution boundaries before exposing anything to the
    operator surface. No returned field is an executable control channel.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    scenarios : Sequence[Mapping[str, object]]
        Scenario records.

    Returns
    -------
    dict[str, object]
        A Studio panel for information-geometry control evidence.
    """
    normalised_records = _normalise_information_geometry_records(records)
    normalised_scenarios, candidate_rows = _normalise_information_geometry_scenarios(
        scenarios
    )
    fisher_values = [
        cast("float", record["fisher_rao_distance"]) for record in normalised_records
    ]
    wasserstein_values = [
        cast("float", record["wasserstein_distance"]) for record in normalised_records
    ]
    gradient_values = [
        cast("float", record["natural_gradient_norm"]) for record in normalised_records
    ]
    curvature_values = [
        cast("float", record["curvature_proxy"]) for record in normalised_records
    ]
    metric_values = [
        value
        for record in normalised_records
        for row in cast("tuple[tuple[float, ...], ...]", record["metric_tensor"])
        for value in row
    ]
    metric_diagonal_values = [
        row[index]
        for record in normalised_records
        for index, row in enumerate(
            cast("tuple[tuple[float, ...], ...]", record["metric_tensor"])
        )
    ]
    backends = tuple(
        sorted({cast("str", record["backend"]) for record in normalised_records})
    )
    scenario_domains = tuple(
        sorted({cast("str", scenario["domain"]) for scenario in normalised_scenarios})
    )
    return {
        "panel_kind": "studio_information_geometry_panel",
        "supervisor": "information_geometry_control",
        "proposal_count": len(normalised_records),
        "scenario_count": len(normalised_scenarios),
        "claim_boundary": _INFORMATION_GEOMETRY_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "backends": backends,
        "scenario_domains": scenario_domains,
        "latest": normalised_records[-1],
        "series": normalised_records,
        "candidate_rows": candidate_rows,
        "fisher_rao_range": {
            "minimum": min(fisher_values),
            "maximum": max(fisher_values),
        },
        "wasserstein_range": {
            "minimum": min(wasserstein_values),
            "maximum": max(wasserstein_values),
        },
        "natural_gradient_range": {
            "minimum": min(gradient_values),
            "maximum": max(gradient_values),
        },
        "curvature_range": {
            "minimum": min(curvature_values),
            "maximum": max(curvature_values),
        },
        "metric_tensor_range": {
            "minimum": min(metric_values),
            "maximum": max(metric_values),
        },
        "metric_diagonal_range": {
            "minimum": min(metric_diagonal_values),
            "maximum": max(metric_diagonal_values),
        },
        "operator_summary": (
            "information-geometry review: "
            f"{len(normalised_records)} proposal record(s) across "
            f"{len(backends)} backend(s); max Fisher-Rao distance "
            f"{max(fisher_values):.6g}"
        ),
        "operator_action": (
            "render as non-actuating geometry-aware control evidence; compare "
            "Fisher-Rao/Wasserstein distances, metric conditioning, and "
            "natural-gradient magnitude before any separately gated policy use"
        ),
    }


def _normalise_information_geometry_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and normalise a non-empty sequence of information-geometry records.

    Each record must keep the review-safe claim boundary and the non-actuating and
    execution-disabled flags, name a supported backend, and carry a valid state
    (simplex point and Fisher-information matrix). Raises ``ValueError`` otherwise.
    """
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("information-geometry records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("information-geometry record must be a mapping")
        label = f"information-geometry record {index}"
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _INFORMATION_GEOMETRY_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if record.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        backend = _require_non_empty_text(record.get("backend"), f"{label} backend")
        if backend not in _INFORMATION_GEOMETRY_BACKENDS:
            raise ValueError(f"{label} backend is not supported")
        state_raw = record.get("state")
        if not isinstance(state_raw, Mapping):
            raise ValueError(f"{label} state must be a mapping")
        simplex = _normalise_simplex_sequence(
            state_raw.get("simplex_coordinates"),
            f"{label} simplex_coordinates",
        )
        target = _normalise_simplex_sequence(
            state_raw.get("target_coordinates"),
            f"{label} target_coordinates",
        )
        if len(simplex) != len(target):
            raise ValueError(f"{label} target_coordinates must match simplex shape")
        metric_tensor = _normalise_square_float_matrix(
            state_raw.get("metric_tensor"),
            f"{label} metric_tensor",
            expected_size=len(simplex),
            positive_diagonal=True,
        )
        tangent_vector = _normalise_float_sequence(
            state_raw.get("tangent_vector"),
            f"{label} tangent_vector",
        )
        if len(tangent_vector) != len(simplex):
            raise ValueError(f"{label} tangent_vector must match simplex shape")
        geodesic_length = _non_negative_float(
            state_raw.get("geodesic_length"),
            f"{label} geodesic_length",
        )
        curvature_proxy = _non_negative_float(
            record.get("curvature_proxy"),
            f"{label} curvature_proxy",
        )
        state_curvature = _non_negative_float(
            state_raw.get("curvature_proxy"),
            f"{label} state.curvature_proxy",
        )
        if abs(state_curvature - curvature_proxy) > 1e-12:
            raise ValueError(f"{label} state curvature_proxy must match proposal")
        fisher_rao = _non_negative_float(
            record.get("fisher_rao_distance"),
            f"{label} fisher_rao_distance",
        )
        if abs(geodesic_length - fisher_rao) > 1e-12:
            raise ValueError(f"{label} geodesic_length must match Fisher-Rao distance")
        normalised.append(
            {
                "step": index + 1,
                "backend": backend,
                "claim_boundary": claim_boundary,
                "non_actuating": True,
                "execution_disabled": True,
                "proposal_hash": _require_sha256_hex(
                    record.get("proposal_hash"),
                    f"{label} proposal_hash",
                ),
                "knob": _single_information_geometry_action(record, label)["knob"],
                "scope": _single_information_geometry_action(record, label)["scope"],
                "action_value": _single_information_geometry_action(record, label)[
                    "value"
                ],
                "ttl_s": _single_information_geometry_action(record, label)["ttl_s"],
                "justification": _single_information_geometry_action(record, label)[
                    "justification"
                ],
                "fisher_rao_distance": fisher_rao,
                "wasserstein_distance": _non_negative_float(
                    record.get("wasserstein_distance"),
                    f"{label} wasserstein_distance",
                ),
                "natural_gradient_norm": _non_negative_float(
                    record.get("natural_gradient_norm"),
                    f"{label} natural_gradient_norm",
                ),
                "curvature_proxy": curvature_proxy,
                "simplex_coordinates": list(simplex),
                "target_coordinates": list(target),
                "metric_tensor": metric_tensor,
                "tangent_vector": list(tangent_vector),
                "geodesic_length": geodesic_length,
            }
        )
    return tuple(normalised)


def _normalise_information_geometry_scenarios(
    scenarios: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Validate information-geometry scenarios and return summaries and table rows.

    Each scenario must keep the review-safe claim boundary and non-actuating flags;
    returns the per-scenario summaries and the flattened candidate rows.
    """
    if isinstance(scenarios, Mapping) or not isinstance(scenarios, Sequence):
        raise ValueError("information-geometry scenarios must be a sequence")
    normalised_scenarios: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError("information-geometry scenario must be a mapping")
        label = f"information-geometry scenario {index}"
        claim_boundary = _require_non_empty_text(
            scenario.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _INFORMATION_GEOMETRY_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if scenario.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if scenario.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        current = _normalise_simplex_sequence(
            scenario.get("current_distribution"),
            f"{label} current_distribution",
        )
        target = _normalise_simplex_sequence(
            scenario.get("target_distribution"),
            f"{label} target_distribution",
        )
        if len(current) != len(target):
            raise ValueError(f"{label} target_distribution must match current shape")
        objectives = _normalise_text_sequence(
            scenario.get("objective_labels"),
            f"{label} objective_labels",
        )
        knob_hints = _normalise_text_sequence(
            scenario.get("knob_hints"),
            f"{label} knob_hints",
        )
        control_gradient = _normalise_information_geometry_gradient(
            scenario.get("control_gradient"),
            f"{label} control_gradient",
        )
        scenario_id = _require_non_empty_text(
            scenario.get("scenario_id"),
            f"{label} scenario_id",
        )
        domain = _require_non_empty_text(scenario.get("domain"), f"{label} domain")
        scenario_hash = _require_sha256_hex(
            scenario.get("scenario_hash"),
            f"{label} scenario_hash",
        )
        max_step = _positive_float(scenario.get("max_step"), f"{label} max_step")
        normalised_scenario = {
            "domain": domain,
            "scenario_id": scenario_id,
            "scenario_hash": scenario_hash,
            "claim_boundary": claim_boundary,
            "non_actuating": True,
            "execution_disabled": True,
            "objective_labels": list(objectives),
            "control_gradient": [
                {"knob": knob, "value": value} for knob, value in control_gradient
            ],
            "knob_hints": list(knob_hints),
            "max_step": max_step,
            "dimension": len(current),
        }
        normalised_scenarios.append(normalised_scenario)
        candidate_rows.append(
            {
                "domain": domain,
                "scenario_id": scenario_id,
                "scenario_hash": scenario_hash,
                "dimension": len(current),
                "objective_count": len(objectives),
                "control_knobs": tuple(knob for knob, _ in control_gradient),
                "max_step": max_step,
            }
        )
    return (tuple(normalised_scenarios), tuple(candidate_rows))


def _normalise_simplex_sequence(value: object, name: str) -> tuple[float, ...]:
    """Return ``value`` as a non-negative float tuple normalised to unit mass."""
    values = _normalise_float_sequence(value, name)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must be non-negative")
    mass = sum(values)
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    if abs(mass - 1.0) > 1e-9:
        raise ValueError(f"{name} must be normalised to unit mass")
    return values


def _normalise_square_float_matrix(
    value: object,
    name: str,
    *,
    expected_size: int,
    positive_diagonal: bool,
) -> tuple[tuple[float, ...], ...]:
    """Return ``value`` as a symmetric square matrix of ``expected_size``, else raise.

    Optionally requires a strictly positive diagonal; the matrix must be square,
    of the expected size, and symmetric to a tight tolerance.
    """
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a square matrix")
    if len(value) != expected_size:
        raise ValueError(f"{name} row count must match simplex shape")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        row_values = _normalise_float_sequence(row, name)
        if len(row_values) != expected_size:
            raise ValueError(f"{name} column count must match simplex shape")
        if positive_diagonal and row_values[row_index] <= 0.0:
            raise ValueError(f"{name} diagonal must be strictly positive")
        rows.append(row_values)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return tuple(rows)


def _single_information_geometry_action(
    record: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    """Return the single validated review action (knob, scope, value, ttl, reason)."""
    actions = record.get("action_proposals")
    if isinstance(actions, str | bytes) or not isinstance(actions, Sequence):
        raise ValueError(f"{label} action_proposals must be a sequence")
    if len(actions) != 1:
        raise ValueError(f"{label} action_proposals must contain one review action")
    action = actions[0]
    if not isinstance(action, Mapping):
        raise ValueError(f"{label} action_proposals entries must be mappings")
    return {
        "knob": _require_non_empty_text(action.get("knob"), f"{label} knob"),
        "scope": _require_non_empty_text(action.get("scope"), f"{label} scope"),
        "value": _finite_number(action.get("value"), f"{label} action value"),
        "ttl_s": _positive_float(action.get("ttl_s"), f"{label} ttl_s"),
        "justification": _require_non_empty_text(
            action.get("justification"),
            f"{label} justification",
        ),
    }
