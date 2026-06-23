# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio chart payload builders

"""Series, regime, and integrated-information chart payload builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from ._shared import (
    _finite_number,
    _non_negative_float,
    _positive_int,
    _require_non_empty_text,
    _unit_interval_number,
)


def build_series_chart_payload(
    label: str,
    values: Sequence[float],
) -> list[dict[str, float | int]]:
    """Return dense chart rows for a scalar time-series.

    Parameters
    ----------
    label : str
        Series or chart label.
    values : Sequence[float]
        Scalar time-series values.

    Returns
    -------
    list[dict[str, float | int]]
        Dense chart rows for a scalar time-series.
    """
    _require_non_empty_text(label, "label")
    return [
        {"step": index, label: _finite_number(value, label)}
        for index, value in enumerate(values, 1)
    ]


def build_regime_chart_payload(regimes: Sequence[str]) -> list[dict[str, object]]:
    """Return deterministic chart rows for regime timelines.

    Parameters
    ----------
    regimes : Sequence[str]
        Per-step regime labels.

    Returns
    -------
    list[dict[str, object]]
        Deterministic chart rows for regime timelines.
    """
    regime_levels = {
        "critical": 0.0,
        "degraded": 1.0,
        "recovery": 1.5,
        "nominal": 2.0,
    }
    rows: list[dict[str, object]] = []
    for index, regime in enumerate(regimes, 1):
        regime_text = _require_non_empty_text(regime, "regime")
        rows.append(
            {
                "step": index,
                "regime": regime_text,
                "regime_level": regime_levels.get(regime_text, 0.0),
            }
        )
    return rows


def build_integrated_information_panel(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel payload for integrated-information audit records.

    The panel is deliberately pure and non-actuating: it converts validated
    monitor audit records into a deterministic operator payload suitable for
    rendering charts, latest-value tiles, and partition review cards. The input
    must preserve the monitor's explicit claim boundary so Studio cannot display
    the Phi proxy as a theoretical IIT or consciousness claim.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for integrated-information audit records.
    """
    normalised_records = _normalise_integrated_information_records(records)
    latest = normalised_records[-1]
    strongest = max(
        normalised_records,
        key=lambda item: cast("float", item["phi"]),
    )
    phi_values = [cast("float", item["phi"]) for item in normalised_records]
    normalised_phi_values = [
        cast("float", item["normalised_phi"]) for item in normalised_records
    ]
    total_integration_values = [
        cast("float", item["total_integration"]) for item in normalised_records
    ]
    return {
        "panel_kind": "studio_integrated_information_panel",
        "monitor": "integrated_information",
        "record_count": len(normalised_records),
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
        "latest": latest,
        "strongest_partition": strongest,
        "series": normalised_records,
        "phi_range": {
            "min": min(phi_values),
            "max": max(phi_values),
        },
        "normalised_phi_range": {
            "min": min(normalised_phi_values),
            "max": max(normalised_phi_values),
        },
        "total_integration_range": {
            "min": min(total_integration_values),
            "max": max(total_integration_values),
        },
        "operator_summary": (
            "latest Phi proxy "
            f"{cast('float', latest['phi']):.6g}; latest normalised Phi "
            f"{cast('float', latest['normalised_phi']):.6g}; records "
            f"{len(normalised_records)}"
        ),
        "operator_action": (
            "render as an engineering integration proxy; preserve the claim "
            "boundary and review the minimum partition before operational use"
        ),
        "actuation_permitted": False,
        "consciousness_claim_permitted": False,
    }


def _normalise_integrated_information_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate a non-empty sequence of integrated-information (IIT) records."""
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("integrated-information records must be a sequence")
    if not records:
        raise ValueError("integrated-information records must not be empty")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise ValueError("integrated-information records must be mappings")
        monitor = record.get("monitor", "integrated_information")
        if monitor != "integrated_information":
            raise ValueError("integrated-information monitor tag is required")
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"),
            "claim_boundary",
        )
        if claim_boundary != "engineering_proxy_not_theoretical_iit":
            raise ValueError("integrated-information claim boundary is required")
        n_bins = _positive_int(record.get("n_bins"), "n_bins", minimum=2)
        phi = _bounded_information_scalar(record.get("phi"), "phi", n_bins)
        normalised_phi = _unit_interval_number(
            record.get("normalised_phi"),
            "normalised_phi",
        )
        expected_normalised_phi = min(1.0, phi / float(np.log(n_bins)))
        if abs(normalised_phi - expected_normalised_phi) > 1e-12:
            raise ValueError("normalised_phi must match phi/log(n_bins)")
        total_integration = _bounded_information_scalar(
            record.get("total_integration"),
            "total_integration",
            n_bins,
        )
        if phi > total_integration + 1e-12:
            raise ValueError("phi must not exceed total_integration")
        minimum_partition = _normalise_integrated_information_partition(
            record.get("minimum_partition"),
        )
        pairwise_shape = _integrated_information_pairwise_shape(
            record.get("pairwise_mi"),
            n_bins,
        )
        if pairwise_shape is not None:
            partition_nodes = {node for side in minimum_partition for node in side}
            if partition_nodes != set(range(pairwise_shape[0])):
                raise ValueError("minimum_partition must cover pairwise_mi nodes")
        normalised.append(
            {
                "step": index,
                "phi": phi,
                "normalised_phi": normalised_phi,
                "total_integration": total_integration,
                "minimum_partition": [list(side) for side in minimum_partition],
                "n_bins": n_bins,
                "claim_boundary": claim_boundary,
            }
        )
    return tuple(normalised)


def _bounded_information_scalar(
    value: object,
    name: str,
    n_bins: int,
) -> float:
    """Return a non-negative scalar not exceeding ``log(n_bins)``, else raise."""
    scalar = _non_negative_float(value, name)
    if scalar > float(np.log(n_bins)) + 1e-12:
        raise ValueError(f"{name} must not exceed log(n_bins)")
    return scalar


def _normalise_integrated_information_partition(
    value: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate the minimum-information partition's two groups of unique indices."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("minimum_partition must contain two index groups")
    if len(value) != 2:
        raise ValueError("minimum_partition must contain two index groups")
    sides: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for side in value:
        if isinstance(side, str | bytes) or not isinstance(side, Sequence):
            raise ValueError("minimum_partition groups must be index sequences")
        if not side:
            raise ValueError("minimum_partition groups must not be empty")
        normalised_side: list[int] = []
        for node in side:
            if isinstance(node, bool) or not isinstance(node, int):
                raise ValueError("minimum_partition indices must be integers")
            checked = int(node)
            if checked < 0:
                raise ValueError("minimum_partition indices must be non-negative")
            if checked in seen:
                raise ValueError("minimum_partition indices must be unique")
            seen.add(checked)
            normalised_side.append(checked)
        sides.append(tuple(sorted(normalised_side)))
    return (sides[0], sides[1])


def _integrated_information_pairwise_shape(
    value: object,
    n_bins: int,
) -> tuple[int, int] | None:
    """Return the square pairwise-MI matrix shape (at least 2x2), or ``None``."""
    if value is None:
        return None
    matrix = np.asarray(value)
    if matrix.dtype == np.bool_ or np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("pairwise_mi must be finite real-valued")
    try:
        checked = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("pairwise_mi must be finite real-valued") from exc
    if checked.ndim != 2 or checked.shape[0] != checked.shape[1]:
        raise ValueError("pairwise_mi must be a square matrix")
    if checked.shape[0] < 2:
        raise ValueError("pairwise_mi must contain at least two oscillators")
    if not np.all(np.isfinite(checked)):
        raise ValueError("pairwise_mi must be finite real-valued")
    if np.any(checked < -1e-12):
        raise ValueError("pairwise_mi must be non-negative")
    if np.any(checked > float(np.log(n_bins)) + 1e-12):
        raise ValueError("pairwise_mi entries must not exceed log(n_bins)")
    if not np.allclose(checked, checked.T, rtol=1e-12, atol=1e-12):
        raise ValueError("pairwise_mi must be symmetric")
    return (int(checked.shape[0]), int(checked.shape[1]))
