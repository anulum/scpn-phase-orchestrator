# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Infrastructure replay monitor suite

"""Deterministic infrastructure replay records for the integrated-information proxy.

These records are empirical benchmark proxies over circular phase trajectories
and are explicitly not theoretical IIT claims.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.information_integration import (
    integrated_information,
)

__all__ = [
    "build_infrastructure_integrated_information_replays",
]

FloatArray: TypeAlias = NDArray[np.float64]

_TWO_PI = 2.0 * np.pi


def build_infrastructure_integrated_information_replays(
    *,
    n_samples: int = 256,
    n_bins: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic infrastructure replay records.

    Args:
        n_samples: Number of trajectory samples per case. Must be an int >= 32.
        n_bins: Histogram bins for ``integrated_information``. Must be an int >= 2.

    Returns:
        Tuple of JSON-safe infrastructure replay records.

    Raises:
        ValueError: If parameters are invalid or the corpus does not satisfy
            ordering and schema validation.
    """
    _validate_replay_parameters(n_samples=n_samples, n_bins=n_bins)

    records = (
        _build_islanding_case(n_samples=n_samples, n_bins=n_bins),
        _build_resynchronisation_case(n_samples=n_samples, n_bins=n_bins),
        _build_traffic_spillback_case(n_samples=n_samples, n_bins=n_bins),
        _build_traffic_recovery_case(n_samples=n_samples, n_bins=n_bins),
    )

    _validate_replay_records(records)
    return records


def _validate_replay_parameters(*, n_samples: int, n_bins: int) -> None:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 32:
        raise ValueError("n_samples must be an integer >= 32")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")


def _validate_replay_records(records: tuple[dict[str, Any], ...]) -> None:
    if len(records) < 2:
        raise ValueError("replay corpus must contain at least two records")

    required_fields = {
        "domain",
        "case_name",
        "description",
        "n_oscillators",
        "n_samples",
        "n_bins",
        "phi",
        "normalised_phi",
        "total_integration",
        "minimum_partition",
        "expected_relationship",
        "claim_boundary",
        "non_actuating",
    }
    for record in records:
        missing = required_fields - set(record.keys())
        if missing:
            raise ValueError(f"replay record missing fields: {sorted(missing)}")
        if record["domain"] != "infrastructure":
            raise ValueError("all replay records must use domain=infrastructure")
        if record["claim_boundary"] != "engineering_proxy_not_theoretical_iit":
            raise ValueError("invalid claim boundary value")
        if record["non_actuating"] is not True:
            raise ValueError("replay records must be non-actuating")
        if not isinstance(record["n_oscillators"], int) or record["n_oscillators"] < 2:
            raise ValueError("n_oscillators must be at least two")
        if (
            not isinstance(record["minimum_partition"], list)
            or len(record["minimum_partition"]) != 2
        ):
            raise ValueError("minimum_partition must be a pair of index lists")
        if any(not isinstance(part, list) for part in record["minimum_partition"]):
            raise ValueError("minimum_partition entries must be lists")
        if not isinstance(record["phi"], float) or not np.isfinite(record["phi"]):
            raise ValueError("phi must be finite float")
        if not isinstance(record["normalised_phi"], float) or not np.isfinite(
            record["normalised_phi"]
        ):
            raise ValueError("normalised_phi must be finite float")
        if not isinstance(record["total_integration"], float) or not np.isfinite(
            record["total_integration"]
        ):
            raise ValueError("total_integration must be finite float")

    case_by_name = {record["case_name"]: record for record in records}
    for key in (
        "power_grid_islanding",
        "power_grid_resynchronisation",
        "traffic_spillback_fragmentation",
        "traffic_platoon_recovery",
    ):
        if key not in case_by_name:
            raise ValueError(f"missing replay case: {key}")

    if not (
        case_by_name["power_grid_resynchronisation"]["phi"]
        > case_by_name["power_grid_islanding"]["phi"]
    ):
        raise ValueError(
            "power grid re-synchronisation integration must exceed islanding"
        )
    if not (
        case_by_name["traffic_platoon_recovery"]["phi"]
        > case_by_name["traffic_spillback_fragmentation"]["phi"]
    ):
        raise ValueError(
            "traffic platoon recovery integration must exceed spillback fragmentation"
        )


def _build_record(
    *,
    case_name: str,
    description: str,
    phase_series: FloatArray,
    n_bins: int,
    expected_relationship: str,
) -> dict[str, Any]:
    result = integrated_information(phase_series, n_bins=n_bins)
    left, right = result.minimum_partition

    return {
        "domain": "infrastructure",
        "case_name": case_name,
        "description": description,
        "n_oscillators": int(phase_series.shape[0]),
        "n_samples": int(phase_series.shape[1]),
        "n_bins": int(result.n_bins),
        "phi": float(result.phi),
        "normalised_phi": float(result.normalised_phi),
        "total_integration": float(result.total_integration),
        "minimum_partition": [list(left), list(right)],
        "expected_relationship": expected_relationship,
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
        "non_actuating": True,
    }


def _time_axis(n_samples: int) -> NDArray[np.float64]:
    return np.arange(n_samples, dtype=np.float64)


def _power_grid_islanding_series(n_samples: int) -> FloatArray:
    t = _time_axis(n_samples) / n_samples
    group_a = np.array([0.08, 0.11, 0.14], dtype=np.float64)[:, None]
    group_b = np.array([0.31, 0.34, 0.37], dtype=np.float64)[:, None]
    phase_a = (group_a * t[None, :] + np.array([0.0, 0.2, 0.4])[:, None]) % 1.0
    phase_b = (group_b * t[None, :] + np.array([1.0, 1.2, 1.4])[:, None]) % 1.0
    return (np.vstack((phase_a, phase_b)) * _TWO_PI) % _TWO_PI


def _power_grid_resynchronisation_series(n_samples: int) -> FloatArray:
    islanding = _power_grid_islanding_series(n_samples)
    t = _time_axis(n_samples) / n_samples
    restored = (
        np.array([0.22, 0.24, 0.26, 0.28, 0.3, 0.32], dtype=np.float64)[:, None]
        * t[None, :]
    )
    restored = (
        restored + np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)[:, None]
    ) % 1.0
    switch = n_samples // 2
    return np.where(
        np.arange(n_samples)[None, :] < switch,
        islanding,
        (restored * _TWO_PI),
    )


def _traffic_spillback_fragmentation_series(n_samples: int) -> FloatArray:
    t = _time_axis(n_samples) / n_samples
    front_platoon = np.array([0.09, 0.13], dtype=np.float64)[:, None]
    rear_platoon = np.array([0.24, 0.28, 0.35, 0.39], dtype=np.float64)[:, None]
    phase_front = (
        front_platoon * t[None, :] + np.array([0.0, 0.15], dtype=np.float64)[:, None]
    ) % 1.0
    phase_rear = (
        rear_platoon * t[None, :]
        + np.array([0.6, 0.75, 0.9, 1.05], dtype=np.float64)[:, None]
    ) % 1.0
    return (np.vstack((phase_front, phase_rear)) * _TWO_PI) % _TWO_PI


def _traffic_platoon_recovery_series(n_samples: int) -> FloatArray:
    spillback = _traffic_spillback_fragmentation_series(n_samples)
    t = _time_axis(n_samples) / n_samples
    recovered = (
        np.array([0.18, 0.2, 0.22, 0.24, 0.26, 0.28], dtype=np.float64)[:, None]
        * t[None, :]
    )
    recovered = (
        recovered
        + np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float64)[:, None]
    ) % 1.0
    switch = 2 * n_samples // 3
    return np.where(
        np.arange(n_samples)[None, :] < switch,
        spillback,
        (recovered * _TWO_PI) % _TWO_PI,
    )


def _build_islanding_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="power_grid_islanding",
        description=(
            "Power-grid frequency replay with two islands drifting on divergent "
            "trajectories and no re-synchronisation in view."
        ),
        phase_series=_power_grid_islanding_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "power_grid_islanding < power_grid_resynchronisation "
            "in engineering-information proxy integration"
        ),
    )


def _build_resynchronisation_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="power_grid_resynchronisation",
        description=(
            "Power-grid replay where frequency islands are re-synchronised mid-run, "
            "restoring phase coherence."
        ),
        phase_series=_power_grid_resynchronisation_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "power_grid_resynchronisation > power_grid_islanding "
            "in engineering-information proxy integration"
        ),
    )


def _build_traffic_spillback_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="traffic_spillback_fragmentation",
        description=(
            "Traffic-corridor replay with fragmented platoons: front and rear groups "
            "decohere and propagate different phases."
        ),
        phase_series=_traffic_spillback_fragmentation_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "traffic_spillback_fragmentation < traffic_platoon_recovery "
            "in engineering-information proxy integration"
        ),
    )


def _build_traffic_recovery_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="traffic_platoon_recovery",
        description=(
            "Traffic-corridor replay with platoon spillback recovery where groups "
            "re-align and recover phase coherence."
        ),
        phase_series=_traffic_platoon_recovery_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "traffic_platoon_recovery > traffic_spillback_fragmentation "
            "in engineering-information proxy integration"
        ),
    )
