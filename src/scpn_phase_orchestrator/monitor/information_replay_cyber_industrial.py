# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cyber-industrial replay monitor suite

"""Deterministic cyber-industrial replay benchmark records for the
engineering integration proxy.

These are proxy-only monitors for empirical replay corpora and are not
theoretical IIT claims.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.information_integration import (
    integrated_information,
)

__all__ = [
    "build_cyber_industrial_integrated_information_replays",
]

FloatArray: TypeAlias = NDArray[np.float64]

_TWO_PI = 2.0 * np.pi


def build_cyber_industrial_integrated_information_replays(
    *,
    n_samples: int = 256,
    n_bins: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic cyber-industrial replay audit records.

    Args:
        n_samples: Number of time samples in each phase trajectory.
            Must be at least 32.
        n_bins: Bin count passed to ``integrated_information``. Must be an int > 1.

    Returns:
        Tuple of JSON-safe replay records (one per case).

    Raises:
        ValueError: If ``n_samples`` or ``n_bins`` are invalid.
    """
    _validate_replay_parameters(n_samples=n_samples, n_bins=n_bins)

    records = (
        _build_cyber_disruption_case(n_samples=n_samples, n_bins=n_bins),
        _build_cyber_recontainment_case(n_samples=n_samples, n_bins=n_bins),
        _build_spc_fragmentation_case(n_samples=n_samples, n_bins=n_bins),
        _build_spc_recovery_case(n_samples=n_samples, n_bins=n_bins),
    )

    _validate_replay_records(records)

    return records


def _validate_replay_parameters(*, n_samples: int, n_bins: int) -> None:
    if not isinstance(n_samples, int) or n_samples < 32:
        raise ValueError("n_samples must be an integer >= 32")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")


def _validate_replay_records(
    records: tuple[dict[str, Any], ...],
) -> None:
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

        if record["domain"] != "cyber_industrial":
            raise ValueError("all replay records must use domain=cyber_industrial")
        if record["claim_boundary"] != "engineering_proxy_not_theoretical_iit":
            raise ValueError("invalid claim boundary value")
        if record["non_actuating"] is not True:
            raise ValueError("replay records must be non-actuating")
        if not isinstance(record["n_oscillators"], int) or record["n_oscillators"] < 2:
            raise ValueError("n_oscillators must be at least two")
        if (
            not isinstance(record["minimum_partition"], list)
            or len(record["minimum_partition"]) != 2
            or any(not isinstance(part, list) for part in record["minimum_partition"])
        ):
            raise ValueError("minimum_partition must be a list pair")

    case_by_name = {record["case_name"]: record for record in records}
    for key in (
        "cyber_disruption",
        "cyber_recontainment",
        "spc_fragmentation",
        "spc_recovery",
    ):
        if key not in case_by_name:
            raise ValueError(f"missing replay case: {key}")

    if not (
        case_by_name["cyber_recontainment"]["phi"]
        > case_by_name["cyber_disruption"]["phi"]
    ):
        raise ValueError(
            "cyber recontainment integration must exceed disruption integration"
        )
    if not (
        case_by_name["spc_recovery"]["phi"]
        > case_by_name["spc_fragmentation"]["phi"]
    ):
        raise ValueError(
            "SPC recovery integration must exceed fragmentation integration"
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
    left_list = list(left)
    right_list = list(right)

    return {
        "domain": "cyber_industrial",
        "case_name": case_name,
        "description": description,
        "n_oscillators": int(phase_series.shape[0]),
        "n_samples": int(phase_series.shape[1]),
        "n_bins": int(result.n_bins),
        "phi": float(result.phi),
        "normalised_phi": float(result.normalised_phi),
        "total_integration": float(result.total_integration),
        "minimum_partition": [left_list, right_list],
        "expected_relationship": expected_relationship,
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
        "non_actuating": True,
    }


def _time_axis(n_samples: int) -> NDArray[np.float64]:
    return np.arange(n_samples, dtype=np.float64)


def _lateral_movement_disruption_series(n_samples: int) -> FloatArray:
    rng = np.random.default_rng(4201)
    return rng.uniform(0.0, _TWO_PI, size=(6, n_samples)).astype(np.float64)


def _lateral_movement_recontainment_series(n_samples: int) -> FloatArray:
    disruption = _lateral_movement_disruption_series(n_samples)
    t = _time_axis(n_samples)
    locked = ((0.17 * t[None, :] / n_samples) * _TWO_PI) + np.array(
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float64
    )[:, None]
    locked = locked % _TWO_PI
    switch = n_samples // 2
    return np.where(np.arange(n_samples) < switch, disruption[:, :], locked) % _TWO_PI


def _spc_fragmentation_series(n_samples: int) -> FloatArray:
    rng = np.random.default_rng(4217)
    return rng.uniform(0.0, _TWO_PI, size=(6, n_samples)).astype(np.float64)


def _spc_recovery_series(n_samples: int) -> FloatArray:
    fragmentation = _spc_fragmentation_series(n_samples)
    t = _time_axis(n_samples)
    recovery_phase = (0.14 * t / n_samples) * _TWO_PI
    line_offsets = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)[:, None]
    line_locked = (recovery_phase[None, :] + line_offsets) % _TWO_PI
    switch = n_samples // 3
    return np.where(
        np.arange(n_samples) < switch,
        fragmentation[:, :],
        np.broadcast_to(line_locked, fragmentation.shape),
    ) % _TWO_PI


def _build_cyber_disruption_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="cyber_disruption",
        description=(
            "Network-security lateral movement replay with fragmented control channels "
            "and no containment re-establishment."
        ),
        phase_series=_lateral_movement_disruption_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "cyber_disruption < cyber_recontainment "
            "in engineering proxy integration"
        ),
    )


def _build_cyber_recontainment_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="cyber_recontainment",
        description=(
            "Network-security lateral movement replay where containment is restored "
            "mid-run and cross-subsystem phase alignment increases."
        ),
        phase_series=_lateral_movement_recontainment_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "cyber_recontainment > cyber_disruption in engineering proxy integration"
        ),
    )


def _build_spc_fragmentation_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="spc_fragmentation",
        description=(
            "Manufacturing SPC replay with SPC stations split between independently "
            "oscillating production phases and weak station coupling."
        ),
        phase_series=_spc_fragmentation_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "spc_fragmentation < spc_recovery in engineering proxy integration"
        ),
    )


def _build_spc_recovery_case(*, n_samples: int, n_bins: int) -> dict[str, Any]:
    return _build_record(
        case_name="spc_recovery",
        description=(
            "Manufacturing SPC replay with line coupling restored after a "
            "fragmentation episode, yielding stronger integrated-information proxy."
        ),
        phase_series=_spc_recovery_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "spc_recovery > spc_fragmentation in engineering proxy integration"
        ),
    )
