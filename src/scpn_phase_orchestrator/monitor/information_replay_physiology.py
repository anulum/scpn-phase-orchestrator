# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physiology replay monitor suite

"""Deterministic physiology replay benchmark records for the
engineering integration proxy.

These records are explicit empirical replay cases used as audit-level
indicators, not as theoretical IIT claims.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.information_integration import (
    integrated_information,
)

__all__ = [
    "build_physiology_integrated_information_replays",
]

FloatArray: TypeAlias = NDArray[np.float64]

_TWO_PI = 2.0 * np.pi


def build_physiology_integrated_information_replays(
    *,
    n_samples: int = 256,
    n_bins: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Build deterministic physiology replay audit records.

    Args:
        n_samples: Number of time samples in each trajectory.
            Must be at least 32.
        n_bins: Number of phase bins used by ``integrated_information``.
            Must be an integer > 1.

    Returns:
        Tuple of JSON-safe replay records (one per physiology case).

    Raises:
        ValueError: If inputs are invalid or the benchmark ordering cannot be
            established.
    """
    _validate_replay_parameters(n_samples=n_samples, n_bins=n_bins)

    records = (
        _build_cardiac_respiratory_lock_case(n_samples=n_samples, n_bins=n_bins),
        _build_cardiac_respiratory_recovery_case(n_samples=n_samples, n_bins=n_bins),
        _build_eeg_sleep_spindle_case(n_samples=n_samples, n_bins=n_bins),
        _build_eeg_sleep_baseline_case(n_samples=n_samples, n_bins=n_bins),
    )

    _validate_replay_records(records)

    return records


def _validate_replay_parameters(*, n_samples: int, n_bins: int) -> None:
    if not isinstance(n_samples, int) or n_samples < 32:
        raise ValueError("n_samples must be an integer >= 32")
    if not isinstance(n_bins, int) or n_bins < 2:
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
        if record["domain"] != "physiology":
            raise ValueError("all replay records must use domain=physiology")
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
        if not isinstance(record["minimum_partition"][0], list):
            raise ValueError("minimum_partition must be list-of-lists")

    by_name = {record["case_name"]: record for record in records}

    required = {
        "cardiac_respiratory_lock",
        "cardiac_respiratory_recovery",
        "eeg_sleep_spindle",
        "eeg_sleep_baseline",
    }
    if not required.issubset(by_name.keys()):
        missing = ", ".join(sorted(required - by_name.keys()))
        raise ValueError(f"missing replay cases: {missing}")

    if not (
        by_name["cardiac_respiratory_lock"]["phi"]
        > by_name["cardiac_respiratory_recovery"]["phi"]
    ):
        raise ValueError(
            "physiology replay should show coherent cardiac-respiratory "
            "lock above recovery"
        )
    if not by_name["eeg_sleep_spindle"]["phi"] > by_name["eeg_sleep_baseline"]["phi"]:
        raise ValueError(
            "physiology replay should show sleep-spindle phase coupling above baseline"
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
        "domain": "physiology",
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


def _time_norm(n_samples: int) -> NDArray[np.float64]:
    return np.arange(n_samples, dtype=np.float64) / float(max(n_samples, 1))


def _cardiac_respiratory_lock_series(n_samples: int) -> FloatArray:
    t = _time_norm(n_samples)
    respiratory = _TWO_PI * (2.0 + 0.3 * t)
    cardiac = _TWO_PI * (3.7 + 0.18 * t) + 0.05 * np.sin(6.0 * _TWO_PI * t)

    return np.vstack(
        [
            respiratory % _TWO_PI,
            (respiratory + 0.08) % _TWO_PI,
            (cardiac + 0.12) % _TWO_PI,
            (cardiac + 0.25 + 0.03 * np.sin(3.0 * _TWO_PI * t)) % _TWO_PI,
        ]
    ).astype(np.float64)


def _cardiac_respiratory_recovery_series(n_samples: int) -> FloatArray:
    rng = np.random.default_rng(2026)
    return rng.uniform(0.0, _TWO_PI, size=(4, n_samples)).astype(np.float64)


def _eeg_spindle_fragments_series(n_samples: int) -> FloatArray:
    t = _time_norm(n_samples)
    rng = np.random.default_rng(3030)
    freqs = np.array([0.63, 0.79, 1.01, 1.27], dtype=np.float64)
    phase_offsets = np.array([0.0, 0.8, 1.5, 2.3], dtype=np.float64)
    return (
        _TWO_PI * freqs[:, None] * (7.2 * t[None, :])
        + phase_offsets[:, None]
        + 0.22 * np.sin(2.0 * _TWO_PI * t[None, :] + phase_offsets[:, None])
        + 0.12 * rng.normal(size=(4, n_samples)).astype(np.float64)
    ) % _TWO_PI


def _eeg_sleep_spindle_series(n_samples: int) -> FloatArray:
    t = _time_norm(n_samples)
    fragmented = _eeg_spindle_fragments_series(n_samples)
    spindle_core = (_TWO_PI * 9.8 * t[None, :])
    spindle_offsets = np.array([0.0, 0.14, 0.28, 0.42], dtype=np.float64)
    coherent = (spindle_core + spindle_offsets[:, None]) % _TWO_PI

    first_start = int(0.10 * n_samples)
    first_end = int(0.45 * n_samples)
    second_start = int(0.55 * n_samples)
    second_end = int(0.95 * n_samples)
    spindle_mask = np.zeros(n_samples, dtype=bool)
    spindle_mask[first_start:first_end] = True
    spindle_mask[second_start:second_end] = True

    return np.where(spindle_mask[None, :], coherent, fragmented)


def _build_cardiac_respiratory_lock_case(
    *,
    n_samples: int,
    n_bins: int,
) -> dict[str, Any]:
    return _build_record(
        case_name="cardiac_respiratory_lock",
        description=(
            "Cardiac and respiratory-phase replay showing coupled physiological "
            "phase-locking over recovery and homeostatic coupling windows."
        ),
        phase_series=_cardiac_respiratory_lock_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "cardiac_respiratory_lock > cardiac_respiratory_recovery in "
            "engineering proxy integration."
        ),
    )


def _build_cardiac_respiratory_recovery_case(
    *,
    n_samples: int,
    n_bins: int,
) -> dict[str, Any]:
    return _build_record(
        case_name="cardiac_respiratory_recovery",
        description=(
            "Cardiac/respiratory replay with phase-coupling loss and fragmented "
            "recovery drift after stable lock."
        ),
        phase_series=_cardiac_respiratory_recovery_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "cardiac_respiratory_recovery < cardiac_respiratory_lock in "
            "engineering proxy integration."
        ),
    )


def _build_eeg_sleep_spindle_case(
    *,
    n_samples: int,
    n_bins: int,
) -> dict[str, Any]:
    return _build_record(
        case_name="eeg_sleep_spindle",
        description=(
            "Sleep spindle replay with deterministic coherent phase pulses across "
            "EEG-like channels and lower baseline drift around them."
        ),
        phase_series=_eeg_sleep_spindle_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "eeg_sleep_spindle > eeg_sleep_baseline in engineering proxy integration."
        ),
    )


def _build_eeg_sleep_baseline_case(
    *,
    n_samples: int,
    n_bins: int,
) -> dict[str, Any]:
    return _build_record(
        case_name="eeg_sleep_baseline",
        description=(
            "Sleep baseline replay with fragmented EEG-like channels and weak "
            "cross-channel phase coupling."
        ),
        phase_series=_eeg_spindle_fragments_series(n_samples),
        n_bins=n_bins,
        expected_relationship=(
            "eeg_sleep_baseline < eeg_sleep_spindle in engineering proxy integration."
        ),
    )
