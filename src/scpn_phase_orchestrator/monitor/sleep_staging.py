# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep stage classifier from Kuramoto order parameter

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
StageCodeArray: TypeAlias = NDArray[np.uint8]

try:
    from spo_kernel import (
        classify_sleep_stage_rust as _rust_classify,
    )
    from spo_kernel import (
        ultradian_phase_rust as _rust_ultradian,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["classify_sleep_stage", "ultradian_phase"]

# AASM sleep staging mapped to Kuramoto order parameter R.
# N3 (slow-wave): highly synchronised cortical oscillations (R > 0.7).
# N2 (spindle):   moderate synchrony with K-complex bursts (R ~ 0.4–0.7).
# N1 (drowsy):    partial desynchronisation (R ~ 0.3–0.4).
# REM:            low R (~0.2–0.35) plus functional desynchronisation flag
#                 (distinguishes REM from light wakefulness).
# Wake:           desynchronised cortex (R < 0.3, no functional_desync).
_STAGE_THRESHOLDS = {
    "N3": 0.70,
    "N2": 0.40,
    "N1": 0.30,
    "REM": 0.20,
}


_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def classify_sleep_stage(R: float, functional_desync: bool = False) -> str:
    """Classify sleep stage from Kuramoto order parameter *R*.

    Args:
        R: order parameter in [0, 1].
        functional_desync: True when EEG shows desynchronisation pattern
            characteristic of REM (low-voltage mixed-frequency),
            as opposed to wakeful desynchronisation.

    Returns:
        One of ``"N3"``, ``"N2"``, ``"N1"``, ``"REM"``, ``"Wake"``.
    """
    r_value = _validate_order_parameter(R)
    desync = _validate_functional_desync(functional_desync)
    if _HAS_RUST:
        code = _rust_classify(r_value, desync)
        return _STAGE_NAMES[code]
    if _STAGE_THRESHOLDS["N3"] <= r_value:
        return "N3"
    if _STAGE_THRESHOLDS["N2"] <= r_value:
        return "N2"
    if _STAGE_THRESHOLDS["N1"] <= r_value:
        if desync:
            return "REM"
        return "N1"
    # Below N1 threshold
    if desync and _STAGE_THRESHOLDS["REM"] <= r_value:
        return "REM"
    return "Wake"


# Ultradian NREM–REM cycle period (Rechtschaffen & Kales 1968).
_ULTRADIAN_PERIOD_S = 90.0 * 60.0  # 90 minutes in seconds


_STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def ultradian_phase(
    timestamps: FloatArray,
    stage_history: list[str],
) -> float:
    """Estimate position within the ~90-minute ultradian sleep cycle.

    Finds the most recent N3 epoch (cycle trough = deepest sleep) and
    returns the elapsed fraction of a 90-minute period since that point.

    Args:
        timestamps: monotonic epoch times in seconds, shape (n_epochs,).
        stage_history: sleep stage label per epoch, same length as timestamps.

    Returns:
        Phase in [0, 1) where 0 = cycle start (N3 onset),
        0.5 ≈ mid-cycle (REM), wrapping back toward 0.
        Returns 0.0 if no N3 epoch is found.
    """
    ts = _validate_timestamps(timestamps)
    stages = _validate_stage_history(stage_history, expected_n=int(ts.size))
    if ts.size == 0:
        return 0.0
    if _HAS_RUST:
        rust_ts: FloatArray = np.ascontiguousarray(ts, dtype=np.float64)
        codes: StageCodeArray = np.array(
            [_STAGE_CODES[s] for s in stages],
            dtype=np.uint8,
        )
        return float(_rust_ultradian(rust_ts, codes))
    n = int(ts.size)

    last_n3_idx = -1
    for i in range(n - 1, -1, -1):
        if stages[i] == "N3":
            last_n3_idx = i
            break

    if last_n3_idx < 0:
        return 0.0

    elapsed = float(ts[n - 1] - ts[last_n3_idx])
    return (elapsed % _ULTRADIAN_PERIOD_S) / _ULTRADIAN_PERIOD_S


def _validate_order_parameter(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("R must be a finite real value in [0, 1]")
    r_value = float(value)
    if not np.isfinite(r_value) or r_value < 0.0 or r_value > 1.0:
        raise ValueError("R must be a finite real value in [0, 1]")
    return r_value


def _validate_functional_desync(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("functional_desync must be a bool")
    return value


def _validate_timestamps(value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError("timestamps must not contain boolean values")
    try:
        timestamps = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be a finite 1-D array") from exc
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a finite 1-D array")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must contain only finite values")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamps must be monotonic non-decreasing")
    return timestamps


def _validate_stage_history(stage_history: list[str], *, expected_n: int) -> list[str]:
    if len(stage_history) != expected_n:
        raise ValueError(
            "stage_history must have the same length as timestamps, "
            f"got {len(stage_history)} and {expected_n}"
        )
    invalid = [stage for stage in stage_history if stage not in _STAGE_CODES]
    if invalid:
        raise ValueError(f"stage_history contains unknown sleep stage {invalid[0]!r}")
    return stage_history
