# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep stage classifier from Kuramoto order parameter

"""Sleep staging helpers derived from validated phase-synchrony time series.

The staging path maps Kuramoto R summaries and ultradian phase estimates into
an AASM-like heuristic stage timeline for diagnostics and simulation review.
R values, timestamps, and stage labels are validated before use, and the Rust
accelerator mirrors the deterministic Python fallback rather than changing
classification semantics.

Validation is fail-closed. ``R`` must be a real number in ``[0, 1]`` and the
desynchronisation flag a Python or NumPy boolean. Timestamps must be a
one-dimensional array of finite real seconds that never decreases. Text,
boolean, complex, ``datetime64`` and ``timedelta64`` samples are rejected
rather than coerced: a text sample would be parsed as a number and a
``timedelta64`` sample would be read in its own unit, not in seconds. Stage
labels must be one of ``"Wake"``, ``"N1"``, ``"N2"``, ``"N3"``, ``"REM"``.
"""

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
# REM:            R in [0.2, 0.4) plus the functional desynchronisation flag
#                 (distinguishes REM from light wakefulness and from N1).
# Wake:           desynchronised cortex (R < 0.3 without the flag, or
#                 R < 0.2 with it).
_STAGE_THRESHOLDS = {
    "N3": 0.70,
    "N2": 0.40,
    "N1": 0.30,
    "REM": 0.20,
}


_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def classify_sleep_stage(R: float, functional_desync: bool = False) -> str:
    """Classify sleep stage from Kuramoto order parameter *R*.

    Parameters
    ----------
    R : float
        order parameter in [0, 1].
    functional_desync : bool
        True when EEG shows desynchronisation pattern characteristic of REM (low-voltage
        mixed-frequency), as opposed to wakeful desynchronisation. A NumPy boolean
        is accepted and normalised to ``bool``.

    Returns
    -------
    str
        One of ``"N3"``, ``"N2"``, ``"N1"``, ``"REM"``, ``"Wake"``.

    Raises
    ------
    TypeError
        If ``R`` is not a real number or ``functional_desync`` is not a boolean.
    ValueError
        If ``R`` is not finite or lies outside ``[0, 1]``.
    """
    r_value = _validate_order_parameter(R)
    desync = _validate_functional_desync(functional_desync)
    if _HAS_RUST:
        code = _rust_classify(r_value, desync)
        return _validate_stage_code(code)
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

    Parameters
    ----------
    timestamps : FloatArray
        non-decreasing epoch times in seconds, shape (n_epochs,).
    stage_history : list[str]
        sleep stage label per epoch, same length as timestamps.

    Returns
    -------
    float
        Phase in [0, 1) where 0 = cycle start (N3 onset), 0.5 ≈ mid-cycle (REM),
        wrapping back toward 0. Returns 0.0 if no N3 epoch is found.

    Raises
    ------
    ValueError
        If the timestamps are not finite, real, one-dimensional seconds in
        non-decreasing order, or the stage history does not match them.
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
        return _validate_ultradian_phase(_rust_ultradian(rust_ts, codes))
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
    """Return the order parameter as a float in ``[0, 1]``, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("R must be a finite real value in [0, 1]")
    r_value = float(value)
    if not np.isfinite(r_value) or r_value < 0.0 or r_value > 1.0:
        raise ValueError("R must be a finite real value in [0, 1]")
    return r_value


def _validate_functional_desync(value: object) -> bool:
    """Return the functional-desynchronisation flag as ``bool``, else raise."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("functional_desync must be a bool")
    return bool(value)


def _validate_timestamps(value: object) -> FloatArray:
    """Return finite, non-decreasing timestamps in seconds, else raise."""
    try:
        raw = np.asarray(value)
    except ValueError as exc:
        raise ValueError("timestamps must be a finite 1-D array") from exc
    if raw.dtype == np.bool_:
        raise ValueError("timestamps must not contain boolean values")
    if raw.dtype.kind in "mM":
        raise ValueError(
            "timestamps must be plain seconds; convert datetime64 or "
            "timedelta64 values explicitly"
        )
    if raw.dtype.kind == "O":
        _require_real_object_samples(raw)
    elif raw.dtype.kind not in "fiu":
        raise ValueError("timestamps must contain real-valued samples")
    timestamps = raw.astype(np.float64, copy=True)
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a finite 1-D array")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must contain only finite values")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamps must be monotonic non-decreasing")
    return timestamps


def _require_real_object_samples(raw: NDArray[np.object_]) -> None:
    """Reject object-array timestamps that hold booleans or non-numeric items."""
    for item in raw.flat:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("timestamps must not contain boolean values")
        if not isinstance(item, Real):
            raise ValueError("timestamps must contain real-valued samples")


def _validate_stage_history(stage_history: list[str], *, expected_n: int) -> list[str]:
    """Return the validated sleep-stage history, else raise ``ValueError``."""
    if len(stage_history) != expected_n:
        raise ValueError(
            "stage_history must have the same length as timestamps, "
            f"got {len(stage_history)} and {expected_n}"
        )
    invalid = [
        stage
        for stage in stage_history
        if not isinstance(stage, str) or stage not in _STAGE_CODES
    ]
    if invalid:
        raise ValueError(f"stage_history contains unknown sleep stage {invalid[0]!r}")
    return [str(stage) for stage in stage_history]


def _validate_stage_code(value: object) -> str:
    """Return ``value`` as a supported sleep-stage code, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Rust sleep stage code must be an integer stage code")
    numeric = float(value)
    code = int(numeric)
    if numeric != float(code) or code not in _STAGE_NAMES:
        raise ValueError(
            f"Rust sleep stage code must be in {_STAGE_NAMES}, got {value!r}"
        )
    return _STAGE_NAMES[code]


def _validate_ultradian_phase(value: object) -> float:
    """Return the validated ultradian-cycle phase, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Rust ultradian phase must be a finite real value in [0, 1)")
    phase = float(value)
    if not np.isfinite(phase) or phase < 0.0 or phase >= 1.0:
        raise ValueError("Rust ultradian phase must be a finite real value in [0, 1)")
    return phase
