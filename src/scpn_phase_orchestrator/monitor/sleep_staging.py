# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep stage classifier from Kuramoto order parameter

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["classify_sleep_stage", "ultradian_phase"]

# AASM sleep staging mapped to Kuramoto order parameter R.
# N3 (slow-wave): highly synchronised cortical oscillations (R > 0.7).
# N2 (spindle):   moderate synchrony with K-complex bursts (R ~ 0.4–0.7).
# N1 (drowsy):    partial desynchronisation (R ~ 0.3–0.4).
# REM:            low R (~0.2–0.35) plus functional desynchronisation flag
#                 (distinguishes REM from light wakefulness).
# Wake:           desynchronised cortex (R < 0.3, no functional_desync).
_STAGE_THRESHOLDS = {
    "N3":   0.70,
    "N2":   0.40,
    "N1":   0.30,
    "REM":  0.20,
}


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
    R = float(R)
    if R >= _STAGE_THRESHOLDS["N3"]:
        return "N3"
    if R >= _STAGE_THRESHOLDS["N2"]:
        return "N2"
    if R >= _STAGE_THRESHOLDS["N1"]:
        if functional_desync:
            return "REM"
        return "N1"
    # Below N1 threshold
    if functional_desync and R >= _STAGE_THRESHOLDS["REM"]:
        return "REM"
    return "Wake"


# Ultradian NREM–REM cycle period (Rechtschaffen & Kales 1968).
_ULTRADIAN_PERIOD_S = 90.0 * 60.0  # 90 minutes in seconds


def ultradian_phase(
    timestamps: NDArray,
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
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) == 0 or len(stage_history) == 0:
        return 0.0
    n = min(len(ts), len(stage_history))

    last_n3_idx = -1
    for i in range(n - 1, -1, -1):
        if stage_history[i] == "N3":
            last_n3_idx = i
            break

    if last_n3_idx < 0:
        return 0.0

    elapsed = float(ts[n - 1] - ts[last_n3_idx])
    return (elapsed % _ULTRADIAN_PERIOD_S) / _ULTRADIAN_PERIOD_S
