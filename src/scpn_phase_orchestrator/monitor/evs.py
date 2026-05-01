# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entrainment Verification Score (EVS)

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

try:
    from spo_kernel import (
        frequency_specificity_rust as _rust_freq_spec,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["EVSMonitor", "EVSResult"]


@dataclass(frozen=True, slots=True)
class EVSResult:
    """Entrainment verification outcome."""

    itpc_value: float
    persistence_score: float
    specificity_ratio: float
    is_entrained: bool


class EVSMonitor:
    """Combines ITPC, persistence, and frequency specificity into
    a single entrainment verification score.

    Three criteria must all pass for ``is_entrained=True``:

    1. Mean ITPC across all time points >= ``itpc_threshold``
    2. ITPC during/after stimulus pause >= ``persistence_threshold``
    3. ITPC at the target frequency / ITPC at a control frequency
       >= ``specificity_threshold``

    The specificity test distinguishes frequency-specific entrainment
    from broadband phase-locking artefacts.
    """

    def __init__(
        self,
        itpc_threshold: float = 0.6,
        persistence_threshold: float = 0.4,
        specificity_threshold: float = 1.5,
    ) -> None:
        self.itpc_threshold = itpc_threshold
        self.persistence_threshold = persistence_threshold
        self.specificity_threshold = specificity_threshold

    def evaluate(
        self,
        phases_trials: FloatArray,
        pause_indices: list[int] | IntArray,
        target_freq: float,
        control_freq: float,
    ) -> EVSResult:
        """Run the full EVS battery.

        Args:
            phases_trials: shape (n_trials, n_timepoints), phases in
                radians at the *target* frequency.
            pause_indices: time-point indices within/after a stimulus
                pause window.
            target_freq: stimulus frequency (Hz).
            control_freq: non-stimulus control frequency (Hz).

        Returns:
            EVSResult with all three sub-scores and the overall verdict.
        """
        itpc_vals = compute_itpc(phases_trials)
        itpc_mean = float(np.mean(itpc_vals)) if itpc_vals.size > 0 else 0.0

        persistence = itpc_persistence(phases_trials, pause_indices)

        specificity = self._frequency_specificity(
            phases_trials,
            target_freq,
            control_freq,
        )

        entrained = (
            itpc_mean >= self.itpc_threshold
            and persistence >= self.persistence_threshold
            and specificity >= self.specificity_threshold
        )

        return EVSResult(
            itpc_value=itpc_mean,
            persistence_score=persistence,
            specificity_ratio=specificity,
            is_entrained=entrained,
        )

    @staticmethod
    def _frequency_specificity(
        phases_trials: FloatArray,
        target_freq: float,
        control_freq: float,
    ) -> float:
        """Ratio of mean ITPC at target vs control frequency.

        Uses phase scaling: phases at the control frequency are obtained
        by rescaling the target-frequency phases by the frequency ratio.
        This models the assumption that the raw signal was band-pass
        filtered at each frequency before phase extraction.
        """
        if target_freq <= 0 or control_freq <= 0:
            return 0.0

        if _HAS_RUST:
            p = np.ascontiguousarray(phases_trials, dtype=np.float64)
            n_trials, n_tp = p.shape
            return float(
                _rust_freq_spec(p.ravel(), n_trials, n_tp, target_freq, control_freq)
            )

        itpc_target = compute_itpc(phases_trials)
        target_mean = float(np.mean(itpc_target)) if itpc_target.size > 0 else 0.0

        # Phase at control frequency: rescale by freq ratio
        ratio = control_freq / target_freq
        control_phases = np.asarray(phases_trials, dtype=np.float64) * ratio
        itpc_control = compute_itpc(control_phases)
        control_mean = float(np.mean(itpc_control)) if itpc_control.size > 0 else 0.0

        if control_mean < 1e-12:
            return float("inf") if target_mean > 0 else 0.0

        return target_mean / control_mean
