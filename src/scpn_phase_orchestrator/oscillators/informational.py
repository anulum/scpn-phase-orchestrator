# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational oscillator

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["InformationalExtractor"]


class InformationalExtractor(PhaseExtractor):
    """Extracts phase from event timestamps (spike trains, discrete events).

    Converts inter-event intervals to instantaneous frequency,
    then derives phase via cumulative integral of frequency.
    """

    def __init__(self, node_id: str = "info_0"):
        self._node_id = node_id

    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        """Args:
        signal: 1-D array of event timestamps in seconds (sorted ascending).
        sample_rate: not used for timestamps but kept for interface consistency.
        """
        if len(signal) < 2:
            return [
                PhaseState(
                    theta=0.0,
                    omega=0.0,
                    amplitude=0.0,
                    quality=0.0,
                    channel="I",
                    node_id=self._node_id,
                )
            ]

        intervals = np.diff(signal)
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return [
                PhaseState(
                    theta=0.0,
                    omega=0.0,
                    amplitude=0.0,
                    quality=0.0,
                    channel="I",
                    node_id=self._node_id,
                )
            ]

        inst_freq = 1.0 / intervals  # Hz
        omega_median = float(np.median(inst_freq)) * TWO_PI  # rad/s

        total_time = float(signal[-1] - signal[0])
        omega_median_hz = float(np.median(inst_freq))
        cumulative_phase = TWO_PI * omega_median_hz * total_time
        theta = float(cumulative_phase % TWO_PI)

        # Quality: inverse coefficient of variation of intervals (regularity)
        cv = (
            float(np.std(intervals) / np.mean(intervals))
            if np.mean(intervals) > 0
            else 1.0
        )
        quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

        return [
            PhaseState(
                theta=theta,
                omega=omega_median,
                amplitude=float(np.mean(inst_freq)),
                quality=quality,
                channel="I",
                node_id=self._node_id,
            )
        ]

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Mean interval-regularity quality across phase states."""
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))
