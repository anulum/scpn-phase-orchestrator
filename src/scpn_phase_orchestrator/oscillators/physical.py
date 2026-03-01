# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

TWO_PI = 2.0 * np.pi


class PhysicalExtractor(PhaseExtractor):
    """Extracts instantaneous phase from continuous waveforms via Hilbert transform."""

    def __init__(self, node_id: str = "phys_0"):
        self._node_id = node_id

    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        analytic = hilbert(signal)
        inst_phase = np.angle(analytic) % TWO_PI
        inst_amp = np.abs(analytic)
        inst_freq = np.gradient(np.unwrap(np.angle(analytic))) * sample_rate / TWO_PI

        theta = float(inst_phase[-1])
        omega = float(np.median(inst_freq)) * TWO_PI  # rad/s
        amplitude = float(np.mean(inst_amp))
        quality = self._snr_estimate(signal, analytic)

        return [
            PhaseState(
                theta=theta,
                omega=omega,
                amplitude=amplitude,
                quality=quality,
                channel="P",
                node_id=self._node_id,
            )
        ]

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    @staticmethod
    def _snr_estimate(signal: NDArray, analytic: NDArray) -> float:
        envelope = np.abs(analytic)
        noise = signal - np.real(analytic)
        sig_power = np.mean(envelope**2)
        noise_power = np.mean(noise**2)
        if noise_power < 1e-15:
            return 1.0
        snr_linear = sig_power / noise_power
        # Map SNR to [0, 1] via sigmoid-like saturation
        return float(np.clip(snr_linear / (1.0 + snr_linear), 0.0, 1.0))
