# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import importlib.util

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

_HAS_RUST = importlib.util.find_spec("spo_kernel") is not None

TWO_PI = 2.0 * np.pi


class PhysicalExtractor(PhaseExtractor):
    """Extracts instantaneous phase from continuous waveforms via Hilbert transform."""

    def __init__(self, node_id: str = "phys_0"):
        self._node_id = node_id

    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        analytic = hilbert(signal)

        if _HAS_RUST:
            from spo_kernel import physical_extract

            theta, omega, amplitude, quality = physical_extract(
                np.real(analytic).tolist(), np.imag(analytic).tolist(), sample_rate
            )
        else:
            inst_phase = np.angle(analytic) % TWO_PI
            inst_amp = np.abs(analytic)
            unwrapped = np.unwrap(np.angle(analytic))
            inst_freq = np.gradient(unwrapped) * sample_rate / TWO_PI

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
        mean_env = np.mean(envelope)
        if mean_env < 1e-15:
            return 0.0
        cv = np.std(envelope) / mean_env
        return float(np.clip(1.0 - cv, 0.0, 1.0))
