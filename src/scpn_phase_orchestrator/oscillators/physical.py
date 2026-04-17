# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical oscillator

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["PhysicalExtractor"]


class PhysicalExtractor(PhaseExtractor):
    """Extracts instantaneous phase from continuous waveforms via Hilbert transform."""

    def __init__(self, node_id: str = "phys_0"):
        self._node_id = node_id

    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        """Extract instantaneous phase from a 1-D waveform via Hilbert transform."""
        if signal.ndim != 1 or signal.size < 2:
            raise ValueError(
                f"signal must be 1-D with >= 2 samples, got shape {signal.shape}"
            )
        from scipy.signal import hilbert  # noqa: PLC0415

        analytic = hilbert(signal)

        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import physical_extract

            theta, omega, amplitude, quality = physical_extract(
                np.ascontiguousarray(np.real(analytic)),
                np.ascontiguousarray(np.imag(analytic)),
                sample_rate,
            )
        else:
            inst_phase = np.angle(analytic) % TWO_PI
            inst_amp = np.abs(analytic)
            unwrapped = np.unwrap(np.angle(analytic))
            inst_freq = np.gradient(unwrapped) * sample_rate / TWO_PI

            theta = float(inst_phase[-1])
            omega = float(np.median(inst_freq)) * TWO_PI  # rad/s
            amplitude = float(np.mean(inst_amp))
            quality = self._envelope_quality(signal, analytic)

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
        """Mean extraction quality across phase states."""
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    @staticmethod
    def _envelope_quality(signal: NDArray, analytic: NDArray) -> float:
        envelope = np.abs(analytic)
        mean_env = np.mean(envelope)
        if mean_env < 1e-15:
            return 0.0
        cv = np.std(envelope) / mean_env
        return float(np.clip(1.0 - cv, 0.0, 1.0))
