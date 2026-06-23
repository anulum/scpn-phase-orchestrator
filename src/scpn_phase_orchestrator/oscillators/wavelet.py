# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Wavelet-ridge oscillator

"""Physical-channel phase extraction via a complex Morlet wavelet ridge.

`WaveletExtractor` computes a continuous wavelet transform with a bank of complex
Morlet wavelets, selects the dominant scale (the energy ridge over a
cone-of-influence-safe interior region), and reads the analytic phase along that
ridge. Unlike the broadband Hilbert transform, the wavelet ridge is band-adaptive
and therefore robust to broadband noise and slow drift around a dominant
oscillation. The terminal phase is extrapolated from a COI-safe interior sample
using the ridge frequency, avoiding the corrupted signal edge.

The complex Morlet is ``psi(t) = pi^(-1/4) * exp(i*w0*t/s) * exp(-(t/s)^2/2) /
sqrt(s)`` with ``w0 = 6`` (the standard admissibility-approximating choice). The
peak frequency of scale ``s`` is ``f = w0 * fs / (2*pi*s)``. A pure-Python/NumPy
path is used because SciPy 1.15 removed ``cwt``/``morlet2`` and PyWavelets is not
a required dependency; the extractor preserves the `PhaseState` contract.
"""

from __future__ import annotations

from math import ceil, isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["WaveletExtractor"]

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

_MORLET_W0 = 6.0
_N_SCALES = 48


def _validate_node_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node_id must be a non-empty string")
    return value


def _validate_signal(value: object) -> FloatArray:
    signal = np.asarray(value)
    dtype = signal.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError("signal must be finite")
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError(
            f"signal must be 1-D with >= 2 samples, got shape {signal.shape}"
        )
    parsed = signal.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError("signal must be finite")
    return parsed


def _validate_sample_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("sample_rate must be finite and positive")
    sample_rate = float(value)
    if not isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive")
    return sample_rate


class WaveletExtractor(PhaseExtractor):
    """Extracts instantaneous phase from a waveform via a complex Morlet ridge."""

    def __init__(self, node_id: str = "wav_0"):
        self._node_id = _validate_node_id(node_id)

    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]:
        """Extract phase from a 1-D waveform via the dominant Morlet ridge.

        Parameters
        ----------
        signal : FloatArray
            Input signal, shape ``(T,)``.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        list[PhaseState]
            A single `PhaseState` on channel ``"P"`` carrying the terminal phase,
            the ridge angular frequency, amplitude, and ridge-regularity quality.
        """
        signal = _validate_signal(signal)
        sample_rate = _validate_sample_rate(sample_rate)

        centred = signal - float(np.mean(signal))
        rms = float(np.sqrt(np.mean(np.square(centred))))
        amplitude = float(np.sqrt(2.0) * rms)
        n = centred.size

        freqs = self._frequency_grid(n, sample_rate)
        if freqs.size == 0 or rms < 1e-15:
            theta = 0.0 if centred[-1] >= 0.0 else float(np.pi)
            return [self._state(theta % TWO_PI, 0.0, amplitude, 0.0)]

        scales = _MORLET_W0 * sample_rate / (TWO_PI * freqs)
        ridge_idx, coeffs, ridge_scale = self._ridge(centred, scales)
        f_ridge = float(freqs[ridge_idx])
        omega = float(TWO_PI * f_ridge)

        theta = self._terminal_phase(coeffs, ridge_scale, omega, sample_rate, n)
        quality = self._ridge_quality(coeffs, ridge_scale)

        return [self._state(theta, omega, amplitude, quality)]

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Mean extraction quality across phase states.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states.

        Returns
        -------
        float
            Mean extraction quality across phase states.
        """
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    def _state(
        self, theta: float, omega: float, amplitude: float, quality: float
    ) -> PhaseState:
        return PhaseState(
            theta=float(theta % TWO_PI),
            omega=float(omega),
            amplitude=float(amplitude),
            quality=float(quality),
            channel="P",
            node_id=self._node_id,
        )

    @staticmethod
    def _frequency_grid(n: int, sample_rate: float) -> FloatArray:
        """Log-spaced frequencies that fit at least ~3 cycles in the window."""
        f_max = 0.4 * sample_rate
        f_min = 3.0 * sample_rate / n
        if not (f_min < f_max):
            return np.empty(0)
        return np.geomspace(f_min, f_max, _N_SCALES)

    @staticmethod
    def _morlet(scale: float) -> ComplexArray:
        half = int(ceil(4.0 * scale))
        t = np.arange(-half, half + 1, dtype=np.float64) / scale
        envelope = np.exp(-0.5 * t * t)
        carrier = np.exp(1j * _MORLET_W0 * t)
        psi = (np.pi**-0.25) * carrier * envelope / np.sqrt(scale)
        return np.asarray(psi, dtype=np.complex128)

    def _cwt_row(self, centred: FloatArray, scale: float) -> ComplexArray:
        psi = self._morlet(scale)
        row = np.convolve(centred, np.conjugate(psi), mode="same")
        return np.asarray(row, dtype=np.complex128)

    def _ridge(
        self, centred: FloatArray, scales: FloatArray
    ) -> tuple[int, ComplexArray, float]:
        """Return (ridge frequency index, ridge CWT row, ridge scale).

        The ridge is the scale of maximum energy over a cone-of-influence-safe
        interior region (edges within one scale are excluded per row).
        """
        n = centred.size
        best_idx = 0
        best_energy = -1.0
        best_row: ComplexArray = self._cwt_row(centred, float(scales[0]))
        best_scale = float(scales[0])
        for idx, scale_val in enumerate(scales):
            scale = float(scale_val)
            row = self._cwt_row(centred, scale)
            margin = min(int(ceil(scale)), n // 2)
            interior = row[margin : n - margin] if n - margin > margin else row
            energy = float(np.sum(np.abs(interior) ** 2))
            if energy > best_energy:
                best_energy = energy
                best_idx = idx
                best_row = row
                best_scale = scale
        return best_idx, best_row, best_scale

    @staticmethod
    def _terminal_phase(
        coeffs: ComplexArray,
        ridge_scale: float,
        omega: float,
        sample_rate: float,
        n: int,
    ) -> float:
        """Phase at the final sample, extrapolated from a COI-safe interior point."""
        # margin <= n - 1, so safe_idx is always a valid non-negative index.
        margin = min(int(ceil(ridge_scale)), n - 1)
        safe_idx = n - 1 - margin
        base = float(np.angle(coeffs[safe_idx]))
        elapsed_seconds = (n - 1 - safe_idx) / sample_rate
        return (base + omega * elapsed_seconds) % TWO_PI

    @staticmethod
    def _ridge_quality(coeffs: ComplexArray, ridge_scale: float) -> float:
        """Quality from the amplitude regularity of the ridge envelope (interior)."""
        n = coeffs.size
        margin = min(int(ceil(ridge_scale)), n // 2)
        interior = coeffs[margin : n - margin] if n - margin > margin else coeffs
        envelope = np.abs(interior)
        mean_env = float(np.mean(envelope))
        if mean_env < 1e-15:
            return 0.0
        cv = float(np.std(envelope)) / mean_env
        return float(np.clip(1.0 - cv, 0.0, 1.0))
