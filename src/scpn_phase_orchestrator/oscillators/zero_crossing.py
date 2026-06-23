# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Zero-crossing oscillator

"""Physical-channel phase extraction from zero crossings.

`ZeroCrossingExtractor` recovers instantaneous phase from a real waveform by
locating its zero crossings (with sub-sample linear interpolation), treating each
crossing as a half-cycle (``pi`` of phase advance), and anchoring absolute phase
to the crossing direction (rising crossing ≡ phase 0, falling crossing ≡ phase
``pi``, matching the sine convention). It is robust to a constant offset (the
mean is removed) and reports an angular frequency from the mean half-period and a
quality from the regularity of the half-period intervals. The extractor produces
the same `PhaseState` contract as the Hilbert-based `PhysicalExtractor` but is
preferable for sharply non-sinusoidal periodic signals where a single dominant
analytic phase is ill-defined.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["ZeroCrossingExtractor"]

FloatArray: TypeAlias = NDArray[np.float64]


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


class ZeroCrossingExtractor(PhaseExtractor):
    """Extracts instantaneous phase from a waveform via interpolated zero crossings."""

    def __init__(self, node_id: str = "zc_0", *, hysteresis: float = 0.5):
        self._node_id = _validate_node_id(node_id)
        if not isfinite(hysteresis) or not (0.0 <= hysteresis < 1.0):
            raise ValueError("hysteresis must be a fraction in [0, 1)")
        self._hysteresis = float(hysteresis)

    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]:
        """Extract phase from a 1-D waveform via interpolated zero crossings.

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
            angular frequency, amplitude, and crossing-regularity quality.
        """
        signal = _validate_signal(signal)
        sample_rate = _validate_sample_rate(sample_rate)

        centred = signal - float(np.mean(signal))
        rms = float(np.sqrt(np.mean(np.square(centred))))
        amplitude = float(np.sqrt(2.0) * rms)

        crossings, rising = self._confirmed_crossings(centred, self._hysteresis * rms)

        if crossings.size < 2 or rms < 1e-15:
            # Too few crossings (near-DC or sub-half-cycle window): no reliable
            # phase. Report a terminal phase consistent with the last sample sign.
            theta = 0.0 if centred[-1] >= 0.0 else float(np.pi)
            return [
                PhaseState(
                    theta=theta % TWO_PI,
                    omega=0.0,
                    amplitude=amplitude,
                    quality=0.0,
                    channel="P",
                    node_id=self._node_id,
                )
            ]

        half_periods = np.diff(crossings)  # samples per half-cycle
        mean_half = float(np.mean(half_periods))
        half_seconds = mean_half / sample_rate
        omega = float(np.pi / half_seconds)  # pi advance per half-period

        last_cross = float(crossings[-1])
        anchor = 0.0 if bool(rising[-1]) else float(np.pi)
        elapsed_seconds = (len(centred) - 1 - last_cross) / sample_rate
        theta = (anchor + omega * elapsed_seconds) % TWO_PI

        quality = self._interval_quality(half_periods)

        return [
            PhaseState(
                theta=float(theta),
                omega=omega,
                amplitude=amplitude,
                quality=quality,
                channel="P",
                node_id=self._node_id,
            )
        ]

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

    @staticmethod
    def _confirmed_crossings(
        centred: FloatArray, deadband: float
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Hysteresis-confirmed zero crossings with sub-sample interpolation.

        A Schmitt-trigger deadband of ``deadband`` about zero suppresses spurious
        noise crossings: the confirmed polarity flips only after the signal rises
        above ``+deadband`` or falls below ``-deadband``. Each confirmed flip is
        reported at the interpolated true zero crossing that precedes it. The
        constant band-traversal lag cancels in the half-period differences used
        for frequency.

        Returns the sub-sample crossing positions and per-crossing rising flags.
        """
        n = centred.size
        states = np.zeros(n)
        state = 0.0
        for i in range(n):
            if centred[i] > deadband:
                state = 1.0
            elif centred[i] < -deadband:
                state = -1.0
            states[i] = state
        # When no sample exceeds the deadband, ``argmax`` yields 0 and no
        # transitions are found, so the ``trans`` guard below returns empty.
        first = int(np.argmax(states != 0.0))
        states[:first] = states[first]
        trans = np.nonzero(states[:-1] != states[1:])[0]
        if trans.size == 0:
            return np.empty(0), np.empty(0, dtype=bool)

        positions: list[float] = []
        rising: list[bool] = []
        search_start = 0
        for k in trans:
            seg_sign = np.sign(centred[search_start : int(k) + 2])
            nonzero = seg_sign != 0.0
            if np.any(nonzero):
                carry = seg_sign[nonzero][0]
                for j in range(seg_sign.size):
                    if seg_sign[j] == 0.0:
                        seg_sign[j] = carry
                    else:
                        carry = seg_sign[j]
            changes = np.nonzero(seg_sign[:-1] != seg_sign[1:])[0]
            if changes.size:
                j = search_start + int(changes[-1])
                x0 = centred[j]
                x1 = centred[j + 1]
                denom = x0 - x1
                frac = x0 / denom if abs(denom) > 1e-15 else 0.5
                positions.append(j + float(np.clip(frac, 0.0, 1.0)))
                rising.append(bool(states[int(k) + 1] > 0.0))
            search_start = int(k) + 1
        return np.asarray(positions, dtype=np.float64), np.asarray(rising, dtype=bool)

    @staticmethod
    def _interval_quality(half_periods: FloatArray) -> float:
        mean_half = float(np.mean(half_periods))
        if mean_half < 1e-15:
            return 0.0
        cv = float(np.std(half_periods)) / mean_half
        return float(np.clip(1.0 - cv, 0.0, 1.0))
