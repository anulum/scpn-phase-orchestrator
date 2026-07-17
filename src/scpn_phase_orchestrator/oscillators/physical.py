# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical oscillator

"""Physical-channel phase extraction from continuous numeric waveforms.

`PhysicalExtractor` validates finite one-dimensional real signals and positive
sample rates, then derives instantaneous phase, angular frequency, amplitude,
and envelope-quality metadata via the Hilbert transform. Optional Rust
acceleration preserves the same `PhaseState` contract as the NumPy/SciPy path.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

__all__ = ["PhysicalExtractor"]

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

try:
    from spo_kernel import physical_extract as _rust_physical_extract
except ImportError:
    _rust_physical_extract = None


def _validate_node_id(value: object) -> str:
    """Return the validated node id, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node_id must be a non-empty string")
    return value


def _validate_signal(value: object) -> FloatArray:
    """Return the signal as a validated finite array, else raise."""
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
    """Return the sample rate as a validated positive value, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("sample_rate must be finite and positive")
    sample_rate = float(value)
    if not isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be finite and positive")
    return sample_rate


def _validate_band(value: object) -> tuple[float, float] | None:
    """Return the pass-band as a validated ``(low, high)`` pair, or ``None``."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("band must be a (low_hz, high_hz) pair or None")
    low, high = value
    reals = (
        not isinstance(low, bool)
        and not isinstance(high, bool)
        and isinstance(low, Real)
        and isinstance(high, Real)
    )
    if not reals:
        raise ValueError("band edges must be real numbers")
    low_f, high_f = float(low), float(high)
    if not (isfinite(low_f) and isfinite(high_f)) or not 0.0 < low_f < high_f:
        raise ValueError("band must satisfy 0 < low < high")
    return (low_f, high_f)


def _validate_filter_order(value: object) -> int:
    """Return the band-pass filter order as a validated positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("filter_order must be a positive integer")
    return value


def _validate_edge_trim(value: object) -> int | None:
    """Return the edge-trim count as a validated non-negative int, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("edge_trim must be a non-negative integer or None")
    return value


class PhysicalExtractor(PhaseExtractor):
    """Extracts instantaneous phase from continuous waveforms via Hilbert transform.

    By default the extractor Hilbert-transforms the raw broadband signal and reports
    the trailing (endpoint) instantaneous phase — the historical behaviour, retained
    bit-for-bit so existing bindings and sealed evidence are unchanged. Two optional,
    opt-in refinements are available for callers who need a cleaner estimate:

    - ``band=(low_hz, high_hz)`` applies a zero-phase Butterworth band-pass
      (``scipy.signal.filtfilt``) before the Hilbert transform, isolating the phase
      of interest instead of mixing all spectral content.
    - ``edge_trim`` (or, when a band is set, an automatic filter-transient trim)
      discards edge samples where the FFT-Hilbert and filtfilt transients are worst,
      so the reported phase is taken from the last reliable interior sample rather
      than the artefact-prone endpoint.

    Both refinements are applied identically before the NumPy and Rust paths, so the
    accelerated kernel needs no change and stays bit-parity with the reference.
    """

    def __init__(
        self,
        node_id: str = "phys_0",
        *,
        band: tuple[float, float] | Sequence[float] | None = None,
        filter_order: int = 4,
        edge_trim: int | None = None,
    ):
        self._node_id = _validate_node_id(node_id)
        self._band = _validate_band(band)
        self._filter_order = _validate_filter_order(filter_order)
        self._edge_trim = _validate_edge_trim(edge_trim)

    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]:
        """Extract instantaneous phase from a 1-D waveform via Hilbert transform.

        Parameters
        ----------
        signal : FloatArray
            Input signal, shape ``(T,)``.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        list[PhaseState]
            Instantaneous phase from a 1-D waveform via Hilbert transform.
        """
        signal = _validate_signal(signal)
        sample_rate = _validate_sample_rate(sample_rate)
        from scipy.signal import hilbert  # noqa: PLC0415

        filtered = self._apply_bandpass(signal, sample_rate)
        analytic = hilbert(filtered)

        trim = self._resolve_edge_trim(analytic.shape[0])
        if trim:
            filtered = filtered[trim : filtered.shape[0] - trim]
            analytic = analytic[trim : analytic.shape[0] - trim]

        if _rust_physical_extract is not None:
            try:
                theta, omega, amplitude, quality = _rust_physical_extract(
                    np.ascontiguousarray(np.real(analytic)),
                    np.ascontiguousarray(np.imag(analytic)),
                    sample_rate,
                )
            except Exception:
                theta, omega, amplitude, quality = self._python_extract(
                    filtered, analytic, sample_rate
                )
        else:
            theta, omega, amplitude, quality = self._python_extract(
                filtered, analytic, sample_rate
            )

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

    def _python_extract(
        self,
        signal: FloatArray,
        analytic: ComplexArray,
        sample_rate: float,
    ) -> tuple[float, float, float, float]:
        """Return the reference phase extraction from a waveform (NumPy floor)."""
        inst_phase = np.angle(analytic) % TWO_PI
        inst_amp = np.abs(analytic)
        unwrapped = np.unwrap(np.angle(analytic))
        inst_freq = np.gradient(unwrapped) * sample_rate / TWO_PI

        theta = float(inst_phase[-1])
        omega = float(np.median(inst_freq)) * TWO_PI  # rad/s
        amplitude = float(np.mean(inst_amp))
        quality = self._envelope_quality(signal, analytic)
        return theta, omega, amplitude, quality

    def _apply_bandpass(self, signal: FloatArray, sample_rate: float) -> FloatArray:
        """Return the zero-phase band-passed signal, or the signal unchanged.

        When no band is configured the input is returned as-is, so the default
        extraction path is bit-for-bit identical to the historical behaviour.
        """
        if self._band is None:
            return signal
        low, high = self._band
        nyquist = 0.5 * sample_rate
        if not 0.0 < low < high < nyquist:
            raise ValueError(
                f"band {self._band} must satisfy 0 < low < high < Nyquist "
                f"({nyquist}) for sample_rate {sample_rate}"
            )
        from scipy.signal import butter, filtfilt  # noqa: PLC0415

        coeff_b, coeff_a = butter(
            self._filter_order, (low / nyquist, high / nyquist), btype="band"
        )
        padlen = 3 * max(len(coeff_a), len(coeff_b))
        if signal.shape[0] <= padlen:
            raise ValueError(
                f"signal length {signal.shape[0]} is too short for a band-pass of "
                f"order {self._filter_order} (needs > {padlen} samples)"
            )
        return np.asarray(filtfilt(coeff_b, coeff_a, signal), dtype=np.float64)

    def _resolve_edge_trim(self, n: int) -> int:
        """Return the number of edge samples to trim from each end of the analytic.

        An explicit ``edge_trim`` wins; otherwise a band-pass implies an automatic
        trim of the filtfilt transient (three filter lengths). The result is clamped
        so at least two samples survive, preserving the ``PhaseState`` contract.
        """
        if self._edge_trim is not None:
            trim = self._edge_trim
        elif self._band is not None:
            trim = 3 * (self._filter_order + 1)
        else:
            return 0
        return min(trim, max(0, (n - 2) // 2))

    @staticmethod
    def _envelope_quality(signal: FloatArray, analytic: ComplexArray) -> float:
        """Return the envelope-based extraction quality score."""
        envelope = np.abs(analytic)
        mean_env = np.mean(envelope)
        if mean_env < 1e-15:
            return 0.0
        cv = np.std(envelope) / mean_env
        return float(np.clip(1.0 - cv, 0.0, 1.0))
