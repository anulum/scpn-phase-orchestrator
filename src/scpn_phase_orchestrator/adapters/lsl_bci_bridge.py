# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - LSL BCI Entrainment Bridge

"""Lab Streaming Layer BCI bridge for buffered phase extraction.

The bridge optionally connects to a configured LSL stream, captures one target
channel into a bounded background buffer, and extracts the current Hilbert phase
from recent samples. Configuration rejects invalid stream names, channels, and
buffer durations. When LSL is unavailable or disconnected, it fails without
leaking stream identifiers in shared error messages.
"""

from __future__ import annotations

import threading
from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.signal import hilbert

try:
    import pylsl

    HAS_LSL = True
except ImportError:
    pylsl = None
    HAS_LSL = False

__all__ = ["LSLBCIBridge", "HAS_LSL"]


def _validate_stream_name(stream_name: str) -> str:
    if not isinstance(stream_name, str) or not stream_name:
        raise ValueError("stream_name must be a non-empty string")
    if any(ord(char) < 32 for char in stream_name):
        raise ValueError("stream_name must not contain control characters")
    return stream_name


def _validate_target_channel(target_channel: int) -> int:
    if isinstance(target_channel, bool) or not isinstance(target_channel, Integral):
        raise ValueError("target_channel must be a non-negative integer")
    channel = int(target_channel)
    if channel < 0:
        raise ValueError("target_channel must be a non-negative integer")
    return channel


def _validate_buffer_size_s(buffer_size_s: float) -> float:
    if (
        isinstance(buffer_size_s, bool)
        or not isinstance(buffer_size_s, Real)
        or not isfinite(float(buffer_size_s))
        or float(buffer_size_s) <= 0.0
    ):
        raise ValueError("buffer_size_s must be a finite positive real value")
    return float(buffer_size_s)


class LSLBCIBridge:
    """Real-time BCI Entrainment Bridge via Lab Streaming Layer (LSL).

    This bridge enables direct human-machine synchronization. It streams
    raw EEG data from LSL (e.g., from OpenBCI or Muse), extracts the
    instantaneous phase of target neural oscillations, and provides them
    to the SCPN orchestrator for closed-loop entrainment.

    Attributes:
        stream_name: Name of the LSL stream to listen to.
        target_channel: Index of the EEG channel to use.
        sampling_rate: Sampling rate of the EEG stream (Hz).
    """

    def __init__(
        self,
        stream_name: str = "EEG",
        target_channel: int = 0,
        buffer_size_s: float = 2.0,
    ):
        self.stream_name = _validate_stream_name(stream_name)
        self.target_channel = _validate_target_channel(target_channel)
        self.buffer_size_s = _validate_buffer_size_s(buffer_size_s)

        self._running = False
        self._inlet: Any = None
        self._data_buffer: list[float] = []
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def connect(self, timeout: float = 5.0) -> bool:
        """Resolve and connect to the LSL stream."""
        if not HAS_LSL or pylsl is None:
            return False

        streams = pylsl.resolve_byprop("name", self.stream_name, timeout=timeout)
        if not streams:
            return False

        self._inlet = pylsl.StreamInlet(streams[0])
        info = self._inlet.info()
        self.sampling_rate = info.nominal_srate()
        self._buffer_len = int(self.buffer_size_s * self.sampling_rate)

        return True

    def start(self) -> None:
        """Start the background capture thread."""
        if self._inlet is None and not self.connect():
            # Do not echo the configured stream_name — keep the error
            # message independent of the user-configured identifier so
            # shared logs never broadcast deployment topology.
            raise RuntimeError(
                "Could not connect to configured LSL stream "
                "(check stream configuration)"
            )

        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Stop capture and disconnect."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._inlet = None

    def get_instantaneous_phase(self) -> float:
        """Extract the current phase from the buffered signal.

        Uses Hilbert transform on the recent buffer window.
        Returns phase in [0, 2*pi).
        """
        with self._lock:
            if len(self._data_buffer) < 32:  # Hilbert needs >=32 samples for stability
                return 0.0
            signal = np.array(self._data_buffer)

        # Analytic signal via Hilbert transform
        analytic_signal = hilbert(signal)
        # Instantaneous phase is the angle of the complex signal
        # We take the last value as the 'current' phase
        phase = np.angle(analytic_signal[-1])

        return float(phase % (2 * np.pi))

    def _capture_loop(self) -> None:
        """Background thread reading samples into the buffer."""
        while self._running:
            try:
                sample, timestamp = self._inlet.pull_sample(timeout=0.1)
                if sample:
                    val = sample[self.target_channel]
                    with self._lock:
                        self._data_buffer.append(val)
                        if len(self._data_buffer) > self._buffer_len:
                            self._data_buffer.pop(0)
            except (OSError, ValueError):
                continue
