# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hardware I/O adapter

"""Real-time hardware I/O via BrainFlow (EEG, PPG, EMG) and SCADA/Modbus.

BrainFlow supports: OpenBCI, Muse, Emotiv, NeuroSky, BrainBit, Enobio,
and simulated boards for development. Install: pip install brainflow

SCADA: Modbus TCP via pymodbus. Install: pip install pymodbus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_tcp_port,
)

__all__ = [
    "BrainFlowAdapter",
    "ModbusAdapter",
    "SampleBuffer",
    "SimulatedBoardAdapter",
    "HAS_BRAINFLOW",
    "HAS_MODBUS",
]

FloatArray: TypeAlias = NDArray[np.float64]


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        if field == "count":
            raise ValueError("count must be > 0")
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        if field == "address":
            raise ValueError("address must be >= 0")
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _channel_index(value: object, *, n_channels: int) -> int:
    idx = _non_negative_int(value, field="channel_idx")
    if idx >= n_channels:
        raise ValueError("channel_idx must be within configured channel range")
    return idx


def _has_non_real_numeric_alias(values: object) -> bool:
    array = np.asarray(values)
    if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.complexfloating):
        return True
    if array.dtype == object:
        return any(
            isinstance(item, bool | np.bool_ | complex | np.complexfloating)
            for item in array.flat
        )
    return False


def _validated_frequencies(
    frequencies: FloatArray | None,
    *,
    n_channels: int,
) -> FloatArray:
    if frequencies is None:
        return np.linspace(1.0, 40.0, n_channels, dtype=np.float64)
    if _has_non_real_numeric_alias(frequencies):
        raise ValueError("frequencies must contain real numeric values")
    array = np.asarray(frequencies, dtype=np.float64)
    if array.shape != (n_channels,):
        raise ValueError(f"frequencies must have shape ({n_channels},)")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("frequencies must contain finite positive values")
    return array.copy()


try:
    # type ignore: BrainFlow is optional and lacks complete typing metadata.
    from brainflow.board_shim import (  # type: ignore[import-not-found,import-untyped]  # pragma: no cover
        BoardShim,
        BrainFlowInputParams,
    )

    HAS_BRAINFLOW = True  # pragma: no cover
except ImportError:
    HAS_BRAINFLOW = False

try:
    # type ignore: pymodbus is optional for hardware adapter deployments.
    from pymodbus.client import (  # type: ignore[import-not-found]
        ModbusTcpClient as _PymodbusTcpClient,  # pragma: no cover
    )

    ModbusTcpClient: Any = _PymodbusTcpClient
    HAS_MODBUS = True  # pragma: no cover
except ImportError:
    ModbusTcpClient = None
    HAS_MODBUS = False


@dataclass
class SampleBuffer:
    """Ring buffer for streaming sensor data."""

    capacity: int
    n_channels: int
    buffer: FloatArray = field(init=False)
    write_idx: int = field(init=False, default=0)
    count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.capacity = _positive_int(self.capacity, field="capacity")
        self.n_channels = _positive_int(self.n_channels, field="n_channels")
        self.buffer = np.zeros((self.n_channels, self.capacity))

    def push(self, samples: FloatArray) -> None:
        """Push (n_channels, n_samples) into the ring buffer.

        Parameters
        ----------
        samples : FloatArray
            Sample block, shape ``(n_channels, n_samples)``.

        Raises
        ------
        ValueError
            If the sample block shape is invalid.
        """
        if samples.ndim != 2:
            raise ValueError(
                "samples must be a 2D array shaped (n_channels, n_samples)"
            )
        if samples.shape[0] != self.n_channels:
            raise ValueError(
                f"samples first dimension must match n_channels={self.n_channels}"
            )
        if _has_non_real_numeric_alias(samples):
            raise ValueError("samples must contain real numeric values")
        if not np.issubdtype(samples.dtype, np.number):
            raise ValueError("samples must be numeric")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must contain only finite values")
        n_samples = samples.shape[1]
        for i in range(n_samples):
            self.buffer[:, self.write_idx % self.capacity] = samples[:, i]
            self.write_idx += 1
        self.count = min(self.count + n_samples, self.capacity)

    def get_recent(self, n: int) -> FloatArray:
        """Get the last n samples as (n_channels, n).

        Parameters
        ----------
        n : int
            Number of most-recent samples to return.

        Returns
        -------
        FloatArray
            The most recent ``n`` samples, shape ``(n_channels, n)``.
        """
        n = _positive_int(n, field="n")
        n = min(n, self.count)
        if n == 0:
            return np.zeros((self.n_channels, 0))
        end = self.write_idx % self.capacity
        if end >= n:
            return self.buffer[:, end - n : end]
        return np.concatenate(
            [self.buffer[:, self.capacity - (n - end) :], self.buffer[:, :end]],
            axis=1,
        )


class BrainFlowAdapter:  # pragma: no cover
    """Streams EEG/PPG/EMG data from BrainFlow-supported devices.

    Usage:
        adapter = BrainFlowAdapter(board_id=BoardIds.SYNTHETIC_BOARD)
        adapter.start()
        signal = adapter.get_channel_data(0, n_samples=256)
        adapter.stop()
    """

    def __init__(
        self,
        board_id: int = 0,
        serial_port: str = "",
        buffer_size: int = 4096,
    ) -> None:
        if not HAS_BRAINFLOW:
            msg = "brainflow not installed. pip install brainflow"
            raise ImportError(msg)
        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        self._board = BoardShim(board_id, params)
        self._board_id = board_id
        self._eeg_channels = BoardShim.get_eeg_channels(board_id)
        self._sample_rate = BoardShim.get_sampling_rate(board_id)
        self._buffer_size = buffer_size
        self._running = False

    @property
    def sample_rate(self) -> int:
        """Board sampling rate in Hz.

        Returns
        -------
        int
            Board sampling rate in Hz.
        """
        return int(self._sample_rate)

    @property
    def eeg_channels(self) -> list[int]:
        """BrainFlow EEG channel indices for this board.

        Returns
        -------
        list[int]
            BrainFlow EEG channel indices for this board.
        """
        return list(self._eeg_channels)

    @property
    def n_channels(self) -> int:
        """Number of EEG channels.

        Returns
        -------
        int
            Number of EEG channels.
        """
        return len(self._eeg_channels)

    def start(self) -> None:
        """Prepare and start the BrainFlow data stream."""
        self._board.prepare_session()
        self._board.start_stream(self._buffer_size)
        self._running = True

    def stop(self) -> None:
        """Stop the stream and release the board session."""
        if self._running:
            self._board.stop_stream()
            self._board.release_session()
            self._running = False

    def get_channel_data(self, channel_idx: int, n_samples: int = 256) -> FloatArray:
        """Get recent samples from one EEG channel.

        Parameters
        ----------
        channel_idx : int
            Index of the channel to read.
        n_samples : int
            Number of samples to return.

        Returns
        -------
        FloatArray
            Recent samples from the channel, shape ``(n_samples,)``.
        """
        data = self._board.get_current_board_data(n_samples)
        ch = self._eeg_channels[channel_idx]
        return np.asarray(data[ch])

    def get_all_eeg(self, n_samples: int = 256) -> FloatArray:
        """Get (n_eeg_channels, n_samples) of recent EEG data.

        Parameters
        ----------
        n_samples : int
            Number of samples to return.

        Returns
        -------
        FloatArray
            Recent EEG data, shape ``(n_eeg_channels, n_samples)``.
        """
        data = self._board.get_current_board_data(n_samples)
        return np.asarray(data[self._eeg_channels])


class SimulatedBoardAdapter:
    """Generates synthetic sinusoidal signals for development without hardware.

    Matches the BrainFlowAdapter interface.
    """

    def __init__(
        self,
        n_channels: int = 8,
        sample_rate: int = 256,
        frequencies: FloatArray | None = None,
    ) -> None:
        self._n_channels = _positive_int(n_channels, field="n_channels")
        self._sample_rate = _positive_int(sample_rate, field="sample_rate")
        self._freqs = _validated_frequencies(frequencies, n_channels=self._n_channels)
        self._t = 0.0
        self._running = False

    @property
    def sample_rate(self) -> int:
        """Simulated sampling rate in Hz.

        Returns
        -------
        int
            Simulated sampling rate in Hz.
        """
        return self._sample_rate

    @property
    def n_channels(self) -> int:
        """Number of simulated channels.

        Returns
        -------
        int
            Number of simulated channels.
        """
        return self._n_channels

    def start(self) -> None:
        """Reset time counter and begin generating data."""
        self._t = 0.0
        self._running = True

    def stop(self) -> None:
        """Mark the simulated board as stopped."""
        self._running = False

    def get_channel_data(self, channel_idx: int, n_samples: int = 256) -> FloatArray:
        """Return synthetic sinusoidal samples for one channel.

        Parameters
        ----------
        channel_idx : int
            Index of the channel to read.
        n_samples : int
            Number of samples to return.

        Returns
        -------
        FloatArray
            Synthetic samples for the channel, shape ``(n_samples,)``.
        """
        channel_idx = _channel_index(channel_idx, n_channels=self._n_channels)
        n_samples = _positive_int(n_samples, field="n_samples")
        sr = self._sample_rate
        t = np.arange(n_samples) / sr + self._t
        self._t += n_samples / sr
        return np.asarray(np.sin(2.0 * np.pi * self._freqs[channel_idx] * t))

    def get_all_eeg(self, n_samples: int = 256) -> FloatArray:
        """Return synthetic (n_channels, n_samples) sinusoidal data.

        Parameters
        ----------
        n_samples : int
            Number of samples to return.

        Returns
        -------
        FloatArray
            Synthetic EEG data, shape ``(n_channels, n_samples)``.
        """
        n_samples = _positive_int(n_samples, field="n_samples")
        sr = self._sample_rate
        t = np.arange(n_samples) / sr + self._t
        self._t += n_samples / sr
        return np.array([np.sin(2.0 * np.pi * f * t) for f in self._freqs])


class ModbusAdapter:  # pragma: no cover
    """Reads SCADA/PLC registers via Modbus TCP for industrial control.

    Usage:
        adapter = ModbusAdapter("192.168.1.100", port=502)
        adapter.connect()
        values = adapter.read_holding_registers(0, count=10)
        adapter.disconnect()
    """

    def __init__(self, host: str, port: int = 502) -> None:
        if not HAS_MODBUS:
            msg = "pymodbus not installed. pip install pymodbus"
            raise ImportError(msg)
        host_text = require_non_empty_str(host, field="Modbus host")
        self._host = host_text
        self._port = require_tcp_port(port, field="Modbus port")
        self._client = ModbusTcpClient(host_text, port=self._port)
        self._connected = False

    def connect(self) -> None:
        """Open Modbus TCP connection."""
        self._client.connect()
        self._connected = True

    def disconnect(self) -> None:
        """Close Modbus TCP connection if open."""
        if self._connected:
            self._client.close()
            self._connected = False

    def read_holding_registers(self, address: int, count: int = 1) -> FloatArray:
        """Read holding registers, return as float64 array.

        Parameters
        ----------
        address : int
            Modbus register address.
        count : int
            Number of registers to read.

        Returns
        -------
        FloatArray
            The register values as a float64 array.
        """
        address = _non_negative_int(address, field="address")
        count = _positive_int(count, field="count")
        result = self._client.read_holding_registers(address, count=count)
        if result.isError():
            return np.zeros(count)
        return np.array(result.registers, dtype=np.float64)

    def write_register(self, address: int, value: int) -> bool:
        """Write a single holding register.

        Parameters
        ----------
        address : int
            Modbus register address.
        value : int
            Register value to write.

        Returns
        -------
        bool
            ``True`` when the write succeeds.

        Raises
        ------
        ValueError
            If the value is out of range.
        """
        address = _non_negative_int(address, field="address")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("value must be an integer")
        result = self._client.write_register(address, value)
        return not result.isError()
