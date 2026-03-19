# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BrainFlowAdapter",
    "ModbusAdapter",
    "SimulatedBoardAdapter",
    "HAS_BRAINFLOW",
    "HAS_MODBUS",
]

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams  # pragma: no cover

    HAS_BRAINFLOW = True  # pragma: no cover
except ImportError:
    HAS_BRAINFLOW = False

try:
    from pymodbus.client import ModbusTcpClient  # pragma: no cover

    HAS_MODBUS = True  # pragma: no cover
except ImportError:
    HAS_MODBUS = False


@dataclass
class SampleBuffer:
    """Ring buffer for streaming sensor data."""

    capacity: int
    n_channels: int
    buffer: NDArray = field(init=False)
    write_idx: int = field(init=False, default=0)
    count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.buffer = np.zeros((self.n_channels, self.capacity))

    def push(self, samples: NDArray) -> None:
        """Push (n_channels, n_samples) into the ring buffer."""
        n_samples = samples.shape[1]
        for i in range(n_samples):
            self.buffer[:, self.write_idx % self.capacity] = samples[:, i]
            self.write_idx += 1
        self.count = min(self.count + n_samples, self.capacity)

    def get_recent(self, n: int) -> NDArray:
        """Get the last n samples as (n_channels, n)."""
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
        return int(self._sample_rate)

    @property
    def eeg_channels(self) -> list[int]:
        return list(self._eeg_channels)

    @property
    def n_channels(self) -> int:
        return len(self._eeg_channels)

    def start(self) -> None:
        self._board.prepare_session()
        self._board.start_stream(self._buffer_size)
        self._running = True

    def stop(self) -> None:
        if self._running:
            self._board.stop_stream()
            self._board.release_session()
            self._running = False

    def get_channel_data(self, channel_idx: int, n_samples: int = 256) -> NDArray:
        """Get recent samples from one EEG channel."""
        data = self._board.get_current_board_data(n_samples)
        ch = self._eeg_channels[channel_idx]
        return np.asarray(data[ch])

    def get_all_eeg(self, n_samples: int = 256) -> NDArray:
        """Get (n_eeg_channels, n_samples) of recent EEG data."""
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
        frequencies: NDArray | None = None,
    ) -> None:
        self._n_channels = n_channels
        self._sample_rate = sample_rate
        self._freqs = (
            frequencies
            if frequencies is not None
            else np.linspace(1.0, 40.0, n_channels)
        )
        self._t = 0.0
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def start(self) -> None:
        self._t = 0.0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_channel_data(self, channel_idx: int, n_samples: int = 256) -> NDArray:
        sr = self._sample_rate
        t = np.arange(n_samples) / sr + self._t
        self._t += n_samples / sr
        return np.asarray(np.sin(2.0 * np.pi * self._freqs[channel_idx] * t))

    def get_all_eeg(self, n_samples: int = 256) -> NDArray:
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
        self._client = ModbusTcpClient(host, port=port)
        self._connected = False

    def connect(self) -> None:
        self._client.connect()
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self._client.close()
            self._connected = False

    def read_holding_registers(self, address: int, count: int = 1) -> NDArray:
        """Read holding registers, return as float64 array."""
        result = self._client.read_holding_registers(address, count)
        if result.isError():
            return np.zeros(count)
        return np.array(result.registers, dtype=np.float64)

    def write_register(self, address: int, value: int) -> bool:
        """Write a single holding register."""
        result = self._client.write_register(address, value)
        return not result.isError()
