# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hardware I/O adapter tests

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scpn_phase_orchestrator.adapters.hardware_io import (
    HAS_BRAINFLOW,
    HAS_MODBUS,
    ModbusAdapter,
    SampleBuffer,
    SimulatedBoardAdapter,
)

# ---------------------------------------------------------------------------
# SampleBuffer: ring buffer for streaming acquisition
# ---------------------------------------------------------------------------


class TestSampleBuffer:
    """Verify the ring buffer preserves data ordering, wraps correctly,
    and handles edge cases."""

    def test_push_and_get_exact(self):
        buf = SampleBuffer(capacity=10, n_channels=2)
        samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        buf.push(samples)
        recent = buf.get_recent(3)
        assert recent.shape == (2, 3)
        np.testing.assert_allclose(recent, samples)

    def test_ring_buffer_wraps_preserving_order(self):
        """After overflow, oldest samples must be evicted first (FIFO)."""
        buf = SampleBuffer(capacity=4, n_channels=1)
        for i in range(6):
            buf.push(np.array([[float(i)]]))
        recent = buf.get_recent(4)
        np.testing.assert_allclose(recent[0], [2.0, 3.0, 4.0, 5.0])

    def test_empty_buffer_returns_zero_width(self):
        buf = SampleBuffer(capacity=10, n_channels=2)
        recent = buf.get_recent(5)
        assert recent.shape == (2, 0)

    def test_partial_fill_returns_available(self):
        buf = SampleBuffer(capacity=10, n_channels=1)
        buf.push(np.array([[1.0, 2.0]]))
        recent = buf.get_recent(5)
        assert recent.shape == (1, 2), "Should return only available samples"

    def test_multi_push_accumulates(self):
        """Multiple pushes accumulate in the buffer."""
        buf = SampleBuffer(capacity=10, n_channels=1)
        buf.push(np.array([[1.0, 2.0]]))
        buf.push(np.array([[3.0, 4.0]]))
        recent = buf.get_recent(4)
        np.testing.assert_allclose(recent[0], [1.0, 2.0, 3.0, 4.0])

    def test_capacity_respected(self):
        """Buffer should never store more than capacity samples."""
        buf = SampleBuffer(capacity=5, n_channels=1)
        for i in range(20):
            buf.push(np.array([[float(i)]]))
        recent = buf.get_recent(10)
        assert recent.shape[1] <= 5

    def test_push_rejects_non_2d(self):
        buf = SampleBuffer(capacity=8, n_channels=2)
        with pytest.raises(ValueError, match="2D"):
            buf.push(np.array([1.0, 2.0]))

    def test_push_rejects_channel_mismatch(self):
        buf = SampleBuffer(capacity=8, n_channels=2)
        with pytest.raises(ValueError, match="n_channels=2"):
            buf.push(np.array([[1.0], [2.0], [3.0]]))

    def test_push_rejects_non_finite(self):
        buf = SampleBuffer(capacity=8, n_channels=2)
        bad = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="finite"):
            buf.push(bad)

    @given(
        n_channels=st.integers(min_value=1, max_value=6),
        capacity=st.integers(min_value=1, max_value=20),
        n_samples=st.integers(min_value=1, max_value=40),
    )
    def test_push_fuzz_valid_numeric_shapes(self, n_channels, capacity, n_samples):
        """Valid numeric 2D arrays must always be accepted."""
        buf = SampleBuffer(capacity=capacity, n_channels=n_channels)
        samples = np.arange(n_channels * n_samples, dtype=np.float64).reshape(
            n_channels, n_samples
        )
        buf.push(samples)
        recent = buf.get_recent(n_samples)
        assert recent.shape[0] == n_channels
        assert recent.shape[1] == min(n_samples, capacity)


# ---------------------------------------------------------------------------
# SimulatedBoardAdapter: synthetic EEG generation
# ---------------------------------------------------------------------------


class TestSimulatedBoard:
    """Verify that SimulatedBoardAdapter produces physically plausible
    sinusoidal signals for testing without real hardware."""

    def test_properties_match_constructor(self):
        board = SimulatedBoardAdapter(n_channels=4, sample_rate=256)
        assert board.n_channels == 4
        assert board.sample_rate == 256

    def test_channel_data_bounded(self):
        """Simulated sinusoidal data must stay within [-1, 1]."""
        board = SimulatedBoardAdapter(n_channels=4, sample_rate=256)
        board.start()
        data = board.get_channel_data(0, n_samples=512)
        assert np.all(np.abs(data) <= 1.0 + 1e-10)
        board.stop()

    def test_all_eeg_shape_and_finiteness(self):
        board = SimulatedBoardAdapter(n_channels=8, sample_rate=256)
        board.start()
        data = board.get_all_eeg(n_samples=64)
        assert data.shape == (8, 64)
        assert np.all(np.isfinite(data))
        board.stop()

    def test_peak_frequency_matches_configured(self):
        """FFT of simulated signal must peak at the configured frequency."""
        freq = 10.0
        board = SimulatedBoardAdapter(
            n_channels=1,
            sample_rate=256,
            frequencies=np.array([freq]),
        )
        board.start()
        data = board.get_channel_data(0, n_samples=256)
        fft_mag = np.abs(np.fft.rfft(data))
        peak_bin = np.argmax(fft_mag[1:]) + 1
        peak_freq = peak_bin * 256.0 / 256
        assert abs(peak_freq - freq) < 2.0, (
            f"Peak freq {peak_freq:.1f} Hz should match configured {freq:.1f} Hz"
        )

    def test_different_channels_have_different_signals(self):
        """Each channel must have a distinct signal (not all identical)."""
        board = SimulatedBoardAdapter(
            n_channels=3,
            sample_rate=256,
            frequencies=np.array([5.0, 10.0, 20.0]),
        )
        board.start()
        data = board.get_all_eeg(n_samples=256)
        # Channels with different frequencies must differ
        assert not np.allclose(data[0], data[1]), "Channels 0 and 1 must differ"
        assert not np.allclose(data[1], data[2]), "Channels 1 and 2 must differ"
        board.stop()

    def test_pipeline_wiring_to_phase_extractor(self):
        """SimulatedBoard → PhysicalExtractor: proves the adapter produces
        valid input for the phase extraction pipeline."""
        from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

        board = SimulatedBoardAdapter(
            n_channels=1,
            sample_rate=256,
            frequencies=np.array([10.0]),
        )
        board.start()
        signal = board.get_channel_data(0, n_samples=256)
        board.stop()

        extractor = PhysicalExtractor()
        states = extractor.extract(signal, 256.0)
        assert len(states) == 1
        assert 0.0 <= states[0].theta < 2 * np.pi
        assert states[0].quality > 0.5, (
            f"Clean sinusoid should give quality>0.5, got {states[0].quality:.3f}"
        )


# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------


class TestHardwareFlags:
    """Verify optional dependency detection matches actual availability."""

    def test_has_brainflow_matches_importlib(self):
        expected = importlib.util.find_spec("brainflow") is not None
        assert expected == HAS_BRAINFLOW

    def test_has_modbus_matches_importlib(self):
        expected = importlib.util.find_spec("pymodbus") is not None
        assert expected == HAS_MODBUS


# ---------------------------------------------------------------------------
# Adapter registry wiring
# ---------------------------------------------------------------------------


class TestHardwareRegistryWiring:
    """Verify hardware_io is intentionally public optional adapter surface."""

    def test_adapters_package_exports_hardware_io_surface(self):
        import scpn_phase_orchestrator.adapters as adapters

        assert adapters.SampleBuffer is SampleBuffer
        assert adapters.SimulatedBoardAdapter is SimulatedBoardAdapter
        assert adapters.ModbusAdapter is ModbusAdapter


# ---------------------------------------------------------------------------
# ModbusAdapter: mock-backed branch coverage without SCADA hardware
# ---------------------------------------------------------------------------


class TestModbusAdapter:
    """Verify ModbusAdapter semantics using a mocked pymodbus client."""

    def test_missing_optional_dependency_raises(self, monkeypatch):
        import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

        monkeypatch.setattr(hardware_io, "HAS_MODBUS", False)

        with pytest.raises(ImportError, match="pymodbus not installed"):
            ModbusAdapter("plc.local")

    def test_connect_disconnect_read_write_success(self, monkeypatch):
        import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

        client = MagicMock()
        read_result = MagicMock()
        read_result.isError.return_value = False
        read_result.registers = [10, 20]
        write_result = MagicMock()
        write_result.isError.return_value = False
        client.read_holding_registers.return_value = read_result
        client.write_register.return_value = write_result
        factory = MagicMock(return_value=client)

        monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
        monkeypatch.setattr(hardware_io, "ModbusTcpClient", factory, raising=False)

        adapter = ModbusAdapter("plc.local", port=1502)
        adapter.connect()
        values = adapter.read_holding_registers(7, count=2)
        wrote = adapter.write_register(9, 123)
        adapter.disconnect()

        factory.assert_called_once_with("plc.local", port=1502)
        client.connect.assert_called_once()
        client.read_holding_registers.assert_called_once_with(7, count=2)
        client.write_register.assert_called_once_with(9, 123)
        client.close.assert_called_once()
        np.testing.assert_allclose(values, np.array([10.0, 20.0]))
        assert wrote is True

    def test_read_error_returns_zero_vector_and_write_reports_false(self, monkeypatch):
        import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

        client = MagicMock()
        read_result = MagicMock()
        read_result.isError.return_value = True
        write_result = MagicMock()
        write_result.isError.return_value = True
        client.read_holding_registers.return_value = read_result
        client.write_register.return_value = write_result

        monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
        monkeypatch.setattr(
            hardware_io,
            "ModbusTcpClient",
            MagicMock(return_value=client),
            raising=False,
        )

        adapter = ModbusAdapter("plc.local")

        np.testing.assert_allclose(
            adapter.read_holding_registers(0, count=3),
            np.zeros(3),
        )
        assert adapter.write_register(0, 1) is False

    def test_read_rejects_invalid_address_and_count(self, monkeypatch):
        import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

        monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
        monkeypatch.setattr(
            hardware_io,
            "ModbusTcpClient",
            MagicMock(return_value=MagicMock()),
            raising=False,
        )
        adapter = ModbusAdapter("plc.local")

        with pytest.raises(ValueError, match="address must be >= 0"):
            adapter.read_holding_registers(-1, count=1)
        with pytest.raises(ValueError, match="count must be > 0"):
            adapter.read_holding_registers(0, count=0)

    def test_write_rejects_invalid_address_and_non_int_value(self, monkeypatch):
        import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

        monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
        monkeypatch.setattr(
            hardware_io,
            "ModbusTcpClient",
            MagicMock(return_value=MagicMock()),
            raising=False,
        )
        adapter = ModbusAdapter("plc.local")

        with pytest.raises(ValueError, match="address must be >= 0"):
            adapter.write_register(-1, 1)
        with pytest.raises(ValueError, match="value must be an integer"):
            adapter.write_register(0, 1.5)  # type: ignore[arg-type]
