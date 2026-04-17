# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hardware I/O adapter tests

from __future__ import annotations

import importlib.util

import numpy as np

from scpn_phase_orchestrator.adapters.hardware_io import (
    HAS_BRAINFLOW,
    HAS_MODBUS,
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
