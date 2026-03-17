# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hardware I/O adapter tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.adapters.hardware_io import (
    HAS_BRAINFLOW,
    HAS_MODBUS,
    SampleBuffer,
    SimulatedBoardAdapter,
)


class TestSampleBuffer:
    def test_push_and_get(self):
        buf = SampleBuffer(capacity=10, n_channels=2)
        samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        buf.push(samples)
        recent = buf.get_recent(3)
        assert recent.shape == (2, 3)
        np.testing.assert_allclose(recent, samples)

    def test_ring_buffer_wraps(self):
        buf = SampleBuffer(capacity=4, n_channels=1)
        for i in range(6):
            buf.push(np.array([[float(i)]]))
        recent = buf.get_recent(4)
        assert recent.shape == (1, 4)
        np.testing.assert_allclose(recent[0], [2.0, 3.0, 4.0, 5.0])

    def test_get_recent_empty(self):
        buf = SampleBuffer(capacity=10, n_channels=2)
        recent = buf.get_recent(5)
        assert recent.shape == (2, 0)

    def test_get_recent_partial(self):
        buf = SampleBuffer(capacity=10, n_channels=1)
        buf.push(np.array([[1.0, 2.0]]))
        recent = buf.get_recent(5)
        assert recent.shape == (1, 2)


class TestSimulatedBoard:
    def test_start_stop(self):
        board = SimulatedBoardAdapter(n_channels=4, sample_rate=256)
        board.start()
        assert board.n_channels == 4
        assert board.sample_rate == 256
        board.stop()

    def test_channel_data_shape(self):
        board = SimulatedBoardAdapter(n_channels=4, sample_rate=256)
        board.start()
        data = board.get_channel_data(0, n_samples=128)
        assert data.shape == (128,)
        board.stop()

    def test_all_eeg_shape(self):
        board = SimulatedBoardAdapter(n_channels=8, sample_rate=256)
        board.start()
        data = board.get_all_eeg(n_samples=64)
        assert data.shape == (8, 64)
        board.stop()

    def test_sinusoidal_output(self):
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
        assert abs(peak_freq - freq) < 2.0

    def test_custom_frequencies(self):
        freqs = np.array([5.0, 10.0, 20.0])
        board = SimulatedBoardAdapter(
            n_channels=3,
            sample_rate=256,
            frequencies=freqs,
        )
        board.start()
        data = board.get_all_eeg(n_samples=256)
        assert data.shape == (3, 256)
        board.stop()


class TestHasFlags:
    def test_has_brainflow_is_bool(self):
        assert isinstance(HAS_BRAINFLOW, bool)

    def test_has_modbus_is_bool(self):
        assert isinstance(HAS_MODBUS, bool)
