# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LSL BCI bridge tests

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.lsl_bci_bridge import HAS_LSL, LSLBCIBridge


class TestInit:
    def test_defaults(self) -> None:
        bridge = LSLBCIBridge()
        assert bridge.stream_name == "EEG"
        assert bridge.target_channel == 0
        assert bridge.buffer_size_s == 2.0

    def test_custom_params(self) -> None:
        bridge = LSLBCIBridge(stream_name="BCI", target_channel=3, buffer_size_s=5.0)
        assert bridge.stream_name == "BCI"
        assert bridge.target_channel == 3
        assert bridge.buffer_size_s == 5.0

    def test_initial_state(self) -> None:
        bridge = LSLBCIBridge()
        assert bridge._running is False
        assert bridge._inlet is None
        assert len(bridge._data_buffer) == 0


class TestConnectWithoutLSL:
    def test_returns_false(self) -> None:
        bridge = LSLBCIBridge()
        if not HAS_LSL:
            assert bridge.connect(timeout=0.1) is False


class TestGetInstantaneousPhase:
    def test_empty_buffer_returns_zero(self) -> None:
        bridge = LSLBCIBridge()
        assert bridge.get_instantaneous_phase() == 0.0

    def test_short_buffer_returns_zero(self) -> None:
        bridge = LSLBCIBridge()
        bridge._data_buffer = [0.0] * 10
        assert bridge.get_instantaneous_phase() == 0.0

    def test_sufficient_buffer_returns_phase(self) -> None:
        bridge = LSLBCIBridge()
        t = np.linspace(0, 2 * np.pi, 64)
        bridge._data_buffer = list(np.sin(t))
        phase = bridge.get_instantaneous_phase()
        assert 0.0 <= phase < 2 * np.pi

    def test_phase_in_range(self) -> None:
        bridge = LSLBCIBridge()
        rng = np.random.default_rng(42)
        bridge._data_buffer = list(rng.normal(0, 1, 128))
        phase = bridge.get_instantaneous_phase()
        assert 0.0 <= phase < 2 * np.pi

    def test_constant_signal_phase(self) -> None:
        bridge = LSLBCIBridge()
        bridge._data_buffer = [1.0] * 64
        phase = bridge.get_instantaneous_phase()
        assert np.isfinite(phase)

    def test_sinusoidal_advances(self) -> None:
        bridge = LSLBCIBridge()
        t1 = np.linspace(0, 4 * np.pi, 128)
        bridge._data_buffer = list(np.sin(t1[:64]))
        p1 = bridge.get_instantaneous_phase()
        bridge._data_buffer = list(np.sin(t1[32:96]))
        p2 = bridge.get_instantaneous_phase()
        # Phases should differ (signal shifted)
        assert p1 != p2


class TestStartWithoutInlet:
    def test_start_without_connect_raises(self) -> None:
        bridge = LSLBCIBridge(stream_name="TOPOLOGY_DETAIL")
        if not HAS_LSL:
            with pytest.raises(RuntimeError) as excinfo:
                bridge.start()
            msg = str(excinfo.value)
            assert "Could not connect" in msg
            # Stream name must not be echoed (may reveal deployment topology).
            assert "TOPOLOGY_DETAIL" not in msg


class TestStopIdempotent:
    def test_stop_before_start(self) -> None:
        bridge = LSLBCIBridge()
        bridge.stop()  # should not raise
        assert bridge._running is False
        assert bridge._inlet is None

    def test_double_stop(self) -> None:
        bridge = LSLBCIBridge()
        bridge.stop()
        bridge.stop()


class TestCaptureLoop:
    def _run_capture(self, bridge, samples):
        """Helper: feed samples into capture loop then stop it."""
        call_count = 0

        def pull_side_effect(timeout=0.1):
            nonlocal call_count
            if call_count < len(samples):
                val = samples[call_count]
                call_count += 1
                if isinstance(val, Exception):
                    raise val
                return val
            bridge._running = False
            raise OSError("stop")

        mock_inlet = MagicMock()
        mock_inlet.pull_sample.side_effect = pull_side_effect
        bridge._inlet = mock_inlet
        bridge._running = True
        bridge._capture_loop()

    def test_appends_to_buffer(self) -> None:
        bridge = LSLBCIBridge(target_channel=0)
        bridge._buffer_len = 10
        self._run_capture(bridge, [([0.5, 0.1], 100.0), ([0.6, 0.2], 100.1)])
        assert len(bridge._data_buffer) == 2
        assert bridge._data_buffer[0] == 0.5
        assert bridge._data_buffer[1] == 0.6

    def test_buffer_capped(self) -> None:
        bridge = LSLBCIBridge(target_channel=0)
        bridge._buffer_len = 3
        samples = [([float(i)], float(i)) for i in range(5)]
        self._run_capture(bridge, samples)
        assert len(bridge._data_buffer) == 3
        assert bridge._data_buffer[0] == 2.0

    def test_none_sample_skipped(self) -> None:
        bridge = LSLBCIBridge(target_channel=0)
        bridge._buffer_len = 100
        self._run_capture(bridge, [(None, None), ([1.0], 100.0)])
        assert len(bridge._data_buffer) == 1
