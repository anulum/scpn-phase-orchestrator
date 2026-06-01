# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LSL BCI bridge tests

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import scpn_phase_orchestrator.adapters.lsl_bci_bridge as lsl_mod
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

    @pytest.mark.parametrize("stream_name", ["", "bad\nstream", object()])
    def test_rejects_invalid_stream_name(self, stream_name: object) -> None:
        with pytest.raises(ValueError, match="stream_name"):
            LSLBCIBridge(stream_name=stream_name)  # type: ignore[arg-type]

    @pytest.mark.parametrize("target_channel", [-1, True, 1.5, "0"])
    def test_rejects_invalid_target_channel(self, target_channel: object) -> None:
        with pytest.raises(ValueError, match="target_channel"):
            LSLBCIBridge(target_channel=target_channel)  # type: ignore[arg-type]

    @pytest.mark.parametrize("buffer_size_s", [0.0, -1.0, True, float("nan"), "2.0"])
    def test_rejects_invalid_buffer_size_s(self, buffer_size_s: object) -> None:
        with pytest.raises(ValueError, match="buffer_size_s"):
            LSLBCIBridge(buffer_size_s=buffer_size_s)  # type: ignore[arg-type]

    def test_rejects_stream_name_control_character_without_echoing_value(self) -> None:
        stream_name = "PRIVATE_TOPOLOGY\rEEG"

        with pytest.raises(ValueError) as excinfo:
            LSLBCIBridge(stream_name=stream_name)

        message = str(excinfo.value)
        assert message == "stream_name must not contain control characters"
        assert "PRIVATE_TOPOLOGY" not in message

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


class TestConnectWithLSL:
    def test_returns_false_when_named_stream_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pylsl = types.SimpleNamespace(
            resolve_byprop=MagicMock(return_value=()),
            StreamInlet=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
        reloaded = importlib.reload(lsl_mod)
        try:
            bridge = reloaded.LSLBCIBridge(stream_name="EEG")

            assert bridge.connect(timeout=0.25) is False

            fake_pylsl.resolve_byprop.assert_called_once_with(
                "name",
                "EEG",
                timeout=0.25,
            )
            fake_pylsl.StreamInlet.assert_not_called()
        finally:
            monkeypatch.delitem(sys.modules, "pylsl", raising=False)
            importlib.reload(lsl_mod)

    def test_resolves_stream_and_records_buffer_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeInfo:
            def nominal_srate(self) -> float:
                return 128.0

        class FakeInlet:
            def __init__(self, stream) -> None:
                self.stream = stream

            def info(self) -> FakeInfo:
                return FakeInfo()

        fake_pylsl = types.SimpleNamespace(
            resolve_byprop=MagicMock(return_value=("stream-1",)),
            StreamInlet=FakeInlet,
        )
        monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
        reloaded = importlib.reload(lsl_mod)
        try:
            bridge = reloaded.LSLBCIBridge(stream_name="EEG", buffer_size_s=0.5)

            assert bridge.connect(timeout=0.25) is True

            fake_pylsl.resolve_byprop.assert_called_once_with(
                "name",
                "EEG",
                timeout=0.25,
            )
            assert bridge._inlet.stream == "stream-1"
            assert bridge.sampling_rate == 128.0
            assert bridge._buffer_len == 64
        finally:
            monkeypatch.delitem(sys.modules, "pylsl", raising=False)
            importlib.reload(lsl_mod)

    @pytest.mark.parametrize("timeout", [True, 0.0, -0.1, float("nan"), "0.25"])
    def test_rejects_invalid_connect_timeout_before_stream_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        timeout: object,
    ) -> None:
        fake_pylsl = types.SimpleNamespace(
            resolve_byprop=MagicMock(return_value=("stream-1",)),
            StreamInlet=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
        reloaded = importlib.reload(lsl_mod)
        try:
            bridge = reloaded.LSLBCIBridge(stream_name="EEG")

            with pytest.raises(ValueError, match="connect_timeout"):
                bridge.connect(timeout=timeout)  # type: ignore[arg-type]

            fake_pylsl.resolve_byprop.assert_not_called()
            fake_pylsl.StreamInlet.assert_not_called()
        finally:
            monkeypatch.delitem(sys.modules, "pylsl", raising=False)
            importlib.reload(lsl_mod)

    @pytest.mark.parametrize("nominal_srate", [0.0, -1.0, True, float("nan"), "250"])
    def test_invalid_nominal_srate_fails_closed_without_retaining_inlet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        nominal_srate: object,
    ) -> None:
        class FakeInfo:
            def nominal_srate(self) -> object:
                return nominal_srate

        class FakeInlet:
            def info(self) -> FakeInfo:
                return FakeInfo()

        fake_pylsl = types.SimpleNamespace(
            resolve_byprop=MagicMock(return_value=("stream-1",)),
            StreamInlet=MagicMock(return_value=FakeInlet()),
        )
        monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
        reloaded = importlib.reload(lsl_mod)
        try:
            bridge = reloaded.LSLBCIBridge(stream_name="EEG")

            assert bridge.connect(timeout=0.25) is False
            assert bridge._inlet is None
            assert bridge._buffer_len == 0
        finally:
            monkeypatch.delitem(sys.modules, "pylsl", raising=False)
            importlib.reload(lsl_mod)


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

    def test_boolean_sample_buffer_is_ignored_before_phase_extraction(self) -> None:
        bridge = LSLBCIBridge()
        bridge._data_buffer = [True] * 64
        assert bridge.get_instantaneous_phase() == 0.0

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


class TestStartWithInlet:
    def test_start_uses_existing_inlet_and_starts_capture_thread(self) -> None:
        class FakeThread:
            def __init__(self) -> None:
                self.started = False

            def start(self) -> None:
                self.started = True

        bridge = LSLBCIBridge()
        fake_thread = FakeThread()
        bridge._inlet = object()
        bridge._thread = fake_thread

        bridge.start()

        assert bridge._running is True
        assert fake_thread.started is True


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

    def test_stop_joins_live_capture_thread(self) -> None:
        class FakeThread:
            def __init__(self) -> None:
                self.join_timeout = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float) -> None:
                self.join_timeout = timeout

        bridge = LSLBCIBridge()
        fake_thread = FakeThread()
        bridge._thread = fake_thread
        bridge._inlet = object()
        bridge._running = True

        bridge.stop()

        assert bridge._running is False
        assert bridge._inlet is None
        assert fake_thread.join_timeout == 1.0


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

    def test_boolean_samples_and_invalid_timestamps_are_skipped(self) -> None:
        bridge = LSLBCIBridge(target_channel=0)
        bridge._buffer_len = 100
        self._run_capture(
            bridge,
            [
                ([True], 100.0),
                ([0.5], -1.0),
                ([0.6], np.nan),
                ([0.7], 100.1),
            ],
        )
        assert bridge._data_buffer == [0.7]
