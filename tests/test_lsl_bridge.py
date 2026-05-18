# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LSL Bridge tests

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.lsl_bci_bridge import LSLBCIBridge


def test_default_constructor_values():
    bridge = LSLBCIBridge()
    assert bridge.stream_name == "EEG"
    assert bridge.target_channel == 0
    assert bridge.buffer_size_s == 2.0


def test_constructor_rejects_invalid_values():
    with pytest.raises(ValueError, match="stream_name"):
        LSLBCIBridge(stream_name="")
    with pytest.raises(ValueError, match="target_channel"):
        LSLBCIBridge(target_channel=-1)
    with pytest.raises(ValueError, match="target_channel"):
        LSLBCIBridge(target_channel=True)
    with pytest.raises(ValueError, match="buffer_size_s"):
        LSLBCIBridge(buffer_size_s=0.0)


def test_phase_extraction_default_with_insufficient_finite_samples():
    bridge = LSLBCIBridge()
    bridge._data_buffer = [np.nan, np.inf, 1.0, 2.0]
    assert bridge.get_instantaneous_phase() == 0.0


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", False)
def test_connect_returns_false_when_lsl_missing():
    bridge = LSLBCIBridge()
    assert bridge.connect() is False


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_connect_rejects_nonfinite_or_nonpositive_timeout(mock_pylsl):
    bridge = LSLBCIBridge()
    mock_pylsl.resolve_byprop.return_value = []

    with pytest.raises(ValueError, match="connect_timeout"):
        bridge.connect(timeout=0.0)
    with pytest.raises(ValueError, match="connect_timeout"):
        bridge.connect(timeout=float("nan"))


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_connect_rejects_invalid_nominal_srate(mock_pylsl):
    mock_pylsl.resolve_byprop.return_value = [MagicMock()]
    mock_info = MagicMock(nominal_srate=lambda: 0.0)
    mock_pylsl.StreamInlet.return_value.info.return_value = mock_info

    bridge = LSLBCIBridge()
    assert bridge.connect() is False
    assert bridge._inlet is None


def test_start_is_idempotent_when_already_running():
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    bridge = LSLBCIBridge(stream_name="EEG")
    bridge._thread = mock_thread
    bridge._running = True

    with patch.object(bridge, "connect") as connect_mock:
        bridge.start()
        connect_mock.assert_not_called()


def test_stop_without_start_is_idempotent():
    bridge = LSLBCIBridge()
    bridge.stop()
    bridge.stop()
    assert bridge._running is False


@patch(
    "scpn_phase_orchestrator.adapters.lsl_bci_bridge.threading.Thread",
)
def test_start_and_thread_idempotence(mock_thread_class):
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False
    mock_thread_class.return_value = mock_thread

    bridge = LSLBCIBridge()
    with patch.object(bridge, "connect", return_value=True):
        bridge.start()
        bridge.start()

    assert mock_thread.start.call_count == 1


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_phase_extraction_filters_non_finite_samples(mock_pylsl):
    mock_pylsl.resolve_byprop.return_value = [MagicMock()]
    mock_pylsl.StreamInlet.return_value.info().nominal_srate.return_value = 250

    bridge = LSLBCIBridge()
    bridge._data_buffer = list(np.sin(np.linspace(0, 1, 64)))
    bridge._data_buffer.extend([float("nan"), float("inf")])
    phase = bridge.get_instantaneous_phase()

    assert np.isfinite(phase)


def test_phase_extraction_returns_default_for_no_sample():
    bridge = LSLBCIBridge()
    bridge._data_buffer = [float("nan")] * 32
    assert bridge.get_instantaneous_phase() == 0.0


def test_capture_loop_ignores_malformed_and_oob_samples():
    bridge = LSLBCIBridge(target_channel=1)
    bridge._running = True
    bridge._buffer_len = 8

    samples = [
        ("malformed", 1.0),
        ([1.0], 1.0),
        ([7.0, 8.0], 1.0),
    ]

    def pull_side_effect(timeout: float):
        sample, _timestamp = samples.pop(0)
        if not samples:
            bridge._running = False
        return sample, _timestamp

    bridge._inlet = MagicMock()
    bridge._inlet.pull_sample.side_effect = pull_side_effect
    thread = threading.Thread(target=bridge._capture_loop)
    thread.start()
    thread.join(timeout=1.0)

    assert bridge._data_buffer == [8.0]


def test_connect_timeout_validation_in_start_error_message_is_scrubbed():
    bridge = LSLBCIBridge(stream_name="TOP-SECRET")
    with patch.object(
        bridge, "connect", side_effect=ValueError("stream_name=TOP-SECRET")
    ):
        with pytest.raises(RuntimeError) as exc:
            bridge.start()
        assert "TOP-SECRET" not in str(exc.value)
        assert "Could not connect to configured LSL stream" in str(exc.value)


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_connect_returns_false_when_no_streams_found(mock_pylsl):
    mock_pylsl.resolve_byprop.return_value = []
    bridge = LSLBCIBridge()
    assert bridge.connect() is False
