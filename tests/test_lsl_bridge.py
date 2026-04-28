# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - LSL Bridge tests

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.lsl_bci_bridge import LSLBCIBridge


def test_lsl_bridge_initialization():
    bridge = LSLBCIBridge(stream_name="TestEEG", target_channel=1)
    assert bridge.stream_name == "TestEEG"
    assert bridge.target_channel == 1


def test_phase_extraction_logic():
    bridge = LSLBCIBridge()
    # Mock some data: a simple sine wave at 10Hz
    fs = 250
    t = np.linspace(0, 1, fs)
    freq = 10
    signal = np.sin(2 * np.pi * freq * t)

    bridge._data_buffer = list(signal)

    phase = bridge.get_instantaneous_phase()
    assert 0 <= phase < 2 * np.pi


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_lsl_connection(mock_pylsl):
    mock_pylsl.resolve_byprop.return_value = [MagicMock()]
    mock_pylsl.StreamInlet.return_value.info().nominal_srate.return_value = 250

    bridge = LSLBCIBridge()
    success = bridge.connect()

    assert success is True
    assert bridge.sampling_rate == 250


def test_default_constructor_values():
    """Default constructor produces stream_name='EEG', channel=0, 2s buffer."""
    bridge = LSLBCIBridge()
    assert bridge.stream_name == "EEG"
    assert bridge.target_channel == 0
    assert bridge.buffer_size_s == 2.0


def test_custom_constructor_values_propagate():
    """All three constructor arguments are honoured."""
    bridge = LSLBCIBridge(
        stream_name="AlphaStream", target_channel=3, buffer_size_s=5.0
    )
    assert bridge.stream_name == "AlphaStream"
    assert bridge.target_channel == 3
    assert bridge.buffer_size_s == 5.0


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", False)
def test_connect_returns_false_when_lsl_missing():
    """Without pylsl available, connect() returns False rather than raising."""
    bridge = LSLBCIBridge()
    assert bridge.connect() is False


@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", True)
@patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl")
def test_connect_returns_false_when_no_streams_found(mock_pylsl):
    """resolve_byprop returning empty list → connect() is False."""
    mock_pylsl.resolve_byprop.return_value = []
    bridge = LSLBCIBridge()
    assert bridge.connect() is False


def test_stop_without_start_is_idempotent():
    """Calling stop() before start() must not crash (T4 resource hygiene)."""
    bridge = LSLBCIBridge()
    bridge.stop()
    assert bridge._running is False
    # Second stop — still safe.
    bridge.stop()
    assert bridge._running is False


def test_phase_extraction_synchronous_signal():
    """Pure sine at 5 Hz should produce a well-defined phase in [0, 2π)."""
    bridge = LSLBCIBridge(buffer_size_s=1.0)
    fs = 250
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    bridge._data_buffer = list(signal)

    phase = bridge.get_instantaneous_phase()
    assert 0 <= phase < 2 * np.pi
    assert np.isfinite(phase)


def test_phase_extraction_handles_noisy_signal():
    """Additive Gaussian noise should not break phase extraction — the
    Hilbert transform is tolerant to noise."""
    rng = np.random.default_rng(3)
    bridge = LSLBCIBridge()
    fs = 250
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 8 * t) + 0.3 * rng.standard_normal(fs)
    bridge._data_buffer = list(signal)

    phase = bridge.get_instantaneous_phase()
    assert 0 <= phase < 2 * np.pi
    assert np.isfinite(phase)


def test_phase_extraction_empty_buffer_is_handled():
    """Empty buffer must not crash — contract returns a finite default."""
    bridge = LSLBCIBridge()
    bridge._data_buffer = []
    phase = bridge.get_instantaneous_phase()
    assert np.isfinite(phase)
    assert 0 <= phase < 2 * np.pi


def test_start_without_lsl_raises_scrubbed_message():
    """start() when pylsl is absent must raise RuntimeError without leaking
    the configured stream_name (T3 scrub regression)."""
    bridge = LSLBCIBridge(stream_name="OPERATOR_IDENT")
    with (
        patch("scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL", False),
        pytest.raises(RuntimeError) as exc,
    ):
        bridge.start()
    assert "OPERATOR_IDENT" not in str(exc.value)
    assert "Could not connect" in str(exc.value)
