# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
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

@patch('scpn_phase_orchestrator.adapters.lsl_bci_bridge.HAS_LSL', True)
@patch('scpn_phase_orchestrator.adapters.lsl_bci_bridge.pylsl')
def test_lsl_connection(mock_pylsl):
    mock_pylsl.resolve_byprop.return_value = [MagicMock()]
    mock_pylsl.StreamInlet.return_value.info().nominal_srate.return_value = 250
    
    bridge = LSLBCIBridge()
    success = bridge.connect()
    
    assert success is True
    assert bridge.sampling_rate == 250
