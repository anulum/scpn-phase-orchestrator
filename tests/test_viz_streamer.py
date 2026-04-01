# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Visualizer Streamer tests

import numpy as np
import pytest
from scpn_phase_orchestrator.visualization.streamer import VisualizerStreamer

def test_viz_streamer_initialization():
    streamer = VisualizerStreamer(host="127.0.0.1", port=9999)
    assert streamer.host == "127.0.0.1"
    assert streamer.port == 9999

def test_json_safe_conversion():
    streamer = VisualizerStreamer()
    data = {
        "array": np.array([1.0, 2.0, 3.0]),
        "scalar": np.float64(4.2),
        "int": np.int64(7),
        "nested": {"val": np.array([0.0])}
    }
    safe = streamer._json_safe(data)
    assert isinstance(safe["array"], list)
    assert isinstance(safe["scalar"], float)
    assert isinstance(safe["int"], int)
    assert isinstance(safe["nested"]["val"], list)
