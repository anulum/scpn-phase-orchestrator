# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Quantum Bridge tests

import numpy as np
import pytest
from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge

def test_quantum_bridge_import_artifact():
    bridge = QuantumControlBridge(n_oscillators=4)
    artifact = {
        "phases": [0.0, 0.1, 3.14, 3.2],
        "fidelity": 0.95,
        "regime": "COHERENT",
        "layer_assignments": [[0, 1], [2, 3]]
    }
    state = bridge.import_artifact(artifact)
    assert state.stability_proxy == 0.95
    assert state.regime_id == "COHERENT"
    assert len(state.layers) == 2
    assert state.layers[0].R > 0.9
    assert state.layers[1].R > 0.9

def test_quantum_solve():
    import importlib.util
    if importlib.util.find_spec("qiskit") is None:
        pytest.skip("Requires qiskit")
    if importlib.util.find_spec("scpn_quantum_control") is None:
        pytest.skip("Requires scpn-quantum-control")

    bridge = QuantumControlBridge(n_oscillators=4)
    knm = 0.5 * np.ones((4, 4))
    omegas = np.ones(4)
    result = bridge.solve_q_upde(knm, omegas, t_max=0.2, dt=0.1)
    assert "R" in result
    assert len(result["R"]) == 3 # 0.0, 0.1, 0.2
