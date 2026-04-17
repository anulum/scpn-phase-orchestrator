# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Live quantum-control bridge tests

from __future__ import annotations

import numpy as np
import pytest

try:
    import importlib.util

    HAS_QC = importlib.util.find_spec("scpn_quantum_control") is not None
except (ImportError, ModuleNotFoundError):
    HAS_QC = False

from scpn_phase_orchestrator.adapters.quantum_control_bridge import (
    QuantumControlBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

pytestmark = pytest.mark.skipif(not HAS_QC, reason="scpn-quantum-control not installed")


def _make_state() -> UPDEState:
    return UPDEState(
        layers=[
            LayerState(R=0.9, psi=0.5),
            LayerState(R=0.7, psi=1.2),
        ],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.8,
        regime_id="nominal",
    )


class TestQuantumBridgeLive:
    def test_orchestrator_to_quantum_roundtrip(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        state = _make_state()

        q_phases = bridge.orchestrator_to_quantum(state)
        assert isinstance(q_phases, np.ndarray)
        assert q_phases.ndim == 1

        result = bridge.quantum_to_orchestrator(q_phases)
        assert isinstance(result, dict)

    def test_export_import_roundtrip(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        state = _make_state()

        bridge.export_artifact(state)
        imported = bridge.import_artifact(
            {
                "phases": np.array([0.5, 1.2, 0.8, 0.3]),
                "fidelity": 0.95,
            }
        )
        assert len(imported.layers) == 2
        assert imported.stability_proxy == 0.95

    def test_import_knm(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        knm = np.array(
            [
                [0.0, 0.3, 0.1, 0.05],
                [0.3, 0.0, 0.2, 0.1],
                [0.1, 0.2, 0.0, 0.15],
                [0.05, 0.1, 0.15, 0.0],
            ]
        )
        coupling = bridge.import_knm(knm)
        assert coupling.knm.shape == (4, 4)
        np.testing.assert_allclose(coupling.knm, knm)


# Pipeline wiring: TestQuantumBridgeLive.test_import_knm_round_trip exercises
# QuantumControlBridge → import_knm → CouplingState — the cross-project
# quantum control bridge. Requires scpn_quantum_control installed.
