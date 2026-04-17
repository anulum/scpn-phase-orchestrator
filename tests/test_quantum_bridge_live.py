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


# The tests below exercise the adapter paths that do NOT require the
# scpn_quantum_control dependency — import_knm, import_artifact,
# export_artifact. They were previously gated on ``HAS_QC`` for the
# whole module, which skipped useful coverage when the optional
# dependency was absent. Only the orchestrator_to_quantum /
# quantum_to_orchestrator / solve_q_upde paths need HAS_QC.


class TestQuantumBridgeAdapterLocal:
    """Paths that do not reach into the scpn-quantum-control library."""

    def test_export_artifact_populates_all_fields(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        state = _make_state()
        d = bridge.export_artifact(state)
        assert set(d) >= {"regime", "fidelity", "layers", "cross_alignment"}
        assert d["fidelity"] == 0.8
        assert len(d["layers"]) == 2

    def test_export_import_roundtrip_local(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        state = _make_state()
        bridge.export_artifact(state)
        imported = bridge.import_artifact(
            {"phases": np.array([0.5, 1.2, 0.8, 0.3]), "fidelity": 0.95}
        )
        assert len(imported.layers) == 2
        assert imported.stability_proxy == 0.95

    def test_import_knm_preserves_values(self):
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

    def test_import_knm_initialises_zero_alpha(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        knm = np.array(
            [[0.0, 0.4, 0.1], [0.4, 0.0, 0.2], [0.1, 0.2, 0.0]]
        )
        coupling = bridge.import_knm(knm)
        assert np.all(coupling.alpha == 0.0)

    def test_import_knm_rejects_non_square(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        with pytest.raises(ValueError, match="Knm must be square"):
            bridge.import_knm(np.zeros((2, 4)))

    def test_import_artifact_defaults_when_fields_missing(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact({"phases": [0.0, np.pi]})
        assert state.stability_proxy == 0.0
        assert state.regime_id == "NOMINAL"


@pytest.mark.skipif(not HAS_QC, reason="scpn-quantum-control not installed")
class TestQuantumBridgeLive:
    """Paths that dispatch into the scpn_quantum_control library."""

    def test_orchestrator_to_quantum_roundtrip(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        state = _make_state()

        q_phases = bridge.orchestrator_to_quantum(state)
        assert isinstance(q_phases, np.ndarray)
        assert q_phases.ndim == 1

        result = bridge.quantum_to_orchestrator(q_phases)
        assert isinstance(result, dict)

    def test_build_hamiltonian_returns_object(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        knm = np.eye(3) * 0.0 + 0.2  # uniform non-zero off-diagonal
        np.fill_diagonal(knm, 0.0)
        omegas = np.ones(3)
        ham = bridge.build_hamiltonian(knm, omegas)
        assert ham is not None


# Pipeline wiring: the adapter-local tests above cover every code path
# that does not cross into scpn_quantum_control, so the file still
# contributes coverage in minimal CI environments. The live suite
# activates only when the optional dependency is present.
