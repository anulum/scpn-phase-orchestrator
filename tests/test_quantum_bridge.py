# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum Bridge tests

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge


def test_quantum_bridge_import_artifact():
    bridge = QuantumControlBridge(n_oscillators=4)
    artifact = {
        "phases": [0.0, 0.1, 3.14, 3.2],
        "fidelity": 0.95,
        "regime": "COHERENT",
        "layer_assignments": [[0, 1], [2, 3]],
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
    assert len(result["R"]) == 3  # 0.0, 0.1, 0.2


class TestConstructorValidation:
    def test_rejects_zero_oscillators(self):
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            QuantumControlBridge(n_oscillators=0)

    def test_rejects_negative_oscillators(self):
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            QuantumControlBridge(n_oscillators=-3)

    def test_default_trotter_order_is_one(self):
        b = QuantumControlBridge(n_oscillators=2)
        assert b._trotter_order == 1

    def test_custom_trotter_order_propagates(self):
        b = QuantumControlBridge(n_oscillators=2, trotter_order=4)
        assert b._trotter_order == 4


class TestImportArtifactEdges:
    def test_default_layer_split_when_assignments_missing(self):
        """No layer_assignments → artefact is bisected at the midpoint."""
        bridge = QuantumControlBridge(n_oscillators=4)
        state = bridge.import_artifact(
            {"phases": [0.0, 0.1, 0.2, 0.3], "fidelity": 0.5, "regime": "DEGRADED"}
        )
        assert len(state.layers) == 2

    def test_missing_fidelity_defaults_to_zero(self):
        """``fidelity`` key is optional; absent → stability_proxy = 0."""
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact({"phases": [0.0, np.pi], "regime": "NOMINAL"})
        assert state.stability_proxy == 0.0

    def test_missing_regime_defaults_to_nominal(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact({"phases": [0.0, 0.1], "fidelity": 0.3})
        assert state.regime_id == "NOMINAL"

    def test_empty_layer_group_gives_zero_order(self):
        """A layer with no oscillator indices must produce R = 0, ψ = 0."""
        bridge = QuantumControlBridge(n_oscillators=4)
        state = bridge.import_artifact(
            {
                "phases": [0.0, 0.1, 0.2, 0.3],
                "fidelity": 1.0,
                "regime": "NOMINAL",
                "layer_assignments": [[0, 1, 2, 3], []],
            }
        )
        assert state.layers[1].R == 0.0
        assert state.layers[1].psi == 0.0

    def test_antiphase_group_has_low_order(self):
        """Two oscillators at opposite phases → R ≈ 0."""
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact(
            {
                "phases": [0.0, np.pi],
                "fidelity": 0.1,
                "regime": "CRITICAL",
                "layer_assignments": [[0, 1]],
            }
        )
        assert state.layers[0].R < 1e-10

    def test_phases_wrapped_modulo_two_pi(self):
        """Input phases outside [0, 2π) must be wrapped on import."""
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact(
            {
                "phases": [2 * np.pi, 4 * np.pi + 0.3],
                "fidelity": 0.8,
                "regime": "NOMINAL",
                "layer_assignments": [[0, 1]],
            }
        )
        # Both phases wrap to ~0.0 and ~0.3 → still highly coherent
        assert state.layers[0].R > 0.9


class TestImportKnm:
    def test_square_matrix_accepted(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        knm = 0.2 * np.ones((3, 3))
        state = bridge.import_knm(knm)
        assert state.knm.shape == (3, 3)
        assert state.active_template == "quantum_import"

    def test_rejects_non_square_matrix(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        with pytest.raises(ValueError, match="Knm must be square"):
            bridge.import_knm(np.zeros((2, 3)))

    def test_rejects_one_dimensional_input(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        with pytest.raises(ValueError, match="Knm must be square"):
            bridge.import_knm(np.arange(9).astype(np.float64))

    def test_alpha_initialised_to_zero(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_knm(np.array([[0.0, 0.4], [0.4, 0.0]]))
        assert np.all(state.alpha == 0.0)


class TestExportArtifact:
    def test_round_trip_preserves_fields(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        original = {
            "phases": [0.1, 0.2],
            "fidelity": 0.73,
            "regime": "RECOVERY",
            "layer_assignments": [[0], [1]],
        }
        state = bridge.import_artifact(original)
        exported = bridge.export_artifact(state)
        assert exported["fidelity"] == 0.73
        assert exported["regime"] == "RECOVERY"
        assert len(exported["layers"]) == 2
        assert "cross_alignment" in exported


# Pipeline wiring: the quantum bridge is how classical SPO state crosses
# into scpn-quantum-control and back. Every case above guards a field
# that downstream consumers rely on — the constructor, the import_artifact
# contract with its defaults, knm validation, and the export round trip.
