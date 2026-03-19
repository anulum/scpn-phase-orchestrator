# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum-control bridge tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge

TWO_PI = 2.0 * np.pi


class TestConstructorValidation:
    def test_n_oscillators_zero_raises(self):
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            QuantumControlBridge(n_oscillators=0)


class TestQuantumControlBridge:
    def test_import_artifact_empty_layer_group(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        artifact = {
            "phases": [0.1, 0.2],
            "fidelity": 0.7,
            "layer_assignments": [[0, 1], []],
        }
        state = bridge.import_artifact(artifact)
        assert len(state.layers) == 2
        assert pytest.approx(0.0) == state.layers[1].R

    def test_import_export_roundtrip(self):
        bridge = QuantumControlBridge(n_oscillators=8)
        phases = np.linspace(0, TWO_PI, 8, endpoint=False)
        artifact = {
            "phases": phases.tolist(),
            "fidelity": 0.95,
            "regime": "NOMINAL",
            "layer_assignments": [[0, 1, 2, 3], [4, 5, 6, 7]],
        }
        state = bridge.import_artifact(artifact)
        assert len(state.layers) == 2
        assert state.stability_proxy == pytest.approx(0.95)
        assert state.regime_id == "NOMINAL"

        exported = bridge.export_artifact(state)
        assert exported["fidelity"] == pytest.approx(0.95)
        assert exported["regime"] == "NOMINAL"
        assert len(exported["layers"]) == 2

    def test_import_default_layer_split(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        artifact = {"phases": [0.0, 0.1, 0.2, 0.3], "fidelity": 0.8}
        state = bridge.import_artifact(artifact)
        assert len(state.layers) == 2
        assert 0.0 <= state.layers[0].R <= 1.0
        assert 0.0 <= state.layers[1].R <= 1.0

    def test_import_knm(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        knm = np.eye(4) * 0.5
        np.fill_diagonal(knm, 0.0)
        coupling = bridge.import_knm(knm)
        assert coupling.active_template == "quantum_import"
        assert coupling.knm.shape == (4, 4)
        assert coupling.alpha.shape == (4, 4)

    def test_import_knm_rejects_non_square(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm(np.zeros((3, 4)))

    def test_phases_wrapped_to_two_pi(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        artifact = {"phases": [7.0, -1.0, 0.0, 3.0], "fidelity": 0.5}
        state = bridge.import_artifact(artifact)
        for ls in state.layers:
            assert 0.0 <= ls.psi < TWO_PI

    def test_import_knm_square(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        knm = np.diag([0.0, 0.0, 0.0, 0.0]) + 0.1
        np.fill_diagonal(knm, 0.0)
        cs = bridge.import_knm(knm)
        assert cs.knm.shape == (4, 4)
        assert cs.active_template == "quantum_import"

    def test_import_knm_non_square_error(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm(np.ones((3, 4)))


class TestQuantumDomainpack:
    def test_binding_spec_loads(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "quantum_simulation" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        assert spec.name == "quantum_simulation"
        assert len(spec.layers) == 2
        n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        assert n_osc == 8

    def test_binding_spec_validates(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding import validate_binding_spec
        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "quantum_simulation" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        errors = validate_binding_spec(spec)
        assert errors == [], f"Validation errors: {errors}"
