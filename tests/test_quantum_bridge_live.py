# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Live quantum-control bridge tests

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from types import ModuleType

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.quantum_control_bridge import (
    QuantumControlBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

HAS_QC = importlib.util.find_spec("scpn_quantum_control") is not None


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


def _install_fake_quantum_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    orchestrator_to_quantum_phases=None,
    quantum_to_orchestrator_phases=None,
) -> None:
    module = ModuleType("scpn_quantum_control")
    if orchestrator_to_quantum_phases is not None:
        module.orchestrator_to_quantum_phases = orchestrator_to_quantum_phases
    if quantum_to_orchestrator_phases is not None:
        module.quantum_to_orchestrator_phases = quantum_to_orchestrator_phases
    monkeypatch.setitem(sys.modules, "scpn_quantum_control", module)


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
        knm = np.array([[0.0, 0.4, 0.1], [0.4, 0.0, 0.2], [0.1, 0.2, 0.0]])
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

    def test_export_artifact_rejects_non_upde_state(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="state must be a UPDEState"):
            bridge.export_artifact(object())

    def test_export_artifact_rejects_non_finite_layer_metrics(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = UPDEState(
            layers=[
                LayerState(R=float("nan"), psi=0.2),
            ],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.4,
            regime_id="NOMINAL",
        )
        with pytest.raises(ValueError, match="layers\\[0\\]\\.R must be finite"):
            bridge.export_artifact(state)

    def test_export_artifact_rejects_non_finite_cross_alignment(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        state = UPDEState(
            layers=[LayerState(R=1.0, psi=0.0)],
            cross_layer_alignment=np.array([[float("nan")]]),
            stability_proxy=0.4,
            regime_id="NOMINAL",
        )
        with pytest.raises(ValueError, match="must contain finite values"):
            bridge.export_artifact(state)

    def test_export_artifact_rejects_cross_layer_shape_mismatch(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        state = UPDEState(
            layers=[
                LayerState(R=1.0, psi=0.0),
                LayerState(R=1.0, psi=1.0),
            ],
            cross_layer_alignment=np.eye(3),
            stability_proxy=0.4,
            regime_id="NOMINAL",
        )
        with pytest.raises(ValueError, match="shape must match number of layers"):
            bridge.export_artifact(state)

    def test_build_quantum_compiler_manifest_has_reproducible_handoff_metadata(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        knm = np.array(
            [[0.0, 0.1, 0.0], [0.1, 0.0, 0.05], [0.0, 0.05, 0.0]],
            dtype=np.float64,
        )
        omegas = np.array([0.4, 0.5, 0.6], dtype=np.float64)
        manifest = bridge.build_quantum_compiler_manifest(knm, omegas, dt=0.2)
        assert manifest["manifest_kind"] == "quantum_compiler_manifest"
        assert manifest["schema_version"] == 1
        assert manifest["n_qubits"] == 3
        assert manifest["status"] == "co_simulation_parity_passed"
        assert manifest["qpu_execution_permitted"] is False
        assert manifest["actuation_permitted"] is False
        assert manifest["target_backends"] == ["qiskit_openqasm3", "pennylane_qasm"]
        assert manifest["co_simulation_parity"]["term_count"] == len(
            manifest["frequency_terms"] + manifest["coupling_terms"]
        )
        assert manifest["openqasm"].startswith("OPENQASM 3.0;")
        parity_projection = json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        )
        assert (
            manifest["manifest_sha256"]
            == sha256(parity_projection.encode("utf-8")).hexdigest()
        )


class TestQuantumBridgeDependencyShim:
    def test_orchestrator_to_quantum_honours_contract_with_fake_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured: dict[str, object] = {}

        def fake_orchestrator_to_quantum_phases(payload: object) -> np.ndarray:
            captured["payload"] = dict(payload)  # type: ignore[arg-type]
            return np.array([0.1, 0.2], dtype=np.float64)

        _install_fake_quantum_module(
            monkeypatch,
            orchestrator_to_quantum_phases=fake_orchestrator_to_quantum_phases,
        )

        bridge = QuantumControlBridge(n_oscillators=4)
        phases = bridge.orchestrator_to_quantum(_make_state())
        np.testing.assert_allclose(phases, np.array([0.1, 0.2], dtype=np.float64))
        assert captured["payload"] == {"layer_0": 0.5, "layer_1": 1.2}

    def test_orchestrator_to_quantum_contract_import_fails_if_contract_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _install_fake_quantum_module(monkeypatch)
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(ImportError, match="orchestrator_to_quantum_phases"):
            bridge.orchestrator_to_quantum(_make_state())

    def test_quantum_to_orchestrator_honours_contract_with_fake_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured: dict[str, object] = {}

        def fake_quantum_to_orchestrator_phases(payload: object) -> dict[str, object]:
            captured["payload"] = payload
            return {"phases": list(np.asarray(payload, dtype=np.float64))}

        _install_fake_quantum_module(
            monkeypatch,
            quantum_to_orchestrator_phases=fake_quantum_to_orchestrator_phases,
        )

        bridge = QuantumControlBridge(n_oscillators=4)
        payload = bridge.quantum_to_orchestrator(
            np.array([0.8, 1.1], dtype=np.float64),
        )
        assert payload["phases"] == [0.8, 1.1]
        np.testing.assert_allclose(captured["payload"], np.array([0.8, 1.1]))

    def test_quantum_to_orchestrator_contract_import_fails_if_contract_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _install_fake_quantum_module(monkeypatch)
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(ImportError, match="quantum_to_orchestrator_phases"):
            bridge.quantum_to_orchestrator(np.array([0.8, 1.1]))

    def test_build_hamiltonian_rejects_invalid_shapes_without_dependency(self):
        bridge = QuantumControlBridge(3)
        with pytest.raises(ValueError, match="does not match n_oscillators=3"):
            bridge.build_hamiltonian(np.ones((2, 2)), np.ones(3))

    def test_solve_q_upde_rejects_invalid_t_max_before_dependency_import(self):
        bridge = QuantumControlBridge(2)
        with pytest.raises(ValueError, match="t_max must be finite and positive"):
            bridge.solve_q_upde(np.ones((2, 2)), np.ones(2), t_max=0.0)


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
