# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum-control bridge tests

from __future__ import annotations

import builtins
import importlib.util
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

_BRIDGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "scpn_phase_orchestrator"
    / "adapters"
    / "quantum_control_bridge.py"
)
_BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "scpn_phase_orchestrator.adapters.quantum_control_bridge",
    _BRIDGE_PATH,
)
assert _BRIDGE_SPEC is not None
assert _BRIDGE_SPEC.loader is not None
_BRIDGE_MODULE = importlib.util.module_from_spec(_BRIDGE_SPEC)
_BRIDGE_SPEC.loader.exec_module(_BRIDGE_MODULE)
QuantumControlBridge = _BRIDGE_MODULE.QuantumControlBridge

TWO_PI = 2.0 * np.pi


def _canonical_digest(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(text.encode("utf-8")).hexdigest()


def _numeric_digest(value: object, *, name: str) -> str:
    return _canonical_digest({"name": name, "value": value})


def _scpn_upde_edge_payload() -> dict[str, object]:
    knm = [[0.0, 0.25], [0.25, 0.0]]
    omega = [1.0, -0.5]
    payload: dict[str, object] = {
        "schema": "knm.scpn-upde.v1",
        "producer": "scpn-quantum-control",
        "consumer": "scpn-phase-orchestrator",
        "scope_envelope": "computational-agreement",
        "claim_boundary": "Paper-27 provisional; computational agreement only.",
        "n_oscillators": 2,
        "K_nm": knm,
        "omega": omega,
        "trotter": {"time": 0.1, "steps": 1, "order": 1, "dt": 0.1},
        "compiler": {
            "kind": "qiskit-pauli-evolution",
            "num_qubits": 2,
            "depth": 1,
            "operation_counts": {"PauliEvolution": 1},
        },
        "permissions": {
            "qpu_execution_permitted": False,
            "actuation_permitted": False,
        },
        "digests": {
            "K_nm_sha256": _numeric_digest(knm, name="K_nm"),
            "omega_sha256": _numeric_digest(omega, name="omega"),
        },
    }
    payload["edge_sha256"] = _canonical_digest(payload)
    return payload


class TestConstructorValidation:
    def test_n_oscillators_zero_raises(self):
        with pytest.raises(ValueError, match="n_oscillators"):
            QuantumControlBridge(n_oscillators=0)

    @pytest.mark.parametrize("n_oscillators", [True, 1.5])
    def test_n_oscillators_must_be_positive_integer(self, n_oscillators):
        with pytest.raises(ValueError, match="n_oscillators"):
            QuantumControlBridge(n_oscillators=n_oscillators)

    @pytest.mark.parametrize("trotter_order", [True, 0, 1.5])
    def test_trotter_order_must_be_positive_integer(self, trotter_order):
        with pytest.raises(ValueError, match="trotter_order must be an integer >= 1"):
            QuantumControlBridge(n_oscillators=2, trotter_order=trotter_order)


class TestQuantumControlBridge:
    def test_import_artifact_empty_layer_group(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        artifact = {
            "phases": [0.1, 0.2, 0.3, 0.4],
            "fidelity": 0.7,
            "layer_assignments": [[0, 1, 2, 3], []],
        }
        state = bridge.import_artifact(artifact)
        assert len(state.layers) == 2
        assert pytest.approx(0.0) == state.layers[1].R

    @pytest.mark.parametrize(
        ("artifact", "field"),
        [
            ({"phases": [0.0, np.nan], "fidelity": 0.7}, "phases"),
            ({"phases": [[0.0, 0.1]], "fidelity": 0.7}, "phases"),
            ({"phases": [0.0], "fidelity": 0.7}, "phases"),
            ({"phases": [0.0, 0.1], "fidelity": np.inf}, "fidelity"),
            (
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.7,
                    "layer_assignments": [[0, 2]],
                },
                "layer_assignments",
            ),
            (
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.7,
                    "layer_assignments": [[True]],
                },
                "layer_assignments",
            ),
            (
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.7,
                    "layer_assignments": [[0], [0]],
                },
                "layer_assignments",
            ),
        ],
    )
    def test_import_artifact_rejects_invalid_payload(self, artifact, field):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match=field):
            bridge.import_artifact(artifact)

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

    def test_import_knm_rejects_size_mismatch_and_nonfinite_values(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(ValueError, match="n_oscillators"):
            bridge.import_knm(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="finite"):
            bridge.import_knm(np.array([[0.0, np.nan], [0.0, 0.0]]))

    def test_import_knm_copies_input_matrix(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        coupling = bridge.import_knm(knm)
        knm[0, 1] = 99.0

        assert coupling.knm[0, 1] == 0.5

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

    def test_quantum_compiler_manifest_emits_qasm_with_parity_evidence(self):
        bridge = QuantumControlBridge(n_oscillators=2, trotter_order=2)
        knm = np.array([[0.0, 0.25], [0.5, 0.0]])
        omegas = np.array([1.0, -0.5])

        manifest = bridge.build_quantum_compiler_manifest(knm, omegas, dt=0.125)
        repeated = bridge.build_quantum_compiler_manifest(knm, omegas, dt=0.125)

        assert manifest == repeated
        assert manifest["manifest_kind"] == "quantum_compiler_manifest"
        assert manifest["schema_version"] == 1
        assert manifest["status"] == "co_simulation_parity_passed"
        assert manifest["target_backends"] == ["qiskit_openqasm3", "pennylane_qasm"]
        assert manifest["n_qubits"] == 2
        assert manifest["trotter_order"] == 2
        assert manifest["qpu_execution_permitted"] is False
        assert manifest["actuation_permitted"] is False
        assert len(manifest["qasm_sha256"]) == 64
        assert "OPENQASM 3.0;" in manifest["openqasm"]
        assert "qubit[2] q;" in manifest["openqasm"]
        assert "rz(0.125000000000) q[0];" in manifest["openqasm"]
        assert "rz(-0.062500000000) q[1];" in manifest["openqasm"]
        assert "rxx(0.046875000000) q[0], q[1];" in manifest["openqasm"]
        assert "ryy(0.046875000000) q[0], q[1];" in manifest["openqasm"]
        assert manifest["co_simulation_parity"] == {
            "engine": "deterministic_xy_term_reconstruction",
            "max_abs_frequency_error": 0.0,
            "max_abs_coupling_error": 0.0,
            "term_count": 3,
        }
        assert manifest["operator_commands"] == [
            "review quantum_compiler_manifest.json",
            "run Qiskit or PennyLane simulator parity before QPU handoff",
        ]

    def test_quantum_compiler_manifest_rejects_invalid_inputs(self):
        bridge = QuantumControlBridge(n_oscillators=2)

        with pytest.raises(ValueError, match="knm shape"):
            bridge.build_quantum_compiler_manifest(
                np.zeros((3, 3)),
                np.ones(2),
                dt=0.1,
            )
        with pytest.raises(ValueError, match="finite"):
            bridge.build_quantum_compiler_manifest(
                np.array([[0.0, np.nan], [0.0, 0.0]]),
                np.ones(2),
                dt=0.1,
            )
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            bridge.build_quantum_compiler_manifest(
                np.zeros((2, 2)),
                np.ones(2),
                dt=0.0,
            )

    def test_quantum_compiler_manifest_rejects_omega_shape_and_nonfinite_values(self):
        bridge = QuantumControlBridge(n_oscillators=2)

        with pytest.raises(ValueError, match="omegas shape"):
            bridge.build_quantum_compiler_manifest(
                np.zeros((2, 2)),
                np.ones(3),
                dt=0.1,
            )
        with pytest.raises(ValueError, match="omegas must contain finite values"):
            bridge.build_quantum_compiler_manifest(
                np.zeros((2, 2)),
                np.array([1.0, np.inf]),
                dt=0.1,
            )

    def test_quantum_compiler_manifest_does_not_mutate_numeric_inputs(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.0, 0.25], [0.5, 0.0]])
        omegas = np.array([1.0, -0.5])
        expected_knm = knm.copy()
        expected_omegas = omegas.copy()

        manifest = bridge.build_quantum_compiler_manifest(knm, omegas, dt=0.125)

        np.testing.assert_allclose(knm, expected_knm)
        np.testing.assert_allclose(omegas, expected_omegas)
        assert manifest["qpu_execution_permitted"] is False
        assert manifest["actuation_permitted"] is False

    def test_build_hamiltonian_rejects_invalid_inputs_before_backend_import(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="knm shape"):
            bridge.build_hamiltonian(np.zeros((3, 3)), np.ones(2))

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"t_max": 0.0, "dt": 0.1, "trotter_per_step": 1}, "t_max"),
            ({"t_max": 1.0, "dt": 0.0, "trotter_per_step": 1}, "dt"),
            ({"t_max": 1.0, "dt": 0.1, "trotter_per_step": 0}, "trotter_per_step"),
            ({"t_max": 1.0, "dt": 0.1, "trotter_per_step": True}, "trotter_per_step"),
        ],
    )
    def test_solve_q_upde_rejects_invalid_runtime_config_before_backend_import(
        self,
        kwargs,
        field,
    ):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match=field):
            bridge.solve_q_upde(np.zeros((2, 2)), np.ones(2), **kwargs)

    def test_quantum_compiler_manifest_omits_zero_and_cancelling_couplings(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        knm = np.array(
            [
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.3],
                [-0.2, 0.5, 0.0],
            ],
            dtype=np.float64,
        )

        manifest = bridge.build_quantum_compiler_manifest(knm, np.zeros(3), dt=0.25)

        assert manifest["coupling_terms"] == [
            {
                "source": 1,
                "target": 2,
                "forward_coupling": 0.3,
                "reverse_coupling": 0.5,
                "symmetric_coupling": 0.4,
                "xx_angle": pytest.approx(0.1),
                "yy_angle": pytest.approx(0.1),
            }
        ]
        assert "q[0], q[1]" not in manifest["openqasm"]
        assert "q[0], q[2]" not in manifest["openqasm"]
        assert "rxx(0.100000000000) q[1], q[2];" in manifest["openqasm"]

    def test_compiler_term_validation_rejects_nonnumeric_qasm_angle(self):
        bridge = QuantumControlBridge(n_oscillators=1)

        with pytest.raises(ValueError, match="QASM angle must be numeric"):
            bridge._render_openqasm([{"angle": object(), "qubit": 0}], [])

    def test_compiler_parity_rejects_noninteger_frequency_qubit(self):
        bridge = QuantumControlBridge(n_oscillators=2)

        with pytest.raises(ValueError, match="qubit must be an integer"):
            bridge._quantum_compiler_parity(
                np.zeros(2),
                [{"qubit": True, "omega": 0.0}],
                np.zeros((2, 2)),
                [],
            )

    def test_compiler_parity_rejects_nonnumeric_coupling_field(self):
        bridge = QuantumControlBridge(n_oscillators=2)

        with pytest.raises(ValueError, match="symmetric_coupling must be numeric"):
            bridge._quantum_compiler_parity(
                np.zeros(2),
                [{"qubit": 0, "omega": 0.0}, {"qubit": 1, "omega": 0.0}],
                np.zeros((2, 2)),
                [{"source": 0, "target": 1, "symmetric_coupling": False}],
            )

    def test_manifest_reports_failed_parity_when_terms_are_mutated(
        self,
        monkeypatch,
    ):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.0, 0.4], [0.2, 0.0]])
        omegas = np.array([1.0, -0.5])

        def tampered_coupling_terms(*, dt: float) -> list[dict[str, object]]:
            return [
                {
                    "source": 0,
                    "target": 1,
                    "forward_coupling": 0.0,
                    "reverse_coupling": 0.0,
                    "symmetric_coupling": 0.0,
                    "xx_angle": 0.4 * dt,
                    "yy_angle": 0.4 * dt,
                }
            ]

        monkeypatch.setattr(
            bridge,
            "_quantum_coupling_terms",
            lambda knm, dt: tampered_coupling_terms(dt=dt),
        )

        manifest = bridge.build_quantum_compiler_manifest(
            knm,
            omegas,
            dt=0.1,
        )

        assert manifest["status"] == "co_simulation_parity_failed"
        assert manifest["co_simulation_parity"]["max_abs_coupling_error"] > 0.0
        assert manifest["co_simulation_parity"]["term_count"] == 3

    def test_qpu_target_readiness_passes_only_when_preconditions_met(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            np.array([[0.0, 0.4], [0.0, 0.0]]),
            np.array([1.0, -0.5]),
            dt=0.1,
        )

        ready = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="local",
            credentials_configured=True,
            operator_approved=True,
        )

        assert ready["status"] == "ready_not_executed"
        assert ready["blocked_reasons"] == []
        assert ready["qpu_execution_permitted"] is False
        assert ready["actuation_permitted"] is False
        assert ready["operator_approved"] is True
        assert ready["credentials_configured"] is True
        assert len(ready["readiness_sha256"]) == 64

    def test_audit_qpu_target_readiness_reports_blocking_reasons(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            np.array([[0.0, 0.4], [0.0, 0.0]]),
            np.array([1.0, -0.5]),
            dt=0.1,
        )

        manifest["status"] = "co_simulation_parity_failed"

        blocked = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="local",
            credentials_configured=False,
            operator_approved=False,
        )

        assert blocked["status"] == "blocked"
        assert set(blocked["blocked_reasons"]) == {
            "co_simulation_parity_not_passed",
            "credentials_not_configured",
            "operator_approval_missing",
        }

    def test_audit_qpu_target_readiness_rejects_manifest_validation_errors(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = {
            "manifest_kind": "quantum_compiler_manifest",
            "status": "co_simulation_parity_passed",
            "target_backends": "qiskit_openqasm3",
            "qpu_execution_permitted": False,
            "actuation_permitted": False,
            "manifest_sha256": "a" * 64,
        }

        with pytest.raises(ValueError, match="target_backends must be a list"):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="qiskit_openqasm3",
                provider="local",
            )

        with pytest.raises(ValueError, match="target_backend"):
            bridge.audit_qpu_target_readiness(
                {
                    "manifest_kind": "quantum_compiler_manifest",
                    "status": "co_simulation_parity_passed",
                    "target_backends": ["qiskit_openqasm3"],
                    "qpu_execution_permitted": False,
                    "actuation_permitted": False,
                    "manifest_sha256": "a" * 64,
                },
                target_backend="missing_backend",
                provider="local",
                credentials_configured=True,
                operator_approved=True,
            )

    def test_import_scpn_upde_edge_accepts_computational_agreement_payload(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        payload = _scpn_upde_edge_payload()

        record = bridge.import_scpn_upde_edge(payload)

        assert record["schema"] == "spo.quantum-control.scpn-upde-import.v1"
        assert record["status"] == "accepted_computational_agreement"
        assert record["accepted_schema"] == "knm.scpn-upde.v1"
        assert record["scope_envelope"] == "computational-agreement"
        assert record["edge_sha256"] == payload["edge_sha256"]
        assert record["qpu_execution_permitted"] is False
        assert record["actuation_permitted"] is False
        assert isinstance(record["import_sha256"], str)
        assert len(record["import_sha256"]) == 64
        coupling = cast(CouplingState, record["coupling_state"])
        assert coupling.active_template == "quantum_import"
        np.testing.assert_allclose(coupling.knm, np.array(payload["K_nm"]))
        manifest = cast(dict[str, object], record["compiler_manifest"])
        assert manifest["manifest_kind"] == "quantum_compiler_manifest"
        assert manifest["status"] == "co_simulation_parity_passed"
        assert manifest["qpu_execution_permitted"] is False
        assert manifest["actuation_permitted"] is False

    def test_import_scpn_upde_edge_rejects_broader_scope(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        payload = _scpn_upde_edge_payload()
        payload["scope_envelope"] = "physical-validation"

        with pytest.raises(ValueError, match="scope_envelope"):
            bridge.import_scpn_upde_edge(payload)

    def test_import_scpn_upde_edge_rejects_execution_permission(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        payload = _scpn_upde_edge_payload()
        cast(dict[str, object], payload["permissions"])["actuation_permitted"] = True

        with pytest.raises(ValueError, match="actuation_permitted"):
            bridge.import_scpn_upde_edge(payload)

    def test_import_scpn_upde_edge_rejects_tampered_knm_digest(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        payload = _scpn_upde_edge_payload()
        cast(list[list[float]], payload["K_nm"])[0][1] += 0.01

        with pytest.raises(ValueError, match="K_nm_sha256"):
            bridge.import_scpn_upde_edge(payload)

    def test_import_scpn_upde_edge_rejects_bridge_size_mismatch(self):
        bridge = QuantumControlBridge(n_oscillators=3)

        with pytest.raises(ValueError, match="n_oscillators"):
            bridge.import_scpn_upde_edge(_scpn_upde_edge_payload())


class TestExportArtifactValidation:
    def test_rejects_non_upde_state(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="state must be a UPDEState"):
            bridge.export_artifact(cast(Any, {"layers": []}))

    def test_rejects_non_list_layers(self):
        bridge = QuantumControlBridge(n_oscillators=1)
        state = UPDEState(
            layers=cast(Any, (LayerState(R=1.0, psi=0.0),)),
            cross_layer_alignment=np.eye(1, dtype=np.float64),
            stability_proxy=0.2,
            regime_id="SINGLET",
        )
        with pytest.raises(ValueError, match="UPDEState.layers must be a list"):
            bridge.export_artifact(state)

    def test_rejects_non_layer_state_entry(self):
        bridge = QuantumControlBridge(n_oscillators=1)
        state = UPDEState(
            layers=cast(Any, [{"R": 0.5, "psi": 0.0}]),
            cross_layer_alignment=np.eye(1, dtype=np.float64),
            stability_proxy=0.2,
            regime_id="SINGLET",
        )
        expected_error = "UPDEState.layers[0] must be a LayerState"
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            bridge.export_artifact(state)

    def test_rejects_non_square_cross_layer_alignment(self):
        bridge = QuantumControlBridge(n_oscillators=1)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.array([1.0]),
            stability_proxy=0.2,
            regime_id="SINGLET",
        )
        with pytest.raises(
            ValueError, match="cross_layer_alignment must be a square matrix"
        ):
            bridge.export_artifact(state)

    def test_rejects_cross_layer_alignment_shape_mismatch(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0), LayerState(R=0.6, psi=1.0)],
            cross_layer_alignment=np.eye(1, dtype=np.float64),
            stability_proxy=0.2,
            regime_id="BILAYER",
        )
        with pytest.raises(
            ValueError,
            match="cross_layer_alignment shape must match number of layers",
        ):
            bridge.export_artifact(state)

    def test_rejects_out_of_range_stability_proxy_as_exported_fidelity(self):
        bridge = QuantumControlBridge(n_oscillators=1)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1, dtype=np.float64),
            stability_proxy=1.5,
            regime_id="SINGLET",
        )

        with pytest.raises(ValueError, match="state.stability_proxy"):
            bridge.export_artifact(state)

    def test_live_phase_export_validates_state_before_optional_backend_import(
        self,
        monkeypatch,
    ):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0), LayerState(R=0.6, psi=1.0)],
            cross_layer_alignment=np.eye(1, dtype=np.float64),
            stability_proxy=0.2,
            regime_id="BILAYER",
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("scpn_quantum_control"):
                raise AssertionError("optional backend import attempted")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(ValueError, match="cross_layer_alignment shape"):
            bridge.orchestrator_to_quantum(state)

    def test_live_phase_import_validates_quantum_theta_before_optional_backend_import(
        self,
        monkeypatch,
    ):
        bridge = QuantumControlBridge(n_oscillators=2)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith("scpn_quantum_control"):
                raise AssertionError("optional backend import attempted")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(ValueError, match="quantum_theta must contain finite"):
            bridge.quantum_to_orchestrator(np.array([0.0, np.nan]))

        with pytest.raises(ValueError, match="quantum_theta shape"):
            bridge.quantum_to_orchestrator(np.array([0.0, 1.0, 2.0]))


class TestOptionalQuantumControlBackend:
    def test_build_hamiltonian_delegates_to_optional_backend(self, monkeypatch):
        root = ModuleType("scpn_quantum_control")
        bridge_pkg = ModuleType("scpn_quantum_control.bridge")
        hamiltonian_mod = ModuleType("scpn_quantum_control.bridge.knm_hamiltonian")
        calls = []

        def knm_to_hamiltonian(knm, omegas):
            calls.append((knm.copy(), omegas.copy()))
            return {"hamiltonian": "xy", "n": len(omegas)}

        hamiltonian_mod.knm_to_hamiltonian = knm_to_hamiltonian
        monkeypatch.setitem(sys.modules, "scpn_quantum_control", root)
        monkeypatch.setitem(sys.modules, "scpn_quantum_control.bridge", bridge_pkg)
        monkeypatch.setitem(
            sys.modules,
            "scpn_quantum_control.bridge.knm_hamiltonian",
            hamiltonian_mod,
        )
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.0, 0.4], [0.4, 0.0]])
        omegas = np.array([1.0, -1.0])

        hamiltonian = bridge.build_hamiltonian(knm, omegas)

        assert hamiltonian == {"hamiltonian": "xy", "n": 2}
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0][0], knm)
        np.testing.assert_allclose(calls[0][1], omegas)

    def test_solve_q_upde_passes_physical_payload_to_optional_solver(self, monkeypatch):
        root = ModuleType("scpn_quantum_control")
        phase_pkg = ModuleType("scpn_quantum_control.phase")
        solver_mod = ModuleType("scpn_quantum_control.phase.xy_kuramoto")
        calls = {}

        class QuantumKuramotoSolver:
            def __init__(
                self, *, n_oscillators, K_coupling, omega_natural, trotter_order
            ):
                calls["init"] = {
                    "n_oscillators": n_oscillators,
                    "K_coupling": K_coupling.copy(),
                    "omega_natural": omega_natural.copy(),
                    "trotter_order": trotter_order,
                }

            def run(self, *, t_max, dt, trotter_per_step):
                calls["run"] = {
                    "t_max": t_max,
                    "dt": dt,
                    "trotter_per_step": trotter_per_step,
                }
                return {"phases": [0.0, 0.5], "fidelity": 0.99}

        solver_mod.QuantumKuramotoSolver = QuantumKuramotoSolver
        monkeypatch.setitem(sys.modules, "scpn_quantum_control", root)
        monkeypatch.setitem(sys.modules, "scpn_quantum_control.phase", phase_pkg)
        monkeypatch.setitem(
            sys.modules,
            "scpn_quantum_control.phase.xy_kuramoto",
            solver_mod,
        )
        bridge = QuantumControlBridge(n_oscillators=2, trotter_order=3)
        knm = np.array([[0.0, 0.25], [0.25, 0.0]])
        omegas = np.array([0.5, -0.25])

        result = bridge.solve_q_upde(
            knm,
            omegas,
            t_max=2.0,
            dt=0.2,
            trotter_per_step=7,
        )

        assert result == {"phases": [0.0, 0.5], "fidelity": 0.99}
        assert calls["init"]["n_oscillators"] == 2
        assert calls["init"]["trotter_order"] == 3
        np.testing.assert_allclose(calls["init"]["K_coupling"], knm)
        np.testing.assert_allclose(calls["init"]["omega_natural"], omegas)
        assert calls["run"] == {"t_max": 2.0, "dt": 0.2, "trotter_per_step": 7}

    def test_phase_payload_converters_delegate_to_optional_backend(self, monkeypatch):
        root = ModuleType("scpn_quantum_control")
        calls = {}

        def orchestrator_to_quantum_phases(layer_phases):
            calls["to_quantum"] = dict(layer_phases)
            return np.array([layer_phases["layer_0"], layer_phases["layer_1"]])

        def quantum_to_orchestrator_phases(theta):
            calls["to_orchestrator"] = theta.copy()
            return {"layer_0": float(theta[0]), "layer_1": float(theta[1])}

        root.orchestrator_to_quantum_phases = orchestrator_to_quantum_phases
        root.quantum_to_orchestrator_phases = quantum_to_orchestrator_phases
        monkeypatch.setitem(sys.modules, "scpn_quantum_control", root)
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact(
            {
                "phases": [0.25, 1.25],
                "fidelity": 0.8,
                "layer_assignments": [[0], [1]],
            }
        )

        quantum_theta = bridge.orchestrator_to_quantum(state)
        roundtrip = bridge.quantum_to_orchestrator(quantum_theta)

        assert calls["to_quantum"] == {
            "layer_0": pytest.approx(0.25),
            "layer_1": pytest.approx(1.25),
        }
        np.testing.assert_allclose(calls["to_orchestrator"], quantum_theta)
        np.testing.assert_allclose(quantum_theta, [0.25, 1.25])
        assert roundtrip == {"layer_0": 0.25, "layer_1": 1.25}

    def test_phase_converters_require_optional_quantum_backend(self, monkeypatch):
        root = ModuleType("scpn_quantum_control")
        monkeypatch.setitem(sys.modules, "scpn_quantum_control", root)
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_artifact({"phases": [0.25, 1.25], "fidelity": 0.5})

        with pytest.raises(ImportError, match="orchestrator_to_quantum_phases"):
            bridge.orchestrator_to_quantum(state)

        with pytest.raises(ImportError, match="quantum_to_orchestrator_phases"):
            bridge.quantum_to_orchestrator(np.array([0.25, 1.25]))


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


class TestQuantumBridgePipelineWiring:
    """Pipeline: QuantumControlBridge → K_nm → engine → R."""

    def test_quantum_imported_knm_drives_engine(self):
        """import_knm → CouplingState → engine → R∈[0,1]."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        bridge = QuantumControlBridge(n_oscillators=n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        cs = bridge.import_knm(knm)

        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0


# Salvaged module-specific behavioural contracts from deleted mixed tests.
class TestQuantumControlBridgeImportExportContracts:
    def test_import_export_roundtrip(self):
        bridge = QuantumControlBridge(4)
        artifact = {"phases": [0.1, 0.2, 0.3, 0.4], "fidelity": 0.9}
        state = bridge.import_artifact(artifact)
        exported = bridge.export_artifact(state)
        assert exported["fidelity"] == state.stability_proxy

    def test_import_knm_non_square_error(self):
        bridge = QuantumControlBridge(4)
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm(np.ones((3, 4)))


# ──────────────────────────────────────────────────────────────────────
# audit/logger.py: phases without omegas raises ValueError
# ──────────────────────────────────────────────────────────────────────


def test_require_positive_real_rejects_boolean() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _BRIDGE_MODULE._require_positive_real(True, name="dt")


def test_require_finite_real_rejects_boolean_and_non_finite() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _BRIDGE_MODULE._require_finite_real(True, name="R")
    with pytest.raises(ValueError, match="must be finite"):
        _BRIDGE_MODULE._require_finite_real(float("inf"), name="R")


def test_finite_array_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        _BRIDGE_MODULE._finite_array(np.array(["a", "b"], dtype=object), name="knm")


def test_require_sha256_rejects_non_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256 hex digest"):
        _BRIDGE_MODULE._require_sha256("not-a-digest", name="edge_sha256")


def test_validate_layer_assignments_rejects_non_list_group() -> None:
    with pytest.raises(ValueError, match="must contain list groups"):
        _BRIDGE_MODULE._validate_layer_assignments([5], n_phases=1)


def test_validate_upde_state_rejects_non_square_alignment() -> None:
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.1), LayerState(R=0.6, psi=0.2)],
        cross_layer_alignment=np.zeros((2, 3), dtype=np.float64),
        stability_proxy=0.0,
        regime_id="probe",
    )
    with pytest.raises(ValueError, match="must be a square matrix"):
        _BRIDGE_MODULE._validate_upde_state(state)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "wrong", "edge schema must be"),
        ("producer", "wrong", "edge producer must be"),
        ("consumer", "wrong", "edge consumer must be"),
    ],
)
def test_validate_scpn_upde_edge_rejects_bad_headers(
    field: str, value: str, match: str
) -> None:
    bridge = QuantumControlBridge(n_oscillators=2)
    edge = _scpn_upde_edge_payload()
    edge[field] = value
    with pytest.raises(ValueError, match=match):
        bridge._validate_scpn_upde_edge(edge)


def test_validate_scpn_upde_edge_rejects_permitted_qpu_execution() -> None:
    bridge = QuantumControlBridge(n_oscillators=2)
    edge = _scpn_upde_edge_payload()
    edge["permissions"] = {
        "qpu_execution_permitted": True,
        "actuation_permitted": False,
    }
    with pytest.raises(ValueError, match="qpu_execution_permitted must be false"):
        bridge._validate_scpn_upde_edge(edge)
