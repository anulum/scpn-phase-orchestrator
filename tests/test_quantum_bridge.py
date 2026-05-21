# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum Bridge tests

from typing import Any, cast

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
        with pytest.raises(ValueError, match="n_oscillators"):
            QuantumControlBridge(n_oscillators=0)

    def test_rejects_negative_oscillators(self):
        with pytest.raises(ValueError, match="n_oscillators"):
            QuantumControlBridge(n_oscillators=-3)

    @pytest.mark.parametrize("n_oscillators", [True, 2.0, "2"])
    def test_rejects_non_integer_oscillator_count(self, n_oscillators: object):
        with pytest.raises(ValueError, match="n_oscillators must be an integer"):
            QuantumControlBridge(n_oscillators=cast("Any", n_oscillators))

    def test_normalises_integer_oscillator_count(self):
        b = QuantumControlBridge(n_oscillators=np.int64(2))
        assert b._n == 2

    def test_default_trotter_order_is_one(self):
        b = QuantumControlBridge(n_oscillators=2)
        assert b._trotter_order == 1

    def test_custom_trotter_order_propagates(self):
        b = QuantumControlBridge(n_oscillators=2, trotter_order=4)
        assert b._trotter_order == 4

    @pytest.mark.parametrize("trotter_order", [0, -1, 1.5, "2", True])
    def test_rejects_invalid_trotter_order(self, trotter_order: object):
        with pytest.raises(ValueError, match="trotter_order"):
            QuantumControlBridge(
                n_oscillators=2,
                trotter_order=cast("Any", trotter_order),
            )


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


class TestImportArtifactValidation:
    def test_rejects_non_mapping_artifact(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="artifact_dict must be a mapping"):
            bridge.import_artifact(cast("Any", [0.0, 0.1]))

    def test_rejects_missing_phases(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must include 'phases'"):
            bridge.import_artifact({"fidelity": 0.5})

    def test_rejects_non_finite_fidelity(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="fidelity must be finite"):
            bridge.import_artifact({"phases": [0.0, 1.0], "fidelity": float("inf")})

    def test_rejects_invalid_fidelity(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="fidelity must be finite"):
            bridge.import_artifact({"phases": [0.0, 1.0], "fidelity": "high"})

    def test_rejects_out_of_range_fidelity(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="fidelity must be finite"):
            bridge.import_artifact({"phases": [0.0, 1.0], "fidelity": 1.7})

    def test_rejects_non_list_layer_assignments(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must be a list of index groups"):
            bridge.import_artifact(
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.3,
                    "layer_assignments": "bad",
                }
            )

    def test_rejects_non_integral_layer_index(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must contain integer indexes"):
            bridge.import_artifact(
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.3,
                    "layer_assignments": [[0, 1.1]],
                }
            )

    def test_rejects_out_of_range_layer_index(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="out of phase range"):
            bridge.import_artifact(
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.3,
                    "layer_assignments": [[0, 2]],
                }
            )

    def test_rejects_repeated_layer_index(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must not repeat phase indexes"):
            bridge.import_artifact(
                {
                    "phases": [0.0, 0.1],
                    "fidelity": 0.3,
                    "layer_assignments": [[0], [0]],
                }
            )

    def test_rejects_incomplete_layer_assignments(self):
        bridge = QuantumControlBridge(n_oscillators=4)
        with pytest.raises(
            ValueError, match="must cover every phase index exactly once"
        ):
            bridge.import_artifact(
                {
                    "phases": [0.0, 0.1, 0.2, 0.3],
                    "fidelity": 0.3,
                    "layer_assignments": [[0, 1, 2]],
                }
            )


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

    def test_accepts_array_like_input(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        state = bridge.import_knm([[0.0, 0.4], [0.4, 0.0]])
        assert state.knm.shape == (2, 2)

    def test_rejects_non_finite_knm(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must contain finite values"):
            bridge.import_knm([[0.0, float("nan")], [0.0, 0.0]])


class TestCompilerManifestValidation:
    def test_rejects_invalid_shapes(self):
        bridge = QuantumControlBridge(n_oscillators=3)
        with pytest.raises(ValueError, match="does not match n_oscillators=3"):
            bridge.build_quantum_compiler_manifest(
                knm=np.ones((2, 2)),
                omegas=np.ones(2),
                dt=0.1,
            )

    def test_rejects_non_finite_dt(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            bridge.build_quantum_compiler_manifest(
                knm=np.ones((2, 2)),
                omegas=np.ones(2),
                dt=float("nan"),
            )

    def test_rejects_non_finite_compiler_arrays(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        with pytest.raises(ValueError, match="must contain finite values"):
            bridge.build_quantum_compiler_manifest(
                knm=np.array([[0.0, float("inf")], [0.0, 0.0]]),
                omegas=np.ones(2),
                dt=0.1,
            )

    def test_marks_parity_status_as_failed_when_terms_differ(self, monkeypatch: Any):
        bridge = QuantumControlBridge(n_oscillators=2)

        def _fake_parity(
            _self: QuantumControlBridge,
            omegas_in: np.ndarray,
            frequency_terms: list[dict[str, object]],
            knm: np.ndarray,
            coupling_terms: list[dict[str, object]],
        ) -> dict[str, object]:
            return {
                "engine": "deterministic_xy_term_reconstruction",
                "max_abs_frequency_error": 0.75,
                "max_abs_coupling_error": 0.0,
                "term_count": len(frequency_terms) + len(coupling_terms),
            }

        monkeypatch.setattr(
            QuantumControlBridge,
            "_quantum_compiler_parity",
            _fake_parity,
        )

        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.4], [0.4, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )
        assert manifest["status"] == "co_simulation_parity_failed"
        assert manifest["co_simulation_parity"]["max_abs_frequency_error"] == 0.75

    def test_qpu_target_readiness_blocks_without_operator_preconditions(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        readiness = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="ibm_quantum",
        )
        repeated = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="ibm_quantum",
        )

        assert readiness["schema"] == "scpn_quantum_target_readiness_v1"
        assert readiness["status"] == "blocked"
        assert readiness["manifest_sha256"] == manifest["manifest_sha256"]
        assert readiness["qpu_execution_permitted"] is False
        assert readiness["actuation_permitted"] is False
        assert readiness["blocked_reasons"] == [
            "credentials_not_configured",
            "operator_approval_missing",
        ]
        assert readiness["readiness_sha256"] == repeated["readiness_sha256"]

    def test_qpu_target_readiness_can_be_ready_but_never_executing(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        readiness = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="pennylane_qasm",
            provider="pennylane",
            credentials_configured=True,
            operator_approved=True,
        )

        assert readiness["status"] == "ready_not_executed"
        assert readiness["blocked_reasons"] == []
        assert readiness["qpu_execution_permitted"] is False
        assert readiness["actuation_permitted"] is False
        assert len(str(readiness["readiness_sha256"])) == 64

    def test_qpu_target_readiness_rejects_invalid_provider_and_target_metadata(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        bad_manifest = dict(manifest)
        bad_manifest["target_backends"] = "qiskit_openqasm3"
        with pytest.raises(
            ValueError, match="target_backends must be a list of strings"
        ):
            bridge.audit_qpu_target_readiness(
                bad_manifest,
                target_backend="qiskit_openqasm3",
                provider="ibm_quantum",
            )

        with pytest.raises(ValueError, match="target_backend"):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="   ",
                provider="ibm_quantum",
            )
        with pytest.raises(ValueError, match="provider"):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="qiskit_openqasm3",
                provider="\x01",
            )

    def test_qpu_target_readiness_rejects_bad_flag_types(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        with pytest.raises(
            ValueError, match="credentials_configured must be a boolean"
        ):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="qiskit_openqasm3",
                provider="ibm_quantum",
                credentials_configured="yes",
            )
        with pytest.raises(ValueError, match="operator_approved must be a boolean"):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="qiskit_openqasm3",
                provider="ibm_quantum",
                credentials_configured=True,
                operator_approved=1,
            )

    def test_qpu_target_readiness_blocks_if_execution_permissions_escape_sandbox(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )
        manifest["qpu_execution_permitted"] = True
        manifest["actuation_permitted"] = True

        readiness = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="ibm_quantum",
            credentials_configured=True,
            operator_approved=True,
        )

        assert readiness["status"] == "blocked"
        assert (
            "qpu_execution_permission_must_remain_false" in readiness["blocked_reasons"]
        )
        assert "actuation_permission_must_remain_false" in readiness["blocked_reasons"]
        assert readiness["qpu_execution_permitted"] is False
        assert readiness["actuation_permitted"] is False

    def test_qpu_target_readiness_record_is_deterministic(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        readiness_a = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="ibm_quantum",
            credentials_configured=True,
            operator_approved=True,
        )
        readiness_b = bridge.audit_qpu_target_readiness(
            manifest,
            target_backend="qiskit_openqasm3",
            provider="ibm_quantum",
            credentials_configured=True,
            operator_approved=True,
        )

        assert readiness_a["readiness_sha256"] == readiness_b["readiness_sha256"]
        assert readiness_a["status"] == readiness_b["status"]

    def test_qpu_target_readiness_rejects_bad_manifest_and_target(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        manifest = bridge.build_quantum_compiler_manifest(
            knm=np.array([[0.0, 0.2], [0.2, 0.0]]),
            omegas=np.array([0.1, 0.2]),
            dt=0.05,
        )

        with pytest.raises(ValueError, match="target_backend"):
            bridge.audit_qpu_target_readiness(
                manifest,
                target_backend="dwave",
                provider="ibm_quantum",
            )
        with pytest.raises(ValueError, match="quantum_compiler_manifest"):
            bridge.audit_qpu_target_readiness(
                {"manifest_kind": "other", "target_backends": ["qiskit_openqasm3"]},
                target_backend="qiskit_openqasm3",
                provider="ibm_quantum",
            )


class TestSolveQUPDEValidation:
    def test_rejects_invalid_t_max(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.5, 0.5], [0.5, 0.5]])
        omegas = np.ones(2)
        with pytest.raises(ValueError, match="t_max must be finite and positive"):
            bridge.solve_q_upde(knm, omegas, t_max=0.0)

    def test_rejects_invalid_dt(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.5, 0.5], [0.5, 0.5]])
        omegas = np.ones(2)
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            bridge.solve_q_upde(knm, omegas, t_max=1.0, dt=float("inf"))

    def test_rejects_invalid_trotter_per_step(self):
        bridge = QuantumControlBridge(n_oscillators=2)
        knm = np.array([[0.5, 0.5], [0.5, 0.5]])
        omegas = np.ones(2)
        with pytest.raises(
            ValueError, match="trotter_per_step must be an integer >= 1"
        ):
            bridge.solve_q_upde(knm, omegas, trotter_per_step=0)


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
