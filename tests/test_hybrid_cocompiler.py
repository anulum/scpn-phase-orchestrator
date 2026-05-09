# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid co-compiler tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.hybrid_cocompiler import (
    build_hybrid_cocompiler_manifest,
)
from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge
from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state() -> UPDEState:
    state = UPDEState(
        layers=[LayerState(R=0.25, psi=0.1), LayerState(R=0.75, psi=0.3)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.5,
        regime_id="nominal",
    )
    state.cross_layer_alignment[0, 1] = 0.4
    return state


def test_hybrid_cocompiler_manifest_links_quantum_and_spiking_targets() -> None:
    quantum = QuantumControlBridge(n_oscillators=2, trotter_order=2)
    quantum_manifest = quantum.build_quantum_compiler_manifest(
        np.array([[0.0, 0.25], [0.5, 0.0]]),
        np.array([1.0, -0.5]),
        dt=0.125,
    )
    spiking = SNNControllerBridge(n_neurons=32)
    spiking_manifest = spiking.build_neuromorphic_schedule_manifest(
        _state(),
        i_scale=2.0,
        threshold_hz=20.0,
    )

    manifest = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        spiking_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    repeated = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        spiking_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )

    assert manifest == repeated
    assert manifest["manifest_kind"] == "hybrid_neuromorphic_quantum_cocompiler"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "co_simulation_parity_passed"
    assert manifest["target_backends"] == [
        "qiskit_openqasm3",
        "pennylane_qasm",
        "lava",
        "pynn",
    ]
    assert manifest["n_channel_semantics"] == ["Q_control", "S_spike", "audit"]
    assert manifest["qpu_execution_permitted"] is False
    assert manifest["hardware_write_permitted"] is False
    assert manifest["actuation_permitted"] is False
    assert len(manifest["hybrid_manifest_sha256"]) == 64
    assert manifest["component_hashes"] == {
        "quantum_qasm_sha256": quantum_manifest["qasm_sha256"],
        "quantum_manifest_sha256": quantum_manifest["manifest_sha256"],
        "neuromorphic_schedule_sha256": spiking_manifest["schedule_sha256"],
    }
    assert manifest["co_simulation_parity"]["quantum_status"] == (
        "co_simulation_parity_passed"
    )
    assert manifest["co_simulation_parity"]["neuromorphic_status"] == (
        "simulator_parity_passed"
    )
    assert manifest["operator_commands"] == [
        "review hybrid_neuromorphic_quantum_cocompiler.json",
        "run quantum and neuromorphic simulators under the shared audit envelope",
    ]


def test_hybrid_cocompiler_manifest_blocks_mismatched_or_invalid_inputs() -> None:
    quantum = QuantumControlBridge(n_oscillators=2).build_quantum_compiler_manifest(
        np.zeros((2, 2)),
        np.ones(2),
        dt=0.1,
    )
    spiking = SNNControllerBridge().build_neuromorphic_schedule_manifest(_state())

    broken_quantum = dict(quantum)
    broken_quantum["status"] = "co_simulation_parity_failed"
    blocked = build_hybrid_cocompiler_manifest(broken_quantum, spiking)
    assert blocked["status"] == "blocked"
    assert "quantum compiler parity must pass" in blocked["blocked_reasons"]
    assert blocked["actuation_permitted"] is False

    with pytest.raises(ValueError, match="quantum manifest kind"):
        build_hybrid_cocompiler_manifest({"manifest_kind": "wrong"}, spiking)
    with pytest.raises(ValueError, match="n_channel_semantics"):
        build_hybrid_cocompiler_manifest(quantum, spiking, n_channel_semantics=())
