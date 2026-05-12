# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid co-compiler tests

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_COCOMPILER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "scpn_phase_orchestrator"
    / "adapters"
    / "hybrid_cocompiler.py"
)
_COCOMPILER_SPEC = importlib.util.spec_from_file_location(
    "scpn_phase_orchestrator.adapters.hybrid_cocompiler",
    _COCOMPILER_PATH,
)
assert _COCOMPILER_SPEC is not None
assert _COCOMPILER_SPEC.loader is not None
_COCOMPILER_MODULE = importlib.util.module_from_spec(_COCOMPILER_SPEC)
_COCOMPILER_SPEC.loader.exec_module(_COCOMPILER_MODULE)
build_hybrid_cocompiler_manifest = (
    _COCOMPILER_MODULE.build_hybrid_cocompiler_manifest
)


def _quantum_manifest() -> dict[str, object]:
    return {
        "manifest_kind": "quantum_compiler_manifest",
        "schema_version": 1,
        "status": "co_simulation_parity_passed",
        "target_backends": ["qiskit_openqasm3", "pennylane_qasm"],
        "n_qubits": 2,
        "trotter_order": 2,
        "dt": 0.125,
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
        "frequency_terms": [
            {"qubit": 0, "omega": 1.0, "rz_angle": 0.125},
            {"qubit": 1, "omega": -0.5, "rz_angle": -0.0625},
        ],
        "coupling_terms": [
            {
                "source": 0,
                "target": 1,
                "forward_coupling": 0.25,
                "reverse_coupling": 0.5,
                "symmetric_coupling": 0.375,
                "xx_angle": 0.046875,
                "yy_angle": 0.046875,
            },
        ],
        "openqasm": "OPENQASM 3.0;\nqubit[2] q;\n",
        "qasm_sha256": "a" * 64,
        "co_simulation_parity": {
            "engine": "deterministic_xy_term_reconstruction",
            "max_abs_frequency_error": 0.0,
            "max_abs_coupling_error": 0.0,
            "term_count": 3,
        },
        "operator_commands": [
            "review quantum_compiler_manifest.json",
            "run Qiskit or PennyLane simulator parity before QPU handoff",
        ],
        "manifest_sha256": "b" * 64,
    }


def _neuromorphic_manifest() -> dict[str, object]:
    return {
        "manifest_kind": "neuromorphic_schedule_manifest",
        "schema_version": 1,
        "status": "simulator_parity_passed",
        "target_backends": ["lava", "pynn"],
        "n_layers": 2,
        "n_neurons_per_population": 32,
        "tau_rc_s": 0.02,
        "tau_ref_s": 0.002,
        "input_scale": 2.0,
        "threshold_hz": 20.0,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
        "populations": [
            {"layer": 0, "R": 0.25, "psi": 0.1, "estimated_rate_hz": 5.0},
            {"layer": 1, "R": 0.75, "psi": 0.3, "estimated_rate_hz": 15.0},
        ],
        "projections": [
            {"source": 0, "target": 1, "weight": 0.4, "delay_ms": 1.0},
        ],
        "control_actions": [
            {
                "knob": "spike_rate_bias",
                "scope": "layer_1",
                "value": 15.0,
                "ttl_s": 0.125,
                "justification": "deterministic schedule parity",
            },
        ],
        "simulator_parity": {
            "engine": "numpy_lif_rate_estimate",
            "max_abs_rate_error_hz": 0.0,
            "sample_count": 2,
        },
        "operator_commands": [
            "review neuromorphic_schedule_manifest.json",
            "run Lava or PyNN simulator parity before hardware handoff",
        ],
        "schedule_sha256": "c" * 64,
    }


def test_hybrid_cocompiler_manifest_links_quantum_and_spiking_targets() -> None:
    quantum_manifest = _quantum_manifest()
    spiking_manifest = _neuromorphic_manifest()

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
    assert manifest["co_simulation_parity"]["quantum_term_count"] == 3
    assert manifest["co_simulation_parity"]["neuromorphic_sample_count"] == 2
    assert manifest["operator_commands"] == [
        "review hybrid_neuromorphic_quantum_cocompiler.json",
        "run quantum and neuromorphic simulators under the shared audit envelope",
    ]


def test_hybrid_cocompiler_manifest_blocks_mismatched_or_invalid_inputs() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()

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


def test_hybrid_cocompiler_strips_semantics_and_rejects_invalid_channels() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()

    manifest = build_hybrid_cocompiler_manifest(
        quantum,
        spiking,
        n_channel_semantics=(" Q_control ", "S_spike\t", "audit"),
    )
    assert manifest["n_channel_semantics"] == ["Q_control", "S_spike", "audit"]

    with pytest.raises(ValueError, match="non-empty sequence"):
        build_hybrid_cocompiler_manifest(
            quantum,
            spiking,
            n_channel_semantics="Q_control",
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        build_hybrid_cocompiler_manifest(
            quantum,
            spiking,
            n_channel_semantics=("Q_control", " "),
        )


def test_hybrid_cocompiler_fails_closed_for_all_component_permission_leaks() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    quantum["qpu_execution_permitted"] = True
    quantum["actuation_permitted"] = None
    spiking["hardware_write_permitted"] = True
    spiking["actuation_permitted"] = "pending-review"

    manifest = build_hybrid_cocompiler_manifest(quantum, spiking)

    assert manifest["status"] == "blocked"
    assert manifest["qpu_execution_permitted"] is False
    assert manifest["hardware_write_permitted"] is False
    assert manifest["actuation_permitted"] is False
    assert manifest["blocked_reasons"] == [
        "quantum qpu_execution_permitted must remain false",
        "quantum actuation_permitted must remain false",
        "neuromorphic hardware_write_permitted must remain false",
        "neuromorphic actuation_permitted must remain false",
    ]


def test_hybrid_cocompiler_reports_neuromorphic_parity_failure() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    spiking["status"] = "simulator_parity_failed"

    manifest = build_hybrid_cocompiler_manifest(quantum, spiking)

    assert manifest["status"] == "blocked"
    assert manifest["blocked_reasons"] == ["neuromorphic simulator parity must pass"]
    assert manifest["co_simulation_parity"]["neuromorphic_status"] == (
        "simulator_parity_failed"
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("qasm_sha256", "a" * 63),
        ("manifest_sha256", 123),
    ],
)
def test_hybrid_cocompiler_rejects_invalid_quantum_hashes(
    field: str,
    bad_value: object,
) -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    quantum[field] = bad_value

    with pytest.raises(ValueError, match=f"{field} must be a 64-character"):
        build_hybrid_cocompiler_manifest(quantum, spiking)


def test_hybrid_cocompiler_rejects_invalid_neuromorphic_hash() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    spiking["schedule_sha256"] = ""

    with pytest.raises(ValueError, match="schedule_sha256 must be a 64-character"):
        build_hybrid_cocompiler_manifest(quantum, spiking)


@pytest.mark.parametrize(
    ("owner", "bad_backends", "message"),
    [
        ("quantum", "qiskit_openqasm3", "quantum target_backends must be a sequence"),
        ("neuromorphic", 7, "neuromorphic target_backends must be a sequence"),
        (
            "quantum",
            ["qiskit_openqasm3", ""],
            "quantum target_backends entries must be non-empty strings",
        ),
        (
            "neuromorphic",
            ["lava", object()],
            "neuromorphic target_backends entries must be non-empty strings",
        ),
    ],
)
def test_hybrid_cocompiler_rejects_invalid_backend_lists(
    owner: str,
    bad_backends: object,
    message: str,
) -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    if owner == "quantum":
        quantum["target_backends"] = bad_backends
    else:
        spiking["target_backends"] = bad_backends

    with pytest.raises(ValueError, match=message):
        build_hybrid_cocompiler_manifest(quantum, spiking)


def test_hybrid_cocompiler_defaults_missing_or_non_integer_parity_counts() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    quantum["co_simulation_parity"] = {"term_count": "3"}
    spiking["simulator_parity"] = {"sample_count": 2.0}

    manifest = build_hybrid_cocompiler_manifest(quantum, spiking)

    assert manifest["co_simulation_parity"]["quantum_term_count"] == 0
    assert manifest["co_simulation_parity"]["neuromorphic_sample_count"] == 0

    quantum["co_simulation_parity"] = None
    spiking["simulator_parity"] = None
    manifest = build_hybrid_cocompiler_manifest(quantum, spiking)
    assert manifest["co_simulation_parity"]["quantum_term_count"] == 0
    assert manifest["co_simulation_parity"]["neuromorphic_sample_count"] == 0
