# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid co-compiler tests

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
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
build_hybrid_cocompiler_manifest = _COCOMPILER_MODULE.build_hybrid_cocompiler_manifest
audit_hybrid_target_readiness = _COCOMPILER_MODULE.audit_hybrid_target_readiness
build_hybrid_operator_handoff_package = (
    _COCOMPILER_MODULE.build_hybrid_operator_handoff_package
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


def _quantum_readiness(status: str = "ready_not_executed") -> dict[str, object]:
    blocked_reasons = [] if status == "ready_not_executed" else ["operator_missing"]
    return {
        "schema": "scpn_quantum_target_readiness_v1",
        "provider": "pennylane",
        "target_backend": "pennylane_qasm",
        "manifest_sha256": "b" * 64,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "credentials_configured": status == "ready_not_executed",
        "operator_approved": status == "ready_not_executed",
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
        "operator_commands": [
            "review quantum_compiler_manifest.json",
            "run simulator parity outside SPO before target handoff",
            "submit QPU job only from an approved external operator workflow",
        ],
        "readiness_sha256": "d" * 64,
    }


def _neuromorphic_readiness(status: str = "ready_not_executed") -> dict[str, object]:
    blocked_reasons = [] if status == "ready_not_executed" else ["operator_missing"]
    return {
        "schema": "scpn_neuromorphic_target_readiness_v1",
        "target_backend": "pynn",
        "hardware_site": "brainscales_review_lane",
        "manifest_sha256": "c" * 64,
        "status": status,
        "blocked_reasons": blocked_reasons,
        "credentials_configured": status == "ready_not_executed",
        "operator_approved": status == "ready_not_executed",
        "external_simulator_parity_verified": status == "ready_not_executed",
        "hardware_write_permitted": False,
        "actuation_permitted": False,
        "operator_commands": [
            "review neuromorphic_schedule_manifest.json",
            "run target simulator parity outside SPO before hardware handoff",
            "submit neuromorphic hardware job only from an approved operator workflow",
        ],
        "readiness_sha256": "e" * 64,
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
    with pytest.raises(ValueError, match="quantum_manifest"):
        build_hybrid_cocompiler_manifest([], spiking)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="neuromorphic_manifest"):
        build_hybrid_cocompiler_manifest(quantum, [])  # type: ignore[arg-type]
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
    with pytest.raises(ValueError, match="non-empty sequence"):
        build_hybrid_cocompiler_manifest(
            quantum,
            spiking,
            n_channel_semantics={"Q_control": "active"},
        )


def test_hybrid_cocompiler_fails_closed_for_all_component_permission_leaks() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    quantum["qpu_execution_permitted"] = True
    quantum["actuation_permitted"] = True
    spiking["hardware_write_permitted"] = True
    spiking["actuation_permitted"] = True

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
        ("qasm_sha256", "g" * 64),
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


def test_hybrid_cocompiler_hash_errors_do_not_echo_payload_values() -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    quantum["qasm_sha256"] = "/private/topology/qpu-audit"

    with pytest.raises(ValueError) as excinfo:
        build_hybrid_cocompiler_manifest(quantum, spiking)

    message = str(excinfo.value)
    assert message == "qasm_sha256 must be a 64-character SHA-256 hex string"
    assert "private" not in message
    assert "qpu-audit" not in message


@pytest.mark.parametrize(
    ("manifest_name", "field"),
    [
        ("quantum", "qpu_execution_permitted"),
        ("quantum", "actuation_permitted"),
        ("neuromorphic", "hardware_write_permitted"),
        ("neuromorphic", "actuation_permitted"),
    ],
)
def test_hybrid_cocompiler_rejects_non_bool_permission_fields(
    manifest_name: str,
    field: str,
) -> None:
    quantum = _quantum_manifest()
    spiking = _neuromorphic_manifest()
    target = quantum if manifest_name == "quantum" else spiking
    target[field] = 0

    with pytest.raises(ValueError, match=field):
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


def test_hybrid_cocompiler_hash_and_field_order_is_deterministic() -> None:
    """Manifest hashes and content must not depend on input field order."""

    quantum = dict(reversed(list(_quantum_manifest().items())))
    neuromorphic = dict(reversed(list(_neuromorphic_manifest().items())))

    manifest_one = build_hybrid_cocompiler_manifest(quantum, neuromorphic)
    manifest_two = build_hybrid_cocompiler_manifest(quantum, neuromorphic)
    assert manifest_one == manifest_two

    manifest_body = {
        k: v for k, v in manifest_one.items() if k != "hybrid_manifest_sha256"
    }
    expected_hash = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert manifest_one["hybrid_manifest_sha256"] == expected_hash


def test_hybrid_cocompiler_does_not_mutate_input_manifests() -> None:
    quantum = _quantum_manifest()
    neuromorphic = _neuromorphic_manifest()
    expected_quantum = deepcopy(quantum)
    expected_neuromorphic = deepcopy(neuromorphic)

    build_hybrid_cocompiler_manifest(quantum, neuromorphic)

    assert quantum == expected_quantum
    assert neuromorphic == expected_neuromorphic


@pytest.mark.parametrize(
    "missing_key", ["qasm_sha256", "manifest_sha256", "schedule_sha256"]
)
def test_hybrid_cocompiler_rejects_missing_component_hashes(missing_key: str) -> None:
    quantum = _quantum_manifest()
    neuromorphic = _neuromorphic_manifest()

    if missing_key in ("qasm_sha256", "manifest_sha256"):
        quantum.pop(missing_key)
    else:
        neuromorphic.pop(missing_key)

    with pytest.raises(ValueError, match=f"{missing_key} must be a 64-character"):
        build_hybrid_cocompiler_manifest(quantum, neuromorphic)


def test_hybrid_target_readiness_requires_component_readiness_and_approval() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )

    record = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(),
        _neuromorphic_readiness(),
        hybrid_operator_approved=False,
    )

    assert record["schema"] == "scpn_hybrid_target_readiness_v1"
    assert record["status"] == "blocked"
    assert record["blocked_reasons"] == ["hybrid_operator_approval_missing"]
    assert record["quantum_readiness_sha256"] == "d" * 64
    assert record["neuromorphic_readiness_sha256"] == "e" * 64
    assert record["hybrid_manifest_sha256"] == hybrid["hybrid_manifest_sha256"]
    assert record["qpu_execution_permitted"] is False
    assert record["hardware_write_permitted"] is False
    assert record["actuation_permitted"] is False
    assert len(record["readiness_sha256"]) == 64


def test_hybrid_target_readiness_ready_not_executed_is_deterministic() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )

    record = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(),
        _neuromorphic_readiness(),
        hybrid_operator_approved=True,
    )
    repeated = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(),
        _neuromorphic_readiness(),
        hybrid_operator_approved=True,
    )

    assert record == repeated
    assert record["status"] == "ready_not_executed"
    assert record["blocked_reasons"] == []
    assert record["component_statuses"] == {
        "hybrid": "co_simulation_parity_passed",
        "neuromorphic": "ready_not_executed",
        "quantum": "ready_not_executed",
    }
    assert record["operator_commands"] == [
        "review hybrid_neuromorphic_quantum_cocompiler.json",
        "verify quantum and neuromorphic readiness hashes before handoff",
        "submit hybrid execution only from an approved external operator workflow",
    ]


def test_hybrid_target_readiness_blocks_component_or_hash_mismatch() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )
    quantum = _quantum_readiness(status="blocked")
    neuromorphic = _neuromorphic_readiness()
    neuromorphic["manifest_sha256"] = "f" * 64

    record = audit_hybrid_target_readiness(
        hybrid,
        quantum,
        neuromorphic,
        hybrid_operator_approved=True,
    )

    assert record["status"] == "blocked"
    assert record["blocked_reasons"] == [
        "quantum_target_readiness_not_ready",
        "neuromorphic_manifest_hash_mismatch",
    ]


def test_hybrid_target_readiness_rejects_invalid_readiness_schema() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )
    quantum = _quantum_readiness()
    quantum["schema"] = "wrong"

    with pytest.raises(ValueError, match="quantum_readiness"):
        audit_hybrid_target_readiness(
            hybrid,
            quantum,
            _neuromorphic_readiness(),
            hybrid_operator_approved=True,
        )


def test_hybrid_operator_handoff_package_is_non_executing_and_hash_linked() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )
    readiness = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(),
        _neuromorphic_readiness(),
        hybrid_operator_approved=True,
    )

    package = build_hybrid_operator_handoff_package(hybrid, readiness)
    repeated = build_hybrid_operator_handoff_package(hybrid, readiness)

    assert package == repeated
    assert package["schema"] == "scpn_hybrid_operator_handoff_package_v1"
    assert package["status"] == "ready_not_executed"
    assert package["hybrid_manifest_sha256"] == hybrid["hybrid_manifest_sha256"]
    assert package["hybrid_readiness_sha256"] == readiness["readiness_sha256"]
    assert package["execution_permitted"] is False
    assert package["qpu_execution_permitted"] is False
    assert package["hardware_write_permitted"] is False
    assert package["actuation_permitted"] is False
    assert package["target_backends"] == [
        "qiskit_openqasm3",
        "pennylane_qasm",
        "lava",
        "pynn",
    ]
    assert package["operator_commands"] == [
        "review hybrid_neuromorphic_quantum_cocompiler.json",
        "review scpn_hybrid_target_readiness_v1.json",
        "verify package_sha256 before external operator handoff",
        "execute only outside SPO from an approved operator workflow",
    ]
    assert len(package["package_sha256"]) == 64


def test_hybrid_operator_handoff_package_preserves_blocked_readiness() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )
    readiness = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(status="blocked"),
        _neuromorphic_readiness(),
        hybrid_operator_approved=True,
    )

    package = build_hybrid_operator_handoff_package(hybrid, readiness)

    assert package["status"] == "blocked"
    assert package["blocked_reasons"] == readiness["blocked_reasons"]
    assert package["execution_permitted"] is False


def test_hybrid_operator_handoff_package_rejects_mismatched_readiness() -> None:
    hybrid = build_hybrid_cocompiler_manifest(
        _quantum_manifest(),
        _neuromorphic_manifest(),
    )
    readiness = audit_hybrid_target_readiness(
        hybrid,
        _quantum_readiness(),
        _neuromorphic_readiness(),
        hybrid_operator_approved=True,
    )
    readiness["hybrid_manifest_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="hybrid readiness manifest hash"):
        build_hybrid_operator_handoff_package(hybrid, readiness)


def test_hybrid_pipeline_enforces_no_actuation_review_boundaries_end_to_end() -> None:
    quantum = _quantum_manifest()
    neuromorphic = _neuromorphic_manifest()
    quantum["actuation_permitted"] = True
    neuromorphic["actuation_permitted"] = True

    manifest = build_hybrid_cocompiler_manifest(
        quantum,
        neuromorphic,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    repeated_manifest = build_hybrid_cocompiler_manifest(
        quantum,
        neuromorphic,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    assert manifest == repeated_manifest
    assert manifest["actuation_permitted"] is False
    assert (
        "quantum actuation_permitted must remain false"
        in manifest["blocked_reasons"]
    )
    assert (
        "neuromorphic actuation_permitted must remain false"
        in manifest["blocked_reasons"]
    )

    quantum_readiness = _quantum_readiness()
    neuromorphic_readiness = _neuromorphic_readiness()
    quantum_readiness["actuation_permitted"] = True
    neuromorphic_readiness["actuation_permitted"] = True

    readiness = audit_hybrid_target_readiness(
        manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=True,
    )
    assert readiness["status"] == "blocked"
    assert readiness["actuation_permitted"] is False
    assert (
        "quantum_readiness actuation_permitted must remain false"
        in readiness["blocked_reasons"]
    )
    assert (
        "neuromorphic_readiness actuation_permitted must remain false"
        in readiness["blocked_reasons"]
    )

    package = build_hybrid_operator_handoff_package(manifest, readiness)
    repeated_package = build_hybrid_operator_handoff_package(manifest, readiness)
    assert package == repeated_package
    assert package["execution_permitted"] is False
    assert package["actuation_permitted"] is False
    assert package["qpu_execution_permitted"] is False
    assert package["hardware_write_permitted"] is False
    assert package["status"] == "blocked"
    assert "execute only outside SPO from an approved operator workflow" in package[
        "operator_commands"
    ]
