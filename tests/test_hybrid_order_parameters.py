# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid order-parameter tests

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest


def test_hybrid_order_product_vs_bell_entropy_ordering() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    phases = np.array([0.0, np.pi / 3, np.pi / 2, np.pi], dtype=np.float64)
    product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    product = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=product_state,
        bipartition=((0,), (1,)),
    )
    bell = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_state,
        bipartition=((0,), (1,)),
    )

    assert product.entanglement_entropy < 0.05
    assert bell.entanglement_entropy > 0.9
    assert (
        bell.normalised_entanglement_entropy > product.normalised_entanglement_entropy
    )


def test_hybrid_order_density_matrix_matches_statevector_path() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    phases = np.array([0.4, 1.1, 2.2], dtype=np.float64)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )
    bell_density = np.outer(bell_state, np.conj(bell_state))

    vector_result = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_state,
        bipartition=((0,), (1,)),
    )
    density_result = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_density,
        bipartition=((0,), (1,)),
    )

    assert density_result.entanglement_entropy == pytest.approx(
        vector_result.entanglement_entropy,
        rel=1e-12,
        abs=1e-12,
    )
    assert density_result.normalised_entanglement_entropy == pytest.approx(
        vector_result.normalised_entanglement_entropy,
        rel=1e-12,
        abs=1e-12,
    )


def test_hybrid_order_rejects_non_positive_semidefinite_density_matrix() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    invalid_density = np.diag(np.array([1.2, -0.2, 0.0, 0.0], dtype=np.complex128))

    with pytest.raises(ValueError, match="positive semidefinite"):
        compute_hybrid_entanglement_order_parameter(
            phases=np.array([0.0, 1.0], dtype=np.float64),
            quantum_state=invalid_density,
            bipartition=((0,), (1,)),
        )


def test_hybrid_order_audit_record_json_safe_and_deterministic_hash() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    phases = np.array([0.2, 0.8, 1.4, 2.0], dtype=np.float64)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )

    first = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_state,
        qubit_count=2,
        bipartition=((0,), (1,)),
    )
    second = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_state,
        qubit_count=2,
        bipartition=((0,), (1,)),
    )

    first_record = first.to_audit_record()
    second_record = second.to_audit_record()
    assert first_record == second_record
    assert (
        first_record["claim_boundary"]
        == "quantum_cosimulation_monitor_not_qpu_execution"
    )
    assert first_record["non_actuating"] is True
    assert first_record["execution_disabled"] is True
    assert first_record["backend"] == "numpy_statevector_density_matrix"

    json_dump = json.dumps(first_record, sort_keys=True, separators=(",", ":"))
    assert json.loads(json_dump) == first_record

    record_body = dict(first_record)
    record_hash = record_body.pop("record_hash")
    expected_hash = hashlib.sha256(
        json.dumps(
            record_body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    ).hexdigest()
    assert expected_hash == record_hash


def test_hybrid_order_classical_order_consistency() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )
    from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

    phases = np.array([0.0, 0.8, 1.6, 2.4], dtype=np.float64)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )
    result = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=bell_state,
        bipartition=((0,), (1,)),
    )
    r_value, psi_value = compute_order_parameter(phases)

    assert pytest.approx(r_value, rel=1e-12, abs=1e-12) == result.R
    assert pytest.approx(psi_value, rel=1e-12, abs=1e-12) == result.Psi


@pytest.mark.parametrize(
    "bad_input",
    [
        {"phases": np.array([0.0, np.nan])},
        {"phases": np.array([0.0, True], dtype=object)},
        {"quantum_state": np.array([1.0, False, 0.0, 0.0], dtype=object)},
        {"quantum_state": np.array([1.0, 2.0, 3.0], dtype=np.complex128)},
        {"bipartition": ((0, 0), (1,))},
        {"bipartition": ((0,), (0, 1))},
        {"bipartition": ((0, 1),)},
        {"bipartition": ((0,), (np.bool_(True),))},
        {
            "quantum_state": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        },
        {
            "quantum_state": np.array([1, 0, 0, 0], dtype=np.complex128),
            "qubit_count": 3,
        },
        {
            "quantum_state": np.array([1, 0, 0, 0], dtype=np.complex128),
            "qubit_count": True,
        },
    ],
)
def test_hybrid_order_invalid_inputs_fail_closed(bad_input: dict[str, object]) -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    defaults = {
        "phases": np.array([0.0, 1.0], dtype=np.float64),
        "quantum_state": np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.complex128,
        ),
        "bipartition": ((0,), (1,)),
        "qubit_count": None,
    }
    defaults.update(bad_input)

    with pytest.raises(ValueError):
        compute_hybrid_entanglement_order_parameter(
            phases=defaults["phases"],
            quantum_state=defaults["quantum_state"],
            bipartition=defaults.get("bipartition"),
            qubit_count=defaults.get("qubit_count"),
        )


def test_hybrid_order_accepts_numpy_integer_qubit_contracts() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    result = compute_hybrid_entanglement_order_parameter(
        phases=np.array([0.0, 1.0], dtype=np.float64),
        quantum_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        qubit_count=np.int64(2),
        bipartition=((np.int64(0),), (np.int64(1),)),
    )

    assert result.qubit_count == 2
    assert result.bipartition == ((0,), (1,))


def test_hybrid_order_explicit_simulator_backend_contracts() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    phases = np.array([0.0, 1.0], dtype=np.float64)
    statevector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    density = np.outer(statevector, np.conj(statevector))

    vector_result = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=statevector,
        bipartition=((0,), (1,)),
        simulator_backend="numpy_statevector",
    )
    density_result = compute_hybrid_entanglement_order_parameter(
        phases=phases,
        quantum_state=density,
        bipartition=((0,), (1,)),
        simulator_backend="numpy_density_matrix",
    )

    assert vector_result.backend == "numpy_statevector"
    assert density_result.backend == "numpy_density_matrix"

    with pytest.raises(ValueError, match="statevector"):
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=density,
            bipartition=((0,), (1,)),
            simulator_backend="numpy_statevector",
        )
    with pytest.raises(ValueError, match="density matrix"):
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=statevector,
            bipartition=((0,), (1,)),
            simulator_backend="numpy_density_matrix",
        )
    with pytest.raises(ValueError, match="simulator_backend"):
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=statevector,
            bipartition=((0,), (1,)),
            simulator_backend="qpu_live",
        )
