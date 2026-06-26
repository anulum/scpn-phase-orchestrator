# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio hybrid-order panel tests

"""Studio facade contract tests for the hybrid-order review panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.monitor.hybrid_order import (
    compute_hybrid_entanglement_order_parameter,
)
from scpn_phase_orchestrator.monitor.hybrid_order_examples import (
    build_hybrid_order_parameter_scenarios,
)


def _monitor_records() -> list[dict[str, object]]:
    """Return production hybrid-order monitor records for panel tests."""
    phases = np.array([0.0, 0.4, 0.9, 1.7], dtype=np.float64)
    product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )
    bell_density = np.outer(bell_state, np.conj(bell_state))
    return [
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=product_state,
            bipartition=((0,), (1,)),
            simulator_backend="numpy_statevector",
        ).to_audit_record(),
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=bell_density,
            bipartition=((0,), (1,)),
            simulator_backend="numpy_density_matrix",
        ).to_audit_record(),
    ]


def _record() -> dict[str, object]:
    """Return one mutable monitor record."""
    return _copy_mapping(_monitor_records()[0])


def _scenarios() -> list[dict[str, object]]:
    """Return mutable production hybrid-order scenario records."""
    return [
        _copy_mapping(record) for record in build_hybrid_order_parameter_scenarios()
    ]


def _scenario() -> dict[str, object]:
    """Return one mutable hybrid-order scenario record."""
    return _scenarios()[0]


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def _state_candidates(scenario: dict[str, object]) -> list[dict[str, object]]:
    """Return a scenario's candidate list with strict test-time typing."""
    return cast("list[dict[str, object]]", scenario["state_candidates"])


def test_hybrid_order_panel_renders_monitor_and_scenario_evidence() -> None:
    """The public Studio facade renders non-actuating hybrid-order evidence."""
    records = _monitor_records()
    scenarios = build_hybrid_order_parameter_scenarios()

    panel = studio.build_hybrid_order_studio_panel(records, scenarios=scenarios)

    assert panel["panel_kind"] == "studio_hybrid_order_panel"
    assert panel["monitor"] == "hybrid_entanglement_order_parameter"
    assert panel["claim_boundary"] == "quantum_cosimulation_monitor_not_qpu_execution"
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["qpu_execution_permitted"] is False
    assert panel["record_count"] == 2
    assert panel["scenario_count"] == len(scenarios)
    assert panel["latest"]["backend"] == "numpy_density_matrix"
    assert panel["strongest_entanglement"]["backend"] == "numpy_density_matrix"
    assert panel["simulator_backends"] == ("numpy_density_matrix", "numpy_statevector")
    assert set(cast("tuple[str, ...]", panel["scenario_domains"])) == {
        "cardiac_rhythm",
        "power_grid",
        "quantum_simulation",
    }
    assert panel["entropy_range"]["minimum"] < panel["entropy_range"]["maximum"]
    assert panel["normalised_entanglement_range"]["minimum"] <= 1.0
    assert panel["participation_ratio_range"]["minimum"] > 0.0
    assert {row["state_type"] for row in panel["candidate_rows"]} >= {
        "entangled",
        "product",
    }
    assert "actions_to_apply" not in panel
    assert "qpu_commands" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]
    assert len(decoded_panel["candidate_rows"]) == len(panel["candidate_rows"])


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ({}, "non-empty sequence"),
        ([], "non-empty sequence"),
        ([42], "record must be a mapping"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_record_sequence(
    records: object,
    match: str,
) -> None:
    """Record sequence validation fails closed before rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel(
            cast("list[dict[str, object]]", records)
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("claim_boundary", "qpu_execution_claim", "claim boundary"),
        ("non_actuating", False, "non_actuating"),
        ("execution_disabled", False, "execution_disabled"),
        ("backend", "qpu_live_backend", "backend"),
        ("qubit_count", True, "qubit_count"),
        ("qubit_count", 0, "qubit_count"),
        ("R", 1.2, "R"),
        ("entanglement_entropy", -0.1, "entanglement_entropy"),
        ("participation_ratio", 0.0, "participation_ratio"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_record_shape(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Top-level monitor schema and boundary violations fail closed."""
    record = _record()
    record[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel([record])


@pytest.mark.parametrize(
    ("bipartition", "match"),
    [
        ("bad", "two-group bipartition"),
        ([[0]], "two groups"),
        (["left", [1]], "group 0"),
        ([[], [0, 1]], "group 0"),
        ([[0], [2]], "out of range"),
        ([[0], [0]], "cover each qubit"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_bipartitions(
    bipartition: object,
    match: str,
) -> None:
    """Bipartition validation rejects non-covering or malformed qubit groups."""
    record = _record()
    record["bipartition"] = bipartition

    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel([record])


def test_hybrid_order_panel_rejects_payload_hash_mismatch() -> None:
    """A syntactically valid but mismatched monitor record hash fails closed."""
    record = _record()
    record["record_hash"] = "0" * 64

    with pytest.raises(ValueError, match="record_hash does not match payload"):
        studio.build_hybrid_order_studio_panel([record])


@pytest.mark.parametrize(
    ("scenarios", "match"),
    [
        ({}, "scenarios must be a sequence"),
        ([42], "scenario must be a mapping"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_scenario_sequence(
    scenarios: object,
    match: str,
) -> None:
    """Scenario sequence validation fails closed before candidate flattening."""
    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel(
            _monitor_records(),
            scenarios=cast("list[dict[str, object]]", scenarios),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("claim_boundary", "qpu_execution_claim", "claim boundary"),
        ("non_actuating", False, "non_actuating"),
        ("execution_disabled", False, "execution_disabled"),
        ("state_candidates", [], "state_candidates"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_scenario_shape(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Scenario-level schema and boundary violations fail closed."""
    scenario = _scenario()
    scenario[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel(_monitor_records(), scenarios=[scenario])


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        (42, "candidate must be a mapping"),
        ({"claim_boundary": "qpu_execution_claim"}, "claim boundary"),
        ({"non_actuating": False}, "non_actuating"),
        ({"execution_disabled": False}, "execution_disabled"),
        ({"state_type": "qpu"}, "state_type"),
    ],
)
def test_hybrid_order_panel_rejects_malformed_state_candidates(
    candidate: object,
    match: str,
) -> None:
    """Candidate-level schema and boundary violations fail closed."""
    scenario = _scenario()
    candidate_rows = _state_candidates(scenario)
    if isinstance(candidate, dict):
        mutated_candidate = dict(candidate_rows[0])
        mutated_candidate.update(candidate)
        candidate_rows[0] = mutated_candidate
    else:
        candidate_rows[0] = cast("dict[str, object]", candidate)

    with pytest.raises(ValueError, match=match):
        studio.build_hybrid_order_studio_panel(_monitor_records(), scenarios=[scenario])
