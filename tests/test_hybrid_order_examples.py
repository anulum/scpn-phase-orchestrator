# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid order-parameter fixture tests

"""Tests for deterministic entanglement-aware hybrid order-parameter fixtures."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.hybrid_order_examples import (
    HybridOrderScenario,
    HybridStateCandidate,
    _compute_scenario_hash,
    _validate_scenario,
    build_hybrid_order_parameter_scenarios,
)


def _contains_array(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(_contains_array(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_array(value) for value in payload)
    return isinstance(payload, np.ndarray)


def test_build_scenarios_have_expected_domains_and_candidate_types() -> None:
    scenarios = build_hybrid_order_parameter_scenarios()
    assert len(scenarios) >= 3

    domains = {scenario["domain"] for scenario in scenarios}
    assert {"quantum_simulation", "power_grid", "cardiac_rhythm"} <= domains

    for scenario in scenarios:
        assert scenario["non_actuating"] is True
        assert scenario["execution_disabled"] is True
        assert (
            scenario["claim_boundary"]
            == "quantum_cosimulation_monitor_not_qpu_execution"
        )
        assert isinstance(scenario["state_candidates"], list)
        assert len(scenario["state_candidates"]) >= 2
        candidate_types = {
            candidate["state_type"] for candidate in scenario["state_candidates"]
        }
        assert "product" in candidate_types
        assert "entangled" in candidate_types
        assert isinstance(scenario["scenario_hash"], str)
        assert scenario["scenario_hash"] == _compute_scenario_hash(
            HybridOrderScenario(
                domain=scenario["domain"],
                scenario_id=scenario["scenario_id"],
                phases=np.array(scenario["phases"], dtype=np.float64),
                qubit_count=int(scenario["qubit_count"]),
                bipartition=tuple(tuple(part) for part in scenario["bipartition"]),
                state_candidates=tuple(
                    HybridStateCandidate(
                        state_id=candidate["state_id"],
                        candidate_type=candidate["state_type"],
                        amplitudes=np.asarray(
                            [
                                complex(real, imag)
                                for real, imag in candidate["amplitudes"]
                            ],
                            dtype=np.complex128,
                        ),
                        entanglement_entropy=float(candidate["entanglement_entropy"]),
                        order_metric_r=float(candidate["order_metric_r"]),
                        order_metric_psi=float(candidate["order_metric_psi"]),
                        objective_labels=tuple(candidate["objective_labels"]),
                    )
                    for candidate in scenario["state_candidates"]
                ),
                objective_labels=tuple(scenario["objective_labels"]),
                non_actuating=scenario["non_actuating"],
                execution_disabled=scenario["execution_disabled"],
                claim_boundary=scenario["claim_boundary"],
                scenario_hash=scenario["scenario_hash"],
            )
        )


def test_scenarios_are_deterministic() -> None:
    first = build_hybrid_order_parameter_scenarios()
    second = build_hybrid_order_parameter_scenarios()

    assert [scenario["scenario_hash"] for scenario in first] == [
        scenario["scenario_hash"] for scenario in second
    ]
    assert first == second


def test_json_safe_and_no_raw_ndarrays() -> None:
    scenarios = build_hybrid_order_parameter_scenarios()

    for scenario in scenarios:
        assert not _contains_array(scenario)
        json.dumps(scenario, allow_nan=False, sort_keys=True)


def test_numeric_summaries_are_finite() -> None:
    scenarios = build_hybrid_order_parameter_scenarios()

    for scenario in scenarios:
        phases_summary = scenario["phases_summary"]
        assert isinstance(phases_summary["count"], int)
        assert phases_summary["count"] == len(scenario["phases"])
        assert math.isfinite(float(phases_summary["min"]))
        assert math.isfinite(float(phases_summary["max"]))
        assert math.isfinite(float(phases_summary["mean"]))
        assert math.isfinite(float(phases_summary["std"]))

        for candidate in scenario["state_candidates"]:
            amplitude_summary = candidate["amplitude_summary"]
            assert math.isfinite(float(amplitude_summary["min"]))
            assert math.isfinite(float(amplitude_summary["max"]))
            assert math.isfinite(float(amplitude_summary["mean"]))
            assert math.isfinite(float(amplitude_summary["std"]))
            assert math.isfinite(float(candidate["entanglement_entropy"]))
            assert math.isfinite(float(candidate["order_metric_r"]))
            assert math.isfinite(float(candidate["order_metric_psi"]))


def _build_valid_scenario() -> HybridOrderScenario:
    return HybridOrderScenario(
        domain="quantum_simulation",
        scenario_id="test_hybrid_quantum_validation",
        phases=np.array([0.1, 0.9], dtype=np.float64),
        qubit_count=2,
        bipartition=((0,), (1,)),
        state_candidates=(
            HybridStateCandidate(
                state_id="unit_product",
                candidate_type="product",
                amplitudes=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                entanglement_entropy=0.0,
                order_metric_r=0.55,
                order_metric_psi=0.62,
                objective_labels=("low_entanglement", "classical_sync"),
            ),
            HybridStateCandidate(
                state_id="unit_entangled",
                candidate_type="entangled",
                amplitudes=np.array(
                    [1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)],
                    dtype=np.complex128,
                ),
                entanglement_entropy=1.0,
                order_metric_r=0.50,
                order_metric_psi=0.60,
                objective_labels=("high_entanglement", "quantum_signal"),
            ),
        ),
        objective_labels=("quantum_cosimulation", "review_audit"),
        non_actuating=True,
        execution_disabled=True,
        claim_boundary="quantum_cosimulation_monitor_not_qpu_execution",
        scenario_hash="",
    )


def test_flags_and_boundaries() -> None:
    scenarios = build_hybrid_order_parameter_scenarios()

    for scenario in scenarios:
        assert scenario["non_actuating"] is True
        assert scenario["execution_disabled"] is True
        assert (
            scenario["claim_boundary"]
            == "quantum_cosimulation_monitor_not_qpu_execution"
        )
        for candidate in scenario["state_candidates"]:
            assert candidate["non_actuating"] is True
            assert candidate["execution_disabled"] is True
            assert (
                candidate["claim_boundary"]
                == "quantum_cosimulation_monitor_not_qpu_execution"
            )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda scenario: setattr(scenario, "domain", "invalid_domain"),
        lambda scenario: scenario.phases.__setitem__(0, float("nan")),
        lambda scenario: setattr(
            scenario,
            "phases",
            np.array([0.1, True], dtype=object),
        ),
        lambda scenario: setattr(scenario, "qubit_count", 4),
        lambda scenario: setattr(scenario, "qubit_count", True),
        lambda scenario: setattr(scenario, "bipartition", ((0,), (np.bool_(True),))),
        lambda scenario: setattr(scenario, "non_actuating", False),
        lambda scenario: setattr(scenario, "scenario_hash", "bad_hash"),
        lambda scenario: setattr(
            scenario.state_candidates[0],
            "candidate_type",
            "unknown",
        ),
        lambda scenario: setattr(
            scenario.state_candidates[0],
            "entanglement_entropy",
            -0.1,
        ),
        lambda scenario: setattr(scenario.state_candidates[0], "order_metric_r", 1.1),
        lambda scenario: setattr(
            scenario.state_candidates[0],
            "order_metric_psi",
            True,
        ),
        lambda scenario: setattr(
            scenario.state_candidates[0],
            "amplitudes",
            np.zeros(4, dtype=np.complex128),
        ),
    ],
)
def test_rejects_malformed_scenarios(
    mutator,
) -> None:
    # Start from a valid model and mutate a single field to create malformed input.
    scenario = _build_valid_scenario()
    _validate_scenario(scenario)
    _hash = _compute_scenario_hash(scenario)
    scenario.scenario_hash = _hash

    mutated = deepcopy(scenario)
    mutator(mutated)
    if mutated is scenario:
        raise AssertionError("mutation helper failed")

    with pytest.raises(ValueError):
        _validate_scenario(mutated)


def test_accepts_numpy_integer_qubit_scenario_contracts() -> None:
    scenario = _build_valid_scenario()
    scenario.qubit_count = np.int64(2)
    scenario.bipartition = ((np.int64(0),), (np.int64(1),))

    _validate_scenario(scenario)
