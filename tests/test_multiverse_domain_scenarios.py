# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multiverse domain scenario fixtures

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.supervisor.multiverse_examples import (
    CounterfactualBoundary,
    DomainScenario,
    _validate_scenario,
    build_multiverse_domain_scenarios,
)

SUPPORTED_COUNTERFACTUAL_KNOBS = {"K", "alpha", "zeta", "Psi"}


def _collect_arrays(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(_collect_arrays(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_collect_arrays(value) for value in payload)
    return isinstance(payload, np.ndarray)


def test_build_returns_expected_domains_and_candidate_counts() -> None:
    scenarios = build_multiverse_domain_scenarios()

    assert isinstance(scenarios, tuple)
    assert len(scenarios) >= 6

    domain_map = {scenario["domain"]: scenario for scenario in scenarios}
    assert set(domain_map).issuperset(
        {
            "power_grid",
            "cardiac_rhythm",
            "cyber_industrial",
            "traffic_flow",
            "manufacturing_spc",
            "plasma_control",
        }
    )

    for domain in {
        "power_grid",
        "cardiac_rhythm",
        "cyber_industrial",
        "traffic_flow",
        "manufacturing_spc",
        "plasma_control",
    }:
        payload = domain_map[domain]
        assert payload["claim_boundary"] == CounterfactualBoundary
        assert payload["non_actuating"] is True
        assert payload["execution_disabled"] is True
        assert isinstance(payload["branch_candidates"], list)
        assert len(payload["branch_candidates"]) >= 2


def test_build_is_deterministic() -> None:
    first = build_multiverse_domain_scenarios()
    second = build_multiverse_domain_scenarios()

    assert [scenario["scenario_hash"] for scenario in first] == [
        scenario["scenario_hash"] for scenario in second
    ]

    payload_one = {
        scenario["scenario_id"]: scenario["scenario_hash"] for scenario in first
    }
    payload_two = {
        scenario["scenario_id"]: scenario["scenario_hash"] for scenario in second
    }
    assert payload_one == payload_two


def test_json_safe_and_no_raw_ndarrays() -> None:
    scenarios = build_multiverse_domain_scenarios()

    for scenario in scenarios:
        assert not _collect_arrays(scenario)
        json.dumps(scenario, allow_nan=False, sort_keys=True)


def test_numeric_summaries_are_finite() -> None:
    scenarios = build_multiverse_domain_scenarios()

    for scenario in scenarios:
        for summary_key in ("initial_phases_summary", "initial_omegas_summary"):
            summary = scenario[summary_key]
            assert isinstance(summary["count"], int)
            assert summary["count"] == len(
                scenario[
                    "initial_phases" if "phases" in summary_key else "initial_omegas"
                ]
            )
            assert math.isfinite(summary["min"])
            assert math.isfinite(summary["max"])
            assert math.isfinite(summary["mean"])
            assert math.isfinite(summary["std"])


def test_domain_specific_objective_labels_and_candidate_flags() -> None:
    scenarios = build_multiverse_domain_scenarios()
    expected = {
        "power_grid": {
            "load_stability",
            "frequency_regulation",
            "islanding_resilience",
        },
        "cardiac_rhythm": {
            "arrhythmia_suppression",
            "cardio_stability",
            "oxygenation_support",
        },
        "cyber_industrial": {
            "attack_surface_reduction",
            "service_containment",
            "latency_regulation",
        },
        "traffic_flow": {
            "congestion_wave_damping",
            "spillback_prevention",
            "green_wave_stability",
        },
        "manufacturing_spc": {
            "process_drift_recovery",
            "scrap_rate_reduction",
            "line_balance_stability",
        },
        "plasma_control": {
            "mode_locking_stability",
            "edge_localised_mode_mitigation",
            "confinement_margin",
        },
    }

    for scenario in scenarios:
        expected_labels = expected[scenario["domain"]]
        labels = set(scenario["objective_labels"])
        assert expected_labels.issubset(labels)

        for candidate in scenario["branch_candidates"]:
            assert candidate["non_actuating"] is True
            assert candidate["execution_disabled"] is True
            assert candidate["claim_boundary"] == CounterfactualBoundary
            assert candidate["topology_variations"]
            assert isinstance(candidate["knob_variations"], list)
            assert isinstance(candidate["objective_labels"], list)
            for knob_name, knob_value in candidate["knob_variations"]:
                assert isinstance(knob_name, str)
                assert knob_name.strip()
                assert knob_name in SUPPORTED_COUNTERFACTUAL_KNOBS
                assert math.isfinite(float(knob_value))


def test_scenario_corpus_has_physics_safe_phase_vectors() -> None:
    scenarios = build_multiverse_domain_scenarios()
    total_candidate_count = 0

    for scenario in scenarios:
        phases = scenario["initial_phases"]
        omegas = scenario["initial_omegas"]
        total_candidate_count += len(scenario["branch_candidates"])

        assert len(phases) == len(omegas)
        assert len(phases) >= 5
        assert all(0.0 <= float(phase) < 2.0 * math.pi for phase in phases)
        assert all(math.isfinite(float(omega)) for omega in omegas)
        assert not np.isclose(np.std(np.asarray(omegas, dtype=np.float64)), 0.0)

    assert total_candidate_count >= 12


def test_rejects_malformed_scenarios() -> None:
    bad_scenario = DomainScenario(
        domain="power_grid",
        scenario_id="bad_nan_scenario",
        initial_phases=np.array([0.0, 0.1], dtype=np.float64),
        initial_omegas=np.array([np.nan, 60.0], dtype=np.float64),
        branch_candidates=(),
        objective_labels=("load_stability",),
    )

    with pytest.raises(ValueError):
        _validate_scenario(bad_scenario)
