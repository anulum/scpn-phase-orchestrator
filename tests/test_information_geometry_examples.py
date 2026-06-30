# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Information-geometry control fixture tests

"""Tests for deterministic information-geometry control fixtures."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from scpn_phase_orchestrator.supervisor.information_geometry_examples import (
    DistributionPair,
    InformationGeometryBoundary,
    InformationGeometryScenario,
    _compute_scenario_hash,
    _validate_control_gradient,
    _validate_information_geometry_scenario,
    _validate_scenario_record,
    build_information_geometry_control_scenarios,
)


def _contains_numpy(payload: object) -> bool:
    if isinstance(payload, dict):
        return any(_contains_numpy(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_numpy(value) for value in payload)
    return isinstance(payload, np.ndarray)


def _bad_scenario(**overrides: object) -> InformationGeometryScenario:
    params = {
        "domain": "power_grid",
        "scenario_id": "unit_test_bad",
        "distributions": DistributionPair(
            current_distribution=np.array([0.2, 0.3, 0.5], dtype=np.float64),
            target_distribution=np.array([0.4, 0.1, 0.5], dtype=np.float64),
        ),
        "objective_labels": ("load_stability", "frequency_damping"),
        "control_gradient": (("K", 0.02),),
        "max_step": 0.1,
    }
    params.update(overrides)
    return InformationGeometryScenario(**params)


def test_build_scenarios_have_expected_domains_and_required_fields() -> None:
    scenarios = build_information_geometry_control_scenarios()

    assert isinstance(scenarios, tuple)
    assert len(scenarios) >= 3
    domains = {scenario["domain"] for scenario in scenarios}
    assert {
        "power_grid",
        "cardiac_rhythm",
        "cyber_industrial",
    } <= domains

    for scenario in scenarios:
        assert scenario["claim_boundary"] == InformationGeometryBoundary
        assert scenario["non_actuating"] is True
        assert scenario["execution_disabled"] is True
        assert scenario["control_gradient"]
        assert scenario["max_step"] > 0
        assert isinstance(scenario["scenario_hash"], str)
        assert len(scenario["scenario_hash"]) == 64


def test_build_scenarios_are_deterministic() -> None:
    first = build_information_geometry_control_scenarios()
    second = build_information_geometry_control_scenarios()

    assert [scenario["scenario_hash"] for scenario in first] == [
        scenario["scenario_hash"] for scenario in second
    ]
    assert [scenario["scenario_id"] for scenario in first] == [
        scenario["scenario_id"] for scenario in second
    ]


def test_records_are_json_safe() -> None:
    scenarios = build_information_geometry_control_scenarios()

    for scenario in scenarios:
        assert not _contains_numpy(scenario)
        json.dumps(scenario, allow_nan=False, sort_keys=True)


def test_distributions_are_finite_and_have_positive_mass() -> None:
    scenarios = build_information_geometry_control_scenarios()

    for scenario in scenarios:
        current = scenario["current_distribution"]
        target = scenario["target_distribution"]
        assert len(current) == len(target)
        assert all(value >= 0.0 for value in current)
        assert all(value >= 0.0 for value in target)
        assert sum(current) > 0.0
        assert sum(target) > 0.0

        for summary in (
            scenario["current_distribution_summary"],
            scenario["target_distribution_summary"],
        ):
            assert math.isfinite(summary["sum"])
            assert summary["sum"] > 0.0
            assert math.isfinite(summary["mean"])
            assert math.isfinite(summary["std"])
            assert math.isfinite(summary["min"])
            assert math.isfinite(summary["max"])
            assert isinstance(summary["count"], int)
            assert summary["count"] == len(current)

        assert scenario["control_gradient"]
        for knob_name, value in scenario["control_gradient"]:
            assert isinstance(knob_name, str)
            assert math.isfinite(float(value))


def test_flags_and_boundary_are_review_only() -> None:
    scenarios = build_information_geometry_control_scenarios()
    for scenario in scenarios:
        assert scenario["claim_boundary"] == InformationGeometryBoundary
        assert scenario["non_actuating"] is True
        assert scenario["execution_disabled"] is True


def test_rejects_malformed_scenarios() -> None:
    with pytest.raises(ValueError, match="unsupported domain"):
        _validate_information_geometry_scenario(_bad_scenario(domain="invalid_domain"))

    with pytest.raises(ValueError, match="must contain only finite values"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array([np.nan, 0.3, 0.4], dtype=np.float64),
                    target_distribution=np.array([0.4, 0.2, 0.4], dtype=np.float64),
                )
            )
        )

    with pytest.raises(ValueError, match="non-negative"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array([-0.2, 0.3, 0.9], dtype=np.float64),
                    target_distribution=np.array([0.3, 0.3, 0.4], dtype=np.float64),
                )
            )
        )

    with pytest.raises(ValueError, match="positive total mass"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    target_distribution=np.array([0.2, 0.3, 0.5], dtype=np.float64),
                )
            )
        )

    with pytest.raises(ValueError, match="matching shape"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array([0.2, 0.3, 0.5], dtype=np.float64),
                    target_distribution=np.array([0.5, 0.5], dtype=np.float64),
                )
            )
        )

    with pytest.raises(ValueError, match="strictly positive"):
        _validate_information_geometry_scenario(_bad_scenario(max_step=0.0))

    with pytest.raises(ValueError, match="non_actuating=True"):
        _validate_information_geometry_scenario(_bad_scenario(non_actuating=False))

    with pytest.raises(ValueError, match="non-empty"):
        _validate_control_gradient(())


def test_rejects_boolean_alias_distribution_values() -> None:
    with pytest.raises(ValueError, match="current_distribution"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array([True, 0.3, 0.7], dtype=object),
                    target_distribution=np.array([0.4, 0.2, 0.4], dtype=np.float64),
                )
            )
        )


def test_rejects_complex_alias_distribution_values() -> None:
    with pytest.raises(ValueError, match="current_distribution"):
        _validate_information_geometry_scenario(
            _bad_scenario(
                distributions=DistributionPair(
                    current_distribution=np.array(
                        [0.2 + 0.1j, 0.3, 0.5],
                        dtype=object,
                    ),
                    target_distribution=np.array([0.4, 0.2, 0.4], dtype=np.float64),
                )
            )
        )


def test_rejects_boolean_alias_control_gradient_and_max_step() -> None:
    with pytest.raises(ValueError, match="control_gradient"):
        _validate_information_geometry_scenario(
            _bad_scenario(control_gradient=(("K", np.bool_(True)),))
        )

    with pytest.raises(ValueError, match="max_step"):
        _validate_information_geometry_scenario(_bad_scenario(max_step=np.bool_(True)))


def test_scenario_hash_rejects_non_finite_canonical_payload() -> None:
    scenario = _bad_scenario(control_gradient=(("K", float("nan")),))

    with pytest.raises(ValueError):
        _compute_scenario_hash(
            domain=scenario.domain,
            scenario_id=scenario.scenario_id,
            distributions=scenario.distributions,
            objective_labels=scenario.objective_labels,
            control_gradient=scenario.control_gradient,
            knob_hints=scenario.knob_hints,
            max_step=scenario.max_step,
            non_actuating=scenario.non_actuating,
            execution_disabled=scenario.execution_disabled,
            claim_boundary=scenario.claim_boundary,
        )


def test_rejects_records_with_invalid_hash_and_malformed_shapes() -> None:
    scenario = build_information_geometry_control_scenarios()[0]
    bad_record = deepcopy(scenario)
    bad_record["scenario_hash"] = "0" * 64
    with pytest.raises(ValueError, match="invalid scenario_hash"):
        _validate_scenario_record(bad_record)

    bad_shape = deepcopy(scenario)
    bad_shape["current_distribution"] = [0.4, 0.4]
    with pytest.raises(ValueError, match="matching shape"):
        _validate_scenario_record(bad_shape)


def test_record_rejects_complex_alias_distribution_and_gradient_values() -> None:
    scenario = build_information_geometry_control_scenarios()[0]

    bad_distribution = deepcopy(scenario)
    bad_distribution["current_distribution"] = [0.4 + 0.1j, 0.3, 0.2, 0.1]
    with pytest.raises(ValueError, match="current_distribution"):
        _validate_scenario_record(bad_distribution)

    bad_gradient = deepcopy(scenario)
    bad_gradient["control_gradient"] = [["K", np.complex128(0.1 + 0.2j)]]
    with pytest.raises(ValueError, match="control_gradient"):
        _validate_scenario_record(bad_gradient)


_IG_MISSING = object()


def _ig_record(field: str, value: object) -> dict:
    record = deepcopy(build_information_geometry_control_scenarios()[0])
    if value is _IG_MISSING:
        del record[field]
    else:
        record[field] = value
    return record


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("domain", _IG_MISSING, "missing required record field"),
        ("domain", 123, "record domain must be a string"),
        ("scenario_id", 123, "record scenario_id must be a string"),
        ("objective_labels", "x", "record objective_labels must be a list"),
        ("control_gradient", "x", "record control_gradient must be a list"),
        ("knob_hints", "x", "record knob_hints must be a list"),
        ("non_actuating", "x", "record non_actuating must be a boolean"),
        ("execution_disabled", "x", "record execution_disabled must be a boolean"),
        ("claim_boundary", 123, "record claim_boundary must be a string"),
        ("current_distribution", 5, "record current_distribution must be a sequence"),
        ("target_distribution", 5, "record target_distribution must be a sequence"),
        ("max_step", "x", "record max_step must be numeric"),
        ("objective_labels", [123], "record objective_labels must contain strings"),
        ("knob_hints", [123], "record knob_hints must contain strings"),
        ("control_gradient", [[1, 2, 3]], "control_gradient entries must be pairs"),
        ("control_gradient", [[123, 0.5]], "control_gradient knobs must be strings"),
        ("control_gradient", [["k", "x"]], "control_gradient values must be numeric"),
        (
            "control_gradient",
            [["k", float("inf")]],
            "control_gradient values must be finite",
        ),
        ("current_distribution", [[0.1, 0.2]], "must be a 1D array"),
        ("current_distribution", [], "must not be empty"),
        ("non_actuating", False, "requires non_actuating=True"),
        ("execution_disabled", False, "requires execution_disabled=True"),
        ("claim_boundary", "live_actuation", "requires claim_boundary="),
        ("scenario_hash", 123, "record scenario_hash must be a string"),
        ("scenario_hash", "0" * 64, "has invalid scenario_hash"),
    ],
)
def test_ig_validate_scenario_record_rejects_corruptions(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_scenario_record(_ig_record(field, value))


def test_ig_record_rejects_empty_domain_and_non_finite_max_step() -> None:
    with pytest.raises(ValueError, match="domain must be a non-empty string"):
        _validate_scenario_record(_ig_record("domain", ""))
    with pytest.raises(ValueError, match="must be finite"):
        _validate_scenario_record(_ig_record("max_step", float("inf")))


def test_ig_validate_distribution_pair_rejects_non_pair() -> None:
    with pytest.raises(ValueError, match="distributions must be a DistributionPair"):
        from scpn_phase_orchestrator.supervisor.information_geometry_examples import (
            _validate_distribution_pair,
        )

        _validate_distribution_pair("not-a-pair")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("gradient", "match"),
    [
        ((), "control_gradient must be a non-empty tuple"),
        ((("  ", 0.5),), "control_gradient keys must be non-empty strings"),
        ((("k", True),), "control_gradient values must be finite numbers"),
        ((("k", float("inf")),), "control_gradient values must be finite"),
    ],
)
def test_ig_validate_control_gradient_rejects_corruptions(gradient, match) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_control_gradient(gradient)


def test_ig_validate_scenario_rejects_non_scenario() -> None:
    with pytest.raises(
        ValueError, match="scenario must be an InformationGeometryScenario"
    ):
        _validate_information_geometry_scenario("not-a-scenario")  # type: ignore[arg-type]


def test_rejects_scenario_identity_and_objective_label_violations() -> None:
    with pytest.raises(ValueError, match="scenario_id must be a non-empty string"):
        _validate_information_geometry_scenario(_bad_scenario(scenario_id="  "))

    with pytest.raises(ValueError, match="objective_labels must be a non-empty tuple"):
        _validate_information_geometry_scenario(_bad_scenario(objective_labels=()))

    with pytest.raises(ValueError, match="objective_labels must contain non-empty"):
        _validate_information_geometry_scenario(
            _bad_scenario(objective_labels=("load_stability", "  "))
        )

    with pytest.raises(ValueError, match="must be a numeric value"):
        _validate_information_geometry_scenario(_bad_scenario(max_step="fast"))
