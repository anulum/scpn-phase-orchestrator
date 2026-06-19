# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Self-model reconfiguration fixture tests

"""Tests for deterministic self-model reconfiguration replay evidence."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor.self_model_examples as examples


def test_build_examples_have_required_domains_and_safe_gating() -> None:
    scenarios = examples.build_self_model_reconfiguration_examples()

    assert len(scenarios) >= 3
    domains = {scenario["domain"] for scenario in scenarios}
    assert {
        "power_grid",
        "cardiac_rhythm",
        "traffic_flow",
        "cyber_industrial",
    } <= domains
    for scenario in scenarios:
        assert scenario["claim_boundary"] == examples.SelfModelBoundary
        assert scenario["operator_review_required"] is True
        assert scenario["execution_disabled"] is True
        assert isinstance(scenario["blocked_live_execution_fields"], list)
        assert scenario["blocked_live_execution_fields"]
        assert all(
            isinstance(field, str) and field.strip()
            for field in scenario["blocked_live_execution_fields"]
        )
        assert isinstance(scenario["self_model_error"], dict)
        assert "within_threshold" in scenario["self_model_error"]
        assert isinstance(scenario["self_model_error"]["within_threshold"], bool)
        assert scenario["self_model_error"]["threshold"] > 0.0
        assert scenario["proposed_reconfiguration_action"]
        assert isinstance(scenario["serialisable_evidence"], dict)
        assert scenario["serialisable_evidence"]
        assert scenario["scenario_hash"]


def test_records_are_deterministic() -> None:
    first = examples.build_self_model_reconfiguration_examples()
    second = examples.build_self_model_reconfiguration_examples()

    assert [scenario["scenario_id"] for scenario in first] == [
        scenario["scenario_id"] for scenario in second
    ]
    assert [scenario["scenario_hash"] for scenario in first] == [
        scenario["scenario_hash"] for scenario in second
    ]
    assert first == second


def test_records_are_json_serialisable_and_no_ndarrays() -> None:
    scenarios = examples.build_self_model_reconfiguration_examples()

    for scenario in scenarios:
        json.dumps(scenario, allow_nan=False, sort_keys=True)
        assert not examples._contains_arrays(scenario)


def test_noisy_scenario_flags_threshold_exceedance() -> None:
    scenarios = examples.build_self_model_reconfiguration_examples()
    unsafe = [
        scenario
        for scenario in scenarios
        if scenario["unsafe_due_to_threshold"] is True
    ]
    assert unsafe, "expected at least one unsafe scenario"

    for scenario in unsafe:
        assert scenario["self_model_error"]["within_threshold"] is False
        assert (
            scenario["self_model_error"]["error_norm"]
            > scenario["self_model_error"]["threshold"]
        )


def test_invalid_hash_and_shape_are_rejected() -> None:
    good = examples.build_self_model_reconfiguration_examples()
    bad = copy.deepcopy(good[0])
    bad["scenario_hash"] = "0" * 64
    with pytest.raises(ValueError, match="mismatched scenario_hash"):
        examples._validate_scenario_record(bad)

    bad_shape = copy.deepcopy(good[0])
    bad_shape["observed_phase"] = bad_shape["observed_phase"][:-1]
    with pytest.raises(ValueError, match="mismatch"):
        examples._validate_scenario_record(bad_shape)


def test_build_examples_integrate_monitor_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[float, ...], tuple[float, ...], float]] = []

    def fake_compute_self_model_error(
        *,
        predicted_phases: np.ndarray,
        observed_phases: np.ndarray,
        tolerance: float,
        max_abs_tolerance: float,
        domain: str,
        scenario_id: str,
        channel_labels: tuple[str, ...],
    ) -> dict[str, float | bool | str]:
        calls.append(
            (
                tuple(float(v) for v in np.asarray(predicted_phases).tolist()),
                tuple(float(v) for v in np.asarray(observed_phases).tolist()),
                float(tolerance),
            )
        )
        return {
            "error_norm": 0.42,
            "max_abs_error": 0.45,
            "mean_abs_error": 0.30,
            "threshold": tolerance,
            "within_threshold": False,
            "metric": "test_metric",
        }

    monkeypatch.setattr(
        examples,
        "compute_self_model_error",
        fake_compute_self_model_error,
    )

    scenarios = examples.build_self_model_reconfiguration_examples()

    assert len(calls) == len(scenarios)
    assert all(
        scenario["self_model_error"]["metric"] == "test_metric"
        for scenario in scenarios
    )
    assert all(
        scenario["self_model_error"]["within_threshold"] is False
        for scenario in scenarios
    )
    assert all(scenario["unsafe_due_to_threshold"] for scenario in scenarios)


_MISSING = object()


def _corrupt_record(field: str, value: object) -> dict:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[0])
    if value is _MISSING:
        del record[field]
    else:
        record[field] = value
    return record


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("domain", _MISSING, "missing required fields"),
        ("scenario_hash", 123, "scenario_hash must be a string"),
        ("domain", "not_a_domain", "invalid domain"),
        ("scenario_id", "  ", "scenario_id must be a non-empty string"),
        ("error_threshold", -1.0, "error_threshold must be positive"),
        ("error_threshold", True, "must be numeric, got bool"),
        ("error_threshold", "x", "must be a numeric value"),
        ("proposed_reconfiguration_action", "  ", "needs non-empty proposed action"),
        ("operator_review_required", False, "requires operator_review_required=True"),
        ("operator_review_required", "yes", "must be boolean"),
        ("execution_disabled", False, "requires execution_disabled=True"),
        ("claim_boundary", "live_actuation", "has invalid claim boundary"),
        ("blocked_live_execution_fields", [], "requires blocked fields"),
        ("blocked_live_execution_fields", ["  "], "must be non-empty strings"),
        ("serialisable_evidence", "not-a-dict", "serialisable_evidence must be a dict"),
        ("predicted_phase", "not-a-vector", "must be a float-convertible vector"),
        ("predicted_phase", [[0.1, 0.2]], "must be one-dimensional"),
        ("predicted_phase", [], "must contain at least one value"),
        ("predicted_phase", [float("nan"), 0.1], "must contain only finite values"),
    ],
)
def test_validate_scenario_record_rejects_corruptions(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        examples._validate_scenario_record(_corrupt_record(field, value))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("within_threshold", "maybe", "within_threshold must be boolean"),
        ("error_norm", float("inf"), "error_norm must be finite"),
        ("max_abs_error", float("inf"), "max_abs_error must be finite"),
        ("mean_abs_error", float("inf"), "mean_abs_error must be finite"),
        ("threshold", -1.0, "threshold must be finite and positive"),
    ],
)
def test_validate_scenario_record_rejects_corrupt_error_payload(
    field, value, match
) -> None:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[0])
    record["self_model_error"][field] = value
    with pytest.raises(ValueError, match=match):
        examples._validate_scenario_record(record)
