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
import dataclasses
import json
from typing import Any, cast

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
        ("scenario_id", "  ", "scenario_id must be a non-empty canonical string"),
        ("error_threshold", -1.0, "error_threshold must be finite and positive"),
        ("error_threshold", True, "must be numeric, got bool"),
        ("error_threshold", "x", "must be a numeric value"),
        (
            "proposed_reconfiguration_action",
            "  ",
            "proposed_reconfiguration_action must be a non-empty canonical string",
        ),
        ("operator_review_required", False, "operator_review_required must be true"),
        ("operator_review_required", "yes", "must be boolean"),
        ("execution_disabled", False, "execution_disabled must be true"),
        (
            "claim_boundary",
            "live_actuation",
            "claim_boundary must preserve the review-only boundary",
        ),
        (
            "blocked_live_execution_fields",
            [],
            "blocked_live_execution_fields must be a non-empty tuple",
        ),
        (
            "blocked_live_execution_fields",
            ["  "],
            "blocked_live_execution_fields must be a non-empty canonical string",
        ),
        ("serialisable_evidence", "not-a-dict", "serialisable_evidence must be a dict"),
        ("predicted_phase", "not-a-vector", "phase vectors must be JSON arrays"),
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
        ("error_norm", -0.1, "error_norm must be non-negative"),
        ("max_abs_error", float("inf"), "max_abs_error must be finite"),
        ("mean_abs_error", float("inf"), "mean_abs_error must be finite"),
        ("threshold", -1.0, "threshold must be finite and positive"),
        (
            "threshold",
            0.123456,
            "self-model error threshold contradicts error_threshold",
        ),
        ("mean_abs_error", 1.0, "mean_abs_error must not exceed error_norm"),
        ("error_norm", 1.0, "error_norm must not exceed max_abs_error"),
        ("metric", "  ", "metric must be a non-empty canonical string"),
    ],
)
def test_validate_scenario_record_rejects_corrupt_error_payload(
    field, value, match
) -> None:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[0])
    record["self_model_error"][field] = value
    with pytest.raises(ValueError, match=match):
        examples._validate_scenario_record(record)


def test_coerce_scalar_accepts_numpy_scalars() -> None:
    assert examples._coerce_scalar(np.float64(1.5), label="x") == 1.5
    assert examples._coerce_scalar(np.int64(3), label="y") == 3.0


def test_coerce_error_payload_reads_object_error_result_fields() -> None:
    class _ObjectErrorResult:
        within_threshold = True
        metric = "object_metric"

    payload = examples._coerce_error_payload(
        cast("Any", _ObjectErrorResult()),
        predicted_phase=np.zeros(4, dtype=np.float64),
        observed_phase=np.zeros(4, dtype=np.float64),
        error_threshold=0.5,
    )

    assert payload["within_threshold"] is True
    assert payload["metric"] == "object_metric"


def test_coerce_error_payload_defaults_metric_for_sparse_mapping() -> None:
    payload = examples._coerce_error_payload(
        {},
        predicted_phase=np.zeros(4, dtype=np.float64),
        observed_phase=np.zeros(4, dtype=np.float64),
        error_threshold=0.5,
    )

    assert payload["metric"] == "circular_rms_error"
    assert payload["within_threshold"] is True


def test_sparse_error_payload_default_includes_maximum_error_gate() -> None:
    payload = examples._coerce_error_payload(
        {},
        predicted_phase=np.zeros(4, dtype=np.float64),
        observed_phase=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        error_threshold=0.6,
    )

    assert payload["error_norm"] < payload["threshold"]
    assert payload["max_abs_error"] > payload["threshold"]
    assert payload["within_threshold"] is False


def test_to_audit_record_rejects_mismatched_stored_hash() -> None:
    proposal = examples._build_static_proposals()[0]
    with pytest.raises(ValueError, match="mismatched scenario_hash"):
        dataclasses.replace(proposal, scenario_hash="f" * 64)


def test_to_audit_record_accepts_matching_stored_hash() -> None:
    record = examples.build_self_model_reconfiguration_examples()[0]
    proposal = dataclasses.replace(
        examples._build_static_proposals()[0],
        scenario_hash=record["scenario_hash"],
    )

    assert proposal.to_audit_record() == record


def test_validate_proposal_rejects_phase_shape_mismatch() -> None:
    proposal = examples._build_static_proposals()[0]
    shorter = np.asarray(proposal.observed_phase, dtype=np.float64)[:-1]
    with pytest.raises(ValueError, match="predicted and observed phase vectors must"):
        dataclasses.replace(proposal, observed_phase=shorter)


def test_validate_scenario_record_rejects_empty_stored_hash() -> None:
    with pytest.raises(ValueError, match="has invalid scenario_hash"):
        examples._validate_scenario_record(_corrupt_record("scenario_hash", ""))


def test_coerce_error_payload_rejects_non_bool_object_within_threshold() -> None:
    class _BadObjectErrorResult:
        within_threshold = "maybe"
        metric = "m"

    with pytest.raises(ValueError, match="within_threshold must be boolean"):
        examples._coerce_error_payload(
            cast("Any", _BadObjectErrorResult()),
            predicted_phase=np.zeros(4, dtype=np.float64),
            observed_phase=np.zeros(4, dtype=np.float64),
            error_threshold=0.5,
        )


@pytest.mark.parametrize(
    "value",
    [
        [True, False],
        np.array([np.bool_(True), np.bool_(False)]),
        ["0.1", "0.2"],
        [0.1, "0.2"],
        [0.1 + 0.0j, 0.2 + 0.0j],
        np.array([0.1, object()], dtype=object),
    ],
)
def test_coerce_vector_rejects_coercive_numeric_aliases(value: object) -> None:
    with pytest.raises(ValueError, match="must be a real numeric vector"):
        examples._coerce_vector(value, label="phase")


def test_coerce_vector_normalises_array_protocol_failures() -> None:
    class _BrokenArray:
        def __array__(self) -> np.ndarray:
            raise RuntimeError("broken protocol")

    with pytest.raises(ValueError, match="must be a real numeric vector"):
        examples._coerce_vector(_BrokenArray(), label="phase")


def test_coerce_vector_preserves_real_numeric_object_arrays() -> None:
    vector = examples._coerce_vector(
        np.array([np.float64(0.1), 2], dtype=object),
        label="phase",
    )

    assert vector.tolist() == [0.1, 2.0]
    assert vector.dtype == np.float64


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scenario_id", 17, "scenario_id must be a non-empty canonical string"),
        (
            "proposed_reconfiguration_action",
            17,
            "proposed_reconfiguration_action must be a non-empty canonical string",
        ),
        ("error_threshold", True, "error_threshold must be numeric, got bool"),
        ("error_threshold", "0.2", "error_threshold must be a numeric value"),
        (
            "error_threshold",
            float("inf"),
            "error_threshold must be finite and positive",
        ),
        ("scenario_hash", 17, "scenario_hash must be a string"),
        (
            "scenario_hash",
            "A" * 64,
            "scenario_hash must be 64 lowercase hexadecimal characters",
        ),
        (
            "scenario_hash",
            "abc",
            "scenario_hash must be 64 lowercase hexadecimal characters",
        ),
    ],
)
def test_direct_proposal_rejects_noncanonical_fields(
    field: str, value: object, match: str
) -> None:
    proposal = examples._build_static_proposals()[0]
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(proposal, **{field: value})


def test_direct_proposal_canonicalises_vectors_and_json_evidence() -> None:
    proposal = examples._build_static_proposals()[0]
    replacement = dataclasses.replace(
        proposal,
        predicted_phase=proposal.predicted_phase.tolist(),
        observed_phase=proposal.observed_phase.tolist(),
        serialisable_evidence={"nested": [1, 2.5, "three"]},
    )

    record = replacement.to_audit_record()

    assert isinstance(replacement.predicted_phase, np.ndarray)
    assert replacement.predicted_phase.dtype == np.float64
    assert replacement.predicted_phase.flags.writeable is False
    assert record["serialisable_evidence"] == {"nested": [1, 2.5, "three"]}


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        ({"array": np.array([1.0])}, "JSON-serialisable"),
        ({"nan": float("nan")}, "JSON-serialisable"),
        ({1: "non-string key"}, "string keys"),
    ],
)
def test_direct_proposal_rejects_non_json_evidence(
    evidence: dict[object, object], match: str
) -> None:
    proposal = examples._build_static_proposals()[0]
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(proposal, serialisable_evidence=evidence)


def test_direct_proposal_rejects_duplicate_blocked_fields() -> None:
    proposal = examples._build_static_proposals()[0]
    duplicate = ("live_actuation", "live_actuation")

    with pytest.raises(
        ValueError, match="blocked_live_execution_fields must be unique"
    ):
        dataclasses.replace(proposal, blocked_live_execution_fields=duplicate)


def test_direct_proposal_rejects_contradictory_error_payload() -> None:
    proposal = examples._build_static_proposals()[0]
    contradictory = {
        "error_norm": 0.5,
        "max_abs_error": 0.5,
        "mean_abs_error": 0.4,
        "threshold": proposal.error_threshold,
        "within_threshold": True,
        "metric": "circular_rms_error",
    }

    with pytest.raises(ValueError, match="within_threshold contradicts"):
        dataclasses.replace(proposal, self_model_error=cast("Any", contradictory))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda record: record.__setitem__(
                "unsafe_due_to_threshold",
                not record["unsafe_due_to_threshold"],
            ),
            "unsafe_due_to_threshold contradicts",
        ),
        (
            lambda record: record["phase_error_summary"].__setitem__("max", 0.0),
            "phase_error_summary contradicts",
        ),
        (
            lambda record: record.__setitem__("unexpected_unsigned_field", True),
            "unexpected fields",
        ),
    ],
)
def test_record_replay_rejects_unbound_or_derived_tampering(
    mutate: Any, match: str
) -> None:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[2])
    mutate(record)

    with pytest.raises(ValueError, match=match):
        examples._validate_scenario_record(record)


def test_record_replay_requires_blocked_fields_as_json_array() -> None:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[0])
    record["blocked_live_execution_fields"] = "live_actuation"

    with pytest.raises(
        ValueError, match="blocked_live_execution_fields must be a list"
    ):
        examples._validate_scenario_record(record)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("predicted_phase", np.array([0.1]), "phase vectors must be JSON arrays"),
        ("self_model_error", "invalid", "self_model_error must be a JSON object"),
        (
            "phase_error_summary",
            [],
            "phase_error_summary must be a JSON object",
        ),
    ],
)
def test_record_replay_requires_json_container_types(
    field: str, value: object, match: str
) -> None:
    record = copy.deepcopy(examples.build_self_model_reconfiguration_examples()[0])
    record[field] = value

    with pytest.raises(ValueError, match=match):
        examples._validate_scenario_record(record)
