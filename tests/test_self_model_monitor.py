# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for self-model error monitor

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.self_model import (
    SelfModelErrorThresholdConfig,
    compute_self_model_error,
)


def test_self_model_error_threshold_pass_and_fail() -> None:
    observed = np.array(
        [[0.0, 0.4, 0.8], [1.0, 1.2, 1.4]],
        dtype=np.float64,
    )
    predicted = np.array(
        [[0.1, 0.1, 1.0], [1.2, 1.3, 1.1]],
        dtype=np.float64,
    )

    pass_case = compute_self_model_error(
        observed,
        predicted,
        tolerance=1.0,
        max_abs_tolerance=1.0,
        channel_labels=("a", "b"),
    )
    fail_case = compute_self_model_error(
        observed,
        predicted,
        tolerance=0.05,
        max_abs_tolerance=0.05,
    )

    assert pass_case.breached is False
    assert fail_case.breached is True
    assert pass_case.order_breached is None


def test_self_model_error_shape_and_finite_validation() -> None:
    observed = np.array([0.0, 1.0, 2.0])
    predicted = np.array([0.0, 1.1, 2.1])
    compute_self_model_error(observed, predicted, tolerance=1.0, max_abs_tolerance=1.0)

    with pytest.raises(ValueError, match="matching shapes"):
        compute_self_model_error(
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0, 3.0]]),
        )

    with pytest.raises(ValueError, match="finite"):
        compute_self_model_error(
            np.array([0.0, np.nan]),
            np.array([0.0, 0.0]),
        )

    with pytest.raises(ValueError, match="must be strictly positive"):
        compute_self_model_error(
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            channel_weights=(1.0, -1.0),
        )

    with pytest.raises(ValueError, match="channel_labels length"):
        compute_self_model_error(
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            channel_labels=("left",),
        )

    with pytest.raises(ValueError, match="must contain at least one"):
        compute_self_model_error(
            np.array([]),
            np.array([]),
        )


def test_self_model_error_deterministic_hashes() -> None:
    observed = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    predicted = np.array([[0.1, 0.9], [1.2, 2.1]], dtype=np.float64)

    first = compute_self_model_error(
        observed,
        predicted,
        tolerance=2.0,
        max_abs_tolerance=1.0,
        channel_labels=("left", "right"),
        domain="integration",
        scenario_id="sm-test-1",
    )
    second = compute_self_model_error(
        observed,
        predicted,
        tolerance=2.0,
        max_abs_tolerance=1.0,
        channel_labels=("left", "right"),
        domain="integration",
        scenario_id="sm-test-1",
    )

    assert first.record_hash == second.record_hash
    assert first.record_hash == first.to_audit_record()["record_hash"]
    assert json.loads(
        json.dumps(first.to_audit_record(), allow_nan=False)
    ) == json.loads(
        json.dumps(second.to_audit_record(), allow_nan=False),
    )


def test_self_model_error_weighted_metrics() -> None:
    observed = np.array(
        [[0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    predicted = np.array(
        [[1.0, 1.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    result = compute_self_model_error(
        observed,
        predicted,
        channel_weights=(1.0, 3.0),
    )

    assert result.weighted_rmse == pytest.approx(
        np.sqrt(0.25 * 1.0**2 + 0.75 * 10.0**2)
    )
    assert result.weighted_mae == pytest.approx(0.25 * 1.0 + 0.75 * 10.0)
    assert result.weighted_max_abs_error == pytest.approx(0.75 * 10.0)


def test_self_model_error_channel_labels_default_and_explicit() -> None:
    observed = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    predicted = np.array([0.1, 0.1, 0.5], dtype=np.float64)

    labeled = compute_self_model_error(
        observed,
        predicted,
        channel_labels=("x",),
    )
    unlabeled = compute_self_model_error(observed, predicted)

    assert labeled.channel_labels == ("x",)
    assert unlabeled.channel_labels == ("channel_0",)
    assert labeled.channel_count == 1
    assert unlabeled.sample_count == 3


def test_self_model_error_optional_order_metrics() -> None:
    observed = np.array(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    predicted = np.array(
        [[0.1, 0.2], [1.0, 1.4]],
        dtype=np.float64,
    )

    result = compute_self_model_error(
        observed,
        predicted,
        observed_order=np.array([0.0, 1.0]),
        predicted_order=np.array([0.2, 1.4]),
        tolerance=0.5,
        max_abs_tolerance=0.5,
    )

    assert result.order_rmse is not None
    assert result.order_mae is not None
    assert result.order_max_abs_error is not None
    assert result.order_breached is False

    with pytest.raises(
        ValueError,
        match="both observed_order and predicted_order must be provided together",
    ):
        compute_self_model_error(
            observed,
            predicted,
            observed_order=np.array([0.0, 1.0]),
        )


def test_self_model_error_threshold_config_and_audit_serialisation() -> None:
    thresholds = SelfModelErrorThresholdConfig(
        tolerance=0.1,
        max_abs_tolerance=0.2,
    )
    assert thresholds.tolerance == 0.1
    assert thresholds.max_abs_tolerance == 0.2

    result = compute_self_model_error(
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array([0.05, 1.01, 2.08], dtype=np.float64),
        channel_labels=("phases",),
        channel_weights=(2.0,),
        tolerance=thresholds.tolerance,
        max_abs_tolerance=thresholds.max_abs_tolerance,
        domain="digital_twin",
        scenario_id="sm-serialise",
    )

    record = result.to_audit_record()

    assert record["domain"] == "digital_twin"
    assert record["scenario_id"] == "sm-serialise"
    assert (
        record["claim_boundary"] == "self_model_error_monitor_not_live_reconfiguration"
    )
    assert record["non_actuating"] is True
    assert record["execution_disabled"] is True
    assert json.loads(json.dumps(record, allow_nan=False))
