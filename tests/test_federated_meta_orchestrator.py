# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated meta-orchestrator tests

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.supervisor.federated import (
    FederatedAggregationConfig,
    FederatedNodeUpdate,
    FederatedPolicyAggregationReport,
    build_federated_meta_orchestrator_manifest,
)


def _updates() -> tuple[dict[str, object], ...]:
    return (
        {
            "node_id": "site-a",
            "policy_delta": {"K": 0.10, "alpha": -0.02},
            "sample_count": 120,
            "local_loss": 0.21,
            "previous_audit_hash": "a" * 64,
            "privacy_epsilon_spent": 0.8,
        },
        {
            "node_id": "site-b",
            "policy_delta": {"K": 0.04, "alpha": -0.01},
            "sample_count": 80,
            "local_loss": 0.24,
            "previous_audit_hash": "b" * 64,
            "privacy_epsilon_spent": 0.6,
        },
        {
            "node_id": "site-c",
            "policy_delta": {"K": 0.08, "alpha": -0.03},
            "sample_count": 100,
            "local_loss": 0.19,
            "previous_audit_hash": "c" * 64,
            "privacy_epsilon_spent": 0.7,
        },
    )


def test_federated_manifest_is_deterministic_review_only_and_json_safe() -> None:
    first = build_federated_meta_orchestrator_manifest(
        _updates(),
        required_policy_keys=("K", "alpha"),
        clipping_norm=0.2,
        epsilon=1.0,
        delta=1e-6,
    )
    second = build_federated_meta_orchestrator_manifest(
        _updates(),
        required_policy_keys=("K", "alpha"),
        clipping_norm=0.2,
        epsilon=1.0,
        delta=1e-6,
    )

    assert first == second
    assert isinstance(first, FederatedPolicyAggregationReport)
    assert first.accepted_node_count == 3
    assert first.rejected_node_count == 0
    assert first.raw_time_series_received is False
    assert first.raw_data_export_permitted is False
    assert first.live_transport_permitted is False
    assert first.actuation_permitted is False
    assert first.non_actuating is True
    assert first.execution_disabled is True
    assert len(first.report_hash) == 64
    assert len(first.aggregate_hash) == 64
    json.loads(json.dumps(first.to_audit_record(), allow_nan=False))


def test_federated_manifest_weighted_aggregation_and_budget_accounting() -> None:
    report = build_federated_meta_orchestrator_manifest(
        _updates(),
        required_policy_keys=("K", "alpha"),
        clipping_norm=1.0,
        epsilon=1.0,
        delta=1e-6,
    )
    aggregate = dict(report.aggregate_delta)

    assert report.total_sample_count == 300
    assert aggregate["K"] == pytest.approx((0.10 * 120 + 0.04 * 80 + 0.08 * 100) / 300)
    assert aggregate["alpha"] == pytest.approx(
        (-0.02 * 120 + -0.01 * 80 + -0.03 * 100) / 300
    )
    assert report.privacy_budget_spent == pytest.approx(0.8)
    assert report.privacy_budget_remaining == pytest.approx(0.2)


def test_federated_manifest_clips_large_updates_without_raw_data() -> None:
    updates = list(_updates())
    updates[0] = {
        **updates[0],
        "policy_delta": {"K": 10.0, "alpha": 0.0},
    }
    report = build_federated_meta_orchestrator_manifest(
        tuple(updates),
        required_policy_keys=("K", "alpha"),
        clipping_norm=0.5,
        epsilon=1.0,
    )

    clipped = report.node_updates[0]
    assert isinstance(clipped, FederatedNodeUpdate)
    assert clipped.accepted is True
    assert clipped.clip_scale == pytest.approx(0.05)
    assert dict(clipped.policy_delta)["K"] == pytest.approx(0.5)


def test_federated_manifest_rejects_privacy_budget_exceedance() -> None:
    updates = list(_updates())
    updates[2] = {**updates[2], "privacy_epsilon_spent": 1.5}

    report = build_federated_meta_orchestrator_manifest(
        tuple(updates),
        required_policy_keys=("K", "alpha"),
        epsilon=1.0,
    )

    assert report.accepted_node_count == 2
    assert report.rejected_node_count == 1
    rejected = next(update for update in report.node_updates if not update.accepted)
    assert rejected.rejection_reasons == ("privacy_epsilon_exceeds_budget",)


def test_federated_manifest_fails_closed_for_raw_data_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="raw time-series"):
        build_federated_meta_orchestrator_manifest(
            ({**_updates()[0], "raw_time_series": [1.0, 2.0]},),
            required_policy_keys=("K", "alpha"),
        )
    with pytest.raises(ValueError, match="missing required keys"):
        build_federated_meta_orchestrator_manifest(
            ({**_updates()[0], "policy_delta": {"K": 0.1}},),
            required_policy_keys=("K", "alpha"),
        )
    with pytest.raises(ValueError, match="previous_audit_hash"):
        build_federated_meta_orchestrator_manifest(
            ({**_updates()[0], "previous_audit_hash": "bad"},),
            required_policy_keys=("K", "alpha"),
        )
    with pytest.raises(ValueError, match="delta"):
        FederatedAggregationConfig(delta=1.0)
