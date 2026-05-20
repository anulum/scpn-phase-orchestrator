# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated DP noise service tests

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceReadiness,
    DpNoiseServiceRequestManifest,
    build_dp_noise_service_manifest,
)


def _seed_request() -> DpNoiseServiceRequestManifest:
    return DpNoiseServiceRequestManifest(
        epsilon=2.5,
        delta=1e-6,
        sensitivity=1.75,
        noise_multiplier=0.9,
        node_count=2,
        seed_hash="a" * 64,
        policy_keys=("alpha", "beta", "gamma"),
        node_budgets=(
            DpNoiseNodePrivacyBudget(node_id="node-a", epsilon_spent=0.9),
            DpNoiseNodePrivacyBudget(node_id="node-b", epsilon_spent=1.2),
        ),
    )


def test_dp_noise_manifest_deterministic_and_review_only() -> None:
    first = build_dp_noise_service_manifest(_seed_request())
    second = build_dp_noise_service_manifest(_seed_request())

    assert first == second
    assert first.service_execution_permitted is False
    assert first.raw_data_export_permitted is False
    assert first.operator_review_required is True
    assert first.non_actuating is True
    assert first.service_readiness == DpNoiseServiceReadiness(
        ready=True, reason="offline_review_only_manifest_ready"
    )


def test_dp_noise_manifest_budget_accounting_and_noise_vector() -> None:
    manifest = build_dp_noise_service_manifest(_seed_request())

    assert manifest.privacy_budget_spent == pytest.approx(2.1)
    assert manifest.privacy_budget_remaining == pytest.approx(0.4)
    assert len(manifest.policy_noise_audit_vector) == 3
    assert manifest.policy_noise_audit_vector[0][0] == "alpha"
    assert manifest.privacy_budget_spent <= manifest.epsilon


def test_dp_noise_manifest_malformed_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon must be greater than 0"):
        DpNoiseServiceRequestManifest(
            epsilon=-0.1,
            delta=1e-6,
            sensitivity=1.0,
            noise_multiplier=1.0,
            node_count=1,
            seed_hash="b" * 64,
            policy_keys=("alpha",),
            node_budgets=(DpNoiseNodePrivacyBudget("node-a", 0.1),),
        )
    with pytest.raises(ValueError, match="policy_keys must be unique"):
        DpNoiseServiceRequestManifest(
            epsilon=1.0,
            delta=1e-6,
            sensitivity=1.0,
            noise_multiplier=1.0,
            node_count=1,
            seed_hash="c" * 64,
            policy_keys=("alpha", "alpha"),
            node_budgets=(DpNoiseNodePrivacyBudget("node-a", 0.1),),
        )
    with pytest.raises(ValueError, match="seed_hash is required"):
        DpNoiseServiceRequestManifest(
            epsilon=1.0,
            delta=1e-6,
            sensitivity=1.0,
            noise_multiplier=1.0,
            node_count=1,
            seed_hash=" ",
            policy_keys=("alpha",),
            node_budgets=(DpNoiseNodePrivacyBudget("node-a", 0.1),),
        )


def test_dp_noise_manifest_budget_failure_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="privacy budget exceeded"):
        request = DpNoiseServiceRequestManifest(
            epsilon=1.0,
            delta=1e-6,
            sensitivity=1.0,
            noise_multiplier=1.0,
            node_count=2,
            seed_hash="d" * 64,
            policy_keys=("alpha",),
            node_budgets=(
                DpNoiseNodePrivacyBudget(node_id="node-a", epsilon_spent=0.7),
                DpNoiseNodePrivacyBudget(node_id="node-b", epsilon_spent=0.7),
            ),
        )
        build_dp_noise_service_manifest(request)


def test_dp_noise_manifest_audit_payload_is_json_safe() -> None:
    manifest = build_dp_noise_service_manifest(_seed_request())
    record = manifest.to_audit_record()
    serialised = json.dumps(record, allow_nan=False, sort_keys=True)
    round_trip = json.loads(serialised)

    assert isinstance(round_trip["policy_noise_audit_vector"], list)
    assert isinstance(round_trip["policy_keys"], list)
    assert round_trip["audit_record_hash"] == manifest.audit_record_hash
