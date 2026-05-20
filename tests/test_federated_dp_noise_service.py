# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated DP noise service tests

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceReadiness,
    DpNoiseServiceRequestManifest,
    DpNoiseServiceResponseManifest,
    build_dp_noise_service_deployment_preflight_manifest,
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


def _seed_response() -> DpNoiseServiceResponseManifest:
    return build_dp_noise_service_manifest(_seed_request())


def _preflight_labels(operator_approved: bool = True) -> dict[str, object]:
    return {
        "mechanism_label": "mechanism-v1",
        "privacy_accountant_owner": "accountant-a",
        "seed_custody_label": "seed-custody-a",
        "budget_issuer_label": "budget-issuer-a",
        "service_endpoint_label": "https://dp-noise.internal",
        "operator_approved": operator_approved,
    }


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


def test_dp_noise_service_preflight_is_deterministic_and_review_only() -> None:
    request = _seed_request()
    response = _seed_response()
    preflight_a = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )
    preflight_b = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )

    assert preflight_a == preflight_b
    assert preflight_a.deployment_readiness == DpNoiseServiceReadiness(
        ready=True, reason="offline_deployment_preflight_ready"
    )
    assert preflight_a.request_hash != ""
    assert preflight_a.response_hash != ""
    assert preflight_a.service_execution_permitted is False
    assert preflight_a.raw_data_export_permitted is False
    assert preflight_a.operator_review_required is True
    assert preflight_a.non_actuating is True


def test_dp_noise_service_preflight_blocks_missing_prerequisites() -> None:
    request = _seed_request()
    response = _seed_response()
    missing_custody = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels() | {"seed_custody_label": ""},
    )
    missing_accountant = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels() | {"privacy_accountant_owner": ""},
    )
    missing_operator = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(operator_approved=False),
    )
    bad_link = build_dp_noise_service_deployment_preflight_manifest(
        request,
        replace(response, request_hash="0" * 64),
        **_preflight_labels(),
    )

    assert missing_custody.deployment_readiness.ready is False
    assert (
        "seed_custody_label is required" in missing_custody.deployment_readiness.reason
    )
    assert missing_accountant.deployment_readiness.ready is False
    assert (
        "privacy_accountant_owner is required"
        in missing_accountant.deployment_readiness.reason
    )
    assert missing_operator.deployment_readiness.ready is False
    assert "operator approval required" in missing_operator.deployment_readiness.reason
    assert bad_link.deployment_readiness.ready is False
    assert (
        "request and response hash linkage broken"
        in bad_link.deployment_readiness.reason
    )


def test_dp_noise_service_deployment_preflight_manifest_rejects_malformed_input() -> (
    None
):
    request = _seed_request()
    response = _seed_response()

    with pytest.raises(ValueError, match="operator_approved must be a boolean"):
        build_dp_noise_service_deployment_preflight_manifest(
            request,
            response,
            **_preflight_labels(operator_approved=False) | {"operator_approved": "no"},
        )
    with pytest.raises(
        ValueError, match="request_manifest must be a DpNoiseServiceRequestManifest"
    ):
        build_dp_noise_service_deployment_preflight_manifest(
            None,
            response_manifest=response,
            **_preflight_labels(),
        )  # type: ignore[arg-type]


def test_dp_noise_service_deployment_preflight_payload_is_json_safe() -> None:
    request = _seed_request()
    response = _seed_response()
    manifest = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )
    record = manifest.to_audit_record()
    serialised = json.dumps(record, allow_nan=False, sort_keys=True)
    round_trip = json.loads(serialised)

    assert isinstance(round_trip["request_hash"], str)
    assert isinstance(round_trip["response_hash"], str)
    assert round_trip["deployment_readiness"]["ready"]
    assert round_trip["audit_record_hash"] == manifest.audit_record_hash


def test_dp_noise_service_deployment_preflight_hash_is_stable() -> None:
    request = _seed_request()
    response = _seed_response()
    preflight_a = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )
    preflight_b = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )
    preflight_mutated = build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels() | {"mechanism_label": "mechanism-v2"},
    )

    assert preflight_a.audit_record_hash == preflight_b.audit_record_hash
    assert preflight_a.audit_record_hash != preflight_mutated.audit_record_hash
