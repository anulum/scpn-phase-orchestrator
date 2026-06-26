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
from typing import cast

import pytest

from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceDeploymentPreflightManifest,
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


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"epsilon": float("nan")}, "epsilon must be a finite float"),
        ({"epsilon": 2_000_000.0}, "epsilon exceeds max bound"),
        ({"delta": float("nan")}, "delta must be a finite float"),
        ({"delta": 2.0}, r"delta must be in \(0, 1\)"),
        ({"sensitivity": float("inf")}, "sensitivity must be a finite float"),
        ({"sensitivity": 0.0}, "sensitivity must be greater than 0"),
        ({"noise_multiplier": float("nan")}, "noise_multiplier must be a finite float"),
        ({"noise_multiplier": 0.0}, "noise_multiplier must be greater than 0"),
        ({"node_count": 1.5}, "node_count must be an integer"),
        ({"node_count": 0}, "node_count must be greater than 0"),
        ({"seed_hash": "a" * 32}, "seed_hash must be 64 hex characters"),
        ({"seed_hash": "z" * 64}, "seed_hash must be a hex string"),
        ({"policy_keys": ["alpha", "beta"]}, "policy_keys must be a tuple"),
        ({"policy_keys": ()}, "policy_keys must be non-empty"),
        ({"policy_keys": ("alpha", "")}, "policy key must be a non-empty string"),
        ({"policy_keys": ("alpha", 1)}, "policy key must be a non-empty string"),
    ],
)
def test_request_manifest_rejects_scalar_fields(changes, match) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_seed_request(), **changes)


def _budgets(*pairs: tuple[str, float]) -> tuple[DpNoiseNodePrivacyBudget, ...]:
    return tuple(DpNoiseNodePrivacyBudget(node_id=n, epsilon_spent=e) for n, e in pairs)


@pytest.mark.parametrize(
    ("node_budgets", "match"),
    [
        (list(_budgets(("a", 0.1), ("b", 0.1))), "node_budgets must be a tuple"),
        (_budgets(("a", 0.1)), "node_budgets length must match node_count"),
        (
            (_budgets(("a", 0.1))[0], "not-a-budget"),
            "node_budgets must only contain",
        ),
        (_budgets(("", 0.1), ("b", 0.1)), "node_id is required"),
        (_budgets(("dup", 0.1), ("dup", 0.1)), "node_id values must be unique"),
        (_budgets(("a", -0.1), ("b", 0.1)), "epsilon spent must be non-negative"),
    ],
)
def test_request_manifest_rejects_node_budgets(node_budgets, match) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_seed_request(), node_budgets=node_budgets)


def test_preflight_manifest_rejects_non_boolean_operator_flag() -> None:
    request = _seed_request()
    response = build_dp_noise_service_manifest(request)
    labels = _preflight_labels() | {"operator_approved": "yes"}
    with pytest.raises(ValueError, match="operator_approved must be a boolean"):
        build_dp_noise_service_deployment_preflight_manifest(
            request, response, **labels
        )


def test_preflight_rejects_non_response_manifest() -> None:
    with pytest.raises(
        ValueError, match="response_manifest must be a DpNoiseServiceResponseManifest"
    ):
        build_dp_noise_service_deployment_preflight_manifest(
            _seed_request(), "not-a-response", **_preflight_labels()
        )


def _preflight_manifest() -> DpNoiseServiceDeploymentPreflightManifest:
    request = _seed_request()
    response = build_dp_noise_service_manifest(request)
    return build_dp_noise_service_deployment_preflight_manifest(
        request,
        response,
        **_preflight_labels(),
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"mechanism_label": object()},
            "mechanism_label must be a non-empty string",
        ),
        (
            {"privacy_accountant_owner": object()},
            "privacy_accountant_owner must be a non-empty string",
        ),
        (
            {"seed_custody_label": object()},
            "seed_custody_label must be a non-empty string",
        ),
        (
            {"budget_issuer_label": object()},
            "budget_issuer_label must be a non-empty string",
        ),
        (
            {"service_endpoint_label": object()},
            "service_endpoint_label must be a non-empty string",
        ),
        ({"operator_approved": "yes"}, "operator_approved must be a boolean"),
        ({"request_hash": "0" * 63}, "request_hash must be a 64-char hex hash"),
        ({"response_hash": "0" * 63}, "response_hash must be a 64-char hex hash"),
        ({"epsilon": float("nan")}, "epsilon must be a finite float"),
        ({"epsilon": 0.0}, "epsilon must be greater than 0"),
        ({"delta": float("nan")}, "delta must be a finite float"),
        ({"delta": 1.0}, r"delta must be in \(0, 1\)"),
        ({"request_hash": "z" * 64}, "request_hash must be hexadecimal"),
        ({"response_hash": "z" * 64}, "response_hash must be hexadecimal"),
    ],
)
def test_deployment_preflight_manifest_rejects_malformed_fields(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_preflight_manifest(), **changes)


def test_preflight_rejects_non_string_label_through_builder() -> None:
    request = _seed_request()
    response = _seed_response()

    with pytest.raises(ValueError, match="mechanism_label must be a non-empty string"):
        build_dp_noise_service_deployment_preflight_manifest(
            request,
            response,
            **_preflight_labels()
            | {"mechanism_label": cast(str, object())},
        )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"epsilon": 9.0}, "epsilon mismatch"),
        ({"delta": 0.5}, "delta mismatch"),
        ({"epsilon": float("nan")}, "response epsilon must be finite and positive"),
        ({"epsilon": 0.0}, "response epsilon must be finite and positive"),
        ({"delta": float("nan")}, "response delta must be finite in"),
        ({"delta": 2.0}, "response delta must be finite in"),
    ],
)
def test_preflight_detects_request_response_inconsistency(changes, reason) -> None:
    request = _seed_request()
    response = replace(_seed_response(), **changes)
    preflight = build_dp_noise_service_deployment_preflight_manifest(
        request, response, **_preflight_labels()
    )
    assert preflight.deployment_readiness.ready is False
    assert reason in preflight.deployment_readiness.reason
