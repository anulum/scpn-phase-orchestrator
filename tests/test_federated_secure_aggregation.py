# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Secure aggregation manifest tests

from __future__ import annotations

import hashlib
import json

import pytest

from scpn_phase_orchestrator.supervisor.federated_secure_aggregation import (
    FederatedSecureAggregationManifest,
    build_federated_secure_aggregation_manifest,
)


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _node_commitment(
    node_id: str,
    masked_policy_delta: dict[str, float],
    sample_count: int,
) -> dict[str, object]:
    delta_items = [
        [key, float(value)] for key, value in sorted(masked_policy_delta.items())
    ]
    return {
        "node_id": node_id,
        "masked_policy_delta": dict(sorted(masked_policy_delta.items())),
        "sample_count": sample_count,
        "share_commitment": f"commit-{node_id}",
        "share_commitment_hash": _stable_hash(
            {"node_id": node_id, "share_commitment": f"commit-{node_id}"}
        ),
        "share_hash": _stable_hash(
            {"node_id": node_id, "masked_policy_delta": delta_items}
        ),
    }


def test_secure_aggregation_manifest_is_deterministic_and_review_only() -> None:
    first = build_federated_secure_aggregation_manifest(
        (
            _node_commitment("node-a", {"alpha": 0.2, "theta": 1.0}, 100),
            _node_commitment("node-b", {"theta": 0.4, "alpha": 0.1}, 40),
            _node_commitment("node-c", {"alpha": 0.0, "theta": -0.2}, 60),
        ),
        required_policy_keys=("theta", "alpha"),
        clipping_norm=2.0,
        min_node_count=3,
    )
    second = build_federated_secure_aggregation_manifest(
        (
            _node_commitment("node-c", {"theta": -0.2, "alpha": 0.0}, 60),
            _node_commitment("node-a", {"theta": 1.0, "alpha": 0.2}, 100),
            _node_commitment("node-b", {"theta": 0.4, "alpha": 0.1}, 40),
        ),
        required_policy_keys=("theta", "alpha"),
        clipping_norm=2.0,
        min_node_count=3,
    )

    assert first == second
    assert first.report_hash == second.report_hash
    assert first.accepted_node_count == 3
    assert first.rejected_node_count == 0
    assert first.secure_aggregation_execution_permitted is False
    assert first.raw_data_export_permitted is False
    assert first.operator_review_required is True
    assert first.non_actuating is True
    assert len(first.aggregate_masked_delta_hash) == 64
    assert len(first.report_hash) == 64
    assert isinstance(json.loads(json.dumps(first.to_audit_record())), dict)


def test_secure_aggregation_manifest_aggregates_weighted_masked_updates() -> None:
    report = build_federated_secure_aggregation_manifest(
        (
            _node_commitment("site-a", {"alpha": 0.0, "theta": 1.0}, 120),
            _node_commitment("site-b", {"theta": 3.0, "alpha": 1.0}, 80),
            _node_commitment("site-c", {"theta": 1.0, "alpha": -1.0}, 100),
        ),
        required_policy_keys=("theta", "alpha"),
        clipping_norm=5.0,
    )
    aggregate = dict(report.aggregate_masked_delta)
    assert report.total_sample_count == 300
    assert aggregate["theta"] == pytest.approx((1.0 * 120 + 3.0 * 80 + 1.0 * 100) / 300)
    assert aggregate["alpha"] == pytest.approx((0.0 * 120 + 1.0 * 80 - 1.0 * 100) / 300)


def test_secure_aggregation_manifest_rejects_quorum_failure() -> None:
    with pytest.raises(ValueError, match="quorum_not_met"):
        build_federated_secure_aggregation_manifest(
            (
                _node_commitment("node-a", {"alpha": 0.2, "theta": 1.0}, 10),
                _node_commitment("node-b", {"alpha": 0.1, "theta": 0.4}, 10),
            ),
            required_policy_keys=("theta", "alpha"),
            min_node_count=3,
        )


def test_secure_aggregation_manifest_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_federated_secure_aggregation_manifest(
            (
                _node_commitment("dup", {"alpha": 0.0, "theta": 0.5}, 10),
                _node_commitment("dup", {"alpha": 0.0, "theta": 0.5}, 10),
            ),
            required_policy_keys=("theta", "alpha"),
        )

    with pytest.raises(ValueError, match="missing required keys"):
        build_federated_secure_aggregation_manifest(
            (
                {
                    "node_id": "site-a",
                    "masked_policy_delta": {"alpha": 0.2},
                    "sample_count": 12,
                    "share_commitment": "commit-site-a",
                    "share_commitment_hash": _stable_hash(
                        {"node_id": "site-a", "share_commitment": "commit-site-a"}
                    ),
                    "share_hash": _stable_hash(
                        {"node_id": "site-a", "masked_policy_delta": [["alpha", 0.2]]}
                    ),
                },
            ),
            required_policy_keys=("theta", "alpha"),
        )

    with pytest.raises(ValueError, match="raw deltas or time-series"):
        build_federated_secure_aggregation_manifest(
            (
                {
                    **_node_commitment("site-a", {"theta": 0.1, "alpha": 0.2}, 8),
                    "raw_policy_delta": {"theta": 0.1, "alpha": 0.2},
                },
            ),
            required_policy_keys=("theta", "alpha"),
        )


def test_secure_aggregation_manifest_rejects_invalid_hashes() -> None:
    base = _node_commitment("site-a", {"theta": 1.0, "alpha": 0.0}, 10)
    base["share_hash"] = "not-a-sha256"
    with pytest.raises(ValueError, match="share_hash"):
        build_federated_secure_aggregation_manifest(
            (base,),
            required_policy_keys=("theta", "alpha"),
            min_node_count=1,
        )


def test_secure_aggregation_manifest_report_type() -> None:
    report = build_federated_secure_aggregation_manifest(
        (_node_commitment("site-a", {"theta": 0.4, "alpha": 0.2}, 1),),
        required_policy_keys=("theta", "alpha"),
        min_node_count=1,
    )
    assert isinstance(report, FederatedSecureAggregationManifest)
