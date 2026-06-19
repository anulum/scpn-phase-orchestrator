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
from dataclasses import replace

import pytest

from scpn_phase_orchestrator.supervisor.federated_secure_aggregation import (
    FederatedSecureAggregationManifest,
    FederatedSecureAggregationPreflightManifest,
    build_federated_secure_aggregation_manifest,
    build_federated_secure_aggregation_preflight_manifest,
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


def _label(node_id: str, kind: str, tag: str) -> str:
    return _stable_hash({"node_id": node_id, "kind": kind, "tag": tag})


def _custody_record(
    node_id: str, rotation_policy: str, *, tag: str = "current"
) -> dict[str, str]:
    key_previous = _label(node_id, "key", f"previous-{tag}")
    share_previous = _label(node_id, "share", f"previous-{tag}")
    key_label = _label(node_id, "key", tag)
    share_label = _label(node_id, "share", tag)
    return {
        "node_id": node_id,
        "key_custody_label": key_label,
        "share_custody_label": share_label,
        "previous_key_custody_label": key_previous,
        "previous_share_custody_label": share_previous,
        "key_custody_continuity_hash": _stable_hash(
            {
                "node_id": node_id,
                "rotation_policy": rotation_policy,
                "previous_key_custody_label": key_previous,
                "key_custody_label": key_label,
            }
        ),
        "share_custody_continuity_hash": _stable_hash(
            {
                "node_id": node_id,
                "rotation_policy": rotation_policy,
                "previous_share_custody_label": share_previous,
                "share_custody_label": share_label,
            }
        ),
    }


def _quorum_evidence(node_id: str) -> dict[str, str]:
    return {
        "node_id": node_id,
        "evidence_hash": _stable_hash({"node_id": node_id, "kind": "quorum"}),
    }


def _build_ready_manifest() -> FederatedSecureAggregationManifest:
    return build_federated_secure_aggregation_manifest(
        (
            _node_commitment("node-a", {"alpha": 0.2, "theta": 1.0}, 100),
            _node_commitment("node-b", {"theta": 0.4, "alpha": 0.1}, 40),
            _node_commitment("node-c", {"alpha": 0.0, "theta": -0.2}, 60),
        ),
        required_policy_keys=("theta", "alpha"),
        clipping_norm=2.0,
        min_node_count=3,
    )


def _build_preflight(
    manifest: FederatedSecureAggregationManifest,
    *,
    rotation_policy: str = "continuous",
    accepted_node_threshold: int = 3,
    operator_approved: bool = True,
    operator_id: str = "ops-1",
    service_owner: str = "svc-phase-orchestrator",
) -> FederatedSecureAggregationPreflightManifest:
    return build_federated_secure_aggregation_preflight_manifest(
        manifest,
        quorum_evidence=(
            _quorum_evidence("node-a"),
            _quorum_evidence("node-b"),
            _quorum_evidence("node-c"),
        ),
        custody_rotation_policy=rotation_policy,
        custody_records=(
            _custody_record("node-a", rotation_policy),
            _custody_record("node-b", rotation_policy),
            _custody_record("node-c", rotation_policy),
        ),
        accepted_node_threshold=accepted_node_threshold,
        operator_approved=operator_approved,
        operator_id=operator_id,
        service_owner=service_owner,
    )


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


def test_federated_secure_aggregation_preflight_manifest_is_deterministic() -> None:
    manifest = _build_ready_manifest()
    first = _build_preflight(manifest, rotation_policy="continuous")
    second = _build_preflight(manifest, rotation_policy="continuous")
    assert first == second
    assert first.report_hash == second.report_hash


def test_federated_secure_preflight_ready_with_complete_prerequisites() -> None:
    manifest = _build_ready_manifest()
    preflight = _build_preflight(manifest, rotation_policy="scheduled")

    assert preflight.accepted_node_threshold == 3
    assert preflight.accepted_node_count == 3
    assert preflight.operator_approved is True
    assert preflight.operator_id == "ops-1"
    assert preflight.service_owner == "svc-phase-orchestrator"
    assert preflight.secure_aggregation_execution_permitted is False
    assert preflight.raw_data_export_permitted is False
    assert preflight.operator_review_required is True
    assert preflight.non_actuating is True
    assert preflight.secure_aggregation_report_hash == manifest.report_hash
    assert len(preflight.report_hash) == 64
    assert isinstance(json.loads(json.dumps(preflight.to_audit_record())), dict)


def test_federated_secure_aggregation_preflight_manifest_blocks_quorum_gap() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(
        ValueError, match="quorum evidence below accepted-node threshold"
    ):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            quorum_evidence=(
                _quorum_evidence("node-a"),
                _quorum_evidence("node-b"),
            ),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=3,
            operator_approved=True,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )


def test_federated_secure_preflight_blocks_hash_and_custody_gaps() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="key custody continuity hash mismatch"):
        invalid = _custody_record("node-a", "continuous")
        invalid["key_custody_continuity_hash"] = "0" * 64
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            quorum_evidence=(
                _quorum_evidence("node-a"),
                _quorum_evidence("node-b"),
                _quorum_evidence("node-c"),
            ),
            custody_rotation_policy="continuous",
            custody_records=(
                invalid,
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=3,
            operator_approved=True,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )

    with pytest.raises(
        ValueError, match="custody labels must cover all accepted nodes"
    ):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            quorum_evidence=(
                _quorum_evidence("node-a"),
                _quorum_evidence("node-b"),
                _quorum_evidence("node-c"),
            ),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
            ),
            accepted_node_threshold=3,
            operator_approved=True,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )


def test_federated_secure_aggregation_preflight_manifest_blocks_operator_gap() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="operator approval is required"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            quorum_evidence=(
                _quorum_evidence("node-a"),
                _quorum_evidence("node-b"),
                _quorum_evidence("node-c"),
            ),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=3,
            operator_approved=False,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )


def test_federated_secure_aggregation_preflight_manifest_malformed_inputs() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="accepted-node threshold not met"):
        _build_preflight(
            manifest,
            accepted_node_threshold=4,
        )

    with pytest.raises(ValueError, match="unsupported custody rotation policy"):
        _build_preflight(manifest, rotation_policy="bad")

    with pytest.raises(ValueError, match="operator_id must be a non-empty string"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            quorum_evidence=(_quorum_evidence("node-a"),),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=1,
            operator_approved=True,
            operator_id="",
            service_owner="svc-phase-orchestrator",
        )

    with pytest.raises(ValueError, match="report hash mismatch"):
        broken = replace(
            manifest,
            report_hash="0" * 64,
        )
        build_federated_secure_aggregation_preflight_manifest(
            broken,
            quorum_evidence=(
                _quorum_evidence("node-a"),
                _quorum_evidence("node-b"),
                _quorum_evidence("node-c"),
            ),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=3,
            operator_approved=True,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )

    with pytest.raises(TypeError, match="FederatedSecureAggregationManifest"):
        build_federated_secure_aggregation_preflight_manifest(
            "not-a-manifest",  # type: ignore[arg-type]
            quorum_evidence=(_quorum_evidence("node-a"),),
            custody_rotation_policy="continuous",
            custody_records=(
                _custody_record("node-a", "continuous"),
                _custody_record("node-b", "continuous"),
                _custody_record("node-c", "continuous"),
            ),
            accepted_node_threshold=1,
            operator_approved=True,
            operator_id="ops-1",
            service_owner="svc-phase-orchestrator",
        )


def test_federated_secure_preflight_stable_hash_changes_with_inputs() -> None:
    manifest = _build_ready_manifest()
    first = _build_preflight(manifest, rotation_policy="continuous")
    second = _build_preflight(
        manifest,
        rotation_policy="continuous",
        service_owner="svc-phase-orchestrator-alt",
    )
    assert first.report_hash != second.report_hash


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


# ---------------------------------------------------------------------
# Build manifest rejection surface
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("container", "match"),
    [
        ("not-a-sequence", "node_commitments must be a sequence of mappings"),
        ((), "node_commitments must be non-empty"),
    ],
)
def test_build_manifest_rejects_bad_container(container, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_manifest(
            container, required_policy_keys=("theta",)
        )


def test_build_manifest_rejects_delta_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
        build_federated_secure_aggregation_manifest(
            (_node_commitment("node-a", {"theta": 0.1}, 5),),
            required_policy_keys=("theta",),
            min_node_count=1,
            delta=1.0,
        )


@pytest.mark.parametrize(
    ("clipping_norm", "match"),
    [
        (0.0, "clipping_norm must be a finite positive float"),
        ("x", "clipping_norm must be a real number"),
    ],
)
def test_build_manifest_rejects_bad_clipping_norm(clipping_norm, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_manifest(
            (_node_commitment("node-a", {"theta": 0.1}, 5),),
            required_policy_keys=("theta",),
            min_node_count=1,
            clipping_norm=clipping_norm,
        )


@pytest.mark.parametrize(
    ("keys", "match"),
    [
        (123, "required_policy_keys must be a sequence of strings"),
        ((), "required_policy_keys must be non-empty and unique"),
        (("theta", "theta"), "required_policy_keys must be non-empty and unique"),
    ],
)
def test_build_manifest_rejects_bad_required_policy_keys(keys, match) -> None:
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_manifest(
            (_node_commitment("node-a", {"theta": 0.1}, 5),),
            required_policy_keys=keys,
            min_node_count=1,
        )


def test_build_manifest_discovers_policy_keys_when_unspecified() -> None:
    report = build_federated_secure_aggregation_manifest(
        (_node_commitment("node-a", {"theta": 0.1, "alpha": 0.2}, 5),),
        min_node_count=1,
    )
    assert set(report.required_policy_keys) == {"alpha", "theta"}


@pytest.mark.parametrize(
    ("commitments", "match"),
    [
        ((123,), "each node commitment must be a mapping"),
        (
            ({"masked_policy_delta": 123},),
            "masked_policy_delta must be a non-empty mapping",
        ),
        (
            ({"masked_policy_delta": {}},),
            "masked_policy_delta must be a non-empty mapping",
        ),
    ],
)
def test_build_manifest_discovery_path_rejects_bad_commitments(
    commitments, match
) -> None:
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_manifest(commitments, min_node_count=1)


def test_build_manifest_rejects_non_mapping_commitment_with_keys() -> None:
    with pytest.raises(ValueError, match="each node commitment must be a mapping"):
        build_federated_secure_aggregation_manifest(
            (123,), required_policy_keys=("theta",), min_node_count=1
        )


def test_build_manifest_rejects_empty_masked_delta_and_zero_sample_count() -> None:
    empty_delta = _node_commitment("node-a", {"theta": 0.1}, 5)
    empty_delta["masked_policy_delta"] = {}
    with pytest.raises(ValueError, match="masked_policy_delta must be a non-empty"):
        build_federated_secure_aggregation_manifest(
            (empty_delta,), required_policy_keys=("theta",), min_node_count=1
        )

    zero_samples = _node_commitment("node-a", {"theta": 0.1}, 0)
    with pytest.raises(ValueError, match="sample_count must be a positive int"):
        build_federated_secure_aggregation_manifest(
            (zero_samples,), required_policy_keys=("theta",), min_node_count=1
        )


def test_build_manifest_rejects_non_finite_policy_delta_value() -> None:
    record = _node_commitment("node-a", {"theta": 0.1}, 5)
    record["masked_policy_delta"] = {"theta": float("inf")}
    with pytest.raises(ValueError, match=r"policy_delta\[theta\] must be a finite"):
        build_federated_secure_aggregation_manifest(
            (record,), required_policy_keys=("theta",), min_node_count=1
        )


def test_build_manifest_marks_node_rejected_on_commitment_hash_mismatch() -> None:
    bad = _node_commitment("node-d", {"theta": 0.1, "alpha": 0.0}, 30)
    bad["share_commitment_hash"] = "0" * 64
    report = build_federated_secure_aggregation_manifest(
        (
            _node_commitment("node-a", {"alpha": 0.2, "theta": 1.0}, 100),
            _node_commitment("node-b", {"theta": 0.4, "alpha": 0.1}, 40),
            _node_commitment("node-c", {"alpha": 0.0, "theta": -0.2}, 60),
            bad,
        ),
        required_policy_keys=("theta", "alpha"),
        clipping_norm=2.0,
        min_node_count=3,
    )
    assert report.accepted_node_count == 3
    assert report.rejected_node_count == 1
    rejected = next(n for n in report.node_commitments if not n.accepted)
    assert "share_commitment_hash_mismatch" in rejected.rejection_reasons


# ---------------------------------------------------------------------
# Preflight rejection surface
# ---------------------------------------------------------------------


def _preflight_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "quorum_evidence": (
            _quorum_evidence("node-a"),
            _quorum_evidence("node-b"),
            _quorum_evidence("node-c"),
        ),
        "custody_rotation_policy": "continuous",
        "custody_records": (
            _custody_record("node-a", "continuous"),
            _custody_record("node-b", "continuous"),
            _custody_record("node-c", "continuous"),
        ),
        "accepted_node_threshold": 3,
        "operator_approved": True,
        "operator_id": "ops-1",
        "service_owner": "svc-phase-orchestrator",
    }
    base.update(overrides)
    return base


def test_preflight_rejects_manifest_without_quorum_met() -> None:
    manifest = replace(_build_ready_manifest(), quorum_met=False)
    with pytest.raises(ValueError, match="quorum not met"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest, **_preflight_kwargs()
        )


def test_preflight_rejects_zero_threshold() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="accepted_node_threshold must be a positive"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest, **_preflight_kwargs(accepted_node_threshold=0)
        )


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        ("not-a-sequence", "quorum_evidence must be a sequence of mappings"),
        ((), "quorum_evidence must be non-empty"),
        ((123,), "each quorum evidence entry must be a mapping"),
    ],
)
def test_preflight_rejects_bad_quorum_evidence_container(evidence, match) -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_preflight_manifest(
            manifest, **_preflight_kwargs(quorum_evidence=evidence)
        )


def test_preflight_rejects_duplicate_and_unknown_quorum_nodes() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="quorum evidence duplicated node_id"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            **_preflight_kwargs(
                quorum_evidence=(
                    _quorum_evidence("node-a"),
                    _quorum_evidence("node-a"),
                    _quorum_evidence("node-c"),
                )
            ),
        )

    with pytest.raises(ValueError, match="quorum evidence must reference accepted"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            **_preflight_kwargs(
                quorum_evidence=(
                    _quorum_evidence("node-a"),
                    _quorum_evidence("node-b"),
                    _quorum_evidence("ghost"),
                )
            ),
        )


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ("not-a-sequence", "custody_records must be a sequence of mappings"),
        ((), "custody_records must be non-empty"),
        ((123,), "each custody record must be a mapping"),
    ],
)
def test_preflight_rejects_bad_custody_container(records, match) -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match=match):
        build_federated_secure_aggregation_preflight_manifest(
            manifest, **_preflight_kwargs(custody_records=records)
        )


def test_preflight_rejects_duplicate_and_unknown_custody_nodes() -> None:
    manifest = _build_ready_manifest()
    with pytest.raises(ValueError, match="custody record duplicated node_id"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            **_preflight_kwargs(
                custody_records=(
                    _custody_record("node-a", "continuous"),
                    _custody_record("node-a", "continuous"),
                    _custody_record("node-c", "continuous"),
                )
            ),
        )

    with pytest.raises(ValueError, match="custody labels must reference accepted"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            **_preflight_kwargs(
                custody_records=(
                    _custody_record("node-a", "continuous"),
                    _custody_record("node-b", "continuous"),
                    _custody_record("ghost", "continuous"),
                )
            ),
        )


def test_preflight_rejects_share_custody_continuity_mismatch() -> None:
    manifest = _build_ready_manifest()
    broken = _custody_record("node-a", "continuous")
    broken["share_custody_continuity_hash"] = "0" * 64
    with pytest.raises(ValueError, match="share custody continuity hash mismatch"):
        build_federated_secure_aggregation_preflight_manifest(
            manifest,
            **_preflight_kwargs(
                custody_records=(
                    broken,
                    _custody_record("node-b", "continuous"),
                    _custody_record("node-c", "continuous"),
                )
            ),
        )
