# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated meta-orchestrator manifest guards

from __future__ import annotations

from typing import Any

import pytest

from scpn_phase_orchestrator.supervisor.federated import (
    build_federated_meta_orchestrator_manifest as _build_manifest,
)

_HASH = "0" * 64


def _node(**overrides: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "node_id": "node-a",
        "sample_count": 10,
        "local_loss": 0.5,
        "previous_audit_hash": _HASH,
        "policy_delta": {"alpha": 0.1, "beta": 0.2},
    }
    node.update(overrides)
    return node


def _build(nodes: Any = None, **config: Any) -> Any:
    config.setdefault("required_policy_keys", ["alpha", "beta"])
    config.setdefault("min_node_count", 1)
    return _build_manifest(
        [_node()] if nodes is None else nodes,
        **config,
    )


class TestManifestShapeGuards:
    def test_rejects_non_sequence_node_updates(self) -> None:
        with pytest.raises(ValueError, match="node_updates must be a sequence"):
            _build(nodes="not-a-sequence")

    def test_rejects_empty_node_updates(self) -> None:
        with pytest.raises(ValueError, match="node_updates must be non-empty"):
            _build(nodes=[])

    def test_rejects_non_positive_clipping_norm(self) -> None:
        with pytest.raises(ValueError, match="clipping_norm must be positive"):
            _build(clipping_norm=0.0)


class TestRequiredKeyGuards:
    def test_rejects_non_sequence_required_keys(self) -> None:
        with pytest.raises(ValueError, match="required_policy_keys must be a sequence"):
            _build(required_policy_keys=5)

    def test_rejects_non_unique_required_keys(self) -> None:
        with pytest.raises(ValueError, match="non-empty and unique"):
            _build(required_policy_keys=["alpha", "alpha"])


class TestKeyDiscoveryPath:
    """With no required keys the keys are discovered from the node deltas."""

    def test_discovers_keys_from_node_deltas(self) -> None:
        report = _build(required_policy_keys=None)
        assert report.required_policy_keys == ("alpha", "beta")

    def test_discovery_rejects_non_mapping_node(self) -> None:
        with pytest.raises(ValueError, match="each node update must be a mapping"):
            _build(nodes=["not-a-mapping"], required_policy_keys=None)

    def test_discovery_rejects_empty_policy_delta(self) -> None:
        with pytest.raises(
            ValueError, match="policy_delta must be a non-empty mapping"
        ):
            _build(nodes=[_node(policy_delta={})], required_policy_keys=None)


class TestNodeUpdateGuards:
    def test_rejects_non_mapping_node(self) -> None:
        with pytest.raises(ValueError, match="each node update must be a mapping"):
            _build(nodes=["not-a-mapping"])

    def test_rejects_empty_policy_delta(self) -> None:
        with pytest.raises(
            ValueError, match="policy_delta must be a non-empty mapping"
        ):
            _build(nodes=[_node(policy_delta={})])

    def test_rejects_empty_node_id(self) -> None:
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            _build(nodes=[_node(node_id="")])

    def test_rejects_non_real_local_loss(self) -> None:
        with pytest.raises(ValueError, match="local_loss must be finite"):
            _build(nodes=[_node(local_loss="low")])

    def test_rejects_non_finite_local_loss(self) -> None:
        with pytest.raises(ValueError, match="local_loss must be finite"):
            _build(nodes=[_node(local_loss=float("inf"))])

    def test_rejects_negative_local_loss(self) -> None:
        with pytest.raises(ValueError, match="local_loss must be non-negative"):
            _build(nodes=[_node(local_loss=-1.0)])

    def test_rejects_boolean_sample_count(self) -> None:
        with pytest.raises(ValueError, match="sample_count must be a positive integer"):
            _build(nodes=[_node(sample_count=True)])

    def test_rejects_non_positive_sample_count(self) -> None:
        with pytest.raises(ValueError, match="sample_count must be a positive integer"):
            _build(nodes=[_node(sample_count=0)])

    def test_rejects_non_hex_previous_audit_hash(self) -> None:
        with pytest.raises(ValueError, match="64-character SHA-256 hex string"):
            _build(nodes=[_node(previous_audit_hash="z" * 64)])
