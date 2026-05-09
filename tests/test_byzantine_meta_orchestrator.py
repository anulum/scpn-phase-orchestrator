# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Byzantine meta-orchestrator tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.byzantine import (
    build_bft_meta_orchestrator_manifest,
    sign_policy_proposal,
)


def test_bft_meta_orchestrator_accepts_two_of_three_signed_proposals() -> None:
    keyring = {"node-a": "alpha", "node-b": "bravo", "node-c": "charlie"}
    payload = {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False}
    parent_hash = "a" * 64
    proposals = [
        sign_policy_proposal("node-a", payload, parent_hash, keyring["node-a"]),
        sign_policy_proposal("node-b", payload, parent_hash, keyring["node-b"]),
        sign_policy_proposal(
            "node-c",
            {"policy": "raise", "knobs": {"K": 0.8}, "actuation": False},
            parent_hash,
            keyring["node-c"],
        ),
    ]

    manifest = build_bft_meta_orchestrator_manifest(proposals, keyring, quorum=2)
    repeated = build_bft_meta_orchestrator_manifest(proposals, keyring, quorum=2)

    assert manifest == repeated
    assert manifest["manifest_kind"] == "bft_meta_orchestrator_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "accepted"
    assert manifest["quorum"] == 2
    assert manifest["accepted_node_ids"] == ["node-a", "node-b"]
    assert manifest["rejected_node_ids"] == ["node-c"]
    assert manifest["actuation_permitted"] is False
    assert manifest["network_opened"] is False
    assert len(manifest["consensus_hash"]) == 64
    assert len(manifest["audit_chain_hash"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["operator_commands"] == [
        "review bft_meta_orchestrator_manifest.json",
        "apply accepted policy only through the normal supervisor review gate",
    ]


def test_bft_meta_orchestrator_blocks_invalid_signature_and_missing_quorum() -> None:
    keyring = {"node-a": "alpha", "node-b": "bravo", "node-c": "charlie"}
    payload = {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False}
    parent_hash = "b" * 64
    valid = sign_policy_proposal("node-a", payload, parent_hash, keyring["node-a"])
    invalid = dict(
        sign_policy_proposal("node-b", payload, parent_hash, keyring["node-b"])
    )
    invalid["signature"] = "0" * 64

    manifest = build_bft_meta_orchestrator_manifest(
        [valid, invalid],
        keyring,
        quorum=2,
    )

    assert manifest["status"] == "blocked"
    assert manifest["accepted_node_ids"] == []
    assert manifest["rejected_node_ids"] == ["node-b"]
    assert "valid quorum not reached" in manifest["blocked_reasons"]
    assert "node-b signature verification failed" in manifest["blocked_reasons"]
    assert manifest["actuation_permitted"] is False


def test_bft_meta_orchestrator_rejects_invalid_configuration() -> None:
    keyring = {"node-a": "alpha"}
    payload = {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False}
    proposal = sign_policy_proposal("node-a", payload, "c" * 64, keyring["node-a"])

    with pytest.raises(ValueError, match="quorum"):
        build_bft_meta_orchestrator_manifest([proposal], keyring, quorum=0)
    with pytest.raises(ValueError, match="previous_audit_hash"):
        sign_policy_proposal("node-a", payload, "bad", keyring["node-a"])
