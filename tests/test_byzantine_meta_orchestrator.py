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


def test_bft_meta_orchestrator_rejects_invalid_collection_shapes() -> None:
    keyring = {"node-a": "alpha"}
    proposal = sign_policy_proposal(
        "node-a",
        {"policy": "hold", "actuation": False},
        "d" * 64,
        keyring["node-a"],
    )

    with pytest.raises(ValueError, match="proposals"):
        build_bft_meta_orchestrator_manifest({"node-a": proposal}, keyring, quorum=1)
    with pytest.raises(ValueError, match="proposals"):
        build_bft_meta_orchestrator_manifest([], keyring, quorum=1)
    with pytest.raises(ValueError, match="keyring"):
        build_bft_meta_orchestrator_manifest([proposal], ["alpha"], quorum=1)
    with pytest.raises(ValueError, match="keyring"):
        build_bft_meta_orchestrator_manifest([proposal], {}, quorum=1)


def test_bft_meta_orchestrator_marks_duplicate_node_as_rejected() -> None:
    keyring = {"node-a": "alpha", "node-b": "bravo"}
    payload = {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False}
    parent_hash = "e" * 64
    node_a = sign_policy_proposal("node-a", payload, parent_hash, keyring["node-a"])
    duplicate_node_a = sign_policy_proposal(
        "node-a",
        payload,
        parent_hash,
        keyring["node-a"],
    )
    node_b = sign_policy_proposal("node-b", payload, parent_hash, keyring["node-b"])

    manifest = build_bft_meta_orchestrator_manifest(
        [node_a, duplicate_node_a, node_b],
        keyring,
        quorum=2,
    )

    assert manifest["status"] == "accepted"
    assert manifest["accepted_node_ids"] == ["node-a", "node-b"]
    assert manifest["rejected_node_ids"] == ["node-a"]
    assert manifest["blocked_reasons"] == ["node-a duplicate proposal"]


def test_bft_meta_orchestrator_fails_closed_for_malformed_or_tampered_records() -> None:
    keyring = {"node-a": "alpha", "node-b": "bravo"}
    payload = {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False}
    parent_hash = "f" * 64
    missing_key = sign_policy_proposal(
        "node-b",
        payload,
        parent_hash,
        keyring["node-b"],
    )
    tampered_payload = dict(
        sign_policy_proposal("node-a", payload, parent_hash, keyring["node-a"])
    )
    tampered_payload["payload"] = {"policy": "raise", "knobs": {"K": 0.8}}

    manifest = build_bft_meta_orchestrator_manifest(
        [missing_key, tampered_payload],
        {"node-a": "alpha"},
        quorum=2,
    )

    assert manifest["status"] == "blocked"
    assert manifest["accepted_node_ids"] == []
    assert manifest["rejected_node_ids"] == ["node-a", "node-b"]
    assert "node-b missing signing key" in manifest["blocked_reasons"]
    assert "node-a payload hash mismatch" in manifest["blocked_reasons"]
    assert "valid quorum not reached" in manifest["blocked_reasons"]


def test_bft_meta_orchestrator_rejects_non_mapping_payload_and_blank_text() -> None:
    keyring = {"node-a": "alpha"}
    proposal = sign_policy_proposal(
        "node-a",
        {"policy": "hold", "actuation": False},
        "1" * 64,
        keyring["node-a"],
    )
    malformed = dict(proposal)
    malformed["payload"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="payload must be a mapping"):
        build_bft_meta_orchestrator_manifest([malformed], keyring, quorum=1)
    with pytest.raises(ValueError, match="payload must encode as a JSON object"):
        sign_policy_proposal("node-a", ["not", "a", "mapping"], "1" * 64, "alpha")
    with pytest.raises(ValueError, match="node_id"):
        sign_policy_proposal("   ", {"policy": "hold"}, "1" * 64, "alpha")


def test_bft_meta_orchestrator_manifest_tiebreaks_by_payload_hash() -> None:
    keyring = {
        "node-a": "alpha",
        "node-b": "bravo",
        "node-c": "charlie",
        "node-d": "delta",
    }
    parent_hash = "0" * 64
    proposal_fast = sign_policy_proposal(
        "node-a",
        {"policy": "hold", "priority": 1, "actuation": False},
        parent_hash,
        keyring["node-a"],
    )
    proposal_fast_copy = sign_policy_proposal(
        "node-b",
        {"policy": "hold", "priority": 1, "actuation": False},
        parent_hash,
        keyring["node-b"],
    )
    proposal_slow = sign_policy_proposal(
        "node-c",
        {"policy": "raise", "priority": 2, "actuation": False},
        parent_hash,
        keyring["node-c"],
    )
    proposal_slow_copy = sign_policy_proposal(
        "node-d",
        {"policy": "raise", "priority": 2, "actuation": False},
        parent_hash,
        keyring["node-d"],
    )

    group_a_hash = proposal_fast["payload_hash"]
    group_b_hash = proposal_slow["payload_hash"]
    proposals_a_first = [
        proposal_fast,
        proposal_fast_copy,
        proposal_slow,
        proposal_slow_copy,
    ]
    proposals_b_first = list(reversed(proposals_a_first))

    manifest_a = build_bft_meta_orchestrator_manifest(
        proposals_a_first, keyring, quorum=2
    )
    manifest_b = build_bft_meta_orchestrator_manifest(
        proposals_b_first, keyring, quorum=2
    )

    assert manifest_a["status"] == "accepted"
    assert manifest_a == manifest_b

    expected_winner_hash = min(group_a_hash, group_b_hash)
    expected_accepted = sorted(
        [
            "node-a",
            "node-b",
        ]
        if expected_winner_hash == group_a_hash
        else [
            "node-c",
            "node-d",
        ]
    )
    expected_rejected = sorted(
        [
            "node-c",
            "node-d",
        ]
        if expected_winner_hash == group_a_hash
        else [
            "node-a",
            "node-b",
        ]
    )

    assert manifest_a["accepted_node_ids"] == expected_accepted
    assert manifest_a["rejected_node_ids"] == expected_rejected


def test_bft_meta_orchestrator_keeps_deduplicated_block_reasons() -> None:
    keyring = {"node-a": "alpha", "node-b": "bravo"}
    payload = {"policy": "hold", "actuation": False}
    parent_hash = "1" * 64
    first = sign_policy_proposal("node-a", payload, parent_hash, keyring["node-a"])
    duplicate_1 = sign_policy_proposal(
        "node-a",
        payload,
        parent_hash,
        keyring["node-a"],
    )
    duplicate_2 = sign_policy_proposal(
        "node-a",
        payload,
        parent_hash,
        keyring["node-a"],
    )
    valid = sign_policy_proposal("node-b", payload, parent_hash, keyring["node-b"])

    manifest = build_bft_meta_orchestrator_manifest(
        [first, duplicate_1, duplicate_2, valid],
        keyring,
        quorum=2,
    )

    assert manifest["status"] == "accepted"
    assert manifest["accepted_node_ids"] == ["node-a", "node-b"]
    assert manifest["blocked_reasons"] == ["node-a duplicate proposal"]


def test_bft_meta_orchestrator_rejects_invalid_non_mapping_proposal() -> None:
    with pytest.raises(ValueError, match="proposal must be a mapping"):
        build_bft_meta_orchestrator_manifest([1, 2, 3], {"node-a": "alpha"}, quorum=1)

def test_bft_meta_orchestrator_reports_missing_signing_keys_and_quorum_block(
    ) -> None:
    keyring = {"node-a": "alpha"}
    payload = {"policy": "hold", "actuation": False}
    parent_hash = "f" * 64
    proposal_with_key = sign_policy_proposal(
        "node-a", payload, parent_hash, keyring["node-a"]
    )
    proposal_without_key = sign_policy_proposal("node-b", payload, parent_hash, "bravo")

    manifest = build_bft_meta_orchestrator_manifest(
        [proposal_with_key, proposal_without_key],
        keyring,
        quorum=2,
    )

    assert manifest["status"] == "blocked"
    assert manifest["accepted_node_ids"] == []
    assert manifest["rejected_node_ids"] == ["node-b"]
    assert manifest["consensus_hash"] == ""
    assert manifest["audit_chain_hash"] == ""
    assert "node-b missing signing key" in manifest["blocked_reasons"]
    assert "valid quorum not reached" in manifest["blocked_reasons"]
    assert manifest["manifest_sha256"] == manifest["manifest_sha256"]

def test_sign_policy_proposal_does_not_accept_nan_payload() -> None:
    keyring = {"node-a": "alpha"}

    with pytest.raises(
        ValueError,
        match="Out of range float values are not JSON compliant",
    ):
        sign_policy_proposal(
            "node-a",
            {"metric": float("nan"), "actuation": False},
            "f" * 64,
            keyring["node-a"],
        )


def test_build_bft_meta_orchestrator_manifest_treats_malformed_hash_as_invalid() -> (
    None
):
    keyring = {"node-a": "alpha"}
    proposal = sign_policy_proposal(
        "node-a",
        {"policy": "hold", "actuation": False},
        "8" * 64,
        keyring["node-a"],
    )
    proposal["payload_hash"] = "z" * 64

    with pytest.raises(ValueError, match="payload_hash"):
        build_bft_meta_orchestrator_manifest([proposal], keyring, quorum=1)
