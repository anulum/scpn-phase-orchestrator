# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated transport tests

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from scpn_phase_orchestrator.supervisor.federated_transport import (
    FederatedTransportReplayLedger,
    build_signed_transport_envelopes,
    replay_federated_transport_batch,
    validate_federated_transport_batch,
)


def _stable_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _update_hash(
    node_id: str, policy_delta: dict[str, float], sample_count: int
) -> str:
    return _stable_hash(
        {
            "node_id": node_id,
            "policy_delta": [
                [key, value] for key, value in sorted(policy_delta.items())
            ],
            "sample_count": sample_count,
            "local_loss": 0.21,
            "previous_audit_hash": "a" * 64,
            "privacy_epsilon_spent": 0.5,
            "clipped_l2_norm": 0.4,
            "clip_scale": 1.0,
            "accepted": True,
            "rejection_reasons": [],
        }
    )


def _records() -> tuple[dict[str, object], ...]:
    first = {
        "node_id": "node-a",
        "policy_delta": {"K": 0.1, "alpha": -0.01},
        "sample_count": 100,
        "local_loss": 0.21,
        "previous_audit_hash": "a" * 64,
        "privacy_epsilon_spent": 0.4,
        "clipped_l2_norm": 0.25,
        "clip_scale": 1.0,
        "accepted": True,
        "rejection_reasons": [],
    }
    second = {
        **first,
        "node_id": "node-b",
        "policy_delta": {"K": 0.2, "alpha": 0.01},
        "sample_count": 120,
    }
    third = {
        **first,
        "node_id": "node-a",
        "policy_delta": {"K": 0.05, "alpha": 0.02},
        "previous_audit_hash": "b" * 64,
        "sample_count": 80,
        "privacy_epsilon_spent": 0.5,
    }
    first["update_hash"] = _update_hash(
        first["node_id"], first["policy_delta"], first["sample_count"]
    )
    second["update_hash"] = _update_hash(
        second["node_id"], second["policy_delta"], second["sample_count"]
    )
    third["update_hash"] = _update_hash(
        third["node_id"], third["policy_delta"], third["sample_count"]
    )
    return first, second, third


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def test_transport_envelope_build_is_deterministic_and_review_only() -> None:
    first = build_signed_transport_envelopes(_records())
    second = build_signed_transport_envelopes(_records())

    assert first == second
    assert len(first) == 3
    assert first[0].parent_envelope_hash == "0" * 64
    assert first[1].parent_envelope_hash == "0" * 64
    assert first[2].parent_envelope_hash == first[0].envelope_hash
    assert first[0].node_sequence == 1
    assert first[2].node_sequence == 2
    for envelope in first:
        assert envelope.transport_execution_permitted is False
        assert envelope.raw_data_export_permitted is False
        assert envelope.operator_review_required is True
        assert _is_sha256(envelope.envelope_id)
        assert _is_sha256(envelope.node_update_audit_hash)
        assert _is_sha256(envelope.envelope_hash)
        json.loads(json.dumps(envelope.to_audit_record(), allow_nan=False))


def test_transport_replay_is_deterministic() -> None:
    envelopes = build_signed_transport_envelopes(_records())
    first_ledger = replay_federated_transport_batch(envelopes)
    second_ledger = replay_federated_transport_batch(envelopes)

    assert isinstance(first_ledger, FederatedTransportReplayLedger)
    assert first_ledger.envelope_count == len(envelopes)
    assert first_ledger.replay_hash == second_ledger.replay_hash
    assert first_ledger.envelope_ids == tuple(env.envelope_id for env in envelopes)


def test_transport_validation_replays_and_detects_tampering() -> None:
    envelopes = build_signed_transport_envelopes(_records())
    valid = validate_federated_transport_batch(envelopes)
    assert tuple(valid) == envelopes

    bad_parent = replace(
        envelopes[2],
        parent_envelope_hash="1" * 64,
    )
    with pytest.raises(ValueError, match="parent hash mismatch"):
        validate_federated_transport_batch((envelopes[0], envelopes[1], bad_parent))

    duplicated = replace(envelopes[1], envelope_id=envelopes[0].envelope_id)
    with pytest.raises(ValueError, match="duplicate envelope id"):
        validate_federated_transport_batch((envelopes[0], duplicated, envelopes[2]))

    bad_sequence = replace(envelopes[2], node_sequence=envelopes[2].node_sequence + 10)
    with pytest.raises(ValueError, match="non-monotonic node_sequence"):
        validate_federated_transport_batch((envelopes[0], envelopes[1], bad_sequence))


def test_transport_rejects_raw_time_series_and_schema_violations() -> None:
    record = dict(_records()[0])
    record["raw_time_series"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="raw time-series"):
        build_signed_transport_envelopes((record,))

    bad_schema = dict(_records()[1])
    bad_schema["unexpected_key"] = "forbidden"
    with pytest.raises(ValueError, match="unsupported keys"):
        build_signed_transport_envelopes((bad_schema,))


def test_transport_batch_rejects_bad_schema_name_and_hashes() -> None:
    with pytest.raises(ValueError, match="schema_name"):
        build_signed_transport_envelopes(
            _records(), schema_name="invalid_transport_schema"
        )

    bad_hash = dict(_records()[0])
    bad_hash["update_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="update_hash"):
        build_signed_transport_envelopes((bad_hash,))

    envelopes = build_signed_transport_envelopes(_records())
    replay = replay_federated_transport_batch(envelopes)
    json.loads(json.dumps(replay.to_audit_record(), allow_nan=False))
