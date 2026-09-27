# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — federated transport node-update seal tests

"""The federated transport verifies each node update record's seal.

The aggregator seals every node update record with ``update_hash``. The
transport checked only that the value was a SHA-256 digest and then cited it in
signed, hash-linked envelopes. A record edited after sealing, with a lowered
privacy spend or a changed policy delta, was transported under its original
hash.
"""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal

import pytest

from scpn_phase_orchestrator.supervisor.federated import (
    build_federated_meta_orchestrator_manifest,
)
from scpn_phase_orchestrator.supervisor.federated_transport import (
    build_signed_transport_envelopes,
    replay_federated_transport_batch,
    validate_federated_transport_batch,
)

UPDATES = (
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


def _node_records(
    required: tuple[str, ...] = ("K", "alpha"),
) -> list[dict[str, object]]:
    """Return sealed node update records from the federated aggregator."""
    report = build_federated_meta_orchestrator_manifest(
        UPDATES,
        required_policy_keys=required,
        clipping_norm=0.2,
        epsilon=1.0,
        delta=1e-6,
    )
    return [deepcopy(update.to_audit_record()) for update in report.node_updates]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("privacy_epsilon_spent", 0.0),
        ("policy_delta", [["K", 5.0], ["alpha", 0.0]]),
        ("sample_count", 10_000),
        ("accepted", False),
    ],
)
def test_edited_node_update_is_refused(field: str, value: object) -> None:
    """A node update edited after sealing is not transported."""
    records = _node_records()
    records[1][field] = value

    with pytest.raises(ValueError, match="update_hash of node 'site-b' does not match"):
        build_signed_transport_envelopes(records)


def test_aggregator_records_are_transported() -> None:
    """Unedited aggregator records, in either key order, build envelopes."""
    for required in (("K", "alpha"), ("alpha", "K")):
        envelopes = build_signed_transport_envelopes(_node_records(required))

        assert [envelope.node_id for envelope in envelopes] == [
            "site-a",
            "site-b",
            "site-c",
        ]


@pytest.mark.parametrize("payload", ["1e999", "-1e999", "NaN"])
def test_nonfinite_json_batch_identifier_refusal_preserves_sealed_updates(
    payload: str,
) -> None:
    """A decoded nonfinite batch identifier cannot seal or modify node evidence."""
    records = _node_records()
    original = deepcopy(records)
    batch_id = json.loads(payload)
    with pytest.raises(ValueError, match="numbers in transport records must be finite"):
        build_signed_transport_envelopes(records, batch_id=batch_id)
    assert records == original

    envelopes = build_signed_transport_envelopes(records, batch_id="recovered-batch")
    assert validate_federated_transport_batch(envelopes) == envelopes
    ledger = replay_federated_transport_batch(envelopes)
    assert ledger.batch_id == "recovered-batch"
    assert ledger.envelope_count == 3
    assert ledger.node_last_sequences == (("site-a", 1), ("site-b", 1), ("site-c", 1))
    assert all(
        envelope.transport_execution_permitted is False for envelope in envelopes
    )
    assert all(envelope.raw_data_export_permitted is False for envelope in envelopes)
    assert records == original


def test_decimal_json_batch_identifier_refusal_preserves_sealed_updates() -> None:
    """Decimal JSON decoding cannot smuggle a numeric identifier into a seal."""
    request = json.loads('{"batch_id": 0.5}', parse_float=Decimal)
    records = _node_records()
    original = deepcopy(records)
    with pytest.raises(
        ValueError, match="transport payload contains unsupported JSON type"
    ):
        build_signed_transport_envelopes(records, batch_id=request["batch_id"])
    assert records == original

    envelopes = build_signed_transport_envelopes(records, batch_id="decimal-recovery")
    assert validate_federated_transport_batch(envelopes) == envelopes
    assert replay_federated_transport_batch(envelopes).batch_id == "decimal-recovery"
    assert all(envelope.operator_review_required is True for envelope in envelopes)
    assert records == original
