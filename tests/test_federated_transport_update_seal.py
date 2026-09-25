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

from copy import deepcopy

import pytest

from scpn_phase_orchestrator.supervisor.federated import (
    build_federated_meta_orchestrator_manifest,
)
from scpn_phase_orchestrator.supervisor.federated_transport import (
    build_signed_transport_envelopes,
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
