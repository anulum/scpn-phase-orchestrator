# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin registry shared-helper tests

"""Shared plugin-registry validation contracts."""

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.plugins.registry._shared as shared
from scpn_phase_orchestrator.plugins import PluginRuntimeExecutionPolicy


def test_runtime_policy_accepts_valid_approved_target_hash() -> None:
    """Runtime policy accepts canonical SHA-256 target approvals."""
    policy = PluginRuntimeExecutionPolicy(
        approved_target_hashes=("a" * 64,),
        require_target_hash_approval=True,
    )

    assert policy.approved_target_hashes == ("a" * 64,)


@pytest.mark.parametrize(
    ("target_hash", "message"),
    (
        ("a" * 63, "64-character SHA-256 hex digest"),
        ("z" * 64, "SHA-256 hex digest"),
    ),
)
def test_runtime_policy_rejects_malformed_approved_target_hashes(
    target_hash: str,
    message: str,
) -> None:
    """Runtime policy fails closed on malformed target-approval digests."""
    with pytest.raises(ValueError, match=message):
        PluginRuntimeExecutionPolicy(approved_target_hashes=(target_hash,))


def test_record_hash_is_order_insensitive_and_content_sensitive() -> None:
    """Canonical record hashing is stable across JSON key order."""
    left = {"plugin": "grid_pack", "priority": 2}
    right = {"priority": 2, "plugin": "grid_pack"}

    assert shared._record_hash(left) == shared._record_hash(right)
    assert shared._record_hash(left) != shared._record_hash(
        {"plugin": "grid_pack", "priority": 3}
    )
