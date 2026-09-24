# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SHA-256 digest format checks

"""Prove a 64-character string that is not a hex digest is refused as a hash.

Several validators checked the length and then ``int(value, 16)`` (or
``bytes.fromhex``). Python's ``int`` accepts a sign, a ``0x`` prefix,
underscores and surrounding whitespace, and ``bytes.fromhex`` skips whitespace
between bytes, so such strings were accepted as digests. A revocation or an
approved-target entry written that way can never match a real digest, so it
was accepted and silently had no effect.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.plugins.registry.policy import (
    PluginRuntimeExecutionPolicy,
)
from scpn_phase_orchestrator.runtime.audit_pqc import seal_audit_chain
from scpn_phase_orchestrator.supervisor.byzantine import sign_policy_proposal
from scpn_phase_orchestrator.supervisor.federated import (
    build_federated_meta_orchestrator_manifest,
)
from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceRequestManifest,
)

NOT_DIGESTS = [
    "0x" + "a" * 62,
    "-" + "a" * 63,
    "+" + "a" * 63,
    " " + "a" * 63,
    "a" * 32 + "_" + "a" * 31,
    "a" * 63 + "g",
]


@pytest.mark.parametrize("value", NOT_DIGESTS)
def test_plugin_policy_refuses_a_non_digest_target_hash(value: str) -> None:
    with pytest.raises(ValueError, match="SHA-256 hex digest"):
        PluginRuntimeExecutionPolicy(approved_target_hashes=(value,))


def test_plugin_policy_still_accepts_upper_and_lower_case_hex() -> None:
    PluginRuntimeExecutionPolicy(approved_target_hashes=("A" * 64, "b" * 64))


@pytest.mark.parametrize("value", NOT_DIGESTS)
def test_byzantine_proposal_refuses_a_non_digest_audit_hash(value: str) -> None:
    with pytest.raises(ValueError, match="previous_audit_hash must be"):
        sign_policy_proposal(
            node_id="node-a",
            payload={"K": 0.1},
            previous_audit_hash=value,
            signing_key="key",
        )


@pytest.mark.parametrize("value", NOT_DIGESTS)
def test_federated_update_refuses_a_non_digest_audit_hash(value: str) -> None:
    update = {
        "node_id": "site-a",
        "policy_delta": {"K": 0.1},
        "sample_count": 10,
        "local_loss": 0.2,
        "previous_audit_hash": value,
        "privacy_epsilon_spent": 0.5,
    }
    with pytest.raises(ValueError, match="previous_audit_hash must be"):
        build_federated_meta_orchestrator_manifest(
            (update,),
            required_policy_keys=("K",),
            clipping_norm=0.2,
            epsilon=1.0,
            delta=1e-6,
        )


@pytest.mark.parametrize("value", NOT_DIGESTS)
def test_dp_noise_request_refuses_a_non_digest_seed_hash(value: str) -> None:
    with pytest.raises(ValueError, match="seed_hash must be"):
        DpNoiseServiceRequestManifest(
            epsilon=1.0,
            delta=1e-6,
            sensitivity=1.0,
            noise_multiplier=1.0,
            node_count=1,
            seed_hash=value,
            policy_keys=("alpha",),
            node_budgets=(DpNoiseNodePrivacyBudget("node-a", 0.1),),
        )


def test_audit_seal_refuses_a_spaced_tip_hash() -> None:
    """32 bytes of hex separated by spaces is not the chain's tip digest text."""
    spaced = " ".join(["ab"] * 32)
    with pytest.raises(ValueError, match="tip_hash must be a 32-byte"):
        seal_audit_chain(spaced, 1, None)
