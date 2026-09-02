# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — producer evidence-state policy tests

"""Guard the epistemic-state versus physical-regime separation."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    PRODUCER_EVIDENCE_STATE_POLICIES,
    ProducerEvidenceDisposition,
    ProducerEvidenceStatePolicy,
    RegimeState,
    ValidityState,
    producer_evidence_state_policy,
)


def test_every_producer_disposition_maps_to_exact_u0_validity() -> None:
    expected = {
        ProducerEvidenceDisposition.UNKNOWN: ValidityState.UNKNOWN,
        ProducerEvidenceDisposition.OUT_OF_DISTRIBUTION: (
            ValidityState.OUT_OF_DISTRIBUTION
        ),
        ProducerEvidenceDisposition.LOW_OBSERVABILITY: ValidityState.UNOBSERVABLE,
        ProducerEvidenceDisposition.STALE: ValidityState.STALE,
    }

    assert tuple(policy.disposition for policy in PRODUCER_EVIDENCE_STATE_POLICIES) == (
        *ProducerEvidenceDisposition,
    )
    assert {
        policy.disposition: policy.validity_state
        for policy in PRODUCER_EVIDENCE_STATE_POLICIES
    } == expected


def test_producer_dispositions_always_abstain_from_physical_regime() -> None:
    for disposition in ProducerEvidenceDisposition:
        policy = producer_evidence_state_policy(disposition)
        assert policy.regime_state is RegimeState.UNKNOWN
        assert policy.physical_regime_classified is False
        assert policy.quality_may_substitute is False
        assert policy.to_record() == {
            "disposition": disposition.value,
            "meaning": policy.meaning,
            "physical_regime_classified": False,
            "quality_may_substitute": False,
            "regime_state": "unknown",
            "validity_state": policy.validity_state.value,
        }


def test_policy_resolver_refuses_untyped_string_alias() -> None:
    with pytest.raises(
        TypeError,
        match="disposition must be ProducerEvidenceDisposition",
    ):
        producer_evidence_state_policy("unknown")  # type: ignore[arg-type]


def test_policy_constructor_refuses_inconsistent_u0_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="producer evidence disposition has an invalid U0 validity mapping",
    ):
        ProducerEvidenceStatePolicy(
            ProducerEvidenceDisposition.LOW_OBSERVABILITY,
            ValidityState.DEGRADED,
            "below the predeclared gate",
        )


def test_disposition_meanings_fix_domain_gate_and_freshness_semantics() -> None:
    meanings = {
        policy.disposition: policy.meaning
        for policy in PRODUCER_EVIDENCE_STATE_POLICIES
    }

    assert (
        "more specific evidence disposition"
        in meanings[ProducerEvidenceDisposition.UNKNOWN]
    )
    assert (
        "validated applicability domain"
        in meanings[ProducerEvidenceDisposition.OUT_OF_DISTRIBUTION]
    )
    assert (
        "below the predeclared observability"
        in meanings[ProducerEvidenceDisposition.LOW_OBSERVABILITY]
    )
    assert (
        "merely small score above that gate does not qualify"
        in meanings[ProducerEvidenceDisposition.LOW_OBSERVABILITY]
    )
    assert "freshness or validity window" in meanings[ProducerEvidenceDisposition.STALE]
