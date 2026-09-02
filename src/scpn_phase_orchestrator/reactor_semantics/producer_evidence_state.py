# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — producer evidence-state semantics
"""Separate producer evidence disposition from physical reactor regime.

These states describe why producer evidence cannot support a current physical
classification. They are not plasma states, operating modes, phase labels, or
quality grades. The only permitted regime projection is an unclassified
``RegimeState.UNKNOWN`` result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from .vocabulary import RegimeState, ValidityState


class ProducerEvidenceDisposition(StrEnum):
    """Producer-owned reason that current plant truth is not classifiable."""

    UNKNOWN = "unknown"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    LOW_OBSERVABILITY = "low_observability"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class ProducerEvidenceStatePolicy:
    """One exact disposition-to-U0 validity and abstention policy."""

    disposition: ProducerEvidenceDisposition
    validity_state: ValidityState
    meaning: str
    regime_state: RegimeState = field(init=False, default=RegimeState.UNKNOWN)
    physical_regime_classified: bool = field(init=False, default=False)
    quality_may_substitute: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ProducerEvidenceDisposition):
            raise TypeError("disposition must be ProducerEvidenceDisposition")
        if not isinstance(self.validity_state, ValidityState):
            raise TypeError("validity_state must be ValidityState")
        if not isinstance(self.meaning, str) or not self.meaning.strip():
            raise ValueError("producer evidence disposition meaning must be non-empty")
        expected_validity = {
            ProducerEvidenceDisposition.UNKNOWN: ValidityState.UNKNOWN,
            ProducerEvidenceDisposition.OUT_OF_DISTRIBUTION: (
                ValidityState.OUT_OF_DISTRIBUTION
            ),
            ProducerEvidenceDisposition.LOW_OBSERVABILITY: (ValidityState.UNOBSERVABLE),
            ProducerEvidenceDisposition.STALE: ValidityState.STALE,
        }[self.disposition]
        if self.validity_state is not expected_validity:
            raise ValueError(
                "producer evidence disposition has an invalid U0 validity mapping"
            )

    def to_record(self) -> dict[str, object]:
        """Return the deterministic public policy record."""
        return {
            "disposition": self.disposition.value,
            "meaning": self.meaning,
            "physical_regime_classified": self.physical_regime_classified,
            "quality_may_substitute": self.quality_may_substitute,
            "regime_state": self.regime_state.value,
            "validity_state": self.validity_state.value,
        }


PRODUCER_EVIDENCE_STATE_POLICIES: Final = (
    ProducerEvidenceStatePolicy(
        ProducerEvidenceDisposition.UNKNOWN,
        ValidityState.UNKNOWN,
        "The producer cannot assign a more specific evidence disposition and "
        "cannot classify the current physical regime.",
    ),
    ProducerEvidenceStatePolicy(
        ProducerEvidenceDisposition.OUT_OF_DISTRIBUTION,
        ValidityState.OUT_OF_DISTRIBUTION,
        "Evidence or estimator input lies outside its versioned validated "
        "applicability domain.",
    ),
    ProducerEvidenceStatePolicy(
        ProducerEvidenceDisposition.LOW_OBSERVABILITY,
        ValidityState.UNOBSERVABLE,
        "The target lies below the predeclared observability or minimum-evidence "
        "gate; a merely small score above that gate does not qualify.",
    ),
    ProducerEvidenceStatePolicy(
        ProducerEvidenceDisposition.STALE,
        ValidityState.STALE,
        "Evidence lies outside its declared freshness or validity window for the "
        "current classification time.",
    ),
)

_POLICY_BY_DISPOSITION: Final = {
    policy.disposition: policy for policy in PRODUCER_EVIDENCE_STATE_POLICIES
}


def producer_evidence_state_policy(
    disposition: ProducerEvidenceDisposition,
) -> ProducerEvidenceStatePolicy:
    """Resolve one typed producer disposition without accepting string aliases."""
    if not isinstance(disposition, ProducerEvidenceDisposition):
        raise TypeError("disposition must be ProducerEvidenceDisposition")
    return _POLICY_BY_DISPOSITION[disposition]


__all__ = [
    "PRODUCER_EVIDENCE_STATE_POLICIES",
    "ProducerEvidenceDisposition",
    "ProducerEvidenceStatePolicy",
    "producer_evidence_state_policy",
]
