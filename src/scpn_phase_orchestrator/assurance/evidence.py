# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case evidence items

"""Typed, content-addressed evidence items for the assurance-case bundle.

An :class:`EvidenceItem` wraps the JSON-safe audit record produced by an
existing SPO surface (audit-chain integrity, replay determinism, formal
verification, twin-confidence, or the conformal admission gate) together with a
content hash, so the bundle can reference evidence by a stable identifier and
detect any later mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)

AUDIT_LOGGING = "audit_logging"
REPLAY_DETERMINISM = "replay_determinism"
FORMAL_VERIFICATION = "formal_verification"
TWIN_CONFIDENCE = "twin_confidence"
CONFORMAL_GATE = "conformal_gate"

EVIDENCE_CATEGORIES: frozenset[str] = frozenset(
    {
        AUDIT_LOGGING,
        REPLAY_DETERMINISM,
        FORMAL_VERIFICATION,
        TWIN_CONFIDENCE,
        CONFORMAL_GATE,
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One content-addressed piece of assurance evidence.

    Parameters
    ----------
    evidence_id:
        Stable identifier, unique within a bundle (e.g.
        ``"audit-chain-integrity"``).
    category:
        One of :data:`EVIDENCE_CATEGORIES`.
    summary:
        Short human-readable description of what the evidence shows.
    record:
        The JSON-safe audit record produced by the originating surface.
    content_hash:
        SHA-256 of the canonical serialisation of ``record``. Defaults to the
        computed hash; an explicit mismatching value is rejected.
    """

    evidence_id: str
    category: str
    summary: str
    record: Mapping[str, object]
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        """Validate the evidence metadata and compute or verify its content hash."""
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        if self.category not in EVIDENCE_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(EVIDENCE_CATEGORIES)}, "
                f"got {self.category!r}"
            )
        if not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        if not isinstance(self.record, Mapping):
            raise ValueError("record must be a mapping")
        computed = canonical_record_hash(dict(self.record))
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        else:
            require_sha256(self.content_hash, "content_hash")
            if self.content_hash != computed:
                raise ValueError(
                    "content_hash does not match the canonical hash of record"
                )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe evidence record.

        Returns
        -------
        dict[str, object]
            A JSON-safe evidence record.
        """
        return {
            "evidence_id": self.evidence_id,
            "category": self.category,
            "summary": self.summary,
            "content_hash": self.content_hash,
            "record": dict(self.record),
        }


def build_evidence_item(
    evidence_id: str,
    category: str,
    summary: str,
    record: Mapping[str, object],
) -> EvidenceItem:
    """Construct an :class:`EvidenceItem` with a computed content hash.

    Parameters
    ----------
    evidence_id:
        Stable identifier, unique within a bundle.
    category:
        One of :data:`EVIDENCE_CATEGORIES`.
    summary:
        Short human-readable description.
    record:
        The JSON-safe audit record.

    Returns
    -------
    EvidenceItem
        The constructed evidence item.
    """
    return EvidenceItem(
        evidence_id=evidence_id,
        category=category,
        summary=summary,
        record=dict(record),
    )
