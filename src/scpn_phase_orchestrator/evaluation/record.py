# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hash-sealed early-warning audit record

"""Seal a detector audit into a tamper-evident, hash-addressed record.

:func:`seal_detector_audit` binds a
:class:`~scpn_phase_orchestrator.evaluation.auditor.DetectorAudit` to the
provenance of the corpus it was measured on — an identifier and a
caller-supplied capture timestamp — and stamps the whole with a SHA-256 over its
canonical JSON. Recomputing the hash from the recorded fields detects any later
edit, so a published audit verdict cannot be quietly altered. The record makes no
claim beyond the supplied corpus and detector; the disclaimer says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:
    from scpn_phase_orchestrator.evaluation.auditor import DetectorAudit

__all__ = [
    "AUDIT_DISCLAIMER",
    "AUDIT_FRAMEWORK",
    "AuditRecord",
    "seal_detector_audit",
]

#: The methodology the audit implements, carried into every sealed record.
AUDIT_FRAMEWORK = (
    "Matched-false-alarm calibration with a label-permutation significance test "
    "(exchangeability null), after Boettiger & Hastings 2012, J. R. Soc. "
    "Interface 9:2527."
)

#: The honest-scope disclaimer sealed with every audit record.
AUDIT_DISCLAIMER = (
    "This audit record measures a detector's event-vs-null skill on the supplied "
    "corpus only: the matched-false-alarm threshold, the rate it achieved, the "
    "event detection rate, and the permutation p-value of the event alarm count. "
    "It is not a certification of field performance and does not transfer to any "
    "other corpus, detector configuration, or deployment; a low p-value means the "
    "events alarmed more than the matched false alarm on this corpus, nothing more."
)


@dataclass(frozen=True)
class AuditRecord:
    """A hash-sealed early-warning detector audit bound to its corpus provenance.

    Attributes
    ----------
    corpus_id : str
        Provenance identifier of the event-and-null corpus the audit ran on.
    captured_at : str
        Caller-supplied timestamp of the audit, echoed for provenance.
    audit : dict[str, object]
        The JSON-safe audit verdict (:meth:`DetectorAudit.to_record`).
    framework : str
        The methodology label (:data:`AUDIT_FRAMEWORK`).
    disclaimer : str
        The honest-scope disclaimer (:data:`AUDIT_DISCLAIMER`).
    content_hash : str
        SHA-256 over the canonical payload, computed at construction.
    """

    corpus_id: str
    captured_at: str
    audit: dict[str, object]
    framework: str = AUDIT_FRAMEWORK
    disclaimer: str = AUDIT_DISCLAIMER
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute the content hash from the canonical audit payload."""
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload the content hash is taken over."""
        return {
            "corpus_id": self.corpus_id,
            "captured_at": self.captured_at,
            "audit": self.audit,
            "framework": self.framework,
            "disclaimer": self.disclaimer,
        }

    def to_record(self) -> dict[str, object]:
        """Return the canonical payload plus the computed ``content_hash``.

        Returns
        -------
        dict[str, object]
            The corpus provenance, the audit verdict, the framework and
            disclaimer, and the ``content_hash`` sealed over them.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record

    def verify(self) -> bool:
        """Return whether the stored hash still matches the recorded fields.

        Returns
        -------
        bool
            ``True`` when recomputing the content hash from the current fields
            reproduces the stored ``content_hash``; ``False`` after any tampering.
        """
        return self.content_hash == canonical_record_hash(self._canonical_payload())


def seal_detector_audit(
    audit: DetectorAudit, *, corpus_id: str, captured_at: str
) -> AuditRecord:
    """Seal a detector audit into a hash-addressed record.

    Parameters
    ----------
    audit : DetectorAudit
        The audit verdict to seal.
    corpus_id : str
        Provenance identifier of the corpus the audit ran on; must be non-empty.
    captured_at : str
        Caller-supplied timestamp of the audit; must be non-empty.

    Returns
    -------
    AuditRecord
        The sealed, hash-addressed record.

    Raises
    ------
    ValueError
        If ``corpus_id`` or ``captured_at`` is empty.
    """
    if not corpus_id:
        raise ValueError("corpus_id must not be empty")
    if not captured_at:
        raise ValueError("captured_at must not be empty")
    return AuditRecord(
        corpus_id=corpus_id,
        captured_at=captured_at,
        audit=audit.to_record(),
    )
