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

The bare SHA-256 seal is tamper-*evident* but not tamper-*resistant*: anyone who
can edit the record can recompute the hash. Pass a signing ``key`` (or call
:meth:`AuditRecord.sign`) to add an HMAC-SHA256 signature over the content hash,
reusing the same key discovery as the audit log
(:mod:`~scpn_phase_orchestrator.runtime.audit_signing`); a verifier without the
secret then cannot forge an edit. Signing is additive — an unsigned record's
serialisation and content hash are unchanged, so records sealed before signing
existed still verify.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.runtime.audit_signing import (
    SIGNATURE_ALGORITHM,
    key_id_for_secret,
)

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
    signature : str | None
        HMAC-SHA256 over :attr:`content_hash`, or ``None`` for an unsigned record.
        It sits outside the hashed payload, so signing leaves the content hash and
        the serialisation of an unsigned record unchanged.
    signing_key_id : str | None
        Identifier of the key that produced :attr:`signature`
        (:func:`~scpn_phase_orchestrator.runtime.audit_signing.key_id_for_secret`),
        or ``None`` when unsigned.
    content_hash : str
        SHA-256 over the canonical payload, computed at construction.
    """

    corpus_id: str
    captured_at: str
    audit: dict[str, object]
    framework: str = AUDIT_FRAMEWORK
    disclaimer: str = AUDIT_DISCLAIMER
    signature: str | None = None
    signing_key_id: str | None = None
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
        if self.signature is not None:
            record["signature"] = self.signature
            record["signing_key_id"] = self.signing_key_id
            record["signature_algorithm"] = SIGNATURE_ALGORITHM
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

    def sign(self, key: str) -> AuditRecord:
        """Return a copy carrying an HMAC-SHA256 signature over the content hash.

        The signature covers :attr:`content_hash` — itself a hash of the whole
        payload — so it authenticates the record without changing what the content
        hash is taken over. A verifier holding the secret (or a keyring including
        it) can then confirm the record was sealed by a key holder, not merely
        left internally consistent by whoever last edited it.

        Parameters
        ----------
        key : str
            The HMAC signing-key material; must be non-empty.

        Returns
        -------
        AuditRecord
            An otherwise-identical record with :attr:`signature` and
            :attr:`signing_key_id` populated.

        Raises
        ------
        ValueError
            If ``key`` is empty.
        """
        key_id = key_id_for_secret(key)
        signature = hmac.new(
            key.encode(), self.content_hash.encode(), hashlib.sha256
        ).hexdigest()
        return AuditRecord(
            corpus_id=self.corpus_id,
            captured_at=self.captured_at,
            audit=self.audit,
            framework=self.framework,
            disclaimer=self.disclaimer,
            signature=signature,
            signing_key_id=key_id,
        )

    def verify_signature(self, keys: dict[str, str]) -> bool:
        """Return whether the signature verifies against a known key.

        Parameters
        ----------
        keys : dict[str, str]
            Candidate verification keys by key id, e.g. from
            :func:`~scpn_phase_orchestrator.runtime.audit_signing.audit_verification_keys`.

        Returns
        -------
        bool
            ``True`` only when the record is signed, its key id is present in
            ``keys``, and the HMAC over :attr:`content_hash` matches (constant-time
            comparison). An unsigned record, an unknown key id, or a mismatch all
            return ``False``.
        """
        if self.signature is None or self.signing_key_id is None:
            return False
        key = keys.get(self.signing_key_id)
        if key is None:
            return False
        expected = hmac.new(
            key.encode(), self.content_hash.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)


def seal_detector_audit(
    audit: DetectorAudit,
    *,
    corpus_id: str,
    captured_at: str,
    key: str | None = None,
) -> AuditRecord:
    """Seal a detector audit into a hash-addressed record, optionally signed.

    Parameters
    ----------
    audit : DetectorAudit
        The audit verdict to seal.
    corpus_id : str
        Provenance identifier of the corpus the audit ran on; must be non-empty.
    captured_at : str
        Caller-supplied timestamp of the audit; must be non-empty.
    key : str | None
        An HMAC signing key. When given, the sealed record is signed
        (:meth:`AuditRecord.sign`); when ``None`` the record is left unsigned with
        its bare content-hash seal.

    Returns
    -------
    AuditRecord
        The sealed, hash-addressed record — signed when ``key`` was supplied.

    Raises
    ------
    ValueError
        If ``corpus_id`` or ``captured_at`` is empty, or ``key`` is an empty string.
    """
    if not corpus_id:
        raise ValueError("corpus_id must not be empty")
    if not captured_at:
        raise ValueError("captured_at must not be empty")
    record = AuditRecord(
        corpus_id=corpus_id,
        captured_at=captured_at,
        audit=audit.to_record(),
    )
    if key is None:
        return record
    return record.sign(key)
