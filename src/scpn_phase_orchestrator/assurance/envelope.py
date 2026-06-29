# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — signed, replay-anchored certification envelope

"""Bind a certification package to the run that produced it, optionally PQC-signed.

A :class:`~scpn_phase_orchestrator.assurance.certification.CertificationEvidencePackage`
is hash-sealed but free-floating: its ``package_hash`` proves the package's own
contents are consistent, yet nothing ties it to the *specific run* whose evidence
it carries. This module adds that outer binding.

A :class:`SignedCertificationEnvelope` commits, in one deterministic record, to:

* the package's ``package_hash`` (which run-evidence the package describes);
* the **audit-chain tip** of the run — the SHA-256 commitment to the whole audit
  log (``_hash`` of the last record), so the envelope is anchored to a specific,
  tamper-evident execution that can be replayed and re-verified; and
* an optional **post-quantum seal** over that tip
  (:class:`~scpn_phase_orchestrator.runtime.audit_pqc.AuditChainSeal`, ML-DSA /
  FIPS 204), making the binding publicly verifiable long after the run.

The envelope reuses the audit seal verbatim — the seal genuinely commits to an
audit-chain tip under its own domain, so there is no cross-protocol confusion: the
envelope merely records that the package describes *that sealed run*. It performs
no signing or log reading itself (the CLI layer reads the tip and produces the seal
with a signing key); it validates the pieces, requires any attached seal to commit
to the same tip and record count, and seals the binding with a deterministic
``envelope_hash``. Verification re-derives the hash, checks the package binding, and
— when a seal is present — verifies it against a trusted public key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from scpn_phase_orchestrator.assurance._hashing import (
    canonical_record_hash,
    require_sha256,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    AuditChainSeal,
    verify_audit_chain_seal,
)

SIGNED_CERTIFICATION_ENVELOPE_SCHEMA = "scpn_signed_certification_envelope_v1"


def _validate_record_count(record_count: object) -> int:
    """Return the validated audit record count, else raise ``ValueError``."""
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError("audit_record_count must be a non-negative integer")
    if record_count < 0:
        raise ValueError("audit_record_count must be a non-negative integer")
    return record_count


@dataclass(frozen=True, slots=True)
class SignedCertificationEnvelope:
    """A deterministic binding of a certification package to its run.

    Attributes
    ----------
    package_hash:
        The ``package_hash`` of the bound certification package (SHA-256 hex).
    audit_chain_tip:
        The run's audit-chain tip — ``_hash`` of the last audit record, the
        SHA-256 commitment to the whole log (32-byte digest, hex).
    audit_record_count:
        The number of records in the sealed audit chain.
    seal:
        The post-quantum seal over the tip as a JSON-safe mapping
        (:meth:`~scpn_phase_orchestrator.runtime.audit_pqc.AuditChainSeal.to_dict`),
        or ``None`` for an unsigned (anchor-only) envelope.
    envelope_hash:
        SHA-256 over the canonical serialisation of the schema, package hash,
        audit tip, record count, and seal. Defaults to the computed hash; an
        explicit mismatching value is rejected.
    """

    package_hash: str
    audit_chain_tip: str
    audit_record_count: int
    seal: Mapping[str, str | int] | None
    envelope_hash: str = ""

    def __post_init__(self) -> None:
        """Validate the fields and compute or check the sealing ``envelope_hash``."""
        require_sha256(self.package_hash, "package_hash")
        require_sha256(self.audit_chain_tip, "audit_chain_tip")
        _validate_record_count(self.audit_record_count)
        computed = canonical_record_hash(self._sealable())
        if not self.envelope_hash:
            object.__setattr__(self, "envelope_hash", computed)
        else:
            require_sha256(self.envelope_hash, "envelope_hash")
            if self.envelope_hash != computed:
                raise ValueError(
                    "envelope_hash does not match the canonical hash of the envelope"
                )

    def _sealable(self) -> dict[str, object]:
        """Return the canonical mapping the ``envelope_hash`` commits to."""
        return {
            "schema": SIGNED_CERTIFICATION_ENVELOPE_SCHEMA,
            "package_hash": self.package_hash,
            "audit_chain_tip": self.audit_chain_tip,
            "audit_record_count": self.audit_record_count,
            "seal": dict(self.seal) if self.seal is not None else None,
        }

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe record of the envelope.

        Returns
        -------
        dict[str, object]
            The schema tag, package hash, audit anchor, optional seal, and the
            sealing ``envelope_hash``.
        """
        record = self._sealable()
        record["envelope_hash"] = self.envelope_hash
        return record


def build_signed_certification_envelope(
    package_hash: str,
    audit_chain_tip: str,
    audit_record_count: int,
    *,
    seal: AuditChainSeal | None = None,
) -> SignedCertificationEnvelope:
    """Bind a certification package hash to a run's audit-chain tip.

    Parameters
    ----------
    package_hash:
        The ``package_hash`` of the certification package to anchor (SHA-256 hex).
    audit_chain_tip:
        The run's audit-chain tip hash (32-byte SHA-256 digest, hex).
    audit_record_count:
        The number of records in the sealed audit chain.
    seal:
        An optional post-quantum seal over the same tip. When given, it must
        commit to ``audit_chain_tip`` and ``audit_record_count``.

    Returns
    -------
    SignedCertificationEnvelope
        The deterministic, optionally signed binding.

    Raises
    ------
    ValueError
        If a field is malformed or a supplied seal commits to a different tip or
        record count than the anchor.
    """
    require_sha256(package_hash, "package_hash")
    require_sha256(audit_chain_tip, "audit_chain_tip")
    _validate_record_count(audit_record_count)
    seal_record: Mapping[str, str | int] | None = None
    if seal is not None:
        if seal.tip_hash != audit_chain_tip:
            raise ValueError("seal tip_hash does not match the anchor audit_chain_tip")
        if seal.record_count != audit_record_count:
            raise ValueError(
                "seal record_count does not match the anchor audit_record_count"
            )
        seal_record = seal.to_dict()
    return SignedCertificationEnvelope(
        package_hash=package_hash,
        audit_chain_tip=audit_chain_tip,
        audit_record_count=audit_record_count,
        seal=seal_record,
    )


def verify_signed_certification_envelope(
    envelope: SignedCertificationEnvelope,
    *,
    package_hash: str,
    trusted_public_key_hex: str | None = None,
) -> bool:
    """Verify an envelope binds a package and, if signed, carries a valid seal.

    Parameters
    ----------
    envelope:
        The envelope to verify.
    package_hash:
        The ``package_hash`` of the package the envelope is expected to bind. The
        envelope is rejected if it anchors a different package.
    trusted_public_key_hex:
        The hex-encoded raw ML-DSA public key the verifier trusts. Required when
        the envelope carries a seal; ignored for an anchor-only envelope.

    Returns
    -------
    bool
        ``True`` only if the envelope hash re-derives, the bound package hash
        matches, and any attached seal verifies under the trusted key for the
        anchored tip. ``False`` otherwise (including a sealed envelope verified
        without a trusted key).
    """
    if envelope.envelope_hash != canonical_record_hash(envelope._sealable()):
        return False
    if envelope.package_hash != package_hash:
        return False
    if envelope.seal is None:
        return True
    if trusted_public_key_hex is None:
        return False
    seal = AuditChainSeal.from_dict(dict(envelope.seal))
    if seal.tip_hash != envelope.audit_chain_tip:
        return False
    if seal.record_count != envelope.audit_record_count:
        return False
    return verify_audit_chain_seal(seal, trusted_public_key_hex)


__all__ = [
    "SIGNED_CERTIFICATION_ENVELOPE_SCHEMA",
    "SignedCertificationEnvelope",
    "build_signed_certification_envelope",
    "verify_signed_certification_envelope",
]
