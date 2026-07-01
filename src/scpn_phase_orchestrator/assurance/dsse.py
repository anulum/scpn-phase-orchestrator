# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — DSSE attestation envelope (post-quantum signed)

"""Wrap a SLSA provenance statement in a post-quantum-signed DSSE envelope.

`DSSE <https://github.com/secure-systems-lab/dsse>`_ (Dead Simple Signing
Envelope) is the wire format `sigstore/cosign <https://docs.sigstore.dev>`_ produces
for ``cosign attest`` and consumes for ``cosign verify-attestation``: a base64
payload, its ``payloadType``, and a list of signatures over the DSSE
*pre-authentication encoding* (PAE) of the payload. Signing the PAE — rather than the
raw JSON — is what makes the signature bind the payload type as well as the bytes, so
an attestation cannot be re-labelled as a different document type.

This module carries the SLSA provenance statement from
:mod:`scpn_phase_orchestrator.assurance.provenance` as the DSSE payload
(``payloadType`` ``application/vnd.in-toto+json``) and signs the PAE with **ML-DSA**
(FIPS 204), reusing the single post-quantum primitive in
:mod:`scpn_phase_orchestrator.runtime.audit_pqc`. Each signature records its
``algorithm`` so a second scheme can be added without breaking existing envelopes;
**SLH-DSA** (FIPS 205 / SPHINCS+) is the reserved hash-based alternative and will be
added once the ``cryptography`` backend ships it (it does not as of the pinned
version, so no SLH-DSA claim is made here).

The envelope is deterministic and offline: it holds no timestamps and makes no
network call. Verification is self-contained — the verifier supplies the trusted
public key, whose short id must match the signature ``keyid`` — so an attestation can
be checked long after the build and against a future quantum adversary. Pushing the
same envelope to a Rekor transparency log or verifying it with ``cosign`` is an
optional operator step that needs network and OIDC, and is therefore out of this
deterministic core.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scpn_phase_orchestrator.assurance.provenance import SlsaProvenanceStatement
from scpn_phase_orchestrator.runtime.audit_pqc import (
    DEFAULT_VARIANT,
    MLDSA_VARIANTS,
    public_key_id,
    sign_bytes,
    verify_bytes,
)

DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json"
_PAE_PREFIX = b"DSSEv1"


def _pae(payload_type: str, body: bytes) -> bytes:
    """Return the DSSE pre-authentication encoding of a payload.

    ``PAE(type, body) = "DSSEv1" SP len(type) SP type SP len(body) SP body`` where
    the lengths are decimal byte counts. Signing the PAE binds both the payload bytes
    and the payload type.
    """
    type_bytes = payload_type.encode("utf-8")
    return b" ".join(
        [
            _PAE_PREFIX,
            str(len(type_bytes)).encode("ascii"),
            type_bytes,
            str(len(body)).encode("ascii"),
            body,
        ]
    )


def _require_algorithm(algorithm: str) -> str:
    """Return ``algorithm`` if it is a supported signature variant, else raise."""
    if algorithm not in MLDSA_VARIANTS:
        raise ValueError(
            f"algorithm must be one of {MLDSA_VARIANTS}, got {algorithm!r}"
        )
    return algorithm


@dataclass(frozen=True, slots=True)
class DsseSignature:
    """One signature over a DSSE envelope's pre-authentication encoding.

    Attributes
    ----------
    keyid:
        Short identifier of the signing public key (SHA-256 prefix), matching
        :func:`~scpn_phase_orchestrator.runtime.audit_pqc.public_key_id`.
    algorithm:
        The signature scheme (an ML-DSA variant), recorded so a second scheme can
        be added without ambiguity.
    signature_b64:
        The raw signature, standard-base64 encoded.
    """

    keyid: str
    algorithm: str
    signature_b64: str

    def __post_init__(self) -> None:
        """Validate the algorithm and that the signature is decodable base64."""
        _require_algorithm(self.algorithm)
        if not isinstance(self.keyid, str) or not self.keyid:
            raise ValueError("keyid must be a non-empty string")
        _decode_b64(self.signature_b64, "signature_b64")

    def to_dict(self) -> dict[str, str]:
        """Return the DSSE signature mapping (``keyid`` / ``algorithm`` / ``sig``).

        Returns
        -------
        dict[str, str]
            The ``keyid`` / ``algorithm`` / ``sig`` wire mapping.
        """
        return {
            "keyid": self.keyid,
            "algorithm": self.algorithm,
            "sig": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DsseSignature:
        """Return a signature parsed from a DSSE signature mapping.

        Parameters
        ----------
        data:
            A mapping carrying ``keyid``, ``algorithm``, and ``sig``.

        Returns
        -------
        DsseSignature
            The reconstructed signature.

        Raises
        ------
        ValueError
            If a required field is missing or malformed.
        """
        for required in ("keyid", "algorithm", "sig"):
            if required not in data:
                raise ValueError(f"signature is missing field: {required}")
        return cls(
            keyid=str(data["keyid"]),
            algorithm=str(data["algorithm"]),
            signature_b64=str(data["sig"]),
        )


def _encode_b64(raw: bytes) -> str:
    """Return the standard-base64 encoding of ``raw``."""
    return base64.b64encode(raw).decode("ascii")


def _decode_b64(value: str, field_name: str) -> bytes:
    """Return the base64-decoded bytes of ``value``, else raise ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} must be valid base64") from exc


@dataclass(frozen=True, slots=True)
class DsseEnvelope:
    """A DSSE v1 envelope carrying a base64 payload and its signatures.

    Attributes
    ----------
    payload_b64:
        The statement JSON, standard-base64 encoded.
    payload_type:
        The payload media type (``application/vnd.in-toto+json``).
    signatures:
        The signatures over the payload's pre-authentication encoding.
    """

    payload_b64: str
    payload_type: str
    signatures: tuple[DsseSignature, ...]

    def __post_init__(self) -> None:
        """Validate a decodable base64 payload and at least one signature."""
        _decode_b64(self.payload_b64, "payload")
        if not isinstance(self.payload_type, str) or not self.payload_type:
            raise ValueError("payloadType must be a non-empty string")
        if not self.signatures:
            raise ValueError("envelope requires at least one signature")

    def payload_bytes(self) -> bytes:
        """Return the decoded payload bytes.

        Returns
        -------
        bytes
            The base64-decoded payload (the canonical statement JSON bytes).
        """
        return _decode_b64(self.payload_b64, "payload")

    def statement(self) -> dict[str, object]:
        """Return the wrapped in-toto statement as a mapping.

        Returns
        -------
        dict[str, object]
            The decoded, JSON-parsed statement.

        Raises
        ------
        ValueError
            If the payload is not valid JSON object bytes.
        """
        try:
            parsed = json.loads(self.payload_bytes())
        except json.JSONDecodeError as exc:
            raise ValueError("payload is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("payload is not a JSON object")
        return parsed

    def to_dict(self) -> dict[str, object]:
        """Return the DSSE wire mapping (payload / payloadType / signatures).

        Returns
        -------
        dict[str, object]
            The ``payload`` / ``payloadType`` / ``signatures`` wire mapping.
        """
        return {
            "payload": self.payload_b64,
            "payloadType": self.payload_type,
            "signatures": [signature.to_dict() for signature in self.signatures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DsseEnvelope:
        """Return an envelope parsed from a DSSE wire mapping.

        Parameters
        ----------
        data:
            A mapping carrying ``payload``, ``payloadType``, and ``signatures``.

        Returns
        -------
        DsseEnvelope
            The reconstructed envelope.

        Raises
        ------
        ValueError
            If a required field is missing or the signatures are malformed.
        """
        for required in ("payload", "payloadType", "signatures"):
            if required not in data:
                raise ValueError(f"envelope is missing field: {required}")
        raw_signatures = data["signatures"]
        if not isinstance(raw_signatures, (list, tuple)):
            raise ValueError("signatures must be a list")
        signatures = tuple(
            DsseSignature.from_dict(_as_mapping(item)) for item in raw_signatures
        )
        return cls(
            payload_b64=str(data["payload"]),
            payload_type=str(data["payloadType"]),
            signatures=signatures,
        )


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return ``value`` as a mapping, else raise ``ValueError``."""
    if not isinstance(value, Mapping):
        raise ValueError("signature entry must be an object")
    return value


def sign_provenance_statement(
    statement: SlsaProvenanceStatement,
    private_key: Any,
    *,
    algorithm: str = DEFAULT_VARIANT,
) -> DsseEnvelope:
    """Wrap a provenance statement in a DSSE envelope and sign it with ML-DSA.

    Parameters
    ----------
    statement:
        The SLSA provenance statement to attest.
    private_key:
        An ML-DSA private key matching ``algorithm`` (see
        :func:`~scpn_phase_orchestrator.runtime.audit_pqc.signing_key_from_seed`).
    algorithm:
        The ML-DSA variant; must match ``private_key``.

    Returns
    -------
    DsseEnvelope
        The signed attestation envelope.

    Raises
    ------
    ValueError
        If the algorithm or private key is invalid.
    """
    algorithm = _require_algorithm(algorithm)
    body = json.dumps(
        statement.to_statement(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    signature = sign_bytes(
        _pae(DSSE_PAYLOAD_TYPE, body), private_key, algorithm=algorithm
    )
    public_bytes = private_key.public_key().public_bytes_raw()
    dsse_signature = DsseSignature(
        keyid=public_key_id(public_bytes),
        algorithm=algorithm,
        signature_b64=_encode_b64(signature),
    )
    return DsseEnvelope(
        payload_b64=_encode_b64(body),
        payload_type=DSSE_PAYLOAD_TYPE,
        signatures=(dsse_signature,),
    )


def verify_dsse_envelope(
    envelope: DsseEnvelope,
    trusted_public_key_hex: str,
) -> bool:
    """Verify that an envelope carries a valid signature under a trusted key.

    The verifier supplies the public key it trusts; the envelope is accepted only if
    a signature whose ``keyid`` matches that key verifies over the payload's
    pre-authentication encoding. This binds the attestation to a known signer, so a
    forged envelope signed under a different key is rejected.

    Parameters
    ----------
    envelope:
        The DSSE envelope to verify.
    trusted_public_key_hex:
        The hex-encoded raw ML-DSA public key the verifier trusts.

    Returns
    -------
    bool
        ``True`` if a matching signature verifies, else ``False``.

    Raises
    ------
    ValueError
        If the trusted key is not a hex string.
    """
    if not isinstance(trusted_public_key_hex, str):
        raise ValueError("trusted_public_key_hex must be a hex string")
    try:
        trusted_bytes = bytes.fromhex(trusted_public_key_hex)
    except ValueError as exc:
        raise ValueError("trusted_public_key_hex must be valid hex") from exc
    trusted_keyid = public_key_id(trusted_bytes)
    body = envelope.payload_bytes()
    pae = _pae(envelope.payload_type, body)
    for signature in envelope.signatures:
        # Every signature's algorithm is validated at construction, so no algorithm
        # re-check is needed here — only the keyid must match the trusted key.
        if signature.keyid != trusted_keyid:
            continue
        raw_signature = _decode_b64(signature.signature_b64, "signature_b64")
        if verify_bytes(
            pae, raw_signature, trusted_public_key_hex, algorithm=signature.algorithm
        ):
            return True
    return False


__all__ = [
    "DSSE_PAYLOAD_TYPE",
    "DsseEnvelope",
    "DsseSignature",
    "sign_provenance_statement",
    "verify_dsse_envelope",
]
