# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Post-quantum audit-chain seal (ML-DSA / FIPS 204)

"""Post-quantum seal over the audit hash chain.

The audit logger already chains every record with SHA-256 and signs each one
with HMAC. HMAC is symmetric: a verifier needs the secret key, and the scheme
is not post-quantum. This module adds an *additive* second seal — it does not
touch the HMAC flow, so there is no regression risk — that commits to the whole
log with a **post-quantum, publicly verifiable** signature.

The chain tip (`_hash` of the last record) is the SHA-256 commitment to the
entire log: change any record and the tip changes. ``seal_audit_log`` signs that
tip with **ML-DSA** (FIPS 204, the NIST module-lattice signature standard),
producing an :class:`AuditChainSeal` that anyone holding the trusted public key
can verify long after the run — and against a future quantum adversary.

ML-DSA-65 (NIST security category 3) is the default; ML-DSA-44 and ML-DSA-87 are
available for lower/higher assurance. FIPS 205 (SLH-DSA / SPHINCS+) is the
hash-based alternative; the seal records its algorithm so a second scheme can be
added without breaking existing seals.

ML-DSA is provided by the optional ``cryptography`` dependency. Install the
``pqc`` extra (``pip install scpn-phase-orchestrator[pqc]``) to use this module;
``cryptography`` is imported lazily, so importing the module never requires it.
ML-DSA additionally needs an OpenSSL 3.5+ backend, which not every platform wheel
bundles (e.g. the Windows ``cryptography`` wheel); on an older backend the seal
functions raise ``cryptography.exceptions.UnsupportedAlgorithm``.

The seal is signed over a domain-separated message binding the algorithm, the
record count, and the tip hash, so a signature cannot be replayed across schemes
or truncated logs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VARIANT = "ml-dsa-65"
SEED_BYTES = 32
_DOMAIN = b"spo-audit-chain-seal-v1"
MLDSA_VARIANTS = ("ml-dsa-44", "ml-dsa-65", "ml-dsa-87")
_PQC_EXTRA_HINT = (
    "post-quantum audit sealing requires the 'cryptography' package; "
    "install scpn-phase-orchestrator[pqc]"
)

__all__ = [
    "DEFAULT_VARIANT",
    "MLDSA_VARIANTS",
    "AuditChainSeal",
    "generate_signing_seed",
    "public_key_id",
    "read_audit_chain_tip",
    "seal_audit_chain",
    "seal_audit_log",
    "sign_bytes",
    "signing_key_from_seed",
    "verify_audit_chain_seal",
    "verify_audit_log_seal",
    "verify_bytes",
]


def _load_mldsa() -> Any:
    """Return the ``cryptography`` ML-DSA module, or a helpful ImportError."""
    try:
        from cryptography.hazmat.primitives.asymmetric import mldsa
    except ImportError as exc:
        raise ImportError(_PQC_EXTRA_HINT) from exc
    return mldsa


def _require_variant(algorithm: str) -> str:
    """Return the validated post-quantum signature variant, else raise."""
    if algorithm not in MLDSA_VARIANTS:
        raise ValueError(
            f"algorithm must be one of {MLDSA_VARIANTS}, got {algorithm!r}"
        )
    return algorithm


def _private_class(algorithm: str) -> Any:
    """Return the post-quantum private-key class for the variant."""
    mldsa = _load_mldsa()
    return {
        "ml-dsa-44": mldsa.MLDSA44PrivateKey,
        "ml-dsa-65": mldsa.MLDSA65PrivateKey,
        "ml-dsa-87": mldsa.MLDSA87PrivateKey,
    }[algorithm]


def _public_class(algorithm: str) -> Any:
    """Return the post-quantum public-key class for the variant."""
    mldsa = _load_mldsa()
    return {
        "ml-dsa-44": mldsa.MLDSA44PublicKey,
        "ml-dsa-65": mldsa.MLDSA65PublicKey,
        "ml-dsa-87": mldsa.MLDSA87PublicKey,
    }[algorithm]


def generate_signing_seed() -> str:
    """Return a fresh ML-DSA signing seed.

    Returns
    -------
    str
        A cryptographically random 32-byte seed, hex-encoded.
    """
    return os.urandom(SEED_BYTES).hex()


def _validate_seed(seed_hex: object) -> bytes:
    """Return the validated key seed, else raise."""
    if not isinstance(seed_hex, str):
        raise ValueError("seed must be a hex string")
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise ValueError("seed must be valid hex") from exc
    if len(seed) != SEED_BYTES:
        raise ValueError(f"seed must decode to {SEED_BYTES} bytes, got {len(seed)}")
    return seed


def signing_key_from_seed(seed_hex: str, *, algorithm: str = DEFAULT_VARIANT) -> Any:
    """Return a deterministic ML-DSA private key derived from a seed.

    Parameters
    ----------
    seed_hex : str
        A 32-byte seed, hex-encoded (see :func:`generate_signing_seed`).
    algorithm : str
        The ML-DSA variant (one of :data:`MLDSA_VARIANTS`).

    Returns
    -------
    cryptography ML-DSA private key
        The deterministic private key for the seed and variant.

    Raises
    ------
    ValueError
        If the seed or algorithm is invalid.
    """
    algorithm = _require_variant(algorithm)
    seed = _validate_seed(seed_hex)
    return _private_class(algorithm).from_seed_bytes(seed)


def sign_bytes(
    message: bytes,
    private_key: Any,
    *,
    algorithm: str = DEFAULT_VARIANT,
) -> bytes:
    """Sign an arbitrary message with an ML-DSA private key.

    This is the generic post-quantum signing primitive that higher-level sealers
    (the audit-chain seal, the DSSE attestation envelope) build on, so the
    ``cryptography`` backend is loaded in exactly one place.

    Parameters
    ----------
    message : bytes
        The raw message to sign (already domain-separated by the caller).
    private_key : MLDSA private key
        An ML-DSA private key matching ``algorithm``.
    algorithm : str
        The ML-DSA variant; must match ``private_key``.

    Returns
    -------
    bytes
        The raw ML-DSA signature.

    Raises
    ------
    ValueError
        If the algorithm, message, or private key is invalid.
    """
    algorithm = _require_variant(algorithm)
    if not isinstance(message, (bytes, bytearray)):
        raise ValueError("message must be raw bytes")
    if not isinstance(private_key, _private_class(algorithm)):
        raise ValueError(f"private_key must be an {algorithm} private key")
    return bytes(private_key.sign(bytes(message)))


def verify_bytes(
    message: bytes,
    signature: bytes,
    trusted_public_key_hex: str,
    *,
    algorithm: str = DEFAULT_VARIANT,
) -> bool:
    """Verify an ML-DSA signature over a message against a trusted public key.

    Parameters
    ----------
    message : bytes
        The raw message the signature is expected to cover.
    signature : bytes
        The raw ML-DSA signature.
    trusted_public_key_hex : str
        The hex-encoded raw ML-DSA public key the verifier trusts.
    algorithm : str
        The ML-DSA variant to verify under.

    Returns
    -------
    bool
        ``True`` if the signature is valid for the trusted key, else ``False``.

    Raises
    ------
    ValueError
        If the algorithm is unknown, the message or signature is not raw bytes, or
        the trusted key is not a hex string.
    """
    algorithm = _require_variant(algorithm)
    if not isinstance(message, (bytes, bytearray)):
        raise ValueError("message must be raw bytes")
    if not isinstance(signature, (bytes, bytearray)):
        raise ValueError("signature must be raw bytes")
    if not isinstance(trusted_public_key_hex, str):
        raise ValueError("trusted_public_key_hex must be a hex string")
    try:
        trusted_bytes = bytes.fromhex(trusted_public_key_hex)
    except ValueError as exc:
        raise ValueError("trusted_public_key_hex must be valid hex") from exc
    from cryptography.exceptions import InvalidSignature

    try:
        public_key = _public_class(algorithm).from_public_bytes(trusted_bytes)
    except ValueError:
        return False
    try:
        public_key.verify(bytes(signature), bytes(message))
    except InvalidSignature:
        return False
    return True


def public_key_id(public_bytes: bytes) -> str:
    """Return the short identifier of a public key.

    Parameters
    ----------
    public_bytes : bytes
        The raw ML-DSA public key bytes.

    Returns
    -------
    str
        The first 16 hex characters of the key's SHA-256 digest.

    Raises
    ------
    ValueError
        If ``public_bytes`` is not raw bytes.
    """
    if not isinstance(public_bytes, (bytes, bytearray)):
        raise ValueError("public_bytes must be raw bytes")
    return hashlib.sha256(bytes(public_bytes)).hexdigest()[:16]


def _signing_message(algorithm: str, record_count: int, tip_hash: str) -> bytes:
    """Return the domain-separated message bound by a seal signature."""
    body = json.dumps(
        {"algorithm": algorithm, "record_count": record_count, "tip_hash": tip_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _DOMAIN + b"\n" + body.encode("utf-8")


def _validate_tip_hash(tip_hash: object) -> str:
    """Return the validated audit-chain tip hash, else raise."""
    if not isinstance(tip_hash, str) or not tip_hash:
        raise ValueError("tip_hash must be a non-empty hex string")
    try:
        decoded = bytes.fromhex(tip_hash)
    except ValueError as exc:
        raise ValueError("tip_hash must be valid hex") from exc
    if len(decoded) != 32:
        raise ValueError("tip_hash must be a 32-byte SHA-256 digest")
    return tip_hash


def _validate_record_count(record_count: object) -> int:
    """Return the validated audit record count, else raise."""
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError("record_count must be a non-negative integer")
    if record_count < 0:
        raise ValueError("record_count must be a non-negative integer")
    return record_count


@dataclass(frozen=True)
class AuditChainSeal:
    """A post-quantum signature committing to an audit hash chain.

    Attributes
    ----------
    algorithm : str
        The ML-DSA variant used (one of :data:`MLDSA_VARIANTS`).
    public_key_id : str
        Short identifier of the public key (SHA-256 prefix).
    public_key_hex : str
        The raw ML-DSA public key, hex-encoded, for verification.
    tip_hash : str
        The SHA-256 chain tip (``_hash`` of the last record) that is sealed.
    record_count : int
        Number of records the chain contained when sealed.
    signature_hex : str
        The ML-DSA signature over the domain-separated seal message, hex-encoded.
    """

    algorithm: str
    public_key_id: str
    public_key_hex: str
    tip_hash: str
    record_count: int
    signature_hex: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serialisable mapping of the seal.

        Returns
        -------
        dict[str, str | int]
            The six seal fields as plain JSON-serialisable values.
        """
        return {
            "algorithm": self.algorithm,
            "public_key_id": self.public_key_id,
            "public_key_hex": self.public_key_hex,
            "tip_hash": self.tip_hash,
            "record_count": self.record_count,
            "signature_hex": self.signature_hex,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditChainSeal:
        """Return a seal parsed from a mapping (e.g. loaded JSON).

        Parameters
        ----------
        data : dict[str, Any]
            A mapping carrying the six seal fields.

        Returns
        -------
        AuditChainSeal
            The reconstructed seal.

        Raises
        ------
        ValueError
            If any required field is missing.
        """
        required = {
            "algorithm",
            "public_key_id",
            "public_key_hex",
            "tip_hash",
            "record_count",
            "signature_hex",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"seal is missing fields: {sorted(missing)}")
        return cls(
            algorithm=str(data["algorithm"]),
            public_key_id=str(data["public_key_id"]),
            public_key_hex=str(data["public_key_hex"]),
            tip_hash=str(data["tip_hash"]),
            record_count=int(data["record_count"]),
            signature_hex=str(data["signature_hex"]),
        )


def seal_audit_chain(
    tip_hash: str,
    record_count: int,
    private_key: Any,
    *,
    algorithm: str = DEFAULT_VARIANT,
) -> AuditChainSeal:
    """Sign an audit chain tip with ML-DSA and return the seal.

    Parameters
    ----------
    tip_hash : str
        The SHA-256 chain tip (32-byte digest, hex).
    record_count : int
        Number of records in the sealed chain.
    private_key : MLDSA private key
        An ML-DSA private key matching ``algorithm`` (see
        :func:`signing_key_from_seed`).
    algorithm : str
        The ML-DSA variant; must match ``private_key``.

    Returns
    -------
    AuditChainSeal
        The post-quantum seal over the chain tip.

    Raises
    ------
    ValueError
        If the algorithm, tip hash, record count, or private key is invalid.
    """
    algorithm = _require_variant(algorithm)
    tip_hash = _validate_tip_hash(tip_hash)
    record_count = _validate_record_count(record_count)
    if not isinstance(private_key, _private_class(algorithm)):
        raise ValueError(f"private_key must be an {algorithm} private key")
    public_bytes = private_key.public_key().public_bytes_raw()
    message = _signing_message(algorithm, record_count, tip_hash)
    signature = private_key.sign(message)
    return AuditChainSeal(
        algorithm=algorithm,
        public_key_id=public_key_id(public_bytes),
        public_key_hex=public_bytes.hex(),
        tip_hash=tip_hash,
        record_count=record_count,
        signature_hex=signature.hex(),
    )


def verify_audit_chain_seal(seal: AuditChainSeal, trusted_public_key_hex: str) -> bool:
    """Verify a seal's signature against a trusted public key.

    The verifier must supply the public key it trusts; the key embedded in the
    seal is only used as a convenience and is checked to match the trusted one,
    so an attacker cannot re-sign a forged log under their own key.

    Parameters
    ----------
    seal : AuditChainSeal
        The seal to verify.
    trusted_public_key_hex : str
        The hex-encoded raw ML-DSA public key the verifier trusts.

    Returns
    -------
    bool
        ``True`` if the signature is valid for the trusted key, else ``False``.

    Raises
    ------
    ValueError
        If the seal's algorithm is unknown or the trusted key is malformed.
    """
    algorithm = _require_variant(seal.algorithm)
    if not isinstance(trusted_public_key_hex, str):
        raise ValueError("trusted_public_key_hex must be a hex string")
    try:
        trusted_bytes = bytes.fromhex(trusted_public_key_hex)
    except ValueError as exc:
        raise ValueError("trusted_public_key_hex must be valid hex") from exc
    if public_key_id(trusted_bytes) != seal.public_key_id:
        return False
    if trusted_bytes.hex() != seal.public_key_hex:
        return False
    from cryptography.exceptions import InvalidSignature

    try:
        public_key = _public_class(algorithm).from_public_bytes(trusted_bytes)
        signature = bytes.fromhex(seal.signature_hex)
    except ValueError:
        return False
    message = _signing_message(algorithm, seal.record_count, seal.tip_hash)
    try:
        public_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def read_audit_chain_tip(path: Path) -> tuple[str, int]:
    """Return the chain tip ``_hash`` and record count of an audit JSONL file.

    Parameters
    ----------
    path : Path
        Path to the audit JSONL stream.

    Returns
    -------
    tuple[str, int]
        ``(tip_hash, record_count)``.

    Raises
    ------
    ValueError
        If the file is empty or the last record carries no ``_hash``.
    FileNotFoundError
        If the file does not exist.
    """
    text = Path(path).read_text(encoding="utf-8")
    records = [line for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("audit stream is empty; nothing to seal")
    try:
        last = json.loads(records[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("audit stream tail is not valid JSON") from exc
    tip_hash = last.get("_hash")
    if not isinstance(tip_hash, str) or not tip_hash:
        raise ValueError("audit stream tail carries no '_hash' chain tip")
    return _validate_tip_hash(tip_hash), len(records)


def seal_audit_log(
    path: Path,
    private_key: Any,
    *,
    algorithm: str = DEFAULT_VARIANT,
) -> AuditChainSeal:
    """Read an audit JSONL file's chain tip and return a post-quantum seal.

    Parameters
    ----------
    path : Path
        Path to the audit JSONL stream to seal.
    private_key : cryptography ML-DSA private key
        The signing key (see :func:`signing_key_from_seed`).
    algorithm : str
        The ML-DSA variant; must match ``private_key``.

    Returns
    -------
    AuditChainSeal
        The post-quantum seal over the file's chain tip.
    """
    tip_hash, record_count = read_audit_chain_tip(path)
    return seal_audit_chain(tip_hash, record_count, private_key, algorithm=algorithm)


def verify_audit_log_seal(
    path: Path,
    seal: AuditChainSeal,
    trusted_public_key_hex: str,
) -> bool:
    """Verify a seal against an audit file and a trusted public key.

    Parameters
    ----------
    path : Path
        Path to the audit JSONL stream to check.
    seal : AuditChainSeal
        The seal previously produced for this log.
    trusted_public_key_hex : str
        The hex-encoded raw ML-DSA public key the verifier trusts.

    Returns
    -------
    bool
        ``True`` only if the file's current chain tip and record count match the
        seal *and* the signature verifies under the trusted key, so the seal is
        rejected if the log was altered after sealing.
    """
    tip_hash, record_count = read_audit_chain_tip(path)
    if tip_hash != seal.tip_hash or record_count != seal.record_count:
        return False
    return verify_audit_chain_seal(seal, trusted_public_key_hex)
