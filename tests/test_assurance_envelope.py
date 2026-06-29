# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — signed, replay-anchored certification envelope tests

"""Tests for the signed, replay-anchored certification envelope.

``build_signed_certification_envelope`` binds a certification package hash to a
run's audit-chain tip and, optionally, a post-quantum seal over that tip;
``verify_signed_certification_envelope`` re-derives the envelope hash, checks the
package binding, and verifies any attached seal against a trusted key. Most branches
are exercised without a cryptography backend (the seal object is a plain dataclass);
only the genuine end-to-end ML-DSA verification needs an OpenSSL 3.5+ backend.
"""

from __future__ import annotations

from typing import Any

import pytest

import scpn_phase_orchestrator.assurance.envelope as _envelope
from scpn_phase_orchestrator.assurance import (
    SIGNED_CERTIFICATION_ENVELOPE_SCHEMA,
    SignedCertificationEnvelope,
    build_signed_certification_envelope,
    verify_signed_certification_envelope,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    AuditChainSeal,
    generate_signing_seed,
    seal_audit_chain,
    signing_key_from_seed,
)

assert _envelope is not None

_PACKAGE_HASH = "ab" * 32
_TIP = "cd" * 32
_OTHER_TIP = "ef" * 32
_COUNT = 7


def _mldsa_supported() -> bool:
    """Return whether the platform cryptography backend implements ML-DSA."""
    try:
        from cryptography.exceptions import UnsupportedAlgorithm
        from cryptography.hazmat.primitives.asymmetric import mldsa
    except ImportError:
        return False
    try:
        mldsa.MLDSA65PrivateKey.generate()
    except UnsupportedAlgorithm:
        return False
    return True


requires_mldsa = pytest.mark.skipif(
    not _mldsa_supported(),
    reason="ML-DSA requires an OpenSSL 3.5+ cryptography backend",
)


def _fake_seal(tip: str = _TIP, count: int = _COUNT) -> AuditChainSeal:
    """Return a structurally valid seal with dummy key material (no signing)."""
    return AuditChainSeal(
        algorithm="ml-dsa-65",
        public_key_id="0" * 16,
        public_key_hex="11" * 32,
        tip_hash=tip,
        record_count=count,
        signature_hex="22" * 16,
    )


def test_anchor_only_envelope_fields() -> None:
    envelope = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)

    assert envelope.package_hash == _PACKAGE_HASH
    assert envelope.audit_chain_tip == _TIP
    assert envelope.audit_record_count == _COUNT
    assert envelope.seal is None
    record = envelope.to_record()
    assert record["schema"] == SIGNED_CERTIFICATION_ENVELOPE_SCHEMA
    assert record["seal"] is None
    assert record["envelope_hash"] == envelope.envelope_hash


def test_envelope_hash_is_deterministic() -> None:
    first = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)
    second = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)

    assert first.envelope_hash == second.envelope_hash


def test_seal_is_recorded_in_the_envelope() -> None:
    envelope = build_signed_certification_envelope(
        _PACKAGE_HASH, _TIP, _COUNT, seal=_fake_seal()
    )

    assert envelope.seal is not None
    assert envelope.seal["tip_hash"] == _TIP
    assert envelope.to_record()["seal"]["signature_hex"] == "22" * 16  # type: ignore[index]


def test_build_rejects_seal_with_mismatched_tip() -> None:
    with pytest.raises(ValueError, match="seal tip_hash does not match"):
        build_signed_certification_envelope(
            _PACKAGE_HASH, _TIP, _COUNT, seal=_fake_seal(tip=_OTHER_TIP)
        )


def test_build_rejects_seal_with_mismatched_record_count() -> None:
    with pytest.raises(ValueError, match="seal record_count does not match"):
        build_signed_certification_envelope(
            _PACKAGE_HASH, _TIP, _COUNT, seal=_fake_seal(count=_COUNT + 1)
        )


def test_build_rejects_bad_package_hash() -> None:
    with pytest.raises(ValueError, match="package_hash"):
        build_signed_certification_envelope("not-a-hash", _TIP, _COUNT)


def test_build_rejects_bad_tip() -> None:
    with pytest.raises(ValueError, match="audit_chain_tip"):
        build_signed_certification_envelope(_PACKAGE_HASH, "short", _COUNT)


@pytest.mark.parametrize("count", [True, 1.5, -1])
def test_build_rejects_bad_record_count(count: Any) -> None:
    with pytest.raises(ValueError, match="audit_record_count"):
        build_signed_certification_envelope(_PACKAGE_HASH, _TIP, count)


def test_explicit_matching_envelope_hash_is_accepted() -> None:
    built = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)
    rebuilt = SignedCertificationEnvelope(
        package_hash=_PACKAGE_HASH,
        audit_chain_tip=_TIP,
        audit_record_count=_COUNT,
        seal=None,
        envelope_hash=built.envelope_hash,
    )

    assert rebuilt.envelope_hash == built.envelope_hash


def test_explicit_mismatching_envelope_hash_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match the canonical hash"):
        SignedCertificationEnvelope(
            package_hash=_PACKAGE_HASH,
            audit_chain_tip=_TIP,
            audit_record_count=_COUNT,
            seal=None,
            envelope_hash="00" * 32,
        )


def test_verify_anchor_only_succeeds_for_matching_package() -> None:
    envelope = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)

    assert verify_signed_certification_envelope(envelope, package_hash=_PACKAGE_HASH)


def test_verify_rejects_wrong_package_hash() -> None:
    envelope = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)

    assert not verify_signed_certification_envelope(envelope, package_hash="ba" * 32)


def test_verify_rejects_tampered_envelope_hash() -> None:
    envelope = build_signed_certification_envelope(_PACKAGE_HASH, _TIP, _COUNT)
    other = build_signed_certification_envelope(_PACKAGE_HASH, _OTHER_TIP, _COUNT)
    object.__setattr__(envelope, "envelope_hash", other.envelope_hash)

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH
    )


def test_verify_signed_envelope_without_trusted_key_is_rejected() -> None:
    envelope = build_signed_certification_envelope(
        _PACKAGE_HASH, _TIP, _COUNT, seal=_fake_seal()
    )

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH
    )


def test_verify_rejects_seal_whose_tip_drifts_from_anchor() -> None:
    # An envelope whose seal commits to a different tip than the anchor — only
    # constructible by bypassing the builder — must be rejected by verify.
    envelope = SignedCertificationEnvelope(
        package_hash=_PACKAGE_HASH,
        audit_chain_tip=_TIP,
        audit_record_count=_COUNT,
        seal=_fake_seal(tip=_OTHER_TIP).to_dict(),
    )

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH, trusted_public_key_hex="11" * 32
    )


def test_verify_rejects_seal_whose_count_drifts_from_anchor() -> None:
    envelope = SignedCertificationEnvelope(
        package_hash=_PACKAGE_HASH,
        audit_chain_tip=_TIP,
        audit_record_count=_COUNT,
        seal=_fake_seal(count=_COUNT + 3).to_dict(),
    )

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH, trusted_public_key_hex="11" * 32
    )


def test_verify_rejects_seal_under_a_non_matching_key() -> None:
    # A fake seal verified against a key whose id does not match is rejected
    # without needing a cryptography backend.
    envelope = build_signed_certification_envelope(
        _PACKAGE_HASH, _TIP, _COUNT, seal=_fake_seal()
    )

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH, trusted_public_key_hex="33" * 32
    )


@requires_mldsa
def test_signed_envelope_round_trips_under_a_trusted_key() -> None:
    seed = generate_signing_seed()
    key = signing_key_from_seed(seed)
    seal = seal_audit_chain(_TIP, _COUNT, key)
    public_hex = key.public_key().public_bytes_raw().hex()
    envelope = build_signed_certification_envelope(
        _PACKAGE_HASH, _TIP, _COUNT, seal=seal
    )

    assert verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH, trusted_public_key_hex=public_hex
    )


@requires_mldsa
def test_signed_envelope_is_rejected_under_a_foreign_key() -> None:
    key = signing_key_from_seed(generate_signing_seed())
    foreign = signing_key_from_seed(generate_signing_seed())
    seal = seal_audit_chain(_TIP, _COUNT, key)
    foreign_hex = foreign.public_key().public_bytes_raw().hex()
    envelope = build_signed_certification_envelope(
        _PACKAGE_HASH, _TIP, _COUNT, seal=seal
    )

    assert not verify_signed_certification_envelope(
        envelope, package_hash=_PACKAGE_HASH, trusted_public_key_hex=foreign_hex
    )
