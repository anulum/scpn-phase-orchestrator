# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — DSSE attestation envelope tests

"""Coverage for the DSSE attestation envelope.

The wire-format paths (PAE encoding, base64 handling, dataclass validation, and
round-tripping) carry no crypto and run everywhere. The sign/verify paths need an
ML-DSA backend and are guarded by ``requires_mldsa``: they check a good signature
verifies, a wrong key is rejected, and a tampered payload is rejected because the
signature binds the payload's pre-authentication encoding.
"""

from __future__ import annotations

import base64
import json

import pytest

from scpn_phase_orchestrator.assurance import dsse
from scpn_phase_orchestrator.assurance.dsse import (
    DSSE_PAYLOAD_TYPE,
    DsseEnvelope,
    DsseSignature,
    sign_provenance_statement,
    verify_dsse_envelope,
)
from scpn_phase_orchestrator.assurance.provenance import (
    ArtifactSubject,
    BuildDefinition,
    RunDetails,
    build_slsa_provenance_statement,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    DEFAULT_VARIANT,
    generate_signing_seed,
    signing_key_from_seed,
)
from tests.test_audit_pqc import requires_mldsa

_A = "a" * 64
_SIG_B64 = base64.b64encode(b"signature-bytes").decode("ascii")


def _statement() -> object:
    return build_slsa_provenance_statement(
        (ArtifactSubject(name="wheel", sha256=_A),),
        BuildDefinition(
            build_type="https://slsa.dev/build/pypi@v1",
            external_parameters={"ref": "refs/tags/v1.0"},
        ),
        RunDetails(builder_id="https://ci", invocation_id="run-1"),
    )


def _signature(**overrides: str) -> DsseSignature:
    params = {
        "keyid": "abc123",
        "algorithm": DEFAULT_VARIANT,
        "signature_b64": _SIG_B64,
    }
    params.update(overrides)
    return DsseSignature(**params)


def _envelope(**overrides: object) -> DsseEnvelope:
    params: dict[str, object] = {
        "payload_b64": base64.b64encode(b'{"ok":true}').decode("ascii"),
        "payload_type": DSSE_PAYLOAD_TYPE,
        "signatures": (_signature(),),
    }
    params.update(overrides)
    return DsseEnvelope(**params)  # type: ignore[arg-type]


# --- PAE ------------------------------------------------------------------


def test_pae_matches_dsse_specification_vector() -> None:
    # The canonical DSSE spec example.
    assert dsse._pae("http://example.com/HelloWorld", b"hello world") == (
        b"DSSEv1 29 http://example.com/HelloWorld 11 hello world"
    )


def test_require_algorithm_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="algorithm must be one of"):
        dsse._require_algorithm("rsa-2048")


# --- base64 helpers -------------------------------------------------------


class TestBase64:
    def test_roundtrip(self) -> None:
        assert dsse._decode_b64(dsse._encode_b64(b"payload"), "x") == b"payload"

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="x must be a base64 string"):
            dsse._decode_b64(123, "x")  # type: ignore[arg-type]

    def test_rejects_invalid_base64(self) -> None:
        with pytest.raises(ValueError, match="x must be valid base64"):
            dsse._decode_b64("not!valid!base64!", "x")


# --- DsseSignature --------------------------------------------------------


class TestDsseSignature:
    def test_to_dict(self) -> None:
        assert _signature().to_dict() == {
            "keyid": "abc123",
            "algorithm": DEFAULT_VARIANT,
            "sig": _SIG_B64,
        }

    def test_rejects_bad_algorithm(self) -> None:
        with pytest.raises(ValueError, match="algorithm must be one of"):
            _signature(algorithm="rsa")

    def test_rejects_empty_keyid(self) -> None:
        with pytest.raises(ValueError, match="keyid must be a non-empty string"):
            _signature(keyid="")

    def test_rejects_bad_signature_base64(self) -> None:
        with pytest.raises(ValueError, match="signature_b64 must be valid base64"):
            _signature(signature_b64="!!!")

    def test_from_dict_roundtrip(self) -> None:
        assert DsseSignature.from_dict(_signature().to_dict()) == _signature()

    def test_from_dict_rejects_missing_field(self) -> None:
        with pytest.raises(ValueError, match="signature is missing field: sig"):
            DsseSignature.from_dict({"keyid": "a", "algorithm": DEFAULT_VARIANT})


# --- DsseEnvelope ---------------------------------------------------------


class TestDsseEnvelope:
    def test_rejects_bad_payload_base64(self) -> None:
        with pytest.raises(ValueError, match="payload must be valid base64"):
            _envelope(payload_b64="!!!")

    def test_rejects_empty_payload_type(self) -> None:
        with pytest.raises(ValueError, match="payloadType must be a non-empty string"):
            _envelope(payload_type="")

    def test_rejects_no_signatures(self) -> None:
        with pytest.raises(ValueError, match="requires at least one signature"):
            _envelope(signatures=())

    def test_payload_bytes_and_statement(self) -> None:
        payload = base64.b64encode(b'{"key": "value"}').decode("ascii")
        envelope = _envelope(payload_b64=payload)
        assert envelope.payload_bytes() == b'{"key": "value"}'
        assert envelope.statement() == {"key": "value"}

    def test_statement_rejects_non_json_payload(self) -> None:
        envelope = _envelope(payload_b64=base64.b64encode(b"not json").decode("ascii"))
        with pytest.raises(ValueError, match="payload is not valid JSON"):
            envelope.statement()

    def test_statement_rejects_non_object_payload(self) -> None:
        envelope = _envelope(payload_b64=base64.b64encode(b"[1, 2]").decode("ascii"))
        with pytest.raises(ValueError, match="payload is not a JSON object"):
            envelope.statement()

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        envelope = _envelope()
        restored = DsseEnvelope.from_dict(envelope.to_dict())
        assert restored == envelope

    def test_from_dict_rejects_missing_field(self) -> None:
        with pytest.raises(ValueError, match="envelope is missing field: signatures"):
            DsseEnvelope.from_dict(
                {"payload": _envelope().payload_b64, "payloadType": DSSE_PAYLOAD_TYPE}
            )

    def test_from_dict_rejects_non_list_signatures(self) -> None:
        with pytest.raises(ValueError, match="signatures must be a list"):
            DsseEnvelope.from_dict(
                {
                    "payload": _envelope().payload_b64,
                    "payloadType": DSSE_PAYLOAD_TYPE,
                    "signatures": {"not": "a list"},
                }
            )

    def test_from_dict_rejects_non_object_signature_entry(self) -> None:
        with pytest.raises(ValueError, match="signature entry must be an object"):
            DsseEnvelope.from_dict(
                {
                    "payload": _envelope().payload_b64,
                    "payloadType": DSSE_PAYLOAD_TYPE,
                    "signatures": ["not-an-object"],
                }
            )


# --- sign / verify (crypto) -----------------------------------------------


@requires_mldsa
class TestSignVerify:
    def _key_and_pub(self) -> tuple[object, str]:
        key = signing_key_from_seed(generate_signing_seed())
        return key, key.public_key().public_bytes_raw().hex()

    def test_sign_then_verify_roundtrip(self) -> None:
        key, pub = self._key_and_pub()
        envelope = sign_provenance_statement(_statement(), key)  # type: ignore[arg-type]
        assert verify_dsse_envelope(envelope, pub) is True
        # The wrapped statement survives the round-trip.
        assert envelope.statement()["predicateType"].startswith("https://slsa.dev")

    def test_sign_rejects_unknown_algorithm(self) -> None:
        key, _pub = self._key_and_pub()
        with pytest.raises(ValueError, match="algorithm must be one of"):
            sign_provenance_statement(_statement(), key, algorithm="rsa")  # type: ignore[arg-type]

    def test_verify_rejects_wrong_key(self) -> None:
        key, _pub = self._key_and_pub()
        envelope = sign_provenance_statement(_statement(), key)  # type: ignore[arg-type]
        other = signing_key_from_seed(generate_signing_seed())
        other_pub = other.public_key().public_bytes_raw().hex()
        assert verify_dsse_envelope(envelope, other_pub) is False

    def test_verify_rejects_tampered_payload(self) -> None:
        key, pub = self._key_and_pub()
        envelope = sign_provenance_statement(_statement(), key)  # type: ignore[arg-type]
        tampered_body = json.dumps({"_type": "evil"}, sort_keys=True).encode("utf-8")
        tampered = DsseEnvelope(
            payload_b64=base64.b64encode(tampered_body).decode("ascii"),
            payload_type=envelope.payload_type,
            signatures=envelope.signatures,
        )
        assert verify_dsse_envelope(tampered, pub) is False


class TestVerifyKeyValidation:
    def test_verify_rejects_non_string_key(self) -> None:
        with pytest.raises(ValueError, match="trusted_public_key_hex must be a hex"):
            verify_dsse_envelope(_envelope(), 123)  # type: ignore[arg-type]

    def test_verify_rejects_non_hex_key(self) -> None:
        with pytest.raises(ValueError, match="must be valid hex"):
            verify_dsse_envelope(_envelope(), "zz")

    def test_verify_returns_false_when_no_keyid_matches(self) -> None:
        # A valid-hex key whose short id does not match the envelope's signature
        # keyid short-circuits to ``False`` without touching the crypto backend.
        assert verify_dsse_envelope(_envelope(), "ab" * 32) is False
