# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Post-quantum audit-chain seal tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.runtime import audit_pqc
from scpn_phase_orchestrator.runtime.audit_pqc import (
    DEFAULT_VARIANT,
    MLDSA_VARIANTS,
    AuditChainSeal,
    generate_signing_seed,
    public_key_id,
    read_audit_chain_tip,
    seal_audit_chain,
    seal_audit_log,
    sign_bytes,
    signing_key_from_seed,
    verify_audit_chain_seal,
    verify_audit_log_seal,
    verify_bytes,
)

_SEED = "01" * 32
_TIP = "ab" * 32  # a valid 32-byte hex digest


def _key(algorithm: str = DEFAULT_VARIANT, seed: str = _SEED) -> Any:
    return signing_key_from_seed(seed, algorithm=algorithm)


def _pub_hex(key: Any) -> str:
    return key.public_key().public_bytes_raw().hex()


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


# ML-DSA needs an OpenSSL 3.5+ backend, which not every platform wheel bundles
# (e.g. the Windows cryptography wheel); key-using tests skip where unsupported.
requires_mldsa = pytest.mark.skipif(
    not _mldsa_supported(),
    reason="ML-DSA requires an OpenSSL 3.5+ cryptography backend",
)


class TestOptionalDependency:
    def test_missing_cryptography_raises_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            # Simulate the ML-DSA module being absent regardless of prior imports.
            if name.startswith("cryptography.hazmat.primitives.asymmetric") and (
                "mldsa" in (fromlist or ())
            ):
                raise ImportError("simulated missing cryptography mldsa")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match=r"scpn-phase-orchestrator\[pqc\]"):
            audit_pqc._load_mldsa()


class TestSeed:
    def test_generate_signing_seed_is_32_byte_hex(self) -> None:
        seed = generate_signing_seed()
        assert len(bytes.fromhex(seed)) == 32

    def test_generate_signing_seed_is_random(self) -> None:
        assert generate_signing_seed() != generate_signing_seed()

    @requires_mldsa
    def test_key_from_seed_is_deterministic(self) -> None:
        assert _pub_hex(_key()) == _pub_hex(_key())

    @pytest.mark.parametrize("seed", ["", "zz" * 32, "ab" * 16, 1234, "ab" * 40])
    def test_rejects_bad_seed(self, seed: Any) -> None:
        with pytest.raises(ValueError, match="seed"):
            signing_key_from_seed(seed)

    def test_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="algorithm"):
            signing_key_from_seed(_SEED, algorithm="ml-dsa-128")


@requires_mldsa
class TestSealAndVerify:
    def test_round_trip(self) -> None:
        key = _key()
        seal = seal_audit_chain(_TIP, 5, key)
        assert seal.algorithm == DEFAULT_VARIANT
        assert seal.tip_hash == _TIP
        assert seal.record_count == 5
        assert verify_audit_chain_seal(seal, _pub_hex(key)) is True

    @pytest.mark.parametrize("algorithm", list(MLDSA_VARIANTS))
    def test_all_variants_round_trip(self, algorithm: str) -> None:
        key = _key(algorithm)
        seal = seal_audit_chain(_TIP, 3, key, algorithm=algorithm)
        assert verify_audit_chain_seal(seal, _pub_hex(key)) is True

    def test_wrong_public_key_fails(self) -> None:
        seal = seal_audit_chain(_TIP, 5, _key())
        other = _key(seed="02" * 32)
        assert verify_audit_chain_seal(seal, _pub_hex(other)) is False

    def test_tampered_tip_fails(self) -> None:
        key = _key()
        seal = seal_audit_chain(_TIP, 5, key)
        forged = AuditChainSeal(
            algorithm=seal.algorithm,
            public_key_id=seal.public_key_id,
            public_key_hex=seal.public_key_hex,
            tip_hash="cd" * 32,
            record_count=seal.record_count,
            signature_hex=seal.signature_hex,
        )
        assert verify_audit_chain_seal(forged, _pub_hex(key)) is False

    def test_tampered_record_count_fails(self) -> None:
        key = _key()
        seal = seal_audit_chain(_TIP, 5, key)
        forged = AuditChainSeal(
            algorithm=seal.algorithm,
            public_key_id=seal.public_key_id,
            public_key_hex=seal.public_key_hex,
            tip_hash=seal.tip_hash,
            record_count=6,
            signature_hex=seal.signature_hex,
        )
        assert verify_audit_chain_seal(forged, _pub_hex(key)) is False

    def test_mismatched_key_id_fails(self) -> None:
        seal = seal_audit_chain(_TIP, 5, _key())
        other_pub = _pub_hex(_key(seed="03" * 32))
        assert verify_audit_chain_seal(seal, other_pub) is False

    def test_malformed_trusted_key_rejected(self) -> None:
        seal = seal_audit_chain(_TIP, 5, _key())
        with pytest.raises(ValueError, match="hex"):
            verify_audit_chain_seal(seal, "not-hex!!")

    def test_unknown_algorithm_in_seal_rejected(self) -> None:
        seal = seal_audit_chain(_TIP, 5, _key())
        broken = AuditChainSeal(
            algorithm="ml-dsa-128",
            public_key_id=seal.public_key_id,
            public_key_hex=seal.public_key_hex,
            tip_hash=seal.tip_hash,
            record_count=seal.record_count,
            signature_hex=seal.signature_hex,
        )
        with pytest.raises(ValueError, match="algorithm"):
            verify_audit_chain_seal(broken, seal.public_key_hex)

    def test_non_string_trusted_key_rejected(self) -> None:
        seal = seal_audit_chain(_TIP, 5, _key())
        with pytest.raises(ValueError, match="hex string"):
            verify_audit_chain_seal(seal, 12345)  # type: ignore[arg-type]

    def test_matching_key_id_but_different_key_fails(self) -> None:
        key = _key()
        pub = _pub_hex(key)
        forged = AuditChainSeal(
            algorithm=DEFAULT_VARIANT,
            public_key_id=public_key_id(bytes.fromhex(pub)),
            public_key_hex=_pub_hex(_key(seed="07" * 32)),
            tip_hash=_TIP,
            record_count=1,
            signature_hex="00",
        )
        assert verify_audit_chain_seal(forged, pub) is False

    def test_invalid_signature_hex_fails(self) -> None:
        key = _key()
        seal = seal_audit_chain(_TIP, 5, key)
        broken = AuditChainSeal(
            algorithm=seal.algorithm,
            public_key_id=seal.public_key_id,
            public_key_hex=seal.public_key_hex,
            tip_hash=seal.tip_hash,
            record_count=seal.record_count,
            signature_hex="zz",
        )
        assert verify_audit_chain_seal(broken, _pub_hex(key)) is False


@requires_mldsa
class TestSealValidation:
    @pytest.mark.parametrize("tip", ["", "zz" * 32, "ab" * 16, 123])
    def test_rejects_bad_tip(self, tip: Any) -> None:
        with pytest.raises(ValueError, match="tip_hash"):
            seal_audit_chain(tip, 5, _key())

    @pytest.mark.parametrize("count", [-1, True, 1.5, "5"])
    def test_rejects_bad_record_count(self, count: Any) -> None:
        with pytest.raises(ValueError, match="record_count"):
            seal_audit_chain(_TIP, count, _key())

    def test_rejects_wrong_key_type(self) -> None:
        with pytest.raises(ValueError, match="private_key"):
            seal_audit_chain(_TIP, 5, object())

    def test_public_key_id_rejects_non_bytes(self) -> None:
        with pytest.raises(ValueError, match="public_bytes"):
            public_key_id("not-bytes")


@requires_mldsa
class TestSerialisation:
    def test_to_from_dict_round_trip(self) -> None:
        seal = seal_audit_chain(_TIP, 7, _key())
        restored = AuditChainSeal.from_dict(json.loads(json.dumps(seal.to_dict())))
        assert restored == seal

    def test_from_dict_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="missing fields"):
            AuditChainSeal.from_dict({"algorithm": "ml-dsa-65"})


class TestChainTipReading:
    def _write_chain(self, path: Path, tips: list[str]) -> None:
        path.write_text(
            "\n".join(json.dumps({"event": i, "_hash": h}) for i, h in enumerate(tips))
            + "\n",
            encoding="utf-8",
        )

    def test_reads_tip_and_count(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        self._write_chain(log, ["aa" * 32, "bb" * 32, _TIP])
        tip, count = read_audit_chain_tip(log)
        assert tip == _TIP
        assert count == 3

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        log.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            read_audit_chain_tip(log)

    def test_missing_hash_rejected(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        log.write_text(json.dumps({"event": 0}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="_hash"):
            read_audit_chain_tip(log)

    def test_bad_json_tail_rejected(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        log.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON"):
            read_audit_chain_tip(log)


@requires_mldsa
class TestSealAuditFile:
    def _write_chain(self, path: Path, tips: list[str]) -> None:
        path.write_text(
            "\n".join(json.dumps({"event": i, "_hash": h}) for i, h in enumerate(tips))
            + "\n",
            encoding="utf-8",
        )

    def test_seal_and_verify_file(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        self._write_chain(log, ["aa" * 32, _TIP])
        key = _key()
        seal = seal_audit_log(log, key)
        assert seal.tip_hash == _TIP
        assert seal.record_count == 2
        assert verify_audit_log_seal(log, seal, _pub_hex(key)) is True

    def test_appended_record_breaks_seal(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        self._write_chain(log, ["aa" * 32, _TIP])
        key = _key()
        seal = seal_audit_log(log, key)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": 99, "_hash": "ff" * 32}) + "\n")
        assert verify_audit_log_seal(log, seal, _pub_hex(key)) is False

    def test_modified_tip_breaks_seal(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        self._write_chain(log, ["aa" * 32, _TIP])
        key = _key()
        seal = seal_audit_log(log, key)
        self._write_chain(log, ["aa" * 32, "cd" * 32])
        assert verify_audit_log_seal(log, seal, _pub_hex(key)) is False


@requires_mldsa
class TestPipelineWiring:
    def test_seals_a_real_audit_logger_chain(self, tmp_path: Path) -> None:
        """Seal and verify a chain written by the real AuditLogger."""
        from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger

        log = tmp_path / "run.jsonl"
        with AuditLogger(log) as logger:
            logger.log_event("regime_change", {"step": 1, "to": "NOMINAL"})
            logger.log_event("regime_change", {"step": 2, "to": "DEGRADED"})

        seed = generate_signing_seed()
        key = signing_key_from_seed(seed)
        seal = seal_audit_log(log, key)
        assert seal.record_count >= 2
        assert verify_audit_log_seal(log, seal, _pub_hex(key)) is True

        # Any post-hoc edit to the sealed log is detected.
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"injected": True, "_hash": "00" * 32}) + "\n")
        assert verify_audit_log_seal(log, seal, _pub_hex(key)) is False


class TestSignBytesValidation:
    """Validation guards that run without an ML-DSA backend."""

    def test_sign_bytes_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="algorithm must be one of"):
            sign_bytes(b"m", object(), algorithm="rsa")

    def test_sign_bytes_rejects_non_bytes_message(self) -> None:
        with pytest.raises(ValueError, match="message must be raw bytes"):
            sign_bytes("not-bytes", object())  # type: ignore[arg-type]

    def test_verify_bytes_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="algorithm must be one of"):
            verify_bytes(b"m", b"s", "ab" * 32, algorithm="rsa")

    def test_verify_bytes_rejects_non_bytes_message(self) -> None:
        with pytest.raises(ValueError, match="message must be raw bytes"):
            verify_bytes("m", b"s", "ab" * 32)  # type: ignore[arg-type]

    def test_verify_bytes_rejects_non_bytes_signature(self) -> None:
        with pytest.raises(ValueError, match="signature must be raw bytes"):
            verify_bytes(b"m", "s", "ab" * 32)  # type: ignore[arg-type]

    def test_verify_bytes_rejects_non_string_key(self) -> None:
        with pytest.raises(ValueError, match="trusted_public_key_hex must be a hex"):
            verify_bytes(b"m", b"s", 123)  # type: ignore[arg-type]

    def test_verify_bytes_rejects_non_hex_key(self) -> None:
        with pytest.raises(ValueError, match="must be valid hex"):
            verify_bytes(b"m", b"s", "zz")


@requires_mldsa
class TestSignBytesRoundTrip:
    def test_sign_then_verify_generic_message(self) -> None:
        key = _key()
        message = b"arbitrary domain-separated bytes"
        signature = sign_bytes(message, key)
        assert verify_bytes(message, signature, _pub_hex(key)) is True

    def test_sign_bytes_rejects_mismatched_key_type(self) -> None:
        with pytest.raises(ValueError, match="private_key must be an ml-dsa-65"):
            sign_bytes(b"m", object())

    def test_verify_bytes_rejects_wrong_key(self) -> None:
        key = _key()
        signature = sign_bytes(b"payload", key)
        other = signing_key_from_seed(generate_signing_seed())
        assert verify_bytes(b"payload", signature, _pub_hex(other)) is False

    def test_verify_bytes_rejects_tampered_message(self) -> None:
        key = _key()
        signature = sign_bytes(b"payload", key)
        assert verify_bytes(b"tampered", signature, _pub_hex(key)) is False

    def test_verify_bytes_returns_false_on_malformed_key_bytes(self) -> None:
        key = _key()
        signature = sign_bytes(b"payload", key)
        # Valid hex, but the wrong byte length for an ML-DSA public key, so the
        # backend rejects the key material and verification fails closed.
        assert verify_bytes(b"payload", signature, "ab" * 20) is False
