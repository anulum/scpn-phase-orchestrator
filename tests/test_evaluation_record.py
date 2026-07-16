# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sealed audit record tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.evaluation.auditor import audit_detector
from scpn_phase_orchestrator.evaluation.record import (
    AUDIT_DISCLAIMER,
    AUDIT_FRAMEWORK,
    AuditRecord,
    seal_detector_audit,
)
from scpn_phase_orchestrator.runtime.audit_signing import (
    SIGNATURE_ALGORITHM,
    key_id_for_secret,
)


def _audit(target_false_alarm: float = 0.10):
    return audit_detector(
        event_scores=[2.0, 3.0, 4.0],
        null_scores=[0.0, 1.0, 0.5, -1.0],
        detector_name="demo",
        target_false_alarm=target_false_alarm,
        n_permutations=200,
    )


class TestSealDetectorAudit:
    def test_seals_and_verifies(self):
        record = seal_detector_audit(
            _audit(), corpus_id="grid-2026", captured_at="2026-07-07T15:00:00+02:00"
        )
        assert isinstance(record, AuditRecord)
        assert len(record.content_hash) == 64
        assert record.verify() is True
        assert record.framework == AUDIT_FRAMEWORK
        assert record.disclaimer == AUDIT_DISCLAIMER

    def test_open_gate_audit_seals_without_non_finite(self):
        # A -inf threshold must serialise to a string so the hash stays strict JSON.
        record = seal_detector_audit(
            _audit(target_false_alarm=1.0),
            corpus_id="c",
            captured_at="t",
        )
        assert record.audit["matched_threshold"] == "-inf"
        assert record.verify() is True

    def test_hash_matches_independent_recompute(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        payload = record.to_record()
        del payload["content_hash"]
        assert record.content_hash == canonical_record_hash(payload)

    def test_to_record_includes_hash_and_provenance(self):
        record = seal_detector_audit(
            _audit(), corpus_id="grid-2026", captured_at="2026-07-07T15:00:00+02:00"
        )
        payload = record.to_record()
        assert payload["corpus_id"] == "grid-2026"
        assert payload["captured_at"] == "2026-07-07T15:00:00+02:00"
        assert payload["content_hash"] == record.content_hash
        assert "audit" in payload

    def test_tampering_is_detected(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        # Mutate the sealed payload in place; verify must fail.
        object.__setattr__(record, "corpus_id", "forged")
        assert record.verify() is False

    def test_two_corpora_hash_differently(self):
        audit = _audit()
        a = seal_detector_audit(audit, corpus_id="corpus-a", captured_at="t")
        b = seal_detector_audit(audit, corpus_id="corpus-b", captured_at="t")
        assert a.content_hash != b.content_hash

    def test_empty_corpus_id_rejected(self):
        with pytest.raises(ValueError, match="corpus_id must not be empty"):
            seal_detector_audit(_audit(), corpus_id="", captured_at="t")

    def test_empty_captured_at_rejected(self):
        with pytest.raises(ValueError, match="captured_at must not be empty"):
            seal_detector_audit(_audit(), corpus_id="c", captured_at="")


class TestSignedAuditRecord:
    _KEY = "s3cr3t-audit-key-material"

    def _keyring(self, key: str = _KEY) -> dict[str, str]:
        return {key_id_for_secret(key): key}

    def test_unsigned_record_carries_no_signature(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        assert record.signature is None
        assert record.signing_key_id is None
        payload = record.to_record()
        assert "signature" not in payload
        assert "signing_key_id" not in payload
        assert "signature_algorithm" not in payload

    def test_signing_leaves_the_content_hash_unchanged(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        signed = record.sign(self._KEY)
        # The signature sits outside the hashed payload, so the seal is identical.
        assert signed.content_hash == record.content_hash
        assert signed.verify() is True

    def test_signed_record_verifies_against_the_key(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        signed = record.sign(self._KEY)
        assert signed.signing_key_id == key_id_for_secret(self._KEY)
        assert signed.verify_signature(self._keyring()) is True

    def test_seal_with_key_signs_in_one_call(self):
        signed = seal_detector_audit(
            _audit(), corpus_id="c", captured_at="t", key=self._KEY
        )
        assert signed.signature is not None
        assert signed.verify_signature(self._keyring()) is True
        payload = signed.to_record()
        assert payload["signature"] == signed.signature
        assert payload["signing_key_id"] == signed.signing_key_id
        assert payload["signature_algorithm"] == SIGNATURE_ALGORITHM

    def test_unsigned_record_signature_does_not_verify(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        assert record.verify_signature(self._keyring()) is False

    def test_unknown_key_id_does_not_verify(self):
        signed = seal_detector_audit(
            _audit(), corpus_id="c", captured_at="t", key=self._KEY
        )
        assert signed.verify_signature(self._keyring("a-different-key")) is False

    def test_signature_from_another_record_does_not_verify(self):
        signed = seal_detector_audit(
            _audit(), corpus_id="corpus-a", captured_at="t", key=self._KEY
        )
        # Graft the signature onto a record with different content (different hash).
        forged = AuditRecord(
            corpus_id="corpus-b",
            captured_at="t",
            audit=signed.audit,
            signature=signed.signature,
            signing_key_id=signed.signing_key_id,
        )
        assert forged.content_hash != signed.content_hash
        assert forged.verify_signature(self._keyring()) is False

    def test_rotation_keyring_still_verifies_an_old_signature(self):
        signed = seal_detector_audit(
            _audit(), corpus_id="c", captured_at="t", key=self._KEY
        )
        rotated = {
            key_id_for_secret("new-current-key"): "new-current-key",
            key_id_for_secret(self._KEY): self._KEY,
        }
        assert signed.verify_signature(rotated) is True

    def test_empty_signing_key_rejected(self):
        record = seal_detector_audit(_audit(), corpus_id="c", captured_at="t")
        with pytest.raises(ValueError, match="audit signing key must not be empty"):
            record.sign("")
        with pytest.raises(ValueError, match="audit signing key must not be empty"):
            seal_detector_audit(_audit(), corpus_id="c", captured_at="t", key="")
