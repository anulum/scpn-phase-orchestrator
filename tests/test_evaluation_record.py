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
