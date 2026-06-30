# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AuditLogger construction and JSON guard contracts

"""Construction-boundary and payload-decode contracts for ``AuditLogger``.

These cover the rejection and skip paths the audit logger relies on to fail
closed: empty audit/key inputs, blank-line skipping while replaying an existing
chain, the signed-key refusal of a pre-existing unsigned log, the defensive
key-presence assertion in signature construction, the idle event-stream
integrity accessor, and the ``_loads_audit_json`` decode guards (malformed
JSON, non-finite constants, and non-object payloads).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.runtime.audit_logger import (
    AuditLogger,
    _loads_audit_json,
)

_PROFILE_KEYS = (
    "SPO_AUDIT_ENV",
    "SPO_AUDIT_PROFILE",
    "SPO_ENV",
    "SPO_PROFILE",
)


def _clear_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force non-production, unsigned-by-default audit behaviour."""
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    for key in _PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_blank_audit_path_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only audit path fails closed before any file is opened."""
    _clear_profile(monkeypatch)

    with pytest.raises(AuditError, match="audit path must be a non-empty path"):
        AuditLogger("   ")


def test_empty_audit_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty ``SPO_AUDIT_KEY`` is refused rather than silently ignored."""
    _clear_profile(monkeypatch)
    monkeypatch.setenv("SPO_AUDIT_KEY", "")

    with pytest.raises(AuditError, match="SPO_AUDIT_KEY must not be empty"):
        AuditLogger(tmp_path / "audit.jsonl")


def test_blank_lines_in_existing_log_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replaying an existing chain ignores blank separator lines."""
    _clear_profile(monkeypatch)
    path = tmp_path / "audit.jsonl"
    path.write_text(
        '{"step": 1, "_hash": "%s", "_audit_sequence": 1}\n\n' % ("a" * 64),
        encoding="utf-8",
    )

    logger = AuditLogger(path)
    try:
        assert logger._sequence == 1
    finally:
        logger.close()


def test_existing_unsigned_log_is_refused_when_key_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a key configured, a pre-existing unsigned record is refused.

    The leading blank line also exercises the blank-line skip inside the
    signed-stream assertion.
    """
    _clear_profile(monkeypatch)
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    path = tmp_path / "audit.jsonl"
    path.write_text('\n{"step": 1}\n', encoding="utf-8")

    with pytest.raises(AuditError, match="unsigned record at line 2"):
        AuditLogger(path)


def test_attach_signature_requires_a_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The signature builder asserts a key is present (defensive contract)."""
    _clear_profile(monkeypatch)
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        assert logger._audit_key is None
        with pytest.raises(AuditError, match="audit key missing during signature"):
            logger._attach_signature_metadata({"step": 1})
    finally:
        logger.close()


def test_event_stream_integrity_is_none_without_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A JSONL-only logger exposes no event-stream integrity summary."""
    _clear_profile(monkeypatch)
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        assert logger.event_stream_integrity is None
    finally:
        logger.close()


def test_loads_audit_json_rejects_malformed_payload() -> None:
    """A syntactically invalid payload surfaces the JSON decode error."""
    with pytest.raises(json.JSONDecodeError):
        _loads_audit_json("{not valid json")


def test_loads_audit_json_rejects_non_finite_constant() -> None:
    """Non-finite JSON constants (NaN/Infinity) are refused as audit numbers."""
    with pytest.raises(AuditError, match="only finite JSON numbers"):
        _loads_audit_json('{"value": NaN}')


def test_loads_audit_json_rejects_non_object_payload() -> None:
    """A valid JSON scalar/array is not a usable audit record."""
    with pytest.raises(AuditError, match="must be a JSON object"):
        _loads_audit_json("123")
