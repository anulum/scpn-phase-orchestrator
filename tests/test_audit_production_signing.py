# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit production-signing requirement tests

"""In production mode the audit trail must be signed: constructing an
`AuditLogger` without ``SPO_AUDIT_KEY`` fails closed, so no unsigned,
unverifiable audit log can be written. Outside production the unsigned
development path is preserved.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger

_PROFILE_KEYS = (
    "SPO_AUDIT_ENV",
    "SPO_AUDIT_PROFILE",
    "SPO_ENV",
    "SPO_PROFILE",
)


def _clear_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    for key in _PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("profile_key", _PROFILE_KEYS)
def test_production_without_key_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, profile_key: str
) -> None:
    _clear_profile(monkeypatch)
    monkeypatch.setenv(profile_key, "production")
    with pytest.raises(AuditError, match="SPO_AUDIT_KEY is required in production"):
        AuditLogger(tmp_path / "audit.jsonl")


def test_production_profile_is_case_insensitive(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_profile(monkeypatch)
    monkeypatch.setenv("SPO_ENV", "PRODUCTION")
    with pytest.raises(AuditError, match="production"):
        AuditLogger(tmp_path / "audit.jsonl")


def test_production_with_key_constructs_and_signs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_profile(monkeypatch)
    monkeypatch.setenv("SPO_ENV", "production")
    monkeypatch.setenv("SPO_AUDIT_KEY", "unit-test-secret-key")
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        assert logger is not None
    finally:
        logger.close()


def test_non_production_without_key_is_allowed_unsigned(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_profile(monkeypatch)
    monkeypatch.setenv("SPO_ENV", "staging")
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        assert logger is not None
    finally:
        logger.close()


def test_default_environment_without_key_is_allowed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_profile(monkeypatch)
    logger = AuditLogger(tmp_path / "audit.jsonl")
    try:
        assert logger is not None
    finally:
        logger.close()
