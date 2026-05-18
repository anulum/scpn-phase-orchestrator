# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit signing helper tests

from __future__ import annotations

import hashlib
import json

import pytest

from scpn_phase_orchestrator.audit.signing import (
    SIGNATURE_ALGORITHM,
    audit_verification_keys,
    key_id_for_secret,
)


def test_key_id_for_secret_matches_audit_metadata_prefix() -> None:
    key_material = "audit-signing-material"

    key_id = key_id_for_secret(key_material)

    assert key_id == hashlib.sha256(key_material.encode()).hexdigest()[:16]
    assert len(key_id) == 16


def test_key_id_for_secret_rejects_empty_secret() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        key_id_for_secret("")


def test_audit_verification_keys_loads_current_and_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_material = "current-audit-key-material"
    historical_material = "historical-audit-key-material"
    keyring = {key_id_for_secret(historical_material): historical_material}
    monkeypatch.setenv("SPO_AUDIT_KEY", current_material)
    monkeypatch.setenv("SPO_AUDIT_KEYRING", json.dumps(keyring))

    keys = audit_verification_keys()

    assert keys == {
        key_id_for_secret(current_material): current_material,
        key_id_for_secret(historical_material): historical_material,
    }


def test_audit_verification_keys_rejects_mismatched_keyring_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    monkeypatch.setenv("SPO_AUDIT_KEYRING", json.dumps({"bad-key-id": "material"}))

    with pytest.raises(ValueError, match="key id does not match"):
        audit_verification_keys()


def test_signature_algorithm_is_hmac_sha256() -> None:
    assert SIGNATURE_ALGORITHM == "HMAC-SHA256"
