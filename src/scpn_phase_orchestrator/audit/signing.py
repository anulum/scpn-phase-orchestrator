# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit signing helpers

from __future__ import annotations

import hashlib
import json
import os

SIGNATURE_ALGORITHM = "HMAC-SHA256"

__all__ = [
    "SIGNATURE_ALGORITHM",
    "audit_verification_keys",
    "key_id_for_secret",
]


def key_id_for_secret(secret: str) -> str:
    """Return the audit key identifier stored in signed audit metadata."""

    if secret == "":
        raise ValueError("audit signing key must not be empty")
    return hashlib.sha256(secret.encode()).hexdigest()[:16]


def audit_verification_keys() -> dict[str, str]:
    """Load current and historical audit verification keys from the environment.

    `SPO_AUDIT_KEY` supplies the current operational key. `SPO_AUDIT_KEYRING`
    supplies historical keys as a JSON object mapping `sha256(secret)[:16]` to
    the corresponding secret. Invalid or mismatched keyrings fail closed by
    raising `ValueError`.
    """

    keys: dict[str, str] = {}
    current_key = os.environ.get("SPO_AUDIT_KEY")
    if current_key == "":
        raise ValueError("SPO_AUDIT_KEY must not be empty")
    if current_key is not None:
        keys[key_id_for_secret(current_key)] = current_key

    keyring_json = os.environ.get("SPO_AUDIT_KEYRING")
    if keyring_json is None:
        return keys
    if keyring_json == "":
        raise ValueError("SPO_AUDIT_KEYRING must not be empty")
    try:
        loaded = json.loads(keyring_json)
    except json.JSONDecodeError as exc:
        raise ValueError("SPO_AUDIT_KEYRING must be JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError("SPO_AUDIT_KEYRING must be a JSON object")
    for key_id, key_value in loaded.items():
        if not isinstance(key_id, str) or key_id == "":
            raise ValueError("SPO_AUDIT_KEYRING key ids must be non-empty strings")
        if not isinstance(key_value, str) or key_value == "":
            raise ValueError("SPO_AUDIT_KEYRING keys must be non-empty strings")
        if key_id != key_id_for_secret(key_value):
            raise ValueError("SPO_AUDIT_KEYRING key id does not match key material")
        keys[key_id] = key_value
    return keys
