# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance bundle canonical hashing

"""Deterministic canonical-JSON hashing for assurance-case records.

The canonical hash and the sealed-JSON loader are implemented once in the core
``monitor._sealed_record`` module (core code may not import this runtime
package) and re-exported here for the assurance records.
"""

from __future__ import annotations

from scpn_phase_orchestrator.monitor._sealed_record import (
    canonical_record_hash,
    load_sealed_json,
)

__all__ = ["canonical_record_hash", "load_sealed_json", "require_sha256"]

_SHA256_LENGTH = 64
_SHA256_ALPHABET = set("0123456789abcdef")


def require_sha256(value: object, field_name: str) -> str:
    """Return ``value`` if it is a lowercase hex SHA-256 digest, else raise.

    Parameters
    ----------
    value:
        The candidate digest.
    field_name:
        Name used in the error message.

    Returns
    -------
    str
        The validated digest.

    Raises
    ------
    ValueError
        If ``value`` is not a 64-character lowercase hexadecimal string.
    """
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    if not set(value) <= _SHA256_ALPHABET:
        raise ValueError(f"{field_name} must be lowercase hexadecimal")
    return value
