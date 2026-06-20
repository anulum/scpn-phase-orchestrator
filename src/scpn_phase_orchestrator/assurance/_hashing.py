# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance bundle canonical hashing

"""Deterministic canonical-JSON hashing for assurance-case records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

_SHA256_LENGTH = 64
_SHA256_ALPHABET = set("0123456789abcdef")


def canonical_record_hash(record: Mapping[str, object]) -> str:
    """Return the SHA-256 of a record under canonical JSON serialisation.

    The canonical form sorts object keys and removes incidental whitespace so a
    record always hashes to the same digest regardless of construction order.

    Parameters
    ----------
    record:
        A JSON-safe mapping.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


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
