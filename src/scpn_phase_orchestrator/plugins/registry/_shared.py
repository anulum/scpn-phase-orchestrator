# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin registry shared validation primitives

"""Shared validation, hashing, and capability-kind primitives for the registry."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, TypeAlias

PluginKind: TypeAlias = Literal[
    "domainpack",
    "extractor",
    "monitor",
    "actuator",
    "bridge",
]


_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")

_VALID_KINDS = {"domainpack", "extractor", "monitor", "actuator", "bridge"}


def _require_identifier(value: str, label: str) -> None:
    """Validate that ``value`` is a whitespace-free identifier."""
    _require_non_empty(value, label)
    if any(char.isspace() for char in value):
        raise ValueError(f"{label} must not contain whitespace")


def _require_non_empty(value: str, label: str) -> None:
    """Validate that ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _validate_sha256(value: str, label: str) -> None:
    """Validate that ``value`` is a SHA-256 hex digest."""
    # int(value, 16) also accepts a sign, a 0x prefix, underscores and
    # surrounding whitespace, so a 64-character non-digest passed as a hash.
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")


def _record_hash(record: dict[str, object]) -> str:
    """Return the canonical hash of a record."""
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
