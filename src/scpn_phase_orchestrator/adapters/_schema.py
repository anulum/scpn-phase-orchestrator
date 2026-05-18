# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Shared adapter input validators

"""Shared primitive validators used by adapter constructors and endpoints.

The helpers normalize common adapter inputs such as non-empty strings,
non-negative integers, and TCP ports before clients, sockets, or stores are
created. They raise ``ValueError`` with field-specific messages and perform no
I/O or dependency imports.
"""

from __future__ import annotations

__all__ = [
    "require_non_empty_str",
    "require_non_negative_int",
    "require_tcp_port",
]


def require_non_empty_str(value: object, *, field: str) -> str:
    """Return a stripped non-empty string value or raise a `ValueError`."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def require_non_negative_int(value: object, *, field: str) -> int:
    """Return a non-negative int or raise a `ValueError`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def require_tcp_port(value: object, *, field: str) -> int:
    """Return a valid TCP port (1..65535) or raise a `ValueError`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not (1 <= value <= 65535):
        raise ValueError(f"{field} must be in the range 1..65535")
    return value
