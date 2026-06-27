# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin shared validation and authorization

"""Shared validation and request-authorization primitives for digital-twin sync."""

from __future__ import annotations

from collections.abc import Mapping


def _has_authorization(headers: Mapping[str, str] | None) -> bool:
    """Return whether the request carries valid authorization."""
    if headers is None:
        return False
    normalised = {key.lower(): value for key, value in headers.items()}
    token = normalised.get("authorization")
    return isinstance(token, str) and bool(token.strip())


def _require_non_empty(value: str, name: str) -> None:
    """Validate that ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
