# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin shared-helper tests

"""Shared digital-twin validation helper contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from scpn_phase_orchestrator.binding.digital_twin import _shared as shared


def test_has_authorization_accepts_case_insensitive_bearer_header() -> None:
    """Authorization is present when a decoded header has non-blank text."""
    assert shared._has_authorization({"Authorization": "  Bearer token  "}) is True
    assert shared._has_authorization({"x-request-id": "req-1"}) is False


def test_has_authorization_fails_closed_for_missing_or_malformed_header() -> None:
    """Missing, blank, and non-string authorization values are rejected."""
    malformed = cast("Mapping[str, str]", {"authorization": object()})

    assert shared._has_authorization(None) is False
    assert shared._has_authorization({"authorization": "   "}) is False
    assert shared._has_authorization(malformed) is False


def test_require_non_empty_accepts_trimmed_text() -> None:
    """Non-empty string values pass through the shared guard."""
    assert shared._require_non_empty("  machine-line  ", "adapter name") is None


@pytest.mark.parametrize("value", ["", "   ", cast("str", object())])
def test_require_non_empty_rejects_empty_or_non_string_values(value: str) -> None:
    """Empty and malformed string fields fail with the requested field name."""
    with pytest.raises(ValueError, match="adapter name must be a non-empty string"):
        shared._require_non_empty(value, "adapter name")
