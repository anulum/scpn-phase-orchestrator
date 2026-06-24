# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI value-coercion helpers

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations


def _string_list(value: object) -> list[str]:
    """Return ``value`` as a list of strings, else raise ``ValueError``."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _float_list(value: object) -> list[float]:
    """Return ``value`` as a list of floats, else raise ``ValueError``."""
    if isinstance(value, list):
        return [float(item) for item in value if isinstance(item, int | float)]
    return []


def _float_value(value: object) -> float:
    """Return ``value`` as a float, else raise ``ValueError``."""
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_value(value: object) -> int:
    """Return ``value`` as an integer, else raise ``ValueError``."""
    if isinstance(value, int):
        return value
    return 0


def _count_dict(value: object) -> dict[str, int]:
    """Return ``value`` as a mapping of names to integer counts, else raise."""
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        if isinstance(raw_count, int):
            counts[str(key)] = raw_count
    return counts
