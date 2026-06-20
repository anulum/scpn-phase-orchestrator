# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal exporter shared identifiers and result types

"""Shared identifier sanitisation helpers and PRISM/TLA export result types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class PrismExport:
    """PRISM model text plus the identifier mapping used during export."""

    model: str
    place_names: dict[str, str]
    metric_names: dict[str, str]
    transition_names: dict[str, str]
    rule_names: dict[str, str] = field(default_factory=dict)
    action_names: dict[str, str] = field(default_factory=dict)
    stl_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TLAExport:
    """TLA+ module text plus the identifier mapping used during export."""

    module: str
    place_names: dict[str, str]
    metric_names: dict[str, str]
    transition_names: dict[str, str]
    rule_names: dict[str, str] = field(default_factory=dict)
    action_names: dict[str, str] = field(default_factory=dict)


def _identifier(raw: str, *, prefix: str) -> str:
    cleaned = _IDENT_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _tla_module_identifier(raw: str) -> str:
    identifier = _identifier(raw, prefix="SpoModule")
    if not identifier[0].isalpha():
        identifier = f"Spo{identifier}"
    return identifier


def _unique_identifier(
    raw: str,
    *,
    prefix: str,
    used: set[str],
) -> str:
    base = _identifier(raw, prefix=prefix)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
