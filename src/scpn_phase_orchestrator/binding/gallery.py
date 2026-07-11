# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Domainpack gallery filtering by validation tier

"""Filter and group binding specs by their validation posture for a Hub gallery.

A Studio Hub that lists SPO's domainpacks needs to keep a broad gallery from
reading as a broad set of validated solutions. Every :class:`BindingSpec` carries
a ``validation_tier`` — one of
:data:`~scpn_phase_orchestrator.binding.types.VALID_VALIDATION_TIERS`. These
helpers let a gallery select a single tier (for example show only
externally-validated packs) or group every pack by tier for a tiered display. The
grouping always covers every tier — including empty ones — so the gallery shape is
stable as packs are promoted.
"""

from __future__ import annotations

from collections.abc import Iterable

from scpn_phase_orchestrator.binding.types import (
    VALID_VALIDATION_TIERS,
    BindingSpec,
)

__all__ = [
    "group_specs_by_validation_tier",
    "select_specs_by_validation_tier",
]


def _require_known_tier(tier: str) -> None:
    """Raise ``ValueError`` if ``tier`` is not a known validation tier.

    Parameters
    ----------
    tier:
        The candidate validation tier.

    Raises
    ------
    ValueError
        If ``tier`` is not in
        :data:`~scpn_phase_orchestrator.binding.types.VALID_VALIDATION_TIERS`,
        so a gallery can never silently filter on a typo and show nothing.
    """
    if tier not in VALID_VALIDATION_TIERS:
        known = ", ".join(sorted(VALID_VALIDATION_TIERS))
        msg = f"unknown validation tier {tier!r}; known tiers: {known}"
        raise ValueError(msg)


def select_specs_by_validation_tier(
    specs: Iterable[BindingSpec], tier: str
) -> tuple[BindingSpec, ...]:
    """Return the specs at one validation tier, preserving input order.

    Parameters
    ----------
    specs:
        The binding specs to filter.
    tier:
        The validation tier to keep, one of
        :data:`~scpn_phase_orchestrator.binding.types.VALID_VALIDATION_TIERS`.

    Returns
    -------
    tuple[BindingSpec, ...]
        The specs whose ``validation_tier`` equals ``tier`` (possibly empty).

    Raises
    ------
    ValueError
        If ``tier`` is not a known validation tier.
    """
    _require_known_tier(tier)
    return tuple(spec for spec in specs if spec.validation_tier == tier)


def group_specs_by_validation_tier(
    specs: Iterable[BindingSpec],
) -> dict[str, tuple[BindingSpec, ...]]:
    """Group specs by validation tier, covering every tier for a stable shape.

    Parameters
    ----------
    specs:
        The binding specs to group.

    Returns
    -------
    dict[str, tuple[BindingSpec, ...]]
        A mapping from every tier in
        :data:`~scpn_phase_orchestrator.binding.types.VALID_VALIDATION_TIERS`
        (in sorted order) to the specs at that tier, preserving input order
        within a tier. Tiers with no specs map to an empty tuple. A spec whose
        ``validation_tier`` is not a known tier is ignored, since the validator
        is the gate that rejects such specs.
    """
    grouped: dict[str, list[BindingSpec]] = {
        tier: [] for tier in sorted(VALID_VALIDATION_TIERS)
    }
    for spec in specs:
        bucket = grouped.get(spec.validation_tier)
        if bucket is not None:
            bucket.append(spec)
    return {tier: tuple(members) for tier, members in grouped.items()}
