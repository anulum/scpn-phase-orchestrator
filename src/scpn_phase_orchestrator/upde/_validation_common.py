# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - shared UPDE validation helpers

"""Shared guard helpers for private UPDE accelerator validation modules."""

from __future__ import annotations

import numpy as np

__all__ = ["contains_boolean_alias", "validate_non_negative_tolerance"]


def contains_boolean_alias(value: object) -> bool:
    """Return whether ``value`` is or contains a Python or NumPy boolean."""
    if isinstance(value, (bool, np.bool_)):
        return True
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def validate_non_negative_tolerance(value: object, *, name: str = "tolerance") -> float:
    """Return ``value`` as a finite, non-negative comparison tolerance."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed
