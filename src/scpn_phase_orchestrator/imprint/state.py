# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint state representation

"""Immutable per-oscillator imprint state containers.

`ImprintState` stores the L9 memory vector, its last update timestamp, and
optional attribution weights. Validation is intentionally performed by the
update/model layer so serialized historical states can still be inspected
before being accepted into active dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["ImprintState"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ImprintState:
    """L9 memory imprint per oscillator: accumulation vector, timestamp, attribution."""

    m_k: FloatArray
    last_update: float
    attribution: dict[str, float] = field(default_factory=dict)
