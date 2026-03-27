# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint state representation

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

__all__ = ["ImprintState"]


@dataclass(frozen=True)
class ImprintState:
    """L9 memory imprint per oscillator: accumulation vector, timestamp, attribution."""

    m_k: NDArray
    last_update: float
    attribution: dict[str, float] = field(default_factory=dict)
