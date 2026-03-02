# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

__all__ = ["ImprintState"]


@dataclass(frozen=True)
class ImprintState:
    m_k: NDArray
    last_update: float
    attribution: dict[str, float] = field(default_factory=dict)
