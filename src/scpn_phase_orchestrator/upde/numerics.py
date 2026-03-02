# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["IntegrationConfig", "check_stability"]


@dataclass(frozen=True)
class IntegrationConfig:
    dt: float
    substeps: int = 1
    method: str = "euler"
    max_dt: float = 0.01
    atol: float = 1e-6
    rtol: float = 1e-3


def check_stability(dt: float, max_omega: float, max_coupling: float) -> bool:
    """CFL-like stability bound for explicit Kuramoto integration.

    Analogous to Courant–Friedrichs–Lewy (1928); see docs/specs/upde_numerics.md.
    dt * max_deriv < pi ensures phase change stays below half-cycle per step.
    """
    max_deriv = max_omega + max_coupling
    if max_deriv == 0.0:
        return True
    # pi threshold: phase change per step must stay below half-cycle
    return dt * max_deriv < math.pi
