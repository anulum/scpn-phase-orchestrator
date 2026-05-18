# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Numerical utilities

"""Numerical integration configuration and explicit-step stability checks.

``IntegrationConfig`` records solver tolerances and method selection, while
``check_stability`` provides a CFL-like phase-step bound for explicit Kuramoto
integration. The helper is deliberately conservative and side-effect free: it
does not adapt solvers or clamp parameters, it only reports whether the supplied
derivative bound keeps a single step below a half-cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["IntegrationConfig", "check_stability"]


@dataclass(frozen=True)
class IntegrationConfig:
    """Numerical integration parameters for the phase ODE solver."""

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
