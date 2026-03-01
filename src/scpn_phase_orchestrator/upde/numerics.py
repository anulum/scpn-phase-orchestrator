# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationConfig:
    dt: float
    substeps: int = 1
    method: str = "euler"
    max_dt: float = 0.01


def check_stability(dt: float, max_omega: float, max_coupling: float) -> bool:
    """CFL-like stability bound for explicit Kuramoto integration.

    The derivative magnitude is bounded by max_omega + N * max_coupling.
    For euler, we require dt * max_deriv < pi to avoid phase jumps
    exceeding half a cycle per step.
    """
    max_deriv = max_omega + max_coupling
    if max_deriv == 0.0:
        return True
    # pi threshold: phase change per step must stay below half-cycle
    return dt * max_deriv < 3.14159265358979
