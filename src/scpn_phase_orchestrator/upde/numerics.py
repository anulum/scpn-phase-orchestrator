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
from numbers import Integral, Real

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

    def __post_init__(self) -> None:
        dt = _validate_positive_finite(self.dt, name="dt")
        max_dt = _validate_positive_finite(self.max_dt, name="max_dt")
        substeps = _validate_positive_int(self.substeps, name="substeps")
        atol = _validate_positive_finite(self.atol, name="atol")
        rtol = _validate_positive_finite(self.rtol, name="rtol")
        if not isinstance(self.method, str) or self.method not in {
            "euler",
            "rk4",
            "rk45",
        }:
            raise ValueError("method must be one of: euler, rk4, rk45")
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "max_dt", max_dt)
        object.__setattr__(self, "substeps", substeps)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)


def check_stability(dt: float, max_omega: float, max_coupling: float) -> bool:
    """CFL-like stability bound for explicit Kuramoto integration.

    Analogous to Courant–Friedrichs–Lewy (1928); see docs/specs/upde_numerics.md.
    dt * max_deriv < pi ensures phase change stays below half-cycle per step.

    Parameters
    ----------
    dt : float
        Integration step size.
    max_omega : float
        Largest absolute natural frequency in the system.
    max_coupling : float
        Largest absolute coupling magnitude in the system.

    Returns
    -------
    bool
        ``True`` when the timestep satisfies the CFL-like stability bound.

    Raises
    ------
    ValueError
        If ``dt``, ``max_omega``, or ``max_coupling`` is not finite and positive.
    """
    if (
        type(dt) in {float, int}
        and type(max_omega) in {float, int}
        and type(max_coupling) in {float, int}
    ):
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive real")
        if not math.isfinite(max_omega) or max_omega < 0.0:
            raise ValueError("max_omega must be a finite non-negative real")
        if not math.isfinite(max_coupling) or max_coupling < 0.0:
            raise ValueError("max_coupling must be a finite non-negative real")
        max_deriv = max_omega + max_coupling
        if max_deriv == 0.0:
            return True
        return dt * max_deriv < math.pi
    dt_value = _validate_positive_finite(dt, name="dt")
    omega_bound = _validate_non_negative_finite(max_omega, name="max_omega")
    coupling_bound = _validate_non_negative_finite(max_coupling, name="max_coupling")
    max_deriv = omega_bound + coupling_bound
    if max_deriv == 0.0:
        return True
    # pi threshold: phase change per step must stay below half-cycle
    return dt_value * max_deriv < math.pi


def _validate_positive_finite(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real")
    coerced = float(value)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return coerced


def _validate_non_negative_finite(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real")
    coerced = float(value)
    if not math.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real")
    return coerced


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
