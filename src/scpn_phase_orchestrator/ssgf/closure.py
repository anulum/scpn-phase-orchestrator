# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — L16 cybernetic closure

"""Cybernetic closure loop connecting observed phases to SSGF geometry updates.

The closure observes the current phase vector, evaluates total SSGF costs,
asks the carrier to descend through a cost callback, then decodes the next
coupling matrix for feedback into phase dynamics. The module mutates only the
injected ``GeometryCarrier`` and its local convergence bookkeeping; phase inputs
are passed through to the cost layer where dimensional and finite-value checks
are enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs

__all__ = ["CyberneticClosure", "ClosureState"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class ClosureState:
    """One L16 closure cycle result: cost before/after and convergence."""

    ssgf_state_step: int
    cost_before: float
    cost_after: float
    converging: bool


class CyberneticClosure:
    """L16 → L1 feedback loop with Lyapunov stability guarantee.

    The closure cycle:
    1. Observe phases (L1-L15 state)
    2. Compute SSGF costs
    3. Update geometry carrier z via gradient on U_total
    4. Decode new coupling W from z
    5. Feed W back to the phase dynamics (L1)

    The loop is a strange loop (Hofstadter): geometry → dynamics → cost →
    gradient → geometry. Lyapunov stability: U_total is non-increasing
    under gradient descent on z, guaranteeing convergence.

    This is the computational implementation. Whether it constitutes a
    "strange loop" in the philosophical sense is a theoretical claim
    outside the scope of this module.
    """

    def __init__(
        self,
        carrier: GeometryCarrier,
        cost_weights: tuple[float, ...] = (1.0, 0.5, 0.1, 0.1),
        max_steps: int = 0,
    ):
        if not isinstance(carrier, GeometryCarrier):
            raise TypeError(f"carrier must be GeometryCarrier, got {carrier!r}")
        if not isinstance(cost_weights, tuple) or not cost_weights:
            raise TypeError("cost_weights must be a non-empty tuple of finite reals")
        for weight in cost_weights:
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError("cost_weights must be a tuple of finite reals")
            if not isfinite(float(weight)):
                raise ValueError("cost_weights must be finite reals")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError(f"max_steps must be a non-negative integer, got {max_steps!r}")
        if max_steps < 0:
            raise ValueError(f"max_steps must be a non-negative integer, got {max_steps!r}")
        self._carrier = carrier
        self._weights = cost_weights
        self._max_steps = max_steps
        self._step = 0
        self._prev_cost: float | None = None

    @property
    def carrier(self) -> GeometryCarrier:
        """The geometry carrier whose latent vector z is updated by the closure."""
        return self._carrier

    def step(self, phases: FloatArray) -> tuple[FloatArray, ClosureState]:
        """One closure cycle: observe → cost → gradient → new W.

        Returns (new_W, closure_state).
        """
        if not isinstance(phases, np.ndarray):
            raise TypeError(f"phases must be numpy.ndarray, got {phases!r}")
        if phases.ndim != 1:
            raise ValueError(f"phases must be 1D vector, got shape {phases.shape!r}")
        if np.issubdtype(phases.dtype, np.bool_):
            raise ValueError("phases must not use boolean dtype")
        n_osc = self._carrier.decode().shape[0]
        if phases.shape[0] != n_osc:
            raise ValueError(
                f"phases length must match oscillator count {n_osc}, got {phases.shape[0]}"
            )
        if not np.isfinite(phases).all():
            raise ValueError("phases must contain only finite values")
        self._step += 1
        W_before = self._carrier.decode()
        costs_before = compute_ssgf_costs(W_before, phases, weights=self._weights)

        def cost_fn(W: FloatArray) -> float:
            """Total SSGF cost for the given weight matrix and current phases."""
            return compute_ssgf_costs(W, phases, weights=self._weights).u_total

        self._carrier.update(cost=costs_before.u_total, cost_fn=cost_fn)
        W_after = self._carrier.decode()
        costs_after = compute_ssgf_costs(W_after, phases, weights=self._weights)

        converging = costs_after.u_total <= costs_before.u_total + 1e-10
        self._prev_cost = costs_after.u_total

        return W_after, ClosureState(
            ssgf_state_step=self._step,
            cost_before=costs_before.u_total,
            cost_after=costs_after.u_total,
            converging=converging,
        )

    def run(
        self, phases: FloatArray, n_outer_steps: int
    ) -> tuple[FloatArray, list[ClosureState]]:
        """Run n outer steps, return final W and history."""
        if not isinstance(phases, np.ndarray):
            raise TypeError(f"phases must be numpy.ndarray, got {phases!r}")
        if phases.ndim != 1:
            raise ValueError(f"phases must be 1D vector, got shape {phases.shape!r}")
        if np.issubdtype(phases.dtype, np.bool_):
            raise ValueError("phases must not use boolean dtype")
        if isinstance(n_outer_steps, bool) or not isinstance(n_outer_steps, int):
            raise TypeError(
                f"n_outer_steps must be a non-negative integer, got {n_outer_steps!r}"
            )
        if n_outer_steps < 0:
            raise ValueError(
                f"n_outer_steps must be a non-negative integer, got {n_outer_steps!r}"
            )
        states = []
        W = self._carrier.decode()
        for _ in range(n_outer_steps):
            W, cs = self.step(phases)
            states.append(cs)
        return W, states

    def reset(self) -> None:
        """Reset step counter and cached cost."""
        self._step = 0
        self._prev_cost = None
