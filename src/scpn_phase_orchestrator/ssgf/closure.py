# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — L16 cybernetic closure

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs

__all__ = ["CyberneticClosure", "ClosureState"]


@dataclass
class ClosureState:
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
        self._carrier = carrier
        self._weights = cost_weights
        self._max_steps = max_steps
        self._step = 0
        self._prev_cost: float | None = None

    @property
    def carrier(self) -> GeometryCarrier:
        return self._carrier

    def step(self, phases: NDArray) -> tuple[NDArray, ClosureState]:
        """One closure cycle: observe → cost → gradient → new W.

        Returns (new_W, closure_state).
        """
        self._step += 1
        W_before = self._carrier.decode()
        costs_before = compute_ssgf_costs(W_before, phases, weights=self._weights)

        def cost_fn(W: NDArray) -> float:
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
        self, phases: NDArray, n_outer_steps: int
    ) -> tuple[NDArray, list[ClosureState]]:
        """Run n outer steps, return final W and history."""
        states = []
        W = self._carrier.decode()
        for _ in range(n_outer_steps):
            W, cs = self.step(phases)
            states.append(cs)
        return W, states

    def reset(self) -> None:
        self._step = 0
        self._prev_cost = None
