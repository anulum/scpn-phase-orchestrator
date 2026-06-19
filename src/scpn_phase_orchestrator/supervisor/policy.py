# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy engine

"""Reactive supervisor policy that maps regimes and state into control proposals.

``SupervisorPolicy`` derives a proposed regime from direct metrics or an
optional Petri adapter, commits it through ``RegimeManager``, and emits bounded
``ControlAction`` proposals for degraded, critical, or recovery states. Petri
failures fall back to direct regime logic. The policy only proposes actions; it
does not apply actuation or mutate coupling matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["SupervisorPolicy", "SupervisorPolicyGains"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisorPolicyGains:
    """Tunable regime-action gains for a deployment-specific supervisor."""

    k_bump: float = 0.05
    zeta_bump: float = 0.1
    k_reduce: float = -0.03
    restore_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.k_bump <= 0.0:
            raise ValueError("k_bump must be positive")
        if self.zeta_bump <= 0.0:
            raise ValueError("zeta_bump must be positive")
        if self.k_reduce >= 0.0:
            raise ValueError("k_reduce must be negative")
        if not 0.0 < self.restore_fraction <= 1.0:
            raise ValueError("restore_fraction must be in (0, 1]")


class SupervisorPolicy:
    """Decide control actions based on regime and system state.

    When *petri_adapter* is provided, regime is derived from the Petri net
    marking instead of RegimeManager.evaluate().
    """

    def __init__(
        self,
        regime_manager: RegimeManager,
        petri_adapter: PetriNetAdapter | None = None,
        gains: SupervisorPolicyGains | None = None,
    ):
        self._regime_manager = regime_manager
        self._petri_adapter = petri_adapter
        self._gains = gains or SupervisorPolicyGains()

    def decide(
        self,
        upde_state: UPDEState,
        boundary_state: BoundaryState,
        petri_ctx: dict[str, float] | None = None,
    ) -> list[ControlAction]:
        """Evaluate regime and return control actions for the current state.

        Parameters
        ----------
        upde_state : UPDEState
            The current UPDE state.
        boundary_state : BoundaryState
            The current boundary-observer state.
        petri_ctx : dict[str, float] | None
            Petri context metric values, or ``None``.

        Returns
        -------
        list[ControlAction]
            The control actions proposed for the current state.
        """
        proposed = self._proposed_regime(upde_state, boundary_state, petri_ctx)
        regime = self._regime_manager.transition(proposed)

        actions = self._actions_for_regime(regime, upde_state)
        logger.info(
            "supervisor decide: regime=%s actions=%d",
            regime.value,
            len(actions),
            extra={
                "regime": regime.value,
                "n_actions": len(actions),
                "n_violations": len(boundary_state.violations),
                "stability_proxy": upde_state.stability_proxy,
                "knobs": [a.knob for a in actions],
            },
        )
        return actions

    def _actions_for_regime(
        self, regime: Regime, upde_state: UPDEState
    ) -> list[ControlAction]:
        if regime == Regime.NOMINAL:
            return []

        if regime == Regime.DEGRADED:
            return [
                ControlAction(
                    knob="K",
                    scope="global",
                    value=self._gains.k_bump,
                    ttl_s=10.0,
                    justification="degraded: boost global coupling",
                )
            ]

        if regime == Regime.CRITICAL:
            actions = [
                ControlAction(
                    knob="zeta",
                    scope="global",
                    value=self._gains.zeta_bump,
                    ttl_s=5.0,
                    justification="critical: increase damping",
                )
            ]
            worst = self._worst_layer(upde_state)
            if worst is not None:
                actions.append(
                    ControlAction(
                        knob="K",
                        scope=f"layer_{worst}",
                        value=self._gains.k_reduce,
                        ttl_s=5.0,
                        justification=f"critical: reduce coupling on layer {worst}",
                    )
                )
            return actions

        # RECOVERY
        return [
            ControlAction(
                knob="K",
                scope="global",
                value=self._gains.k_bump * self._gains.restore_fraction,
                ttl_s=15.0,
                justification="recovery: gradual coupling restore",
            )
        ]

    def _proposed_regime(
        self,
        upde_state: UPDEState,
        boundary_state: BoundaryState,
        petri_ctx: dict[str, float] | None,
    ) -> Regime:
        if self._petri_adapter is not None and petri_ctx is not None:
            try:
                return self._petri_adapter.step(petri_ctx)
            except Exception:
                # Fault-injection hardening: supervisor remains operational if
                # Petri stepping fails and falls back to direct regime logic.
                return self._regime_manager.evaluate(upde_state, boundary_state)
        return self._regime_manager.evaluate(upde_state, boundary_state)

    def _worst_layer(self, upde_state: UPDEState) -> int | None:
        if not upde_state.layers:
            return None
        return min(range(len(upde_state.layers)), key=lambda i: upde_state.layers[i].R)
