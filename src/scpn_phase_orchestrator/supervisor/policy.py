# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import UPDEState

# Default adjustment magnitudes
_K_BUMP = 0.05
_ZETA_BUMP = 0.1
_K_REDUCE = -0.03
_RESTORE_FRACTION = 0.5


class SupervisorPolicy:
    """Decide control actions based on regime and system state.

    Default policy:
        NOMINAL  -> no-op
        DEGRADED -> increase global coupling K
        CRITICAL -> increase damping zeta, reduce coupling on worst layers
        RECOVERY -> gradual restore toward nominal parameters
    """

    def __init__(self, regime_manager: RegimeManager):
        self._regime_manager = regime_manager

    def decide(
        self, upde_state: UPDEState, boundary_state: BoundaryState
    ) -> list[ControlAction]:
        proposed = self._regime_manager.evaluate(upde_state, boundary_state)
        regime = self._regime_manager.transition(
            self._regime_manager.current_regime, proposed
        )

        if regime == Regime.NOMINAL:
            return []

        if regime == Regime.DEGRADED:
            return [
                ControlAction(
                    knob="K",
                    scope="global",
                    value=_K_BUMP,
                    ttl_s=10.0,
                    justification="degraded: boost global coupling",
                )
            ]

        if regime == Regime.CRITICAL:
            actions = [
                ControlAction(
                    knob="zeta",
                    scope="global",
                    value=_ZETA_BUMP,
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
                        value=_K_REDUCE,
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
                value=_K_BUMP * _RESTORE_FRACTION,
                ttl_s=15.0,
                justification="recovery: gradual coupling restore",
            )
        ]

    def _worst_layer(self, upde_state):
        if not upde_state.layers:
            return None
        return min(range(len(upde_state.layers)), key=lambda i: upde_state.layers[i].R)
