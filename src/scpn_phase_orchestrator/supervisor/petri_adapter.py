# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net adapter

from __future__ import annotations

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.events import EventBus, RegimeEvent
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet
from scpn_phase_orchestrator.supervisor.regimes import Regime

__all__ = ["PetriNetAdapter"]

_REGIME_LOOKUP = {r.value.upper(): r for r in Regime}


class PetriNetAdapter:
    """Map Petri net markings to Regime values.

    Each place in the net maps to a Regime via *place_to_regime*.
    When multiple places are marked, the highest-severity regime wins
    (CRITICAL > RECOVERY > DEGRADED > NOMINAL).
    """

    _PRIORITY = {
        Regime.NOMINAL: 0,
        Regime.DEGRADED: 1,
        Regime.RECOVERY: 2,
        Regime.CRITICAL: 3,
    }

    def __init__(
        self,
        net: PetriNet,
        initial_marking: Marking,
        place_to_regime: dict[str, str],
        event_bus: EventBus | None = None,
    ) -> None:
        self._net = net
        self._marking = initial_marking
        self._place_to_regime: dict[str, Regime] = {}
        for place, regime_str in place_to_regime.items():
            key = regime_str.upper()
            if key not in _REGIME_LOOKUP:
                raise PolicyError(f"unknown regime {regime_str!r} for place {place!r}")
            self._place_to_regime[place] = _REGIME_LOOKUP[key]
        self._event_bus = event_bus
        self._step = 0

    @property
    def marking(self) -> Marking:
        """Current Petri net marking (token distribution)."""
        return self._marking

    @property
    def net(self) -> PetriNet:
        """The underlying Petri net structure."""
        return self._net

    def step(self, ctx: dict[str, float]) -> Regime:
        """Advance the Petri net one step and return the active regime."""
        self._step += 1
        new_marking, fired = self._net.step(self._marking, ctx)
        if fired is not None:
            self._marking = new_marking
            if self._event_bus is not None:
                self._event_bus.post(
                    RegimeEvent(
                        kind="petri_transition",
                        step=self._step,
                        detail=fired.name,
                    )
                )
        return self._active_regime()

    def _active_regime(self) -> Regime:
        best = Regime.NOMINAL
        for place in self._marking.active_places():
            regime = self._place_to_regime.get(place, Regime.NOMINAL)
            if self._PRIORITY[regime] > self._PRIORITY[best]:
                best = regime
        return best
