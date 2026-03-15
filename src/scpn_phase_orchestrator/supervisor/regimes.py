# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from collections import deque
from enum import Enum

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus, RegimeEvent
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["Regime", "RegimeManager"]

_R_CRITICAL = 0.3  # Acebrón et al. 2005 §2.3 — incoherence boundary
_R_DEGRADED = 0.6  # Acebrón et al. 2005 §2.3 — partial sync threshold


class Regime(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"


class RegimeManager:
    def __init__(
        self,
        hysteresis: float = 0.05,
        cooldown_steps: int = 10,
        event_bus: EventBus | None = None,
        hysteresis_hold_steps: int = 0,
    ) -> None:
        self._hysteresis = hysteresis
        self._cooldown_steps = cooldown_steps
        self._current = Regime.NOMINAL
        self._step_counter = 0
        self._last_transition_step = -cooldown_steps
        self._event_bus = event_bus
        self._hysteresis_hold_steps = hysteresis_hold_steps
        self._downward_streak = 0
        self.transition_history: deque[tuple[int, Regime, Regime]] = deque(maxlen=100)

    @property
    def current_regime(self) -> Regime:
        return self._current

    def evaluate(self, upde_state: UPDEState, boundary_state: BoundaryState) -> Regime:
        if boundary_state.hard_violations:
            return Regime.CRITICAL

        avg_r = self._mean_r(upde_state)

        if avg_r < _R_CRITICAL:
            return Regime.CRITICAL

        is_recovering = self._current in (Regime.CRITICAL, Regime.RECOVERY)

        if avg_r < _R_DEGRADED:
            if is_recovering:
                return Regime.RECOVERY
            return Regime.DEGRADED

        if self._current == Regime.DEGRADED and avg_r < _R_DEGRADED + self._hysteresis:
            return Regime.DEGRADED
        if is_recovering and avg_r < _R_DEGRADED + self._hysteresis:
            return Regime.RECOVERY

        if self._current == Regime.CRITICAL:
            return Regime.RECOVERY

        return Regime.NOMINAL

    def transition(self, proposed: Regime) -> Regime:
        self._step_counter += 1

        if proposed == self._current:
            self._downward_streak = 0
            return self._current

        # Soft downward transitions (non-critical) require N consecutive steps
        is_downward = self._regime_rank(proposed) > self._regime_rank(self._current)
        if (
            is_downward
            and proposed != Regime.CRITICAL
            and self._hysteresis_hold_steps > 0
        ):
            self._downward_streak += 1
            if self._downward_streak < self._hysteresis_hold_steps:
                return self._current
        else:
            self._downward_streak = 0

        in_cooldown = (
            self._step_counter - self._last_transition_step
        ) < self._cooldown_steps
        if in_cooldown and proposed != Regime.CRITICAL:
            return self._current

        prev = self._current
        self._last_transition_step = self._step_counter
        self._current = proposed
        self._downward_streak = 0
        self.transition_history.append((self._step_counter, prev, proposed))
        self._emit_transition(prev, proposed)
        return proposed

    def force_transition(self, regime: Regime) -> Regime:
        """Bypass cooldown and hysteresis hold."""
        self._step_counter += 1
        prev = self._current
        if regime == prev:
            return prev
        self._last_transition_step = self._step_counter
        self._current = regime
        self._downward_streak = 0
        self.transition_history.append((self._step_counter, prev, regime))
        self._emit_transition(prev, regime)
        return regime

    def _emit_transition(self, prev: Regime, new: Regime) -> None:
        if self._event_bus is None:
            return
        self._event_bus.post(
            RegimeEvent(
                kind="regime_transition",
                step=self._step_counter,
                detail=f"{prev.value}->{new.value}",
            )
        )

    @staticmethod
    def _regime_rank(regime: Regime) -> int:
        return {
            Regime.NOMINAL: 0,
            Regime.DEGRADED: 1,
            Regime.RECOVERY: 2,
            Regime.CRITICAL: 3,
        }[regime]

    def _mean_r(self, upde_state: UPDEState) -> float:
        if not upde_state.layers:
            return 0.0
        return sum(s.R for s in upde_state.layers) / len(upde_state.layers)
