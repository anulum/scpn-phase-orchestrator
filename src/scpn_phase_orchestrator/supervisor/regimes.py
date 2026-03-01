# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from enum import Enum

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.upde.metrics import UPDEState

try:
    from spo_kernel import PyRegimeManager as _RustRegimeManager  # noqa: F401

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

_R_CRITICAL = 0.3  # Acebrón et al. 2005 §2.3 — incoherence boundary
_R_DEGRADED = 0.6  # Acebrón et al. 2005 §2.3 — partial sync threshold


class Regime(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"


class RegimeManager:
    def __init__(self, hysteresis=0.05, cooldown_steps=10):
        self._hysteresis = hysteresis
        self._cooldown_steps = cooldown_steps
        self._current = Regime.NOMINAL
        self._step_counter = 0
        self._last_transition_step = -cooldown_steps  # allow immediate first transition

    @property
    def current_regime(self) -> Regime:
        return self._current

    def evaluate(self, upde_state: UPDEState, boundary_state: BoundaryState) -> Regime:
        """Determine regime from R values and boundary violations."""
        if boundary_state.hard_violations:
            return Regime.CRITICAL

        avg_r = self._mean_r(upde_state)

        if avg_r < _R_CRITICAL:
            return Regime.CRITICAL
        if avg_r < _R_DEGRADED:
            return Regime.DEGRADED
        if self._current == Regime.CRITICAL:
            return Regime.RECOVERY
        return Regime.NOMINAL

    def transition(self, current: Regime, proposed: Regime) -> Regime:
        """Apply hysteresis and cooldown to proposed transition."""
        self._step_counter += 1

        if proposed == current:
            return current

        in_cooldown = (
            self._step_counter - self._last_transition_step
        ) < self._cooldown_steps
        if in_cooldown and proposed != Regime.CRITICAL:
            return current

        self._last_transition_step = self._step_counter
        self._current = proposed
        return proposed

    def _mean_r(self, upde_state):
        if not upde_state.layers:
            return 0.0
        return sum(s.R for s in upde_state.layers) / len(upde_state.layers)
