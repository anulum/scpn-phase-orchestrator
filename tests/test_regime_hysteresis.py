# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Regime hysteresis tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(r_values, regime_id="nominal"):
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=min(r_values) if r_values else 0.0,
        regime_id=regime_id,
    )


def _clean_boundary():
    return BoundaryState()


def _hard_violation():
    return BoundaryState(
        violations=["R below floor"],
        hard_violations=["R below floor"],
    )


# --- original tests ---


def test_nominal_when_r_high():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.9, 0.8, 0.85])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.NOMINAL


def test_critical_on_hard_violation():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.9, 0.8])
    regime = mgr.evaluate(state, _hard_violation())
    assert regime == Regime.CRITICAL


def test_degraded_when_r_low():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.4, 0.45, 0.5])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.DEGRADED


def test_critical_when_r_very_low():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.1, 0.15, 0.2])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.CRITICAL


def test_cooldown_prevents_rapid_transition():
    mgr = RegimeManager(cooldown_steps=5)
    result1 = mgr.transition(Regime.DEGRADED)
    assert result1 == Regime.DEGRADED
    result2 = mgr.transition(Regime.NOMINAL)
    assert result2 == Regime.DEGRADED


def test_always_escalate_to_critical():
    mgr = RegimeManager(cooldown_steps=100)
    mgr.transition(Regime.DEGRADED)
    result = mgr.transition(Regime.CRITICAL)
    assert result == Regime.CRITICAL


def test_recovery_when_current_is_critical():
    mgr = RegimeManager(cooldown_steps=0)
    mgr._current = Regime.CRITICAL
    state = _make_state([0.45, 0.50])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.RECOVERY


def test_critical_must_pass_through_recovery():
    mgr = RegimeManager(cooldown_steps=0)
    mgr._current = Regime.CRITICAL
    state = _make_state([0.7, 0.75])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.RECOVERY


def test_recovery_to_nominal_when_r_exceeds_band():
    mgr = RegimeManager(cooldown_steps=0)
    mgr._current = Regime.RECOVERY
    state = _make_state([0.7, 0.75])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.NOMINAL


def test_mean_r_empty_layers():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.CRITICAL


# --- v0.3 event bus tests ---


def test_transition_emits_event():
    bus = EventBus()
    mgr = RegimeManager(cooldown_steps=0, event_bus=bus)
    mgr.transition(Regime.DEGRADED)
    assert bus.count == 1
    e = bus.history[0]
    assert e.kind == "regime_transition"
    assert "nominal->degraded" in e.detail


def test_no_event_on_same_regime():
    bus = EventBus()
    mgr = RegimeManager(cooldown_steps=0, event_bus=bus)
    mgr.transition(Regime.NOMINAL)
    assert bus.count == 0


def test_force_transition_bypasses_cooldown():
    bus = EventBus()
    mgr = RegimeManager(cooldown_steps=100, event_bus=bus)
    mgr.transition(Regime.DEGRADED)
    result = mgr.force_transition(Regime.NOMINAL)
    assert result == Regime.NOMINAL
    assert bus.count == 2


def test_force_transition_same_regime_noop():
    mgr = RegimeManager(cooldown_steps=0)
    result = mgr.force_transition(Regime.NOMINAL)
    assert result == Regime.NOMINAL
    assert len(mgr.transition_history) == 0


def test_transition_history_records():
    mgr = RegimeManager(cooldown_steps=0)
    mgr.transition(Regime.DEGRADED)
    mgr.transition(Regime.CRITICAL)
    assert len(mgr.transition_history) == 2
    step, prev, new = mgr.transition_history[0]
    assert prev == Regime.NOMINAL
    assert new == Regime.DEGRADED


def test_transition_history_bounded():
    mgr = RegimeManager(cooldown_steps=0)
    for _ in range(150):
        mgr.transition(Regime.DEGRADED)
        mgr.transition(Regime.NOMINAL)
    assert len(mgr.transition_history) == 100


def test_hysteresis_hold_blocks_downward():
    mgr = RegimeManager(cooldown_steps=0, hysteresis_hold_steps=3)
    # First downward attempt
    result = mgr.transition(Regime.DEGRADED)
    assert result == Regime.NOMINAL
    # Second
    result = mgr.transition(Regime.DEGRADED)
    assert result == Regime.NOMINAL
    # Third — meets threshold, transitions
    result = mgr.transition(Regime.DEGRADED)
    assert result == Regime.DEGRADED


def test_hysteresis_hold_reset_on_same():
    mgr = RegimeManager(cooldown_steps=0, hysteresis_hold_steps=3)
    mgr.transition(Regime.DEGRADED)  # streak=1
    mgr.transition(Regime.NOMINAL)  # same → resets streak
    mgr.transition(Regime.DEGRADED)  # streak=1 again
    assert mgr.current_regime == Regime.NOMINAL


def test_critical_bypasses_hysteresis_hold():
    mgr = RegimeManager(cooldown_steps=0, hysteresis_hold_steps=100)
    result = mgr.transition(Regime.CRITICAL)
    assert result == Regime.CRITICAL


class TestRegimeHysteresisPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine → R → RegimeManager → transitions.

    Proves RegimeManager is structurally wired into the SPO pipeline.
    """

    def test_engine_trajectory_drives_regime_transitions(self):
        """Run engine with varying coupling → R changes → regime transitions."""
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        cb = CouplingBuilder()
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        mgr = RegimeManager(cooldown_steps=0)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        # Phase 1: strong coupling → high R → NOMINAL
        cs_strong = cb.build(n, 2.0, 0.2)
        for _ in range(300):
            phases = eng.step(phases, omegas, cs_strong.knm, 0.0, 0.0, cs_strong.alpha)
        r, psi = compute_order_parameter(phases)
        state = _make_state([r])
        regime = mgr.evaluate(state, _clean_boundary())
        assert regime in {Regime.NOMINAL, Regime.RECOVERY}

        # Phase 2: zero coupling → R drops → DEGRADED or CRITICAL
        phases_uncoupled = rng.uniform(0, 2 * np.pi, n)
        knm_zero = np.zeros((n, n))
        alpha_zero = np.zeros((n, n))
        for _ in range(100):
            phases_uncoupled = eng.step(
                phases_uncoupled, rng.uniform(-5, 5, n), knm_zero, 0.0, 0.0, alpha_zero
            )
        r_low, _ = compute_order_parameter(phases_uncoupled)
        state_low = _make_state([r_low])
        regime_low = mgr.evaluate(state_low, _clean_boundary())
        # With low R, should not be NOMINAL
        assert regime_low in {Regime.DEGRADED, Regime.CRITICAL, Regime.RECOVERY}

    def test_hysteresis_prevents_oscillation_in_live_sim(self):
        """Fluctuating R near boundary with hysteresis → stable regime."""
        mgr = RegimeManager(cooldown_steps=5, hysteresis=0.05)
        # Simulate R near NOMINAL/DEGRADED boundary
        regimes = []
        for r in [0.62, 0.58, 0.63, 0.57, 0.61, 0.59, 0.64]:
            state = _make_state([r])
            regime = mgr.evaluate(state, _clean_boundary())
            regimes.append(regime)
        # With cooldown=5, should not oscillate every step
        transitions = sum(
            1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]
        )
        assert transitions <= 3, f"Too many transitions: {transitions}"

    def test_event_bus_captures_engine_driven_transitions(self):
        """EventBus records transitions driven by engine R trajectory."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        bus = EventBus()
        mgr = RegimeManager(cooldown_steps=0, event_bus=bus)
        n = 4
        eng = UPDEEngine(n, dt=0.01)
        # High R → NOMINAL (no event, already nominal)
        phases_sync = np.full(n, 1.0)
        omegas = np.ones(n)
        knm = 2.0 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        phases_sync = eng.run(phases_sync, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        r_high, _ = compute_order_parameter(phases_sync)
        mgr.evaluate(_make_state([r_high]), _clean_boundary())
        initial_count = bus.count
        # Force hard violation → CRITICAL (event emitted)
        mgr.evaluate(_make_state([0.1]), _hard_violation())
        assert bus.count > initial_count

    def test_performance_evaluate_under_10us(self):
        """RegimeManager.evaluate() < 10μs per call."""
        import time

        mgr = RegimeManager(cooldown_steps=0)
        state = _make_state([0.8, 0.75])
        boundary = _clean_boundary()
        mgr.evaluate(state, boundary)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10000):
            mgr.evaluate(state, boundary)
        elapsed = (time.perf_counter() - t0) / 10000
        assert elapsed < 1e-5, f"evaluate() took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: RegimeManager tested via CouplingBuilder → UPDEEngine(RK4)
# → compute_order_parameter → evaluate(). Hysteresis: cooldown prevents oscillation,
# EventBus captures transitions. Performance: evaluate()<10μs.
