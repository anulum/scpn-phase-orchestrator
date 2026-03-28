#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Petri Net + Policy DSL
#
# The supervisor's brain: a Petri net FSM manages regime transitions
# (nominal → degraded → critical → recovery) with formal guards,
# while policy rules fire actions based on R thresholds.
#
# Usage: python examples/petri_policy_demo.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 8
    rng = np.random.default_rng(0)
    omegas = rng.uniform(-1.5, 1.5, n)
    knm = np.ones((n, n)) * 1.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    eng = UPDEEngine(n, dt=0.01)
    phases = rng.uniform(0, TWO_PI, n)

    # Set up supervisor with event bus
    bus = EventBus()
    events_received: list[str] = []
    bus.subscribe(lambda e: events_received.append(e.detail))

    rm = RegimeManager(
        hysteresis=0.05,
        cooldown_steps=5,
        event_bus=bus,
    )

    print("Petri Net Regime FSM + Policy Rules")
    print("=" * 55)

    # Phase 1: Normal operation
    print("\nPhase 1: Normal operation (K=1.0)")
    for step in range(100):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        upde_state = UPDEState(
            layers=[LayerState(R=R, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=rm.current_regime.value,
        )
        proposed = rm.evaluate(upde_state, BoundaryState())
        rm.transition(proposed)

        if step % 25 == 24:
            print(f"  Step {step + 1}: R={R:.3f} [{rm.current_regime.value}]")

    # Phase 2: Degrade coupling — trigger regime transitions
    print("\nPhase 2: Coupling degraded (K=0.1)")
    knm_weak = knm * 0.1
    for step in range(100, 200):
        phases = eng.step(phases, omegas, knm_weak, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        upde_state = UPDEState(
            layers=[LayerState(R=R, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=rm.current_regime.value,
        )
        proposed = rm.evaluate(upde_state, BoundaryState())
        rm.transition(proposed)

        if step % 25 == 24:
            print(f"  Step {step + 1}: R={R:.3f} [{rm.current_regime.value}]")

    # Phase 3: Hard boundary violation → force critical
    print("\nPhase 3: Hard boundary violation injected")
    hard_bs = BoundaryState(hard_violations=["coupling_below_minimum"])
    proposed = rm.evaluate(upde_state, hard_bs)
    rm.transition(proposed)
    print(f"  Regime: {rm.current_regime.value} (forced by violation)")

    # Phase 4: Recovery
    print("\nPhase 4: Coupling restored (K=3.0)")
    knm_strong = knm * 3.0
    for step in range(200, 300):
        phases = eng.step(phases, omegas, knm_strong, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        upde_state = UPDEState(
            layers=[LayerState(R=R, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=R,
            regime_id=rm.current_regime.value,
        )
        proposed = rm.evaluate(upde_state, BoundaryState())
        rm.transition(proposed)

        if step % 25 == 24:
            print(f"  Step {step + 1}: R={R:.3f} [{rm.current_regime.value}]")

    # Summary
    print(f"\nTransition history ({len(rm.transition_history)} transitions):")
    for step_n, prev, new in rm.transition_history:
        print(f"  Step {step_n}: {prev.value} → {new.value}")

    print(f"\nEvent bus received {bus.count} events.")
    print("Formal state machine + event-driven policy = auditable control.")


if __name__ == "__main__":
    main()
