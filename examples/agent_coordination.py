#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Multi-Agent AI Coordination
#
# Models AI agent collaboration as a synchronisation problem.
# 4 agents work with a human operator on a shared codebase.
# Synchronisation (high R) = harmony. Desync (low R) = conflicts.
#
# Demonstrates: SYNAPSE_CHANNEL-style coordination modelled as phase
# oscillators with SPO's supervisor detecting conflicts and
# redistributing work.
#
# Usage: python examples/agent_coordination.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    agents = ["Agent-A", "Agent-B", "Agent-C", "Human"]
    n = len(agents)
    rng = np.random.default_rng(42)

    # Natural work rhythms (cycles/hour): each agent has a different pace
    omegas = TWO_PI * np.array([0.8, 1.2, 0.6, 0.3])

    # Coupling: agents working on related repos couple
    knm = np.array(
        [
            [0.0, 1.5, 0.5, 2.0],  # Agent-A: strong with Human, medium Agent-B
            [1.5, 0.0, 1.0, 0.5],  # Agent-B: strong with Agent-A, medium Agent-C
            [0.5, 1.0, 0.0, 0.8],  # Agent-C: medium with Agent-B and Human
            [2.0, 0.5, 0.8, 0.0],  # Human: directs Agent-A primarily
        ]
    )

    eng = UPDEEngine(n, dt=0.01)
    alpha = np.zeros((n, n))
    phases = rng.uniform(0, TWO_PI, n)

    print("Multi-Agent AI Coordination as Synchronisation")
    print("=" * 55)

    # Phase 1: Normal parallel work
    print("\nPhase 1: Coordinated parallel work")
    for epoch in range(5):
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        status = "harmony" if R > 0.6 else "drifting" if R > 0.3 else "CONFLICT"
        print(f"  t={epoch + 1}: R={R:.3f} [{status}]")

    # Phase 2: Agent conflict — two agents work on the same file
    print("\nPhase 2: CONFLICT — Agent-B and Agent-C overlap on same module")
    knm_conflict = knm.copy()
    knm_conflict[1, 2] = -1.0  # anti-coupling: competing changes
    knm_conflict[2, 1] = -1.0

    for epoch in range(5, 10):
        for _ in range(100):
            phases = eng.step(phases, omegas, knm_conflict, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        status = "harmony" if R > 0.6 else "drifting" if R > 0.3 else "CONFLICT"
        print(f"  t={epoch + 1}: R={R:.3f} [{status}]")

    # Phase 3: Supervisor intervenes — redistribute tasks
    print("\nPhase 3: Supervisor redistributes (task lock + coupling boost)")
    knm_resolved = knm.copy()
    knm_resolved[1, 2] = 0.0  # decouple competing agents
    knm_resolved[2, 1] = 0.0
    knm_resolved *= 2.0  # boost remaining links

    for epoch in range(10, 15):
        for _ in range(100):
            phases = eng.step(phases, omegas, knm_resolved, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        status = "harmony" if R > 0.6 else "drifting" if R > 0.3 else "CONFLICT"
        print(f"  t={epoch + 1}: R={R:.3f} [{status}]")

    print("\nAgent phases (work rhythm alignment):")
    for i, agent in enumerate(agents):
        print(f"  {agent}: {np.degrees(phases[i]):.0f} deg")

    print("\nWhen R drops, agents are stepping on each other.")
    print("The supervisor (SYNAPSE_CHANNEL hub) redistributes tasks.")
    print("This IS what SYNAPSE_CHANNEL does — modelled as phase dynamics.")


if __name__ == "__main__":
    main()
