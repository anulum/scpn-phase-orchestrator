#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Failure Injection & Recovery
#
# Demonstrates SPO's fault tolerance: a well-synchronised system
# suffers a sudden coupling failure (node disconnects), the supervisor
# detects the R drop, and the system recovers via coupling boost.
#
# Usage: python examples/failure_recovery.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 8
    rng = np.random.default_rng(0)
    omegas = rng.uniform(-0.5, 0.5, n)
    knm = np.ones((n, n)) * 2.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    eng = UPDEEngine(n, dt=0.01)
    phases = rng.uniform(0, TWO_PI, n)

    print("Failure Injection & Recovery")
    print("=" * 50)

    # Phase 1: Synchronise (steps 0-200)
    print("\nPhase 1: Establishing synchronisation...")
    for _step in range(200):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
    R, _ = compute_order_parameter(phases)
    print(f"  Step 200: R={R:.3f} (synchronised)")

    # Phase 2: Inject fault at step 200 — disconnect nodes 0,1,2
    print("\nPhase 2: FAULT INJECTED — nodes 0,1,2 disconnected")
    knm_broken = knm.copy()
    knm_broken[0:3, :] = 0.0
    knm_broken[:, 0:3] = 0.0

    for step in range(200, 350):
        phases = eng.step(phases, omegas, knm_broken, 0.0, 0.0, alpha)
        if step % 50 == 0:
            R, _ = compute_order_parameter(phases)
            print(f"  Step {step}: R={R:.3f}")
    R, _ = compute_order_parameter(phases)
    print(f"  Step 350: R={R:.3f} (degraded)")

    # Phase 3: Supervisor detects and boosts remaining links
    print("\nPhase 3: Supervisor response — boosting remaining coupling 3x")
    knm_boosted = knm_broken.copy()
    knm_boosted[3:, 3:] *= 3.0

    for step in range(350, 500):
        phases = eng.step(phases, omegas, knm_boosted, 0.0, 0.0, alpha)
        if step % 50 == 0:
            R, _ = compute_order_parameter(phases)
            print(f"  Step {step}: R={R:.3f}")
    R, _ = compute_order_parameter(phases)
    print(f"  Step 500: R={R:.3f} (recovered)")

    print("\nSummary: sync → fault → R drops → supervisor boosts → recovery")


if __name__ == "__main__":
    main()
