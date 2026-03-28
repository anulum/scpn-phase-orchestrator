#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Imprint Memory
#
# Coupling that remembers: oscillators that synchronise frequently
# accumulate an imprint that strengthens their connection permanently.
# Like Hebbian learning, but at the coupling topology level.
#
# Usage: python examples/imprint_memory.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-0.5, 0.5, n)
    knm_base = np.ones((n, n)) * 0.5
    np.fill_diagonal(knm_base, 0.0)
    alpha = np.zeros((n, n))

    eng = UPDEEngine(n, dt=0.01)
    model = ImprintModel(decay_rate=0.005, saturation=3.0)
    state = ImprintState(m_k=np.zeros(n), last_update=0.0)

    phases = rng.uniform(0, TWO_PI, n)

    print("Imprint Memory: Coupling That Remembers")
    print("=" * 50)
    print("Decay rate: 0.005, saturation: 3.0\n")
    print(f"{'Epoch':>6s}  {'R':>6s}  {'m_mean':>7s}  {'m_max':>6s}  {'K_eff':>6s}")
    print("-" * 40)

    for epoch in range(20):
        # Compute exposure from current sync (high R → high exposure)
        R, _ = compute_order_parameter(phases)
        exposure = np.full(n, R)

        # Update imprint
        state = model.update(state, exposure, dt=1.0)

        # Modulate coupling by imprint
        knm = model.modulate_coupling(knm_base, state)

        # Run 50 steps with modulated coupling
        for _ in range(50):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        R, _ = compute_order_parameter(phases)
        m_mean = float(np.mean(state.m_k))
        m_max = float(np.max(state.m_k))
        k_eff = float(knm[knm > 0].mean())

        if epoch % 4 == 0 or epoch == 19:
            print(
                f"{epoch + 1:>6d}  {R:>6.3f}  {m_mean:>7.3f}  "
                f"{m_max:>6.3f}  {k_eff:>6.3f}"
            )

    print(f"\nImprint accumulated: m_mean={np.mean(state.m_k):.3f}")
    print("Coupling strengthened by memory of past synchronisation.")
    print("Remove the stimulus — coupling persists. That's memory.")


if __name__ == "__main__":
    main()
