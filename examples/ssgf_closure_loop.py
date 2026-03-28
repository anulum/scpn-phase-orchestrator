#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: SSGF Cybernetic Closure Loop
#
# The geometry produces the dynamics, the dynamics produce the cost,
# the cost gradient reshapes the geometry. A self-organising loop
# where coupling topology emerges from optimisation — not design.
#
# Usage: python examples/ssgf_closure_loop.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import CyberneticClosure
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, TWO_PI, n)

    carrier = GeometryCarrier(n, z_dim=8, lr=0.05, seed=42)
    closure = CyberneticClosure(carrier, cost_weights=(1.0, 0.5, 0.1, 0.1))

    print("SSGF Cybernetic Closure Loop")
    print("=" * 55)
    print("Geometry → Dynamics → Cost → Gradient → Geometry")
    print()

    W_init = carrier.decode()
    R_init, _ = compute_order_parameter(phases)
    print(f"Initial: R={R_init:.3f}, W_mean={W_init.mean():.3f}")
    hdr = f"{'Step':>5s}  {'Before':>8s}  {'After':>8s}  {'Conv':>5s}"
    print(hdr)
    print("-" * 45)

    W, history = closure.run(phases, n_outer_steps=20)

    for cs in history:
        conv = "yes" if cs.converging else "NO"
        print(
            f"{cs.ssgf_state_step:>5d}  "
            f"{cs.cost_before:>12.4f}  "
            f"{cs.cost_after:>11.4f}  "
            f"{conv:>11s}"
        )

    R_final, _ = compute_order_parameter(phases)
    print(f"\nFinal W_mean: {W.mean():.3f} (was {W_init.mean():.3f})")
    print(f"Cost: {history[0].cost_before:.4f} → {history[-1].cost_after:.4f}")
    print("\nThe coupling topology self-organised via gradient descent on z.")


if __name__ == "__main__":
    main()
