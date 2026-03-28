#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Swarmalator Dynamics
#
# Swarmalators are agents that simultaneously swarm (move in space)
# and oscillate (have internal phase). The phase-spatial coupling J
# controls whether phases cluster by location (static sync),
# form rotating patterns, or decouple entirely.
#
# O'Keeffe et al. 2017, Nature Communications 8:1504.
#
# Usage: python examples/swarmalator_dynamics.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 20
    rng = np.random.default_rng(42)

    omegas = rng.uniform(-0.5, 0.5, n)
    positions = rng.standard_normal((n, 2))
    phases = rng.uniform(0, TWO_PI, n)

    print("Swarmalator Dynamics (N=20)")
    print("=" * 50)

    for J_val in [0.0, 0.5, 1.0, 2.0]:
        eng = SwarmalatorEngine(n, dim=2, dt=0.01, J=J_val)
        pos = positions.copy()
        ph = phases.copy()

        for _ in range(500):
            pos, ph = eng.step(pos, ph, omegas)

        R, _ = compute_order_parameter(ph)
        spatial_spread = np.std(np.linalg.norm(pos, axis=1))
        phase_spatial_corr = abs(np.corrcoef(ph, np.linalg.norm(pos, axis=1))[0, 1])

        regime = "decoupled" if J_val == 0 else "static sync" if R > 0.7 else "active"
        print(
            f"  J={J_val:.1f}: R={R:.3f}, spatial_spread={spatial_spread:.2f}, "
            f"phase-space corr={phase_spatial_corr:.3f} [{regime}]"
        )


if __name__ == "__main__":
    main()
