# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — time-varying omega quickstart

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde import UPDEEngine, compute_order_parameter


def main() -> None:
    n = 6
    dt = 1.0e-4
    phases = np.linspace(0.0, 0.5, n, dtype=np.float64)
    omega0 = np.linspace(0.8, 1.2, n, dtype=np.float64)
    drift = np.linspace(-0.02, 0.02, n, dtype=np.float64)
    knm = np.full((n, n), 0.04, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)

    engine = UPDEEngine(
        n,
        dt=dt,
        method="rk4",
        omega=lambda t: omega0 + drift * t,
    )
    final_phases = engine.run(phases, knm=knm, alpha=alpha, n_steps=25)
    r, psi = compute_order_parameter(final_phases)
    print({"R": float(r), "psi": float(psi), "time": engine.time})


if __name__ == "__main__":
    main()
