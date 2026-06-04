# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Doppler UPDE quickstart

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde import DopplerEngine, doppler_term

knm = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
velocities = np.array([300.0, -300.0], dtype=np.float64)
shift = doppler_term(velocities, knm, doppler_epsilon=1.0e-12)

engine = DopplerEngine(
    2,
    omega=-shift,
    k_nm=knm,
    alpha=0.0,
    dt=1.0e-3,
    velocities=velocities,
    doppler_epsilon=1.0e-12,
    solver="euler",
    phases=np.array([0.5, 0.0], dtype=np.float64),
)

phases = engine.run(n_steps=2_000)
print("final phases:", phases)
print("doppler correction:", engine.doppler_term)
