# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - SINDy tests

import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

def test_sindy_recovery():
    n = 2
    dt = 0.05
    t_max = 20.0
    steps = int(t_max / dt)
    
    omega = np.array([1.5, 2.0])
    K = np.array([[0.0, 0.5], [0.5, 0.0]])
    
    phases = np.zeros((steps, n))
    phases[0] = [0.0, 1.0]
    
    for t in range(1, steps):
        # Euler integration
        d0 = omega[0] + K[0, 1] * np.sin(phases[t-1, 1] - phases[t-1, 0])
        d1 = omega[1] + K[1, 0] * np.sin(phases[t-1, 0] - phases[t-1, 1])
        phases[t] = (phases[t-1] + dt * np.array([d0, d1])) % (2 * np.pi)
        
    sindy = PhaseSINDy(threshold=0.1)
    coeffs = sindy.fit(phases, dt)
    
    # Coeffs is list of arrays
    # node 0: [omega[0], K[0,1]]
    # node 1: [omega[1], K[1,0]]
    
    assert abs(coeffs[0][0] - 1.5) < 0.01
    assert abs(coeffs[0][1] - 0.5) < 0.01
    assert abs(coeffs[1][0] - 2.0) < 0.01
    assert abs(coeffs[1][1] - 0.5) < 0.01
    
    equations = sindy.get_equations()
    assert "1.5000 * 1" in equations[0]
    assert "0.5000 * sin(theta_1 - theta_0)" in equations[0]
