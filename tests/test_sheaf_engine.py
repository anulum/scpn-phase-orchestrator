# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Cellular Sheaf Engine tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine

class TestSheafUPDEEngine:
    def test_compare_with_dense_1d(self):
        # A Sheaf with D=1 should be mathematically identical to the scalar UPDEEngine
        n = 4
        dt = 0.01
        
        engine_dense = UPDEEngine(n, dt=dt, method='euler')
        engine_sheaf = SheafUPDEEngine(n, d_dimensions=1, dt=dt, method='euler')
        
        phases = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64)
        
        knm = np.array([
            [0.0, 0.5, 0.0, 0.1],
            [0.5, 0.0, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.3],
            [0.1, 0.0, 0.3, 0.0]
        ], dtype=np.float64)
        
        alpha = np.zeros((n, n), dtype=np.float64)
        zeta = 0.2
        psi = 0.0
        
        # Dense step
        p_dense = engine_dense.step(phases, omegas, knm, zeta, psi, alpha)
        
        # Sheaf step
        phases_d = phases.reshape(n, 1)
        omegas_d = omegas.reshape(n, 1)
        restriction_maps = np.zeros((n, n, 1, 1), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                restriction_maps[i, j, 0, 0] = knm[i, j]
                
        psi_d = np.array([psi], dtype=np.float64)
        
        p_sheaf = engine_sheaf.step(phases_d, omegas_d, restriction_maps, zeta, psi_d)
        
        np.testing.assert_allclose(p_dense, p_sheaf.flatten(), atol=1e-12)

    def test_run_sheaf_2d(self):
        n = 4
        d = 2
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method='rk45')
        
        phases = np.zeros((n, d), dtype=np.float64)
        omegas = np.ones((n, d), dtype=np.float64)
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        
        # Cross-frequency coupling: dim 0 of node j drives dim 1 of node i
        for i in range(n):
            for j in range(n):
                if i != j:
                    restriction_maps[i, j, 1, 0] = 0.5
                    
        psi = np.zeros(d, dtype=np.float64)
        
        # Run 100 steps
        p_final = engine.run(phases, omegas, restriction_maps, 0.0, psi, 100)
        
        assert p_final.shape == (n, d)
        assert np.all(p_final >= 0)
        assert np.all(p_final < 2 * np.pi)
