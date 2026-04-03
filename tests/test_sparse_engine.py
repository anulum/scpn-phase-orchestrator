# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse UPDE engine tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


def dense_to_csr(knm, alpha):
    n = knm.shape[0]
    row_ptr = [0]
    col_indices = []
    knm_values = []
    alpha_values = []

    for i in range(n):
        for j in range(n):
            if knm[i, j] != 0:
                col_indices.append(j)
                knm_values.append(knm[i, j])
                alpha_values.append(alpha[i, j])
        row_ptr.append(len(col_indices))

    return (
        np.array(row_ptr, dtype=np.uint64),
        np.array(col_indices, dtype=np.uint64),
        np.array(knm_values, dtype=np.float64),
        np.array(alpha_values, dtype=np.float64),
    )


class TestSparseUPDEEngine:
    def test_compare_with_dense(self):
        n = 4
        dt = 0.01
        engine_dense = UPDEEngine(n, dt=dt, method="euler")
        engine_sparse = SparseUPDEEngine(n, dt=dt, method="euler")

        phases = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64)

        knm = np.array(
            [
                [0.0, 0.5, 0.0, 0.1],
                [0.5, 0.0, 0.2, 0.0],
                [0.0, 0.2, 0.0, 0.3],
                [0.1, 0.0, 0.3, 0.0],
            ],
            dtype=np.float64,
        )

        alpha = np.zeros((n, n), dtype=np.float64)
        zeta = 0.2
        psi = 0.0

        # Dense step
        p_dense = engine_dense.step(phases, omegas, knm, zeta, psi, alpha)

        # Sparse step
        row_ptr, col_indices, knm_values, alpha_values = dense_to_csr(knm, alpha)
        p_sparse = engine_sparse.step(
            phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values
        )

        np.testing.assert_allclose(p_dense, p_sparse, atol=1e-12)

    def test_run_sparse(self):
        n = 8
        dt = 0.01
        engine = SparseUPDEEngine(n, dt=dt, method="rk45")

        phases = np.zeros(n, dtype=np.float64)
        omegas = np.ones(n, dtype=np.float64)

        # Sparse ring topology
        knm = np.zeros((n, n))
        for i in range(n):
            knm[i, (i + 1) % n] = 0.5
            knm[i, (i - 1) % n] = 0.5

        alpha = np.zeros((n, n))
        row_ptr, col_indices, knm_values, alpha_values = dense_to_csr(knm, alpha)

        # Run 100 steps
        p_final = engine.run(
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            0.0,
            0.0,
            alpha_values,
            100,
        )

        assert len(p_final) == n
        assert np.all(p_final >= 0)
        assert np.all(p_final < 2 * np.pi)

    def test_sparse_plasticity(self):
        n = 4
        dt = 0.01
        engine = SparseUPDEEngine(n, dt=dt, method="euler")

        # Enable plasticity
        if hasattr(engine._rust, "set_plasticity"):
            engine._rust.set_plasticity(lr=1.0, decay=0.0, modulator=1.0)
        else:
            pytest.skip("Rust plasticity not available")

        phases = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(n, dtype=np.float64)

        # Initial sparse coupling (weak)
        knm = np.zeros((n, n), dtype=np.float64)
        knm[0, 1] = 0.1
        alpha = np.zeros((n, n))
        row_ptr, col_indices, knm_values, alpha_values = dense_to_csr(knm, alpha)

        # Step
        engine.step(
            phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
        )

        # knm_values[0] (which is knm[0,1]) should have increased
        # delta = lr * mod * cos(th_j - th_i) * dt
        # delta = cos(0.1) * 0.01 ≈ 0.00995
        # new_knm = 0.1 + 0.00995 = 0.10995
        assert knm_values[0] > 0.1
        assert abs(knm_values[0] - (0.1 + np.cos(0.1) * 0.01)) < 1e-6
