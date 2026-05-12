# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse UPDE engine tests

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import sparse_engine as sparse_mod
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


class TestSparseEngineEdgeCases:
    """Edge cases and error paths a prior audit flagged as missing."""

    def test_zero_coupling_matches_dense(self):
        """Empty CSR → sparse should reduce to the pure ω·dt Euler step."""
        n = 3
        dt = 0.01
        sparse = SparseUPDEEngine(n, dt=dt, method="euler")
        dense = UPDEEngine(n, dt=dt, method="euler")
        phases = np.array([0.1, 0.2, 0.3])
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))

        row_ptr, col, kv, av = dense_to_csr(knm, alpha)
        p_sparse = sparse.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        p_dense = dense.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(p_sparse, p_dense, atol=1e-12)

    def test_rk45_parity_with_dense(self):
        """Adaptive-step integrator parity across sparse and dense paths."""
        n = 6
        dt = 0.01
        rng = np.random.default_rng(31)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.uniform(0.9, 1.1, n)
        # Sparse pattern: 50% density
        mask = rng.random((n, n)) < 0.5
        knm = np.where(mask, 0.3, 0.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        dense = UPDEEngine(n, dt=dt, method="rk45")
        sparse = SparseUPDEEngine(n, dt=dt, method="rk45")
        row_ptr, col, kv, av = dense_to_csr(knm, alpha)

        p_dense = dense.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)
        p_sparse = sparse.step(phases.copy(), omegas, row_ptr, col, kv, 0.0, 0.0, av)
        np.testing.assert_allclose(p_dense, p_sparse, atol=1e-7)

    def test_fully_dense_matrix_still_matches(self):
        """If every K_ij > 0, sparse must still agree with dense path."""
        n = 5
        dt = 0.01
        rng = np.random.default_rng(5)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.2 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        dense = UPDEEngine(n, dt=dt, method="rk4")
        sparse = SparseUPDEEngine(n, dt=dt, method="rk4")
        row_ptr, col, kv, av = dense_to_csr(knm, alpha)

        for _ in range(30):
            phases = dense.step(phases, omegas, knm, 0.0, 0.0, alpha)
        phases2 = np.zeros(n)  # fresh start not strictly needed; re-init below
        rng = np.random.default_rng(5)
        phases2 = rng.uniform(0, 2 * np.pi, n)
        for _ in range(30):
            phases2 = sparse.step(phases2, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        np.testing.assert_allclose(phases, phases2, atol=1e-7)

    def test_sakaguchi_lag_parity(self):
        """Non-zero α on each edge must propagate through CSR representation."""
        n = 4
        dt = 0.01
        phases = np.array([0.0, 0.4, 0.8, 1.2])
        omegas = np.ones(n)
        knm = np.array(
            [
                [0.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0],
            ]
        )
        alpha = np.where(knm > 0, 0.2, 0.0)

        dense = UPDEEngine(n, dt=dt, method="rk4")
        sparse = SparseUPDEEngine(n, dt=dt, method="rk4")
        row_ptr, col, kv, av = dense_to_csr(knm, alpha)

        p_dense = dense.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)
        p_sparse = sparse.step(phases.copy(), omegas, row_ptr, col, kv, 0.0, 0.0, av)
        np.testing.assert_allclose(p_dense, p_sparse, atol=1e-8)

    def test_invalid_method_rejected(self):
        """Unknown integration method must raise, mirroring UPDEEngine."""
        with pytest.raises((ValueError, Exception)):  # noqa: BLE001
            SparseUPDEEngine(4, dt=0.01, method="midpoint")

    def test_single_oscillator_decouples(self):
        """N=1 has no neighbours — output is pure ω·dt Euler step."""
        dt = 0.01
        engine = SparseUPDEEngine(1, dt=dt, method="euler")
        phases = np.array([0.5])
        omegas = np.array([2.0])
        knm = np.zeros((1, 1))
        alpha = np.zeros((1, 1))
        row_ptr, col, kv, av = dense_to_csr(knm, alpha)
        out = engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        # θ(dt) = (θ(0) + ω·dt) mod 2π = 0.5 + 0.02 = 0.52
        np.testing.assert_allclose(out, [0.52], atol=1e-10)

    def test_last_dt_reports_configured_python_timestep(self):
        engine = SparseUPDEEngine(3, dt=0.0125, method="rk4")
        assert engine.last_dt == pytest.approx(0.0125)

    def test_rust_import_error_falls_back_to_python(self, monkeypatch):
        """If the compatibility flag is true but the sparse Rust class is
        absent, construction must keep the Python fallback usable."""
        fake_spo = types.ModuleType("spo_kernel")
        monkeypatch.setattr(sparse_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SparseUPDEEngine(2, dt=0.01, method="euler")
        assert engine._rust is None

        phases = np.array([0.1, 0.2], dtype=np.float64)
        omegas = np.array([1.0, 1.5], dtype=np.float64)
        row_ptr = np.array([0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        alpha = np.array([], dtype=np.float64)
        out = engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out, phases + 0.01 * omegas, atol=1e-12)

    def test_rust_stepper_dispatches_contiguous_arrays(self, monkeypatch):
        """The optional Rust path receives flattened contiguous CSR arrays and
        its step/run outputs are returned as NumPy arrays without mutation."""

        class FakeSparseStepper:
            def __init__(self, n, dt, method, *, atol, rtol):
                assert (n, dt, method, atol, rtol) == (3, 0.01, "rk4", 1e-6, 1e-3)

            def step(
                self,
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
            ):
                assert phases.flags.c_contiguous
                assert omegas.flags.c_contiguous
                assert row_ptr.dtype == np.uint64
                assert col_indices.dtype == np.uint64
                assert knm_values.flags.c_contiguous
                assert alpha_values.flags.c_contiguous
                assert (zeta, psi) == (0.2, 0.3)
                return np.array([0.4, 0.5, 0.6], dtype=np.float64)

            def run(
                self,
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
                n_steps,
            ):
                assert n_steps == 4
                return np.array([0.7, 0.8, 0.9], dtype=np.float64)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.PySparseUPDEStepper = FakeSparseStepper
        monkeypatch.setattr(sparse_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SparseUPDEEngine(3, dt=0.01, method="rk4")
        phases = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        omegas = np.array([[1.0, 1.1, 1.2]], dtype=np.float64)
        row_ptr = np.array([0, 1, 2, 2], dtype=np.uint64)
        col = np.array([1, 2], dtype=np.uint64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        alpha = np.array([0.0, 0.1], dtype=np.float64)

        step = engine.step(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha)
        run = engine.run(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha, 4)
        np.testing.assert_allclose(step, [0.4, 0.5, 0.6], atol=1e-12)
        np.testing.assert_allclose(run, [0.7, 0.8, 0.9], atol=1e-12)


# Pipeline wiring: the sparse engine swaps in for UPDEEngine when the
# coupling matrix is O(N²) in allocation but O(N·<k>) in non-zero
# entries. The dense-parity tests above guarantee substitutability —
# if this file turns green, sparse is safe to use as the hot-path
# for large, low-density networks.
