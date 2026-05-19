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

    def test_run_rejects_invalid_row_ptr_shape(self):
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="row_ptr.shape"):
            engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 1)

    def test_run_rejects_non_finite_inputs(self):
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, float("nan"), 1.2], dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="omegas contains NaN/Inf"):
            engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 1)

    def test_run_with_empty_boundary_decouples(self):
        n = 4
        dt = 0.02
        n_steps = 5
        engine = SparseUPDEEngine(n, dt=dt, method="euler")
        phases = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        omegas = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        out = engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, n_steps)
        expected = (phases + n_steps * dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(out, expected, atol=1e-12)

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

    def test_constructor_rejects_invalid_scalar_parameters(self):
        for value in (0, -1, 0.0, float("nan"), float("inf"), "4", False):
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=value)
        for value in (0.0, -1e-6, float("nan"), float("inf"), "1e-6", False):
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=0.01, atol=value)
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=0.01, rtol=value)

    def test_step_rejects_non_real_zeta_psi(self):
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.zeros(3, dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError):
            engine.step(phases, omegas, row_ptr, col, kv, True, 0.0, av)
        with pytest.raises(ValueError):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 1j, av)

    def test_step_rejects_array_like_inputs_not_strict_numpy_arrays(self):
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = [0.1, 0.2, 0.3]
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = [0, 0, 0, 0]
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)

    def test_rejects_bool_object_and_nonfinite_csr_arrays(self):
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        base_row_ptr = np.array([0, 1, 2, 3], dtype=np.uint64)

        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                np.array([False, False, False, False]),
                np.array([0, 1, 2], dtype=np.uint64),
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                0.0,
                0.0,
                np.array([0.0, 0.1, 0.2], dtype=np.float64),
            )
        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                base_row_ptr,
                np.array([0, 1, "2"], dtype=object),
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                0.0,
                0.0,
                np.array([0.0, 0.1, 0.2], dtype=np.float64),
            )
        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                base_row_ptr,
                np.array([0, 1, 2], dtype=np.uint64),
                np.array([0.1, float("nan"), 0.3], dtype=np.float64),
                0.0,
                0.0,
                np.array([0.0, 0.1, 0.2], dtype=np.float64),
            )

    def test_step_rejects_row_ptr_invariants(self):
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        knm = np.array([0.2], dtype=np.float64)
        av = np.array([0.0], dtype=np.float64)

        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64),
                np.array([0], dtype=np.uint64),
                knm,
                0.0,
                0.0,
                av,
            )
        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                np.array([0, 2, 1, 1], dtype=np.uint64),
                np.array([0], dtype=np.uint64),
                knm,
                0.0,
                0.0,
                av,
            )
        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                np.array([0, 3, 3, 3], dtype=np.uint64),
                np.array([0, 1, 2], dtype=np.uint64),
                knm,
                0.0,
                0.0,
                av,
            )

    def test_step_rejects_col_indices_out_of_bounds(self):
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        kv = np.array([0.1], dtype=np.float64)
        av = np.array([0.0], dtype=np.float64)

        with pytest.raises(ValueError):
            engine.step(
                phases,
                omegas,
                np.array([0, 2, 2, 2], dtype=np.uint64),
                np.array([3], dtype=np.uint64),
                kv,
                0.0,
                0.0,
                av,
            )

    def test_run_zero_steps_returns_copy(self, monkeypatch):
        class FakeSparseStepper:
            def __init__(self, n, dt, method, *, atol, rtol):
                assert (n, dt, method, atol, rtol) == (3, 0.01, "euler", 1e-6, 1e-3)

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
                raise AssertionError("Rust run should not be called for n_steps=0")

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
                return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.PySparseUPDEStepper = FakeSparseStepper
        monkeypatch.setattr(sparse_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        out = engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 0)
        np.testing.assert_allclose(out, phases)
        assert out is not phases

    def test_rust_output_shape_and_finite_validation(self, monkeypatch):
        class FakeSparseStepper:
            def __init__(self, n, dt, method, *, atol, rtol):
                assert (n, dt, method, atol, rtol) == (3, 0.01, "rk4", 1e-6, 1e-3)

            def step(self, *args):
                return np.array([0.1, 0.2], dtype=np.float64)

            def run(self, *args):
                return np.array([0.1, 0.2, float("nan")], dtype=np.float64)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.PySparseUPDEStepper = FakeSparseStepper
        monkeypatch.setattr(sparse_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SparseUPDEEngine(3, dt=0.01, method="rk4")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="Rust output has malformed shape"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        with pytest.raises(ValueError, match="Rust output contains NaN/Inf"):
            engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 3)

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
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        row_ptr = np.array([0, 1, 2, 2], dtype=np.uint64)
        col = np.array([1, 2], dtype=np.uint64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        alpha = np.array([0.0, 0.1], dtype=np.float64)

        step = engine.step(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha)
        run = engine.run(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha, 4)
        np.testing.assert_allclose(step, [0.4, 0.5, 0.6], atol=1e-12)
        np.testing.assert_allclose(run, [0.7, 0.8, 0.9], atol=1e-12)

    def test_step_rejects_malformed_csr_before_dispatch(self):
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 1, 3, 2], dtype=np.uint64)
        col = np.array([1, 2], dtype=np.uint64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        alpha = np.array([0.0, 0.1], dtype=np.float64)

        with pytest.raises(ValueError, match="row_ptr must be monotonic"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)

    def test_step_rejects_undocumented_flattening(self):
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        alpha = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="phases.*shape|one-dimensional"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)

    def test_rk45_sparse_fallback_uses_error_control(self, monkeypatch):
        monkeypatch.setattr(sparse_mod, "_HAS_RUST", False)
        n = 4
        dt = 0.5
        rng = np.random.default_rng(113)
        phases = rng.uniform(0.0, 2 * np.pi, n)
        omegas = rng.uniform(1.0, 2.0, n)
        knm = np.array(
            [
                [0.0, 1.2, 0.8, 0.0],
                [0.7, 0.0, 1.1, 0.5],
                [0.4, 0.9, 0.0, 1.0],
                [0.6, 0.0, 0.3, 0.0],
            ],
            dtype=np.float64,
        )
        alpha = np.where(knm > 0.0, 0.15, 0.0)
        row_ptr, col, kv, av = dense_to_csr(knm, alpha)
        rk45 = SparseUPDEEngine(n, dt=dt, method="rk45", atol=1e-12, rtol=1e-12)
        rk4 = SparseUPDEEngine(n, dt=dt, method="rk4", atol=1e-12, rtol=1e-12)

        out_rk45 = rk45.step(phases, omegas, row_ptr, col, kv, 0.3, -0.2, av)
        out_rk4 = rk4.step(phases, omegas, row_ptr, col, kv, 0.3, -0.2, av)

        assert rk45.last_dt < dt
        assert np.all(np.isfinite(out_rk45))
        assert np.all(out_rk45 >= 0.0)
        assert np.all(out_rk45 < 2 * np.pi)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(out_rk45, out_rk4, atol=1e-12, rtol=1e-12)


# Pipeline wiring: the sparse engine swaps in for UPDEEngine when the
# coupling matrix is O(N²) in allocation but O(N·<k>) in non-zero
# entries. The dense-parity tests above guarantee substitutability —
# if this file turns green, sparse is safe to use as the hot-path
# for large, low-density networks.
