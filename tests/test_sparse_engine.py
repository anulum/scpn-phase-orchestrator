# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse UPDE engine tests

"""Real sparse numerical, input-admission and buffer contracts."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import (
    FloatArray,
    IntArray,
    SparseUPDEEngine,
)


def dense_to_csr(
    knm: FloatArray, alpha: FloatArray
) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    """Encode nonzero dense coupling and aligned phase lags as CSR arrays."""
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
    """Equation and coupling behaviour through the actual selected runtime."""

    def test_compare_with_dense(self) -> None:
        """Sparse Euler reproduces the dense coupled equation with external forcing."""
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

    def test_run_sparse(self) -> None:
        """Adaptive batched integration returns finite phases inside the torus."""
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

    def test_public_coupling_remains_fixed(self) -> None:
        """Public sparse construction preserves coupling in both real runtimes."""
        n = 4
        dt = 0.01
        engine = SparseUPDEEngine(n, dt=dt, method="euler")

        # Public construction leaves plasticity disabled in either real runtime.
        # Native plasticity and aliased snapshots have dedicated native tests.

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

        np.testing.assert_array_equal(knm_values, [0.1])


# Casts retain runtime data for inadmissible inputs and the int32 admission
# case, which is broader than the static float64 hint. They never coerce data
# or replace a producer.
class TestSparseEngineEdgeCases:
    """Edge cases and error paths a prior audit flagged as missing."""

    def test_zero_coupling_matches_dense(self) -> None:
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

    def test_run_rejects_invalid_row_ptr_shape(self) -> None:
        """A batch refuses CSR row pointers without one boundary per oscillator."""
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="row_ptr.shape"):
            engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 1)

    def test_run_rejects_non_finite_inputs(self) -> None:
        """A batch refuses non-finite frequencies before integration."""
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([1.0, float("nan"), 1.2], dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="omegas contains NaN/Inf"):
            engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 1)

    def test_run_with_empty_boundary_decouples(self) -> None:
        """An empty graph advances each oscillator by its independent frequency."""
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

    def test_rk45_parity_with_dense(self) -> None:
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

    def test_fully_dense_matrix_still_matches(self) -> None:
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
        rng = np.random.default_rng(5)
        phases2 = rng.uniform(0, 2 * np.pi, n)
        for _ in range(30):
            phases2 = sparse.step(phases2, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        np.testing.assert_allclose(phases, phases2, atol=1e-7)

    def test_sakaguchi_lag_parity(self) -> None:
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

    def test_constructor_rejects_invalid_scalar_parameters(self) -> None:
        """Construction refuses nonpositive or non-finite timesteps and tolerances."""
        for value in (0, -1, 0.0, float("nan"), float("inf"), "4", False):
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=cast(float, value))
        for value in (0.0, -1e-6, float("nan"), float("inf"), "1e-6", False):
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=0.01, atol=cast(float, value))
            with pytest.raises(ValueError):
                SparseUPDEEngine(n_oscillators=4, dt=0.01, rtol=cast(float, value))

    def test_step_rejects_non_real_zeta_psi(self) -> None:
        """External-drive strength and target must be finite real scalars."""
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
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, cast(float, 1j), av)
        with pytest.raises(ValueError, match="zeta"):
            engine.step(phases, omegas, row_ptr, col, kv, float("nan"), 0.0, av)

    def test_step_rejects_array_like_inputs_not_strict_numpy_arrays(self) -> None:
        """List-valued CSR row pointers refuse before native dispatch."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = cast(IntArray, [0, 0, 0, 0])
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)

    def test_step_rejects_non_numpy_phase_arrays(self) -> None:
        """List-valued phases refuse instead of being silently converted."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = cast(FloatArray, [0.1, 0.2, 0.3])
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="phases must be a NumPy ndarray"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)

    def test_step_rejects_wrong_rank_integer_csr_arrays(self) -> None:
        """CSR row pointers must be one-dimensional even with integer dtype."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([[0, 0, 0, 0]], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        av = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="row_ptr must be one-dimensional"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)

    def test_rejects_bool_object_and_nonfinite_csr_arrays(self) -> None:
        """CSR buffers refuse boolean, object and non-finite numeric data."""
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
        with pytest.raises(ValueError, match="phases must be a real numeric ndarray"):
            engine.step(
                np.array([True, False, True], dtype=np.bool_),
                omegas,
                np.array([0, 0, 0, 0], dtype=np.uint64),
                np.array([], dtype=np.uint64),
                np.array([], dtype=np.float64),
                0.0,
                0.0,
                np.array([], dtype=np.float64),
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

    def test_step_rejects_row_ptr_invariants(self) -> None:
        """CSR boundaries refuse negative, nonzero-start and decreasing pointers."""
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
        with pytest.raises(ValueError, match="row_ptr entries must be non-negative"):
            engine.step(
                phases,
                omegas,
                np.array([0, -1, 1, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                knm,
                0.0,
                0.0,
                av,
            )
        with pytest.raises(ValueError, match="row_ptr must start at 0"):
            engine.step(
                phases,
                omegas,
                np.array([1, 1, 1, 1], dtype=np.uint64),
                np.array([0], dtype=np.uint64),
                knm,
                0.0,
                0.0,
                av,
            )

    def test_step_rejects_col_indices_out_of_bounds(self) -> None:
        """Every CSR column must identify an oscillator inside the graph."""
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
        with pytest.raises(ValueError, match="col_indices entries"):
            engine.step(
                phases,
                omegas,
                np.array([0, 1, 1, 1], dtype=np.uint64),
                np.array([3], dtype=np.uint64),
                kv,
                0.0,
                0.0,
                av,
            )

    def test_step_rejects_shape_mismatched_vectors_and_edge_payloads(self) -> None:
        """Phase cardinality and all edge-buffer lengths must match the CSR topology."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 1, 2, 2], dtype=np.uint64)
        col = np.array([1, 2], dtype=np.uint64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        av = np.array([0.0, 0.1], dtype=np.float64)

        with pytest.raises(ValueError, match="phases.shape"):
            engine.step(phases[:2], omegas, row_ptr, col, kv, 0.0, 0.0, av)
        with pytest.raises(ValueError, match="omegas.shape"):
            engine.step(phases, omegas[:2], row_ptr, col, kv, 0.0, 0.0, av)
        with pytest.raises(ValueError, match="knm_values.shape"):
            engine.step(phases, omegas, row_ptr, col, kv[:1], 0.0, 0.0, av)
        with pytest.raises(ValueError, match="alpha_values.shape"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av[:1])

    def test_run_zero_steps_returns_copy(self) -> None:
        """Zero steps return an independent copy through the real runtime."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases, omegas, row_ptr, col, kv, av = _empty_sparse_inputs(3)
        out = engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 0)
        np.testing.assert_array_equal(out, phases)
        assert not np.shares_memory(out, phases)
        np.testing.assert_array_equal(kv, [])

    def test_real_integer_phases_produce_float64_torus_output(self) -> None:
        """Actual runtime conversion retains the uncoupled RK4 phase equation."""
        engine = SparseUPDEEngine(3, dt=0.01, method="rk4")
        # Integer arrays are accepted by the runtime's real-numeric input contract.
        phases = cast(FloatArray, np.array([0, -1, 7], dtype=np.int32))
        omegas = np.array([1.0, 1.1, 1.2])
        rows = np.array([0, 0, 0, 0], dtype=np.int64)
        columns = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        result = engine.step(phases, omegas, rows, columns, values, 0.0, 0.0, values)
        expected = (phases.astype(np.float64) + 0.01 * omegas) % (2 * np.pi)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(phases, [0, -1, 7])

    def test_nonnumeric_input_refusal_leaves_public_solver_usable(self) -> None:
        """Real invalid source types refuse before a subsequent correct Euler step."""
        engine = SparseUPDEEngine(3, dt=0.01)
        phases, omegas, rows, columns, values, lag = _empty_sparse_inputs(3)
        invalid = cast(FloatArray, np.array(["bad", "input", "dtype"], dtype=object))
        with pytest.raises(ValueError, match="phases must be a real numeric"):
            engine.step(invalid, omegas, rows, columns, values, 0.0, 0.0, lag)
        result = engine.step(phases, omegas, rows, columns, values, 0.0, 0.0, lag)
        np.testing.assert_allclose(result, phases + 0.01 * omegas, atol=1e-12, rtol=0.0)

    def test_invalid_method_rejected(self) -> None:
        """Unknown integration method must raise, mirroring UPDEEngine."""
        with pytest.raises(ValueError):
            SparseUPDEEngine(4, dt=0.01, method="midpoint")

    def test_single_oscillator_decouples(self) -> None:
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

    def test_last_dt_reports_configured_python_timestep(self) -> None:
        """A fixed-step solver initially exposes its configured timestep."""
        engine = SparseUPDEEngine(3, dt=0.0125, method="rk4")
        assert engine.last_dt == pytest.approx(0.0125)

    def test_actual_runtime_batch_preserves_uncoupled_equation(self) -> None:
        """Installed and genuinely absent environments execute the same public batch."""
        engine = SparseUPDEEngine(2, dt=0.01)
        phases = np.array([0.1, 0.2])
        omegas = np.array([1.0, 1.5])
        rows = np.array([0, 0, 0], dtype=np.int64)
        columns = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        result = engine.run(phases, omegas, rows, columns, values, 0.0, 0.0, values, 7)
        np.testing.assert_allclose(result, phases + 0.07 * omegas, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(phases, [0.1, 0.2])

    def test_rust_stepper_dispatches_contiguous_arrays(self) -> None:
        """Real step and batch calls retain CSR Euler dynamics on strided inputs."""
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 9.0, 0.2, 9.0, 0.3, 9.0])[::2]
        omegas = np.array([1.0, 9.0, 1.1, 9.0, 1.2, 9.0])[::2]
        row_ptr = np.array([0, 1, 2, 2], dtype=np.int64)
        col = np.array([1, 2], dtype=np.int64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        alpha = np.array([0.0, 0.1], dtype=np.float64)
        expected = phases.copy()
        for _ in range(4):
            derivative = omegas + 0.2 * np.sin(0.3 - expected)
            derivative[:2] += kv * np.sin(expected[col] - expected[:2] - alpha)
            expected = (expected + 0.01 * derivative) % (2 * np.pi)
        result = engine.run(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha, 4)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
        first = omegas + 0.2 * np.sin(0.3 - phases)
        first[:2] += kv * np.sin(phases[col] - phases[:2] - alpha)
        np.testing.assert_allclose(
            engine.step(phases, omegas, row_ptr, col, kv, 0.2, 0.3, alpha),
            (phases + 0.01 * first) % (2 * np.pi),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_array_equal(phases, [0.1, 0.2, 0.3])

    def test_step_rejects_malformed_csr_before_dispatch(self) -> None:
        """Decreasing CSR pointers refuse before either real runtime advances."""
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 1, 3, 2], dtype=np.uint64)
        col = np.array([1, 2], dtype=np.uint64)
        kv = np.array([0.4, 0.5], dtype=np.float64)
        alpha = np.array([0.0, 0.1], dtype=np.float64)

        with pytest.raises(ValueError, match="row_ptr must be monotonic"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)

    def test_step_rejects_undocumented_flattening(self) -> None:
        """Rank-two phases refuse instead of being silently flattened."""
        engine = SparseUPDEEngine(3, dt=0.01, method="euler")
        phases = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        row_ptr = np.array([0, 0, 0, 0], dtype=np.uint64)
        col = np.array([], dtype=np.uint64)
        kv = np.array([], dtype=np.float64)
        alpha = np.array([], dtype=np.float64)

        with pytest.raises(ValueError, match="phases.*shape|one-dimensional"):
            engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)

    def test_rk45_sparse_fallback_uses_error_control(self) -> None:
        """Adaptive error control differs from the large fixed-step RK4 trajectory."""
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

    def test_rk45_sparse_fallback_returns_after_reject_budget(self) -> None:
        """A genuine stiff trajectory returns after exhausting adaptive rejections."""
        engine = SparseUPDEEngine(2, dt=10.0, method="rk45", atol=1e-16, rtol=1e-16)
        phases = np.array([0.1, 2.0], dtype=np.float64)
        omegas = np.array([10.0, -20.0], dtype=np.float64)
        row_ptr = np.array([0, 1, 2], dtype=np.int64)
        col = np.array([1, 0], dtype=np.int64)
        kv = np.array([100.0, 100.0], dtype=np.float64)
        alpha = np.zeros(2, dtype=np.float64)

        out = engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, alpha)

        assert engine.last_dt == pytest.approx(10.0 * (0.2**4))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)
        assert np.all(out < 2 * np.pi)


def _empty_sparse_inputs(
    n: int,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
    FloatArray,
    FloatArray,
]:
    """Return valid empty-graph inputs for optional-Rust contract tests."""
    return (
        np.linspace(0.1, 0.1 * n, n, dtype=np.float64),
        np.ones(n, dtype=np.float64),
        np.zeros(n + 1, dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )


def _call_sparse_operation(
    engine: SparseUPDEEngine,
    operation: str,
    phases: FloatArray,
    omegas: FloatArray,
) -> FloatArray:
    """Advance an actual public solver with a valid empty three-node graph."""
    rows = np.zeros(4, dtype=np.int64)
    columns = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float64)
    if operation == "step":
        return engine.step(phases, omegas, rows, columns, values, 0.0, 0.0, values)
    return engine.run(phases, omegas, rows, columns, values, 0.0, 0.0, values, 1)


@pytest.mark.parametrize("operation", ["step", "run"])
def test_sparse_real_overflow_refuses_nonfinite_output(operation: str) -> None:
    """Finite frequencies can overflow the real integrator and must fail closed."""
    engine = SparseUPDEEngine(3, dt=10.0)
    phases = np.zeros(3)
    frequencies = np.full(3, 1e308)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="Sparse output contains NaN/Inf"),
    ):
        _call_sparse_operation(engine, operation, phases, frequencies)
    assert engine.last_dt == 10.0
    np.testing.assert_array_equal(phases, np.zeros(3))
    recovered = _call_sparse_operation(engine, operation, phases, np.ones(3))
    np.testing.assert_allclose(recovered, np.full(3, 10.0 % (2 * np.pi)), atol=1e-12)


@pytest.mark.parametrize("operation", ["step", "run"])
def test_sparse_real_wrap_rounding_refuses_upper_endpoint(operation: str) -> None:
    """A tiny negative phase can wrap to the excluded upper endpoint."""
    engine = SparseUPDEEngine(3, dt=0.01)
    phases = np.array([-1e-300, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"outside \[0, 2\*pi\)"):
        _call_sparse_operation(engine, operation, phases, np.zeros(3))
    assert engine.last_dt == 0.01
    np.testing.assert_array_equal(phases, [-1e-300, 0.0, 0.0])


@pytest.mark.parametrize("operation", ["step", "run"])
def test_sparse_real_adaptive_underflow_refuses_invalid_last_dt(operation: str) -> None:
    """Real adaptive rejection can underflow its next-step proposal to zero."""
    minimum = float(np.nextafter(0.0, 1.0))
    engine = SparseUPDEEngine(3, minimum, method="rk45", atol=minimum, rtol=minimum)
    phases = np.zeros(3)
    with (
        np.errstate(over="ignore", invalid="ignore", divide="ignore"),
        pytest.raises(ValueError, match="Sparse last_dt"),
    ):
        _call_sparse_operation(engine, operation, phases, np.full(3, 1e308))
    assert engine.last_dt == minimum
    np.testing.assert_array_equal(phases, np.zeros(3))


@pytest.mark.parametrize("operation", ["step", "run"])
def test_sparse_rust_output_publishes_valid_last_dt(operation: str) -> None:
    """Real adaptive calls preserve uncoupled phases and the next-step proposal."""
    engine = SparseUPDEEngine(3, dt=0.01, method="rk45")
    phases, omegas, row_ptr, col, kv, av = _empty_sparse_inputs(3)
    if operation == "step":
        result = engine.step(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av)
        elapsed = 0.01
        next_dt = 0.05
    else:
        result = engine.run(phases, omegas, row_ptr, col, kv, 0.0, 0.0, av, 2)
        elapsed = 0.06
        next_dt = 0.1
    np.testing.assert_allclose(result, phases + elapsed * omegas, atol=1e-12, rtol=0.0)
    assert engine.last_dt == pytest.approx(next_dt)


# Coverage exception: admitted inputs and both real producers always return an
# n-element real numeric array (native Vec<f64> -> PyArray1; Python arithmetic).
# Shape/dtype refusal guards require a corrupt producer and remain uncovered;
# the integer-input contract and overflow/wrap tests exercise the nearest real
# output behaviour. No producer is replaced to manufacture malformed returns.
# These equation and parity checks establish only the exercised sparse contracts.
# They do not establish general substitutability or a deployment scaling budget.


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize("dtype", ["int64", "uint8", "int8"])
def test_sparse_integer_buffers_follow_real_equation(
    method: str, operation: str, dtype: str
) -> None:
    """Admitted integer data retains fractional coupling and forcing terms."""
    initial_values = {"int64": [0, 1], "uint8": [0, 255], "int8": [-128, 127]}[dtype]
    phases = np.array(initial_values, dtype=dtype)
    frequencies = np.array([1, 2], dtype=dtype)
    rows = np.array([0, 1, 2])
    columns = np.array([1, 0])
    coupling = np.array([1, 2], dtype=dtype)
    lags = np.array([0, 0], dtype=dtype)
    dt = 0.001
    engine = SparseUPDEEngine(2, dt, method=method)
    if operation == "step":
        actual = engine.step(
            phases, frequencies, rows, columns, coupling, 0.2, 0.5, lags
        )
    else:
        actual = engine.run(
            phases, frequencies, rows, columns, coupling, 0.2, 0.5, lags, 1
        )

    def derivative(theta: FloatArray) -> FloatArray:
        """Evaluate the two-node coupled phase equation independently."""
        return np.array(
            [
                1.0 + np.sin(theta[1] - theta[0]) + 0.2 * np.sin(0.5 - theta[0]),
                2.0 + 2.0 * np.sin(theta[0] - theta[1]) + 0.2 * np.sin(0.5 - theta[1]),
            ]
        )

    initial = phases.astype(np.float64)
    k1 = derivative(initial)
    if method == "euler":
        expected = initial + dt * k1
    else:
        k2 = derivative(initial + dt * k1 / 2.0)
        k3 = derivative(initial + dt * k2 / 2.0)
        k4 = derivative(initial + dt * k3)
        expected = initial + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    np.testing.assert_allclose(actual, expected % (2.0 * np.pi), rtol=0.0, atol=1e-12)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(phases, initial_values)
    np.testing.assert_array_equal(frequencies, [1, 2])
    np.testing.assert_array_equal(coupling, [1, 2])


def test_sparse_batch_numerical_refusal_restores_entry_diagnostic() -> None:
    """A genuine second-step proposal overflow must retain the pre-run value."""
    initial_dt = 2e307
    phases, omegas, rows, columns, coupling, lags = _empty_sparse_inputs(3)
    original_phases = phases.copy()
    omegas.fill(0.0)
    single = SparseUPDEEngine(3, initial_dt, method="rk45")
    first = single.step(phases, omegas, rows, columns, coupling, 0.0, 0.0, lags)
    np.testing.assert_array_equal(first, phases)
    assert single.last_dt == 1e308

    batch = SparseUPDEEngine(3, initial_dt, method="rk45")
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="Sparse last_dt"),
    ):
        batch.run(phases, omegas, rows, columns, coupling, 0.0, 0.0, lags, 2)
    assert batch.last_dt == initial_dt
    np.testing.assert_array_equal(phases, original_phases)
