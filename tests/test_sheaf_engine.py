# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cellular Sheaf Engine tests

"""Verify public sheaf equations, elapsed intervals and refusal recovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, cast

import numpy as np
import pytest
from numpy.typing import ArrayLike, DTypeLike
from scipy.integrate import solve_ivp

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine


class SheafConfig(TypedDict, total=False):
    """Constructor fields used to preserve malformed runtime configurations."""

    n_oscillators: int
    d_dimensions: int
    dt: float
    method: str


class TestSheafUPDEEngine:
    """Public vector-phase integration and scalar-engine interoperability."""

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    @pytest.mark.parametrize("dt", [0.01, 1e-14])
    def test_repeated_steps_and_batch_have_exact_elapsed_time(
        self, method: str, dt: float
    ) -> None:
        """Every successful call advances dt, independently of adaptive proposals."""
        phase = np.zeros((2, 2))
        omega = np.array([[1.0, 0.2], [0.5, 1.2]])
        maps = np.zeros((2, 2, 2, 2))
        psi = np.zeros(2)
        engine = SheafUPDEEngine(2, 2, dt, method=method)
        current = phase.copy()
        for _ in range(7):
            current = engine.step(current, omega, maps, 0.0, psi)
        expected = 7 * dt * omega
        np.testing.assert_allclose(current, expected, atol=dt * 1e-13, rtol=0)
        np.testing.assert_allclose(
            engine.run(phase, omega, maps, 0.0, psi, 7),
            expected,
            atol=dt * 1e-13,
            rtol=0,
        )
        assert 0.0 < engine.last_dt <= dt

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_concurrent_batches_preserve_independent_equations(
        self, method: str
    ) -> None:
        """Shared solver scratch state cannot contaminate independent threaded calls."""
        engine = SheafUPDEEngine(2, 2, 0.01, method=method)
        omega = np.array([[1.0, 0.2], [0.5, 1.2]])
        maps = np.zeros((2, 2, 2, 2))
        psi = np.zeros(2)

        def advance(offset: float) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            """Run the real public batch with one thread's independent initial state."""
            return engine.run(np.full((2, 2), offset), omega, maps, 0.0, psi, 20)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(advance, [0.1, 0.2, 0.3, 0.4]))
        for offset, result in zip([0.1, 0.2, 0.3, 0.4], results, strict=True):
            np.testing.assert_allclose(result, offset + 0.2 * omega, atol=1e-13, rtol=0)

    @pytest.mark.parametrize(("n", "d"), [(2**32, 1), (1, 2**32), (2**64, 2)])
    def test_overflowing_restriction_geometry_refuses_before_runtime_allocation(
        self, n: int, d: int
    ) -> None:
        """Both runtimes reject overflowing restriction-tensor cardinality."""
        with pytest.raises(ValueError, match="geometry overflows"):
            SheafUPDEEngine(n, d, 0.01)

    def test_compare_with_dense_1d(self) -> None:
        # A Sheaf with D=1 should be mathematically identical to the scalar UPDEEngine
        """D=1 Euler matches the wired scalar engine with nonzero graph and drive."""
        n = 4
        dt = 0.01

        engine_dense = UPDEEngine(n, dt=dt, method="euler")
        engine_sheaf = SheafUPDEEngine(n, d_dimensions=1, dt=dt, method="euler")

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

    def test_run_sheaf_2d(self) -> None:
        """Cross-frequency RK45 matches an independent one-second trajectory."""
        n = 4
        d = 2
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="rk45")

        phases = np.zeros((n, d), dtype=np.float64)
        phases[:, 1] = 0.3
        omegas = np.ones((n, d), dtype=np.float64)
        omegas[:, 1] = 0.7
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)

        # Cross-frequency coupling: dim 0 of node j drives dim 1 of node i
        for i in range(n):
            for j in range(n):
                if i != j:
                    restriction_maps[i, j, 1, 0] = 0.5

        psi = np.zeros(d, dtype=np.float64)

        # Run 100 steps
        p_final = engine.run(phases, omegas, restriction_maps, 0.0, psi, 100)

        reference = solve_ivp(
            lambda _t, y: np.array([1.0, 0.7 + 1.5 * np.sin(y[0] - y[1])]),
            (0.0, 1.0),
            np.array([0.0, 0.3]),
            method="DOP853",
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            p_final, np.tile(reference.y[:, -1], (n, 1)), atol=3e-6, rtol=0
        )


class TestSheafEngineEdgeCases:
    """Edge cases and error paths a prior audit flagged as missing."""

    @pytest.mark.parametrize(
        ("dt", "omega", "zeta", "tolerance", "match"),
        [
            (0.1, 1.0, 1.0, 5e-324, "non-finite"),
            (5e-324, 1e308, 0.0, 1e-320, "cannot advance"),
            (1e16, 0.0, 1.0, 1e-12, "cannot advance"),
            (1e300, 0.0, 1.0, 1e-300, "rejection limit"),
            (0.01, 200.0, 0.0, 1e308, "non-finite"),
        ],
    )
    @pytest.mark.parametrize("operation", ["step", "run"])
    def test_real_adaptive_limits_refuse_and_recover(
        self,
        dt: float,
        omega: float,
        zeta: float,
        tolerance: float,
        match: str,
        operation: str,
    ) -> None:
        """Extreme real controls expose estimator, proposal and progress limits."""
        engine = SheafUPDEEngine(
            1, 1, dt, method="rk45", atol=tolerance, rtol=tolerance
        )
        phases = np.array([[0.1]])
        maps = np.zeros((1, 1, 1, 1))
        psi = np.ones(1)
        with pytest.raises(ValueError, match=match):
            if operation == "step":
                engine.step(phases, np.full((1, 1), omega), maps, zeta, psi)
            else:
                engine.run(phases, np.full((1, 1), omega), maps, zeta, psi, 2)
        assert engine.last_dt == dt
        np.testing.assert_array_equal(phases, [[0.1]])
        np.testing.assert_allclose(
            engine.step(phases, np.zeros((1, 1)), maps, 0.0, psi),
            phases,
            atol=1e-15,
            rtol=0,
        )

    def test_rk45_long_forced_interval_refuses_at_work_limit_and_recovers(
        self,
    ) -> None:
        """A genuine long driven trajectory reaches the bounded adaptive work limit."""
        engine = SheafUPDEEngine(1, 1, 1e8, method="rk45", atol=1e-12, rtol=1e-12)
        phases = np.zeros((1, 1))
        maps = np.zeros((1, 1, 1, 1))
        psi = np.zeros(1)
        with pytest.raises(ValueError, match="substep limit"):
            engine.step(phases, np.full((1, 1), 2.0), maps, 1.0, psi)
        np.testing.assert_array_equal(phases, [[0.0]])
        assert engine.last_dt == 1e8
        np.testing.assert_array_equal(
            engine.step(phases, np.zeros((1, 1)), maps, 0.0, psi), [[0.0]]
        )

    def test_rk45_rejects_relative_tolerance_below_absolute(self) -> None:
        """Both runtimes enforce the shared adaptive tolerance order."""
        with pytest.raises(ValueError, match="rtol must be >= atol"):
            SheafUPDEEngine(1, 1, 0.01, method="rk45", atol=1e-2, rtol=1e-3)

    @pytest.mark.parametrize(
        "count", [True, "1", 1.5, -1, np.bool_(True), 2**64, 10**400]
    )
    def test_batch_count_refuses_aliases_and_native_overflow(
        self, count: object
    ) -> None:
        """Both runtimes reject malformed counts before consuming solver state."""
        engine = SheafUPDEEngine(1, 1, 0.01, method="rk45")
        phases = np.array([[0.2]])
        frequency = np.ones((1, 1))
        maps = np.zeros((1, 1, 1, 1))
        psi = np.zeros(1)
        with pytest.raises(ValueError, match="n_steps"):
            engine.run(phases, frequency, maps, 0.0, psi, cast(int, count))
        assert engine.last_dt == 0.01
        np.testing.assert_array_equal(phases, [[0.2]])
        np.testing.assert_allclose(
            engine.run(phases, frequency, maps, 0.0, psi, 2), [[0.22]], atol=1e-15
        )

    def test_unrepresentable_drive_refuses_before_runtime_dispatch(self) -> None:
        """A genuine integer drive beyond float64 refuses without coercion overflow."""
        engine = SheafUPDEEngine(1, 1, 0.01)
        with pytest.raises(ValueError, match="zeta must be finite real"):
            engine.step(
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1, 1, 1)),
                10**400,
                np.zeros(1),
            )
        assert engine.last_dt == 0.01

    def test_late_batch_overflow_preserves_original_phase_and_recovers(self) -> None:
        """A valid first increment cannot partially publish a failed batch."""
        angle = float(np.remainder(1e308, 2 * np.pi) + np.pi / 2)
        phases = np.array([[angle]])
        psi = np.array([angle])
        maps = np.zeros((1, 1, 1, 1))
        engine = SheafUPDEEngine(1, 1, 1.0)
        first = engine.step(phases, np.full((1, 1), 1e308), maps, 1e308, psi)
        np.testing.assert_allclose(
            first, [[np.remainder(1e308, 2 * np.pi)]], atol=1e-15
        )
        with pytest.raises(ValueError):
            engine.run(phases, np.full((1, 1), 1e308), maps, 1e308, psi, 2)
        np.testing.assert_array_equal(phases, [[angle]])
        assert engine.last_dt == 1.0
        np.testing.assert_allclose(
            engine.run(phases, np.full((1, 1), 0.1), maps, 0.0, psi, 2),
            (phases + 0.2) % (2 * np.pi),
            atol=1e-14,
            rtol=0,
        )

    def test_zero_restriction_maps_decouple_oscillators(self) -> None:
        """Empty restriction maps leave each component's frequency and drive."""
        n = 3
        d = 2
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="euler")

        phases = np.zeros((n, d), dtype=np.float64)
        omegas = np.full((n, d), 0.5, dtype=np.float64)
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        psi = np.zeros(d, dtype=np.float64)

        p = engine.step(phases, omegas, restriction_maps, 0.0, psi)
        np.testing.assert_allclose(p, 0.5 * dt * np.ones((n, d)), atol=1e-12)

    def test_run_rejects_shape_mismatch(self) -> None:
        """Wrong phase geometry refuses the public batch entry point."""
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.01, method="euler")
        phases = np.zeros(2, dtype=np.float64)
        omegas = np.ones((2, 2), dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        with pytest.raises(ValueError, match="phases.shape"):
            engine.run(phases, omegas, restriction_maps, 0.0, psi, 1)

    def test_run_rejects_non_finite_inputs(self) -> None:
        """Nonfinite frequencies refuse before a real batch advances."""
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.01, method="euler")
        phases = np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float64)
        omegas = np.array([[1.0, 0.0], [np.inf, 0.2]], dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        with pytest.raises(ValueError, match="omegas contains NaN/Inf"):
            engine.run(phases, omegas, restriction_maps, 0.0, psi, 1)

    def test_run_with_empty_restriction_maps_decouples(self) -> None:
        """Uncoupled multidimensional batch follows exact omega times elapsed time."""
        n = 3
        d = 3
        dt = 0.02
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="euler")
        phases = np.array(
            [[0.0, 0.5, 1.0], [0.2, 0.7, 1.2], [0.4, 0.9, 1.4]],
            dtype=np.float64,
        )
        omegas = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            dtype=np.float64,
        )
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        psi = np.zeros(d, dtype=np.float64)
        n_steps = 4

        out = engine.run(phases, omegas, restriction_maps, 0.0, psi, n_steps)
        expected = (phases + n_steps * dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_d_dimensions_one_matches_scalar_engine_over_many_steps(self) -> None:
        """D=1 sheaf tracks the scalar UPDEEngine across 50 real steps."""
        n = 5
        dt = 0.01
        rng = np.random.default_rng(13)
        dense = UPDEEngine(n, dt=dt, method="rk4")
        sheaf = SheafUPDEEngine(n, d_dimensions=1, dt=dt, method="rk4")

        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.uniform(0.9, 1.1, n)
        knm = 0.25 * (np.ones((n, n)) - np.eye(n))
        alpha = np.zeros((n, n))

        p_dense = phases.copy()
        p_sheaf = phases.reshape(n, 1).copy()
        omegas_d = omegas.reshape(n, 1)
        restrict = np.zeros((n, n, 1, 1))
        restrict[:, :, 0, 0] = knm
        psi = np.zeros(1)

        for _ in range(50):
            p_dense = dense.step(p_dense, omegas, knm, 0.0, 0.0, alpha)
            p_sheaf = sheaf.step(p_sheaf, omegas_d, restrict, 0.0, psi)
        np.testing.assert_allclose(p_dense, p_sheaf.flatten(), atol=1e-7)

    def test_external_drive_is_applied_per_dimension(self) -> None:
        """ζ·sin(Ψ_d − θ_d) must drive each dimension's phases toward Ψ_d."""
        n = 3
        d = 2
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="rk4")

        phases = np.zeros((n, d), dtype=np.float64)
        omegas = np.zeros((n, d), dtype=np.float64)
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        psi = np.array([1.0, -1.0], dtype=np.float64)
        zeta = 0.5

        p = engine.run(phases, omegas, restriction_maps, zeta, psi, 200)
        # With no intrinsic drift and a strong attractor, dim 0 → +1,
        # dim 1 → −1 (wrapped into [0, 2π)).
        # Values clamped to [0, 2π) so θ ≈ 1 stays as ~1 and θ ≈ -1 wraps to ~2π-1.
        assert np.all(p[:, 0] > 0.5) and np.all(p[:, 0] < 1.5)
        assert np.all((p[:, 1] > 2 * np.pi - 1.5) & (p[:, 1] < 2 * np.pi - 0.5))

    def test_single_oscillator_multi_dim(self) -> None:
        """N=1, D=3: no neighbours, so each dimension evolves under ω·dt."""
        n = 1
        d = 3
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="euler")

        phases = np.zeros((n, d), dtype=np.float64)
        omegas = np.array([[0.5, 1.0, 1.5]], dtype=np.float64)
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        psi = np.zeros(d, dtype=np.float64)

        p = engine.step(phases, omegas, restriction_maps, 0.0, psi)
        np.testing.assert_allclose(p[0], [0.005, 0.01, 0.015], atol=1e-12)

    def test_output_bounded_to_unit_circle(self) -> None:
        """After ``run`` every phase must live in [0, 2π) — wrap contract."""
        n = 4
        d = 2
        dt = 0.05
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="rk4")

        rng = np.random.default_rng(88)
        phases = rng.uniform(0, 2 * np.pi, (n, d))
        omegas = rng.uniform(1.0, 5.0, (n, d))  # strong ω to force wrapping
        restriction_maps = np.zeros((n, n, d, d))
        for i in range(n):
            for j in range(n):
                if i != j:
                    restriction_maps[i, j] = 0.1 * np.eye(d)
        psi = np.zeros(d)

        p = engine.run(phases, omegas, restriction_maps, 0.0, psi, 300)
        assert np.all(p >= 0)
        assert np.all(p < 2 * np.pi)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_oscillators": 0, "d_dimensions": 2, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": True, "d_dimensions": 2, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": 2, "d_dimensions": 0, "dt": 0.01}, "d_dimensions"),
            ({"n_oscillators": 2, "d_dimensions": 2, "dt": False}, "dt"),
            ({"n_oscillators": 2, "d_dimensions": 2, "dt": np.inf}, "dt"),
            (
                {"n_oscillators": 2, "d_dimensions": 2, "dt": 0.01, "method": "heun"},
                "Unknown method",
            ),
        ],
    )
    def test_constructor_rejects_invalid_configuration(
        self, kwargs: SheafConfig, match: str
    ) -> None:
        """Public construction rejects malformed geometry, timestep and method."""
        with pytest.raises(ValueError, match=match):
            SheafUPDEEngine(**kwargs)

    def test_last_dt_reports_configured_python_timestep(self) -> None:
        """Fixed-step proposal agrees with the real uncoupled phase increment."""
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.0125, method="rk4")
        assert engine.last_dt == pytest.approx(0.0125)
        out = engine.step(
            np.zeros((2, 2)), np.ones((2, 2)), np.zeros((2, 2, 2, 2)), 0.0, np.zeros(2)
        )
        np.testing.assert_allclose(
            out, np.full((2, 2), engine.last_dt), atol=1e-15, rtol=0
        )

    def test_available_runtime_executes_uncoupled_equation(self) -> None:
        """Installed native and genuinely absent runs both advance the same equation.

        Partial-wheel ImportError is not reachable with these two real runtimes.
        No incompatible released wheel is available here; no synthetic module
        is inserted to manufacture constructor-branch coverage.
        """
        engine = SheafUPDEEngine(2, 1, 0.01)
        phases = np.array([[0.1], [0.2]])
        omegas = np.array([[1.0], [1.5]])
        maps = np.zeros((2, 2, 1, 1))
        out = engine.step(phases, omegas, maps, 0.0, np.zeros(1))
        np.testing.assert_allclose(out, phases + 0.01 * omegas, atol=1e-12)
        np.testing.assert_array_equal(phases, [[0.1], [0.2]])

    def test_runtime_preserves_row_major_phase_geometry(self) -> None:
        """Real forced RK4 preserves distinct node/dimension coordinates."""
        engine = SheafUPDEEngine(2, 2, 0.01, method="rk4")
        phases = np.array([[0.0, 0.1], [0.2, 0.3]])
        omegas = np.array([[1.0, 1.5], [0.5, -0.2]])
        maps = np.zeros((2, 2, 2, 2))
        psi = np.array([0.0, 1.0])
        expected = solve_ivp(
            lambda _t, state: omegas.ravel() + 0.25 * np.sin(np.tile(psi, 2) - state),
            (0.0, 0.05),
            phases.ravel(),
            method="DOP853",
            atol=1e-13,
            rtol=1e-13,
            t_eval=[0.01, 0.05],
        ).y
        np.testing.assert_allclose(
            engine.step(phases, omegas, maps, 0.25, psi),
            expected[:, 0].reshape(2, 2) % (2 * np.pi),
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            engine.run(phases, omegas, maps, 0.25, psi, 5),
            expected[:, 1].reshape(2, 2) % (2 * np.pi),
            atol=1e-10,
            rtol=0,
        )
        assert engine.last_dt == 0.01

    def test_step_rejects_malformed_shapes_and_non_finite_values(self) -> None:
        """Real inputs refuse wrong shapes and nonfinite controls."""
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.01, method="euler")
        phases = np.zeros((2, 2), dtype=np.float64)
        omegas = np.ones((2, 2), dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        with pytest.raises(ValueError, match="restriction_maps.shape"):
            engine.step(phases, omegas, restriction_maps[:, :, :, :1], 0.0, psi)

        bad_phases = phases.copy()
        bad_phases[0, 0] = np.nan
        with pytest.raises(ValueError, match="phases contains NaN/Inf"):
            engine.step(bad_phases, omegas, restriction_maps, 0.0, psi)

        with pytest.raises(ValueError, match="zeta must be finite"):
            engine.step(phases, omegas, restriction_maps, np.inf, psi)

        with pytest.raises(ValueError, match="zeta must be finite real"):
            engine.step(phases, omegas, restriction_maps, True, psi)

        with pytest.raises(ValueError, match="psi.shape"):
            engine.step(phases, omegas, restriction_maps, 0.0, np.zeros((2, 1)))

        with pytest.raises(ValueError, match="omegas contains NaN/Inf"):
            engine.step(
                phases,
                np.array([[1.0, 2.0], [3.0, np.inf]]),
                restriction_maps,
                0.0,
                psi,
            )

    def test_step_accepts_numeric_array_like_inputs(self) -> None:
        """Nested Python sequences advance the real uncoupled Euler equation."""
        engine = SheafUPDEEngine(2, d_dimensions=1, dt=0.01, method="euler")
        out = engine.step(
            [[0.0], [0.1]],
            [[1.0], [2.0]],
            [[[[0.0]], [[0.0]]], [[[0.0]], [[0.0]]]],
            0,
            [0.0],
        )

        np.testing.assert_allclose(out, [[0.01], [0.12]], atol=1e-12)

    @pytest.mark.parametrize(
        "field",
        ["phases", "omegas", "restriction_maps", "psi"],
    )
    @pytest.mark.parametrize(
        ("alias", "dtype", "match"),
        [
            (True, None, "real-valued, not boolean"),
            (np.timedelta64(1, "s"), None, "must be numeric"),
            (np.datetime64("2026-09-30"), None, "must be numeric"),
            ("0.1", None, "must be numeric"),
            (0.1 + 0.2j, None, "real-valued, not complex"),
            (np.bool_(True), object, "real-valued, not boolean"),
            (0.1 + 0.2j, object, "real-valued, not complex"),
        ],
    )
    def test_step_rejects_coercive_array_aliases(
        self, field: str, alias: object, dtype: DTypeLike, match: str
    ) -> None:
        """Boolean, string and complex arrays refuse without numeric coercion."""
        engine = SheafUPDEEngine(2, d_dimensions=1, dt=0.01, method="euler")
        inputs = {
            "phases": np.array([[0.0], [0.1]], dtype=np.float64),
            "omegas": np.array([[1.0], [2.0]], dtype=np.float64),
            "restriction_maps": np.zeros((2, 2, 1, 1), dtype=np.float64),
            "psi": np.array([0.0], dtype=np.float64),
        }
        inputs[field] = np.full(inputs[field].shape, alias, dtype=dtype)

        with pytest.raises(ValueError, match=match):
            engine.step(
                inputs["phases"],
                inputs["omegas"],
                inputs["restriction_maps"],
                0.0,
                inputs["psi"],
            )

    @pytest.mark.parametrize("field", ["phases", "omegas", "restriction_maps", "psi"])
    @pytest.mark.parametrize("operation", ["step", "run", "zero"])
    def test_mixed_boolean_sequences_refuse_and_recover(
        self, field: str, operation: str
    ) -> None:
        """Nested lists retain boolean source values before numeric promotion."""
        inputs: dict[str, ArrayLike] = {
            "phases": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "omegas": np.ones((2, 2)),
            "restriction_maps": np.zeros((2, 2, 2, 2)),
            "psi": np.zeros(2),
        }
        original = inputs[field]
        mixed = np.asarray(original, dtype=object)
        mixed.flat[0] = True
        inputs[field] = mixed.tolist()
        engine = SheafUPDEEngine(2, 2, 0.01, method="rk45")
        with pytest.raises(ValueError, match="not boolean"):
            if operation == "step":
                engine.step(
                    inputs["phases"],
                    inputs["omegas"],
                    inputs["restriction_maps"],
                    0.0,
                    inputs["psi"],
                )
            else:
                engine.run(
                    inputs["phases"],
                    inputs["omegas"],
                    inputs["restriction_maps"],
                    0.0,
                    inputs["psi"],
                    0 if operation == "zero" else 2,
                )
        assert engine.last_dt == 0.01
        inputs[field] = original
        result = engine.run(
            inputs["phases"],
            inputs["omegas"],
            inputs["restriction_maps"],
            0.0,
            inputs["psi"],
            2,
        )
        np.testing.assert_allclose(result, [[0.12, 0.22], [0.32, 0.42]], atol=1e-15)

    def test_step_accepts_real_numeric_object_arrays(self) -> None:
        """Real object arrays normalise to float64 before the phase equation."""
        engine = SheafUPDEEngine(2, d_dimensions=1, dt=0.01, method="euler")
        out = engine.step(
            np.array([[0], [0.1]], dtype=object),
            np.array([[1], [2.0]], dtype=object),
            np.zeros((2, 2, 1, 1), dtype=object),
            0.0,
            np.array([0], dtype=object),
        )

        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [[0.01], [0.12]], atol=1e-12)

    @pytest.mark.parametrize(
        "phases",
        [
            [[0.0], [0.1, 0.2]],
            np.array([[10**400], [0]], dtype=object),
        ],
    )
    def test_step_rejects_unrepresentable_numeric_inputs(
        self, phases: ArrayLike
    ) -> None:
        """Ragged phases and unrepresentable integers refuse public admission."""
        engine = SheafUPDEEngine(2, d_dimensions=1, dt=0.01, method="euler")

        with pytest.raises(ValueError, match="phases must be a numeric array"):
            engine.step(
                phases,
                np.ones((2, 1)),
                np.zeros((2, 2, 1, 1)),
                0.0,
                np.zeros(1),
            )

    def test_run_zero_steps_returns_independent_copy(self) -> None:
        """Zero-step batches validate and copy without sharing input storage."""
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.01, method="rk4")
        phases = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        omegas = np.ones((2, 2), dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        psi = np.zeros(2, dtype=np.float64)

        out = engine.run(phases, omegas, restriction_maps, 0.0, psi, 0)
        np.testing.assert_allclose(out, phases, atol=0.0)
        assert not np.shares_memory(out, phases)

    def test_rk45_sheaf_fallback_uses_error_control(self) -> None:
        """Adaptive full-interval dynamics agree with a fine fixed-step reference."""
        n = 3
        d = 2
        dt = 0.4
        phases = np.array(
            [[0.0, 0.4], [1.0, 1.5], [2.0, 2.8]],
            dtype=np.float64,
        )
        omegas = np.array(
            [[1.1, 1.4], [0.9, 1.7], [1.3, 1.0]],
            dtype=np.float64,
        )
        restriction_maps = np.zeros((n, n, d, d), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    restriction_maps[i, j] = np.array(
                        [[0.8, 0.25], [0.35, 0.9]],
                        dtype=np.float64,
                    )
        psi = np.array([0.2, -0.3], dtype=np.float64)
        rk45 = SheafUPDEEngine(
            n,
            d_dimensions=d,
            dt=dt,
            method="rk45",
            atol=1e-12,
            rtol=1e-12,
        )
        rk4 = SheafUPDEEngine(
            n,
            d_dimensions=d,
            dt=dt,
            method="rk4",
            atol=1e-12,
            rtol=1e-12,
        )

        out_rk45 = rk45.step(phases, omegas, restriction_maps, 0.4, psi)
        out_rk4 = rk4.step(phases, omegas, restriction_maps, 0.4, psi)

        reference = SheafUPDEEngine(n, d, dt / 1000, method="rk4").run(
            phases, omegas, restriction_maps, 0.4, psi, 1000
        )
        np.testing.assert_allclose(out_rk45, reference, atol=2e-10, rtol=0)
        assert 0.0 < rk45.last_dt <= dt
        assert np.all(np.isfinite(out_rk45))
        assert np.all(out_rk45 >= 0.0)
        assert np.all(out_rk45 < 2 * np.pi)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(out_rk45, out_rk4, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("operation", ["step", "run"])
    def test_rk45_nonfinite_arithmetic_refuses_and_recovers(
        self, operation: str
    ) -> None:
        """Finite overflowing frequencies cannot publish a fabricated bounded phase."""
        engine = SheafUPDEEngine(1, 1, 10.0, method="rk45")
        phases = np.zeros((1, 1))
        maps = np.zeros((1, 1, 1, 1))
        psi = np.zeros(1)
        with pytest.raises(ValueError):
            if operation == "step":
                engine.step(phases, np.full((1, 1), 1e308), maps, 0.0, psi)
            else:
                engine.run(phases, np.full((1, 1), 1e308), maps, 0.0, psi, 2)
        assert engine.last_dt == 10.0
        np.testing.assert_array_equal(phases, [[0.0]])
        np.testing.assert_allclose(
            engine.step(phases, np.full((1, 1), 0.1), maps, 0.0, psi),
            [[1.0]],
            atol=1e-14,
            rtol=0,
        )

    def test_real_phase_shape_refusal_preserves_state_and_recovers(self) -> None:
        """Malformed public geometry refuses before either real runtime advances.

        Current native Vec<f64>/PyArray1 always has the validated N*D cardinality;
        malformed native output cannot be produced by admitted real inputs.
        Rust refuses non-finite values and canonicalises rounded torus endpoints
        before publishing. Real overflow and crossing cases cover that producer.
        Retain the defensive consumer guard without injecting a fake producer.
        """
        engine = SheafUPDEEngine(2, 2, 0.01)
        phases = np.zeros((2, 2))
        maps = np.zeros((2, 2, 2, 2))
        with pytest.raises(ValueError, match="phases.shape"):
            engine.step(np.zeros(3), np.ones((2, 2)), maps, 0.0, np.zeros(2))
        assert engine.last_dt == 0.01
        np.testing.assert_allclose(
            engine.step(phases, np.ones((2, 2)), maps, 0.0, np.zeros(2)),
            np.full((2, 2), 0.01),
            atol=1e-15,
            rtol=0,
        )

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_real_overflowing_batch_refuses_without_publishing(
        self, method: str
    ) -> None:
        """Overflowing finite input refuses a real batch and retains input/proposal."""
        engine = SheafUPDEEngine(1, 1, 10.0, method=method)
        phases = np.zeros((1, 1))
        maps = np.zeros((1, 1, 1, 1))
        with pytest.raises(ValueError):
            engine.run(phases, np.full((1, 1), 1e308), maps, 0.0, np.zeros(1), 2)
        np.testing.assert_array_equal(phases, [[0.0]])
        assert engine.last_dt == 10.0
        np.testing.assert_allclose(
            engine.run(phases, np.full((1, 1), 0.1), maps, 0.0, np.zeros(1), 2),
            [[2.0]],
            atol=1e-14,
            rtol=0,
        )

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    @pytest.mark.parametrize("operation", ["step", "run"])
    @pytest.mark.parametrize(
        ("phase", "omega"),
        [(-1e-300, 0.0), (0.0, -1e-15), (-2 * np.pi, 0.0), (-4 * np.pi, 0.0)],
    )
    def test_real_rounded_endpoint_and_negative_crossing_return_zero(
        self, method: str, operation: str, phase: float, omega: float
    ) -> None:
        """Near-zero crossings and exact torus multiples return positive zero."""
        engine = SheafUPDEEngine(1, 1, 0.01, method=method)
        phases = np.array([[phase]])
        frequencies = np.array([[omega]])
        maps = np.zeros((1, 1, 1, 1))
        output = (
            engine.step(phases, frequencies, maps, 0.0, np.zeros(1))
            if operation == "step"
            else engine.run(phases, frequencies, maps, 0.0, np.zeros(1), 3)
        )
        np.testing.assert_array_equal(output, [[0.0]])
        assert not np.any(np.signbit(output))
        np.testing.assert_array_equal(phases, [[phase]])
        assert engine.last_dt == 0.01

    @pytest.mark.parametrize("dt", [False, "0.01", 0.0, -0.01, np.inf, 10**400])
    def test_real_constructor_rejects_invalid_timestep(self, dt: object) -> None:
        """Actual configuration admission prevents invalid producer timesteps.

        Cast preserves the deliberately invalid object; it does not coerce it.
        Current successful solver proposals are positive finite f64 values.
        """
        with pytest.raises(ValueError, match="dt must be positive finite"):
            SheafUPDEEngine(2, 2, cast(float, dt))


# Pipeline wiring: SheafUPDEEngine generalises the scalar phase equation to
# multi-dimensional
# phase vectors with matrix-valued restriction maps. The D=1 parity case
# above guarantees backwards compatibility; the higher-D cases pin
# external drive, degenerate connectivity, single-oscillator decoupling
# and the [0, 2π) wrap contract used by the supervisor downstream.
