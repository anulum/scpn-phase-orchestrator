# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Cellular Sheaf Engine tests

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import sheaf_engine as sheaf_mod
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine


class TestSheafUPDEEngine:
    def test_compare_with_dense_1d(self):
        # A Sheaf with D=1 should be mathematically identical to the scalar UPDEEngine
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

    def test_run_sheaf_2d(self):
        n = 4
        d = 2
        dt = 0.01
        engine = SheafUPDEEngine(n, d_dimensions=d, dt=dt, method="rk45")

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


class TestSheafEngineEdgeCases:
    """Edge cases and error paths a prior audit flagged as missing."""

    def test_zero_restriction_maps_decouple_oscillators(self):
        """Empty coupling (all zero restriction maps) → each dim evolves
        independently with only ω·dt plus the external drive term."""
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

    def test_d_dimensions_one_matches_scalar_engine_over_many_steps(self):
        """Long-run parity: D=1 sheaf tracks scalar UPDEEngine across 50
        steps (single-step parity is already covered above)."""
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

    def test_external_drive_is_applied_per_dimension(self):
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

    def test_single_oscillator_multi_dim(self):
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

    def test_output_bounded_to_unit_circle(self):
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
    def test_constructor_rejects_invalid_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SheafUPDEEngine(**kwargs)

    def test_last_dt_reports_configured_python_timestep(self):
        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.0125, method="rk4")
        assert engine.last_dt == pytest.approx(0.0125)

    def test_rust_import_error_falls_back_to_python(self, monkeypatch):
        """A missing optional Rust sheaf class must leave the Python path
        available instead of failing construction."""
        fake_spo = types.ModuleType("spo_kernel")
        monkeypatch.setattr(sheaf_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SheafUPDEEngine(2, d_dimensions=1, dt=0.01, method="euler")
        assert engine._rust is None

        phases = np.array([[0.1], [0.2]], dtype=np.float64)
        omegas = np.array([[1.0], [1.5]], dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 1, 1), dtype=np.float64)
        psi = np.zeros(1, dtype=np.float64)
        out = engine.step(phases, omegas, restriction_maps, 0.0, psi)
        np.testing.assert_allclose(out, phases + 0.01 * omegas, atol=1e-12)

    def test_rust_stepper_dispatches_and_reshapes_outputs(self, monkeypatch):
        """The optional Rust path is flattened at the FFI boundary and
        reshaped back to the engine's (N, D) phase matrix."""

        class FakeSheafStepper:
            def __init__(self, n, d, dt, method, *, atol, rtol):
                assert (n, d, dt, method, atol, rtol) == (
                    2,
                    2,
                    0.01,
                    "rk4",
                    1e-6,
                    1e-3,
                )

            def step(self, phases, omegas, restriction_maps, zeta, psi):
                assert phases.flags.c_contiguous
                assert omegas.flags.c_contiguous
                assert restriction_maps.flags.c_contiguous
                assert psi.flags.c_contiguous
                assert zeta == 0.25
                return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

            def run(self, phases, omegas, restriction_maps, zeta, psi, n_steps):
                assert n_steps == 5
                return np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.PySheafUPDEStepper = FakeSheafStepper
        monkeypatch.setattr(sheaf_mod, "_HAS_RUST", True)
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        engine = SheafUPDEEngine(2, d_dimensions=2, dt=0.01, method="rk4")
        phases = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float64)
        omegas = np.ones((2, 2), dtype=np.float64)
        restriction_maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        psi = np.array([0.0, 1.0], dtype=np.float64)

        step = engine.step(phases, omegas, restriction_maps, 0.25, psi)
        run = engine.run(phases, omegas, restriction_maps, 0.25, psi, 5)
        np.testing.assert_allclose(step, [[0.1, 0.2], [0.3, 0.4]], atol=1e-12)
        np.testing.assert_allclose(run, [[0.5, 0.6], [0.7, 0.8]], atol=1e-12)


# Pipeline wiring: SheafUPDEEngine extends UPDEEngine to multi-dimensional
# phase vectors with matrix-valued restriction maps. The D=1 parity case
# above guarantees backwards compatibility; the higher-D cases pin
# external drive, degenerate connectivity, single-oscillator decoupling
# and the [0, 2π) wrap contract used by the supervisor downstream.
