# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for geometric, numerics, OA reduction

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability
from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── TorusEngine ──────────────────────────────────────────────────────────


class TestTorusEngine:
    def test_step_matches_exact_free_rotation_on_circle(self) -> None:
        """With K=0 and no drive, torus dynamics are exact S1 rotations."""
        n = 5
        dt = 0.01
        eng = TorusEngine(n, dt=dt)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (phases + dt * omegas) % TWO_PI
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)
        assert np.all(out >= 0)
        assert np.all(out < TWO_PI + 1e-10)

    def test_identical_phases_remain_fully_synchronised(self) -> None:
        """All-to-all identical phases have zero coupling torque and R=1."""
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.full(n, 0.7)
        omegas = np.full(n, 1.25)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out, np.full(n, (0.7 + 0.0125) % TWO_PI))
        assert eng.order_parameter(out) == pytest.approx(1.0, abs=1e-15)

    def test_finite_output(self) -> None:
        n = 6
        eng = TorusEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-5, 5, n)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))

    def test_run_n_steps(self) -> None:
        n = 3
        dt = 0.01
        n_steps = 100
        eng = TorusEngine(n, dt=dt)
        phases = np.array([0.2, 1.0, 5.8])
        omegas = np.array([1.0, 2.0, 3.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        expected = (phases + n_steps * dt * omegas) % TWO_PI
        np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)

    def test_zeta_driving(self) -> None:
        n = 3
        eng = TorusEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out_no_drive = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_drive = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
        assert not np.allclose(out_no_drive, out_drive)

    def test_zero_coupling_free_rotation(self) -> None:
        """K=0 → each oscillator rotates at its own ω."""
        n = 2
        eng = TorusEngine(n, dt=0.01)
        phases = np.array([0.0, 0.0])
        omegas = np.array([1.0, -1.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert out[0] > 0  # positive ω → advances
        assert out[1] > np.pi  # negative ω → wraps backward

    @pytest.mark.parametrize(
        ("constructor_args", "message"),
        [
            ((True, 0.01), "n_oscillators"),
            ((4, True), "dt"),
        ],
    )
    def test_constructor_rejects_boolean_aliases(
        self,
        constructor_args: tuple[object, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TorusEngine(*constructor_args)

    def test_run_rejects_boolean_step_count(self) -> None:
        n = 3
        eng = TorusEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        with pytest.raises(ValueError, match="n_steps"):
            eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=True)


# ── Numerics ─────────────────────────────────────────────────────────────


class TestIntegrationConfig:
    def test_defaults(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.substeps == 1
        assert cfg.method == "euler"
        assert cfg.max_dt == 0.01

    def test_custom(self) -> None:
        cfg = IntegrationConfig(dt=0.005, substeps=4, method="rk4")
        assert cfg.substeps == 4
        assert cfg.method == "rk4"

    def test_frozen(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        with pytest.raises(AttributeError):
            cfg.dt = 0.02  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dt": True},
            {"dt": 0.01, "substeps": True},
            {"dt": 0.01, "max_dt": True},
            {"dt": 0.01, "atol": True},
            {"dt": 0.01, "rtol": True},
        ],
    )
    def test_rejects_boolean_numeric_aliases(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            IntegrationConfig(**kwargs)

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            IntegrationConfig(dt=0.01, method="verlet")


class TestCheckStability:
    def test_stable(self) -> None:
        assert check_stability(0.01, max_omega=10.0, max_coupling=5.0) is True

    def test_unstable(self) -> None:
        assert check_stability(1.0, max_omega=10.0, max_coupling=5.0) is False

    def test_zero_deriv(self) -> None:
        assert check_stability(100.0, max_omega=0.0, max_coupling=0.0) is True

    def test_boundary(self) -> None:
        # dt * (omega + coupling) = pi → False (not strictly less)
        dt = math.pi / 15.0
        assert check_stability(dt, 10.0, 5.0) is False

    @pytest.mark.parametrize("dt", [0.001, 0.005, 0.01])
    def test_small_dt_stable(self, dt: float) -> None:
        assert check_stability(dt, max_omega=5.0, max_coupling=5.0) is True

    @pytest.mark.parametrize(
        ("dt", "max_omega", "max_coupling"),
        [
            (True, 1.0, 1.0),
            (0.01, True, 1.0),
            (0.01, 1.0, True),
        ],
    )
    def test_rejects_boolean_numeric_aliases(
        self,
        dt: object,
        max_omega: object,
        max_coupling: object,
    ) -> None:
        with pytest.raises(ValueError):
            check_stability(dt, max_omega, max_coupling)  # type: ignore[arg-type]


# ── OttAntonsenReduction ─────────────────────────────────────────────────


class TestOttAntonsenReduction:
    def test_k_critical(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        assert oa.K_c == 1.0

    def test_steady_state_above_kc(self) -> None:
        """K > K_c → R_ss > 0."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0)
        R_ss = oa.steady_state_R()
        assert 0.0 < R_ss < 1.0
        expected = np.sqrt(1.0 - 2 * 0.5 / 4.0)
        assert abs(R_ss - expected) < 1e-12

    def test_steady_state_below_kc(self) -> None:
        """K < K_c → R_ss = 0."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=1.0)
        assert oa.steady_state_R() == 0.0

    def test_steady_state_zero_width_locks_to_unit_order(self) -> None:
        """Identical-frequency OA manifold has R_ss=1 for positive K."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.0, K=2.0)
        assert oa.K_c == 0.0
        assert oa.steady_state_R() == pytest.approx(1.0, abs=1e-15)

    def test_negative_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            OttAntonsenReduction(omega_0=0.0, delta=-1.0, K=1.0)

    def test_step_preserves_boundedness(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        z = complex(0.5, 0.0)
        for _ in range(100):
            z = oa.step(z)
        assert abs(z) <= 1.0 + 1e-6

    def test_step_moves_subcritical_seed_toward_above_kc_attractor(self) -> None:
        """For K>Kc and real z below R_ss, OA radius must increase."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0)
        z0 = complex(0.1, 0.0)
        z1 = oa.step(z0)
        assert abs(z1) > abs(z0)
        assert abs(z1) < oa.steady_state_R()

    def test_zero_width_uncoupled_flow_conserves_order_radius(self) -> None:
        """With Δ=K=0, OA flow is pure phase rotation, not damping."""
        oa = OttAntonsenReduction(omega_0=1.25, delta=0.0, K=0.0, dt=0.001)
        state = oa.run(complex(0.4, 0.0), n_steps=200)
        assert pytest.approx(0.4, abs=1e-10) == state.R

    def test_run_returns_oa_state(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        state = oa.run(complex(0.1, 0.0), n_steps=100)
        assert isinstance(state, OAState)
        assert 0.0 <= state.R <= 1.0 + 1e-6
        assert state.K_c == 1.0

    def test_run_converges_above_kc(self) -> None:
        """K > K_c → R converges to R_ss."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0, dt=0.01)
        state = oa.run(complex(0.01, 0.0), n_steps=5000)
        R_ss = oa.steady_state_R()
        assert abs(state.R - R_ss) < 0.05

    def test_predict_from_oscillators(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=5.0)
        omegas = np.random.default_rng(0).standard_cauchy(50) * 0.5
        omegas = np.clip(omegas, -5, 5)
        state = oa.predict_from_oscillators(omegas, K=5.0)
        assert isinstance(state, OAState)
        assert 0.0 <= state.R <= 1.0 + 1e-6


class TestUPDEMathPipelineWiring:
    """Pipeline: OA predicts R → engine verifies → torus extends."""

    def test_oa_prediction_vs_engine_simulation(self):
        """Workflow contract: identical-frequency OA theory matches engine locking."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 24
        K = 4.0
        delta = 0.0
        oa = OttAntonsenReduction(omega_0=0.0, delta=delta, K=K, dt=0.01)
        R_theory = oa.steady_state_R()

        omegas = np.zeros(n)
        eng = UPDEEngine(n, dt=0.01)
        phases = np.linspace(-0.6, 0.6, n) % TWO_PI
        R_initial, _ = compute_order_parameter(phases)
        knm = np.ones((n, n)) * (K / n)
        np.fill_diagonal(knm, 0.0)
        for _ in range(400):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))
        R_sim, _ = compute_order_parameter(phases)

        assert R_theory == pytest.approx(1.0, abs=1e-15)
        assert R_sim > R_initial + 0.02
        assert R_sim > 0.99
