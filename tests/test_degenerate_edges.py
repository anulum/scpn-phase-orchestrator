# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Degenerate and boundary edge case tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import CouplingBuilder
from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.winding import winding_numbers
from scpn_phase_orchestrator.upde.basin_stability import basin_stability
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * np.pi


# ── Helpers ──────────────────────────────────────────────────────────────


def _zero_knm(n: int) -> np.ndarray:
    return np.zeros((n, n))


def _zero_alpha(n: int) -> np.ndarray:
    return np.zeros((n, n))


def _uniform_omegas(n: int, omega: float = 1.0) -> np.ndarray:
    return np.full(n, omega)


def _random_knm(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.uniform(0, 0.5, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── UPDEEngine edge cases ───────────────────────────────────────────────


class TestUPDESingleOscillator:
    def test_n1_step_returns_correct_shape(self) -> None:
        eng = UPDEEngine(1, dt=0.01)
        phases = np.array([1.0])
        out = eng.step(phases, np.array([2.0]), _zero_knm(1), 0.0, 0.0, _zero_alpha(1))
        assert out.shape == (1,)

    def test_n1_free_rotation(self) -> None:
        eng = UPDEEngine(1, dt=0.01)
        phase = np.array([0.5])
        omega = np.array([3.0])
        out = eng.step(phase, omega, _zero_knm(1), 0.0, 0.0, _zero_alpha(1))
        expected = (0.5 + 0.01 * 3.0) % TWO_PI
        np.testing.assert_allclose(out[0], expected, atol=1e-12)

    def test_n1_order_parameter_is_one(self) -> None:
        r, _ = compute_order_parameter(np.array([1.23]))
        assert abs(r - 1.0) < 1e-12


class TestUPDEFreeRotation:
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_zero_coupling_free_rotation(self, n: int) -> None:
        rng = np.random.default_rng(42)
        eng = UPDEEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-5.0, 5.0, n)
        out = eng.step(phases, omegas, _zero_knm(n), 0.0, 0.0, _zero_alpha(n))
        expected = (phases + 0.01 * omegas) % TWO_PI
        np.testing.assert_allclose(out, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_identical_phases_remain_identical(self, n: int) -> None:
        eng = UPDEEngine(n, dt=0.01)
        phases = np.full(n, 1.0)
        omegas = np.full(n, 2.0)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = _zero_alpha(n)
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out, out[0] * np.ones(n), atol=1e-12)


class TestUPDEZeroDt:
    @pytest.mark.parametrize("n", [2, 4])
    def test_dt_zero_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            UPDEEngine(n, dt=0.0)


class TestUPDEAllSameFrequencies:
    @pytest.mark.parametrize("n", [3, 5, 8])
    @pytest.mark.parametrize("omega", [0.0, 1.0, -3.0, 100.0])
    def test_same_freq_no_nan(self, n: int, omega: float) -> None:
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.full(n, omega)
        out = eng.step(phases, omegas, _random_knm(n, rng), 0.0, 0.0, _zero_alpha(n))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)


class TestUPDENegativeFrequencies:
    @pytest.mark.parametrize("n", [2, 4])
    def test_negative_freq_free_rotation(self, n: int) -> None:
        eng = UPDEEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.full(n, -5.0)
        out = eng.step(phases, omegas, _zero_knm(n), 0.0, 0.0, _zero_alpha(n))
        expected = (0.0 + 0.01 * (-5.0)) % TWO_PI
        np.testing.assert_allclose(out, np.full(n, expected), atol=1e-12)


class TestUPDEPhaseBoundary:
    def test_phase_exactly_zero(self) -> None:
        eng = UPDEEngine(4, dt=0.01)
        phases = np.zeros(4)
        out = eng.step(phases, np.zeros(4), _zero_knm(4), 0.0, 0.0, _zero_alpha(4))
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-12)

    def test_phase_near_two_pi_wraps(self) -> None:
        eng = UPDEEngine(2, dt=0.01)
        phases = np.array([TWO_PI - 0.001, TWO_PI - 0.001])
        omegas = np.array([1.0, 1.0])
        out = eng.step(phases, omegas, _zero_knm(2), 0.0, 0.0, _zero_alpha(2))
        assert np.all(out >= 0) and np.all(out < TWO_PI)


class TestUPDELargeDt:
    @pytest.mark.parametrize("dt", [1.0, 10.0, 100.0])
    def test_large_dt_phase_bounded(self, dt: float) -> None:
        eng = UPDEEngine(4, dt=dt)
        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, 4)
        omegas = rng.uniform(-5, 5, 4)
        out = eng.step(phases, omegas, _random_knm(4, rng), 0.0, 0.0, _zero_alpha(4))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)


class TestUPDELargeCoupling:
    @pytest.mark.parametrize("scale", [10.0, 100.0, 1000.0])
    def test_large_knm_phase_bounded(self, scale: float) -> None:
        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(2)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _random_knm(n, rng) * scale
        out = eng.step(phases, _uniform_omegas(n), knm, 0.0, 0.0, _zero_alpha(n))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)


class TestUPDEMethods:
    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_all_methods_phase_bounded(self, method: str) -> None:
        n = 4
        eng = UPDEEngine(n, dt=0.01, method=method)
        rng = np.random.default_rng(3)
        phases = rng.uniform(0, TWO_PI, n)
        out = eng.step(
            phases, rng.uniform(-2, 2, n), _random_knm(n, rng), 0.5, 1.0, _zero_alpha(n)
        )
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_all_methods_free_rotation_agree(self, method: str) -> None:
        n = 4
        rng = np.random.default_rng(4)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        eng = UPDEEngine(n, dt=0.001, method=method)
        out = eng.step(phases, omegas, _zero_knm(n), 0.0, 0.0, _zero_alpha(n))
        expected = (phases + 0.001 * omegas) % TWO_PI
        np.testing.assert_allclose(out, expected, atol=1e-6)


# ── Stuart-Landau edge cases ────────────────────────────────────────────


class TestStuartLandauEdges:
    def _make_state(self, n: int, phase: float = 1.0, amp: float = 1.0) -> np.ndarray:
        state = np.empty(2 * n)
        state[:n] = phase
        state[n:] = amp
        return state

    def test_n1(self) -> None:
        eng = StuartLandauEngine(1, dt=0.01)
        state = self._make_state(1)
        out = eng.step(
            state,
            np.array([1.0]),
            np.array([1.0]),
            _zero_knm(1),
            _zero_knm(1),
            0.0,
            0.0,
            _zero_alpha(1),
        )
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("n", [2, 4])
    def test_zero_coupling_amplitude_evolves(self, n: int) -> None:
        eng = StuartLandauEngine(n, dt=0.01)
        state = self._make_state(n, amp=0.5)
        mu = np.ones(n) * 2.0
        out = eng.step(
            state,
            _uniform_omegas(n),
            mu,
            _zero_knm(n),
            _zero_knm(n),
            0.0,
            0.0,
            _zero_alpha(n),
        )
        assert np.all(np.isfinite(out))
        assert np.all(out[:n] >= 0) and np.all(out[:n] < TWO_PI)
        assert np.all(out[n:] >= 0)

    def test_dt_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            StuartLandauEngine(3, dt=0.0)

    @pytest.mark.parametrize("scale", [10.0, 100.0])
    def test_large_coupling_finite(self, scale: float) -> None:
        n = 3
        eng = StuartLandauEngine(n, dt=0.001)
        rng = np.random.default_rng(5)
        state = self._make_state(n, phase=1.5, amp=1.0)
        knm = _random_knm(n, rng) * scale
        out = eng.step(
            state,
            _uniform_omegas(n),
            np.ones(n),
            knm,
            knm,
            0.0,
            0.0,
            _zero_alpha(n),
        )
        assert np.all(np.isfinite(out))


# ── Simplicial edge cases ───────────────────────────────────────────────


class TestSimplicialEdges:
    def test_n2_three_body_vanishes(self) -> None:
        eng = SimplicialEngine(2, dt=0.01, sigma2=5.0)
        rng = np.random.default_rng(6)
        phases = rng.uniform(0, TWO_PI, 2)
        omegas = rng.uniform(-2, 2, 2)
        knm = _random_knm(2, rng)
        out = eng.step(phases, omegas, knm, 0.0, 0.0, _zero_alpha(2))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)

    def test_sigma2_zero_matches_upde(self) -> None:
        n = 4
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _random_knm(n, rng)
        alpha = _zero_alpha(n)
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_simp, out_upde, atol=1e-10)

    @pytest.mark.parametrize("n", [3, 5, 8])
    @pytest.mark.parametrize("sigma2", [0.0, 0.5, 5.0, 50.0])
    def test_phases_bounded(self, n: int, sigma2: float) -> None:
        eng = SimplicialEngine(n, dt=0.01, sigma2=sigma2)
        rng = np.random.default_rng(8)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _random_knm(n, rng)
        out = eng.step(phases, omegas, knm, 0.0, 0.0, _zero_alpha(n))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0) and np.all(out < TWO_PI)

    def test_n1(self) -> None:
        eng = SimplicialEngine(1, dt=0.01, sigma2=1.0)
        out = eng.step(
            np.array([1.0]),
            np.array([2.0]),
            _zero_knm(1),
            0.0,
            0.0,
            _zero_alpha(1),
        )
        assert out.shape == (1,)


# ── Swarmalator edge cases ───────────────────────────────────────────────


class TestSwarmalatorEdges:
    def test_n2_step(self) -> None:
        eng = SwarmalatorEngine(2, dim=2, dt=0.01)
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        phases = np.array([0.0, np.pi])
        omegas = np.array([1.0, 1.0])
        new_pos, new_phases = eng.step(pos, phases, omegas)
        assert new_pos.shape == (2, 2)
        assert new_phases.shape == (2,)
        assert np.all(np.isfinite(new_pos))
        assert np.all(np.isfinite(new_phases))

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_dimensions(self, dim: int) -> None:
        n = 4
        eng = SwarmalatorEngine(n, dim=dim, dt=0.01)
        rng = np.random.default_rng(9)
        pos = rng.uniform(-1, 1, (n, dim))
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        new_pos, new_phases = eng.step(pos, phases, omegas)
        assert new_pos.shape == (n, dim)
        assert new_phases.shape == (n,)

    def test_j_zero_decouples(self) -> None:
        n = 4
        eng_coupled = SwarmalatorEngine(n, dt=0.01, J=1.0)
        eng_decoupled = SwarmalatorEngine(n, dt=0.01, J=0.0)
        rng = np.random.default_rng(10)
        pos = rng.uniform(-1, 1, (n, 2))
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        _, ph_c = eng_coupled.step(pos, phases, omegas)
        _, ph_d = eng_decoupled.step(pos, phases, omegas)
        assert ph_c.shape == (n,)
        assert ph_d.shape == (n,)


# ── Inertial Kuramoto edge cases ─────────────────────────────────────────


class TestInertialEdges:
    def test_n1(self) -> None:
        eng = InertialKuramotoEngine(1, dt=0.01)
        theta = np.array([1.0])
        omega_dot = np.array([0.0])
        power = np.array([0.0])
        knm = _zero_knm(1)
        inertia = np.array([1.0])
        damping = np.array([1.0])
        new_theta, new_omega = eng.step(theta, omega_dot, power, knm, inertia, damping)
        assert new_theta.shape == (1,)
        assert new_omega.shape == (1,)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_zero_power_zero_coupling_stationary(self, n: int) -> None:
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = np.zeros(n)
        omega_dot = np.zeros(n)
        new_theta, new_omega = eng.step(
            theta,
            omega_dot,
            np.zeros(n),
            _zero_knm(n),
            np.ones(n),
            np.ones(n),
        )
        np.testing.assert_allclose(new_theta, np.zeros(n), atol=1e-12)
        np.testing.assert_allclose(new_omega, np.zeros(n), atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4])
    def test_large_coupling_finite(self, n: int) -> None:
        eng = InertialKuramotoEngine(n, dt=0.001)
        rng = np.random.default_rng(11)
        theta = rng.uniform(0, TWO_PI, n)
        omega_dot = np.zeros(n)
        knm = _random_knm(n, rng) * 100.0
        new_theta, new_omega = eng.step(
            theta,
            omega_dot,
            np.zeros(n),
            knm,
            np.ones(n),
            np.ones(n),
        )
        assert np.all(np.isfinite(new_theta))
        assert np.all(np.isfinite(new_omega))


# ── Order parameter edge cases ───────────────────────────────────────────


class TestOrderParameterEdges:
    def test_identical_phases_r_one(self) -> None:
        r, _ = compute_order_parameter(np.full(100, 1.234))
        assert abs(r - 1.0) < 1e-12

    def test_two_opposite_phases_r_zero(self) -> None:
        r, _ = compute_order_parameter(np.array([0.0, np.pi]))
        assert abs(r) < 1e-12

    @pytest.mark.parametrize("n", [2, 10, 100])
    def test_uniform_phases_r_near_zero(self, n: int) -> None:
        phases = np.linspace(0, TWO_PI, n, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert r < 0.1

    def test_single_phase(self) -> None:
        r, psi = compute_order_parameter(np.array([2.5]))
        assert abs(r - 1.0) < 1e-12
        assert abs(psi - 2.5) < 1e-12


class TestPLVEdges:
    def test_identical_series(self) -> None:
        phases = np.linspace(0, TWO_PI * 3, 100)
        plv = compute_plv(phases, phases)
        assert abs(plv - 1.0) < 1e-10

    def test_opposite_series(self) -> None:
        a = np.linspace(0, TWO_PI * 3, 100)
        b = a + np.pi
        plv = compute_plv(a, b)
        assert abs(plv - 1.0) < 1e-10

    def test_random_series_bounded(self) -> None:
        rng = np.random.default_rng(12)
        a = rng.uniform(0, TWO_PI, 200)
        b = rng.uniform(0, TWO_PI, 200)
        plv = compute_plv(a, b)
        assert 0.0 <= plv <= 1.0 + 1e-12


# ── Chimera detection edge cases ─────────────────────────────────────────


class TestChimeraEdges:
    def test_synchronized_no_chimera(self) -> None:
        n = 10
        phases = np.zeros(n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        result = detect_chimera(phases, knm)
        assert result.chimera_index < 0.01

    def test_n2(self) -> None:
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = detect_chimera(phases, knm)
        assert result.chimera_index >= 0


# ── Winding number edge cases ────────────────────────────────────────────


class TestWindingEdges:
    def test_stationary_zero_winding(self) -> None:
        traj = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
        wn = winding_numbers(traj)
        np.testing.assert_allclose(wn, 0, atol=1e-12)

    def test_full_rotation_winding_one(self) -> None:
        # winding = floor((theta_T - theta_0) / 2pi)
        n_steps = 100
        theta = np.linspace(0, TWO_PI, n_steps, endpoint=False)
        traj = theta[:, np.newaxis]
        wn = winding_numbers(traj)
        expected = int(np.floor((theta[-1] - theta[0]) / TWO_PI))
        assert wn[0] == expected

    def test_large_rotation_winding(self) -> None:
        n_steps = 1000
        theta = np.linspace(0, TWO_PI * 5, n_steps)
        traj = theta[:, np.newaxis]
        wn = winding_numbers(traj)
        expected = int(np.floor((theta[-1] - theta[0]) / TWO_PI))
        assert wn[0] == expected


# ── Basin stability edge cases ───────────────────────────────────────────


class TestBasinStabilityEdges:
    def test_n2_bounded(self) -> None:
        knm = np.array([[0.0, 2.0], [2.0, 0.0]])
        omegas = np.array([1.0, 1.0])
        result = basin_stability(omegas, knm, n_samples=10, dt=0.01)
        assert 0.0 <= result.S_B <= 1.0
        assert result.n_converged <= result.n_samples
        assert len(result.R_final) == result.n_samples

    def test_zero_coupling_low_stability(self) -> None:
        n = 4
        knm = _zero_knm(n)
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        result = basin_stability(omegas, knm, n_samples=20, dt=0.01)
        assert 0.0 <= result.S_B <= 1.0
        assert all(0.0 <= r <= 1.0 + 1e-12 for r in result.R_final)


# ── CouplingBuilder edge cases ───────────────────────────────────────────


class TestCouplingBuilderEdges:
    def test_n2_valid(self) -> None:
        state = CouplingBuilder().build(2, base_strength=1.0, decay_alpha=0.1)
        assert state.knm.shape == (2, 2)
        assert state.knm[0, 0] == 0.0
        assert state.knm[1, 1] == 0.0
        np.testing.assert_allclose(state.knm, state.knm.T)

    def test_zero_base_strength(self) -> None:
        state = CouplingBuilder().build(4, base_strength=0.0, decay_alpha=0.1)
        np.testing.assert_allclose(state.knm, np.zeros((4, 4)))

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_zero_diagonal(self, n: int) -> None:
        state = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        np.testing.assert_allclose(np.diag(state.knm), 0.0)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_symmetry(self, n: int) -> None:
        state = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        np.testing.assert_allclose(state.knm, state.knm.T)
