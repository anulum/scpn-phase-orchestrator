# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn.functional (JAX Kuramoto)

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_phase_orchestrator.nn.functional import (
    coupling_laplacian,
    kuramoto_forward,
    kuramoto_forward_masked,
    kuramoto_rk4_step,
    kuramoto_rk4_step_masked,
    kuramoto_step,
    kuramoto_step_masked,
    order_parameter,
    plv,
    saf_loss,
    saf_order_parameter,
    simplicial_forward,
    simplicial_rk4_step,
    simplicial_step,
    stuart_landau_forward,
    stuart_landau_rk4_step,
    stuart_landau_step,
    winfree_forward,
    winfree_rk4_step,
    winfree_step,
)

TWO_PI = 2.0 * math.pi


@pytest.fixture()
def uniform_phases():
    return jnp.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])


@pytest.fixture()
def sync_phases():
    return jnp.zeros(8)


@pytest.fixture()
def omegas4():
    return jnp.ones(4)


@pytest.fixture()
def zero_coupling4():
    return jnp.zeros((4, 4))


@pytest.fixture()
def uniform_coupling4():
    K = jnp.ones((4, 4)) * 0.5
    return K - jnp.diag(jnp.diag(K))


# ── Kuramoto Euler ──


class TestKuramotoStep:
    def test_uncoupled_advances_by_omega_dt(self, omegas4, zero_coupling4):
        phases = jnp.zeros(4)
        dt = 0.01
        result = kuramoto_step(phases, omegas4, zero_coupling4, dt)
        np.testing.assert_allclose(result, jnp.full(4, 0.01), atol=1e-12)

    def test_phases_wrapped_to_0_2pi(self, omegas4, zero_coupling4):
        phases = jnp.full(4, TWO_PI - 0.005)
        result = kuramoto_step(phases, omegas4, zero_coupling4, 0.01)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result < TWO_PI + 1e-10)

    def test_coupling_drives_sync(self, omegas4, uniform_coupling4):
        phases = jnp.array([0.1, 0.3, 0.5, 0.7])
        r_before = float(order_parameter(phases))
        for _ in range(500):
            phases = kuramoto_step(phases, jnp.zeros(4), uniform_coupling4, 0.01)
        r_after = float(order_parameter(phases))
        assert r_after > r_before


# ── Kuramoto RK4 ──


class TestKuramotoRK4:
    def test_uncoupled_exact(self, omegas4, zero_coupling4):
        phases = jnp.zeros(4)
        result = kuramoto_rk4_step(phases, omegas4, zero_coupling4, 0.01)
        np.testing.assert_allclose(result, jnp.full(4, 0.01), atol=1e-12)

    def test_rk4_more_accurate_than_euler(self, uniform_phases, uniform_coupling4):
        omegas = jnp.ones(4)
        dt = 0.05
        e1 = kuramoto_step(uniform_phases, omegas, uniform_coupling4, dt)
        r1 = kuramoto_rk4_step(uniform_phases, omegas, uniform_coupling4, dt)
        # Both should produce valid phases
        assert jnp.all(jnp.isfinite(e1))
        assert jnp.all(jnp.isfinite(r1))


# ── Kuramoto Forward ──


class TestKuramotoForward:
    def test_trajectory_shape(self, omegas4, zero_coupling4):
        phases = jnp.zeros(4)
        final, traj = kuramoto_forward(phases, omegas4, zero_coupling4, 0.01, 50)
        assert final.shape == (4,)
        assert traj.shape == (50, 4)

    def test_euler_method(self, omegas4, zero_coupling4):
        phases = jnp.zeros(4)
        final, traj = kuramoto_forward(
            phases, omegas4, zero_coupling4, 0.01, 10, method="euler"
        )
        assert final.shape == (4,)

    def test_final_matches_last_trajectory(self, omegas4, zero_coupling4):
        phases = jnp.zeros(4)
        final, traj = kuramoto_forward(phases, omegas4, zero_coupling4, 0.01, 20)
        np.testing.assert_allclose(final, traj[-1], atol=1e-12)


# ── Masked Kuramoto ──


class TestMaskedKuramoto:
    def test_zero_mask_equals_uncoupled(self, omegas4, uniform_coupling4):
        phases = jnp.array([0.1, 0.5, 1.0, 1.5])
        mask = jnp.zeros((4, 4))
        dt = 0.01
        masked = kuramoto_step_masked(phases, omegas4, uniform_coupling4, mask, dt)
        uncoupled = kuramoto_step(phases, omegas4, jnp.zeros((4, 4)), dt)
        np.testing.assert_allclose(masked, uncoupled, atol=1e-12)

    def test_ones_mask_equals_full(self, uniform_phases, omegas4, uniform_coupling4):
        mask = jnp.ones((4, 4))
        dt = 0.01
        masked = kuramoto_step_masked(
            uniform_phases, omegas4, uniform_coupling4, mask, dt
        )
        full = kuramoto_step(uniform_phases, omegas4, uniform_coupling4, dt)
        np.testing.assert_allclose(masked, full, atol=1e-12)

    def test_rk4_masked_works(self, uniform_phases, omegas4, uniform_coupling4):
        mask = jnp.eye(4)
        dt = 0.01
        result = kuramoto_rk4_step_masked(
            uniform_phases, omegas4, uniform_coupling4, mask, dt
        )
        assert jnp.all(jnp.isfinite(result))

    def test_forward_masked(self, omegas4, uniform_coupling4):
        phases = jnp.zeros(4)
        mask = jnp.ones((4, 4))
        final, traj = kuramoto_forward_masked(
            phases, omegas4, uniform_coupling4, mask, 0.01, 10
        )
        assert final.shape == (4,)
        assert traj.shape == (10, 4)


# ── Winfree ──


class TestWinfree:
    def test_zero_coupling_matches_uncoupled_rotation(self):
        phases = jnp.array([0.1, 0.5, 1.0, 1.5])
        omegas = jnp.array([0.2, 0.4, 0.6, 0.8])
        dt = 0.25

        result = winfree_step(phases, omegas, K=0.0, dt=dt)

        np.testing.assert_allclose(result, (phases + dt * omegas) % TWO_PI, atol=1e-7)

    def test_winfree_step_advances(self):
        phases = jnp.array([0.1, 0.5, 1.0, 1.5])
        omegas = jnp.ones(4)
        result = winfree_step(phases, omegas, K=1.0, dt=0.01)
        assert jnp.all(jnp.isfinite(result))
        assert not jnp.allclose(result, phases)

    def test_winfree_rk4_step(self):
        phases = jnp.array([0.1, 0.5, 1.0, 1.5])
        omegas = jnp.ones(4)
        result = winfree_rk4_step(phases, omegas, K=1.0, dt=0.01)
        assert jnp.all(jnp.isfinite(result))

    def test_winfree_forward(self):
        phases = jnp.zeros(4)
        omegas = jnp.ones(4)
        final, traj = winfree_forward(phases, omegas, K=1.0, dt=0.01, n_steps=20)
        assert final.shape == (4,)
        assert traj.shape == (20, 4)

    def test_winfree_forward_euler_records_last_state(self):
        phases = jnp.array([0.0, 0.2, 0.4, 0.6])
        omegas = jnp.ones(4)

        final, traj = winfree_forward(
            phases, omegas, K=0.5, dt=0.01, n_steps=5, method="euler"
        )

        np.testing.assert_allclose(final, traj[-1], atol=1e-12)


# ── Simplicial ──


class TestSimplicial:
    def test_zero_sigma2_equals_kuramoto(
        self, uniform_phases, omegas4, uniform_coupling4
    ):
        dt = 0.01
        simp = simplicial_step(
            uniform_phases, omegas4, uniform_coupling4, dt, sigma2=0.0
        )
        kura = kuramoto_step(uniform_phases, omegas4, uniform_coupling4, dt)
        np.testing.assert_allclose(simp, kura, atol=1e-10)

    def test_nonzero_sigma2_differs(self, omegas4, uniform_coupling4):
        phases = jnp.array([0.1, 0.3, 0.8, 1.5])
        dt = 0.01
        s0 = simplicial_step(phases, omegas4, uniform_coupling4, dt, sigma2=0.0)
        s5 = simplicial_step(phases, omegas4, uniform_coupling4, dt, sigma2=5.0)
        assert not jnp.allclose(s0, s5)

    def test_rk4_simplicial(self, uniform_phases, omegas4, uniform_coupling4):
        result = simplicial_rk4_step(
            uniform_phases, omegas4, uniform_coupling4, 0.01, sigma2=1.0
        )
        assert jnp.all(jnp.isfinite(result))

    def test_forward_simplicial(self, omegas4, uniform_coupling4):
        phases = jnp.zeros(4)
        final, traj = simplicial_forward(
            phases, omegas4, uniform_coupling4, 0.01, 10, sigma2=0.5
        )
        assert final.shape == (4,)
        assert traj.shape == (10, 4)

    def test_forward_simplicial_euler_sigma_zero_matches_kuramoto(
        self, omegas4, uniform_coupling4
    ):
        phases = jnp.array([0.1, 0.3, 0.8, 1.5])

        simp_final, simp_traj = simplicial_forward(
            phases,
            omegas4,
            uniform_coupling4,
            0.01,
            7,
            sigma2=0.0,
            method="euler",
        )
        kura_final, kura_traj = kuramoto_forward(
            phases, omegas4, uniform_coupling4, 0.01, 7, method="euler"
        )

        np.testing.assert_allclose(simp_final, kura_final, atol=1e-10)
        np.testing.assert_allclose(simp_traj, kura_traj, atol=1e-10)


# ── Stuart-Landau ──


class TestStuartLandau:
    def test_step_basic(self):
        n = 4
        phases = jnp.array([0.0, 0.5, 1.0, 1.5])
        amps = jnp.ones(n) * 0.5
        omegas = jnp.ones(n)
        mu = jnp.ones(n) * 0.5
        K = jnp.zeros((n, n))
        K_r = jnp.zeros((n, n))
        new_p, new_r = stuart_landau_step(phases, amps, omegas, mu, K, K_r, dt=0.01)
        assert new_p.shape == (n,)
        assert new_r.shape == (n,)
        assert jnp.all(new_r >= 0.0)

    def test_rk4_step(self):
        n = 4
        phases = jnp.zeros(n)
        amps = jnp.ones(n)
        omegas = jnp.ones(n)
        mu = jnp.ones(n)
        K = jnp.eye(n) * 0.0
        K_r = jnp.eye(n) * 0.0
        new_p, new_r = stuart_landau_rk4_step(phases, amps, omegas, mu, K, K_r, dt=0.01)
        assert jnp.all(jnp.isfinite(new_p))
        assert jnp.all(new_r >= 0.0)

    def test_forward(self):
        n = 4
        phases = jnp.zeros(n)
        amps = jnp.ones(n) * 0.1
        omegas = jnp.ones(n)
        mu = jnp.ones(n) * 0.5
        K = jnp.zeros((n, n))
        K_r = jnp.zeros((n, n))
        fp, fr, tp, tr = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, dt=0.01, n_steps=20
        )
        assert fp.shape == (n,)
        assert fr.shape == (n,)
        assert tp.shape == (20, n)
        assert tr.shape == (20, n)

    def test_forward_euler_records_final_phase_and_amplitude(self):
        n = 3
        phases = jnp.array([0.0, 0.2, 0.4])
        amps = jnp.ones(n) * 0.2
        omegas = jnp.ones(n) * 0.5
        mu = jnp.ones(n) * 0.3
        K = jnp.zeros((n, n))
        K_r = jnp.zeros((n, n))

        fp, fr, tp, tr = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, dt=0.01, n_steps=6, method="euler"
        )

        np.testing.assert_allclose(fp, tp[-1], atol=1e-12)
        np.testing.assert_allclose(fr, tr[-1], atol=1e-12)

    def test_negative_amplitudes_are_clipped_to_physical_radius(self):
        n = 2
        phases = jnp.zeros(n)
        amps = jnp.array([-0.5, 0.1])
        omegas = jnp.zeros(n)
        mu = jnp.ones(n)
        K = jnp.zeros((n, n))
        K_r = jnp.ones((n, n)) - jnp.eye(n)

        _, new_r = stuart_landau_step(phases, amps, omegas, mu, K, K_r, dt=0.1)

        assert jnp.all(new_r >= 0.0)

    def test_amplitude_grows_when_mu_positive(self):
        n = 2
        phases = jnp.zeros(n)
        amps = jnp.ones(n) * 0.01
        omegas = jnp.zeros(n)
        mu = jnp.ones(n) * 2.0
        K = jnp.zeros((n, n))
        K_r = jnp.zeros((n, n))
        fp, fr, _, _ = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, dt=0.001, n_steps=500
        )
        assert float(jnp.mean(fr)) > 0.01


# ── Order Parameter & PLV ──


class TestOrderParameter:
    def test_sync_phases_give_r_one(self, sync_phases):
        r = float(order_parameter(sync_phases))
        assert abs(r - 1.0) < 1e-6

    def test_uniform_phases_give_r_near_zero(self):
        n = 100
        phases = jnp.linspace(0, TWO_PI, n, endpoint=False)
        r = float(order_parameter(phases))
        assert r < 0.1

    def test_trajectory_input(self):
        traj = jnp.zeros((10, 8))
        r = order_parameter(traj)
        assert r.shape == (10,)


class TestPLV:
    def test_sync_trajectory_gives_plv_one(self):
        traj = jnp.zeros((50, 4))
        plv_mat = plv(traj)
        assert plv_mat.shape == (4, 4)
        np.testing.assert_allclose(plv_mat, jnp.ones((4, 4)), atol=1e-6)

    def test_plv_symmetric(self):
        key = jax.random.PRNGKey(42)
        traj = jax.random.uniform(key, (50, 4)) * TWO_PI
        plv_mat = plv(traj)
        np.testing.assert_allclose(plv_mat, plv_mat.T, atol=1e-12)

    def test_constant_phase_offset_is_locked(self):
        base = jnp.linspace(0.0, 1.0, 25)
        traj = jnp.stack([base, base + 0.75], axis=1)

        plv_mat = plv(traj)

        np.testing.assert_allclose(plv_mat, jnp.ones((2, 2)), atol=1e-6)


# ── SAF & Laplacian ──


class TestSAF:
    def test_laplacian_symmetric(self):
        K = jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]])
        L = coupling_laplacian(K)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_laplacian_row_sums_zero(self):
        K = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        L = coupling_laplacian(K)
        row_sums = jnp.sum(L, axis=1)
        np.testing.assert_allclose(row_sums, jnp.zeros(2), atol=1e-12)

    @pytest.mark.parametrize(
        ("K", "match"),
        [
            (jnp.array([[False, True], [True, False]]), "boolean"),
            (jnp.array([[0.0, 1.0j], [1.0, 0.0]]), "real-valued"),
            (jnp.array([0.0, 1.0]), "square"),
            (jnp.ones((2, 3)), "square"),
        ],
    )
    def test_laplacian_rejects_non_real_or_malformed_coupling(self, K, match):
        with pytest.raises(ValueError, match=match):
            coupling_laplacian(K)

    def test_saf_order_parameter_in_range(self):
        K = jnp.ones((4, 4)) * 5.0 - jnp.eye(4) * 5.0
        omegas = jnp.array([1.0, 1.1, 0.9, 1.05])
        r = float(saf_order_parameter(K, omegas))
        assert 0.0 <= r <= 1.0

    def test_saf_identical_frequencies_are_perfectly_aligned(self):
        K = jnp.ones((4, 4)) - jnp.eye(4)
        omegas = jnp.ones(4) * 2.0

        r = float(saf_order_parameter(K, omegas, solver="eigh"))

        assert r == pytest.approx(1.0, abs=1e-7)

    def test_saf_cg_solver_matches_eigendecomposition_contract(self):
        K = jnp.array(
            [
                [0.0, 1.4, 0.6, 0.3],
                [1.4, 0.0, 0.8, 0.5],
                [0.6, 0.8, 0.0, 1.1],
                [0.3, 0.5, 1.1, 0.0],
            ]
        )
        omegas = jnp.array([1.0, 1.2, 0.85, 1.05])

        exact = saf_order_parameter(K, omegas, solver="eigh")
        iterative = saf_order_parameter(K, omegas, solver="cg", cg_tol=1e-6)

        np.testing.assert_allclose(iterative, exact, atol=2e-3)

    def test_saf_auto_uses_cg_path_above_size_limit(self):
        K = jnp.ones((4, 4)) * 2.0 - jnp.eye(4) * 2.0
        omegas = jnp.array([1.0, 1.2, 0.9, 1.05])

        forced_auto = saf_order_parameter(K, omegas, exact_size_limit=2, cg_tol=1e-6)
        forced_cg = saf_order_parameter(K, omegas, solver="cg", cg_tol=1e-6)

        np.testing.assert_allclose(forced_auto, forced_cg, atol=1e-12)

    def test_saf_rejects_unknown_solver(self):
        K = jnp.ones((3, 3)) - jnp.eye(3)
        omegas = jnp.array([1.0, 1.1, 0.9])

        with pytest.raises(ValueError, match="solver"):
            saf_order_parameter(K, omegas, solver="not-a-solver")

    @pytest.mark.parametrize(
        ("K", "omegas", "match"),
        [
            (
                jnp.array([[False, True], [True, False]]),
                jnp.array([1.0, 1.1]),
                "K must not contain boolean",
            ),
            (
                jnp.array([[0.0, 1.0j], [1.0, 0.0]]),
                jnp.array([1.0, 1.1]),
                "K must contain real-valued",
            ),
            (
                jnp.ones((2, 3)),
                jnp.array([1.0, 1.1]),
                "K must be a square",
            ),
            (
                jnp.ones((2, 2)) - jnp.eye(2),
                jnp.array([[1.0, 1.1]]),
                "omegas must be a one-dimensional",
            ),
            (
                jnp.ones((2, 2)) - jnp.eye(2),
                jnp.array([1.0 + 0.0j, 1.1 + 0.0j]),
                "omegas must contain real-valued",
            ),
            (
                jnp.ones((2, 2)) - jnp.eye(2),
                jnp.array([1.0, 1.1, 1.2]),
                "omegas length must match",
            ),
        ],
    )
    def test_saf_rejects_non_real_or_malformed_public_inputs(self, K, omegas, match):
        with pytest.raises(ValueError, match=match):
            saf_order_parameter(K, omegas)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"eps": 0.0}, "eps"),
            ({"exact_size_limit": True}, "exact_size_limit"),
            ({"exact_size_limit": 0}, "exact_size_limit"),
            ({"cg_tol": -1.0}, "cg_tol"),
            ({"cg_maxiter": 0}, "cg_maxiter"),
        ],
    )
    def test_saf_rejects_invalid_solver_controls(self, kwargs, match):
        K = jnp.ones((2, 2)) - jnp.eye(2)
        omegas = jnp.array([1.0, 1.1])

        with pytest.raises(ValueError, match=match):
            saf_order_parameter(K, omegas, **kwargs)

    def test_saf_loss_negative_r(self):
        K = jnp.ones((4, 4)) * 5.0 - jnp.eye(4) * 5.0
        omegas = jnp.array([1.0, 1.1, 0.9, 1.05])
        loss = float(saf_loss(K, omegas))
        assert loss <= 0.0

    def test_saf_loss_with_budget(self):
        K = jnp.ones((4, 4)) * 5.0 - jnp.eye(4) * 5.0
        omegas = jnp.array([1.0, 1.1, 0.9, 1.05])
        loss_no_budget = float(saf_loss(K, omegas, budget=0.0))
        loss_budget = float(saf_loss(K, omegas, budget=1.0, budget_weight=1.0))
        assert loss_budget > loss_no_budget

    def test_saf_loss_zero_budget_disables_budget_penalty(self):
        K = jnp.ones((3, 3)) * 0.2 - jnp.eye(3) * 0.2
        omegas = jnp.array([1.0, 1.05, 0.95])

        no_budget = float(saf_loss(K, omegas, budget=0.0))

        assert no_budget == pytest.approx(float(saf_loss(K, omegas)), abs=1e-12)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"budget": True}, "budget"),
            ({"budget": math.nan}, "budget"),
            ({"budget": -1.0}, "budget"),
            ({"budget_weight": False}, "budget_weight"),
            ({"budget_weight": -0.1}, "budget_weight"),
        ],
    )
    def test_saf_loss_rejects_invalid_budget_controls(self, kwargs, match):
        K = jnp.ones((2, 2)) - jnp.eye(2)
        omegas = jnp.array([1.0, 1.1])

        with pytest.raises(ValueError, match=match):
            saf_loss(K, omegas, **kwargs)


# ── Differentiability (JAX grad) ──


class TestDifferentiability:
    def test_order_parameter_has_grad(self):
        phases = jnp.array([0.1, 0.2, 0.3, 0.4])
        grad_fn = jax.grad(lambda p: order_parameter(p).real)
        g = grad_fn(phases)
        assert g.shape == (4,)
        assert jnp.all(jnp.isfinite(g))

    def test_saf_loss_has_grad(self):
        # Non-degenerate coupling to avoid repeated eigenvalues in eigh grad
        K = jnp.array(
            [
                [0.0, 1.0, 0.5, 0.2],
                [1.0, 0.0, 0.8, 0.3],
                [0.5, 0.8, 0.0, 1.2],
                [0.2, 0.3, 1.2, 0.0],
            ]
        )
        omegas = jnp.array([1.0, 1.1, 0.9, 1.05])
        grad_fn = jax.grad(lambda k: saf_loss(k, omegas))
        g = grad_fn(K)
        assert g.shape == (4, 4)
        assert jnp.all(jnp.isfinite(g))

    def test_saf_cg_loss_has_grad(self):
        K = jnp.array(
            [
                [0.0, 1.0, 0.5, 0.2],
                [1.0, 0.0, 0.8, 0.3],
                [0.5, 0.8, 0.0, 1.2],
                [0.2, 0.3, 1.2, 0.0],
            ]
        )
        omegas = jnp.array([1.0, 1.1, 0.9, 1.05])
        grad_fn = jax.grad(lambda k: saf_loss(k, omegas, solver="cg"))
        g = grad_fn(K)
        assert g.shape == (4, 4)
        assert jnp.all(jnp.isfinite(g))


# ── Empty/Edge Inputs ──


class TestEmptyInputs:
    def test_single_oscillator_kuramoto(self):
        phases = jnp.array([1.0])
        omegas = jnp.array([1.0])
        K = jnp.zeros((1, 1))
        result = kuramoto_step(phases, omegas, K, 0.01)
        assert result.shape == (1,)
        assert jnp.all(jnp.isfinite(result))

    def test_single_oscillator_order_parameter(self):
        r = float(order_parameter(jnp.array([0.5])))
        assert abs(r - 1.0) < 1e-10

    def test_two_oscillators_plv(self):
        traj = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        plv_mat = plv(traj)
        assert plv_mat.shape == (2, 2)

    def test_zero_coupling_no_interaction(self):
        n = 8
        phases = jnp.linspace(0, 5.0, n)
        omegas = jnp.zeros(n)
        K = jnp.zeros((n, n))
        result = kuramoto_step(phases, omegas, K, 0.01)
        np.testing.assert_allclose(result, phases % TWO_PI, atol=1e-12)

    def test_zero_dt_no_change(self):
        phases = jnp.array([0.1, 0.5, 1.0, 1.5])
        omegas = jnp.ones(4)
        K = jnp.ones((4, 4))
        result = kuramoto_step(phases, omegas, K, 0.0)
        np.testing.assert_allclose(result, phases, atol=1e-12)


# ── Performance Budget ──


class TestPerformance:
    def test_kuramoto_step_budget_n64(self):
        """Single kuramoto_step(N=64) must stay inside the per-platform budget."""
        import time

        n = 64
        phases = jnp.zeros(n)
        omegas = jnp.ones(n)
        K = jnp.eye(n)

        # JIT warm-up
        kuramoto_step(phases, omegas, K, 0.01)
        kuramoto_step(phases, omegas, K, 0.01)

        t0 = time.perf_counter()
        for _ in range(100):
            kuramoto_step(phases, omegas, K, 0.01)
        elapsed = (time.perf_counter() - t0) / 100
        budget = 0.02 if sys.platform == "win32" else 0.01
        assert elapsed < budget, (
            f"kuramoto_step(N=64) = {elapsed * 1000:.2f}ms "
            f"> {budget * 1000:.0f}ms on {sys.platform}"
        )

    def test_order_parameter_budget(self):
        """order_parameter(N=64) must complete in <5ms (JAX per-call overhead)."""
        import time

        phases = jnp.zeros(64)
        # JIT warm-up
        order_parameter(phases)
        order_parameter(phases)

        t0 = time.perf_counter()
        for _ in range(100):
            order_parameter(phases)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005, f"order_parameter(N=64) = {elapsed * 1000:.2f}ms > 5ms"
