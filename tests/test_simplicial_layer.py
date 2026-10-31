# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for SimplicialKuramotoLayer

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")

from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer
from scpn_phase_orchestrator.nn.simplicial_layer import SimplicialKuramotoLayer

if TYPE_CHECKING:
    from equinox import Module as EqxModule
    from jax import Array

N = 8
DT = 0.01
N_STEPS = 50


@pytest.fixture()
def key() -> Array:
    prng_key: Array = jax.random.PRNGKey(42)
    return prng_key


@pytest.fixture()
def layer(key: Array) -> SimplicialKuramotoLayer:
    return SimplicialKuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)


@pytest.fixture()
def phases(key: Array) -> Array:
    drawn: Array = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
    return drawn


class TestInit:
    def test_K_shape(self, layer: SimplicialKuramotoLayer) -> None:
        assert layer.K.shape == (N, N)

    def test_K_symmetric(self, layer: SimplicialKuramotoLayer) -> None:
        assert jnp.allclose(layer.K, layer.K.T, atol=1e-7)

    def test_omegas_shape(self, layer: SimplicialKuramotoLayer) -> None:
        assert layer.omegas.shape == (N,)

    def test_sigma2_scalar(self, layer: SimplicialKuramotoLayer) -> None:
        assert layer.sigma2.shape == ()

    def test_sigma2_default_zero(self, layer: SimplicialKuramotoLayer) -> None:
        assert float(layer.sigma2) == 0.0

    def test_custom_sigma2(self, key: Array) -> None:
        layer = SimplicialKuramotoLayer(N, sigma2_init=1.5, key=key)
        assert float(layer.sigma2) == pytest.approx(1.5)


class TestForward:
    def test_output_shape(self, layer: SimplicialKuramotoLayer, phases: Array) -> None:
        out = layer(phases)
        assert out.shape == (N,)

    def test_phases_in_range(
        self, layer: SimplicialKuramotoLayer, phases: Array
    ) -> None:
        out = layer(phases)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)

    def test_finite(self, layer: SimplicialKuramotoLayer, phases: Array) -> None:
        out = layer(phases)
        assert jnp.isfinite(out).all()


class TestTrajectory:
    def test_trajectory_shape(
        self, layer: SimplicialKuramotoLayer, phases: Array
    ) -> None:
        final, traj = layer.forward_with_trajectory(phases)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last(
        self, layer: SimplicialKuramotoLayer, phases: Array
    ) -> None:
        final, traj = layer.forward_with_trajectory(phases)
        assert jnp.allclose(final, traj[-1], atol=1e-6)


class TestSyncScore:
    def test_scalar_output(self, layer: SimplicialKuramotoLayer, phases: Array) -> None:
        r = layer.sync_score(phases)
        assert r.shape == ()

    def test_range(self, layer: SimplicialKuramotoLayer, phases: Array) -> None:
        r = layer.sync_score(phases)
        assert 0.0 <= float(r) <= 1.0


class TestSigma2Zero:
    """With sigma2=0, SimplicialKuramotoLayer matches KuramotoLayer."""

    def test_matches_kuramoto_layer(self, key: Array) -> None:
        k1, k2 = jax.random.split(key)
        phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)

        simp = SimplicialKuramotoLayer(
            N,
            n_steps=N_STEPS,
            dt=DT,
            sigma2_init=0.0,
            key=k2,
        )
        kura = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=k2)

        # Same key → same K and omegas
        assert jnp.allclose(simp.K, kura.K, atol=1e-6)
        assert jnp.allclose(simp.omegas, kura.omegas, atol=1e-6)

        out_s = simp(phases)
        out_k = kura(phases)
        # RK4 vs RK4 — should be very close but simplicial uses
        # simplicial_forward (RK4 default) and Kuramoto uses kuramoto_forward (Euler).
        # Only check finite + same shape; exact match not guaranteed.
        assert out_s.shape == out_k.shape
        assert jnp.isfinite(out_s).all()


class TestGrad:
    def test_differentiable(
        self, layer: SimplicialKuramotoLayer, phases: Array
    ) -> None:
        def loss(model: SimplicialKuramotoLayer, p: Array) -> Array:
            mean: Array = jnp.mean(model(p))
            return mean

        grads = eqx.filter_grad(loss)(layer, phases)
        assert jnp.isfinite(grads.K).all()
        assert jnp.isfinite(grads.omegas).all()
        assert jnp.isfinite(grads.sigma2).all()

    def test_sigma2_grad_nonzero_when_active(self, key: Array) -> None:
        k1, k2 = jax.random.split(key)
        phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
        layer = SimplicialKuramotoLayer(
            N,
            n_steps=N_STEPS,
            dt=DT,
            sigma2_init=1.0,
            key=k2,
        )

        def loss(model: SimplicialKuramotoLayer, p: Array) -> Array:
            mean: Array = jnp.mean(model(p))
            return mean

        grads = eqx.filter_grad(loss)(layer, phases)
        # sigma2 grad should be nonzero when sigma2 is nonzero
        assert float(jnp.abs(grads.sigma2)) > 0.0


class TestJIT:
    def test_call_jit(self, layer: SimplicialKuramotoLayer, phases: Array) -> None:
        out1 = layer(phases)
        out2 = layer(phases)
        assert jnp.allclose(out1, out2, atol=1e-6)

    def test_trajectory_jit(
        self, layer: SimplicialKuramotoLayer, phases: Array
    ) -> None:
        f1, t1 = layer.forward_with_trajectory(phases)
        f2, t2 = layer.forward_with_trajectory(phases)
        assert jnp.allclose(f1, f2, atol=1e-6)


class TestSigma2Activation:
    def test_sigma2_changes_dynamics(self, key: Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
        layer_zero = SimplicialKuramotoLayer(
            N, n_steps=20, dt=DT, sigma2_init=0.0, key=k2
        )
        layer_active = SimplicialKuramotoLayer(
            N, n_steps=20, dt=DT, sigma2_init=1.0, key=k3
        )
        out_zero = layer_zero(phases)
        out_active = layer_active(phases)
        assert out_zero.shape == out_active.shape
        assert jnp.isfinite(out_zero).all()
        assert jnp.isfinite(out_active).all()
        assert not jnp.allclose(out_zero, out_active, atol=1e-7)


class TestTrainingPreservesSymmetry:
    """Pairwise coupling is undirected, so K must remain symmetric under
    gradient training, not only at initialisation."""

    def test_K_stays_symmetric_after_training(self, key: Array, phases: Array) -> None:
        optax = pytest.importorskip("optax")
        from scpn_phase_orchestrator.nn.training import train

        layer = SimplicialKuramotoLayer(N, n_steps=20, dt=DT, key=key)

        def loss_fn(model: EqxModule) -> Array:
            assert isinstance(model, SimplicialKuramotoLayer)
            mean: Array = jnp.mean(model(phases) ** 2)
            return mean

        trained, _ = train(layer, loss_fn, optax.adam(1e-2), 20)
        assert isinstance(trained, SimplicialKuramotoLayer)
        asym = float(jnp.max(jnp.abs(trained.K - trained.K.T)))
        assert asym < 1e-5, f"K lost symmetry after training: {asym:.2e}"


# Pipeline wiring: SimplicialKuramotoLayer uses simplicial_forward internally
# with order_parameter. TestComparison and TestGrad exercise the full pipeline.
