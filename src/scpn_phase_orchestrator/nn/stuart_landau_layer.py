# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable Stuart-Landau layer (equinox)

"""Equinox module wrapping Stuart-Landau dynamics as a differentiable layer.

Unlike the Kuramoto-only KuramotoLayer, this layer has both phase AND
amplitude dynamics, enabling representation of feature presence/absence
(amplitude) alongside binding relationships (phase).

Solves AKOrN's limitations: amplitude allows memory, no N>32 degradation,
supercritical/subcritical bifurcation as a natural activation gate.

Requires: jax>=0.4, equinox>=0.11
"""

from __future__ import annotations

import equinox as eqx
import jax

from .functional import order_parameter, stuart_landau_forward


class StuartLandauLayer(eqx.Module):
    """Differentiable Stuart-Landau oscillator layer.

    Learnable parameters:
        K: (n, n) phase coupling matrix
        K_r: (n, n) amplitude coupling matrix
        omegas: (n,) natural frequencies
        mu: (n,) bifurcation parameters (>0: supercritical, <0: subcritical)

    Static config:
        n_steps: integration steps per forward pass
        dt: integration timestep
        epsilon: amplitude coupling strength
    """

    K: jax.Array
    K_r: jax.Array
    omegas: jax.Array
    mu: jax.Array
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        n_steps: int = 50,
        dt: float = 0.01,
        K_scale: float = 0.1,
        epsilon: float = 1.0,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        raw = K_scale * jax.random.normal(k1, (n, n))
        self.K = (raw + raw.T) / 2.0
        raw_r = K_scale * jax.random.normal(k2, (n, n))
        self.K_r = (raw_r + raw_r.T) / 2.0
        self.omegas = jax.random.normal(k3, (n,))
        # mu > 0 by default: supercritical regime (oscillators have amplitude)
        self.mu = 0.5 + 0.1 * jax.random.normal(k4, (n,))
        self.n_steps = n_steps
        self.dt = dt
        self.epsilon = epsilon
        self.n = n

    @property
    def coupling(self) -> jax.Array:
        """Symmetric phase-coupling matrix used by the dynamics ``(K + Kᵀ)/2``.

        Phase coupling is undirected, so the loss depends only on the symmetric
        part; the gradient w.r.t. ``K`` is therefore symmetric and training from
        a symmetric ``K`` keeps it symmetric instead of drifting directed.

        Returns
        -------
        jax.Array
            Symmetric phase-coupling matrix used by the dynamics ``(K + Kᵀ)/2``.
        """
        return (self.K + self.K.T) * 0.5

    @property
    def coupling_r(self) -> jax.Array:
        """Symmetric amplitude-coupling matrix ``(K_r + K_rᵀ)/2`` (see ``coupling``).

        Returns
        -------
        jax.Array
            Symmetric amplitude-coupling matrix ``(K_r + K_rᵀ)/2`` (see ``coupling``).
        """
        return (self.K_r + self.K_r.T) * 0.5

    @eqx.filter_jit
    def __call__(
        self, phases: jax.Array, amplitudes: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Run Stuart-Landau dynamics on input state.

        Parameters
        ----------
        phases : jax.Array
            (n,) initial phase angles in [0, 2pi).
        amplitudes : jax.Array
            (n,) initial amplitudes (r >= 0).

        Returns
        -------
        tuple[jax.Array, jax.Array]
            (final_phases, final_amplitudes).
        """
        fp, fr, _, _ = stuart_landau_forward(
            phases,
            amplitudes,
            self.omegas,
            self.mu,
            self.coupling,
            self.coupling_r,
            self.dt,
            self.n_steps,
            self.epsilon,
        )
        return fp, fr

    @eqx.filter_jit
    def forward_with_trajectory(
        self, phases: jax.Array, amplitudes: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Run dynamics and return full trajectories.

        Returns
        -------
            (final_phases, final_amplitudes, phase_trajectory, amplitude_trajectory)

        Parameters
        ----------
        phases : jax.Array
            Oscillator phases in radians, shape ``(N,)``.
        amplitudes : jax.Array
            Oscillator amplitudes, shape ``(N,)``.
        """
        return stuart_landau_forward(
            phases,
            amplitudes,
            self.omegas,
            self.mu,
            self.coupling,
            self.coupling_r,
            self.dt,
            self.n_steps,
            self.epsilon,
        )

    @eqx.filter_jit
    def sync_score(self, phases: jax.Array, amplitudes: jax.Array) -> jax.Array:
        """Run dynamics and return final synchronization (order parameter R).

        Parameters
        ----------
        phases : jax.Array
            (n,) initial phases.
        amplitudes : jax.Array
            (n,) initial amplitudes.

        Returns
        -------
        jax.Array
            Scalar R in [0, 1].
        """
        fp, _ = self(phases, amplitudes)
        return order_parameter(fp)

    @eqx.filter_jit
    def mean_amplitude(self, phases: jax.Array, amplitudes: jax.Array) -> jax.Array:
        """Run dynamics and return mean final amplitude.

        Useful as a differentiable activity measure: high mean amplitude
        means oscillators are active (supercritical), low means quiescent.

        Parameters
        ----------
        phases : jax.Array
            (n,) initial phases.
        amplitudes : jax.Array
            (n,) initial amplitudes.

        Returns
        -------
        jax.Array
            Scalar mean amplitude.
        """
        import jax.numpy as jnp

        _, fr = self(phases, amplitudes)
        return jnp.mean(fr)
