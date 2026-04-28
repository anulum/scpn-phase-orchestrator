# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable simplicial Kuramoto layer (equinox)

"""Equinox module wrapping simplicial (3-body) Kuramoto dynamics.

Extends KuramotoLayer with a learnable 3-body coupling strength sigma2.
When sigma2=0, reduces to standard pairwise Kuramoto. Nonzero sigma2
produces explosive (first-order) synchronization transitions
(Gambuzza et al. 2023, Nature Physics).

First differentiable 3-body Kuramoto layer in open source.

Requires: jax>=0.4, equinox>=0.11
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .functional import order_parameter, simplicial_forward


class SimplicialKuramotoLayer(eqx.Module):
    """Differentiable simplicial Kuramoto layer with 3-body interactions.

    Learnable parameters:
        K: (n, n) pairwise coupling matrix
        omegas: (n,) natural frequencies
        sigma2: scalar 3-body coupling strength

    Static config:
        n_steps: integration steps per forward pass
        dt: integration timestep
    """

    K: jax.Array
    omegas: jax.Array
    sigma2: jax.Array
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        n_steps: int = 50,
        dt: float = 0.01,
        K_scale: float = 0.1,
        sigma2_init: float = 0.0,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        raw = K_scale * jax.random.normal(k1, (n, n))
        self.K = (raw + raw.T) / 2.0
        self.omegas = jax.random.normal(k2, (n,))
        self.sigma2 = jnp.array(sigma2_init)
        self.n_steps = n_steps
        self.dt = dt
        self.n = n

    @eqx.filter_jit
    def __call__(self, phases: jax.Array) -> jax.Array:
        """Run simplicial Kuramoto dynamics on input phases.

        Args:
            phases: (n,) initial phase angles in [0, 2pi)

        Returns:
            (n,) phase angles after n_steps of integration
        """
        final, _ = simplicial_forward(
            phases,
            self.omegas,
            self.K,
            self.dt,
            self.n_steps,
            sigma2=self.sigma2,
        )
        return final

    @eqx.filter_jit
    def forward_with_trajectory(
        self,
        phases: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run dynamics and return both final state and full trajectory.

        Args:
            phases: (n,) initial phase angles

        Returns:
            Tuple of (final_phases, trajectory) where trajectory is (n_steps, n)
        """
        return simplicial_forward(
            phases,
            self.omegas,
            self.K,
            self.dt,
            self.n_steps,
            sigma2=self.sigma2,
        )

    @eqx.filter_jit
    def sync_score(self, phases: jax.Array) -> jax.Array:
        """Run dynamics and return final synchronization (order parameter R).

        Args:
            phases: (n,) initial phases

        Returns:
            Scalar R in [0, 1]
        """
        final = self(phases)
        return order_parameter(final)
