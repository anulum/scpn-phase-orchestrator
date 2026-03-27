# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable Kuramoto layer (equinox)

"""Equinox module wrapping Kuramoto dynamics as a differentiable layer.

The KuramotoLayer maps input features to oscillator phases, runs N steps
of Kuramoto dynamics with a learnable coupling matrix K, and returns the
synchronized phase representation. Fully differentiable via JAX autodiff.

Requires: jax>=0.4, equinox>=0.11
"""

from __future__ import annotations

import equinox as eqx
import jax

from .functional import kuramoto_forward, kuramoto_forward_masked, order_parameter


class KuramotoLayer(eqx.Module):
    """Differentiable Kuramoto oscillator layer.

    Learnable parameters:
        K: (n, n) coupling matrix — controls which oscillators synchronize
        omegas: (n,) natural frequencies

    Static config:
        n_steps: integration steps per forward pass
        dt: integration timestep
    """

    K: jax.Array
    omegas: jax.Array
    mask: jax.Array | None = eqx.field(static=True, default=None)
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        n_steps: int = 50,
        dt: float = 0.01,
        K_scale: float = 0.1,
        mask: jax.Array | None = None,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        raw = K_scale * jax.random.normal(k1, (n, n))
        self.K = (raw + raw.T) / 2.0
        self.omegas = jax.random.normal(k2, (n,))
        self.mask = mask
        self.n_steps = n_steps
        self.dt = dt
        self.n = n

    @eqx.filter_jit
    def __call__(self, phases: jax.Array) -> jax.Array:
        """Run Kuramoto dynamics on input phases.

        Args:
            phases: (n,) initial phase angles in [0, 2pi)

        Returns:
            (n,) phase angles after n_steps of Kuramoto integration
        """
        if self.mask is not None:
            final, _ = kuramoto_forward_masked(
                phases,
                self.omegas,
                self.K,
                self.mask,
                self.dt,
                self.n_steps,
            )
        else:
            final, _ = kuramoto_forward(
                phases,
                self.omegas,
                self.K,
                self.dt,
                self.n_steps,
            )
        return final

    @eqx.filter_jit
    def forward_with_trajectory(self, phases: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run dynamics and return both final state and full trajectory.

        Args:
            phases: (n,) initial phase angles

        Returns:
            Tuple of (final_phases, trajectory) where trajectory is (n_steps, n)
        """
        if self.mask is not None:
            return kuramoto_forward_masked(
                phases,
                self.omegas,
                self.K,
                self.mask,
                self.dt,
                self.n_steps,
            )
        return kuramoto_forward(phases, self.omegas, self.K, self.dt, self.n_steps)

    @eqx.filter_jit
    def sync_score(self, phases: jax.Array) -> jax.Array:
        """Run dynamics and return final synchronization (order parameter R).

        Useful as a differentiable loss target: maximize R for sync,
        minimize R for desync.

        Args:
            phases: (n,) initial phases

        Returns:
            Scalar R in [0, 1]
        """
        final = self(phases)
        return order_parameter(final)
