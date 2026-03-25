# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Oscillator Ising Machine (OIM)

"""Kuramoto-based combinatorial optimization via phase clustering.

Maps NP-hard problems (graph coloring, max-cut, QUBO) to coupled
oscillator dynamics. Oscillators settle into k distinct phase clusters,
each cluster corresponding to a color/partition.

The coupling function is modified from standard sin(Δθ) to produce
equidistant phase clusters (Nature Scientific Reports 2017, Böhm &
Schumacher 2020). GPU-accelerated via JAX.

First open-source oscillator Ising machine simulator.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .functional import TWO_PI


def _oim_deriv(
    phases: jax.Array,
    adjacency: jax.Array,
    n_colors: int,
    coupling_strength: float,
) -> jax.Array:
    """Derivative for OIM coloring dynamics.

    Uses sin(n_colors * Δθ) coupling: oscillators connected by an edge
    repel from the SAME phase cluster and attract to DIFFERENT clusters
    spaced at 2π/n_colors intervals.
    """
    # Gradient descent on E = Σ cos(n·Δθ): connected nodes repel
    # from same-phase cluster. dE/dθ_i = -n·Σ_j A_ij sin(n(θ_i-θ_j))
    diff = phases[:, jnp.newaxis] - phases[jnp.newaxis, :]
    coupling = jnp.sum(adjacency * jnp.sin(n_colors * diff), axis=1)
    return coupling_strength * coupling


def oim_step(
    phases: jax.Array,
    adjacency: jax.Array,
    n_colors: int,
    dt: float,
    coupling_strength: float = 1.0,
) -> jax.Array:
    """Single step of OIM coloring dynamics.

    Args:
        phases: (N,) oscillator phases
        adjacency: (N, N) graph adjacency matrix (1 = edge, 0 = no edge)
        n_colors: number of colors (phase clusters)
        dt: integration timestep
        coupling_strength: overall coupling scale

    Returns:
        (N,) updated phases
    """
    dphi = _oim_deriv(phases, adjacency, n_colors, coupling_strength)
    return (phases + dt * dphi) % TWO_PI


def oim_forward(
    phases: jax.Array,
    adjacency: jax.Array,
    n_colors: int,
    dt: float,
    n_steps: int,
    coupling_strength: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Run OIM dynamics for n_steps, returning final phases and trajectory.

    Args:
        phases: (N,) initial random phases
        adjacency: (N, N) graph adjacency matrix
        n_colors: number of colors
        dt: timestep
        n_steps: number of integration steps
        coupling_strength: overall coupling scale

    Returns:
        (final_phases, trajectory) where trajectory is (n_steps, N)
    """

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = oim_step(carry, adjacency, n_colors, dt, coupling_strength)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


def extract_coloring(phases: jax.Array, n_colors: int) -> jax.Array:
    """Extract integer color assignment from oscillator phases.

    Maps each phase to the nearest cluster center at 2πk/n_colors.

    Args:
        phases: (N,) oscillator phases in [0, 2π)
        n_colors: number of colors

    Returns:
        (N,) integer color labels in {0, 1, ..., n_colors-1}
    """
    # Cluster centers at 2πk/n_colors
    bucket_size = TWO_PI / n_colors
    result: jax.Array = jnp.floor(phases / bucket_size).astype(jnp.int32) % n_colors
    return result


def coloring_violations(
    colors: jax.Array,
    adjacency: jax.Array,
) -> jax.Array:
    """Count edges where both endpoints have the same color.

    Args:
        colors: (N,) integer color labels
        adjacency: (N, N) adjacency matrix

    Returns:
        Scalar: number of violated edges
    """
    same_color = (colors[jnp.newaxis, :] == colors[:, jnp.newaxis]).astype(jnp.float32)
    # Count upper triangle only (each edge once)
    violations = jnp.sum(jnp.triu(adjacency * same_color, k=1))
    result: jax.Array = violations
    return result


def coloring_energy(
    phases: jax.Array,
    adjacency: jax.Array,
    n_colors: int,
) -> jax.Array:
    """Continuous energy function for the coloring problem.

    E = Σ_{(i,j)∈E} cos(n_colors · (θ_i - θ_j))

    Minimized when connected nodes are in different phase clusters
    (cos(n·Δθ) = -1 when Δθ = π/n, i.e., maximally separated).
    Differentiable for gradient-based optimization.

    Args:
        phases: (N,) oscillator phases
        adjacency: (N, N) adjacency matrix
        n_colors: number of colors

    Returns:
        Scalar energy (lower = better coloring)
    """
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    cost_matrix = jnp.cos(n_colors * diff)
    energy: jax.Array = jnp.sum(jnp.triu(adjacency * cost_matrix, k=1))
    return energy
