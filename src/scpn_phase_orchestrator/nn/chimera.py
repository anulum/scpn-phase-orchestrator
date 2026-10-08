# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable chimera state detection

"""JAX-based chimera state detection for coupled oscillator networks.

Chimera states are spatiotemporal patterns where synchronised and
incoherent domains coexist (Kuramoto & Battogtokh 2002). This module
provides instantaneous local-coherence diagnostics and phase gradients on
fixed adjacency. Hard ``K != 0`` support decisions give zero coupling-amplitude
gradients away from topology changes; they do not enable gradient-based
topology search. A vanishing neighbourhood phasor has a nondifferentiable
magnitude, and threshold masks are discrete. A snapshot does not certify a
dynamical chimera state.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def local_order_parameter(
    phases: jax.Array,
    K: jax.Array,
) -> jax.Array:
    """Local Kuramoto order parameter R_i for each oscillator.

    R_i = |mean(exp(i·Δθ_j)) for neighbours j of i|

    Neighbours are all nonzero entries in K, including negative and self edges.
    This differs from monitor.chimera's positive non-self adjacency. Vectorised
    without Python loops; empty neighbourhoods return zero.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases.
    K : jax.Array
        (N, N) coupling matrix (nonzero = neighbour).

    Returns
    -------
    jax.Array
        (N,) local order parameters in [0, 1].
    """
    mask = (K != 0).astype(jnp.float32)
    # The centre phasor has unit magnitude, so remove it before summing.
    # This preserves the nonzero adjacency model, including signed/self edges,
    # and avoids overflow in finite unwrapped phase differences.
    cos_diff = jnp.cos(phases)[jnp.newaxis, :] * mask
    sin_diff = jnp.sin(phases)[jnp.newaxis, :] * mask
    n_neighbours = jnp.sum(mask, axis=1).clip(min=1.0)
    mean_cos = jnp.sum(cos_diff, axis=1) / n_neighbours
    mean_sin = jnp.sum(sin_diff, axis=1) / n_neighbours
    return jnp.sqrt(mean_cos**2 + mean_sin**2)


def chimera_index(
    phases: jax.Array,
    K: jax.Array,
) -> jax.Array:
    """Scalar chimera index: variance of local order parameters.

    Variance summarizes instantaneous heterogeneity, without certifying
    dynamical coexistence. Zero variance means equal local-order values.
    Phase gradients are defined on fixed adjacency away from zero phasors;
    hard support decisions do not provide coupling-topology gradients.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases.
    K : jax.Array
        (N, N) coupling matrix.

    Returns
    -------
    jax.Array
        Scalar local-order variance; the nonempty finite range is [0, 0.25].
    """
    R_local = local_order_parameter(phases, K)
    return jnp.var(R_local)


def detect_chimera(
    phases: jax.Array,
    K: jax.Array,
    coherent_threshold: float = 0.8,
    incoherent_threshold: float = 0.3,
) -> tuple[jax.Array, jax.Array]:
    """Classify oscillators as coherent or incoherent.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases.
    K : jax.Array
        (N, N) coupling matrix.
    coherent_threshold : float
        R_i greater than or equal to this → coherent.
    incoherent_threshold : float
        R_i less than or equal to this → incoherent.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (coherent_mask, incoherent_mask): (N,) boolean arrays.
    """
    R_local = local_order_parameter(phases, K)
    coherent = R_local >= coherent_threshold
    incoherent = R_local <= incoherent_threshold
    return coherent, incoherent
