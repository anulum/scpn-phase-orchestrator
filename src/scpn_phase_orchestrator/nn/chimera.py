# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable chimera state detection

"""JAX-based chimera state detection for coupled oscillator networks.

Chimera states are spatiotemporal patterns where synchronised and
incoherent domains coexist (Kuramoto & Battogtokh 2002). This module
provides differentiable detection, enabling gradient-based search for
chimera-producing coupling matrices.

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

    Neighbours defined by nonzero entries in K. Vectorised — no Python loops.

    Args:
        phases: (N,) oscillator phases
        K: (N, N) coupling matrix (nonzero = neighbour)

    Returns:
        (N,) local order parameters in [0, 1]
    """
    mask = (K != 0).astype(jnp.float32)
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]  # (N, N)
    # Complex phasors weighted by adjacency
    cos_diff = jnp.cos(diff) * mask
    sin_diff = jnp.sin(diff) * mask
    n_neighbours = jnp.sum(mask, axis=1).clip(min=1.0)
    mean_cos = jnp.sum(cos_diff, axis=1) / n_neighbours
    mean_sin = jnp.sum(sin_diff, axis=1) / n_neighbours
    return jnp.sqrt(mean_cos**2 + mean_sin**2)


def chimera_index(
    phases: jax.Array,
    K: jax.Array,
) -> jax.Array:
    """Scalar chimera index: variance of local order parameters.

    High variance = coexistence of coherent (R≈1) and incoherent (R≈0)
    domains. Zero variance = uniform state (either all sync or all desync).
    Differentiable.

    Args:
        phases: (N,) oscillator phases
        K: (N, N) coupling matrix

    Returns:
        Scalar chimera index (higher = more chimera-like)
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

    Args:
        phases: (N,) oscillator phases
        K: (N, N) coupling matrix
        coherent_threshold: R_i above this → coherent
        incoherent_threshold: R_i below this → incoherent

    Returns:
        (coherent_mask, incoherent_mask): (N,) boolean arrays
    """
    R_local = local_order_parameter(phases, K)
    coherent = R_local >= coherent_threshold
    incoherent = R_local <= incoherent_threshold
    return coherent, incoherent
