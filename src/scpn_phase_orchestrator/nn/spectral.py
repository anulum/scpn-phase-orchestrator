# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral analysis of coupling matrices

"""Differentiable spectral metrics for coupling matrix analysis.

All functions are differentiable via jnp.linalg.eigh, enabling
gradient-based topology optimisation: find the sparsest K that
maintains synchronisability above a target threshold.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def laplacian_spectrum(K: jax.Array) -> jax.Array:
    """Sorted eigenvalues of the graph Laplacian L = D - K.

    Args:
        K: (N, N) symmetric coupling matrix (non-negative weights)

    Returns:
        (N,) eigenvalues in ascending order. First is ~0 (connected graph).
    """
    D = jnp.diag(jnp.sum(K, axis=1))
    L = D - K
    return jnp.linalg.eigh(L)[0]


def algebraic_connectivity(K: jax.Array) -> jax.Array:
    """Second-smallest Laplacian eigenvalue (Fiedler value).

    Measures how well-connected the network is. Zero iff disconnected.
    Differentiable — gradient flows through eigh.

    Args:
        K: (N, N) symmetric coupling matrix

    Returns:
        Scalar lambda_2
    """
    return laplacian_spectrum(K)[1]


def eigenratio(K: jax.Array) -> jax.Array:
    """Ratio lambda_N / lambda_2 (synchronisability metric).

    Lower eigenratio = more synchronisable (Barahona & Pecora 2002).
    The MSF (master stability function) approach shows that coupled
    oscillators synchronise when all transverse eigenvalues fall
    within the MSF stability interval.

    Args:
        K: (N, N) symmetric coupling matrix

    Returns:
        Scalar lambda_N / lambda_2
    """
    eigs = laplacian_spectrum(K)
    lambda_2 = eigs[1]
    lambda_N = eigs[-1]
    return lambda_N / jnp.maximum(lambda_2, 1e-10)


def sync_threshold(
    K: jax.Array,
    omegas: jax.Array,
) -> jax.Array:
    """Critical coupling strength estimate (Dorfler & Bullo 2014).

    K_c ≈ max|ω_i - ω_j| / lambda_2

    Below K_c, the network cannot synchronise. Above, it can.

    Args:
        K: (N, N) symmetric coupling matrix
        omegas: (N,) natural frequencies

    Returns:
        Scalar estimated critical coupling
    """
    lambda_2 = algebraic_connectivity(K)
    omega_spread = jnp.max(omegas) - jnp.min(omegas)
    return omega_spread / jnp.maximum(lambda_2, 1e-10)
