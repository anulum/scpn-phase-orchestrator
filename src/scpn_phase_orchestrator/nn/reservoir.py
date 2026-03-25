# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Kuramoto reservoir computing

"""Kuramoto-based reservoir computing in JAX.

Uses a Kuramoto oscillator network as a nonlinear reservoir. Input signals
modulate natural frequencies; the reservoir's phase state is read out via
a trained linear layer.

Theory: universal approximation near edge-of-bifurcation
(arXiv:2407.16172, 2024). The Ott-Antonsen critical coupling K_c = 2*Delta
defines the optimal operating point.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .functional import kuramoto_forward, order_parameter


def reservoir_features(phases: jax.Array) -> jax.Array:
    """Extract features from oscillator phases for readout.

    Features: [cos(theta_1), sin(theta_1), ..., cos(theta_N), sin(theta_N), R]
    Total: 2*N + 1 features.

    Args:
        phases: (N,) oscillator phases

    Returns:
        (2*N + 1,) feature vector
    """
    R = order_parameter(phases)
    return jnp.concatenate([jnp.cos(phases), jnp.sin(phases), R[jnp.newaxis]])


def reservoir_drive(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    W_in: jax.Array,
    u: jax.Array,
    dt: float,
    n_steps: int,
) -> jax.Array:
    """Drive reservoir with input signal and collect features at each step.

    Input is injected into natural frequencies: omega_i(t) = omega_i + W_in @ u(t).

    Args:
        phases: (N,) initial oscillator phases
        omegas: (N,) base natural frequencies
        K: (N, N) fixed coupling matrix
        W_in: (N, D_in) input weight matrix
        u: (T, D_in) input signal sequence
        dt: integration timestep
        n_steps: Kuramoto steps per input sample

    Returns:
        (T, 2*N + 1) feature matrix for readout training
    """

    def process_sample(carry, u_t):
        p = carry
        driven_omegas = omegas + W_in @ u_t
        for _ in range(n_steps):
            p, _ = kuramoto_forward(p, driven_omegas, K, dt, 1)
        features = reservoir_features(p)
        return p, features

    _, features = jax.lax.scan(process_sample, phases, u)
    return features


def ridge_readout(
    features: jax.Array,
    targets: jax.Array,
    alpha: float = 1e-4,
) -> jax.Array:
    """Train linear readout via ridge regression.

    W_out = (F^T F + alpha I)^{-1} F^T Y

    Args:
        features: (T, D_feat) reservoir feature matrix
        targets: (T, D_out) target outputs
        alpha: L2 regularization strength

    Returns:
        (D_feat, D_out) readout weight matrix
    """
    FtF = features.T @ features + alpha * jnp.eye(features.shape[1])
    FtY = features.T @ targets
    return jnp.linalg.solve(FtF, FtY)


def reservoir_predict(
    features: jax.Array,
    W_out: jax.Array,
) -> jax.Array:
    """Apply trained readout to reservoir features.

    Args:
        features: (T, D_feat) feature matrix
        W_out: (D_feat, D_out) readout weights

    Returns:
        (T, D_out) predictions
    """
    return features @ W_out
