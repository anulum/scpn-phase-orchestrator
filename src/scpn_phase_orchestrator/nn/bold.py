# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable BOLD signal generator

"""Balloon-Windkessel hemodynamic model in JAX.

Converts neural activity (oscillator amplitude envelope) to simulated
fMRI BOLD signal. Fully differentiable for gradient-based optimization
of oscillator parameters to match empirical fMRI data.

Friston et al. 2000 (Balloon model), Stephan et al. 2007 (parameters).
Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Stephan et al. 2007 default parameters
KAPPA = 0.65  # signal decay rate (1/s)
GAMMA = 0.41  # flow-dependent elimination rate (1/s)
TAU = 0.98  # hemodynamic transit time (s)
ALPHA = 0.32  # Grubb's exponent (stiffness)
E0 = 0.4  # resting oxygen extraction fraction
V0 = 0.02  # resting blood volume fraction

# BOLD signal coefficients (1.5T, Obata et al. 2004)
K1 = 7.0 * E0
K2 = 2.0
K3 = 2.0 * E0 - 0.2


def balloon_windkessel_step(
    s: jax.Array,
    f: jax.Array,
    v: jax.Array,
    q: jax.Array,
    x: jax.Array,
    dt: float,
    kappa: float = KAPPA,
    gamma: float = GAMMA,
    tau: float = TAU,
    alpha: float = ALPHA,
    e0: float = E0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single Euler step of the Balloon-Windkessel hemodynamic model.

    Args:
        s: (N,) vasodilatory signal
        f: (N,) blood inflow (normalized, resting=1)
        v: (N,) blood volume (normalized, resting=1)
        q: (N,) deoxyhemoglobin content (normalized, resting=1)
        x: (N,) neural input (amplitude envelope)
        dt: integration timestep (seconds)
        kappa, gamma, tau, alpha, e0: hemodynamic parameters

    Returns:
        Tuple of (new_s, new_f, new_v, new_q)
    """
    # Oxygen extraction: E(f) = 1 - (1 - E0)^(1/f)
    E_f = 1.0 - (1.0 - e0) ** (1.0 / jnp.maximum(f, 0.01))

    ds = x - kappa * s - gamma * (f - 1.0)
    df = s
    dv = (1.0 / tau) * (f - v ** (1.0 / alpha))
    dq = (1.0 / tau) * (f * E_f / e0 - v ** (1.0 / alpha) * q / jnp.maximum(v, 0.01))

    new_s = s + dt * ds
    new_f = jnp.maximum(f + dt * df, 0.01)
    new_v = jnp.maximum(v + dt * dv, 0.01)
    new_q = jnp.maximum(q + dt * dq, 0.01)

    return new_s, new_f, new_v, new_q


def bold_signal(
    v: jax.Array,
    q: jax.Array,
    v0: float = V0,
    k1: float = K1,
    k2: float = K2,
    k3: float = K3,
) -> jax.Array:
    """Compute BOLD signal from blood volume and deoxyhemoglobin.

    Args:
        v: (N,) or (T, N) blood volume
        q: (N,) or (T, N) deoxyhemoglobin

    Returns:
        BOLD signal, same shape as input
    """
    return v0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))


def bold_from_neural(
    neural: jax.Array,
    dt: float,
    dt_bold: float = 0.5,
    kappa: float = KAPPA,
    gamma: float = GAMMA,
    tau: float = TAU,
    alpha: float = ALPHA,
    e0: float = E0,
) -> jax.Array:
    """Generate BOLD signal from neural activity time series.

    Runs the Balloon-Windkessel model on the neural input and returns
    the BOLD signal at a lower sampling rate (TR = dt_bold).

    Args:
        neural: (T, N) neural activity time series (e.g., amplitude envelope)
        dt: simulation timestep (seconds)
        dt_bold: BOLD sampling period (seconds, default 0.5s = 2Hz)
        kappa, gamma, tau, alpha, e0: hemodynamic parameters

    Returns:
        (T_bold, N) BOLD signal, where T_bold = T * dt / dt_bold
    """
    T, n_regions = neural.shape
    subsample = max(1, int(dt_bold / dt))

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        s, f, v, q = carry
        new_s, new_f, new_v, new_q = balloon_windkessel_step(
            s, f, v, q, x_t, dt, kappa, gamma, tau, alpha, e0
        )
        y = bold_signal(new_v, new_q)
        return (new_s, new_f, new_v, new_q), y

    s0 = jnp.zeros(n_regions)
    f0 = jnp.ones(n_regions)
    v0 = jnp.ones(n_regions)
    q0 = jnp.ones(n_regions)

    _, bold_full = jax.lax.scan(step, (s0, f0, v0, q0), neural)
    result: jax.Array = bold_full[::subsample]
    return result
