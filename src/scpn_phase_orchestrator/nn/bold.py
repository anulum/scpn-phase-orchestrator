# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
TAU = 0.98  # haemodynamic transit time (s)
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

    Parameters
    ----------
    s : jax.Array
        (N,) vasodilatory signal.
    f : jax.Array
        (N,) blood inflow (normalised, resting=1).
    v : jax.Array
        (N,) blood volume (normalised, resting=1).
    q : jax.Array
        (N,) deoxyhaemoglobin content (normalised, resting=1).
    x : jax.Array
        (N,) neural input (amplitude envelope).
    dt : float
        integration timestep (seconds).
    kappa : float
        signal decay rate (default 0.65).
    gamma : float
        flow-dependent elimination (default 0.41).
    tau : float
        haemodynamic transit time (default 0.98).
    alpha : float
        Grubb's vessel stiffness exponent (default 0.32).
    e0 : float
        resting oxygen extraction fraction (default 0.34).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        (new_s, new_f, new_v, new_q).
    """
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

    Parameters
    ----------
    v : jax.Array
        (N,) or (T, N) blood volume.
    q : jax.Array
        (N,) or (T, N) deoxyhaemoglobin.

    Returns
    -------
    jax.Array
        BOLD signal, same shape as input.
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

    Parameters
    ----------
    neural : jax.Array
        (T, N) neural activity time series (e.g., amplitude envelope).
    dt : float
        simulation timestep (seconds).
    dt_bold : float
        BOLD sampling period (seconds, default 0.5s = 2Hz).
    kappa : float
        signal decay rate (default 0.65).
    gamma : float
        flow-dependent elimination (default 0.41).
    tau : float
        haemodynamic transit time (default 0.98).
    alpha : float
        Grubb's vessel stiffness exponent (default 0.32).
    e0 : float
        resting oxygen extraction fraction (default 0.34).

    Returns
    -------
    jax.Array
        (T_bold, N) BOLD signal, where T_bold = T * dt / dt_bold.
    """
    T, n_regions = neural.shape
    subsample = max(1, int(dt_bold / dt))

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        x_t: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        """One scan step: advance hemodynamic state and emit BOLD."""
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
