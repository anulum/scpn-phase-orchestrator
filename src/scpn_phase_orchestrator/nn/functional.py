# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable Kuramoto functional API

"""Pure JAX functions for differentiable Kuramoto dynamics.

All functions are JIT-compilable, vmap-compatible, and differentiable
via JAX autodiff. No NumPy conversions — inputs and outputs stay as
JAX arrays for gradient flow.

Requires: jax>=0.4
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

TWO_PI = 2.0 * jnp.pi


def _as_real_jax_array(name: str, value: jax.Array) -> jax.Array:
    """Return a numeric real-valued JAX array or reject unsafe public inputs."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if jnp.issubdtype(array.dtype, jnp.bool_):
        raise ValueError(f"{name} must not contain boolean values")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must contain real-valued samples")
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise ValueError(f"{name} must contain numeric samples")
    return array.astype(jnp.result_type(array, 1.0))


def _validate_square_coupling(name: str, value: jax.Array) -> jax.Array:
    """Validate real square coupling matrices before Laplacian construction."""
    matrix = _as_real_jax_array(name, value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square coupling matrix")
    return matrix


def _validate_frequency_vector(value: jax.Array, *, n: int) -> jax.Array:
    """Validate real oscillator-frequency vectors against coupling dimension."""
    vector = _as_real_jax_array("omegas", value)
    if vector.ndim != 1:
        raise ValueError("omegas must be a one-dimensional frequency vector")
    if vector.shape[0] != n:
        raise ValueError("omegas length must match K dimensions")
    return vector


def _require_positive_real(value: object, name: str) -> float:
    """Require a finite positive scalar control parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return result


def _require_non_negative_real(value: object, name: str) -> float:
    """Require a finite non-negative scalar control parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real")
    return result


def _require_positive_int(value: object, name: str) -> int:
    """Require a positive integer control parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _require_positive_int_or_none(value: object, name: str) -> int | None:
    """Require a positive integer control parameter or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return result


def kuramoto_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single Euler step of the Kuramoto model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) coupling matrix.
    dt : float
        integration timestep.

    Returns
    -------
    jax.Array
        (N,) updated phases, wrapped to [0, 2pi).
    """
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    coupling = jnp.sum(K * jnp.sin(diff), axis=1)
    return (phases + dt * (omegas + coupling)) % TWO_PI


def kuramoto_rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single RK4 step of the Kuramoto model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) coupling matrix.
    dt : float
        integration timestep.

    Returns
    -------
    jax.Array
        (N,) updated phases, wrapped to [0, 2pi).
    """

    def deriv(p: jax.Array) -> jax.Array:
        diff = p[jnp.newaxis, :] - p[:, jnp.newaxis]
        return omegas + jnp.sum(K * jnp.sin(diff), axis=1)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    new = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new % TWO_PI


def kuramoto_forward(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N Kuramoto steps, returning final state and trajectory.

    Uses jax.lax.scan for efficient compilation and autodiff.

    Parameters
    ----------
    phases : jax.Array
        (N,) initial oscillator phases.
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) coupling matrix.
    dt : float
        integration timestep.
    n_steps : int
        number of integration steps.
    method : str
        "rk4" or "euler".

    Returns
    -------
    tuple[jax.Array, jax.Array]
        : final: (N,) phases after n_steps trajectory: (n_steps, N) full phase
        trajectory.
    """
    step_fn = kuramoto_rk4_step if method == "rk4" else kuramoto_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


# ──────────────────────────────────────────────────
# Masked (sparse) coupling variants
# ──────────────────────────────────────────────────


def _kuramoto_deriv_masked(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Kuramoto derivative with binary mask for sparse coupling."""
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    coupling = jnp.sum(K * mask * jnp.sin(diff), axis=1)
    return omegas + coupling


def kuramoto_step_masked(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    mask: jax.Array,
    dt: float,
) -> jax.Array:
    """Single Euler step with masked coupling.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) coupling weights.
    mask : jax.Array
        (N, N) binary mask (1 = edge exists, 0 = no edge).
    dt : float
        integration timestep.

    Returns
    -------
    jax.Array
        (N,) updated phases.
    """
    dphi = _kuramoto_deriv_masked(phases, omegas, K, mask)
    return (phases + dt * dphi) % TWO_PI


def kuramoto_rk4_step_masked(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    mask: jax.Array,
    dt: float,
) -> jax.Array:
    """Single RK4 step with masked coupling.

    Parameters
    ----------
    phases : jax.Array
        Oscillator phases in radians, shape ``(N,)``.
    omegas : jax.Array
        Natural frequencies in rad/s, shape ``(N,)``.
    K : jax.Array
        Coupling matrix ``K``, shape ``(N, N)``.
    mask : jax.Array
        Boolean coupling mask, shape ``(N, N)``.
    dt : float
        Integration step size.

    Returns
    -------
    jax.Array
        The phases after one masked RK4 step.
    """

    def deriv(p: jax.Array) -> jax.Array:
        return _kuramoto_deriv_masked(p, omegas, K, mask)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    return (phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)) % TWO_PI


def kuramoto_forward_masked(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    mask: jax.Array,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N Kuramoto steps with masked coupling.

    Parameters
    ----------
    phases : jax.Array
        (N,) initial phases.
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) coupling weights.
    mask : jax.Array
        (N, N) binary mask.
    dt : float
        timestep.
    n_steps : int
        integration steps.
    method : str
        "rk4" or "euler".

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (final, trajectory) — same as kuramoto_forward.
    """
    step_fn = kuramoto_rk4_step_masked if method == "rk4" else kuramoto_step_masked

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, mask, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


# ──────────────────────────────────────────────────
# Winfree model (Winfree 1967)
# ──────────────────────────────────────────────────


def _winfree_deriv(
    phases: jax.Array,
    omegas: jax.Array,
    K: float,
) -> jax.Array:
    """Compute the derivative for the Winfree model.

    dθ_i/dt = ω_i + (K/N) · Q(θ_i) · Σ_j P(θ_j)

    P(θ) = (1 + cos(θ)) (pulse), Q(θ) = -sin(θ) (phase response curve).
    """
    N = phases.shape[0]
    P_sum = jnp.sum(1.0 + jnp.cos(phases))  # scalar
    Q = -jnp.sin(phases)  # (N,)
    return omegas + (K / N) * Q * P_sum


def winfree_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: float,
    dt: float,
) -> jax.Array:
    """Single Euler step of the Winfree model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : float
        scalar coupling strength.
    dt : float
        integration timestep.

    Returns
    -------
    jax.Array
        (N,) updated phases.
    """
    return (phases + dt * _winfree_deriv(phases, omegas, K)) % TWO_PI


def winfree_rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: float,
    dt: float,
) -> jax.Array:
    """Single RK4 step of the Winfree model.

    Parameters
    ----------
    phases : jax.Array
        Oscillator phases in radians, shape ``(N,)``.
    omegas : jax.Array
        Natural frequencies in rad/s, shape ``(N,)``.
    K : float
        Coupling matrix ``K``, shape ``(N, N)``.
    dt : float
        Integration step size.

    Returns
    -------
    jax.Array
        The phases after one Winfree RK4 step.
    """

    def deriv(p: jax.Array) -> jax.Array:
        return _winfree_deriv(p, omegas, K)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    return (phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)) % TWO_PI


def winfree_forward(
    phases: jax.Array,
    omegas: jax.Array,
    K: float,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N steps of Winfree dynamics.

    Parameters
    ----------
    phases : jax.Array
        (N,) initial phases.
    omegas : jax.Array
        (N,) natural frequencies.
    K : float
        scalar coupling strength.
    dt : float
        timestep.
    n_steps : int
        integration steps.
    method : str
        "rk4" or "euler".

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (final, trajectory).
    """
    step_fn = winfree_rk4_step if method == "rk4" else winfree_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


def _simplicial_deriv(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    sigma2: float | jax.Array,
) -> jax.Array:
    """Compute the derivative for Kuramoto with pairwise + 3-body coupling.

    Gambuzza et al. 2021, Nature Communications 12:1255; Tang et al. 2025.
    """
    n = phases.shape[0]
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    pairwise = jnp.sum(K * jnp.sin(diff), axis=1)
    result = omegas + pairwise

    # 3-body: σ₂/N² Σ_{j,k} sin((θ_j - θ_i) + (θ_k - θ_i))
    # = σ₂/N² · 2 · S_i · C_i  where S_i = Σ_j sin(θ_j - θ_i), C_i = Σ_j cos(θ_j - θ_i)
    S = jnp.sum(jnp.sin(diff), axis=1)  # (N,)
    C = jnp.sum(jnp.cos(diff), axis=1)  # (N,)
    three_body = sigma2 / (n * n) * 2.0 * S * C
    return result + three_body


def simplicial_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    sigma2: float | jax.Array = 0.0,
) -> jax.Array:
    """Single Euler step of the simplicial (3-body) Kuramoto model.

    Extends standard Kuramoto with higher-order 3-body interactions that
    produce explosive (first-order) synchronization transitions.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) pairwise coupling matrix.
    dt : float
        integration timestep.
    sigma2 : float | jax.Array
        3-body coupling strength (0 = standard Kuramoto).

    Returns
    -------
    jax.Array
        (N,) updated phases, wrapped to [0, 2pi).
    """
    dphi = _simplicial_deriv(phases, omegas, K, sigma2)
    return (phases + dt * dphi) % TWO_PI


def simplicial_rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    sigma2: float | jax.Array = 0.0,
) -> jax.Array:
    """Single RK4 step of the simplicial (3-body) Kuramoto model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) pairwise coupling matrix.
    dt : float
        integration timestep.
    sigma2 : float | jax.Array
        3-body coupling strength (0 = standard Kuramoto).

    Returns
    -------
    jax.Array
        (N,) updated phases, wrapped to [0, 2pi).
    """

    def deriv(p: jax.Array) -> jax.Array:
        return _simplicial_deriv(p, omegas, K, sigma2)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    new = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new % TWO_PI


def simplicial_forward(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    n_steps: int,
    sigma2: float | jax.Array = 0.0,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N steps of simplicial Kuramoto, returning final state and trajectory.

    Parameters
    ----------
    phases : jax.Array
        (N,) initial oscillator phases.
    omegas : jax.Array
        (N,) natural frequencies.
    K : jax.Array
        (N, N) pairwise coupling matrix.
    dt : float
        integration timestep.
    n_steps : int
        number of integration steps.
    sigma2 : float | jax.Array
        3-body coupling strength (0 = standard Kuramoto).
    method : str
        "rk4" or "euler".

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (final, trajectory) where trajectory is (n_steps, N).
    """
    step_fn = simplicial_rk4_step if method == "rk4" else simplicial_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, dt, sigma2)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


def _stuart_landau_deriv(
    phases: jax.Array,
    amplitudes: jax.Array,
    omegas: jax.Array,
    mu: jax.Array,
    K: jax.Array,
    K_r: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute the derivative for Stuart-Landau coupled oscillators.

    Phase: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
    Amplitude: dr_i/dt = (μ_i - r_i²)r_i + ε Σ_j K^r_ij · r_j · cos(θ_j - θ_i)
    """
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    dtheta = omegas + jnp.sum(K * jnp.sin(diff), axis=1)

    r_clamped = jnp.maximum(amplitudes, 0.0)
    amp_coupling = jnp.sum(K_r * r_clamped[jnp.newaxis, :] * jnp.cos(diff), axis=1)
    dr = (mu - amplitudes * amplitudes) * amplitudes + epsilon * amp_coupling

    return dtheta, dr


def stuart_landau_step(
    phases: jax.Array,
    amplitudes: jax.Array,
    omegas: jax.Array,
    mu: jax.Array,
    K: jax.Array,
    K_r: jax.Array,
    dt: float,
    epsilon: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Single Euler step of the Stuart-Landau oscillator model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    amplitudes : jax.Array
        (N,) oscillator amplitudes (r >= 0).
    omegas : jax.Array
        (N,) natural frequencies.
    mu : jax.Array
        (N,) bifurcation parameters (supercritical if mu > 0).
    K : jax.Array
        (N, N) phase coupling matrix.
    K_r : jax.Array
        (N, N) amplitude coupling matrix.
    dt : float
        integration timestep.
    epsilon : float
        amplitude coupling strength.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (new_phases, new_amplitudes).
    """
    dtheta, dr = _stuart_landau_deriv(phases, amplitudes, omegas, mu, K, K_r, epsilon)
    new_phases = (phases + dt * dtheta) % TWO_PI
    new_amplitudes = jnp.maximum(amplitudes + dt * dr, 0.0)
    return new_phases, new_amplitudes


def stuart_landau_rk4_step(
    phases: jax.Array,
    amplitudes: jax.Array,
    omegas: jax.Array,
    mu: jax.Array,
    K: jax.Array,
    K_r: jax.Array,
    dt: float,
    epsilon: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Single RK4 step of the Stuart-Landau oscillator model.

    Parameters
    ----------
    phases : jax.Array
        (N,) oscillator phases in [0, 2pi).
    amplitudes : jax.Array
        (N,) oscillator amplitudes (r >= 0).
    omegas : jax.Array
        (N,) natural frequencies.
    mu : jax.Array
        (N,) bifurcation parameters.
    K : jax.Array
        (N, N) phase coupling matrix.
    K_r : jax.Array
        (N, N) amplitude coupling matrix.
    dt : float
        integration timestep.
    epsilon : float
        amplitude coupling strength.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (new_phases, new_amplitudes).
    """

    def deriv(p: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _stuart_landau_deriv(p, r, omegas, mu, K, K_r, epsilon)

    k1p, k1r = deriv(phases, amplitudes)
    k2p, k2r = deriv(phases + 0.5 * dt * k1p, amplitudes + 0.5 * dt * k1r)
    k3p, k3r = deriv(phases + 0.5 * dt * k2p, amplitudes + 0.5 * dt * k2r)
    k4p, k4r = deriv(phases + dt * k3p, amplitudes + dt * k3r)

    new_phases = (phases + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)) % TWO_PI
    new_amps = amplitudes + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
    return new_phases, jnp.maximum(new_amps, 0.0)


def stuart_landau_forward(
    phases: jax.Array,
    amplitudes: jax.Array,
    omegas: jax.Array,
    mu: jax.Array,
    K: jax.Array,
    K_r: jax.Array,
    dt: float,
    n_steps: int,
    epsilon: float = 1.0,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run N Stuart-Landau steps, returning final state and trajectories.

    Parameters
    ----------
    phases : jax.Array
        (N,) initial phases.
    amplitudes : jax.Array
        (N,) initial amplitudes.
    omegas : jax.Array
        (N,) natural frequencies.
    mu : jax.Array
        (N,) bifurcation parameters.
    K : jax.Array
        (N, N) phase coupling matrix.
    K_r : jax.Array
        (N, N) amplitude coupling matrix.
    dt : float
        integration timestep.
    n_steps : int
        number of steps.
    epsilon : float
        amplitude coupling strength.
    method : str
        "rk4" or "euler".

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        (final_phases, final_amplitudes, phase_traj, amp_traj) where trajectories are
        (n_steps, N).
    """
    step_fn = stuart_landau_rk4_step if method == "rk4" else stuart_landau_step

    def body(
        carry: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        p, r = carry
        new_p, new_r = step_fn(p, r, omegas, mu, K, K_r, dt, epsilon)
        return (new_p, new_r), (new_p, new_r)

    (final_p, final_r), (traj_p, traj_r) = jax.lax.scan(
        body, (phases, amplitudes), None, length=n_steps
    )
    return final_p, final_r, traj_p, traj_r


def order_parameter(phases: jax.Array) -> jax.Array:
    """Kuramoto order parameter R = |<exp(i*phi)>|.

    Differentiable scalar measuring global synchronization.
    R=1 means perfect sync, R~0 means incoherent.

    Parameters
    ----------
    phases : jax.Array
        (N,) or (T, N) oscillator phases.

    Returns
    -------
    jax.Array
        Scalar R value (or (T,) if trajectory input).
    """
    z = jnp.exp(1j * phases)
    return jnp.abs(jnp.mean(z, axis=-1))


def plv(trajectory: jax.Array) -> jax.Array:
    """Phase-Locking Value matrix from a phase trajectory.

    PLV_ij = |<exp(i*(phi_i(t) - phi_j(t)))>_t|

    Parameters
    ----------
    trajectory : jax.Array
        (T, N) phase trajectory.

    Returns
    -------
    jax.Array
        (N, N) PLV matrix, values in [0, 1].
    """
    # (T, N, 1) - (T, 1, N) -> (T, N, N) phase differences
    diff = trajectory[:, :, jnp.newaxis] - trajectory[:, jnp.newaxis, :]
    return jnp.abs(jnp.mean(jnp.exp(1j * diff), axis=0))


# --- Spectral Alignment Function (SAF) ---
# Skardal & Taylor, SIAM J. Appl. Dyn. Syst. 2016;
# Song et al. 2025 (arXiv:2501.18279)  # spectral alignment for Kuramoto


def coupling_laplacian(K: jax.Array) -> jax.Array:
    """Compute the graph Laplacian from a coupling matrix.

    L = D - K, where D_ii = sum_j K_ij.

    Parameters
    ----------
    K : jax.Array
        (N, N) symmetric coupling matrix.

    Returns
    -------
    jax.Array
        (N, N) Laplacian matrix.
    """
    K = _validate_square_coupling("K", K)
    D = jnp.diag(jnp.sum(K, axis=1))
    return D - K


def saf_order_parameter(
    K: jax.Array,
    omegas: jax.Array,
    eps: float = 1e-8,
    solver: str = "auto",
    exact_size_limit: int = 256,
    cg_tol: float = 1e-5,
    cg_maxiter: int | None = None,
) -> jax.Array:
    """Spectral Alignment Function: closed-form order parameter estimate.

    r ≈ 1 - (1/2N) Σ_{j=2}^N λ_j⁻² ⟨v^j, ω⟩²

    where λ_j are Laplacian eigenvalues and v^j are eigenvectors.
    Valid in the strongly-coupled regime. The exact path differentiates through
    Laplacian eigendecomposition. The conjugate-gradient path uses the
    equivalent identity Σ λ_j⁻² ⟨v^j, ω⟩² = ||L⁺ω||², avoids full
    eigendecomposition, and maps large dense problems to GPU-friendly
    matrix-vector operations.

    Parameters
    ----------
    K : jax.Array
        (N, N) symmetric coupling matrix (non-negative).
    omegas : jax.Array
        (N,) natural frequencies.
    eps : float
        regularization for small eigenvalues.
    solver : str
        "auto", "eigh", or "cg". "auto" uses exact eigendecomposition up to
        exact_size_limit and conjugate gradient above it.
    exact_size_limit : int
        Largest N where "auto" keeps the exact eigensolver.
    cg_tol : float
        Relative tolerance for the conjugate-gradient solver.
    cg_maxiter : int | None
        Optional maximum conjugate-gradient iterations.

    Returns
    -------
    jax.Array
        Scalar estimated order parameter in [0, 1].

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    K = _validate_square_coupling("K", K)
    N = K.shape[0]
    omegas = _validate_frequency_vector(omegas, n=N)
    eps = _require_positive_real(eps, "eps")
    exact_size_limit = _require_positive_int(exact_size_limit, "exact_size_limit")
    cg_tol = _require_positive_real(cg_tol, "cg_tol")
    cg_maxiter = _require_positive_int_or_none(cg_maxiter, "cg_maxiter")

    if solver == "auto":
        solver = "eigh" if exact_size_limit >= N else "cg"
    if solver == "cg":
        return _saf_order_parameter_cg(K, omegas, eps, cg_tol, cg_maxiter)
    if solver != "eigh":
        raise ValueError("solver must be 'auto', 'eigh', or 'cg'")
    return _saf_order_parameter_eigh(K, omegas, eps)


def _saf_order_parameter_eigh(
    K: jax.Array,
    omegas: jax.Array,
    eps: float,
) -> jax.Array:
    """Exact SAF estimate via dense symmetric Laplacian eigendecomposition."""
    N = K.shape[0]
    centred_omegas = omegas - jnp.mean(omegas)
    L = coupling_laplacian(K)
    eigenvalues, eigenvectors = jnp.linalg.eigh(L)

    # Skip first eigenvalue (λ_0 = 0 for connected graph)
    lam = eigenvalues[1:]  # (N-1,)
    V = eigenvectors[:, 1:]  # (N, N-1)

    # ⟨v^j, ω⟩² for each non-zero eigenvector
    projections = (V.T @ centred_omegas) ** 2  # (N-1,)

    # r ≈ 1 - (1/2N) Σ λ_j⁻² ⟨v^j, ω⟩²
    inv_lam_sq = 1.0 / (lam**2 + eps)
    r = 1.0 - jnp.sum(inv_lam_sq * projections) / (2.0 * N)
    return jnp.clip(r, 0.0, 1.0)


def _saf_order_parameter_cg(
    K: jax.Array,
    omegas: jax.Array,
    eps: float,
    tol: float,
    maxiter: int | None,
) -> jax.Array:
    """GPU-oriented SAF estimate via Laplacian pseudoinverse solve."""
    N = K.shape[0]
    centred_omegas = omegas - jnp.mean(omegas)
    degree = jnp.sum(K, axis=1)

    def matvec(x: jax.Array) -> jax.Array:
        return degree * x - K @ x + eps * x

    solution, _ = cg(matvec, centred_omegas, tol=tol, maxiter=maxiter)
    solution = solution - jnp.mean(solution)
    r = 1.0 - jnp.vdot(solution, solution).real / (2.0 * N)
    return jnp.clip(r, 0.0, 1.0)


def saf_loss(
    K: jax.Array,
    omegas: jax.Array,
    budget: float = 0.0,
    budget_weight: float = 0.1,
    solver: str = "auto",
    exact_size_limit: int = 256,
    cg_tol: float = 1e-5,
    cg_maxiter: int | None = None,
) -> jax.Array:
    """Loss function for coupling topology optimization via SAF.

    Minimizes -r_SAF (maximize synchronization) with optional L1 budget
    constraint on total coupling strength.

    Parameters
    ----------
    K : jax.Array
        (N, N) symmetric coupling matrix.
    omegas : jax.Array
        (N,) natural frequencies.
    budget : float
        target total coupling strength (0 = no constraint).
    budget_weight : float
        penalty weight for budget violation.
    solver : str
        SAF solver passed to saf_order_parameter.
    exact_size_limit : int
        Largest N where "auto" keeps the exact eigensolver.
    cg_tol : float
        Relative tolerance for the conjugate-gradient solver.
    cg_maxiter : int | None
        Optional maximum conjugate-gradient iterations.

    Returns
    -------
    jax.Array
        Scalar loss (lower = better synchronization).
    """
    budget = _require_non_negative_real(budget, "budget")
    budget_weight = _require_non_negative_real(budget_weight, "budget_weight")

    r = saf_order_parameter(
        K,
        omegas,
        solver=solver,
        exact_size_limit=exact_size_limit,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
    )
    loss = -r
    if budget > 0.0:
        total_coupling = jnp.sum(jnp.abs(K))
        loss = loss + budget_weight * jnp.maximum(total_coupling - budget, 0.0)
    return loss
