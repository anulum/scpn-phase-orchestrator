# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

import jax
import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


def kuramoto_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single Euler step of the Kuramoto model.

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
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

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
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

    Args:
        phases: (N,) initial oscillator phases
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep
        n_steps: number of integration steps
        method: "rk4" or "euler"

    Returns:
        Tuple of:
            final: (N,) phases after n_steps
            trajectory: (n_steps, N) full phase trajectory
    """
    step_fn = kuramoto_rk4_step if method == "rk4" else kuramoto_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


def _simplicial_deriv(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    sigma2: float,
) -> jax.Array:
    """Derivative for Kuramoto with pairwise + 3-body simplicial coupling.

    Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.
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
    sigma2: float = 0.0,
) -> jax.Array:
    """Single Euler step of the simplicial (3-body) Kuramoto model.

    Extends standard Kuramoto with higher-order 3-body interactions that
    produce explosive (first-order) synchronization transitions.

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) pairwise coupling matrix
        dt: integration timestep
        sigma2: 3-body coupling strength (0 = standard Kuramoto)

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
    """
    dphi = _simplicial_deriv(phases, omegas, K, sigma2)
    return (phases + dt * dphi) % TWO_PI


def simplicial_rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    sigma2: float = 0.0,
) -> jax.Array:
    """Single RK4 step of the simplicial (3-body) Kuramoto model.

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) pairwise coupling matrix
        dt: integration timestep
        sigma2: 3-body coupling strength (0 = standard Kuramoto)

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
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
    sigma2: float = 0.0,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N steps of simplicial Kuramoto, returning final state and trajectory.

    Args:
        phases: (N,) initial oscillator phases
        omegas: (N,) natural frequencies
        K: (N, N) pairwise coupling matrix
        dt: integration timestep
        n_steps: number of integration steps
        sigma2: 3-body coupling strength (0 = standard Kuramoto)
        method: "rk4" or "euler"

    Returns:
        Tuple of (final, trajectory) where trajectory is (n_steps, N)
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
    """Derivative for Stuart-Landau coupled oscillators.

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

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        amplitudes: (N,) oscillator amplitudes (r >= 0)
        omegas: (N,) natural frequencies
        mu: (N,) bifurcation parameters (supercritical if mu > 0)
        K: (N, N) phase coupling matrix
        K_r: (N, N) amplitude coupling matrix
        dt: integration timestep
        epsilon: amplitude coupling strength

    Returns:
        Tuple of (new_phases, new_amplitudes)
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

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        amplitudes: (N,) oscillator amplitudes (r >= 0)
        omegas: (N,) natural frequencies
        mu: (N,) bifurcation parameters
        K: (N, N) phase coupling matrix
        K_r: (N, N) amplitude coupling matrix
        dt: integration timestep
        epsilon: amplitude coupling strength

    Returns:
        Tuple of (new_phases, new_amplitudes)
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

    Args:
        phases: (N,) initial phases
        amplitudes: (N,) initial amplitudes
        omegas: (N,) natural frequencies
        mu: (N,) bifurcation parameters
        K: (N, N) phase coupling matrix
        K_r: (N, N) amplitude coupling matrix
        dt: integration timestep
        n_steps: number of steps
        epsilon: amplitude coupling strength
        method: "rk4" or "euler"

    Returns:
        (final_phases, final_amplitudes, phase_traj, amp_traj)
        where trajectories are (n_steps, N)
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

    Args:
        phases: (N,) or (T, N) oscillator phases

    Returns:
        Scalar R value (or (T,) if trajectory input)
    """
    z = jnp.exp(1j * phases)
    return jnp.abs(jnp.mean(z, axis=-1))


def plv(trajectory: jax.Array) -> jax.Array:
    """Phase-Locking Value matrix from a phase trajectory.

    PLV_ij = |<exp(i*(phi_i(t) - phi_j(t)))>_t|

    Args:
        trajectory: (T, N) phase trajectory

    Returns:
        (N, N) PLV matrix, values in [0, 1]
    """
    # (T, N, 1) - (T, 1, N) -> (T, N, N) phase differences
    diff = trajectory[:, :, jnp.newaxis] - trajectory[:, jnp.newaxis, :]
    return jnp.abs(jnp.mean(jnp.exp(1j * diff), axis=0))


# --- Spectral Alignment Function (SAF) ---
# Skardal & Taylor, SIAM J. Appl. Dyn. Syst. 2016;
# Song et al. 2025 (arXiv:2509.18279)


def coupling_laplacian(K: jax.Array) -> jax.Array:
    """Compute the graph Laplacian from a coupling matrix.

    L = D - K, where D_ii = sum_j K_ij.

    Args:
        K: (N, N) symmetric coupling matrix

    Returns:
        (N, N) Laplacian matrix
    """
    D = jnp.diag(jnp.sum(K, axis=1))
    return D - K


def saf_order_parameter(
    K: jax.Array,
    omegas: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Spectral Alignment Function: closed-form order parameter estimate.

    r ≈ 1 - (1/2N) Σ_{j=2}^N λ_j⁻² ⟨v^j, ω⟩²

    where λ_j are Laplacian eigenvalues and v^j are eigenvectors.
    Valid in the strongly-coupled regime. Differentiable through
    eigendecomposition for gradient-based topology optimization.

    Args:
        K: (N, N) symmetric coupling matrix (non-negative)
        omegas: (N,) natural frequencies
        eps: regularization for small eigenvalues

    Returns:
        Scalar estimated order parameter in [0, 1]
    """
    N = K.shape[0]
    L = coupling_laplacian(K)
    eigenvalues, eigenvectors = jnp.linalg.eigh(L)

    # Skip first eigenvalue (λ_0 = 0 for connected graph)
    lam = eigenvalues[1:]  # (N-1,)
    V = eigenvectors[:, 1:]  # (N, N-1)

    # ⟨v^j, ω⟩² for each non-zero eigenvector
    projections = (V.T @ omegas) ** 2  # (N-1,)

    # r ≈ 1 - (1/2N) Σ λ_j⁻² ⟨v^j, ω⟩²
    inv_lam_sq = 1.0 / (lam**2 + eps)
    r = 1.0 - jnp.sum(inv_lam_sq * projections) / (2.0 * N)
    return jnp.clip(r, 0.0, 1.0)


def saf_loss(
    K: jax.Array,
    omegas: jax.Array,
    budget: float = 0.0,
    budget_weight: float = 0.1,
) -> jax.Array:
    """Loss function for coupling topology optimization via SAF.

    Minimizes -r_SAF (maximize synchronization) with optional L1 budget
    constraint on total coupling strength.

    Args:
        K: (N, N) symmetric coupling matrix
        omegas: (N,) natural frequencies
        budget: target total coupling strength (0 = no constraint)
        budget_weight: penalty weight for budget violation

    Returns:
        Scalar loss (lower = better synchronization)
    """
    r = saf_order_parameter(K, omegas)
    loss = -r
    if budget > 0.0:
        total_coupling = jnp.sum(jnp.abs(K))
        loss = loss + budget_weight * jnp.maximum(total_coupling - budget, 0.0)
    return loss
