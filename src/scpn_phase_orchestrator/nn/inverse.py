# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inverse Kuramoto coupling inference

"""Infer coupling matrix K and natural frequencies ω from observed phases.

Three methods, in order of preference:

1. **analytical_inverse** (Pikovsky 2008) — O(N³) linear regression on
   sin(Δθ) basis functions. Exact for noiseless Kuramoto, >0.95
   correlation, completes in seconds. Use this by default.

2. **hybrid_inverse** — analytical init + gradient refinement. Handles
   model mismatch (noise, higher harmonics) by starting from the
   analytical solution and running a few Adam epochs.

3. **infer_coupling** — pure gradient descent through ODE solver.
   Kept for backward compatibility. Slow (minutes), lower accuracy.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .functional import kuramoto_forward


def inverse_loss(
    K: jax.Array,
    omegas: jax.Array,
    observed: jax.Array,
    dt: float,
    l1_weight: float = 0.0,
) -> jax.Array:
    """Loss for inverse Kuramoto: prediction error + optional L1 sparsity.

    Runs the forward model from observed[0] and compares the predicted
    trajectory against the observed trajectory.

    Parameters
    ----------
    K : jax.Array
        (N, N) coupling matrix to optimize.
    omegas : jax.Array
        (N,) natural frequencies to optimize.
    observed : jax.Array
        (T, N) observed phase trajectory.
    dt : float
        integration timestep.
    l1_weight : float
        L1 penalty on K for sparsity (0 = no penalty).

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    n_steps = observed.shape[0] - 1
    initial = observed[0]

    _, predicted = kuramoto_forward(initial, omegas, K, dt, n_steps)

    diff = observed[1:] - predicted
    phase_error = jnp.mean(1.0 - jnp.cos(diff))

    loss = phase_error
    if l1_weight > 0.0:
        loss = loss + l1_weight * jnp.sum(jnp.abs(K))
    return loss


def _shooting_loss(
    K: jax.Array,
    omegas: jax.Array,
    starts: jax.Array,
    targets: jax.Array,
    dt: float,
    window_size: int,
    l1_weight: float = 0.0,
) -> jax.Array:
    """Multiple-shooting loss, fully JIT-compatible via vmap.

    Parameters
    ----------
    K : jax.Array
        (N, N) coupling matrix.
    omegas : jax.Array
        (N,) natural frequencies.
    starts : jax.Array
        (W, N) initial phases for each window.
    targets : jax.Array
        (W, window_size, N) target trajectories per window.
    dt : float
        integration timestep.
    window_size : int
        steps per window (fixed for JIT).
    l1_weight : float
        L1 penalty.

    Returns
    -------
    jax.Array
        The result.
    """

    def window_error(start: jax.Array, target: jax.Array) -> jax.Array:
        """Phase prediction error for a single shooting window."""
        _, predicted = kuramoto_forward(start, omegas, K, dt, window_size)
        return jnp.mean(1.0 - jnp.cos(target - predicted))

    errors = jax.vmap(window_error)(starts, targets)
    loss = jnp.mean(errors)
    if l1_weight > 0.0:
        loss = loss + l1_weight * jnp.sum(jnp.abs(K))
    return loss


def _build_windows(
    observed: jax.Array, window_size: int
) -> tuple[jax.Array, jax.Array]:
    """Split trajectory into fixed-size windows for shooting loss.

    Returns (starts, targets) where starts[i] is the initial phase
    and targets[i] is the next window_size steps.
    """
    T = observed.shape[0]
    n_windows = (T - 1) // window_size
    starts = observed[jnp.arange(n_windows) * window_size]
    indices = (
        jnp.arange(n_windows)[:, None] * window_size
        + jnp.arange(1, window_size + 1)[None, :]
    )
    targets = observed[indices]
    return starts, targets


def _symmetrise_K(K: jax.Array) -> jax.Array:
    """Enforce symmetric coupling with zero diagonal."""
    N = K.shape[0]
    K = (K + K.T) / 2.0
    return K.at[jnp.diag_indices(N)].set(0.0)


def analytical_inverse(
    observed: jax.Array,
    dt: float,
    alpha: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Recover K and ω from observed phases via linear regression.

    Exploits the Kuramoto structure directly (Pikovsky 2008):
      dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)

    Finite-difference dθ/dt, build sin(Δθ) basis, solve via lstsq.
    O(N³) per oscillator, no ODE backprop, no gradient vanishing.

    Parameters
    ----------
    observed : jax.Array
        (T, N) phase trajectory, T >= 3.
    dt : float
        integration timestep.
    alpha : float
        Tikhonov (ridge) regularisation strength. 0 = no reg.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (K, omegas): inferred (N, N) coupling and (N,) frequencies.
    """
    T, N = observed.shape
    # Phase-aware central finite differences: unwrap Δθ via atan2
    # to handle 2π boundary crossings correctly
    raw_diff = observed[2:] - observed[:-2]
    dtheta_dt = jnp.arctan2(jnp.sin(raw_diff), jnp.cos(raw_diff)) / (2.0 * dt)
    phases_mid = observed[1:-1]  # (T_mid, N)

    # Build 3D basis: B_all[i, t, j] = sin(θ_j(t) - θ_i(t))
    # phases_mid[:, :, None] - phases_mid[:, None, :] → (T_mid, N, N)
    # then transpose to (N, T_mid, N) for per-oscillator solve
    diff_3d = phases_mid[:, jnp.newaxis, :] - phases_mid[:, :, jnp.newaxis]
    sin_basis = jnp.sin(diff_3d).transpose(1, 0, 2)  # (N, T_mid, N)
    targets = dtheta_dt.T  # (N, T_mid)

    # Augment each per-oscillator design with an intercept column so ω_i is
    # estimated jointly with its coupling row rather than as a post-hoc residual.
    # Without the intercept the lstsq absorbs the ω-driven phase drift into the
    # sin(Δθ) basis, inflating K for weakly coupled or uncoupled data (the
    # ω/coupling confounding). The intercept removes that bias: uncoupled data
    # recovers K ≈ 0 while coupled recovery is unchanged.
    n_mid = sin_basis.shape[1]
    design = jnp.concatenate([sin_basis, jnp.ones((N, n_mid, 1))], axis=2)

    if alpha > 0:
        # Ridge-penalise the coupling block only; the intercept (ω) is unpenalised.
        reg = alpha * jnp.diag(jnp.concatenate([jnp.ones(N), jnp.zeros(1)]))

        def _solve(design_i: jax.Array, target: jax.Array) -> jax.Array:
            solution_i: jax.Array = jnp.linalg.solve(
                design_i.T @ design_i + reg, design_i.T @ target
            )
            return solution_i

        solution = jax.vmap(_solve)(design, targets)
    else:

        def _solve(design_i: jax.Array, target: jax.Array) -> jax.Array:
            row, _, _, _ = jnp.linalg.lstsq(design_i, target)
            return row

        solution = jax.vmap(_solve)(design, targets)

    K = _symmetrise_K(solution[:, :N])
    omegas = solution[:, N]
    return K, omegas


def hybrid_inverse(
    observed: jax.Array,
    dt: float,
    alpha: float = 0.0,
    n_refine: int = 50,
    lr: float = 0.005,
    window_size: int = 10,
) -> tuple[jax.Array, jax.Array, list[float]]:
    """Analytical inverse + gradient refinement for noisy data.

    Runs analytical_inverse() for the initial estimate, then refines
    with a few Adam epochs using multiple shooting. Handles model
    mismatch (noise, higher harmonics, amplitude effects).

    Parameters
    ----------
    observed : jax.Array
        (T, N) phase trajectory.
    dt : float
        integration timestep.
    alpha : float
        Tikhonov regularisation for analytical step.
    n_refine : int
        Adam refinement epochs (0 = analytical only).
    lr : float
        learning rate for refinement.
    window_size : int
        shooting window size for refinement.

    Returns
    -------
    tuple[jax.Array, jax.Array, list[float]]
        (K, omegas, losses): inferred params + refinement loss history.
    """
    K, omegas = analytical_inverse(observed, dt, alpha=alpha)

    if n_refine <= 0:
        return K, omegas, []

    starts, targets = _build_windows(observed, window_size)

    def loss_fn(k: jax.Array, o: jax.Array) -> jax.Array:
        """Shooting loss for refinement step."""
        return _shooting_loss(k, o, starts, targets, dt, window_size, 0.0)

    loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))

    m_K = jnp.zeros_like(K)
    v_K = jnp.zeros_like(K)
    m_o = jnp.zeros_like(omegas)
    v_o = jnp.zeros_like(omegas)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses: list[float] = []

    for epoch in range(n_refine):
        loss_val, (grad_K, grad_o) = loss_and_grad(K, omegas)
        g_norm = jnp.sqrt(jnp.sum(grad_K**2) + jnp.sum(grad_o**2) + 1e-10)
        scale = jnp.minimum(1.0, 1.0 / g_norm)
        grad_K = grad_K * scale
        grad_o = grad_o * scale

        t = epoch + 1
        m_K = beta1 * m_K + (1 - beta1) * grad_K
        v_K = beta2 * v_K + (1 - beta2) * grad_K**2
        m_o = beta1 * m_o + (1 - beta1) * grad_o
        v_o = beta2 * v_o + (1 - beta2) * grad_o**2
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        K = K - lr * (m_K / bc1) / (jnp.sqrt(v_K / bc2) + eps)
        omegas = omegas - lr * (m_o / bc1) / (jnp.sqrt(v_o / bc2) + eps)
        K = _symmetrise_K(K)
        losses.append(float(loss_val))

    return K, omegas, losses


def infer_coupling(
    observed: jax.Array,
    dt: float,
    n_epochs: int = 200,
    lr: float = 0.01,
    l1_weight: float = 0.001,
    seed: int = 0,
    window_size: int = 0,
    grad_clip: float = 1.0,
) -> tuple[jax.Array, jax.Array, list[float]]:
    """Infer coupling matrix K and frequencies ω from observed phases.

    Uses Adam optimiser with gradient clipping and optional multiple
    shooting for gradient-stable training through ODE solvers.

    Parameters
    ----------
    observed : jax.Array
        (T, N) observed phase trajectory.
    dt : float
        integration timestep used to generate the data.
    n_epochs : int
        optimisation epochs.
    lr : float
        learning rate (for Adam).
    l1_weight : float
        L1 sparsity penalty on K.
    seed : int
        random seed for initialisation.
    window_size : int
        if >0, use multiple shooting with this window size. Recommended: 10-20 steps. 0
        = single-shot (original behaviour).
    grad_clip : float
        maximum gradient norm (0 = no clipping).

    Returns
    -------
    tuple[jax.Array, jax.Array, list[float]]
        (K, omegas, losses) where: K: (N, N) inferred coupling matrix omegas: (N,)
        inferred natural frequencies losses: list of loss values per epoch.
    """
    N = observed.shape[1]
    key = jax.random.PRNGKey(seed)
    k1, _ = jax.random.split(key)

    K = jax.random.normal(k1, (N, N)) * 0.05
    K = _symmetrise_K(K)
    omegas = jnp.zeros(N)

    # Adam state
    m_K = jnp.zeros_like(K)
    v_K = jnp.zeros_like(K)
    m_o = jnp.zeros_like(omegas)
    v_o = jnp.zeros_like(omegas)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    if window_size > 0:
        starts, targets = _build_windows(observed, window_size)

        def loss_fn(k: jax.Array, o: jax.Array) -> jax.Array:
            """Multiple-shooting loss with L1 penalty."""
            return _shooting_loss(k, o, starts, targets, dt, window_size, l1_weight)
    else:

        def loss_fn(k: jax.Array, o: jax.Array) -> jax.Array:
            """Single-shot inverse loss with L1 penalty."""
            return inverse_loss(k, o, observed, dt, l1_weight)

    loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))
    losses: list[float] = []

    for epoch in range(n_epochs):
        loss_val, (grad_K, grad_o) = loss_and_grad(K, omegas)

        # Gradient clipping
        if grad_clip > 0:
            g_norm = jnp.sqrt(jnp.sum(grad_K**2) + jnp.sum(grad_o**2) + 1e-10)
            scale = jnp.minimum(1.0, grad_clip / g_norm)
            grad_K = grad_K * scale
            grad_o = grad_o * scale

        # Adam update
        t = epoch + 1
        m_K = beta1 * m_K + (1 - beta1) * grad_K
        v_K = beta2 * v_K + (1 - beta2) * grad_K**2
        m_o = beta1 * m_o + (1 - beta1) * grad_o
        v_o = beta2 * v_o + (1 - beta2) * grad_o**2

        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        K = K - lr * (m_K / bc1) / (jnp.sqrt(v_K / bc2) + eps)
        omegas = omegas - lr * (m_o / bc1) / (jnp.sqrt(v_o / bc2) + eps)

        K = _symmetrise_K(K)
        losses.append(float(loss_val))

    return K, omegas, losses


def coupling_correlation(K_true: jax.Array, K_inferred: jax.Array) -> jax.Array:
    """Pearson correlation between true and inferred coupling matrices.

    Parameters
    ----------
    K_true : jax.Array
        (N, N) ground truth coupling.
    K_inferred : jax.Array
        (N, N) inferred coupling.

    Returns
    -------
    jax.Array
        Scalar correlation in [-1, 1].
    """
    # Flatten upper triangle (exclude diagonal)
    N = K_true.shape[0]
    idx = jnp.triu_indices(N, k=1)
    a = K_true[idx]
    b = K_inferred[idx]
    a_centered = a - jnp.mean(a)
    b_centered = b - jnp.mean(b)
    num = jnp.sum(a_centered * b_centered)
    denom = jnp.sqrt(jnp.sum(a_centered**2) * jnp.sum(b_centered**2) + 1e-10)
    result: jax.Array = num / denom
    return result
