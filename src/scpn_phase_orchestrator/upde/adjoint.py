# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adjoint sensitivity for SSGF gradient
#
# The accelerated path integrates the Kuramoto-Sakaguchi field with a diffrax
# adaptive solver and differentiates through it with a continuous adjoint, for
# gradients at O(1) memory cost. The numpy finite-difference path is the
# reference estimator and the fallback for environments without JAX.

"""Adjoint and finite-difference sensitivities for UPDE coupling gradients.

The module defines the synchronization cost ``1 - R`` and two gradient paths:
a deterministic NumPy finite-difference estimator over coupling entries and a
diffrax continuous-adjoint implementation when the optional JAX/diffrax stack is
installed. The finite-difference path mutates only local coupling copies for
each perturbation; the continuous-adjoint path integrates the same
Kuramoto-Sakaguchi field and differentiates through the solver, agreeing with
the finite-difference reference in direction and — in the fine-``dt`` limit — in
magnitude. It fails explicitly with ``ImportError`` when the dependency is
absent instead of silently claiming accelerated gradients, and never mutates the
process-global ``jax_enable_x64`` flag.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["cost_R", "gradient_knm_fd", "gradient_knm_jax"]
FloatArray: TypeAlias = NDArray[np.float64]


def cost_R(phases: FloatArray) -> float:
    """Cost ``1 − R`` (minimise to maximise synchronisation).

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.

    Returns
    -------
    float
        The cost ``1 − R`` for the supplied phases.
    """
    R, _ = compute_order_parameter(phases)
    return 1.0 - R


def gradient_knm_fd(
    engine: UPDEEngine,
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    n_steps: int = 100,
    epsilon: float = 1e-4,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> FloatArray:
    """Finite-difference gradient of cost_R w.r.t. knm entries.

    For each K_ij (i≠j), perturbs by ±ε and measures the effect on R
    after n_steps. Returns gradient matrix ∂(1-R)/∂K_ij.

    Complexity: O(N² · N² · n_steps) = O(N⁴ · n_steps) because each of
    the ~N² off-diagonal entries requires a full N-step simulation that is
    itself O(N²) per step. Use gradient_knm_jax() for anything beyond N≈16.
    The adjoint method via diffrax reduces this to O(n_steps) but requires JAX.

    Parameters
    ----------
    engine : UPDEEngine
        The UPDE engine used to integrate each trial.
    phases_init : FloatArray
        Initial oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    n_steps : int
        Number of integration steps to run.
    epsilon : float
        Finite-difference perturbation size.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.

    Returns
    -------
    FloatArray
        The finite-difference gradient of the cost with respect to ``knm``.
    """
    n = knm.shape[0]
    grad = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            knm_plus = knm.copy()
            knm_plus[i, j] += epsilon
            p_plus = engine.run(
                phases_init, omegas, knm_plus, zeta, psi, alpha, n_steps
            )
            c_plus = cost_R(p_plus)

            knm_minus = knm.copy()
            knm_minus[i, j] -= epsilon
            p_minus = engine.run(
                phases_init, omegas, knm_minus, zeta, psi, alpha, n_steps
            )
            c_minus = cost_R(p_minus)

            grad[i, j] = (c_plus - c_minus) / (2 * epsilon)

    return grad


def gradient_knm_jax(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    n_steps: int = 100,
    dt: float = 0.01,
) -> FloatArray:
    """Gradient of ``cost_R`` w.r.t. ``knm`` via a diffrax continuous adjoint.

    Integrates the Kuramoto-Sakaguchi field

        dθ_i/dt = ω_i + Σ_j K_ij · sin(θ_j − θ_i − α_ij)

    over ``[0, n_steps·dt]`` with an adaptive solver (``diffrax.Tsit5``) and a
    ``RecursiveCheckpointAdjoint``, then differentiates ``cost_R`` of the final
    phases with respect to ``knm`` using reverse-mode autodiff. This is the
    ``O(1)``-memory continuous-adjoint path the finite-difference estimator in
    :func:`gradient_knm_fd` approximates; the two agree in direction and, in the
    fine-``dt`` limit, in magnitude (the finite-difference reference
    differentiates the discrete explicit-Euler map, so a fixed ``dt`` leaves an
    ``O(dt)`` discretisation gap).

    The solver runs in JAX's active default precision — it does **not** mutate
    the process-global ``jax_enable_x64`` flag, so it does not perturb the
    float32 default the rest of the differentiable stack relies on. Callers that
    need float64 gradients must enable x64 at process start-up themselves.

    Parameters
    ----------
    phases_init : FloatArray
        Initial oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Phase-lag matrix in radians, shape ``(N, N)``. Use a zero matrix for no
        lag; this is the regime in which the gradient matches the drive-free
        (``ζ = ψ = 0``) finite-difference reference.
    n_steps : int
        Number of nominal steps; the integration horizon is ``n_steps · dt``.
    dt : float
        Nominal step size setting the integration horizon and the adaptive
        solver's initial step.

    Returns
    -------
    FloatArray
        The continuous-adjoint gradient of the cost with respect to ``knm``.

    Raises
    ------
    ImportError
        If the optional JAX/diffrax stack is not installed.
    """
    import diffrax
    import jax
    import jax.numpy as jnp

    theta0 = jnp.asarray(phases_init)
    om = jnp.asarray(omegas)
    al = jnp.asarray(alpha)
    t1 = float(n_steps) * dt

    def _field(_t: Any, theta: Any, coupling: Any) -> Any:
        """Kuramoto-Sakaguchi tangent-space derivative dθ/dt."""
        diff = theta[jnp.newaxis, :] - theta[:, jnp.newaxis] - al
        return om + jnp.sum(coupling * jnp.sin(diff), axis=1)

    def _cost(knm_j: Any) -> Any:
        """Continuous-adjoint roll-out to the sync cost ``1 − R``."""
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_field),
            diffrax.Tsit5(),
            t0=0.0,
            t1=t1,
            dt0=dt,
            y0=theta0,
            args=knm_j,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=16384,
        )
        theta_final = solution.ys[-1]
        z = jnp.mean(jnp.exp(1j * theta_final))
        return 1.0 - jnp.abs(z)

    grad = jax.grad(_cost)(jnp.asarray(knm))
    return np.asarray(grad, dtype=np.float64)
