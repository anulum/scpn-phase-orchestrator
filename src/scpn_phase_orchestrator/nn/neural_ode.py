# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Continuous adjoint UDE-Kuramoto solver (diffrax)

"""Continuous-time adjoint integration of the UDE-Kuramoto vector field.

The explicit ``jax.lax.scan`` Euler map in :mod:`scpn_phase_orchestrator.nn.ude`
stores every intermediate state, so reverse-mode gradients cost ``O(n_steps)``
memory. This module integrates the *same* vector field

    dθ_i/dt = ω_i + Σ_j K_ij · [sin(θ_j − θ_i) + NN_φ(θ_j − θ_i)]

with an adaptive higher-order solver (``diffrax.Tsit5`` by default) under a
configurable adjoint. ``diffrax.RecursiveCheckpointAdjoint`` gives logarithmic
checkpointing; ``diffrax.BacksolveAdjoint`` reconstructs the forward trajectory
backwards for ``O(1)`` memory. Both differentiate through the coupling matrix
``K`` and the learned residual, so this is the production gradient path the
finite-difference estimator in :mod:`scpn_phase_orchestrator.upde.adjoint`
approximates.

The integration runs on the *unwrapped* phase: the coupling depends only on
phase differences and ``sin`` is ``2π``-periodic, so the vector field is
invariant to wrapping, while an adaptive solver must not see the ``% 2π``
discontinuities that the Euler map introduces at each step. Wrapping is applied
once, to the returned states.

The dtype of every intermediate follows the input arrays — the solver never
mutates the global ``jax_enable_x64`` flag, so callers keep the float32 default
of the rest of ``nn`` unless they opt into x64 themselves.

Requires: ``jax>=0.4``, ``equinox>=0.11``, ``diffrax>=0.5``.
"""

from __future__ import annotations

from typing import Any

import diffrax
import jax

from .functional import TWO_PI
from .ude import CouplingResidual, _ude_deriv

__all__ = ["solve_ude_adjoint"]


def _ude_vector_field(
    _t: Any,
    phases: Any,
    args: tuple[jax.Array, jax.Array, CouplingResidual],
) -> jax.Array:
    """Return dθ/dt for the UDE-Kuramoto field, packaged for ``diffrax``.

    Parameters
    ----------
    _t : jax.Array
        Integration time. The autonomous Kuramoto field ignores it.
    phases : jax.Array
        Current unwrapped oscillator phases, shape ``(N,)``.
    args : tuple[jax.Array, jax.Array, CouplingResidual]
        The ``(omegas, K, residual_fn)`` triple carried through the solver so
        gradients flow into the coupling matrix and the learned residual.

    Returns
    -------
    jax.Array
        The time derivative ``dθ/dt``, shape ``(N,)``.
    """
    omegas, coupling, residual_fn = args
    return _ude_deriv(phases, omegas, coupling, residual_fn)


def solve_ude_adjoint(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    residual_fn: CouplingResidual,
    *,
    t1: float,
    dt0: float = 0.01,
    solver: diffrax.AbstractSolver[Any] | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 4096,
    saveat_ts: jax.Array | None = None,
    wrap: bool = True,
    throw: bool = True,
) -> jax.Array:
    """Integrate UDE-Kuramoto with an adaptive solver and continuous adjoint.

    Parameters
    ----------
    phases : jax.Array
        Initial oscillator phases in radians, shape ``(N,)``.
    omegas : jax.Array
        Natural frequencies in rad/s, shape ``(N,)``.
    K : jax.Array
        Coupling matrix, shape ``(N, N)``.
    residual_fn : CouplingResidual
        Learned per-pair coupling correction.
    t1 : float
        Final integration time; the interval is ``[0, t1]``. Must be positive.
    dt0 : float
        Initial step size handed to the adaptive controller. Must be positive.
    solver : diffrax.AbstractSolver or None
        The ODE solver. Defaults to :class:`diffrax.Tsit5` (5th-order adaptive).
    adjoint : diffrax.AbstractAdjoint or None
        The reverse-mode strategy. Defaults to
        :class:`diffrax.RecursiveCheckpointAdjoint`; pass
        :class:`diffrax.BacksolveAdjoint` for ``O(1)`` memory.
    rtol : float
        Relative tolerance for the PID step-size controller. Must be positive.
    atol : float
        Absolute tolerance for the PID step-size controller. Must be positive.
    max_steps : int
        Upper bound on solver steps. Must be positive.
    saveat_ts : jax.Array or None
        Times at which to save the trajectory. ``None`` saves only the final
        state and returns shape ``(N,)``; a length-``T`` array returns shape
        ``(T, N)``.
    wrap : bool
        When ``True`` (default) the returned phases are wrapped into
        ``[0, 2π)``; when ``False`` the unwrapped phases are returned.
    throw : bool
        Stiffness guard. When ``True`` (default) a solve that exhausts
        ``max_steps`` — the symptom of a stiff or blowing-up field — raises
        instead of silently returning non-finite phases, so a diverging
        integration can never masquerade as a valid result. Set ``False`` to
        recover the non-finite solution for inspection (e.g. to locate the
        offending oscillator) rather than raising; raise ``max_steps`` or soften
        the field when this trips.

    Returns
    -------
    jax.Array
        The final phases (shape ``(N,)``) when ``saveat_ts`` is ``None``, else
        the saved trajectory (shape ``(T, N)``).

    Raises
    ------
    ValueError
        If ``t1``, ``dt0``, ``rtol``, ``atol`` or ``max_steps`` is not
        positive, or if ``phases`` is not one-dimensional.
    """
    if phases.ndim != 1:
        raise ValueError("phases must be a one-dimensional array")
    if t1 <= 0.0:
        raise ValueError("t1 must be positive")
    if dt0 <= 0.0:
        raise ValueError("dt0 must be positive")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    active_solver = diffrax.Tsit5() if solver is None else solver
    active_adjoint = (
        diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    )
    if saveat_ts is None:
        saveat = diffrax.SaveAt(t1=True)
    else:
        saveat = diffrax.SaveAt(ts=saveat_ts)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ude_vector_field),
        active_solver,
        t0=0.0,
        t1=t1,
        dt0=dt0,
        y0=phases,
        args=(omegas, K, residual_fn),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=active_adjoint,
        max_steps=max_steps,
        throw=throw,
    )

    saved = solution.ys
    states = saved if saveat_ts is not None else saved[-1]
    if wrap:
        wrapped: jax.Array = states % TWO_PI
        return wrapped
    result: jax.Array = states
    return result
