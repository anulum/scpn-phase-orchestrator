# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator dynamics

"""Swarmalator step (position + phase) with a 5-backend fallback
chain per ``feedback_module_standard_attnres.md``.

Swarmalators combine spatial attraction / repulsion with phase
oscillator dynamics (O'Keeffe & Strogatz 2017). Each agent has a
position ``x_i ∈ ℝ^d`` and a phase ``θ_i``; they co-evolve through
attract/repulse + phase-coupling terms:

    ẋ_i = (1/N) Σ_j (x_j − x_i) [(a + j·cos(θ_j − θ_i)) / |x_j − x_i|
                                 − b / (|x_j − x_i|³ + ε)]
    θ̇_i = ω_i + (k / N) Σ_j sin(θ_j − θ_i) / |x_j − x_i|

Regularisation constants ``1e-6`` inside ``sqrt`` and the repulse
cube guard against singularities at coincident agents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "SwarmalatorEngine",
]

FloatArray: TypeAlias = NDArray[np.float64]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    from spo_kernel import PySwarmalatorStepper

    def _rust(
        pos: FloatArray,
        phases: FloatArray,
        omegas: FloatArray,
        n: int,
        dim: int,
        a: float,
        b: float,
        j: float,
        k: float,
        dt: float,
    ) -> tuple[FloatArray, FloatArray]:
        stepper = PySwarmalatorStepper(int(n), int(dim), float(dt))
        new_pos, new_phases = stepper.step(
            np.ascontiguousarray(pos.ravel(), dtype=np.float64),
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
            float(a),
            float(b),
            float(j),
            float(k),
        )
        return (
            np.asarray(new_pos, dtype=np.float64).reshape(n, dim),
            np.asarray(new_phases, dtype=np.float64),
        )

    return cast("Callable[..., tuple[FloatArray, FloatArray]]", _rust)


def _load_mojo_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._swarmalator_mojo import (
        _ensure_exe,
        swarmalator_step_mojo,
    )

    _ensure_exe()
    return swarmalator_step_mojo


def _load_julia_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.upde._swarmalator_julia import (
        swarmalator_step_julia,
    )

    return swarmalator_step_julia


def _load_go_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._swarmalator_go import (
        _load_lib,
        swarmalator_step_go,
    )

    _load_lib()
    return swarmalator_step_go


_LOADERS: dict[
    str,
    Callable[[], Callable[..., tuple[FloatArray, FloatArray]]],
] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., tuple[FloatArray, FloatArray]] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _python_step(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Python reference matching the Rust kernel exactly.

    Note on ``repulse``: Rust uses ``b / (dist · d²ₛᵤₘ + eps)`` where
    ``d²ₛᵤₘ`` is the pre-eps squared sum, not ``b / (dist³ + eps)``.
    The two formulas agree in the ``eps → 0`` limit but drift at
    small distances; the pre-migration Python reference used the
    ``dist³`` variant. All non-Rust backends now match Rust.
    """
    eps = 1e-6
    new_pos = pos.copy()
    new_phases = phases.copy()
    for i in range(n):
        diff = pos - pos[i]
        d2 = np.sum(diff**2, axis=1)  # pre-eps
        dist = np.sqrt(d2 + eps)
        cos_diff = np.cos(phases - phases[i])
        sin_diff = np.sin(phases - phases[i])
        attract = (a + j * cos_diff) / dist
        repulse = b / (dist * d2 + eps)  # Rust semantics
        vel = (
            np.sum(
                diff * (attract - repulse)[:, np.newaxis],
                axis=0,
            )
            / n
        )
        new_pos[i] = pos[i] + dt * vel
        dth = omegas[i] + k * float(np.mean(sin_diff / dist))
        new_phases[i] = (phases[i] + dt * dth) % TWO_PI
    return new_pos, new_phases


class SwarmalatorEngine:
    """Swarmalator stepper with 5-backend dispatch.

    The engine is stateful in its ``(n_agents, dim, dt)`` geometry
    but the step contract is stateless: ``(pos, phases, omegas) →
    (new_pos, new_phases)``.
    """

    def __init__(self, n_agents: int, dim: int = 2, dt: float = 0.01):
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._n = n_agents
        self._dim = dim
        self._dt = dt

    def step(
        self,
        pos: FloatArray,
        phases: FloatArray,
        omegas: FloatArray,
        a: float = 1.0,
        b: float = 1.0,
        j: float = 1.0,
        k: float = 1.0,
    ) -> tuple[FloatArray, FloatArray]:
        """Advance coupled swarmalator positions and phases by one step.

        Parameters
        ----------
        pos
            Agent positions with shape ``(n_agents, dim)``.
        phases
            Agent phases in radians, shape ``(n_agents,)``.
        omegas
            Natural angular frequencies, shape ``(n_agents,)``.
        a
            Baseline spatial attraction coefficient.
        b
            Spatial repulsion coefficient.
        j
            Phase-dependent attraction modulation.
        k
            Phase-coupling coefficient.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Updated positions with shape ``(n_agents, dim)`` and updated
            phases wrapped into ``[0, 2*pi)``.

        Notes
        -----
        The dispatcher selects the first available accelerated backend and
        falls back to the NumPy reference path with the same state contract.
        """
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                pos,
                phases,
                omegas,
                self._n,
                self._dim,
                a,
                b,
                j,
                k,
                self._dt,
            )
        return _python_step(
            pos,
            phases,
            omegas,
            self._n,
            self._dim,
            a,
            b,
            j,
            k,
            self._dt,
        )

    def run(
        self,
        pos: FloatArray,
        phases: FloatArray,
        omegas: FloatArray,
        a: float = 1.0,
        b: float = 1.0,
        j: float = 1.0,
        k: float = 1.0,
        n_steps: int = 100,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        curr_pos, curr_phases = pos.copy(), phases.copy()
        pos_traj = np.empty((n_steps, self._n, self._dim))
        phase_traj = np.empty((n_steps, self._n))
        for i in range(n_steps):
            curr_pos, curr_phases = self.step(
                curr_pos,
                curr_phases,
                omegas,
                a,
                b,
                j,
                k,
            )
            pos_traj[i] = curr_pos
            phase_traj[i] = curr_phases
        return curr_pos, curr_phases, pos_traj, phase_traj

    def order_parameter(self, phases: FloatArray) -> float:
        return float(np.abs(np.mean(np.exp(1j * phases))))
