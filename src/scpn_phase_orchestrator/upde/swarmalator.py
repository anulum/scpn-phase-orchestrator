# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator dynamics

"""Swarmalator step (position + phase) with a 5-backend fallback chain.

Swarmalators combine spatial attraction / repulsion with phase
oscillator dynamics (O'Keeffe, Hong & Strogatz, *Nat. Commun.* 8:1504,
2017). Each agent has a position ``x_i ∈ ℝ^d`` and a phase ``θ_i``;
they co-evolve through attract/repulse + phase-coupling terms:

    ẋ_i = (1/N) Σ_j (x_j − x_i) [(a + j·cos(θ_j − θ_i)) / |x_j − x_i|
                                 − b / |x_j − x_i|²]
    θ̇_i = ω_i + (k / N) Σ_j sin(θ_j − θ_i) / |x_j − x_i|

The repulsion ``b·(x_j − x_i) / |x_j − x_i|²`` is the canonical
inverse-distance hard core of O'Keeffe-Hong-Strogatz (magnitude
``b / |x_j − x_i|``), with ``a = A = 1``, ``b = B = 1``, ``j = J``,
``k = K`` recovering the original model. A single regularisation
constant ``ε = 1e-6`` is added to ``|x_j − x_i|²`` (and inside the
``sqrt`` for the attraction/phase ``|x_j − x_i|``) so the kernel is
finite at coincident agents; it vanishes in the ``ε → 0`` limit.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.coupling.spatial_modulator import (
    SpatialCouplingModulator,
)

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
    from ..experimental.accelerators.upde._swarmalator_mojo import (
        _ensure_exe,
        swarmalator_step_mojo,
    )

    _ensure_exe()
    return swarmalator_step_mojo


def _load_julia_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._swarmalator_julia import (
        swarmalator_step_julia,
    )

    return swarmalator_step_julia


def _load_go_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.upde._swarmalator_go import (
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
_BACKEND_CACHE: dict[str, Callable[..., tuple[FloatArray, FloatArray]]] = {}


def _load_backend(name: str) -> Callable[..., tuple[FloatArray, FloatArray]]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., tuple[FloatArray, FloatArray]] | None:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)
    for backend in deduped:
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    return coerced


def _validate_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must be real-valued, not boolean")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued, not complex")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_backend_output(
    pos: object,
    phases: object,
    *,
    n: int,
    dim: int,
) -> tuple[FloatArray, FloatArray]:
    out_pos = _validate_state_array(
        pos,
        name="backend output positions",
        shape=(n, dim),
    )
    out_phases = _validate_state_array(
        phases,
        name="backend output phases",
        shape=(n,),
    )
    if np.any(out_phases < 0.0) or np.any(out_phases >= TWO_PI):
        raise ValueError("backend output phases must be in [0, 2*pi)")
    return out_pos, out_phases


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

    Repulsion is the canonical O'Keeffe-Hong-Strogatz inverse-distance
    core ``b·(x_j − x_i) / (|x_j − x_i|² + ε)`` (scalar ``b / (d2 + ε)``
    multiplying the separation vector, where ``d2`` is the pre-ε squared
    distance). All five backends use this identical form bit-for-bit.
    """
    eps = 1e-6
    phase_distance_weights = SpatialCouplingModulator(
        K_base=1.0,
        decay_form="inverse_distance",
        epsilon=eps,
    ).modulation_matrix(pos)
    new_pos = pos.copy()
    new_phases = phases.copy()
    for i in range(n):
        diff = pos - pos[i]
        d2 = np.sum(diff**2, axis=1)  # pre-eps
        dist = np.sqrt(d2 + eps)
        cos_diff = np.cos(phases - phases[i])
        sin_diff = np.sin(phases - phases[i])
        attract = (a + j * cos_diff) / dist
        repulse = b / (d2 + eps)  # OHS inverse-distance core
        vel = (
            np.sum(
                diff * (attract - repulse)[:, np.newaxis],
                axis=0,
            )
            / n
        )
        new_pos[i] = pos[i] + dt * vel
        dth = omegas[i] + k * float(np.mean(sin_diff * phase_distance_weights[i]))
        new_phases[i] = (phases[i] + dt * dth) % TWO_PI
    return new_pos, new_phases


class SwarmalatorEngine:
    """Swarmalator stepper with 5-backend dispatch.

    The engine is stateful in its ``(n_agents, dim, dt)`` geometry
    but the step contract is stateless: ``(pos, phases, omegas) →
    (new_pos, new_phases)``.
    """

    def __init__(self, n_agents: int, dim: int = 2, dt: float = 0.01):
        self._n = _validate_positive_int(n_agents, name="n_agents")
        self._dim = _validate_positive_int(dim, name="dim")
        self._dt = _validate_positive_float(dt, name="dt")

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
        tuple[FloatArray, FloatArray]
            Updated positions with shape ``(n_agents, dim)`` and updated
            phases wrapped into ``[0, 2*pi)``.

        Notes
        -----
        The dispatcher selects the first available accelerated backend and
        falls back to the NumPy reference path with the same state contract.
        """
        pos64 = _validate_state_array(
            pos,
            name="pos",
            shape=(self._n, self._dim),
        )
        phases64 = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        )
        omegas64 = _validate_state_array(
            omegas,
            name="omegas",
            shape=(self._n,),
        )
        a = _validate_finite_float(a, name="a")
        b = _validate_finite_float(b, name="b")
        j = _validate_finite_float(j, name="j")
        k = _validate_finite_float(k, name="k")

        backend_fn = _dispatch()
        if backend_fn is not None:
            return _validate_backend_output(
                *backend_fn(
                    pos64,
                    phases64,
                    omegas64,
                    self._n,
                    self._dim,
                    a,
                    b,
                    j,
                    k,
                    self._dt,
                ),
                n=self._n,
                dim=self._dim,
            )
        return _python_step(
            pos64,
            phases64,
            omegas64,
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
        """Integrate swarmalator positions and phases with trajectory capture."""
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        curr_pos = _validate_state_array(
            pos,
            name="pos",
            shape=(self._n, self._dim),
        ).copy()
        curr_phases = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        ).copy()
        omegas64 = _validate_state_array(
            omegas,
            name="omegas",
            shape=(self._n,),
        )
        pos_traj = np.empty((n_steps, self._n, self._dim))
        phase_traj = np.empty((n_steps, self._n))
        for i in range(n_steps):
            curr_pos, curr_phases = self.step(
                curr_pos,
                curr_phases,
                omegas64,
                a,
                b,
                j,
                k,
            )
            pos_traj[i] = curr_pos
            phase_traj[i] = curr_phases
        return curr_pos, curr_phases, pos_traj, phase_traj

    def order_parameter(self, phases: FloatArray) -> float:
        """Return the Kuramoto order parameter for swarmalator phases."""
        phases64 = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        )
        return float(np.abs(np.mean(np.exp(1j * phases64))))
