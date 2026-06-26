# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE batched-run dispatcher

"""Five-backend dispatcher for the UPDE batched integrator.

Exports:

* :data:`ACTIVE_BACKEND` / :data:`AVAILABLE_BACKENDS` — resolved at
  import time with the fastest-first preference
  (Rust → WebGPU → Mojo → Julia → Go → Python).
* :func:`upde_run` — stateless entry point that any caller can use
  directly; :class:`scpn_phase_orchestrator.upde.engine.UPDEEngine`
  also routes through here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._ref_kernel import (
    upde_run_omega_schedule_python,
    upde_run_python,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "upde_run",
    "upde_run_omega_schedule",
]


_BACKEND_NAMES = ("rust", "webgpu", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., FloatArray]:
    """Load the Rust UPDE step backend callable."""
    from spo_kernel import PyUPDEStepper  # noqa: F401

    def _rust_run(
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
        method: str,
        n_substeps: int,
        atol: float,
        rtol: float,
    ) -> FloatArray:
        """Call the Rust UPDE batched-step kernel."""
        n = int(phases.size)
        stepper = PyUPDEStepper(
            n, dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        return cast(
            "FloatArray",
            np.asarray(
                stepper.run(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                    np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                    float(zeta),
                    float(psi),
                    np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                    int(n_steps),
                ),
                dtype=np.float64,
            ),
        )

    return cast("Callable[..., FloatArray]", _rust_run)


def _load_rust_schedule_fn() -> Callable[..., FloatArray]:
    """Load the Rust UPDE schedule backend callable."""
    from spo_kernel import PyUPDEStepper

    def _rust_run_schedule(
        phases: FloatArray,
        omega_schedule: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        dt: float,
        method: str,
        n_substeps: int,
        atol: float,
        rtol: float,
    ) -> FloatArray:
        """Call the Rust UPDE schedule kernel."""
        n = int(phases.size)
        stepper = PyUPDEStepper(
            n, dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        if not hasattr(stepper, "run_omega_schedule"):
            raise ImportError(
                "spo_kernel PyUPDEStepper does not expose run_omega_schedule; "
                "rebuild spo-kernel to enable Rust omega schedule dispatch"
            )
        return cast(
            "FloatArray",
            np.asarray(
                stepper.run_omega_schedule(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omega_schedule.ravel(), dtype=np.float64),
                    np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                    float(zeta),
                    float(psi),
                    np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                    int(omega_schedule.shape[0]),
                ),
                dtype=np.float64,
            ),
        )

    return cast("Callable[..., FloatArray]", _rust_run_schedule)


def _load_mojo_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Mojo UPDE step backend callable."""
    from ..experimental.accelerators.upde._engine_mojo import (
        _ensure_exe,
        upde_run_mojo,
    )

    _ensure_exe()
    return cast("Callable[..., FloatArray]", upde_run_mojo)


def _load_mojo_schedule_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Mojo UPDE schedule backend callable."""
    from ..experimental.accelerators.upde._engine_mojo import (
        _ensure_exe,
        upde_run_omega_schedule_mojo,
    )

    _ensure_exe()
    return cast("Callable[..., FloatArray]", upde_run_omega_schedule_mojo)


def _load_webgpu_fn() -> Callable[..., FloatArray]:
    """Load the WebGPU UPDE step backend callable."""
    from ..experimental.accelerators.upde._engine_webgpu import (
        load_webgpu_dispatch_bridge,
    )

    return load_webgpu_dispatch_bridge()


def _require_juliacall_runtime() -> None:
    # pragma: no cover — toolchain
    """Import and return the juliacall runtime, else raise."""
    juliacall = importlib.import_module("juliacall")
    # The engine needs ``juliacall.Main``. When juliacall cannot finish
    # initialising the Julia runtime (for example a partial init under a
    # coverage thread tracer) the module imports but ``Main`` is absent; treat
    # that as an unavailable backend rather than letting the later engine call
    # crash with ImportError after dispatch.
    if not hasattr(juliacall, "Main"):
        raise ImportError("juliacall.Main unavailable; Julia runtime not initialised")


def _load_julia_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Julia UPDE step backend callable."""
    _require_juliacall_runtime()

    from ..experimental.accelerators.upde._engine_julia import (
        upde_run_julia,
    )

    return cast("Callable[..., FloatArray]", upde_run_julia)


def _load_julia_schedule_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Julia UPDE schedule backend callable."""
    _require_juliacall_runtime()

    from ..experimental.accelerators.upde._engine_julia import (
        upde_run_omega_schedule_julia,
    )

    return cast("Callable[..., FloatArray]", upde_run_omega_schedule_julia)


def _load_go_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Go UPDE step backend callable."""
    from ..experimental.accelerators.upde._engine_go import (
        _load_lib,
        upde_run_go,
    )

    _load_lib()
    return cast("Callable[..., FloatArray]", upde_run_go)


def _load_go_schedule_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Go UPDE schedule backend callable."""
    from ..experimental.accelerators.upde._engine_go import (
        _load_lib,
        upde_run_omega_schedule_go,
    )

    _load_lib()
    return cast("Callable[..., FloatArray]", upde_run_omega_schedule_go)


_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_fn,
    "webgpu": _load_webgpu_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_SCHEDULE_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_schedule_fn,
    "mojo": _load_mojo_schedule_fn,
    "julia": _load_julia_schedule_fn,
    "go": _load_go_schedule_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}
_SCHEDULE_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}


def _load_backend(name: str) -> Callable[..., FloatArray]:
    """Load and cache the named step backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _load_schedule_backend(name: str) -> Callable[..., FloatArray]:
    """Load and cache the named schedule backend callable."""
    cached = _SCHEDULE_BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _SCHEDULE_LOADERS[name]()
    _SCHEDULE_BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
    _BACKEND_CACHE.clear()
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


def _dispatch() -> Callable[..., FloatArray] | None:
    """Return the fastest available step backend, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            continue
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _dispatch_schedule() -> Callable[..., FloatArray] | None:
    """Return the fastest available schedule backend, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend not in _SCHEDULE_LOADERS:
            continue
        try:
            return _load_schedule_backend(backend)
        except (ImportError, RuntimeError, OSError, AttributeError):
            continue
    return None


def _validate_zero_self_coupling(knm: FloatArray) -> None:
    """Assert the coupling matrix has a zero diagonal, else raise."""
    if not np.allclose(np.diag(knm), 0.0, rtol=0.0, atol=1e-15):
        raise ValueError("knm self-coupling diagonal must be zero")


def _validate_omega_schedule(schedule: FloatArray, *, n: int) -> None:
    """Return the validated natural-frequency schedule, else raise."""
    if schedule.dtype == np.bool_ or np.iscomplexobj(schedule):
        raise ValueError("omega_schedule must be real-valued")
    if schedule.ndim != 2:
        raise ValueError("omega_schedule must be a two-dimensional matrix")
    if schedule.shape[0] < 1:
        raise ValueError("omega_schedule must contain at least one step")
    if schedule.shape[1] != n:
        raise ValueError("omega_schedule column count must match phases")
    if not np.all(np.isfinite(schedule)):
        raise ValueError("omega_schedule contains NaN/Inf")


def upde_run(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str = "euler",
    n_substeps: int = 1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> FloatArray:
    """Stateless batched UPDE integrator.

    Dispatches to the first available backend per the SPO chain
    (Rust → WebGPU → Mojo → Julia → Go → Python). Every backend runs the
    same algorithm: ``method ∈ {"euler", "rk4", "rk45"}`` with
    ``n_substeps`` applied to the fixed-step methods; RK45 is
    adaptive with ``(atol, rtol)`` tolerances. Phases are wrapped
    to ``[0, 2π)`` after each outer step.
    """
    p = np.ascontiguousarray(phases, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    k = np.ascontiguousarray(knm, dtype=np.float64)
    a = np.ascontiguousarray(alpha, dtype=np.float64)
    _validate_zero_self_coupling(k)
    backend_fn = _dispatch()
    if backend_fn is None:
        return upde_run_python(
            p,
            o,
            k,
            a,
            float(zeta),
            float(psi),
            float(dt),
            int(n_steps),
            method,
            int(n_substeps),
            float(atol),
            float(rtol),
        )
    return cast(
        "FloatArray",
        np.asarray(
            backend_fn(
                p,
                o,
                k,
                a,
                float(zeta),
                float(psi),
                float(dt),
                int(n_steps),
                method,
                int(n_substeps),
                float(atol),
                float(rtol),
            ),
            dtype=np.float64,
        ),
    )


def upde_run_omega_schedule(
    phases: FloatArray,
    omega_schedule: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    method: str = "euler",
    n_substeps: int = 1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> FloatArray:
    """Run UPDE with one resolved natural-frequency vector per outer step."""
    p = np.ascontiguousarray(phases, dtype=np.float64)
    raw_schedule = np.asarray(omega_schedule)
    k = np.ascontiguousarray(knm, dtype=np.float64)
    a = np.ascontiguousarray(alpha, dtype=np.float64)
    _validate_zero_self_coupling(k)
    _validate_omega_schedule(raw_schedule, n=int(p.size))
    schedule = np.ascontiguousarray(raw_schedule, dtype=np.float64)
    backend_fn = _dispatch_schedule()
    if backend_fn is None:
        return upde_run_omega_schedule_python(
            p,
            schedule,
            k,
            a,
            float(zeta),
            float(psi),
            float(dt),
            method,
            int(n_substeps),
            float(atol),
            float(rtol),
        )
    try:
        return cast(
            "FloatArray",
            np.asarray(
                backend_fn(
                    p,
                    schedule,
                    k,
                    a,
                    float(zeta),
                    float(psi),
                    float(dt),
                    method,
                    int(n_substeps),
                    float(atol),
                    float(rtol),
                ),
                dtype=np.float64,
            ),
        )
    except (AttributeError, ImportError):
        return upde_run_omega_schedule_python(
            p,
            schedule,
            k,
            a,
            float(zeta),
            float(psi),
            float(dt),
            method,
            int(n_substeps),
            float(atol),
            float(rtol),
        )
