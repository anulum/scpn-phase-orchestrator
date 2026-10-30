# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE batched-run dispatcher

"""Six-backend dispatcher for the UPDE batched integrator.

Exports:

* :data:`ACTIVE_BACKEND` / :data:`AVAILABLE_BACKENDS` — resolved on first
  use with the fastest-first preference
  (Rust → WebGPU → Mojo → Julia → Go → Python).
* :func:`upde_run` — stateless entry point that any caller can use
  directly; :class:`scpn_phase_orchestrator.upde.engine.UPDEEngine`
  also routes through here.
* :func:`upde_run_backend` — names the owner that serves a step or
  schedule run, under ``"auto"`` or an explicit owner name.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
    validate_upde_schedule_backend_inputs,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main
from scpn_phase_orchestrator.upde._ref_kernel import (
    upde_run_omega_schedule_python,
    upde_run_python,
)

FloatArray: TypeAlias = NDArray[np.float64]
ACTIVE_BACKEND: str
AVAILABLE_BACKENDS: list[str]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "upde_run",
    "upde_run_backend",
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
        return validate_upde_backend_output(
            stepper.run(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                int(n_steps),
            ),
            n=n,
            wrapped=n_steps > 0,
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
        return validate_upde_backend_output(
            stepper.run_omega_schedule(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omega_schedule.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                int(omega_schedule.shape[0]),
            ),
            n=n,
        )

    return cast("Callable[..., FloatArray]", _rust_run_schedule)


def _load_mojo_fn() -> Callable[..., FloatArray]:
    """Load the Mojo UPDE step backend callable."""
    from ..experimental.accelerators.upde._engine_mojo import (
        _ensure_exe,
        upde_run_mojo,
    )

    _ensure_exe()
    return cast("Callable[..., FloatArray]", upde_run_mojo)


def _load_mojo_schedule_fn() -> Callable[..., FloatArray]:
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
    """Import the juliacall runtime, else raise ``ImportError``.

    Delegates to the shared UPDE probe, which treats a missing ``Main`` (a
    partial runtime init, for example under a coverage thread tracer) and a
    broken Julia set-up alike as an unavailable backend.
    """
    require_juliacall_main()


def _load_julia_fn() -> Callable[..., FloatArray]:
    """Load the Julia UPDE step backend callable."""
    _require_juliacall_runtime()

    from ..experimental.accelerators.upde._engine_julia import (
        upde_run_julia,
    )

    return cast("Callable[..., FloatArray]", upde_run_julia)


def _load_julia_schedule_fn() -> Callable[..., FloatArray]:
    """Load the Julia UPDE schedule backend callable."""
    _require_juliacall_runtime()

    from ..experimental.accelerators.upde._engine_julia import (
        upde_run_omega_schedule_julia,
    )

    return cast("Callable[..., FloatArray]", upde_run_omega_schedule_julia)


def _load_go_fn() -> Callable[..., FloatArray]:
    """Load the Go UPDE step backend callable."""
    from ..experimental.accelerators.upde._engine_go import (
        _load_lib,
        upde_run_go,
    )

    _load_lib()
    return cast("Callable[..., FloatArray]", upde_run_go)


def _load_go_schedule_fn() -> Callable[..., FloatArray]:
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


_BACKEND_STATE_LOCK = Lock()


def _backend_state() -> tuple[str, list[str]]:
    """Resolve backend availability once, on first computation or inspection."""
    active = globals().get("ACTIVE_BACKEND")
    available = globals().get("AVAILABLE_BACKENDS")
    if isinstance(active, str) and isinstance(available, list):
        return active, available
    with _BACKEND_STATE_LOCK:
        active = globals().get("ACTIVE_BACKEND")
        available = globals().get("AVAILABLE_BACKENDS")
        if not isinstance(active, str) or not isinstance(available, list):
            active, available = _resolve_backends()
            globals()["ACTIVE_BACKEND"] = active
            globals()["AVAILABLE_BACKENDS"] = available
    return active, available


def __getattr__(name: str) -> object:
    """Resolve public backend status only when a caller requests it."""
    if name in {"ACTIVE_BACKEND", "AVAILABLE_BACKENDS"}:
        active, available = _backend_state()
        return {"ACTIVE_BACKEND": active, "AVAILABLE_BACKENDS": available}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_SLOTS = ("step", "schedule")
_STEP_LOAD_FAILURES = (ImportError, RuntimeError, OSError, KeyError)
_SCHEDULE_LOAD_FAILURES = (ImportError, RuntimeError, OSError, AttributeError)


def _auto_select(slot: str) -> tuple[str, Callable[..., FloatArray] | None]:
    """Return the first loadable owner of ``slot`` in preference order.

    The active backend is tried first, then the available list; an owner
    whose loader fails is skipped, and ``python`` ends the search with no
    native callable.
    """
    active, available = _backend_state()
    seen: set[str] = set()
    for backend in [active, *available]:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return "python", None
        if slot == "schedule":
            if backend not in _SCHEDULE_LOADERS:
                continue
            try:
                return backend, _load_schedule_backend(backend)
            except _SCHEDULE_LOAD_FAILURES:
                continue
        try:
            return backend, _load_backend(backend)
        except _STEP_LOAD_FAILURES:
            continue
    return "python", None


def _named_select(slot: str, backend: str) -> Callable[..., FloatArray] | None:
    """Return the named owner's ``slot`` callable, or ``None`` for Python.

    Raises
    ------
    RuntimeError
        If the owner has no kernel for ``slot`` or its loader fails.
    """
    if backend == "python":
        return None
    loaders = _LOADERS if slot == "step" else _SCHEDULE_LOADERS
    if backend not in loaders:
        raise RuntimeError(f"UPDE backend {backend!r} has no {slot} kernel")
    try:
        if slot == "step":
            return _load_backend(backend)
        return _load_schedule_backend(backend)
    except (*_STEP_LOAD_FAILURES, AttributeError) as exc:
        raise RuntimeError(f"UPDE backend {backend!r} unavailable") from exc


def _require_known(slot: str, backend: str) -> None:
    """Refuse an unknown slot or owner name before any computation."""
    if slot not in _SLOTS:
        raise ValueError(f"unknown UPDE run slot: {slot!r}")
    if backend not in (*_BACKEND_NAMES, "auto"):
        raise ValueError(f"unknown UPDE backend: {backend!r}")


def _dispatch() -> Callable[..., FloatArray] | None:
    """Prefer the selected step backend, returning ``None`` for Python."""
    return _auto_select("step")[1]


def _dispatch_schedule() -> Callable[..., FloatArray] | None:
    """Prefer the selected schedule backend, returning ``None`` for Python."""
    return _auto_select("schedule")[1]


def upde_run_backend(slot: str = "step", backend: str = "auto") -> str:
    """Return the owner that serves a UPDE run.

    Parameters
    ----------
    slot : str
        ``"step"`` for :func:`upde_run` or ``"schedule"`` for
        :func:`upde_run_omega_schedule`.
    backend : str
        ``"auto"`` or one of ``rust``, ``webgpu``, ``mojo``, ``julia``,
        ``go``, ``python``.

    Returns
    -------
    str
        The owner name.

    Raises
    ------
    ValueError
        If ``slot`` or ``backend`` is unknown.
    RuntimeError
        If a named owner has no kernel for ``slot`` or is unavailable.
    """
    _require_known(slot, backend)
    if backend == "auto":
        return _auto_select(slot)[0]
    _named_select(slot, backend)
    return backend


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
    backend: str = "auto",
) -> FloatArray:
    """Run the stateless UPDE integrator through the selected real backend.

    Euler, RK4 and RK45 use float64 in the native and NumPy chain. The
    browser WebGPU bridge supports float32 Euler only. Integrated outputs
    lie in [0, 2*pi); rounded endpoints and signed zeros become positive
    zero. Zero steps return an independent unwrapped copy.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Finite phase-lag matrix in radians, shape ``(N, N)``; use zeros for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    dt : float
        Integration step size.
    n_steps : int
        Number of integration steps to run.
    method : str
        Integration method (``euler``, ``rk4``, or ``rk45``).
    n_substeps : int
        Number of inner substeps per outer step.
    atol : float
        Absolute tolerance for the adaptive (rk45) integrator.
    rtol : float
        Relative tolerance for the adaptive (rk45) integrator.
    backend : str
        ``"auto"`` for the fastest loadable owner, or one of ``rust``,
        ``webgpu``, ``mojo``, ``julia``, ``go``, ``python``.

    Returns
    -------
    FloatArray
        The final phases after ``n_steps`` integration steps.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, or the inputs or the backend output are
        invalid.
    RuntimeError
        If the named owner is unavailable.
    """
    _require_known("step", backend)
    (
        p,
        o,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_upde_backend_inputs(
        phases,
        omegas,
        knm,
        alpha,
        zeta,
        psi,
        dt,
        n_steps,
        method,
        n_substeps,
        atol,
        rtol,
    )
    n = int(p.size)
    k_matrix = k.reshape((n, n))
    a_matrix = a.reshape((n, n))
    backend_fn = _dispatch() if backend == "auto" else _named_select("step", backend)
    if backend_fn is None:
        result = upde_run_python(
            p,
            o,
            k_matrix,
            a_matrix,
            zeta_f,
            psi_f,
            dt_f,
            n_steps_i,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        )
    else:
        result = backend_fn(
            p,
            o,
            k_matrix,
            a_matrix,
            zeta_f,
            psi_f,
            dt_f,
            n_steps_i,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        )
    return validate_upde_backend_output(result, n=n, wrapped=n_steps_i > 0)


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
    backend: str = "auto",
) -> FloatArray:
    """Run UPDE with one resolved natural-frequency vector per outer step.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : FloatArray
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Finite phase-lag matrix in radians, shape ``(N, N)``; use zeros for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    dt : float
        Integration step size.
    method : str
        Integration method (``euler``, ``rk4``, or ``rk45``).
    n_substeps : int
        Number of inner substeps per outer step.
    atol : float
        Absolute tolerance for the adaptive (rk45) integrator.
    rtol : float
        Relative tolerance for the adaptive (rk45) integrator.
    backend : str
        ``"auto"`` for the fastest loadable schedule owner, or one of
        ``rust``, ``mojo``, ``julia``, ``go``, ``python``. WebGPU has no
        schedule kernel.

    Returns
    -------
    FloatArray
        The final phases after integrating the omega schedule.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, or the inputs or the backend output are
        invalid.
    RuntimeError
        If the named owner has no schedule kernel, is unavailable, or fails
        for want of a symbol; under ``"auto"`` such a call-time failure falls
        back to the Python reference.
    """
    _require_known("schedule", backend)
    (
        p,
        schedule,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        _n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_upde_schedule_backend_inputs(
        phases,
        omega_schedule,
        knm,
        alpha,
        zeta,
        psi,
        dt,
        method,
        n_substeps,
        atol,
        rtol,
    )
    n = int(p.size)
    k_matrix = k.reshape((n, n))
    a_matrix = a.reshape((n, n))
    backend_fn = (
        _dispatch_schedule()
        if backend == "auto"
        else _named_select("schedule", backend)
    )
    if backend_fn is None:
        result = upde_run_omega_schedule_python(
            p,
            schedule,
            k_matrix,
            a_matrix,
            zeta_f,
            psi_f,
            dt_f,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        )
    else:
        try:
            result = backend_fn(
                p,
                schedule,
                k_matrix,
                a_matrix,
                zeta_f,
                psi_f,
                dt_f,
                method_s,
                n_substeps_i,
                atol_f,
                rtol_f,
            )
        except (AttributeError, ImportError) as exc:
            if backend != "auto":
                raise RuntimeError(f"UPDE backend {backend!r} failed") from exc
            result = upde_run_omega_schedule_python(
                p,
                schedule,
                k_matrix,
                a_matrix,
                zeta_f,
                psi_f,
                dt_f,
                method_s,
                n_substeps_i,
                atol_f,
                rtol_f,
            )
    return validate_upde_backend_output(result, n=n)
