# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Doppler-corrected UPDE engine

"""Doppler-corrected Kuramoto UPDE integration.

``DopplerEngine`` augments the standard Sakaguchi-Kuramoto UPDE with a
relative-velocity correction. For oscillator ``i`` the correction is the
coupling-graph weighted relative velocity

``D_i = s * mean_j(|K_ij| * (v_i - v_j) / (|v_i| + eps))``

where the mean is normalised by the active absolute coupling row. Normalising
by the row mass keeps the Doppler strength independent of graph degree while
still respecting the active coupling topology.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import suppress
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._ref_kernel import upde_run_omega_schedule_python
from scpn_phase_orchestrator.upde.engine import UPDEEngine

__all__ = [
    "DopplerEngine",
    "doppler_run",
    "doppler_run_python",
    "doppler_term",
    "validate_doppler_backend_inputs",
    "validate_doppler_backend_output",
]

FloatArray: TypeAlias = NDArray[np.float64]
ArraySource: TypeAlias = FloatArray | Callable[[float], FloatArray]
BackendFn: TypeAlias = Callable[..., FloatArray]

_TWO_PI = 2.0 * np.pi
_BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")


def _reject_non_real_array(values: object, *, name: str) -> FloatArray:
    """Raise if the array contains non-real boolean or complex values."""
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f"{name} must be real-valued, not boolean")
    if arr.dtype == object:
        raise ValueError(f"{name} must be numeric, not object dtype")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN/Inf")
    return out


def _finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _normalise_axis(axis: object, *, dimension: int) -> FloatArray:
    """Return the motion axis as a validated unit vector, else raise."""
    axis_arr = _reject_non_real_array(axis, name="velocity_axis")
    if axis_arr.shape != (dimension,):
        raise ValueError("velocity_axis shape must match velocity vector dimension")
    norm = float(np.linalg.norm(axis_arr))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("velocity_axis must have non-zero finite norm")
    return np.ascontiguousarray(axis_arr / norm, dtype=np.float64)


def scalarise_velocities(
    velocities: object,
    *,
    n: int,
    velocity_axis: object | None = None,
) -> FloatArray:
    """Convert scalar or vector velocities to one signed scalar per oscillator."""
    arr = _reject_non_real_array(velocities, name="velocities")
    if arr.shape == (n,):
        return arr
    if arr.ndim == 2 and arr.shape[0] == n:
        if velocity_axis is None:
            return np.ascontiguousarray(np.linalg.norm(arr, axis=1), dtype=np.float64)
        axis = _normalise_axis(velocity_axis, dimension=int(arr.shape[1]))
        return np.ascontiguousarray(arr @ axis, dtype=np.float64)
    raise ValueError("velocities must have shape (n,) or (n, d)")


def _validate_knm(knm: object, *, n: int) -> FloatArray:
    """Return the coupling as a validated finite square matrix, else raise."""
    k = _reject_non_real_array(knm, name="knm")
    if k.shape != (n, n):
        raise ValueError("knm shape must be (n, n)")
    if not np.allclose(np.diag(k), 0.0, atol=0.0):
        raise ValueError("knm diagonal must be zero for Doppler coupling")
    return k


def _validate_alpha(alpha: object, *, n: int) -> FloatArray:
    """Return the phase-lag matrix as a validated finite array, else raise."""
    if np.isscalar(alpha):
        scalar = _finite_float(alpha, name="alpha")
        a = np.full((n, n), scalar, dtype=np.float64)
        np.fill_diagonal(a, 0.0)
        return a
    a = _reject_non_real_array(alpha, name="alpha")
    if a.shape != (n, n):
        raise ValueError("alpha shape must be (n, n)")
    return a


def _validate_schedule(schedule: object, *, n: int, name: str) -> FloatArray:
    """Return the validated motion schedule, else raise."""
    out = _reject_non_real_array(schedule, name=name)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if out.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one step")
    if out.shape[1] != n:
        raise ValueError(f"{name} column count must match oscillator count")
    return out


def _validate_phases(phases: object, *, n: int) -> FloatArray:
    """Return the phases as a validated 1-D finite array, else raise."""
    p = _reject_non_real_array(phases, name="phases")
    if p.shape != (n,):
        raise ValueError("phases shape must be (n,)")
    return p


def validate_doppler_backend_output(value: object, *, n: int) -> FloatArray:
    """Validate Doppler backend output before returning it to callers.

    Parameters
    ----------
    value : object
        Backend-produced oscillator phases in radians, shape ``(N,)``.
    n : int
        Expected oscillator count.

    Returns
    -------
    FloatArray
        Contiguous ``float64`` phase vector in the principal ``[0, 2*pi)`` branch.

    Raises
    ------
    ValueError
        If the backend output is non-finite, has the wrong shape, or leaves the
        principal phase branch.
    """
    out = _validate_phases(value, n=n)
    if np.any(out < 0.0) or np.any(out >= _TWO_PI):
        raise ValueError("Doppler backend output phases must be in [0, 2*pi)")
    return np.ascontiguousarray(out, dtype=np.float64)


def _validate_positive_step_count(value: object, *, name: str) -> int:
    """Return the step count as a positive integer, else raise."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    out = int(value)
    if out < 1 or out != value:
        raise ValueError(f"{name} must be a positive integer")
    return out


def _validate_method(method: str) -> str:
    """Return the supported integration-method name, else raise."""
    if method not in {"euler", "rk4", "rk45"}:
        raise ValueError("method must be 'euler', 'rk4', or 'rk45'")
    return method


def doppler_term(
    velocities: object,
    knm: object,
    *,
    doppler_strength: float = 1.0,
    doppler_epsilon: float = 1.0e-9,
    velocity_axis: object | None = None,
) -> FloatArray:
    """Return the graph-weighted Doppler correction for each oscillator.

    Parameters
    ----------
    velocities : object
        Per-oscillator axial velocities, shape ``(N,)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    doppler_strength : float
        Doppler coupling-correction strength.
    doppler_epsilon : float
        Numerical floor guarding the Doppler denominator.
    velocity_axis : object | None
        Optional unit axis along which velocity is projected, or ``None``.

    Returns
    -------
    FloatArray
        The graph-weighted Doppler correction per oscillator.

    Raises
    ------
    ValueError
        If the velocity or coupling inputs are non-finite or mismatched.
    """
    k_raw = _reject_non_real_array(knm, name="knm")
    if k_raw.ndim != 2 or k_raw.shape[0] != k_raw.shape[1]:
        raise ValueError("knm shape must be (n, n)")
    n = int(k_raw.shape[0])
    k = _validate_knm(k_raw, n=n)
    speed = scalarise_velocities(velocities, n=n, velocity_axis=velocity_axis)
    strength = _finite_float(doppler_strength, name="doppler_strength")
    epsilon = _finite_float(doppler_epsilon, name="doppler_epsilon")
    if epsilon <= 0.0:
        raise ValueError("doppler_epsilon must be positive")

    weights = np.abs(k)
    np.fill_diagonal(weights, 0.0)
    row_mass = weights.sum(axis=1)
    relative = (speed[:, None] - speed[None, :]) / (np.abs(speed)[:, None] + epsilon)
    weighted = np.sum(weights * relative, axis=1)
    term = np.divide(
        weighted, row_mass, out=np.zeros(n, dtype=np.float64), where=row_mass > 0.0
    )
    return np.ascontiguousarray(strength * term, dtype=np.float64)


def _effective_omega_schedule(
    omega_schedule: FloatArray,
    velocity_schedule: FloatArray,
    knm: FloatArray,
    *,
    doppler_strength: float,
    doppler_epsilon: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the Doppler-shifted natural-frequency schedule."""
    terms = np.vstack(
        [
            doppler_term(
                velocity_schedule[step],
                knm,
                doppler_strength=doppler_strength,
                doppler_epsilon=doppler_epsilon,
            )
            for step in range(int(omega_schedule.shape[0]))
        ]
    )
    return np.ascontiguousarray(omega_schedule + terms, dtype=np.float64), terms


def validate_doppler_backend_inputs(
    phases: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    doppler_strength: float,
    doppler_epsilon: float,
    zeta: float,
    psi: float,
    dt: float,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    float,
    float,
    int,
    str,
    int,
    float,
    float,
]:
    """Validate the backend-neutral Doppler schedule contract.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    doppler_strength : float
        Doppler coupling-correction strength.
    doppler_epsilon : float
        Numerical floor guarding the Doppler denominator.
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

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float, float,
    float, float, float, int, str, int, float, float]
        The validated, canonicalised Doppler schedule contract tuple.

    Raises
    ------
    ValueError
        If any schedule array is non-finite or has an inconsistent shape.
    """
    p_raw = _reject_non_real_array(phases, name="phases")
    if p_raw.ndim != 1 or p_raw.size < 1:
        raise ValueError("phases must be a non-empty vector")
    n = int(p_raw.size)
    p = _validate_phases(p_raw, n=n)
    omega = _validate_schedule(omega_schedule, n=n, name="omega_schedule")
    velocities = _validate_schedule(velocity_schedule, n=n, name="velocity_schedule")
    if velocities.shape[0] != omega.shape[0]:
        raise ValueError("velocity_schedule step count must match omega_schedule")
    k = _validate_knm(knm, n=n)
    a = _validate_alpha(alpha, n=n)
    strength = _finite_float(doppler_strength, name="doppler_strength")
    epsilon = _finite_float(doppler_epsilon, name="doppler_epsilon")
    if epsilon <= 0.0:
        raise ValueError("doppler_epsilon must be positive")
    zeta_f = _finite_float(zeta, name="zeta")
    psi_f = _finite_float(psi, name="psi")
    dt_f = _finite_float(dt, name="dt")
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    method_s = _validate_method(method)
    n_substeps_i = _validate_positive_step_count(n_substeps, name="n_substeps")
    atol_f = _finite_float(atol, name="atol")
    rtol_f = _finite_float(rtol, name="rtol")
    if atol_f <= 0.0 or rtol_f <= 0.0:
        raise ValueError("atol and rtol must be positive")
    return (
        p,
        omega,
        k,
        a,
        velocities,
        strength,
        epsilon,
        zeta_f,
        psi_f,
        dt_f,
        int(omega.shape[0]),
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    )


def doppler_run_python(
    phases: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    doppler_strength: float,
    doppler_epsilon: float,
    zeta: float,
    psi: float,
    dt: float,
    method: str = "rk45",
    n_substeps: int = 1,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-3,
) -> FloatArray:
    """Run the Doppler-corrected UPDE schedule in the Python reference path.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    doppler_strength : float
        Doppler coupling-correction strength.
    doppler_epsilon : float
        Numerical floor guarding the Doppler denominator.
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

    Returns
    -------
    FloatArray
        The final phases after running the Doppler schedule on the Python path.
    """
    (
        p,
        omega,
        k,
        a,
        velocities,
        strength,
        epsilon,
        zeta_f,
        psi_f,
        dt_f,
        _n_steps,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_doppler_backend_inputs(
        phases,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        doppler_strength,
        doppler_epsilon,
        zeta,
        psi,
        dt,
        method,
        n_substeps,
        atol,
        rtol,
    )
    effective, _terms = _effective_omega_schedule(
        omega,
        velocities,
        k,
        doppler_strength=strength,
        doppler_epsilon=epsilon,
    )
    return upde_run_omega_schedule_python(
        p,
        effective,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    )


def _rust_backend() -> BackendFn:
    """Load the Rust Doppler backend callable."""
    from spo_kernel import PyUPDEStepper

    if not hasattr(PyUPDEStepper, "run_doppler_schedule"):
        raise ImportError("PyUPDEStepper.run_doppler_schedule is unavailable")

    def run(
        phases: FloatArray,
        omega_schedule: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        velocity_schedule: FloatArray,
        doppler_strength: float,
        doppler_epsilon: float,
        zeta: float,
        psi: float,
        dt: float,
        method: str,
        n_substeps: int,
        atol: float,
        rtol: float,
    ) -> FloatArray:
        """Call the Rust Doppler schedule kernel."""
        stepper = PyUPDEStepper(
            int(phases.size), dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        return np.asarray(
            stepper.run_doppler_schedule(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omega_schedule.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                np.ascontiguousarray(velocity_schedule.ravel(), dtype=np.float64),
                float(doppler_strength),
                float(doppler_epsilon),
                int(omega_schedule.shape[0]),
            ),
            dtype=np.float64,
        )

    return run


def _backend_map() -> dict[str, BackendFn]:
    """Return the mapping of backend names to their loaders."""
    backends: dict[str, BackendFn] = {"python": doppler_run_python}
    with suppress(ImportError):
        go_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_go"
        )
        backends["go"] = go_mod.doppler_run_go
    with suppress(ImportError):
        julia_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_julia"
        )
        backends["julia"] = julia_mod.doppler_run_julia
    with suppress(ImportError):
        mojo_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_mojo"
        )
        backends["mojo"] = mojo_mod.doppler_run_mojo
    with suppress(ImportError):
        backends["rust"] = _rust_backend()
    return backends


def doppler_run(
    phases: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    doppler_strength: float = 1.0,
    doppler_epsilon: float = 1.0e-9,
    zeta: float = 0.0,
    psi: float = 0.0,
    dt: float = 0.01,
    method: str = "rk45",
    n_substeps: int = 1,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-3,
    *,
    backend: str = "auto",
) -> FloatArray:
    """Run a Doppler-corrected UPDE schedule through the selected backend.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    doppler_strength : float
        Doppler coupling-correction strength.
    doppler_epsilon : float
        Numerical floor guarding the Doppler denominator.
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
        Name of the compute backend to run.

    Returns
    -------
    FloatArray
        The final phases after running the Doppler schedule on the selected backend.

    Raises
    ------
    ImportError
        If the selected backend's runtime is unavailable.
    ValueError
        If the schedule contract is invalid; the underlying backend error propagates
        when every backend fails.
    """
    validated = validate_doppler_backend_inputs(
        phases,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        doppler_strength,
        doppler_epsilon,
        zeta,
        psi,
        dt,
        method,
        n_substeps,
        atol,
        rtol,
    )
    (
        p,
        omega,
        k,
        a,
        velocities,
        strength,
        epsilon,
        zeta_f,
        psi_f,
        dt_f,
        _n_steps,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validated
    backends = _backend_map()
    if backend != "auto" and backend not in backends:
        raise ImportError(f"Doppler backend {backend!r} is not available")
    order = _BACKEND_ORDER if backend == "auto" else (backend,)
    last_error: Exception | None = None
    for name in order:
        fn = backends.get(name)
        if fn is None:
            continue
        try:
            out = fn(
                p,
                omega,
                k,
                a,
                velocities,
                strength,
                epsilon,
                zeta_f,
                psi_f,
                dt_f,
                method_s,
                n_substeps_i,
                atol_f,
                rtol_f,
            )
            return validate_doppler_backend_output(out, n=int(p.size))
        except (AttributeError, ImportError) as exc:
            last_error = exc
            continue
    if backend != "auto" and last_error is not None:
        raise last_error
    return doppler_run_python(
        p,
        omega,
        k,
        a,
        velocities,
        strength,
        epsilon,
        zeta_f,
        psi_f,
        dt_f,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    )


class DopplerEngine(UPDEEngine):
    """Stateful UPDE engine with graph-weighted Doppler velocity correction."""

    def __init__(
        self,
        n: int,
        omega: object,
        k_nm: object,
        alpha: object = 0.0,
        dt: float = 0.01,
        velocities: object | Callable[[float], object] | None = None,
        doppler_strength: float = 1.0,
        doppler_epsilon: float = 1.0e-9,
        solver: str = "rk45",
        phases: object | None = None,
        velocity_axis: object | None = None,
        t0: float = 0.0,
    ) -> None:
        if velocities is None:
            raise ValueError("velocities are required for DopplerEngine")
        super().__init__(n, dt=dt, method=solver, omega=omega, t0=t0)
        self.k_nm = _validate_knm(k_nm, n=n)
        self.alpha_matrix = _validate_alpha(alpha, n=n)
        self._velocity_source = velocities
        self._velocity_axis = velocity_axis
        self.doppler_strength = _finite_float(doppler_strength, name="doppler_strength")
        self.doppler_epsilon = _finite_float(doppler_epsilon, name="doppler_epsilon")
        if self.doppler_epsilon <= 0.0:
            raise ValueError("doppler_epsilon must be positive")
        if phases is None:
            self.phases = np.zeros(n, dtype=np.float64)
        else:
            self.phases = _validate_phases(phases, n=n)
        self.velocity_current = self._velocity_for_step(self.time)
        self._doppler_term = doppler_term(
            self.velocity_current,
            self.k_nm,
            doppler_strength=self.doppler_strength,
            doppler_epsilon=self.doppler_epsilon,
        )

    @property
    def doppler_term(self) -> FloatArray:
        """Most recently applied Doppler correction vector.

        Returns
        -------
        FloatArray
            Most recently applied Doppler correction vector.
        """
        return self._doppler_term.copy()

    def _velocity_for_step(self, t: float) -> FloatArray:
        """Return the source velocity active at a given step."""
        source = self._velocity_source
        raw = source(t) if callable(source) else source
        return scalarise_velocities(raw, n=self._n, velocity_axis=self._velocity_axis)

    def _omega_at(self, source: object, t: float) -> FloatArray:
        """Return the Doppler-shifted natural frequencies at a step."""
        raw = source(t) if callable(source) else source
        omega = _reject_non_real_array(raw, name="omegas")
        if omega.shape != (self._n,):
            raise ValueError("omegas shape must be (n,)")
        return omega

    def _omega_schedule(self, source: object, n_steps: int) -> FloatArray:
        """Return the per-step natural-frequency schedule."""
        return np.vstack(
            [
                self._omega_at(source, self._time + step * self._dt)
                for step in range(n_steps)
            ]
        )

    def _velocity_schedule(self, n_steps: int) -> FloatArray:
        """Return the per-step source-velocity schedule."""
        return np.vstack(
            [
                self._velocity_for_step(self._time + step * self._dt)
                for step in range(n_steps)
            ]
        )

    def step(
        self,
        phases: object | None = None,
        omegas: object | None = None,
        knm: object | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: object | None = None,
    ) -> FloatArray:
        """Advance one Doppler-corrected UPDE step.

        Parameters
        ----------
        phases : object | None
            Oscillator phases in radians, shape ``(N,)``.
        omegas : object | None
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : object | None
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        alpha : object | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.

        Returns
        -------
        FloatArray
            The phases after one Doppler-corrected UPDE step.
        """
        p = self.phases if phases is None else _validate_phases(phases, n=self._n)
        k = self.k_nm if knm is None else _validate_knm(knm, n=self._n)
        a = self.alpha_matrix if alpha is None else _validate_alpha(alpha, n=self._n)
        omega_source = self._omega_source if omegas is None else omegas
        omega = self._omega_at(omega_source, self._time)
        self.velocity_current = self._velocity_for_step(self._time)
        self._doppler_term = doppler_term(
            self.velocity_current,
            k,
            doppler_strength=self.doppler_strength,
            doppler_epsilon=self.doppler_epsilon,
        )
        out = super().step(p, omega + self._doppler_term, k, zeta, psi, a)
        self.phases = np.ascontiguousarray(out, dtype=np.float64)
        return self.phases.copy()

    def run(
        self,
        phases: object | None = None,
        omegas: object | None = None,
        knm: object | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: object | None = None,
        n_steps: int = 1,
    ) -> FloatArray:
        """Run ``n_steps`` of the Doppler-corrected UPDE dynamics.

        Parameters
        ----------
        phases : object | None
            Oscillator phases in radians, shape ``(N,)``.
        omegas : object | None
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : object | None
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        alpha : object | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        FloatArray
            The final phases after ``n_steps`` Doppler-corrected steps.
        """
        steps = _validate_positive_step_count(n_steps, name="n_steps")
        p = self.phases if phases is None else _validate_phases(phases, n=self._n)
        k = self.k_nm if knm is None else _validate_knm(knm, n=self._n)
        a = self.alpha_matrix if alpha is None else _validate_alpha(alpha, n=self._n)
        omega_source = self._omega_source if omegas is None else omegas
        omega_schedule = self._omega_schedule(omega_source, steps)
        velocity_schedule = self._velocity_schedule(steps)
        out = doppler_run(
            p,
            omega_schedule,
            k,
            a,
            velocity_schedule,
            self.doppler_strength,
            self.doppler_epsilon,
            zeta,
            psi,
            self._dt,
            self._method,
            1,
            self._atol,
            self._rtol,
        )
        _effective, terms = _effective_omega_schedule(
            omega_schedule,
            velocity_schedule,
            k,
            doppler_strength=self.doppler_strength,
            doppler_epsilon=self.doppler_epsilon,
        )
        self._omega_current = omega_schedule[-1].copy()
        self.velocity_current = velocity_schedule[-1]
        self._doppler_term = terms[-1]
        self._time += steps * self._dt
        self.phases = np.ascontiguousarray(out, dtype=np.float64)
        return self.phases.copy()
