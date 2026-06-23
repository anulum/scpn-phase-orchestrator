# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — moving-frame UPDE engine

"""Moving-frame Kuramoto UPDE integration.

``MovingFrameUPDEEngine`` carries one absolute axial coordinate per oscillator
alongside phase. Each outer step evaluates distance-dependent coupling from the
current positions, applies graph-weighted Doppler detuning from the current
velocities, advances phases, then advances positions ballistically over the
same chamber-clock step.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spatial_modulator import SpatialCouplingModulator
from scpn_phase_orchestrator.upde._ref_kernel import upde_run_omega_schedule_python
from scpn_phase_orchestrator.upde.doppler import (
    DopplerEngine,
    _finite_float,
    _reject_non_real_array,
    _validate_alpha,
    _validate_knm,
    _validate_method,
    _validate_phases,
    _validate_positive_step_count,
    _validate_schedule,
    doppler_term,
)

__all__ = [
    "KINEMATIC_RESIDUAL_TOLERANCE_M",
    "KINEMATIC_SUMMARY_REPLAY_TOLERANCE",
    "MovingFrameState",
    "MovingFrameUPDEEngine",
    "moving_frame_run",
    "moving_frame_run_python",
    "validate_moving_frame_backend_inputs",
]

FloatArray: TypeAlias = NDArray[np.float64]
BackendFn: TypeAlias = Callable[..., FloatArray]

_TWO_PI = 2.0 * np.pi
KINEMATIC_RESIDUAL_TOLERANCE_M = 1.0e-9
KINEMATIC_SUMMARY_REPLAY_TOLERANCE = 1.0e-12
_BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
_DECAY_TO_CODE = {
    "inverse_plus_one": 0,
    "exponential": 1,
    "power_law": 2,
    "inverse_distance": 3,
}
_CODE_TO_DECAY = {value: key for key, value in _DECAY_TO_CODE.items()}


@dataclass(frozen=True)
class MovingFrameState:
    """Snapshot of a moving-frame UPDE step or run."""

    phases: FloatArray
    positions: FloatArray
    velocities: FloatArray
    knm_effective: FloatArray
    doppler_term: FloatArray
    time: float
    kinematic_residual_max_m: float = 0.0
    max_abs_velocity_m_per_s: float = 0.0
    path_length_max_m: float = 0.0


def _validate_positions_vector(value: object, *, n: int) -> FloatArray:
    """Return the positions as a validated 1-D finite array, else raise."""
    arr = _reject_non_real_array(value, name="positions")
    if arr.shape != (n,):
        raise ValueError("positions shape must be (n,)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_spatial_modulator(value: object) -> SpatialCouplingModulator:
    """Return the validated spatial-modulator configuration, else raise."""
    if not isinstance(value, SpatialCouplingModulator):
        raise ValueError("spatial_modulator must be a SpatialCouplingModulator")
    return value


def _validate_decay_code(value: object) -> int:
    """Return the supported spatial-decay code, else raise."""
    if isinstance(value, str):
        if value not in _DECAY_TO_CODE:
            valid = ", ".join(sorted(_DECAY_TO_CODE))
            raise ValueError(f"spatial_decay_form must be one of: {valid}")
        return _DECAY_TO_CODE[value]
    if isinstance(value, bool):
        raise ValueError("spatial_decay_form must be a decay name or code")
    if not isinstance(value, (int, np.integer)):
        raise ValueError("spatial_decay_form code must be 0, 1, 2, or 3")
    code = int(value)
    if code != value or code not in _CODE_TO_DECAY:
        raise ValueError("spatial_decay_form code must be 0, 1, 2, or 3")
    return code


def _validate_nonnegative_float(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite float, else raise."""
    out = _finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    out = _finite_float(value, name=name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _spatial_weight(
    distance: FloatArray,
    *,
    k_base: float,
    decay_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    """Return the distance-based spatial coupling weight for a decay code."""
    if decay_code == 0:
        return k_base / (1.0 + distance)
    if decay_code == 1:
        return k_base * np.exp(-distance / decay_length_scale)
    if decay_code == 2:
        return k_base / (1.0 + distance / decay_length_scale) ** decay_exponent
    return k_base / np.sqrt(distance * distance + epsilon)


def _axial_spatial_modulate(
    knm: FloatArray,
    positions: FloatArray,
    *,
    k_base: float,
    decay_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    """Return the coupling modulated along the axial spatial profile."""
    distances = np.abs(positions[:, None] - positions[None, :])
    weights = _spatial_weight(
        distances,
        k_base=k_base,
        decay_code=decay_code,
        decay_exponent=decay_exponent,
        decay_length_scale=decay_length_scale,
        epsilon=epsilon,
    )
    out = np.ascontiguousarray(knm * weights, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    return out


def validate_moving_frame_backend_inputs(
    phases: object,
    positions: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    spatial_k_base: float,
    spatial_decay_form: object,
    spatial_decay_exponent: float,
    spatial_decay_length_scale: float,
    spatial_epsilon: float,
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
    FloatArray,
    float,
    int,
    float,
    float,
    float,
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
    """Validate the backend-neutral moving-frame schedule contract.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    positions : object
        Absolute axial coordinates per oscillator, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    spatial_k_base : float
        Base coupling strength before spatial modulation.
    spatial_decay_form : object
        Spatial decay law name (e.g. ``exponential`` or ``power``).
    spatial_decay_exponent : float
        Exponent of the spatial decay law.
    spatial_decay_length_scale : float
        Characteristic length scale of the spatial decay.
    spatial_epsilon : float
        Numerical floor guarding the spatial-decay denominator.
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
    tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float,
    int, float, float, float, float, float, float, float, float, int, str, int, float,
    float]
        The validated, canonicalised moving-frame schedule contract tuple.

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
    z = _validate_positions_vector(positions, n=n)
    omega = _validate_schedule(omega_schedule, n=n, name="omega_schedule")
    velocities = _validate_schedule(velocity_schedule, n=n, name="velocity_schedule")
    if velocities.shape[0] != omega.shape[0]:
        raise ValueError("velocity_schedule step count must match omega_schedule")
    k = _validate_knm(knm, n=n)
    a = _validate_alpha(alpha, n=n)
    k_base = _validate_nonnegative_float(spatial_k_base, name="spatial_k_base")
    decay_code = _validate_decay_code(spatial_decay_form)
    decay_exponent = _validate_positive_float(
        spatial_decay_exponent, name="spatial_decay_exponent"
    )
    decay_length_scale = _validate_positive_float(
        spatial_decay_length_scale, name="spatial_decay_length_scale"
    )
    spatial_eps = _validate_positive_float(spatial_epsilon, name="spatial_epsilon")
    strength = _finite_float(doppler_strength, name="doppler_strength")
    doppler_eps = _validate_positive_float(doppler_epsilon, name="doppler_epsilon")
    zeta_f = _finite_float(zeta, name="zeta")
    psi_f = _finite_float(psi, name="psi")
    dt_f = _validate_positive_float(dt, name="dt")
    method_s = _validate_method(method)
    n_substeps_i = _validate_positive_step_count(n_substeps, name="n_substeps")
    atol_f = _validate_positive_float(atol, name="atol")
    rtol_f = _validate_positive_float(rtol, name="rtol")
    return (
        p,
        z,
        omega,
        k,
        a,
        velocities,
        k_base,
        decay_code,
        decay_exponent,
        decay_length_scale,
        spatial_eps,
        strength,
        doppler_eps,
        zeta_f,
        psi_f,
        dt_f,
        int(omega.shape[0]),
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    )


def _expected_positions_from_schedule(
    positions: FloatArray,
    velocities: FloatArray,
    dt: float,
) -> FloatArray:
    """Return the expected oscillator positions from the motion schedule."""
    return np.ascontiguousarray(
        positions + dt * np.sum(velocities, axis=0),
        dtype=np.float64,
    )


def _kinematic_residual_max(
    observed_positions: FloatArray,
    expected_positions: FloatArray,
) -> float:
    """Return the maximum kinematic residual against expected positions."""
    return float(np.max(np.abs(observed_positions - expected_positions)))


def _validate_backend_output(
    value: object,
    *,
    n: int,
    expected_positions: FloatArray | None = None,
) -> FloatArray:
    """Return the backend output matching the reference, else raise."""
    out = _reject_non_real_array(value, name="moving_frame_backend_output")
    if out.shape != (2 * n,):
        raise ValueError("moving-frame backend output shape must be (2*n,)")
    phases = out[:n]
    positions = out[n:]
    if np.any(phases < 0.0) or np.any(phases >= _TWO_PI):
        raise ValueError("moving-frame backend phases must be in [0, 2*pi)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("moving-frame backend positions must be finite")
    if expected_positions is not None:
        expected = np.ascontiguousarray(expected_positions, dtype=np.float64)
        if expected.shape != positions.shape or not np.all(np.isfinite(expected)):
            raise ValueError("expected_positions must match finite backend positions")
        residual = _kinematic_residual_max(positions, expected)
        if residual > KINEMATIC_RESIDUAL_TOLERANCE_M:
            raise ValueError(
                "moving-frame backend positions violate ballistic kinematics: "
                f"max residual {residual} m exceeds "
                f"{KINEMATIC_RESIDUAL_TOLERANCE_M} m"
            )
    return np.ascontiguousarray(out, dtype=np.float64)


def moving_frame_run_python(
    phases: object,
    positions: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    spatial_k_base: float,
    spatial_decay_form: object,
    spatial_decay_exponent: float,
    spatial_decay_length_scale: float,
    spatial_epsilon: float,
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
    """Run the moving-frame UPDE schedule in the Python reference path.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    positions : object
        Absolute axial coordinates per oscillator, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    spatial_k_base : float
        Base coupling strength before spatial modulation.
    spatial_decay_form : object
        Spatial decay law name (e.g. ``exponential`` or ``power``).
    spatial_decay_exponent : float
        Exponent of the spatial decay law.
    spatial_decay_length_scale : float
        Characteristic length scale of the spatial decay.
    spatial_epsilon : float
        Numerical floor guarding the spatial-decay denominator.
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
        The final phases after running the moving-frame schedule on the Python path.
    """
    (
        p,
        z,
        omega,
        k,
        a,
        velocities,
        k_base,
        decay_code,
        decay_exponent,
        decay_length_scale,
        spatial_eps,
        strength,
        doppler_eps,
        zeta_f,
        psi_f,
        dt_f,
        n_steps,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_moving_frame_backend_inputs(
        phases,
        positions,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        spatial_k_base,
        spatial_decay_form,
        spatial_decay_exponent,
        spatial_decay_length_scale,
        spatial_epsilon,
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
    p_work = p.copy()
    z_work = z.copy()
    for step in range(n_steps):
        k_effective = _axial_spatial_modulate(
            k,
            z_work,
            k_base=k_base,
            decay_code=decay_code,
            decay_exponent=decay_exponent,
            decay_length_scale=decay_length_scale,
            epsilon=spatial_eps,
        )
        correction = doppler_term(
            velocities[step],
            k_effective,
            doppler_strength=strength,
            doppler_epsilon=doppler_eps,
        )
        p_work = upde_run_omega_schedule_python(
            p_work,
            (omega[step] + correction).reshape(1, -1),
            k_effective,
            a,
            zeta_f,
            psi_f,
            dt_f,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        )
        z_work = np.ascontiguousarray(
            z_work + velocities[step] * dt_f, dtype=np.float64
        )
    return np.ascontiguousarray(np.concatenate([p_work, z_work]), dtype=np.float64)


def _rust_backend() -> BackendFn:
    """Load the Rust moving-frame backend callable."""
    from spo_kernel import PyUPDEStepper

    if not hasattr(PyUPDEStepper, "run_moving_frame_schedule"):
        raise ImportError("PyUPDEStepper.run_moving_frame_schedule is unavailable")

    def run(
        phases: FloatArray,
        positions: FloatArray,
        omega_schedule: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        velocity_schedule: FloatArray,
        spatial_k_base: float,
        spatial_decay_form: int,
        spatial_decay_exponent: float,
        spatial_decay_length_scale: float,
        spatial_epsilon: float,
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
        """Call the Rust moving-frame schedule kernel."""
        stepper = PyUPDEStepper(
            int(phases.size), dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        return np.asarray(
            stepper.run_moving_frame_schedule(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(positions.ravel(), dtype=np.float64),
                np.ascontiguousarray(omega_schedule.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                np.ascontiguousarray(velocity_schedule.ravel(), dtype=np.float64),
                float(spatial_k_base),
                int(spatial_decay_form),
                float(spatial_decay_exponent),
                float(spatial_decay_length_scale),
                float(spatial_epsilon),
                float(doppler_strength),
                float(doppler_epsilon),
                int(omega_schedule.shape[0]),
            ),
            dtype=np.float64,
        )

    return run


def _backend_map() -> dict[str, BackendFn]:
    """Return the mapping of backend names to their loaders."""
    backends: dict[str, BackendFn] = {"python": moving_frame_run_python}
    with suppress(ImportError):
        go_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_go"
        )
        backends["go"] = go_mod.moving_frame_run_go
    with suppress(ImportError):
        julia_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_julia"
        )
        backends["julia"] = julia_mod.moving_frame_run_julia
    with suppress(ImportError):
        mojo_mod = importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._moving_frame_mojo"
        )
        backends["mojo"] = mojo_mod.moving_frame_run_mojo
    with suppress(ImportError):
        backends["rust"] = _rust_backend()
    return backends


def moving_frame_run(
    phases: object,
    positions: object,
    omega_schedule: object,
    knm: object,
    alpha: object,
    velocity_schedule: object,
    spatial_modulator: SpatialCouplingModulator,
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
    """Run a moving-frame UPDE schedule through the selected backend.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    positions : object
        Absolute axial coordinates per oscillator, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    spatial_modulator : SpatialCouplingModulator
        Configured spatial coupling modulator.
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
        The final phases after running the moving-frame schedule on the selected
        backend.

    Raises
    ------
    ImportError
        If the selected backend's runtime is unavailable.
    ValueError
        If the schedule contract is invalid; the underlying backend error propagates
        when every backend fails.
    """
    modulator = _validate_spatial_modulator(spatial_modulator)
    if modulator.distance_fn is not None:
        raise ValueError(
            "moving_frame_run requires the default axial SpatialCouplingModulator "
            "distance kernel"
        )
    validated = validate_moving_frame_backend_inputs(
        phases,
        positions,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        modulator.K_base,
        modulator.decay_form,
        modulator.decay_exponent,
        modulator.decay_length_scale,
        modulator.epsilon,
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
        z,
        omega,
        k,
        a,
        velocities,
        k_base,
        decay_code,
        decay_exponent,
        decay_length_scale,
        spatial_eps,
        strength,
        doppler_eps,
        zeta_f,
        psi_f,
        dt_f,
        _n_steps,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validated
    expected_positions = _expected_positions_from_schedule(z, velocities, dt_f)
    backends = _backend_map()
    if backend != "auto" and backend not in backends:
        raise ImportError(f"moving-frame backend {backend!r} is not available")
    order = _BACKEND_ORDER if backend == "auto" else (backend,)
    last_error: Exception | None = None
    for name in order:
        fn = backends.get(name)
        if fn is None:
            continue
        try:
            out = fn(
                p,
                z,
                omega,
                k,
                a,
                velocities,
                k_base,
                decay_code,
                decay_exponent,
                decay_length_scale,
                spatial_eps,
                strength,
                doppler_eps,
                zeta_f,
                psi_f,
                dt_f,
                method_s,
                n_substeps_i,
                atol_f,
                rtol_f,
            )
            return _validate_backend_output(
                out,
                n=int(p.size),
                expected_positions=expected_positions,
            )
        except (AttributeError, ImportError) as exc:
            last_error = exc
            continue
    if backend != "auto" and last_error is not None:
        raise last_error
    return _validate_backend_output(
        moving_frame_run_python(
            p,
            z,
            omega,
            k,
            a,
            velocities,
            k_base,
            decay_code,
            decay_exponent,
            decay_length_scale,
            spatial_eps,
            strength,
            doppler_eps,
            zeta_f,
            psi_f,
            dt_f,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        ),
        n=int(p.size),
        expected_positions=expected_positions,
    )


class MovingFrameUPDEEngine(DopplerEngine):
    """UPDE engine with chamber-frame axial positions and collision checks."""

    def __init__(
        self,
        n: int,
        omega: object,
        k_nm: object,
        alpha: object = 0.0,
        dt: float = 0.01,
        positions_t0: object | None = None,
        velocities: object | Callable[[float], object] | None = None,
        spatial_modulator: SpatialCouplingModulator | None = None,
        reference_point: float = 0.0,
        doppler_strength: float = 1.0,
        doppler_epsilon: float = 1.0e-9,
        solver: str = "rk45",
        phases: object | None = None,
        velocity_axis: object | None = None,
        t0: float = 0.0,
    ) -> None:
        if positions_t0 is None:
            raise ValueError("positions_t0 is required for MovingFrameUPDEEngine")
        if spatial_modulator is None:
            raise ValueError("spatial_modulator is required for MovingFrameUPDEEngine")
        self.spatial_modulator = _validate_spatial_modulator(spatial_modulator)
        self.reference_point = _finite_float(reference_point, name="reference_point")
        super().__init__(
            n,
            omega=omega,
            k_nm=k_nm,
            alpha=alpha,
            dt=dt,
            velocities=velocities,
            doppler_strength=doppler_strength,
            doppler_epsilon=doppler_epsilon,
            solver=solver,
            phases=phases,
            velocity_axis=velocity_axis,
            t0=t0,
        )
        self._positions = _validate_positions_vector(positions_t0, n=n)
        self._knm_effective = self._modulated_knm(self.k_nm, self._positions)
        self._doppler_term = doppler_term(
            self.velocity_current,
            self._knm_effective,
            doppler_strength=self.doppler_strength,
            doppler_epsilon=self.doppler_epsilon,
        )
        self._kinematic_residual_max_m = 0.0
        self._max_abs_velocity_m_per_s = float(np.max(np.abs(self.velocity_current)))
        self._path_length_max_m = 0.0

    @property
    def positions(self) -> FloatArray:
        """Current absolute axial coordinate for each oscillator.

        Returns
        -------
        FloatArray
            Current absolute axial coordinate for each oscillator.
        """
        return self._positions.copy()

    @property
    def distance_to_reference(self) -> FloatArray:
        """Absolute distance from each oscillator to the chamber reference.

        Returns
        -------
        FloatArray
            Absolute distance from each oscillator to the chamber reference.
        """
        return np.ascontiguousarray(
            np.abs(self._positions - self.reference_point), dtype=np.float64
        )

    @property
    def knm_effective(self) -> FloatArray:
        """Most recently applied distance-modulated coupling matrix.

        Returns
        -------
        FloatArray
            Most recently applied distance-modulated coupling matrix.
        """
        return self._knm_effective.copy()

    @property
    def kinematic_residual_max_m(self) -> float:
        """Maximum residual against ``z_next = z + v*dt`` in the last run.

        Returns
        -------
        float
            Maximum residual against ``z_next = z + v*dt`` in the last run.
        """
        return float(self._kinematic_residual_max_m)

    @property
    def max_abs_velocity_m_per_s(self) -> float:
        """Maximum absolute axial velocity used by the last step or run.

        Returns
        -------
        float
            Maximum absolute axial velocity used by the last step or run.
        """
        return float(self._max_abs_velocity_m_per_s)

    @property
    def path_length_max_m(self) -> float:
        """Maximum per-oscillator axial path length in the last step or run.

        Returns
        -------
        float
            Maximum per-oscillator axial path length in the last step or run.
        """
        return float(self._path_length_max_m)

    @property
    def state(self) -> MovingFrameState:
        """Return the current moving-frame diagnostic snapshot.

        Returns
        -------
        MovingFrameState
            Return the current moving-frame diagnostic snapshot.
        """
        return MovingFrameState(
            phases=self.phases.copy(),
            positions=self._positions.copy(),
            velocities=self.velocity_current.copy(),
            knm_effective=self._knm_effective.copy(),
            doppler_term=self.doppler_term,
            time=float(self._time),
            kinematic_residual_max_m=self._kinematic_residual_max_m,
            max_abs_velocity_m_per_s=self._max_abs_velocity_m_per_s,
            path_length_max_m=self._path_length_max_m,
        )

    def _modulated_knm(self, k_nm: FloatArray, positions: FloatArray) -> FloatArray:
        """Return the spatially-modulated coupling matrix for a frame."""
        modulator = self.spatial_modulator
        if modulator.distance_fn is not None:
            return modulator.modulate(k_nm, positions.reshape(-1, 1))
        return _axial_spatial_modulate(
            k_nm,
            positions,
            k_base=float(modulator.K_base),
            decay_code=_DECAY_TO_CODE[str(modulator.decay_form)],
            decay_exponent=float(modulator.decay_exponent),
            decay_length_scale=float(modulator.decay_length_scale),
            epsilon=float(modulator.epsilon),
        )

    def collision_imminent(self, threshold_m: float = 1.0e-3) -> bool:
        """Return whether any oscillator is at or crosses the reference soon.

        Parameters
        ----------
        threshold_m : float
            Distance threshold in metres.

        Returns
        -------
        bool
            ``True`` when an oscillator is at or crossing the reference within one step.
        """
        threshold = _validate_nonnegative_float(threshold_m, name="threshold_m")
        signed_now = self._positions - self.reference_point
        signed_next = signed_now + self.velocity_current * self._dt
        current_near = np.abs(signed_now) <= threshold
        next_near = np.abs(signed_next) <= threshold
        crosses = signed_now * signed_next <= 0.0
        return bool(np.any(current_near | next_near | crosses))

    def step(
        self,
        phases: object | None = None,
        omegas: object | None = None,
        knm: object | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: object | None = None,
    ) -> FloatArray:
        """Advance one coupled phase/position step.

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
            The phases after one coupled phase/position step.
        """
        k_base = self.k_nm if knm is None else _validate_knm(knm, n=self._n)
        k_effective = self._modulated_knm(k_base, self._positions)
        positions_start = self._positions.copy()
        out = super().step(phases, omegas, k_effective, zeta, psi, alpha)
        self._knm_effective = k_effective
        self._positions = np.ascontiguousarray(
            positions_start + self.velocity_current * self._dt,
            dtype=np.float64,
        )
        expected_positions = positions_start + self.velocity_current * self._dt
        self._kinematic_residual_max_m = _kinematic_residual_max(
            self._positions,
            expected_positions,
        )
        self._max_abs_velocity_m_per_s = float(np.max(np.abs(self.velocity_current)))
        self._path_length_max_m = float(
            np.max(np.abs(self.velocity_current * self._dt))
        )
        return out

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
        """Run ``n_steps`` of joint phase and axial-position dynamics.

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
            The final phases after ``n_steps`` joint phase/position steps.
        """
        steps = _validate_positive_step_count(n_steps, name="n_steps")
        p = self.phases if phases is None else _validate_phases(phases, n=self._n)
        k = self.k_nm if knm is None else _validate_knm(knm, n=self._n)
        a = self.alpha_matrix if alpha is None else _validate_alpha(alpha, n=self._n)
        omega_source = self._omega_source if omegas is None else omegas
        omega_schedule = self._omega_schedule(omega_source, steps)
        velocity_schedule = self._velocity_schedule(steps)
        z0 = self._positions.copy()
        flat = moving_frame_run(
            p,
            z0,
            omega_schedule,
            k,
            a,
            velocity_schedule,
            self.spatial_modulator,
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
        n = self._n
        self.phases = np.ascontiguousarray(flat[:n], dtype=np.float64)
        self._positions = np.ascontiguousarray(flat[n:], dtype=np.float64)
        expected_positions = _expected_positions_from_schedule(
            z0,
            velocity_schedule,
            self._dt,
        )
        self._kinematic_residual_max_m = _kinematic_residual_max(
            self._positions,
            expected_positions,
        )
        self._max_abs_velocity_m_per_s = float(np.max(np.abs(velocity_schedule)))
        self._path_length_max_m = float(
            np.max(np.sum(np.abs(velocity_schedule * self._dt), axis=0))
        )
        if steps > 1:
            last_start_positions = z0 + self._dt * np.sum(
                velocity_schedule[:-1], axis=0
            )
        else:
            last_start_positions = z0
        self._knm_effective = self._modulated_knm(k, last_start_positions)
        self._omega_current = omega_schedule[-1].copy()
        self.velocity_current = velocity_schedule[-1].copy()
        self._doppler_term = doppler_term(
            self.velocity_current,
            self._knm_effective,
            doppler_strength=self.doppler_strength,
            doppler_epsilon=self.doppler_epsilon,
        )
        self._time += steps * self._dt
        return self.phases.copy()
