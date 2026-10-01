# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JAX-accelerated UPDE engine

"""JIT-compiled Kuramoto solver on the configured JAX device.

Raises ImportError if JAX is not installed. Check HAS_JAX before use.
Usage:
    from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX
    if HAS_JAX:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine
        engine = JaxUPDEEngine(n, dt=0.01)
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from numbers import Complex, Integral, Real
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["JaxUPDEEngine", "HAS_JAX"]

FloatArray: TypeAlias = NDArray[np.float64]
ResultArray: TypeAlias = NDArray[np.float32 | np.float64]

TWO_PI = 2.0 * np.pi

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import Array

KuramotoStep: TypeAlias = Callable[
    ["Array", "Array", "Array", float, float, "Array", float], "Array"
]
StuartLandauStep: TypeAlias = Callable[
    ["Array", "Array", "Array", "Array", "Array", float, float, "Array", float, float],
    "Array",
]


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return value


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return value


def _validate_array(value: object, *, name: str, shape: tuple[int, ...]) -> FloatArray:
    """Return the value as a validated finite array, else raise."""
    try:
        object_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        object_array = None
    if object_array is not None and any(
        isinstance(item, (bool, np.bool_)) for item in object_array.flat
    ):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc
    object_values = raw.dtype == np.object_
    if np.iscomplexobj(raw) or (
        object_values
        and any(
            isinstance(item, Complex) and not isinstance(item, Real)
            for item in raw.flat
        )
    ):
        raise ValueError(f"{name} must not contain complex values")
    numeric_object = object_values and all(isinstance(item, Real) for item in raw.flat)
    if not np.issubdtype(raw.dtype, np.number) and not numeric_object:
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_method(value: object) -> str:
    """Return the supported integration-method name, else raise."""
    if not isinstance(value, str) or value not in ("euler", "rk4"):
        raise ValueError(f"unsupported method {value!r}")
    return value


def _build_jax_step() -> tuple[KuramotoStep, KuramotoStep]:
    """Build JIT-compiled Kuramoto step function."""
    from scpn_phase_orchestrator.upde._jax_phase_wrap import wrap_phases

    @jit
    def _kuramoto_step(
        phases: Array,
        omegas: Array,
        knm: Array,
        zeta: float,
        psi: float,
        alpha: Array,
        dt: float,
    ) -> Array:
        """Advance the Kuramoto phases one explicit-Euler step (JAX)."""
        diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
        coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
        dphi = omegas + coupling
        dphi = dphi + zeta * jnp.sin(psi - phases)
        new_phases = phases + dt * dphi
        return wrap_phases(new_phases)

    @jit
    def _kuramoto_rk4(
        phases: Array,
        omegas: Array,
        knm: Array,
        zeta: float,
        psi: float,
        alpha: Array,
        dt: float,
    ) -> Array:
        """Advance the Kuramoto phases one RK4 step (JAX)."""

        def deriv(p: Array) -> Array:
            """Kuramoto coupling derivative at given phases."""
            diff = p[jnp.newaxis, :] - p[:, jnp.newaxis]
            coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
            return omegas + coupling + zeta * jnp.sin(psi - p)

        k1 = deriv(phases)
        k2 = deriv(phases + 0.5 * dt * k1)
        k3 = deriv(phases + 0.5 * dt * k2)
        k4 = deriv(phases + dt * k3)
        new_phases = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return wrap_phases(new_phases)

    return _kuramoto_step, _kuramoto_rk4


def _build_jax_sl_step() -> StuartLandauStep:
    """Build JIT-compiled Stuart-Landau step function."""

    @jit
    def _sl_rk4(
        state: Array,
        omegas: Array,
        mu: Array,
        knm: Array,
        knm_r: Array,
        zeta: float,
        psi: float,
        alpha: Array,
        epsilon: float,
        dt: float,
    ) -> Array:
        """Advance the Stuart-Landau state one RK4 step (JAX)."""
        n = omegas.shape[0]

        def deriv(s: Array) -> Array:
            """Stuart-Landau coupled (phase, amplitude) derivative."""
            th, am = s[:n], s[n:]
            diff = th[jnp.newaxis, :] - th[:, jnp.newaxis]
            phase_coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
            amp_coupling = jnp.sum(
                knm_r * jnp.maximum(am, 0.0)[jnp.newaxis, :] * jnp.cos(diff - alpha),
                axis=1,
            )
            dtheta = omegas + phase_coupling + zeta * jnp.sin(psi - th)
            dr = (mu - am * am) * am + epsilon * amp_coupling
            return jnp.concatenate([dtheta, dr])

        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        new_theta = new_state[:n] % (2.0 * jnp.pi)
        new_r = jnp.maximum(new_state[n:], 0.0)
        return jnp.concatenate([new_theta, new_r])

    return _sl_rk4


class JaxUPDEEngine:
    """JAX-accelerated Kuramoto/UPDE integrator.

    JIT execution uses the available JAX device and configured floating-point
    precision. The first call compiles its shape; later calls reuse that code.
    """

    def __init__(self, n: int, dt: float = 0.01, method: str = "rk4") -> None:
        """Configure the actual JAX device integrator.

        Parameters
        ----------
        n : int
            Positive oscillator count.
        dt : float, default 0.01
            Positive finite integration timestep in seconds.
        method : str, default "rk4"
            Euler or RK4; the first step compiles for the actual shape and dtype.

        Raises
        ------
        ImportError
            JAX is not installed.
        ValueError
            The oscillator count, timestep, or integration method is invalid.
        """
        if not HAS_JAX:
            msg = "JAX not installed. Install with: pip install jax jaxlib"
            raise ImportError(msg)
        self._n = _validate_positive_int(n, name="n")
        self._dt = _validate_positive_float(dt, name="dt")
        self._method = _validate_method(method)
        euler_fn, rk4_fn = _build_jax_step()
        self._euler = euler_fn
        self._rk4 = rk4_fn

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> ResultArray:
        """Advance phases by one Kuramoto step via JIT-compiled JAX.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        alpha : FloatArray
            Finite phase-lag matrix in radians, shape ``(N, N)``.

        Returns
        -------
        ResultArray
            Finite phases in the half-open torus, with canonical positive zero
            and float32 or float64 precision according to JAX configuration.

        Raises
        ------
        ValueError
            If inputs are invalid or the actual configured-precision computation
            produces nonfinite or out-of-domain phases.
        """
        phases = _validate_array(phases, name="phases", shape=(self._n,))
        omegas = _validate_array(omegas, name="omegas", shape=(self._n,))
        knm = _validate_array(knm, name="knm", shape=(self._n, self._n))
        alpha = _validate_array(alpha, name="alpha", shape=(self._n, self._n))
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")

        # A finite float64 input can overflow the actual float32 conversion.
        # Refuse the resulting nonfinite computation below instead of publishing it.
        with np.errstate(over="ignore", invalid="ignore"):
            jp = jnp.asarray(phases)
            jo = jnp.asarray(omegas)
            jk = jnp.asarray(knm)
            ja = jnp.asarray(alpha)
            if self._method == "rk4":
                result = self._rk4(jp, jo, jk, zeta, psi, ja, self._dt)
            else:
                result = self._euler(jp, jo, jk, zeta, psi, ja, self._dt)
            output: ResultArray = np.asarray(result)
        period = np.asarray(TWO_PI, dtype=output.dtype)
        if not np.all(np.isfinite(output)):
            raise ValueError("JAX output contains NaN/Inf")
        if np.any((output < 0.0) | (output >= period)):
            raise ValueError("JAX output phases must be in [0, 2*pi)")
        return output


class JaxStuartLandauEngine:
    """JAX-accelerated Stuart-Landau integrator (RK4 only)."""

    def __init__(self, n: int, dt: float = 0.01) -> None:
        """Configure the existing JAX Stuart-Landau RK4 integrator.

        Parameters
        ----------
        n : int
            Positive oscillator count; state has n phases then n amplitudes.
        dt : float, default 0.01
            Positive finite integration timestep in seconds.

        Raises
        ------
        ImportError
            JAX is not installed.
        ValueError
            The oscillator count or timestep is invalid.
        """
        if not HAS_JAX:
            msg = "JAX not installed. Install with: pip install jax jaxlib"
            raise ImportError(msg)
        self._n = _validate_positive_int(n, name="n")
        self._dt = _validate_positive_float(dt, name="dt")
        self._sl_rk4 = _build_jax_sl_step()

    def step(
        self,
        state: FloatArray,
        omegas: FloatArray,
        mu: FloatArray,
        knm: FloatArray,
        knm_r: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
        epsilon: float = 1.0,
    ) -> ResultArray:
        """Advance the existing Stuart-Landau state via actual JAX RK4.

        Parameters
        ----------
        state : FloatArray
            Shape (2*n,), with phases followed by amplitudes.
        omegas, mu : FloatArray
            Frequencies and amplitude growth parameters, each shape (n,).
        knm, knm_r : FloatArray
            Phase and amplitude coupling matrices, each shape (n, n).
        zeta, psi : float
            Finite external phase drive strength and target phase.
        alpha : FloatArray
            Finite phase-lag matrix, shape (n, n).
        epsilon : float, default 1.0
            Finite amplitude-coupling scale.

        Returns
        -------
        ResultArray
            State in JAX's configured float32 or float64 precision.

        Raises
        ------
        ValueError
            Supplied arrays or scalar inputs fail admission.

        Notes
        -----
        This existing model retains its own phase and amplitude equations;
        it uses neither the Kuramoto projection nor its output admission guard.
        """
        state = _validate_array(state, name="state", shape=(2 * self._n,))
        omegas = _validate_array(omegas, name="omegas", shape=(self._n,))
        mu = _validate_array(mu, name="mu", shape=(self._n,))
        knm = _validate_array(knm, name="knm", shape=(self._n, self._n))
        knm_r = _validate_array(knm_r, name="knm_r", shape=(self._n, self._n))
        alpha = _validate_array(alpha, name="alpha", shape=(self._n, self._n))
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        epsilon = _validate_finite_float(epsilon, name="epsilon")

        js = jnp.asarray(state)
        result = self._sl_rk4(
            js,
            jnp.asarray(omegas),
            jnp.asarray(mu),
            jnp.asarray(knm),
            jnp.asarray(knm_r),
            zeta,
            psi,
            jnp.asarray(alpha),
            epsilon,
            self._dt,
        )
        return np.asarray(result)
