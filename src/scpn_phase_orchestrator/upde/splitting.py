# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strang operator splitting integrator

"""Strang second-order operator splitting for the Kuramoto ODE.

Exposes a 5-backend fallback chain.

Scheme
------
Split ``dθ/dt = ω + Σ_j K_ij · sin(θ_j − θ_i − α_ij) +
ζ · sin(ψ − θ_i)`` into

    A: dθ/dt = ω              (exact rotation)
    B: dθ/dt = coupling       (RK4 on the nonlinear part)

and compose symmetrically as ``A(dt/2) → B(dt) → A(dt/2)``
(Strang scheme, second-order in dt).

Why split?
----------
The ``ω`` flow is linear, so it has no truncation error; folding
it into a monolithic RK45 burns integrator budget on a solvable
direction while damping unrelated accuracy in the nonlinear
direction. Reference: Hairer, Lubich & Wanner 2006, *Geometric
Numerical Integration* §II.5.

Numerics
--------
The B-stage RK4 uses the Rust kernel's
``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) − cos(θ_j)·sin(θ_i)``
expansion on the alpha-zero branch so that floating-point
rounding matches Rust (``spo-engine/src/splitting.rs``)
bit-for-bit. Nonzero alpha falls back to the direct
``sin(diff)`` form in all five backends.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from time import perf_counter
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "SplittingEngine",
]

FloatArray: TypeAlias = NDArray[np.float64]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., FloatArray]:
    """Load the Rust Strang-splitting backend callable."""
    from spo_kernel import splitting_run_rust

    def _rust(
        phases: FloatArray,
        omegas: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        n: int,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        # The Rust FFI reads N from ``phases.len()`` and ignores the
        # positional ``_n`` argument (hence the leading underscore in
        # its signature). We still pass ``n`` for future-proofing.
        """Call the Rust Strang-splitting step kernel."""
        return np.asarray(
            splitting_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(zeta),
                float(psi),
                float(dt),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return _rust


def _load_mojo_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Mojo Strang-splitting backend callable."""
    from ..experimental.accelerators.upde._splitting_mojo import (
        _ensure_exe,
        splitting_run_mojo,
    )

    _ensure_exe()
    return splitting_run_mojo


def _load_julia_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Julia Strang-splitting backend callable."""
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._splitting_julia import (
        splitting_run_julia,
    )

    return splitting_run_julia


def _load_go_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    """Load the Go Strang-splitting backend callable."""
    from ..experimental.accelerators.upde._splitting_go import (
        _load_lib,
        splitting_run_go,
    )

    _load_lib()
    return splitting_run_go


_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}


def _load_backend(name: str) -> Callable[..., FloatArray]:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    active = min(available, key=_splitting_probe_seconds)
    return active, available


def _splitting_probe_seconds(name: str) -> float:
    """Return the per-backend probe timings for the splitting step."""
    n = 64
    phases = np.linspace(0.0, TWO_PI, n, endpoint=False, dtype=np.float64)
    omegas = np.ones(n, dtype=np.float64)
    knm = np.full((n, n), 0.5, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    knm_flat = knm.ravel()
    alpha_flat = alpha.ravel()
    start = perf_counter()
    try:
        if name == "python":
            _python_run(phases, omegas, knm_flat, alpha_flat, n, 0.0, 0.0, 0.01, 1)
        else:
            _load_backend(name)(
                phases,
                omegas,
                knm_flat,
                alpha_flat,
                n,
                0.0,
                0.0,
                0.01,
                1,
            )
    except (ImportError, RuntimeError, OSError, KeyError):
        return float("inf")
    return perf_counter() - start


def _dispatch() -> Callable[..., FloatArray] | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
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
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_nonzero_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a non-zero finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-zero real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced == 0.0:
        raise ValueError(f"{name} must be a finite non-zero real, got {value!r}")
    return coerced


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the state as a validated finite array, else raise."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_backend_output(value: object, *, n: int) -> FloatArray:
    """Return the backend output matching the reference, else raise."""
    try:
        output = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend output must be a finite phase vector") from exc
    if output.shape != (n,):
        raise ValueError(f"backend output shape {output.shape} does not match {(n,)}")
    if not np.all(np.isfinite(output)):
        raise ValueError("backend output must contain only finite phases")
    if np.any(output < 0.0) or np.any(output >= TWO_PI):
        raise ValueError("backend output phases must be in [0, 2*pi)")
    return np.ascontiguousarray(output, dtype=np.float64)


def _coupling_deriv(
    theta: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    alpha_zero: bool,
) -> FloatArray:
    """Return the coupling half-step derivative for the phase state."""
    s = np.sin(theta)
    c = np.cos(theta)
    if alpha_zero:
        # sin(θ_j − θ_i) = s_j·c_i − c_j·s_i
        sin_diff = (
            s[np.newaxis, :] * c[:, np.newaxis] - c[np.newaxis, :] * s[:, np.newaxis]
        )
        out = np.sum(knm * sin_diff, axis=1)
    else:
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        out = np.sum(knm * np.sin(diff), axis=1)
    if zeta != 0.0:
        # ζ · sin(ψ − θ) = ζ·sin(ψ)·cos(θ) − ζ·cos(ψ)·sin(θ)
        out = out + zeta * np.sin(psi) * c - zeta * np.cos(psi) * s
    return cast("FloatArray", out)


def _rk4_coupling(
    p: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    alpha_zero: bool,
) -> FloatArray:
    """Advance the coupling half-step one RK4 step."""
    k1 = _coupling_deriv(p, knm, alpha, zeta, psi, alpha_zero)
    k2 = _coupling_deriv(
        (p + 0.5 * dt * k1) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    k3 = _coupling_deriv(
        (p + 0.5 * dt * k2) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    k4 = _coupling_deriv(
        (p + dt * k3) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    return (p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % TWO_PI


def _python_run(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Python reference aligned to the Rust kernel exactly.

    Uses the sincos expansion on the alpha-zero branch and the
    same ``ζ·sin(ψ−θ) = ζ·sin(ψ)·cos(θ) − ζ·cos(ψ)·sin(θ)``
    expansion as the Rust kernel, giving bit-exact parity.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    om = np.asarray(omegas, dtype=np.float64)
    knm = knm_flat.reshape(n, n)
    alpha = alpha_flat.reshape(n, n)
    alpha_zero = bool(np.all(alpha == 0.0))
    half_dt = 0.5 * dt
    for _ in range(n_steps):
        p = (p + half_dt * om) % TWO_PI
        p = _rk4_coupling(p, knm, alpha, zeta, psi, dt, alpha_zero)
        p = (p + half_dt * om) % TWO_PI
    return p


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


class SplittingEngine:
    """Strang-split Kuramoto stepper with 5-backend dispatch.

    The engine's geometry is ``(n, dt)``; the step is stateless.
    """

    def __init__(self, n_oscillators: int, dt: float):
        n_oscillators = _validate_positive_int(
            n_oscillators,
            name="n_oscillators",
        )
        dt = _validate_nonzero_finite_float(dt, name="dt")
        # Negative dt is intentional for symplectic-reversibility
        # checks — Strang is time-reversible, so stepping forward
        # then with dt → −dt returns to the starting state.
        self._n = n_oscillators
        self._dt = dt

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        """One Strang-split step: A(dt/2) → B(dt) → A(dt/2).

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
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.

        Returns
        -------
        FloatArray
            The phases after one Strang-split step.
        """
        return self.run(phases, omegas, knm, zeta, psi, alpha, n_steps=1)

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
        n_steps: int,
    ) -> FloatArray:
        """Apply repeated Strang-split phase integration steps.

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
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        FloatArray
            The final phases after ``n_steps`` Strang-split steps.

        Raises
        ------
        ValueError
            If ``n_steps`` is negative or the state arrays are invalid.
        """
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        if np.any(np.diag(knm64) != 0.0):
            raise ValueError("knm diagonal must be exactly zero")
        alpha64 = _validate_state_array(
            alpha,
            name="alpha",
            shape=(self._n, self._n),
        )
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        knm_flat = knm64.ravel()
        alpha_flat = alpha64.ravel()
        backend_fn = _dispatch() if self._dt > 0.0 else None
        # Non-Rust backends also validate dt > 0 internally, so
        # negative dt (symplectic-reversibility checks) falls back
        # to the Python reference.
        if backend_fn is not None:
            return _validate_backend_output(
                backend_fn(
                    phases64,
                    omegas64,
                    knm_flat,
                    alpha_flat,
                    self._n,
                    float(zeta),
                    float(psi),
                    float(self._dt),
                    int(n_steps),
                ),
                n=self._n,
            )
        return _python_run(
            phases64,
            omegas64,
            knm_flat,
            alpha_flat,
            self._n,
            float(zeta),
            float(psi),
            float(self._dt),
            int(n_steps),
        )

    def order_parameter(self, phases: FloatArray) -> float:
        """Compute the standard Kuramoto R = |<exp(iθ)>|.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.

        Returns
        -------
        float
            The Kuramoto order parameter ``R``.
        """
        phases64 = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        )
        return float(np.abs(np.mean(np.exp(1j * phases64))))
