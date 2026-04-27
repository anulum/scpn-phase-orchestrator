# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen mean-field reduction

"""Exact mean-field reduction for globally-coupled Kuramoto with
Lorentzian ``g(ω)``, exposed as a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Dynamics
--------
On the Ott-Antonsen manifold the full N-oscillator Kuramoto system
reduces to a single complex-scalar ODE:

    dz/dt = −(Δ + iω₀)·z + (K/2)·(z − |z|²·z)

with ``z = R·e^{iψ}`` the mean-field order parameter, ``Δ`` the
half-width of the Lorentzian ``g(ω)``, ``ω₀`` its centre, and
``K`` the coupling strength.

Steady-state ``R_ss = √(1 − 2Δ/K)`` for ``K > K_c = 2Δ`` and
``R_ss = 0`` below. Reference: Ott & Antonsen 2008, *Chaos*
18(3):037113.

Numerics
--------
``run(z0, n_steps)`` is the compute-kernel path: a tight RK4 loop
on the real/imaginary components of ``z``. This is dispatched
across Rust / Mojo / Julia / Go / Python with bit-exact parity
(scalar ODE, no reduction identities, no global sums — the only
differences between backends are the rounding order of the
``k1..k4`` accumulation, which matches exactly).

The scalar-output helpers ``K_c``, ``steady_state_R`` and
``predict_from_oscillators`` stay native Python + optional Rust —
they are O(1) arithmetic or O(N) percentile work and do not
benefit from multi-language chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        fit_lorentzian_rust as _rust_fit_lorentzian,
    )
    from spo_kernel import (
        steady_state_r_oa_rust as _rust_steady_state_r,
    )

    _HAS_RUST_SCALAR = True
except ImportError:
    _HAS_RUST_SCALAR = False

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "OAState",
    "OttAntonsenReduction",
]


@dataclass
class OAState:
    z: complex
    R: float
    psi: float
    K_c: float


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., tuple[float, float, float, float]]:
    from spo_kernel import oa_run_rust

    def _rust(
        z_re: float,
        z_im: float,
        omega_0: float,
        delta: float,
        k_coupling: float,
        dt: float,
        n_steps: int,
    ) -> tuple[float, float, float, float]:
        re, im, r, psi = oa_run_rust(
            float(z_re),
            float(z_im),
            float(omega_0),
            float(delta),
            float(k_coupling),
            float(dt),
            int(n_steps),
        )
        return float(re), float(im), float(r), float(psi)

    return _rust


def _load_mojo_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._reduction_mojo import (
        _ensure_exe,
        oa_run_mojo,
    )

    _ensure_exe()
    return oa_run_mojo


def _load_julia_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._reduction_julia import oa_run_julia

    return oa_run_julia


def _load_go_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._reduction_go import oa_run_go

    return oa_run_go


_LOADERS: dict[str, Callable[[], Callable[..., tuple[float, float, float, float]]]] = {
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


def _dispatch() -> Callable[..., tuple[float, float, float, float]] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _oa_deriv(
    re: float,
    im: float,
    omega_0: float,
    delta: float,
    half_k: float,
) -> tuple[float, float]:
    abs_sq = re * re + im * im
    lin_re = -delta * re + omega_0 * im
    lin_im = -delta * im - omega_0 * re
    cubic_factor = half_k * (1.0 - abs_sq)
    return (lin_re + cubic_factor * re, lin_im + cubic_factor * im)


def _python_oa_run(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    """Python reference matching the Rust kernel exactly.

    Uses scalar-by-scalar RK4 on (re, im) with the same operation
    order and ``half_k = K/2`` factoring as
    ``spo-engine/src/reduction.rs``.
    """
    re, im = z_re, z_im
    half_k = k_coupling / 2.0
    for _ in range(n_steps):
        k1r, k1i = _oa_deriv(re, im, omega_0, delta, half_k)
        k2r, k2i = _oa_deriv(
            re + 0.5 * dt * k1r,
            im + 0.5 * dt * k1i,
            omega_0,
            delta,
            half_k,
        )
        k3r, k3i = _oa_deriv(
            re + 0.5 * dt * k2r,
            im + 0.5 * dt * k2i,
            omega_0,
            delta,
            half_k,
        )
        k4r, k4i = _oa_deriv(
            re + dt * k3r,
            im + dt * k3i,
            omega_0,
            delta,
            half_k,
        )
        re += (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        im += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
    r = (re * re + im * im) ** 0.5
    import math

    psi = math.atan2(im, re)
    return re, im, r, psi


class OttAntonsenReduction:
    """Ott-Antonsen mean-field reduction for globally-coupled
    Kuramoto with Lorentzian ``g(ω)`` (Ott-Antonsen 2008).

    The class stores ``(ω₀, Δ, K, dt)`` and exposes ``K_c``,
    ``steady_state_R()``, ``step(z)``, ``run(z0, n_steps)`` and
    ``predict_from_oscillators(omegas, K)``. ``run`` is dispatched
    across the 5-backend chain; the scalar helpers stay Python +
    optional Rust.
    """

    def __init__(
        self,
        omega_0: float,
        delta: float,
        K: float,
        dt: float = 0.01,
    ):
        if delta < 0:
            raise ValueError(f"delta (half-width) must be non-negative, got {delta}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._omega_0 = omega_0
        self._delta = delta
        self._K = K
        self._dt = dt

    @property
    def K_c(self) -> float:
        """Critical coupling ``K_c = 2Δ``."""
        return 2.0 * self._delta

    def steady_state_R(self) -> float:
        """Analytical steady-state ``R_ss = √(1 − 2Δ/K)`` for
        ``K > K_c``, zero otherwise."""
        if _HAS_RUST_SCALAR:
            return float(_rust_steady_state_r(self._delta, self._K))
        if self.K_c >= self._K:
            return 0.0
        return float((1.0 - 2.0 * self._delta / self._K) ** 0.5)

    def step(self, z: complex) -> complex:
        """Single RK4 step on the OA ODE."""
        re, im, _, _ = self._run_scalar(z.real, z.imag, n_steps=1)
        return complex(re, im)

    def run(self, z0: complex, n_steps: int) -> OAState:
        """Integrate ``n_steps`` RK4 steps; return final
        ``OAState(z, R, ψ, K_c)`` via the dispatched kernel."""
        re, im, r, psi = self._run_scalar(z0.real, z0.imag, n_steps)
        return OAState(z=complex(re, im), R=r, psi=psi, K_c=self.K_c)

    def _run_scalar(
        self,
        z_re: float,
        z_im: float,
        n_steps: int,
    ) -> tuple[float, float, float, float]:
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                z_re,
                z_im,
                self._omega_0,
                self._delta,
                self._K,
                self._dt,
                int(n_steps),
            )
        return _python_oa_run(
            z_re,
            z_im,
            self._omega_0,
            self._delta,
            self._K,
            self._dt,
            int(n_steps),
        )

    def predict_from_oscillators(
        self,
        omegas: NDArray,
        K: float,
    ) -> OAState:
        """Fit Lorentzian to ``omegas`` (median → ω₀,
        IQR/2 → Δ), run OA reduction ~10 time units from a
        small seed, and return the final ``OAState``."""
        if _HAS_RUST_SCALAR:
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            omega_0, delta = _rust_fit_lorentzian(o)
        else:
            omega_0 = float(np.median(omegas))
            q75, q25 = np.percentile(omegas, [75, 25])
            delta = (q75 - q25) / 2.0 if q75 > q25 else 0.01
        reducer = OttAntonsenReduction(omega_0, delta, K, dt=self._dt)
        return reducer.run(complex(0.01, 0.0), n_steps=int(10.0 / self._dt))
