# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial (higher-order) Kuramoto coupling

"""Pairwise + all-to-all 3-body (simplicial) Kuramoto with a
5-backend fallback chain per ``feedback_module_standard_attnres.md``.

Model
-----
    dθ_i/dt = ω_i
              + (σ₁/N) · Σ_j A_ij · sin(θ_j − θ_i)
              + (σ₂/N²) · Σ_{j,k} sin(θ_j + θ_k − 2θ_i)
              + ζ · sin(ψ − θ_i)

σ₂ > 0 drives **explosive (first-order) transitions** and shrinks
basins of attraction while improving the locking stability of
already-synchronous states (Gambuzza et al. 2023; Tang et al. 2025).

Closed form for the 3-body sum
------------------------------
Expanding ``sin(θ_j + θ_k − 2θ_i) = sin((θ_j − θ_i) + (θ_k − θ_i))``
and separating the cross terms gives

    Σ_{j,k} sin(θ_j + θ_k − 2θ_i) = 2 · S_i · C_i

with

    S_i = Σ_j sin(θ_j − θ_i) = (Σ sin θ)·cos θ_i − (Σ cos θ)·sin θ_i
    C_i = Σ_j cos(θ_j − θ_i) = (Σ cos θ)·cos θ_i + (Σ sin θ)·sin θ_i

So the 3-body contribution is evaluated in **O(N²)** (not O(N³))
using two global sums plus the per-node sincos expansion. All five
backends use this identity; the pairwise path matches the Rust
kernel's sincos expansion on the alpha-zero branch and the direct
``sin(diff)`` form otherwise, giving bit-exact parity.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "SimplicialEngine",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import simplicial_run_rust

    def _rust(
        phases: NDArray,
        omegas: NDArray,
        knm_flat: NDArray,
        alpha_flat: NDArray,
        n: int,
        zeta: float,
        psi: float,
        sigma2: float,
        dt: float,
        n_steps: int,
    ) -> NDArray:
        return np.asarray(
            simplicial_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(zeta),
                float(psi),
                float(sigma2),
                float(dt),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return _rust


def _load_mojo_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._simplicial_mojo import (
        _ensure_exe,
        simplicial_run_mojo,
    )

    _ensure_exe()
    return simplicial_run_mojo


def _load_julia_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._simplicial_julia import (
        simplicial_run_julia,
    )

    return simplicial_run_julia


def _load_go_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._simplicial_go import (
        simplicial_run_go,
    )

    return simplicial_run_go


_LOADERS: dict[str, Callable[[], Callable[..., NDArray]]] = {
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


def _dispatch() -> Callable[..., NDArray] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _python_run(
    phases: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    """Python reference aligned to the Rust kernel exactly.

    Uses the sincos expansion for the pairwise alpha-zero fast
    path and the closed-form ``2·S_i·C_i`` identity (with global
    sin/cos sums) for the 3-body term.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    om = np.asarray(omegas, dtype=np.float64)
    knm = knm_flat.reshape(n, n)
    alpha = alpha_flat.reshape(n, n)
    alpha_zero = bool(np.all(alpha == 0.0))
    use3 = sigma2 != 0.0 and n >= 3
    inv_n2 = sigma2 / (n * n) if n > 0 else 0.0

    for _ in range(n_steps):
        s = np.sin(p)
        c = np.cos(p)
        if alpha_zero:
            sin_diff = (
                s[np.newaxis, :] * c[:, np.newaxis]
                - c[np.newaxis, :] * s[:, np.newaxis]
            )
            pairwise = np.sum(knm * sin_diff, axis=1)
        else:
            diff = p[np.newaxis, :] - p[:, np.newaxis] - alpha
            pairwise = np.sum(knm * np.sin(diff), axis=1)
        deriv = om + pairwise
        if use3:
            gs = float(np.sum(s))
            gc = float(np.sum(c))
            # S_i = gs·c_i − gc·s_i,  C_i = gc·c_i + gs·s_i
            S_i = gs * c - gc * s
            C_i = gc * c + gs * s
            deriv = deriv + 2.0 * S_i * C_i * inv_n2
        if zeta != 0.0:
            deriv = deriv + zeta * np.sin(psi - p)
        p = (p + dt * deriv) % TWO_PI
    return p


class SimplicialEngine:
    """Pairwise + simplicial (3-body, all-to-all) Kuramoto stepper
    with 5-backend dispatch.

    The engine's geometry is ``(n, dt, σ₂)``; the step itself is
    stateless: ``(phases, omegas, K, α, ζ, ψ) → new_phases``.
    """

    def __init__(self, n_oscillators: int, dt: float, sigma2: float = 0.0):
        if n_oscillators < 1:
            raise ValueError(f"n_oscillators must be >= 1, got {n_oscillators}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if sigma2 < 0.0:
            raise ValueError(f"sigma2 must be non-negative, got {sigma2}")
        self._n = n_oscillators
        self._dt = dt
        self._sigma2 = sigma2

    @property
    def sigma2(self) -> float:
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"sigma2 must be non-negative, got {value}")
        self._sigma2 = value

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        return self.run(phases, omegas, knm, zeta, psi, alpha, n_steps=1)

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
        n_steps: int,
    ) -> NDArray:
        knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
        alpha_flat = np.ascontiguousarray(alpha, dtype=np.float64).ravel()
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                knm_flat,
                alpha_flat,
                self._n,
                float(zeta),
                float(psi),
                float(self._sigma2),
                float(self._dt),
                int(n_steps),
            )
        return _python_run(
            np.ascontiguousarray(phases, dtype=np.float64),
            np.ascontiguousarray(omegas, dtype=np.float64),
            knm_flat,
            alpha_flat,
            self._n,
            float(zeta),
            float(psi),
            float(self._sigma2),
            float(self._dt),
            int(n_steps),
        )

    def order_parameter(self, phases: NDArray) -> float:
        """Standard Kuramoto R = |<exp(iθ)>|."""
        return float(np.abs(np.mean(np.exp(1j * phases))))
