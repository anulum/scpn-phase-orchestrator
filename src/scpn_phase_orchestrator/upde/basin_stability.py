# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability analysis

"""Basin stability for Kuramoto synchronisation with a 5-backend
fallback chain per ``feedback_module_standard_attnres.md``.

Monte Carlo estimation of the volume of the basin of attraction for
the synchronised state. Basin stability ``S_B`` is the probability
that a random initial condition converges to the synchronised
attractor (Menck et al. 2013, Ji et al. 2014).

Kernel of the computation
-------------------------
The single-trial primitive is ``steady_state_r(phases_init, omegas,
knm, alpha, dt, n_transient, n_measure) → R`` — explicit Euler
integration of the Kuramoto ODE, transient discarded, time-averaged
order parameter returned. The trial kernel has no RNG and is
dispatched across Rust / Mojo / Julia / Go / Python (bit-exact parity
on deterministic inputs).

RNG ownership
-------------
The Monte Carlo loop lives in Python: ``np.random.default_rng(seed)``
draws ``n_samples`` random phase vectors from ``[0, 2π)^N`` and calls
the dispatched trial kernel once per IC. This is the ``dimension``
pattern — Python owns the randomness so the compute primitive stays
deterministic and parity-testable. The original
``basin_stability_rust`` (seed-in → S_B-out) kernel is preserved as a
one-shot fast path when all four arguments match, but regular use
goes through the dispatched per-trial kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "BasinStabilityResult",
    "basin_stability",
    "multi_basin_stability",
    "steady_state_r",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., float]:
    from spo_kernel import steady_state_r_rust

    def _rust(
        phases_init: NDArray,
        omegas: NDArray,
        knm_flat: NDArray,
        alpha_flat: NDArray,
        n: int,
        k_scale: float,
        dt: float,
        n_transient: int,
        n_measure: int,
    ) -> float:
        return float(
            steady_state_r_rust(
                np.ascontiguousarray(phases_init, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(k_scale),
                float(dt),
                int(n_transient),
                int(n_measure),
            )
        )

    return _rust


def _load_mojo_fn() -> Callable[..., float]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._basin_stability_mojo import (
        _ensure_exe,
        steady_state_r_mojo,
    )

    _ensure_exe()
    return steady_state_r_mojo


def _load_julia_fn() -> Callable[..., float]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.upde._basin_stability_julia import (
        steady_state_r_julia,
    )

    return steady_state_r_julia


def _load_go_fn() -> Callable[..., float]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._basin_stability_go import (
        steady_state_r_go,
    )

    return steady_state_r_go


_LOADERS: dict[str, Callable[[], Callable[..., float]]] = {
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


def _dispatch() -> Callable[..., float] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _python_steady_state_r(
    phases_init: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Python reference matching the Rust ``bifurcation::steady_state_r``.

    Full-snapshot explicit Euler step; identical accumulator order
    (sum across j for each i). The ``|k_ij| < 1e-30`` skip in the
    Rust kernel is purely a performance shortcut — for finite phase
    inputs, ``0 · sin(finite) = 0`` — so the Python reference omits
    it and still produces bit-exact results.
    """
    phases = np.asarray(phases_init, dtype=np.float64).copy()
    knm = np.asarray(knm_flat, dtype=np.float64).reshape(n, n)
    alpha = np.asarray(alpha_flat, dtype=np.float64).reshape(n, n)
    om = np.asarray(omegas, dtype=np.float64)

    for _ in range(n_transient):
        # diff[i, j] = phases[j] - phases[i] - alpha[i, j]
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * k_scale * np.sin(diff), axis=1)
        phases = phases + dt * (om + coupling)

    if n_measure == 0:
        return 0.0
    r_sum = 0.0
    for _ in range(n_measure):
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * k_scale * np.sin(diff), axis=1)
        phases = phases + dt * (om + coupling)
        z = np.mean(np.exp(1j * phases))
        r_sum += float(np.abs(z))
    return r_sum / n_measure


def steady_state_r(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray | None = None,
    k_scale: float = 1.0,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
) -> float:
    """One-trial Kuramoto steady-state R (dispatched).

    Integrates the Kuramoto ODE for ``n_transient + n_measure`` steps
    and returns the time-averaged order parameter over the latter
    window. Delegates to the fastest available backend.
    """
    N = int(len(omegas))
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = np.ascontiguousarray(alpha, dtype=np.float64).ravel()
    knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
    phases_in = np.ascontiguousarray(phases_init, dtype=np.float64)
    backend_fn = _dispatch()
    if backend_fn is not None:
        return float(
            backend_fn(
                phases_in,
                np.ascontiguousarray(omegas, dtype=np.float64),
                knm_flat,
                alpha_flat,
                N,
                float(k_scale),
                float(dt),
                int(n_transient),
                int(n_measure),
            )
        )
    return _python_steady_state_r(
        phases_in,
        omegas,
        knm_flat,
        alpha_flat,
        N,
        float(k_scale),
        float(dt),
        int(n_transient),
        int(n_measure),
    )


@dataclass
class BasinStabilityResult:
    """Basin stability estimation result.

    Attributes:
        S_B: Basin stability (fraction of ICs converging to sync).
        n_samples: Total number of initial conditions tested.
        n_converged: Number that converged to synchronised state.
        R_final: (n_samples,) final order parameter for each trial.
        R_threshold: Threshold used for sync classification.
    """

    S_B: float
    n_samples: int
    n_converged: int
    R_final: NDArray
    R_threshold: float


def _monte_carlo_R_finals(
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    dt: float,
    n_transient: int,
    n_measure: int,
    n_samples: int,
    seed: int,
) -> NDArray:
    rng = np.random.default_rng(seed)
    R_finals = np.zeros(n_samples)
    backend_fn = _dispatch()
    for i in range(n_samples):
        phases_init = rng.uniform(0, 2 * np.pi, n)
        if backend_fn is not None:
            R_finals[i] = float(
                backend_fn(
                    phases_init,
                    np.ascontiguousarray(omegas, dtype=np.float64),
                    knm_flat,
                    alpha_flat,
                    n,
                    1.0,
                    dt,
                    n_transient,
                    n_measure,
                )
            )
        else:
            R_finals[i] = _python_steady_state_r(
                phases_init,
                omegas,
                knm_flat,
                alpha_flat,
                n,
                1.0,
                dt,
                n_transient,
                n_measure,
            )
    return R_finals


def basin_stability(
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_threshold: float = 0.8,
    seed: int = 42,
) -> BasinStabilityResult:
    """Estimate basin stability of the synchronised state.

    Draws ``n_samples`` random initial phase configurations from
    ``[0, 2π)^N``, integrates each to steady state via the dispatched
    trial kernel, and classifies trials by ``R_final ≥ R_threshold``.

    Args:
        omegas: (N,) natural frequencies.
        knm: (N, N) coupling matrix.
        alpha: (N, N) phase lags (default: zeros).
        dt: Integration timestep.
        n_transient: Transient steps to discard.
        n_measure: Steps to average R over.
        n_samples: Number of random initial conditions.
        R_threshold: Threshold for classifying as "synchronised".
        seed: RNG seed (owned by Python).

    Returns:
        BasinStabilityResult with S_B, R_final array, and counts.
    """
    N = int(len(omegas))
    knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = np.ascontiguousarray(alpha, dtype=np.float64).ravel()

    R_finals = _monte_carlo_R_finals(
        np.ascontiguousarray(omegas, dtype=np.float64),
        knm_flat,
        alpha_flat,
        N,
        float(dt),
        int(n_transient),
        int(n_measure),
        int(n_samples),
        int(seed),
    )
    n_converged = int(np.sum(R_finals >= R_threshold))
    return BasinStabilityResult(
        S_B=n_converged / n_samples if n_samples > 0 else 0.0,
        n_samples=n_samples,
        n_converged=n_converged,
        R_final=R_finals,
        R_threshold=R_threshold,
    )


def multi_basin_stability(
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_thresholds: tuple[float, ...] = (0.3, 0.6, 0.8),
    seed: int = 42,
) -> dict[str, BasinStabilityResult]:
    """Basin stability at multiple synchronisation thresholds.

    One Monte Carlo sweep; threshold classification repeated locally
    for each entry of ``R_thresholds``.

    Returns:
        Dict mapping ``"R>={thresh:.2f}"`` to BasinStabilityResult.
    """
    N = int(len(omegas))
    knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = np.ascontiguousarray(alpha, dtype=np.float64).ravel()

    R_finals = _monte_carlo_R_finals(
        np.ascontiguousarray(omegas, dtype=np.float64),
        knm_flat,
        alpha_flat,
        N,
        float(dt),
        int(n_transient),
        int(n_measure),
        int(n_samples),
        int(seed),
    )
    results: dict[str, BasinStabilityResult] = {}
    for thresh in R_thresholds:
        n_above = int(np.sum(R_finals >= thresh))
        results[f"R>={thresh:.2f}"] = BasinStabilityResult(
            S_B=n_above / n_samples if n_samples > 0 else 0.0,
            n_samples=n_samples,
            n_converged=n_above,
            R_final=R_finals,
            R_threshold=thresh,
        )
    return results
