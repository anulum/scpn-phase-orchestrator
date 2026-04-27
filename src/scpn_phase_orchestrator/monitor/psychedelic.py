# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic simulation protocol
#
# Carhart-Harris et al. 2014, Front. Hum. Neurosci. 8:20
# ("The entropic brain: a theory of conscious states informed by
# neuroimaging research with psychedelic drugs")

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.upde.engine import UPDEEngine

try:
    from spo_kernel import (
        reduce_coupling_rust as _rust_reduce,
    )

    _HAS_RUST_REDUCE = True
except ImportError:
    _HAS_RUST_REDUCE = False

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "entropy_from_phases",
    "reduce_coupling",
    "simulate_psychedelic_trajectory",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., float]:
    from spo_kernel import entropy_from_phases_rust

    def _rust(phases: NDArray, n_bins: int) -> float:
        return float(
            entropy_from_phases_rust(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                int(n_bins),
            )
        )

    return cast("Callable[..., float]", _rust)


def _load_mojo_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._psychedelic_mojo import (
        _ensure_exe,
        entropy_from_phases_mojo,
    )

    _ensure_exe()
    return entropy_from_phases_mojo


def _load_julia_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._psychedelic_julia import (
        entropy_from_phases_julia,
    )

    return entropy_from_phases_julia


def _load_go_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._psychedelic_go import (
        entropy_from_phases_go,
    )

    return entropy_from_phases_go


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


def reduce_coupling(knm: NDArray, reduction_factor: float) -> NDArray:
    """Scale coupling matrix by ``(1 − reduction_factor)``.

    Args:
        knm: ``(n, n)`` coupling matrix.
        reduction_factor: fraction to reduce, in ``[0, 1]``.

    Returns:
        Scaled copy. Zero when ``reduction_factor == 1``.
    """
    k = np.asarray(knm, dtype=np.float64)
    if _HAS_RUST_REDUCE:
        flat = np.ascontiguousarray(k.ravel())
        return np.asarray(_rust_reduce(flat, reduction_factor)).reshape(k.shape)
    return k * (1.0 - reduction_factor)


def entropy_from_phases(phases: NDArray, n_bins: int = 36) -> float:
    """Circular Shannon entropy of a phase distribution.

    Wraps phases to ``[0, 2π)``, bins into ``n_bins`` equal-width
    intervals (default 36 = 10° resolution), returns entropy in
    nats.

    Carhart-Harris et al. 2014, Front. Hum. Neurosci. **8**:20
    ("The entropic brain").
    """
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    backend_fn = _dispatch()
    if backend_fn is not None:
        return float(backend_fn(phases, int(n_bins)))

    wrapped = phases % (2.0 * np.pi)
    counts, _ = np.histogram(
        wrapped,
        bins=int(n_bins),
        range=(0, 2.0 * np.pi),
    )
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def simulate_psychedelic_trajectory(
    engine: UPDEEngine,
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    reduction_schedule: list[float],
    n_steps_per_level: int = 100,
) -> list[dict]:
    """Progressively reduce coupling, recording observables at each level.

    Models the entropic brain hypothesis: reduced serotonergic gating
    (coupling reduction) increases neural entropy and breaks coherent
    states into chimera-like patterns.

    Args:
        engine: UPDE integrator instance.
        phases: initial oscillator phases, shape (n,).
        omegas: natural frequencies, shape (n,).
        knm: baseline coupling matrix, shape (n, n).
        alpha: phase-lag matrix, shape (n, n).
        reduction_schedule: list of reduction_factor values (0 to 1).
        n_steps_per_level: integration steps at each coupling level.

    Returns:
        List of dicts, one per level, with keys:
          reduction_factor, R, entropy, chimera_index, phases.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    results: list[dict] = []

    for rf in reduction_schedule:
        k_reduced = reduce_coupling(knm, rf)
        p = engine.run(
            p,
            omegas,
            k_reduced,
            zeta=0.0,
            psi=0.0,
            alpha=alpha,
            n_steps=n_steps_per_level,
        )
        r_val, _ = engine.compute_order_parameter(p)
        ent = entropy_from_phases(p)
        chimera = detect_chimera(p, k_reduced)

        results.append(
            {
                "reduction_factor": rf,
                "R": r_val,
                "entropy": ent,
                "chimera_index": chimera.chimera_index,
                "phases": p.copy(),
            }
        )

    return results
