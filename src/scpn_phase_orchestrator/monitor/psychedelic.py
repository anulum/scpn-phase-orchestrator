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

"""Psychedelic phase-dispersion simulation utilities for research diagnostics.

The helpers model coupling reduction, phase entropy, and trajectory evolution
through an optional backend chain while keeping a deterministic Python fallback.
Inputs are constrained to finite phase vectors, finite square coupling matrices,
and unit-interval coupling factors before simulation begins. The module is a
research simulation surface only; it does not provide clinical guidance,
actuation, dosage advice, or patient-state decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.upde.engine import UPDEEngine

FloatArray: TypeAlias = NDArray[np.float64]

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

    def _rust(phases: FloatArray, n_bins: int) -> float:
        return float(
            entropy_from_phases_rust(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                int(n_bins),
            )
        )

    return cast("Callable[..., float]", _rust)


def _load_mojo_fn() -> Callable[..., float]:
    from ..experimental.accelerators.monitor._psychedelic_mojo import (
        _ensure_exe,
        entropy_from_phases_mojo,
    )

    _ensure_exe()
    return entropy_from_phases_mojo


def _load_julia_fn() -> Callable[..., float]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._psychedelic_julia import (
        entropy_from_phases_julia,
    )

    return entropy_from_phases_julia


def _load_go_fn() -> Callable[..., float]:
    from ..experimental.accelerators.monitor._psychedelic_go import (
        _load_lib,
        entropy_from_phases_go,
    )

    _load_lib()
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


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1-D phase vector") from exc
    if phases.ndim != 1:
        raise ValueError(f"{name} must be a finite 1-D phase vector")
    if not np.all(np.isfinite(phases)):
        raise ValueError(f"{name} must contain only finite values")
    return phases


def _validate_coupling_matrix(
    value: object, *, name: str, expected_n: int | None = None
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 2-D matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a finite 2-D matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")
    if expected_n is not None and matrix.shape != (expected_n, expected_n):
        raise ValueError(
            f"{name} must have shape ({expected_n}, {expected_n}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value in [0, 1]")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must be a finite real value in [0, 1]")
    return scalar


def _validate_n_bins(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    if value < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return int(value)


def _validate_step_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_reduction_schedule(values: list[float]) -> list[float]:
    return [
        _validate_unit_interval(value, name="reduction_schedule") for value in values
    ]


def reduce_coupling(knm: FloatArray, reduction_factor: float) -> FloatArray:
    """Scale coupling matrix by ``(1 − reduction_factor)``.

    Args:
        knm: ``(n, n)`` coupling matrix.
        reduction_factor: fraction to reduce, in ``[0, 1]``.

    Returns:
        Scaled copy. Zero when ``reduction_factor == 1``.
    """
    k = _validate_coupling_matrix(knm, name="knm")
    factor = _validate_unit_interval(reduction_factor, name="reduction_factor")
    if _HAS_RUST_REDUCE:
        flat = np.ascontiguousarray(k.ravel())
        return np.asarray(_rust_reduce(flat, factor)).reshape(k.shape)
    return k * (1.0 - factor)


def entropy_from_phases(phases: FloatArray, n_bins: int = 36) -> float:
    """Circular Shannon entropy of a phase distribution.

    Wraps phases to ``[0, 2π)``, bins into ``n_bins`` equal-width
    intervals (default 36 = 10° resolution), returns entropy in
    nats.

    Carhart-Harris et al. 2014, Front. Hum. Neurosci. **8**:20
    ("The entropic brain").
    """
    bin_count = _validate_n_bins(n_bins)
    phase_values = _validate_phase_vector(phases, name="phases")
    if phase_values.size == 0:
        return 0.0
    backend_fn = _dispatch()
    if backend_fn is not None:
        return float(backend_fn(phase_values, bin_count))

    wrapped = phase_values % (2.0 * np.pi)
    counts, _ = np.histogram(
        wrapped,
        bins=bin_count,
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
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
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
    p = _validate_phase_vector(phases, name="phases").copy()
    n = int(p.size)
    if n == 0:
        raise ValueError("phases must contain at least one oscillator")
    omega_values = _validate_phase_vector(omegas, name="omegas")
    if omega_values.shape != (n,):
        raise ValueError(f"omegas must have shape ({n},), got {omega_values.shape}")
    k_base = _validate_coupling_matrix(knm, name="knm", expected_n=n)
    alpha_values = _validate_coupling_matrix(alpha, name="alpha", expected_n=n)
    schedule = _validate_reduction_schedule(reduction_schedule)
    step_count = _validate_step_count(n_steps_per_level, name="n_steps_per_level")
    results: list[dict] = []

    for rf in schedule:
        k_reduced = reduce_coupling(k_base, rf)
        p = engine.run(
            p,
            omega_values,
            k_reduced,
            zeta=0.0,
            psi=0.0,
            alpha=alpha_values,
            n_steps=step_count,
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
