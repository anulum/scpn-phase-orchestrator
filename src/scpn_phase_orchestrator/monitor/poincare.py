# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincaré section analysis

"""Poincaré-section crossings with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Detects when a trajectory crosses a hyperplane, extracts the
crossing points (Poincaré map), and computes return-time statistics.

For phase oscillators the natural section is the plane where one
oscillator's phase crosses a reference value. The module exposes
:func:`poincare_section` for generic hyperplanes and
:func:`phase_poincare` for the phase-specific case.

References:
    Poincaré 1899, "Les méthodes nouvelles de la mécanique céleste".
    Strogatz 2015, "Nonlinear Dynamics and Chaos", Ch. 8.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "PoincareResult",
    "phase_poincare",
    "poincare_section",
    "return_times",
]


_DIRECTION_IDS = {"positive": 0, "negative": 1, "both": 2}
_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import phase_poincare_rust, poincare_section_rust

    def _rust_section(
        traj_flat: FloatArray,
        t: int,
        d: int,
        normal: FloatArray,
        offset: float,
        direction_id: int,
    ) -> tuple[FloatArray, FloatArray, int]:
        dir_str = {0: "positive", 1: "negative", 2: "both"}[int(direction_id)]
        cr, times, n_cr = poincare_section_rust(
            np.ascontiguousarray(traj_flat.ravel(), dtype=np.float64),
            int(t),
            int(d),
            np.ascontiguousarray(normal.ravel(), dtype=np.float64),
            float(offset),
            dir_str,
        )
        cr_arr = np.asarray(cr, dtype=np.float64)
        times_arr = np.asarray(times, dtype=np.float64)
        pad_cr = np.zeros(int(t) * int(d), dtype=np.float64)
        pad_cr[: cr_arr.size] = cr_arr
        pad_times = np.zeros(int(t), dtype=np.float64)
        pad_times[: times_arr.size] = times_arr
        return pad_cr, pad_times, int(n_cr)

    def _rust_phase(
        phases_flat: FloatArray,
        t: int,
        n: int,
        oscillator_idx: int,
        section_phase: float,
    ) -> tuple[FloatArray, FloatArray, int]:
        cr, times, n_cr = phase_poincare_rust(
            np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64),
            int(t),
            int(n),
            int(oscillator_idx),
            float(section_phase),
        )
        cr_arr = np.asarray(cr, dtype=np.float64)
        times_arr = np.asarray(times, dtype=np.float64)
        pad_cr = np.zeros(int(t) * int(n), dtype=np.float64)
        pad_cr[: cr_arr.size] = cr_arr
        pad_times = np.zeros(int(t), dtype=np.float64)
        pad_times[: times_arr.size] = times_arr
        return pad_cr, pad_times, int(n_cr)

    return {"section": _rust_section, "phase": _rust_phase}


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._poincare_mojo import (
        _ensure_exe,
        phase_poincare_mojo,
        poincare_section_mojo,
    )

    _ensure_exe()
    return {"section": poincare_section_mojo, "phase": phase_poincare_mojo}


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._poincare_julia import (
        phase_poincare_julia,
        poincare_section_julia,
    )

    return {"section": poincare_section_julia, "phase": phase_poincare_julia}


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._poincare_go import (
        _load_lib,
        phase_poincare_go,
        poincare_section_go,
    )

    _load_lib()
    return {"section": poincare_section_go, "phase": phase_poincare_go}


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object | None:
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
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError):
            continue
        if fn is not None:
            return fn
    return None


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_state_history(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1D or 2D float array") from exc
    if array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_normal(normal: object, *, expected_dim: int) -> FloatArray:
    raw = np.asarray(normal)
    if _contains_boolean_alias(normal):
        raise ValueError("normal must not contain boolean values")
    try:
        normal_vec = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("normal must be a finite one-dimensional float array") from exc
    if normal_vec.ndim != 1 or normal_vec.shape != (expected_dim,):
        raise ValueError(
            f"normal shape {normal_vec.shape} does not match ({expected_dim},)"
        )
    if not np.all(np.isfinite(normal_vec)):
        raise ValueError("normal must contain only finite values")
    return np.ascontiguousarray(normal_vec, dtype=np.float64)


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_oscillator_idx(value: object, *, n: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"oscillator_idx must be an integer in [0, {n})")
    idx = int(value)
    if idx < 0 or idx >= n:
        raise ValueError(f"oscillator_idx must be in [0, {n}), got {idx}")
    return idx


@dataclass
class PoincareResult:
    """Poincaré-section output."""

    crossings: FloatArray
    crossing_times: FloatArray
    return_times: FloatArray
    mean_return_time: float
    std_return_time: float

    def __post_init__(self) -> None:
        crossings = _validate_crossings(self.crossings)
        crossing_times = _validate_crossing_times(
            self.crossing_times,
            expected_count=int(crossings.shape[0]),
        )
        return_times_array = _validate_return_times(
            self.return_times,
            crossing_times=crossing_times,
        )
        mean_return_time = _validate_finite_real(
            self.mean_return_time,
            name="mean_return_time",
        )
        std_return_time = _validate_finite_real(
            self.std_return_time,
            name="std_return_time",
        )
        if mean_return_time < 0.0:
            raise ValueError("mean_return_time must be non-negative")
        if std_return_time < 0.0:
            raise ValueError("std_return_time must be non-negative")
        expected_mean = (
            float(np.mean(return_times_array)) if return_times_array.size else 0.0
        )
        expected_std = (
            float(np.std(return_times_array)) if return_times_array.size else 0.0
        )
        if not np.isclose(mean_return_time, expected_mean, rtol=1e-12, atol=1e-12):
            raise ValueError("mean_return_time must match return_times")
        if not np.isclose(std_return_time, expected_std, rtol=1e-12, atol=1e-12):
            raise ValueError("std_return_time must match return_times")

        self.crossings = crossings
        self.crossing_times = crossing_times
        self.return_times = return_times_array
        self.mean_return_time = mean_return_time
        self.std_return_time = std_return_time


def _validate_crossings(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("crossings must not contain boolean values")
    try:
        crossings = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("crossings must be a finite two-dimensional array") from exc
    if crossings.ndim != 2:
        raise ValueError(f"crossings must be two-dimensional, got {crossings.shape}")
    if not np.all(np.isfinite(crossings)):
        raise ValueError("crossings must contain only finite values")
    return np.ascontiguousarray(crossings, dtype=np.float64)


def _validate_crossing_times(value: object, *, expected_count: int) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("crossing_times must not contain boolean values")
    try:
        crossing_times = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "crossing_times must be a finite one-dimensional array"
        ) from exc
    if crossing_times.ndim != 1:
        raise ValueError("crossing_times must be one-dimensional")
    if crossing_times.shape != (expected_count,):
        raise ValueError(
            f"crossing_times shape {crossing_times.shape} does not match "
            f"({expected_count},)"
        )
    if not np.all(np.isfinite(crossing_times)):
        raise ValueError("crossing_times must contain only finite values")
    if crossing_times.size > 1 and np.any(np.diff(crossing_times) <= 1e-12):
        raise ValueError("crossing_times must be strictly increasing")
    return np.ascontiguousarray(crossing_times, dtype=np.float64)


def _validate_return_times(
    value: object,
    *,
    crossing_times: FloatArray,
) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("return_times must not contain boolean values")
    try:
        return_times_array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("return_times must be a finite one-dimensional array") from exc
    if return_times_array.ndim != 1:
        raise ValueError("return_times must be one-dimensional")
    expected = np.diff(crossing_times)
    if return_times_array.shape != expected.shape:
        raise ValueError(
            f"return_times shape {return_times_array.shape} does not match "
            f"{expected.shape}"
        )
    if not np.all(np.isfinite(return_times_array)):
        raise ValueError("return_times must contain only finite values")
    if np.any(return_times_array < -1e-12):
        raise ValueError("return_times must be non-negative")
    if not np.allclose(return_times_array, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("return_times must match crossing_times differences")
    return np.ascontiguousarray(np.maximum(return_times_array, 0.0), dtype=np.float64)


def _assemble_result(
    crossings_flat: FloatArray,
    times: FloatArray,
    n_cr: int,
    dim: int,
) -> PoincareResult:
    if isinstance(n_cr, bool) or not isinstance(n_cr, Integral):
        raise ValueError("n_cr must be an integer")
    n_cr = int(n_cr)
    if n_cr < 0:
        raise ValueError("n_cr must be non-negative")
    if dim < 1:
        raise ValueError("dim must be positive")
    crossings_flat = _validate_state_history(crossings_flat, name="crossings_flat")
    crossings_flat = crossings_flat.ravel()
    times = _validate_state_history(times, name="times").ravel()
    if n_cr > times.size or n_cr * dim > crossings_flat.size:
        raise ValueError("n_cr exceeds backend crossing payload size")
    if n_cr == 0:
        return PoincareResult(
            crossings=np.empty((0, dim)),
            crossing_times=np.array([]),
            return_times=np.array([]),
            mean_return_time=0.0,
            std_return_time=0.0,
        )
    cr = crossings_flat[: n_cr * dim].reshape(n_cr, dim)
    ct = times[:n_cr]
    rt = np.diff(ct)
    return PoincareResult(
        crossings=cr,
        crossing_times=ct,
        return_times=rt,
        mean_return_time=float(np.mean(rt)) if rt.size > 0 else 0.0,
        std_return_time=float(np.std(rt)) if rt.size > 0 else 0.0,
    )


def poincare_section(
    trajectory: object,
    normal: object,
    offset: object = 0.0,
    direction: str = "positive",
) -> PoincareResult:
    """Hyperplane-crossing Poincaré section."""
    traj = _validate_state_history(trajectory, name="trajectory")
    t, d = int(traj.shape[0]), int(traj.shape[1])
    norm_vec = _validate_normal(normal, expected_dim=d)
    offset = _validate_finite_real(offset, name="offset")
    direction_id = _DIRECTION_IDS.get(direction)
    if direction_id is None:
        raise ValueError(
            f"direction must be one of {list(_DIRECTION_IDS)}, got {direction!r}"
        )

    norm_mag = float(np.linalg.norm(norm_vec))
    if norm_mag == 0.0:
        return _assemble_result(np.zeros(t * d), np.zeros(t), 0, d)

    backend_fn = _dispatch("section")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, FloatArray, float, int], "
            "tuple[FloatArray, FloatArray, int]]",
            backend_fn,
        )
        try:
            cr_flat, times, n_cr = fn(
                traj.ravel(),
                t,
                d,
                norm_vec,
                offset,
                int(direction_id),
            )
            return _assemble_result(cr_flat, times, n_cr, d)
        except Exception:
            direction_id = int(direction_id)

    n = norm_vec / norm_mag
    signed_dist = traj @ n - offset

    cr_flat = np.zeros(t * d, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    n_cr = 0
    for i in range(len(signed_dist) - 1):
        d0, d1 = signed_dist[i], signed_dist[i + 1]
        is_cross = (
            (direction == "positive" and d0 < 0 and d1 >= 0)
            or (direction == "negative" and d0 > 0 and d1 <= 0)
            or (direction == "both" and d0 * d1 < 0)
        )
        if not is_cross:
            continue
        alpha = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-15 else 0.5
        pt = traj[i] + alpha * (traj[i + 1] - traj[i])
        cr_flat[n_cr * d : (n_cr + 1) * d] = pt
        times[n_cr] = i + alpha
        n_cr += 1

    return _assemble_result(cr_flat, times, n_cr, d)


def return_times(
    trajectory: object,
    normal: object,
    offset: object = 0.0,
) -> FloatArray:
    """Shortcut: return only the return-time sequence."""
    return poincare_section(
        trajectory,
        normal,
        offset,
        direction="positive",
    ).return_times


def phase_poincare(
    phases: object,
    oscillator_idx: object = 0,
    section_phase: object = 0.0,
) -> PoincareResult:
    """Poincaré section for phase-oscillator trajectories.

    Detects when ``phases[:, oscillator_idx]`` crosses
    ``section_phase (mod 2π)``.
    """
    phases = _validate_state_history(phases, name="phases")
    t, n = int(phases.shape[0]), int(phases.shape[1])
    oscillator_idx = _validate_oscillator_idx(oscillator_idx, n=n)
    section_phase = _validate_finite_real(section_phase, name="section_phase")

    backend_fn = _dispatch("phase")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, int, float], "
            "tuple[FloatArray, FloatArray, int]]",
            backend_fn,
        )
        try:
            cr_flat, times, n_cr = fn(
                phases.ravel(),
                t,
                n,
                oscillator_idx,
                section_phase,
            )
            return _assemble_result(cr_flat, times, n_cr, n)
        except Exception:
            oscillator_idx = int(oscillator_idx)

    target = np.unwrap(phases[:, oscillator_idx])
    shifted = (target - section_phase) % (2 * np.pi)

    cr_flat = np.zeros(t * n, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    n_cr = 0
    for i in range(t - 1):
        if shifted[i] > np.pi and shifted[i + 1] < np.pi:
            alpha = shifted[i] / (shifted[i] - shifted[i + 1] + 2 * np.pi)
            alpha = min(max(alpha, 0.0), 1.0)
            pt = phases[i] + alpha * (phases[i + 1] - phases[i])
            cr_flat[n_cr * n : (n_cr + 1) * n] = pt
            times[n_cr] = i + alpha
            n_cr += 1

    return _assemble_result(cr_flat, times, n_cr, n)
