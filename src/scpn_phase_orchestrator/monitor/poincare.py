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


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._poincare_mojo import (
        _ensure_exe,
        phase_poincare_mojo,
        poincare_section_mojo,
    )

    _ensure_exe()
    return {"section": poincare_section_mojo, "phase": phase_poincare_mojo}


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._poincare_julia import (
        phase_poincare_julia,
        poincare_section_julia,
    )

    return {"section": poincare_section_julia, "phase": phase_poincare_julia}


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._poincare_go import (
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


def _dispatch(fn_name: str) -> object | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


@dataclass
class PoincareResult:
    """Poincaré-section output."""

    crossings: FloatArray
    crossing_times: FloatArray
    return_times: FloatArray
    mean_return_time: float
    std_return_time: float


def _assemble_result(
    crossings_flat: FloatArray,
    times: FloatArray,
    n_cr: int,
    dim: int,
) -> PoincareResult:
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
    trajectory: FloatArray,
    normal: FloatArray,
    offset: float = 0.0,
    direction: str = "positive",
) -> PoincareResult:
    """Hyperplane-crossing Poincaré section."""
    traj = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
    t, d = int(traj.shape[0]), int(traj.shape[1])
    norm_vec = np.asarray(normal, dtype=np.float64)
    direction_id = _DIRECTION_IDS.get(direction)
    if direction_id is None:
        raise ValueError(
            f"direction must be one of {list(_DIRECTION_IDS)}, got {direction!r}"
        )

    backend_fn = _dispatch("section")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, FloatArray, float, int], "
            "tuple[FloatArray, FloatArray, int]]",
            backend_fn,
        )
        cr_flat, times, n_cr = fn(
            traj.ravel(),
            t,
            d,
            norm_vec,
            float(offset),
            int(direction_id),
        )
        return _assemble_result(cr_flat, times, n_cr, d)

    norm_mag = float(np.linalg.norm(norm_vec))
    if norm_mag == 0.0:
        return _assemble_result(np.zeros(t * d), np.zeros(t), 0, d)
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
    trajectory: FloatArray,
    normal: FloatArray,
    offset: float = 0.0,
) -> FloatArray:
    """Shortcut: return only the return-time sequence."""
    return poincare_section(
        trajectory,
        normal,
        offset,
        direction="positive",
    ).return_times


def phase_poincare(
    phases: FloatArray,
    oscillator_idx: int = 0,
    section_phase: float = 0.0,
) -> PoincareResult:
    """Poincaré section for phase-oscillator trajectories.

    Detects when ``phases[:, oscillator_idx]`` crosses
    ``section_phase (mod 2π)``.
    """
    phases = np.atleast_2d(np.asarray(phases, dtype=np.float64))
    t, n = int(phases.shape[0]), int(phases.shape[1])

    backend_fn = _dispatch("phase")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, int, float], "
            "tuple[FloatArray, FloatArray, int]]",
            backend_fn,
        )
        cr_flat, times, n_cr = fn(
            phases.ravel(),
            t,
            n,
            int(oscillator_idx),
            float(section_phase),
        )
        return _assemble_result(cr_flat, times, n_cr, n)

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
