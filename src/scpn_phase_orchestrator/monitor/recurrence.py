# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence analysis for phase dynamics

"""Recurrence analysis with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Compute surface:

* :func:`recurrence_matrix` — ``R_ij = Θ(ε − ‖x_i − x_j‖)``.
* :func:`cross_recurrence_matrix` — cross-recurrence of two
  trajectories.
* :func:`rqa` — full Recurrence Quantification Analysis using the
  dispatched matrix; line-length histograms + RQA statistics stay
  Python-side for uniformity.
* :func:`cross_rqa` — cross-RQA; same pattern.

References: Eckmann, Kamphorst & Ruelle 1987, Europhys. Lett.
**4**:973–977; Zbilut & Webber 1992, Phys. Lett. A **171**:199–203;
Marwan et al. 2007, Phys. Reports **438**:237–329.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "RQAResult",
    "cross_recurrence_matrix",
    "cross_rqa",
    "recurrence_matrix",
    "rqa",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        cross_recurrence_matrix_rust,
        recurrence_matrix_rust,
    )

    def _rust_rm(
        traj_flat: NDArray,
        t: int,
        d: int,
        epsilon: float,
        angular: bool,
    ) -> NDArray:
        return np.asarray(
            recurrence_matrix_rust(traj_flat, t, d, epsilon, angular),
            dtype=np.uint8,
        )

    def _rust_cross(
        a_flat: NDArray,
        b_flat: NDArray,
        t: int,
        d: int,
        epsilon: float,
        angular: bool,
    ) -> NDArray:
        return np.asarray(
            cross_recurrence_matrix_rust(
                a_flat,
                b_flat,
                t,
                d,
                epsilon,
                angular,
            ),
            dtype=np.uint8,
        )

    return {"rm": _rust_rm, "cross_rm": _rust_cross}


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._recurrence_mojo import (
        _ensure_exe,
        cross_recurrence_matrix_mojo,
        recurrence_matrix_mojo,
    )

    _ensure_exe()
    return {
        "rm": recurrence_matrix_mojo,
        "cross_rm": cross_recurrence_matrix_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._recurrence_julia import (
        cross_recurrence_matrix_julia,
        recurrence_matrix_julia,
    )

    return {
        "rm": recurrence_matrix_julia,
        "cross_rm": cross_recurrence_matrix_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._recurrence_go import (
        _load_lib,
        cross_recurrence_matrix_go,
        recurrence_matrix_go,
    )

    _load_lib()
    return {
        "rm": recurrence_matrix_go,
        "cross_rm": cross_recurrence_matrix_go,
    }


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
class RQAResult:
    """Standard RQA measures from Marwan et al. 2007."""

    recurrence_rate: float
    determinism: float
    avg_diagonal: float
    max_diagonal: int
    entropy_diagonal: float
    laminarity: float
    trapping_time: float
    max_vertical: int


def recurrence_matrix(
    trajectory: NDArray,
    epsilon: float,
    metric: str = "euclidean",
) -> NDArray:
    """Binary recurrence matrix ``R_ij = ‖x_i − x_j‖ ≤ ε``.

    Args:
        trajectory: ``(T, d)`` or ``(T,)`` state-space trajectory.
        epsilon: recurrence threshold.
        metric: ``"euclidean"`` or ``"angular"`` (chord distance on
            ``S¹``).

    Returns:
        ``(T, T)`` boolean array.
    """
    traj = np.asarray(trajectory)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    elif traj.ndim != 2:
        raise ValueError(f"trajectory must be 1D or 2D, got shape {traj.shape}")
    t, d = int(traj.shape[0]), int(traj.shape[1])
    angular = metric == "angular"
    flat = np.ascontiguousarray(traj.ravel(), dtype=np.float64)

    backend_fn = _dispatch("rm")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray, int, int, float, bool], NDArray]", backend_fn)
        out = np.asarray(fn(flat, t, d, float(epsilon), angular), dtype=np.uint8)
        return out.reshape(t, t).astype(bool)

    if angular:
        diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
        dist = np.sqrt(np.sum(4 * np.sin(diff / 2) ** 2, axis=2))
    else:
        diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
    result: NDArray = dist <= epsilon
    return result


def cross_recurrence_matrix(
    traj_a: NDArray,
    traj_b: NDArray,
    epsilon: float,
    metric: str = "euclidean",
) -> NDArray:
    """Cross-recurrence matrix ``CR_ij = ‖x_i − y_j‖ ≤ ε``.

    ``traj_a`` and ``traj_b`` must have the same length and
    dimensionality.
    """
    a = np.asarray(traj_a)
    b = np.asarray(traj_b)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    elif a.ndim != 2:
        raise ValueError(f"traj_a must be 1D or 2D, got shape {a.shape}")
    if b.ndim == 1:
        b = b[:, np.newaxis]
    elif b.ndim != 2:
        raise ValueError(f"traj_b must be 1D or 2D, got shape {b.shape}")
    t, d = int(a.shape[0]), int(a.shape[1])
    if b.shape != a.shape:
        raise ValueError(f"trajectories must match: a={a.shape} b={b.shape}")
    angular = metric == "angular"
    a_flat = np.ascontiguousarray(a.ravel(), dtype=np.float64)
    b_flat = np.ascontiguousarray(b.ravel(), dtype=np.float64)

    backend_fn = _dispatch("cross_rm")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, NDArray, int, int, float, bool], NDArray]",
            backend_fn,
        )
        out = np.asarray(
            fn(a_flat, b_flat, t, d, float(epsilon), angular),
            dtype=np.uint8,
        )
        return out.reshape(t, t).astype(bool)

    if angular:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        dist = np.sqrt(np.sum(4 * np.sin(diff / 2) ** 2, axis=2))
    else:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
    result: NDArray = dist <= epsilon
    return result


def _diagonal_lines(R: NDArray, l_min: int = 2) -> list[int]:
    """Diagonal line lengths, excluding the main diagonal."""
    t = R.shape[0]
    lengths: list[int] = []
    for k in range(1, t):
        count = 0
        for val in np.diag(R, k):
            if val:
                count += 1
            else:
                if count >= l_min:
                    lengths.append(count)
                count = 0
        if count >= l_min:
            lengths.append(count)
    return lengths


def _vertical_lines(R: NDArray, v_min: int = 2) -> list[int]:
    """Vertical line lengths."""
    t = R.shape[0]
    lengths: list[int] = []
    for col in range(t):
        count = 0
        for row in range(t):
            if R[row, col]:
                count += 1
            else:
                if count >= v_min:
                    lengths.append(count)
                count = 0
        if count >= v_min:
            lengths.append(count)
    return lengths


def _rqa_from_matrix(
    R: NDArray,
    l_min: int,
    v_min: int,
    exclude_main_diag: bool,
) -> RQAResult:
    """Shared line-analysis path — consumed by both :func:`rqa` and
    :func:`cross_rqa` after the dispatched matrix is obtained."""
    t = R.shape[0]
    if exclude_main_diag:
        R = R.copy()
        np.fill_diagonal(R, False)
        off_diag_total = t * t - t
    else:
        off_diag_total = t * t
    n_recurrent = int(np.sum(R))
    rr = n_recurrent / off_diag_total if off_diag_total > 0 else 0.0

    diag_lengths = _diagonal_lines(R, l_min)
    if diag_lengths:
        det = sum(diag_lengths) / n_recurrent if n_recurrent > 0 else 0.0
        avg_diag = float(np.mean(diag_lengths))
        max_diag = max(diag_lengths)
        counts = np.bincount(diag_lengths)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        ent_diag = float(-np.sum(probs * np.log(probs)))
    else:
        det = 0.0
        avg_diag = 0.0
        max_diag = 0
        ent_diag = 0.0

    vert_lengths = _vertical_lines(R, v_min)
    if vert_lengths:
        lam = sum(vert_lengths) / n_recurrent if n_recurrent > 0 else 0.0
        trap_time = float(np.mean(vert_lengths))
        max_vert = max(vert_lengths)
    else:
        lam = 0.0
        trap_time = 0.0
        max_vert = 0

    return RQAResult(
        recurrence_rate=rr,
        determinism=det,
        avg_diagonal=avg_diag,
        max_diagonal=max_diag,
        entropy_diagonal=ent_diag,
        laminarity=lam,
        trapping_time=trap_time,
        max_vertical=max_vert,
    )


def rqa(
    trajectory: NDArray,
    epsilon: float,
    l_min: int = 2,
    v_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Full Recurrence Quantification Analysis.

    The recurrence matrix is computed via the 5-backend dispatcher;
    line-length histograms and RQA statistics are computed in
    Python for uniformity across backends.
    """
    R = recurrence_matrix(trajectory, epsilon, metric)
    return _rqa_from_matrix(R, l_min, v_min, exclude_main_diag=True)


def cross_rqa(
    traj_a: NDArray,
    traj_b: NDArray,
    epsilon: float,
    l_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Cross-Recurrence Quantification Analysis between two trajectories."""
    CR = cross_recurrence_matrix(traj_a, traj_b, epsilon, metric)
    return _rqa_from_matrix(CR, l_min, l_min, exclude_main_diag=False)
