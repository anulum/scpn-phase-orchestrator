# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence analysis for phase dynamics

"""Recurrence analysis with a 5-backend fallback chain.

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
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
ByteArray: TypeAlias = NDArray[np.uint8]

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
        traj_flat: FloatArray,
        t: int,
        d: int,
        epsilon: float,
        angular: bool,
    ) -> ByteArray:
        return np.asarray(
            recurrence_matrix_rust(traj_flat, t, d, epsilon, angular),
            dtype=np.uint8,
        )

    def _rust_cross(
        a_flat: FloatArray,
        b_flat: FloatArray,
        t: int,
        d: int,
        epsilon: float,
        angular: bool,
    ) -> ByteArray:
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


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._recurrence_mojo import (
        _ensure_exe,
        cross_recurrence_matrix_mojo,
        recurrence_matrix_mojo,
    )

    _ensure_exe()
    return {
        "rm": recurrence_matrix_mojo,
        "cross_rm": cross_recurrence_matrix_mojo,
    }


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._recurrence_julia import (
        cross_recurrence_matrix_julia,
        recurrence_matrix_julia,
    )

    return {
        "rm": recurrence_matrix_julia,
        "cross_rm": cross_recurrence_matrix_julia,
    }


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._recurrence_go import (
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
        except (ImportError, RuntimeError, OSError, KeyError):
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
        if backend not in _LOADERS:
            continue
        try:
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is None:
            continue
        return fn
    return None


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_trajectory(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real-valued trajectory samples")
    try:
        trajectory = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1D or 2D float array") from exc
    if trajectory.ndim == 1:
        trajectory = trajectory[:, np.newaxis]
    elif trajectory.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(trajectory, dtype=np.float64)


def _validate_epsilon(epsilon: object) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError(f"epsilon must be a finite non-negative real, got {epsilon!r}")
    result = float(epsilon)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon!r}")
    return result


def _validate_metric(metric: object) -> bool:
    if metric not in {"euclidean", "angular"}:
        raise ValueError("metric must be 'euclidean' or 'angular'")
    return metric == "angular"


def _validate_line_min(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {result}")
    return result


def _validate_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real in [0, 1], got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return result


def _validate_non_negative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _validate_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _expected_recurrence_matrix(
    traj_a: FloatArray,
    traj_b: FloatArray,
    *,
    epsilon: float,
    angular: bool,
) -> BoolArray:
    diff = traj_a[:, np.newaxis, :] - traj_b[np.newaxis, :, :]
    if angular:
        dist = np.sqrt(np.sum(4.0 * np.sin(diff / 2.0) ** 2, axis=2))
    else:
        dist = np.sqrt(np.sum(diff**2, axis=2))
    result: BoolArray = dist <= epsilon
    return result


def _backend_recurrence_matrix(
    value: object,
    *,
    t: int,
    name: str,
    expected: BoolArray,
) -> BoolArray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} backend output must be array-like") from exc
    if array.size != t * t:
        raise ValueError(
            f"{name} backend output size must be {t * t}, got {array.size}"
        )
    try:
        numeric = array.astype(np.float64, copy=False).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} backend output must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} backend output must contain only finite values")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} backend output must contain only 0/1 values")
    matrix = numeric.reshape(t, t).astype(bool)
    if name == "recurrence_matrix":
        if not np.all(np.diag(matrix)):
            raise ValueError("recurrence_matrix backend output must have true diagonal")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("recurrence_matrix backend output must be symmetric")
    if expected.shape != (t, t):
        raise ValueError(f"{name} expected output shape must be {(t, t)}")
    if not np.array_equal(matrix, expected):
        raise ValueError(f"{name} backend output must match exact recurrence threshold")
    return matrix


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

    def __post_init__(self) -> None:
        self.recurrence_rate = _validate_unit_interval(
            self.recurrence_rate,
            name="recurrence_rate",
        )
        self.determinism = _validate_unit_interval(
            self.determinism,
            name="determinism",
        )
        self.avg_diagonal = _validate_non_negative_float(
            self.avg_diagonal,
            name="avg_diagonal",
        )
        self.max_diagonal = _validate_non_negative_int(
            self.max_diagonal,
            name="max_diagonal",
        )
        self.entropy_diagonal = _validate_non_negative_float(
            self.entropy_diagonal,
            name="entropy_diagonal",
        )
        self.laminarity = _validate_unit_interval(
            self.laminarity,
            name="laminarity",
        )
        self.trapping_time = _validate_non_negative_float(
            self.trapping_time,
            name="trapping_time",
        )
        self.max_vertical = _validate_non_negative_int(
            self.max_vertical,
            name="max_vertical",
        )
        if self.avg_diagonal > self.max_diagonal + 1e-12:
            raise ValueError("avg_diagonal must not exceed max_diagonal")
        if self.max_diagonal == 0 and self.entropy_diagonal > 0.0:
            raise ValueError("entropy_diagonal requires at least one diagonal line")
        if self.trapping_time > self.max_vertical + 1e-12:
            raise ValueError("trapping_time must not exceed max_vertical")


def recurrence_matrix(
    trajectory: FloatArray,
    epsilon: float,
    metric: str = "euclidean",
) -> BoolArray:
    """Binary recurrence matrix ``R_ij = ‖x_i − x_j‖ ≤ ε``.

    Args:
        trajectory: ``(T, d)`` or ``(T,)`` state-space trajectory.
        epsilon: recurrence threshold.
        metric: ``"euclidean"`` or ``"angular"`` (chord distance on
            ``S¹``).

    Returns
    -------
        ``(T, T)`` boolean array.

    Parameters
    ----------
    trajectory : FloatArray
        Phase-space trajectory, shape ``(T, d)``.
    epsilon : float
        Recurrence threshold.
    metric : str
        Distance metric name.
    """
    traj = _validate_trajectory(trajectory, name="trajectory")
    epsilon = _validate_epsilon(epsilon)
    t, d = int(traj.shape[0]), int(traj.shape[1])
    angular = _validate_metric(metric)
    if t == 0:
        return np.zeros((0, 0), dtype=bool)
    flat = traj.ravel()
    expected = _expected_recurrence_matrix(
        traj,
        traj,
        epsilon=epsilon,
        angular=angular,
    )

    backend_fn = _dispatch("rm")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, float, bool], ByteArray]",
            backend_fn,
        )
        try:
            return _backend_recurrence_matrix(
                fn(flat, t, d, epsilon, angular),
                t=t,
                name="recurrence_matrix",
                expected=expected,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            angular = bool(angular)

    return expected


def cross_recurrence_matrix(
    traj_a: FloatArray,
    traj_b: FloatArray,
    epsilon: float,
    metric: str = "euclidean",
) -> BoolArray:
    """Cross-recurrence matrix ``CR_ij = ‖x_i − y_j‖ ≤ ε``.

    ``traj_a`` and ``traj_b`` must have the same length and
    dimensionality.

    Parameters
    ----------
    traj_a : FloatArray
        First trajectory, shape ``(T, d)``.
    traj_b : FloatArray
        Second trajectory, shape ``(T, d)``.
    epsilon : float
        Recurrence threshold.
    metric : str
        Distance metric name.

    Returns
    -------
    BoolArray
        The binary cross-recurrence matrix.

    Raises
    ------
    ValueError
        If the two trajectories are incompatible.
    """
    a = _validate_trajectory(traj_a, name="traj_a")
    b = _validate_trajectory(traj_b, name="traj_b")
    epsilon = _validate_epsilon(epsilon)
    t, d = int(a.shape[0]), int(a.shape[1])
    if b.shape != a.shape:
        raise ValueError(f"trajectories must match: a={a.shape} b={b.shape}")
    angular = _validate_metric(metric)
    if t == 0:
        return np.zeros((0, 0), dtype=bool)
    a_flat = a.ravel()
    b_flat = b.ravel()
    expected = _expected_recurrence_matrix(
        a,
        b,
        epsilon=epsilon,
        angular=angular,
    )

    backend_fn = _dispatch("cross_rm")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, FloatArray, int, int, float, bool], ByteArray]",
            backend_fn,
        )
        try:
            return _backend_recurrence_matrix(
                fn(a_flat, b_flat, t, d, epsilon, angular),
                t=t,
                name="cross_recurrence_matrix",
                expected=expected,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            angular = bool(angular)

    return expected


def _diagonal_lines(R: BoolArray, l_min: int = 2) -> list[int]:
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


def _vertical_lines(R: BoolArray, v_min: int = 2) -> list[int]:
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
    R: BoolArray,
    l_min: int,
    v_min: int,
    exclude_main_diag: bool,
) -> RQAResult:
    """Run the shared line-analysis path for :func:`rqa` and :func:`cross_rqa`."""
    l_min = _validate_line_min(l_min, name="l_min")
    v_min = _validate_line_min(v_min, name="v_min")
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
    trajectory: FloatArray,
    epsilon: float,
    l_min: int = 2,
    v_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Full Recurrence Quantification Analysis.

    The recurrence matrix is computed via the 5-backend dispatcher;
    line-length histograms and RQA statistics are computed in
    Python for uniformity across backends.

    Parameters
    ----------
    trajectory : FloatArray
        Phase-space trajectory, shape ``(T, d)``.
    epsilon : float
        Recurrence threshold.
    l_min : int
        Minimum diagonal-line length counted by RQA.
    v_min : int
        Minimum vertical-line length counted by RQA.
    metric : str
        Distance metric name.

    Returns
    -------
    RQAResult
        The recurrence quantification analysis result.
    """
    R = recurrence_matrix(trajectory, epsilon, metric)
    return _rqa_from_matrix(R, l_min, v_min, exclude_main_diag=True)


def cross_rqa(
    traj_a: FloatArray,
    traj_b: FloatArray,
    epsilon: float,
    l_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Cross-Recurrence Quantification Analysis between two trajectories.

    Parameters
    ----------
    traj_a : FloatArray
        First trajectory, shape ``(T, d)``.
    traj_b : FloatArray
        Second trajectory, shape ``(T, d)``.
    epsilon : float
        Recurrence threshold.
    l_min : int
        Minimum diagonal-line length counted by RQA.
    metric : str
        Distance metric name.

    Returns
    -------
    RQAResult
        The cross-recurrence quantification analysis result.
    """
    CR = cross_recurrence_matrix(traj_a, traj_b, epsilon, metric)
    return _rqa_from_matrix(CR, l_min, l_min, exclude_main_diag=False)
