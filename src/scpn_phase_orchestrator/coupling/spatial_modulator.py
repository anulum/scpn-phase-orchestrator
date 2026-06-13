# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spatial coupling modulation

"""Distance-dependent coupling modulation for moving oscillator systems.

``SpatialCouplingModulator`` converts a base coupling matrix ``K_nm`` and an
absolute position array into a distance-weighted coupling matrix. The default
MIF/general moving-agent contract is

    K'_ij = K_base * K_ij / (1 + ||x_i - x_j||),   K'_ii = 0.

Additional decay forms support exponential, finite power-law, and the exact
regularised inverse-distance kernel used by the existing Swarmalator phase
coupling term. Inputs are validated at the public boundary before any optional
backend is called.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "SpatialCouplingModulator",
    "spatial_modulate",
]

FloatArray: TypeAlias = NDArray[np.float64]
DecayForm = Literal["inverse_plus_one", "exponential", "power_law", "inverse_distance"]
DistanceFn = Callable[[FloatArray, FloatArray], object]

_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")
_DECAY_TO_CODE: dict[str, int] = {
    "inverse_plus_one": 0,
    "exponential": 1,
    "power_law": 2,
    "inverse_distance": 3,
}


def _contains_boolean_alias(value: object) -> bool:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype != object:
            return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        return True
    if isinstance(value, np.ndarray) and raw.dtype != object:
        return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_scalar(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _validate_non_negative_scalar(value: object, *, name: str) -> float:
    resolved = _validate_scalar(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _validate_decay_form(value: object) -> DecayForm:
    if not isinstance(value, str) or value not in _DECAY_TO_CODE:
        valid = ", ".join(sorted(_DECAY_TO_CODE))
        raise ValueError(f"decay_form must be one of: {valid}")
    return cast("DecayForm", value)


def _validate_positions(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("positions must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or _contains_complex_alias(value):
        raise ValueError("positions must be finite real coordinates")
    try:
        positions = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a finite real array") from exc
    if positions.ndim == 1:
        positions = positions.reshape(positions.shape[0], 1)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (n,) or (n, dim)")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("positions must contain at least one oscillator and dimension")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    return np.ascontiguousarray(positions, dtype=np.float64)


def _validate_knm_base(value: object, *, expected_n: int | None = None) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("k_nm_base must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or _contains_complex_alias(value):
        raise ValueError("k_nm_base must be a finite real square matrix")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_nm_base must be a finite real square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("k_nm_base must be a finite real square matrix")
    if expected_n is not None and matrix.shape != (expected_n, expected_n):
        expected_shape = (expected_n, expected_n)
        raise ValueError(
            f"k_nm_base shape {matrix.shape} does not match {expected_shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("k_nm_base must contain only finite values")
    if np.max(np.abs(np.diag(matrix))) > 1.0e-12:
        raise ValueError("k_nm_base diagonal must be zero")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_distance_matrix(value: object, *, n: int) -> FloatArray:
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError("distance matrix must be finite real-valued")
    try:
        distances = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance matrix must be finite real-valued") from exc
    if distances.shape != (n, n):
        raise ValueError(
            f"distance matrix shape {distances.shape} does not match ({n}, {n})"
        )
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance matrix must contain only finite values")
    if np.any(distances < -1.0e-12):
        raise ValueError("distance matrix must be non-negative")
    if np.max(np.abs(np.diag(distances))) > 1.0e-12:
        raise ValueError("distance matrix diagonal must be zero")
    if np.max(np.abs(distances - distances.T)) > 1.0e-10:
        raise ValueError("distance matrix must be symmetric")
    return np.ascontiguousarray(np.maximum(distances, 0.0), dtype=np.float64)


def _pairwise_euclidean(positions: FloatArray) -> FloatArray:
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distances, 0.0)
    return np.ascontiguousarray(distances, dtype=np.float64)


def _decay_weights(
    distances: FloatArray,
    *,
    decay_form: DecayForm,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    if decay_form == "inverse_plus_one":
        weights = 1.0 / (1.0 + distances)
    elif decay_form == "exponential":
        weights = np.exp(-distances / decay_length_scale)
    elif decay_form == "power_law":
        weights = (1.0 + distances / decay_length_scale) ** (-decay_exponent)
    elif decay_form == "inverse_distance":
        weights = 1.0 / np.sqrt(distances * distances + epsilon)
    else:  # pragma: no cover - _validate_decay_form prevents this branch.
        raise ValueError(f"unsupported decay form {decay_form!r}")
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


def _python_modulation_matrix(
    positions: FloatArray,
    *,
    k_base: float,
    decay_form: DecayForm,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
    distance_fn: DistanceFn | None = None,
) -> FloatArray:
    n = positions.shape[0]
    if distance_fn is None:
        distances = _pairwise_euclidean(positions)
    else:
        left = positions[:, np.newaxis, :]
        right = positions[np.newaxis, :, :]
        distances = _validate_distance_matrix(distance_fn(left, right), n=n)
    weights = _decay_weights(
        distances,
        decay_form=decay_form,
        decay_exponent=decay_exponent,
        decay_length_scale=decay_length_scale,
        epsilon=epsilon,
    )
    return np.ascontiguousarray(k_base * weights, dtype=np.float64)


def _python_spatial_modulate(
    k_nm_flat: FloatArray,
    positions_flat: FloatArray,
    n: int,
    dim: int,
    k_base: float,
    decay_form_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    form_by_code = {code: name for name, code in _DECAY_TO_CODE.items()}
    decay_form = cast("DecayForm", form_by_code[int(decay_form_code)])
    base = k_nm_flat.reshape(n, n)
    positions = positions_flat.reshape(n, dim)
    modulation = _python_modulation_matrix(
        positions,
        k_base=k_base,
        decay_form=decay_form,
        decay_exponent=decay_exponent,
        decay_length_scale=decay_length_scale,
        epsilon=epsilon,
    )
    out = base * modulation
    np.fill_diagonal(out, 0.0)
    return np.ascontiguousarray(out, dtype=np.float64)


def _load_rust_fn() -> Callable[..., FloatArray]:
    from spo_kernel import spatial_modulate_rust

    def _rust(
        k_nm_flat: FloatArray,
        positions_flat: FloatArray,
        n: int,
        dim: int,
        k_base: float,
        decay_form_code: int,
        decay_exponent: float,
        decay_length_scale: float,
        epsilon: float,
    ) -> FloatArray:
        values = spatial_modulate_rust(
            k_nm_flat.tolist(),
            positions_flat.tolist(),
            int(n),
            int(dim),
            float(k_base),
            int(decay_form_code),
            float(decay_exponent),
            float(decay_length_scale),
            float(epsilon),
        )
        return np.asarray(values, dtype=np.float64)

    return _rust


def _load_mojo_fn() -> Callable[..., FloatArray]:
    from ..experimental.accelerators.coupling._spatial_modulator_mojo import (
        _ensure_exe,
        spatial_modulate_mojo,
    )

    _ensure_exe()
    return spatial_modulate_mojo


def _load_julia_fn() -> Callable[..., FloatArray]:
    import importlib

    importlib.import_module("juliacall")

    from ..experimental.accelerators.coupling._spatial_modulator_julia import (
        spatial_modulate_julia,
    )

    return spatial_modulate_julia


def _load_go_fn() -> Callable[..., FloatArray]:
    from ..experimental.accelerators.coupling._spatial_modulator_go import (
        _load_lib,
        spatial_modulate_go,
    )

    _load_lib()
    return spatial_modulate_go


_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}


def _load_backend(name: str) -> Callable[..., FloatArray]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
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


def _dispatch() -> Callable[..., FloatArray] | None:
    ordered = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: list[str] = []
    for backend in ordered:
        if backend not in seen:
            seen.append(backend)
    for backend in seen:
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _validate_backend_output(value: object, *, n: int) -> FloatArray:
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError("spatial backend output must be finite real-valued")
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial backend output must be finite real-valued") from exc
    if out.shape == (n * n,):
        out = out.reshape(n, n)
    if out.shape != (n, n):
        raise ValueError(f"spatial backend output shape {out.shape} must be ({n}, {n})")
    if not np.all(np.isfinite(out)):
        raise ValueError("spatial backend output must contain only finite values")
    if np.max(np.abs(np.diag(out))) > 1.0e-10:
        raise ValueError("spatial backend output diagonal must be zero")
    return np.ascontiguousarray(out, dtype=np.float64)


@dataclass(frozen=True)
class SpatialCouplingModulator:
    """Distance-decay modulator for base coupling matrices.

    Parameters are immutable so the same instance can be safely reused across
    simulator steps, benchmarks, and downstream MIF contract tests.
    """

    K_base: float
    distance_fn: DistanceFn | None = None
    decay_form: DecayForm = "inverse_plus_one"
    decay_exponent: float = 1.0
    decay_length_scale: float = 1.0
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        _validate_non_negative_scalar(self.K_base, name="K_base")
        _validate_decay_form(self.decay_form)
        _validate_scalar(self.decay_exponent, name="decay_exponent", positive=True)
        _validate_scalar(
            self.decay_length_scale,
            name="decay_length_scale",
            positive=True,
        )
        _validate_scalar(self.epsilon, name="epsilon", positive=True)
        if self.distance_fn is not None and not callable(self.distance_fn):
            raise ValueError("distance_fn must be callable when provided")

    def distance_matrix(self, positions: object) -> FloatArray:
        """Return the validated pairwise distance matrix for ``positions``."""

        positions64 = _validate_positions(positions)
        if self.distance_fn is None:
            return _pairwise_euclidean(positions64)
        left = positions64[:, np.newaxis, :]
        right = positions64[np.newaxis, :, :]
        return _validate_distance_matrix(
            self.distance_fn(left, right), n=positions64.shape[0]
        )

    def modulation_matrix(self, positions: object) -> FloatArray:
        """Return ``K_base * f(distance)`` with a zero self-coupling diagonal."""

        positions64 = _validate_positions(positions)
        return _python_modulation_matrix(
            positions64,
            k_base=float(self.K_base),
            decay_form=_validate_decay_form(self.decay_form),
            decay_exponent=float(self.decay_exponent),
            decay_length_scale=float(self.decay_length_scale),
            epsilon=float(self.epsilon),
            distance_fn=self.distance_fn,
        )

    def modulate(self, k_nm_base: object, positions: object) -> FloatArray:
        """Return the distance-modulated coupling matrix.

        ``k_nm_base`` must be square, finite, real-valued, and zero diagonal.
        The output preserves that zero self-coupling diagonal.
        """

        positions64 = _validate_positions(positions)
        base = _validate_knm_base(k_nm_base, expected_n=positions64.shape[0])
        form = _validate_decay_form(self.decay_form)
        if self.distance_fn is not None:
            out = base * self.modulation_matrix(positions64)
            np.fill_diagonal(out, 0.0)
            return np.ascontiguousarray(out, dtype=np.float64)
        backend_fn = _dispatch()
        if backend_fn is None:
            return _python_spatial_modulate(
                base.ravel(),
                positions64.ravel(),
                positions64.shape[0],
                positions64.shape[1],
                float(self.K_base),
                _DECAY_TO_CODE[form],
                float(self.decay_exponent),
                float(self.decay_length_scale),
                float(self.epsilon),
            ).reshape(positions64.shape[0], positions64.shape[0])
        return _validate_backend_output(
            backend_fn(
                np.ascontiguousarray(base.ravel(), dtype=np.float64),
                np.ascontiguousarray(positions64.ravel(), dtype=np.float64),
                int(positions64.shape[0]),
                int(positions64.shape[1]),
                float(self.K_base),
                int(_DECAY_TO_CODE[form]),
                float(self.decay_exponent),
                float(self.decay_length_scale),
                float(self.epsilon),
            ),
            n=positions64.shape[0],
        )

    def jacobian_positions(self, positions: object) -> FloatArray:
        """Analytical derivative of ``modulation_matrix`` with respect to positions.

        Returns an array with shape ``(n, n, n, dim)``. Entry
        ``J[i, j, a, d]`` is ``d M[i, j] / d positions[a, d]`` where
        ``M = modulation_matrix(positions)``. Custom distance functions are not
        differentiable through this closed-form path and fail closed.
        """

        if self.distance_fn is not None:
            raise ValueError(
                "jacobian_positions requires the default Euclidean distance"
            )
        positions64 = _validate_positions(positions)
        form = _validate_decay_form(self.decay_form)
        n, dim = positions64.shape
        jac = np.zeros((n, n, n, dim), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                delta = positions64[i] - positions64[j]
                distance = float(np.linalg.norm(delta))
                if form == "inverse_distance":
                    denom = (distance * distance + float(self.epsilon)) ** 1.5
                    grad_i = -float(self.K_base) * delta / denom
                elif distance <= float(self.epsilon):
                    grad_i = np.zeros(dim, dtype=np.float64)
                elif form == "inverse_plus_one":
                    grad_i = (
                        -float(self.K_base) * delta / (distance * (1.0 + distance) ** 2)
                    )
                elif form == "exponential":
                    weight = np.exp(-distance / float(self.decay_length_scale))
                    grad_i = (
                        -float(self.K_base)
                        * weight
                        * delta
                        / (float(self.decay_length_scale) * distance)
                    )
                else:
                    scaled = 1.0 + distance / float(self.decay_length_scale)
                    grad_i = (
                        -float(self.K_base)
                        * float(self.decay_exponent)
                        * scaled ** (-float(self.decay_exponent) - 1.0)
                        * delta
                        / (float(self.decay_length_scale) * distance)
                    )
                jac[i, j, i, :] = grad_i
                jac[i, j, j, :] = -grad_i
        return jac


def spatial_modulate(
    k_nm_base: object,
    positions: object,
    *,
    K_base: float = 1.0,
    decay_form: DecayForm = "inverse_plus_one",
    decay_exponent: float = 1.0,
    decay_length_scale: float = 1.0,
    epsilon: float = 1.0e-12,
) -> FloatArray:
    """Functional wrapper around :class:`SpatialCouplingModulator`."""

    return SpatialCouplingModulator(
        K_base=K_base,
        decay_form=decay_form,
        decay_exponent=decay_exponent,
        decay_length_scale=decay_length_scale,
        epsilon=epsilon,
    ).modulate(k_nm_base, positions)
