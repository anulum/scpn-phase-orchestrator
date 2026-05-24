# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition of coupling dynamics

"""Hodge decomposition with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

For each oscillator ``i``:

* ``gradient_i = Σ_j K_ij^sym · cos(θ_j − θ_i)`` (phase-locking flow
  from the symmetric part of ``K``)
* ``curl_i     = Σ_j K_ij^anti · cos(θ_j − θ_i)`` (rotational flow
  from the antisymmetric part)
* ``harmonic_i = total − gradient_i − curl_i`` (topological residual;
  a clean sym+anti split of ``K`` makes this identically zero up to
  float-precision noise)

Jiang et al. 2011, Math. Program. **127** (1):203–244.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
HodgeTuple: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]
HodgeBackend: TypeAlias = Callable[..., HodgeTuple]


__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "HodgeResult",
    "hodge_decomposition",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> HodgeBackend:
    from spo_kernel import hodge_decomposition_rust

    def _rust(
        knm_flat: FloatArray,
        phases: FloatArray,
        n: int,
    ) -> HodgeTuple:
        g, c, h = hodge_decomposition_rust(
            np.ascontiguousarray(knm_flat.ravel(), dtype=np.float64),
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            int(n),
        )
        return (
            np.asarray(g, dtype=np.float64),
            np.asarray(c, dtype=np.float64),
            np.asarray(h, dtype=np.float64),
        )

    return cast(
        "HodgeBackend",
        _rust,
    )


def _load_mojo_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.coupling._hodge_mojo import (
        _ensure_exe,
        hodge_decomposition_mojo,
    )

    _ensure_exe()
    return hodge_decomposition_mojo


def _load_julia_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401

    from ..experimental.accelerators.coupling._hodge_julia import (
        hodge_decomposition_julia,
    )

    return hodge_decomposition_julia


def _load_go_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.coupling._hodge_go import (
        _load_lib,
        hodge_decomposition_go,
    )

    _load_lib()
    return hodge_decomposition_go


_LOADERS: dict[
    str,
    Callable[[], HodgeBackend],
] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, HodgeBackend] = {}


def _load_backend(name: str) -> HodgeBackend:
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


def _dispatch() -> HodgeBackend | None:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError):
            continue
    return None


@dataclass
class HodgeResult:
    """Hodge decomposition of coupling flow into three orthogonal
    components.

    * ``gradient`` — phase-locking (conservative).
    * ``curl`` — circulation (antisymmetric).
    * ``harmonic`` — topological residual (numerical noise for a
      perfect sym/anti split of ``K``).
    """

    gradient: FloatArray
    curl: FloatArray
    harmonic: FloatArray


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
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


def _validate_coupling_matrix(value: object, *, expected_n: int) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("knm must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError("knm must not contain boolean values")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be a finite square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("knm must be a finite square matrix")
    if matrix.shape != (expected_n, expected_n):
        raise ValueError(
            f"knm shape {matrix.shape} does not match ({expected_n}, {expected_n})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("knm must contain only finite values")
    return matrix


def _python_decomposition(k: FloatArray, phases: FloatArray) -> HodgeTuple:
    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    cos_diff = np.cos(diff)
    k_sym = 0.5 * (k + k.T)
    k_anti = 0.5 * (k - k.T)
    total = np.sum(k * cos_diff, axis=1)
    gradient = np.sum(k_sym * cos_diff, axis=1)
    curl = np.sum(k_anti * cos_diff, axis=1)
    harmonic = total - gradient - curl
    return (
        gradient.astype(np.float64),
        curl.astype(np.float64),
        harmonic.astype(np.float64),
    )


def _normalise_backend_output(
    output: HodgeTuple,
    *,
    expected_n: int,
) -> HodgeTuple:
    gradient, curl, harmonic = (
        np.asarray(output[0], dtype=np.float64),
        np.asarray(output[1], dtype=np.float64),
        np.asarray(output[2], dtype=np.float64),
    )
    expected_shape = (expected_n,)
    if (
        gradient.shape != expected_shape
        or curl.shape != expected_shape
        or harmonic.shape != expected_shape
    ):
        raise ValueError("Hodge backend returned vectors with invalid shape")
    if not (
        np.all(np.isfinite(gradient))
        and np.all(np.isfinite(curl))
        and np.all(np.isfinite(harmonic))
    ):
        raise ValueError("Hodge backend returned non-finite values")
    return gradient, curl, harmonic


def _backend_matches_reference(
    backend_output: HodgeTuple,
    reference: HodgeTuple,
) -> bool:
    return all(
        np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
        for actual, expected in zip(backend_output, reference, strict=True)
    )


def hodge_decomposition(knm: FloatArray, phases: FloatArray) -> HodgeResult:
    """Decompose coupling dynamics into gradient / curl / harmonic
    per-oscillator contributions."""
    phases = _validate_phase_vector(phases, name="phases")
    n = int(phases.size)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return HodgeResult(gradient=empty, curl=empty, harmonic=empty)

    k = _validate_coupling_matrix(knm, expected_n=n)
    k_flat = np.ascontiguousarray(k.ravel())

    reference = _python_decomposition(k, phases)
    backend_fn = _dispatch()
    if backend_fn is not None:
        backend_output = _normalise_backend_output(
            backend_fn(k_flat, phases, n),
            expected_n=n,
        )
        if _backend_matches_reference(backend_output, reference):
            g, c, h = backend_output
            return HodgeResult(gradient=g, curl=c, harmonic=h)

    gradient, curl, harmonic = reference
    return HodgeResult(
        gradient=gradient,
        curl=curl,
        harmonic=harmonic,
    )
