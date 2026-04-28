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
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "HodgeResult",
    "hodge_decomposition",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., tuple[NDArray, NDArray, NDArray]]:
    from spo_kernel import hodge_decomposition_rust

    def _rust(
        knm_flat: NDArray,
        phases: NDArray,
        n: int,
    ) -> tuple[NDArray, NDArray, NDArray]:
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
        "Callable[..., tuple[NDArray, NDArray, NDArray]]",
        _rust,
    )


def _load_mojo_fn() -> Callable[..., tuple[NDArray, NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.coupling._hodge_mojo import (
        _ensure_exe,
        hodge_decomposition_mojo,
    )

    _ensure_exe()
    return hodge_decomposition_mojo


def _load_julia_fn() -> Callable[..., tuple[NDArray, NDArray, NDArray]]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.coupling._hodge_julia import (
        hodge_decomposition_julia,
    )

    return hodge_decomposition_julia


def _load_go_fn() -> Callable[..., tuple[NDArray, NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.coupling._hodge_go import (
        _load_lib,
        hodge_decomposition_go,
    )

    _load_lib()
    return hodge_decomposition_go


_LOADERS: dict[
    str,
    Callable[[], Callable[..., tuple[NDArray, NDArray, NDArray]]],
] = {
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


def _dispatch() -> Callable[..., tuple[NDArray, NDArray, NDArray]] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


@dataclass
class HodgeResult:
    """Hodge decomposition of coupling flow into three orthogonal
    components.

    * ``gradient`` — phase-locking (conservative).
    * ``curl`` — circulation (antisymmetric).
    * ``harmonic`` — topological residual (numerical noise for a
      perfect sym/anti split of ``K``).
    """

    gradient: NDArray
    curl: NDArray
    harmonic: NDArray


def hodge_decomposition(knm: NDArray, phases: NDArray) -> HodgeResult:
    """Decompose coupling dynamics into gradient / curl / harmonic
    per-oscillator contributions."""
    phases = np.asarray(phases, dtype=np.float64)
    n = int(phases.size)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return HodgeResult(gradient=empty, curl=empty, harmonic=empty)

    k = np.asarray(knm, dtype=np.float64)
    k_flat = np.ascontiguousarray(k.ravel())

    backend_fn = _dispatch()
    if backend_fn is not None:
        g, c, h = backend_fn(k_flat, phases, n)
        return HodgeResult(gradient=g, curl=c, harmonic=h)

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    cos_diff = np.cos(diff)
    k_sym = 0.5 * (k + k.T)
    k_anti = 0.5 * (k - k.T)
    total = np.sum(k * cos_diff, axis=1)
    gradient = np.sum(k_sym * cos_diff, axis=1)
    curl = np.sum(k_anti * cos_diff, axis=1)
    harmonic = total - gradient - curl
    return HodgeResult(
        gradient=gradient.astype(np.float64),
        curl=curl.astype(np.float64),
        harmonic=harmonic.astype(np.float64),
    )
