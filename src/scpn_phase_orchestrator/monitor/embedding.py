# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding primitives + wrappers

"""Delay-embedding analysis with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Three compute primitives on the multi-language chain:

* :func:`delay_embed` — time-delay embedding matrix.
* :func:`mutual_information` — Fraser-Swinney 1986 average mutual
  information.
* :func:`nearest_neighbor_distances` — brute-force ``k=1`` kNN in
  the embedded space (consumed by FNN).

Two wrappers stay Python-side (they are control flow over the
primitives):

* :func:`optimal_delay` — first local minimum of MI (Fraser-Swinney).
* :func:`optimal_dimension` — Kennel-Brown-Abarbanel 1992 FNN.
* :func:`auto_embed` — convenience that chains ``optimal_delay``,
  ``optimal_dimension``, and :func:`delay_embed`.

The Rust backend exposes native ``optimal_delay_rust`` and
``optimal_dimension_rust`` entry points; when Rust is active those
wrappers use the native path for maximum throughput. The Python
fallback composes the primitives through the dispatcher.

MI and NN are exposed by Julia / Go / Mojo / Python only — Rust
does not expose standalone MI or kNN FFI; those slots dispatch to
the next available backend in the chain.
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
    "EmbeddingResult",
    "auto_embed",
    "delay_embed",
    "mutual_information",
    "nearest_neighbor_distances",
    "optimal_delay",
    "optimal_dimension",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        delay_embed_rust,
        optimal_delay_rust,
        optimal_dimension_rust,
    )

    def _de(signal: NDArray, delay: int, dimension: int) -> NDArray:
        flat = np.ascontiguousarray(signal, dtype=np.float64)
        return np.asarray(
            delay_embed_rust(flat, int(delay), int(dimension)),
            dtype=np.float64,
        )

    return {
        "de": _de,
        "mi": None,  # Rust has no standalone MI kernel.
        "nn": None,  # Rust has no standalone kNN kernel.
        "optimal_delay": optimal_delay_rust,
        "optimal_dimension": optimal_dimension_rust,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._embedding_mojo import (
        _ensure_exe,
        delay_embed_mojo,
        mutual_information_mojo,
        nearest_neighbor_distances_mojo,
    )

    _ensure_exe()
    return {
        "de": delay_embed_mojo,
        "mi": mutual_information_mojo,
        "nn": nearest_neighbor_distances_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._embedding_julia import (
        delay_embed_julia,
        mutual_information_julia,
        nearest_neighbor_distances_julia,
    )

    return {
        "de": delay_embed_julia,
        "mi": mutual_information_julia,
        "nn": nearest_neighbor_distances_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._embedding_go import (
        _load_lib,
        delay_embed_go,
        mutual_information_go,
        nearest_neighbor_distances_go,
    )

    _load_lib()
    return {
        "de": delay_embed_go,
        "mi": mutual_information_go,
        "nn": nearest_neighbor_distances_go,
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
    """Return the backend function or ``None`` to signal Python fallback.

    If the active backend lacks ``fn_name`` (e.g. Rust has no standalone
    MI kernel), we fall through to the next available backend in the
    chain rather than crashing or silently diverging.
    """
    if ACTIVE_BACKEND == "python":
        return None
    for name in AVAILABLE_BACKENDS:
        if name == "python":
            return None
        fn = _LOADERS[name]()[fn_name]
        if fn is not None:
            return fn
    return None


@dataclass
class EmbeddingResult:
    """Delay-embedding output."""

    trajectory: NDArray
    delay: int
    dimension: int
    T_effective: int


def delay_embed(
    signal: NDArray,
    delay: int,
    dimension: int,
) -> NDArray:
    """Time-delay embedding: ``v(t) = [x(t), x(t+τ), x(t+2τ), …]``."""
    s = np.asarray(signal, dtype=np.float64).ravel()
    t_eff = int(s.size) - (int(dimension) - 1) * int(delay)
    if t_eff <= 0:
        msg = (
            f"Signal too short (T={s.size}) for delay={delay}, "
            f"dimension={dimension}: need T > {(dimension - 1) * delay}"
        )
        raise ValueError(msg)

    backend_fn = _dispatch("de")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray, int, int], NDArray]", backend_fn)
        return np.asarray(
            fn(s, int(delay), int(dimension)),
            dtype=np.float64,
        ).reshape(t_eff, int(dimension))

    indices = np.arange(int(dimension)) * int(delay)
    rows = np.arange(t_eff)[:, np.newaxis] + indices[np.newaxis, :]
    return cast("NDArray", s[rows])


def mutual_information(
    signal: NDArray,
    lag: int,
    n_bins: int = 32,
) -> float:
    """Fraser-Swinney 1986 average mutual information at ``lag``."""
    s = np.asarray(signal, dtype=np.float64).ravel()
    if s.size - int(lag) <= 0:
        return 0.0

    backend_fn = _dispatch("mi")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray, int, int], float]", backend_fn)
        return float(fn(s, int(lag), int(n_bins)))

    t_total = s.size - int(lag)
    x = s[:t_total]
    y = s[int(lag) : int(lag) + t_total]
    hist_xy, _, _ = np.histogram2d(x, y, bins=int(n_bins))
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)
    total = hist_xy.sum()
    if total <= 0:
        return 0.0
    p_xy = hist_xy / total
    p_x = hist_x / total
    p_y = hist_y / total
    mi = 0.0
    for i in range(int(n_bins)):
        for j in range(int(n_bins)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(mi)


def nearest_neighbor_distances(
    embedded: NDArray,
) -> tuple[NDArray, NDArray]:
    """Brute-force ``k = 1`` kNN on the rows of ``embedded``."""
    e = np.atleast_2d(embedded)
    t, m = int(e.shape[0]), int(e.shape[1])
    if t == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    backend_fn = _dispatch("nn")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, int, int], tuple[NDArray, NDArray]]",
            backend_fn,
        )
        dist, idx = fn(e, t, m)
        return (
            np.asarray(dist, dtype=np.float64),
            np.asarray(idx, dtype=np.int64),
        )

    nn_dist = np.full(t, np.inf)
    nn_idx = np.zeros(t, dtype=np.int64)
    for i in range(t):
        diffs = e - e[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        j = int(np.argmin(dists))
        nn_dist[i] = dists[j]
        nn_idx[i] = j
    return nn_dist, nn_idx


def optimal_delay(
    signal: NDArray,
    max_lag: int = 100,
    n_bins: int = 32,
) -> int:
    """First local minimum of :func:`mutual_information` vs ``lag``."""
    s = np.asarray(signal, dtype=np.float64).ravel()

    if ACTIVE_BACKEND == "rust":
        fn = cast(
            "Callable[[NDArray, int, int], int]",
            _LOADERS["rust"]()["optimal_delay"],
        )
        return int(fn(s, int(max_lag), int(n_bins)))

    max_lag = min(int(max_lag), s.size // 2)
    mi_values = np.array([mutual_information(s, lag, n_bins) for lag in range(max_lag)])
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            return i
    return 1


def optimal_dimension(
    signal: NDArray,
    delay: int,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> int:
    """Kennel-Brown-Abarbanel 1992 FNN to select embedding dimension."""
    s = np.asarray(signal, dtype=np.float64).ravel()
    sigma = float(np.std(s))
    if sigma == 0:
        return 1

    if ACTIVE_BACKEND == "rust":
        fn = cast(
            "Callable[..., int]",
            _LOADERS["rust"]()["optimal_dimension"],
        )
        return int(
            fn(s, int(delay), int(max_dim), float(rtol), float(atol)),
        )

    for m in range(1, int(max_dim) + 1):
        t_next = s.size - m * int(delay)
        if t_next <= 1:
            return m
        emb_m = delay_embed(s, int(delay), m)
        t_m = emb_m.shape[0]
        nn_dist, nn_idx = nearest_neighbor_distances(emb_m)

        n_false = 0
        n_valid = 0
        for i in range(t_m):
            j = int(nn_idx[i])
            d = nn_dist[i]
            if d == 0 or not np.isfinite(d):
                continue
            i_next = i + m * int(delay)
            j_next = j + m * int(delay)
            if i_next >= s.size or j_next >= s.size:
                continue
            n_valid += 1
            extra = abs(s[i_next] - s[j_next])
            if extra / d > rtol:
                n_false += 1
                continue
            new_dist = (d * d + extra * extra) ** 0.5
            if new_dist / sigma > atol:
                n_false += 1
        fnn_frac = n_false / n_valid if n_valid > 0 else 0.0
        if fnn_frac < 0.01:
            return m
    return int(max_dim)


def auto_embed(
    signal: NDArray,
    max_lag: int = 100,
    max_dim: int = 10,
) -> EmbeddingResult:
    """``optimal_delay`` ∘ ``optimal_dimension`` ∘ ``delay_embed``."""
    tau = optimal_delay(signal, max_lag)
    m = optimal_dimension(signal, tau, max_dim)
    traj = delay_embed(signal, tau, m)
    return EmbeddingResult(
        trajectory=traj,
        delay=tau,
        dimension=m,
        T_effective=int(traj.shape[0]),
    )
