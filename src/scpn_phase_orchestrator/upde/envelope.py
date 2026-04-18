# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Amplitude envelope solver

"""Sliding-window RMS envelope + modulation-depth statistic with a
5-backend fallback chain per ``feedback_module_standard_attnres.md``.

The sliding-window RMS uses the O(T) cumulative-sum form: compute
``cs[i] = Σ_{k < i} x_k²``, then
``rms[i] = sqrt((cs[i+w] − cs[i]) / w)`` for valid indices, with a
front-pad of the first valid value. The 1-D path is on the
5-backend chain; the 2-D ``(T, N)`` batched path stays pure NumPy
because the Rust FFI is 1-D-only and the vectorised NumPy form is
already near-optimal at realistic ``N``.
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
    "EnvelopeState",
    "envelope_modulation_depth",
    "extract_envelope",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        envelope_modulation_depth_rust,
        extract_envelope_rust,
    )

    def _rust_extract(amps: NDArray, window: int) -> NDArray:
        return np.asarray(
            extract_envelope_rust(
                np.ascontiguousarray(amps.ravel(), dtype=np.float64),
                int(window),
            ),
            dtype=np.float64,
        )

    def _rust_mod(env: NDArray) -> float:
        return float(
            envelope_modulation_depth_rust(
                np.ascontiguousarray(env.ravel(), dtype=np.float64),
            )
        )

    return {"extract": _rust_extract, "mod": _rust_mod}


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._envelope_mojo import (
        _ensure_exe,
        envelope_modulation_depth_mojo,
        extract_envelope_mojo,
    )

    _ensure_exe()
    return {
        "extract": extract_envelope_mojo,
        "mod": envelope_modulation_depth_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._envelope_julia import (
        envelope_modulation_depth_julia,
        extract_envelope_julia,
    )

    return {
        "extract": extract_envelope_julia,
        "mod": envelope_modulation_depth_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._envelope_go import (
        envelope_modulation_depth_go,
        extract_envelope_go,
    )

    return {
        "extract": extract_envelope_go,
        "mod": envelope_modulation_depth_go,
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


def _extract_1d_python(amps: NDArray, window: int) -> NDArray:
    t = amps.size
    sq = amps.astype(np.float64) ** 2
    # Window exceeds series — fall back to a single constant RMS
    # over the whole trace (matches the non-Python backends).
    if window > t:
        v = float(np.sqrt(sq.sum() / t)) if t > 0 else 0.0
        return np.full(t, v, dtype=np.float64)
    cs = np.cumsum(sq)
    cs = np.insert(cs, 0, 0.0)
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    pad = np.full(window - 1, rms[0] if rms.size > 0 else 0.0)
    return np.concatenate([pad, rms])


def extract_envelope(
    amplitudes_history: NDArray, window: int = 10,
) -> NDArray:
    """Sliding-window RMS envelope.

    Args:
        amplitudes_history: ``(T,)`` or ``(T, N)`` amplitude time
            series.
        window: RMS window length in samples.

    Returns:
        Same shape as input; the first ``window − 1`` entries are
        front-padded with the first valid RMS value.
    """
    if amplitudes_history.size == 0:
        return amplitudes_history.copy()
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if amplitudes_history.ndim == 1:
        backend_fn = _dispatch("extract")
        if backend_fn is not None:
            fn = cast(
                "Callable[[NDArray, int], NDArray]", backend_fn,
            )
            return np.asarray(
                fn(amplitudes_history, int(window)),
                dtype=np.float64,
            )
        return _extract_1d_python(amplitudes_history, int(window))

    # 2-D path stays pure NumPy.
    sq = amplitudes_history.astype(np.float64) ** 2
    cs = np.cumsum(sq, axis=0)
    cs = np.vstack([np.zeros((1, sq.shape[1]), dtype=np.float64), cs])
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    first = rms[0] if rms.shape[0] > 0 else np.zeros(sq.shape[1])
    return np.vstack([np.tile(first, (window - 1, 1)), rms])


def envelope_modulation_depth(envelope: NDArray) -> float:
    """Modulation depth ``(max − min) / (max + min) ∈ [0, 1]``.

    Returns ``0.0`` for empty or non-positive envelopes.
    """
    if envelope.size == 0:
        return 0.0
    backend_fn = _dispatch("mod")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray], float]", backend_fn)
        return float(fn(envelope))
    flat = envelope.ravel()
    vmax = float(np.max(flat))
    vmin = float(np.min(flat))
    denom = vmax + vmin
    if denom <= 0.0:
        return 0.0
    return float((vmax - vmin) / denom)


@dataclass(frozen=True)
class EnvelopeState:
    """Snapshot of amplitude envelope statistics."""

    mean_amplitude: float
    amplitude_spread: float
    modulation_depth: float
    subcritical_count: int
