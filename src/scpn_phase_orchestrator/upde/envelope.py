# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Amplitude envelope solver

"""Sliding-window RMS envelope and modulation-depth statistic.

Exposes a 5-backend fallback chain. ``AVAILABLE_BACKENDS`` keeps the canonical
fallback order; ``ACTIVE_BACKEND`` is chosen by a small hot-path probe so slow
external wrappers do not displace the faster local path.

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
from time import perf_counter
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._envelope_validation import (
    _contains_numeric_string_alias,
    _is_numeric_string_alias,
    validate_envelope_modulation_output,
    validate_extract_envelope_output,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "EnvelopeState",
    "envelope_modulation_depth",
    "extract_envelope",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")
FloatArray: TypeAlias = NDArray[np.float64]


def _load_rust_fns() -> dict[str, object]:
    """Load the Rust envelope backend callables."""
    from spo_kernel import (
        envelope_modulation_depth_rust,
        extract_envelope_rust,
    )

    def _rust_extract(amps: FloatArray, window: int) -> FloatArray:
        """Call the Rust sliding-window RMS envelope kernel."""
        return np.asarray(
            extract_envelope_rust(
                np.ascontiguousarray(amps.ravel(), dtype=np.float64),
                int(window),
            ),
            dtype=np.float64,
        )

    def _rust_mod(env: FloatArray) -> float:
        """Call the Rust modulation-depth kernel."""
        return float(
            envelope_modulation_depth_rust(
                np.ascontiguousarray(env.ravel(), dtype=np.float64),
            )
        )

    return {"extract": _rust_extract, "mod": _rust_mod}


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    """Load the Mojo envelope backend callables."""
    from ..experimental.accelerators.upde._envelope_mojo import (
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
    """Load the Julia envelope backend callables."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._envelope_julia import (
        envelope_modulation_depth_julia,
        extract_envelope_julia,
    )

    return {
        "extract": extract_envelope_julia,
        "mod": envelope_modulation_depth_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    """Load the Go envelope backend callables."""
    from ..experimental.accelerators.upde._envelope_go import (
        _load_lib,
        envelope_modulation_depth_go,
        extract_envelope_go,
    )

    _load_lib()
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
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    """Load and cache the named backend callables."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _extract_1d_python(amps: FloatArray, window: int) -> FloatArray:
    """Return the reference 1-D RMS envelope (NumPy floor)."""
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


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    active = min(available, key=_extract_probe_seconds)
    return active, available


def _extract_probe_seconds(name: str) -> float:
    """Return the per-backend probe timings for envelope extraction."""
    amps = np.linspace(0.01, 1.0, 256, dtype=np.float64)
    start = perf_counter()
    try:
        if name == "python":
            _extract_1d_python(amps, 10)
        else:
            fn = cast(
                "Callable[[FloatArray, int], FloatArray]",
                _load_backend(name)["extract"],
            )
            fn(amps, 10)
    except (ImportError, RuntimeError, OSError, KeyError):
        return float("inf")
    return perf_counter() - start


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object | None:
    """Return the fastest available backend callables, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return None
        try:
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is None:
            continue
        return fn
    return None


def extract_envelope(
    amplitudes_history: FloatArray,
    window: int = 10,
) -> FloatArray:
    """Sliding-window RMS envelope.

    Parameters
    ----------
    amplitudes_history : FloatArray
        ``(T,)`` or ``(T, N)`` amplitude time series.
    window : int
        RMS window length in samples.

    Returns
    -------
    FloatArray
        Same shape as input; the first ``window − 1`` entries are front-padded with the
        first valid RMS value.

    Raises
    ------
    ValueError
        If ``window`` is not a positive integer no larger than the history.
    """
    if _contains_numeric_string_alias(amplitudes_history):
        raise ValueError("amplitudes_history must not contain numeric-string aliases")
    if _is_numeric_string_alias(window):
        raise ValueError("window must not be a numeric-string alias")
    amplitudes = np.asarray(amplitudes_history, dtype=np.float64)
    if amplitudes.size == 0:
        return amplitudes.copy()
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes_history must contain only finite values")

    if amplitudes.ndim == 1:
        if window >= amplitudes.size:
            return _extract_1d_python(amplitudes, int(window))
        backend_fn = _dispatch("extract")
        if backend_fn is not None:
            fn = cast("Callable[[FloatArray, int], FloatArray]", backend_fn)
            return validate_extract_envelope_output(
                fn(amplitudes, int(window)),
                n=int(amplitudes.size),
            )
        return _extract_1d_python(amplitudes, int(window))

    if amplitudes.ndim != 2:
        raise ValueError("amplitudes_history must be 1-D or 2-D")

    # 2-D path stays pure NumPy.
    sq = amplitudes**2
    if window >= sq.shape[0]:
        rms = np.sqrt(np.mean(sq, axis=0))
        return np.tile(rms, (sq.shape[0], 1))
    cs = np.cumsum(sq, axis=0)
    cs = np.vstack([np.zeros((1, sq.shape[1]), dtype=np.float64), cs])
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    first = rms[0] if rms.shape[0] > 0 else np.zeros(sq.shape[1])
    return np.vstack([np.tile(first, (window - 1, 1)), rms])


def envelope_modulation_depth(envelope: FloatArray) -> float:
    """Modulation depth ``(max − min) / (max + min) ∈ [0, 1]``.

    Returns ``0.0`` for empty or non-positive envelopes.

    Parameters
    ----------
    envelope : FloatArray
        An amplitude-envelope time series, shape ``(T,)``.

    Returns
    -------
    float
        The modulation depth ``(max − min) / (max + min)`` in ``[0, 1]``.
    """
    if _contains_numeric_string_alias(envelope):
        raise ValueError("envelope must not contain numeric-string aliases")
    if envelope.size == 0:
        return 0.0
    backend_fn = _dispatch("mod")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray], float]", backend_fn)
        return validate_envelope_modulation_output(fn(envelope))
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
