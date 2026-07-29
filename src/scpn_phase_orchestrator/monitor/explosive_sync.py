# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Explosive-synchronisation early-warning monitor

"""Ordinal-pattern transition-entropy early warning for explosive sync.

A first-order (explosive) synchronisation transition — the abrupt, hysteretic
collapse to coherence behind power-grid blackouts and seizure onset — is
preceded by a regularisation of each oscillator's local dynamics. That
regularisation shows up as a drop in the ordinal-pattern transition entropy
of the node's observable *before* the macroscopic order parameter jumps,
where variance / autocorrelation critical-slowing-down indicators are weak.

``explosive_sync_warning`` slides a window across a multi-node signal array,
computes the per-node transition entropy (``monitor/opt_entropy.py``,
five-backend accelerated), aggregates it into a coherence-regularisation
index, and raises a fail-early alarm when the index drops a robust
(median / MAD) margin below its leading baseline. The monitor is passive:
it reads observables and emits a warning record; it never actuates.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — early-warning signals for
  critical transitions (the slowing-down framework this complements).
* Bandt & Pompe 2002, *Phys. Rev. Lett.* 88, 174102 — permutation entropy.
* Gómez-Gardeñes, Gómez, Arenas & Moreno 2011, *Phys. Rev. Lett.* 106,
  128701 — explosive synchronisation as a first-order transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .opt_entropy import (
    DEFAULT_DELAY,
    DEFAULT_DIMENSION,
    _validate_ordinal_params,
    transition_entropy,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Consistent-estimator scale factor making the median absolute deviation an
# unbiased estimate of the standard deviation for a normal distribution.
_MAD_TO_STD = 1.4826
_SCALE_FLOOR = 1.0e-12

__all__ = [
    "ExplosiveSyncWarning",
    "explosive_sync_warning",
]


@dataclass(frozen=True)
class ExplosiveSyncWarning:
    """Result of an explosive-synchronisation early-warning sweep.

    Attributes
    ----------
    window_starts : IntArray
        First sample index of each analysis window, shape ``(W,)``.
    entropy_index : FloatArray
        Mean transition entropy across nodes per window, shape ``(W,)``;
        the headline coherence-regularisation index.
    per_node_entropy : FloatArray
        Per-node transition entropy, shape ``(W, N)`` — the local field.
    robust_z : FloatArray
        Median / MAD robust z-score of ``entropy_index`` against the
        baseline, shape ``(W,)``. Strongly negative means a sharp drop.
    relative_drop : FloatArray
        Fractional drop of ``entropy_index`` below the baseline median,
        shape ``(W,)``.
    baseline_median : float
        Median entropy index over the leading baseline windows.
    baseline_scale : float
        Robust scale (``1.4826 × MAD``) of the baseline windows.
    n_baseline_windows : int
        Number of leading windows used to fit the baseline.
    warning_triggered : bool
        Whether a sustained drop crossed both the z and relative-drop gates.
    warning_window : int | None
        Index of the first window of the triggering run, or ``None``.
    warning_sample : int | None
        Sample index ``window_starts[warning_window]``, or ``None``.
    dimension, delay, window, step : int
        Echoed analysis parameters.
    z_threshold, drop_threshold : float
        Echoed alarm gates.
    persistence : int
        Echoed number of consecutive breaching windows required to alarm.
    """

    window_starts: IntArray = field(repr=False)
    entropy_index: FloatArray = field(repr=False)
    per_node_entropy: FloatArray = field(repr=False)
    robust_z: FloatArray = field(repr=False)
    relative_drop: FloatArray = field(repr=False)
    baseline_median: float
    baseline_scale: float
    n_baseline_windows: int
    warning_triggered: bool
    warning_window: int | None
    warning_sample: int | None
    dimension: int
    delay: int
    window: int
    step: int
    z_threshold: float
    drop_threshold: float
    persistence: int

    def __post_init__(self) -> None:
        starts = _validate_window_starts(self.window_starts)
        if starts[0] != 0 or np.any(starts < 0) or np.any(np.diff(starts) <= 0):
            raise ValueError(
                "window_starts must start at zero and be strictly increasing"
            )

        entropy = _validate_result_array(self.entropy_index, "entropy_index", ndim=1)
        per_node = _validate_result_array(
            self.per_node_entropy,
            "per_node_entropy",
            ndim=2,
        )
        robust_z = _validate_result_array(self.robust_z, "robust_z", ndim=1)
        relative_drop = _validate_result_array(
            self.relative_drop,
            "relative_drop",
            ndim=1,
        )
        n_windows = starts.size
        if (
            entropy.shape != (n_windows,)
            or robust_z.shape != (n_windows,)
            or relative_drop.shape != (n_windows,)
            or per_node.shape[0] != n_windows
        ):
            raise ValueError("warning arrays must share one non-empty window axis")
        tolerance = 1e-12
        if np.any(entropy < -tolerance) or np.any(entropy > 1.0 + tolerance):
            raise ValueError("entropy_index must lie in [0, 1]")
        if np.any(per_node < -tolerance) or np.any(per_node > 1.0 + tolerance):
            raise ValueError("per_node_entropy must lie in [0, 1]")
        entropy = np.clip(entropy, 0.0, 1.0)
        per_node = np.clip(per_node, 0.0, 1.0)
        if not np.allclose(
            entropy,
            per_node.mean(axis=1),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("entropy_index must equal the per-node mean")

        baseline_median = _validate_unit_real(
            self.baseline_median,
            "baseline_median",
        )
        baseline_scale = _validate_non_negative_real(
            self.baseline_scale,
            "baseline_scale",
        )
        n_baseline = _validate_positive_int(
            self.n_baseline_windows,
            "n_baseline_windows",
        )
        if n_baseline > n_windows:
            raise ValueError("n_baseline_windows exceeds the window count")
        expected_median = float(np.median(entropy[:n_baseline]))
        expected_scale = _MAD_TO_STD * float(
            np.median(np.abs(entropy[:n_baseline] - expected_median))
        )
        if not np.isclose(baseline_median, expected_median, rtol=0.0, atol=tolerance):
            raise ValueError("baseline_median does not match baseline evidence")
        if not np.isclose(baseline_scale, expected_scale, rtol=0.0, atol=tolerance):
            raise ValueError("baseline_scale does not match baseline evidence")

        expected_z = (entropy - expected_median) / max(expected_scale, _SCALE_FLOOR)
        expected_drop = (
            (expected_median - entropy) / expected_median
            if expected_median > 0.0
            else np.zeros_like(entropy)
        )
        if not np.allclose(robust_z, expected_z, rtol=0.0, atol=tolerance):
            raise ValueError("robust_z does not match baseline evidence")
        if not np.allclose(
            relative_drop,
            expected_drop,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("relative_drop does not match baseline evidence")

        dimension, delay = _validate_ordinal_params(self.dimension, self.delay)
        window = _validate_positive_int(self.window, "window")
        step = _validate_positive_int(self.step, "step")
        persistence = _validate_positive_int(self.persistence, "persistence")
        z_threshold = _validate_non_negative_real(self.z_threshold, "z_threshold")
        drop_threshold = _validate_non_negative_real(
            self.drop_threshold,
            "drop_threshold",
        )
        if starts.size > 1 and not np.all(np.diff(starts) == step):
            raise ValueError("window_starts must advance by step")
        if not isinstance(self.warning_triggered, (bool, np.bool_)):
            raise ValueError("warning_triggered must be a boolean")
        warning_window = _validate_optional_index(self.warning_window, "warning_window")
        warning_sample = _validate_optional_index(self.warning_sample, "warning_sample")
        breaches = (
            (np.arange(n_windows) >= n_baseline)
            & (robust_z <= -z_threshold)
            & (relative_drop >= drop_threshold)
        )
        expected_window = _first_sustained_breach(breaches, persistence)
        expected_sample = (
            int(starts[expected_window]) if expected_window is not None else None
        )
        if (
            bool(self.warning_triggered) != (expected_window is not None)
            or warning_window != expected_window
            or warning_sample != expected_sample
        ):
            raise ValueError("warning fields do not match threshold evidence")

        arrays = (starts, entropy, per_node, robust_z, relative_drop)
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "window_starts", starts)
        object.__setattr__(self, "entropy_index", entropy)
        object.__setattr__(self, "per_node_entropy", per_node)
        object.__setattr__(self, "robust_z", robust_z)
        object.__setattr__(self, "relative_drop", relative_drop)
        object.__setattr__(self, "baseline_median", baseline_median)
        object.__setattr__(self, "baseline_scale", baseline_scale)
        object.__setattr__(self, "n_baseline_windows", n_baseline)
        object.__setattr__(self, "warning_triggered", bool(self.warning_triggered))
        object.__setattr__(self, "warning_window", warning_window)
        object.__setattr__(self, "warning_sample", warning_sample)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "z_threshold", z_threshold)
        object.__setattr__(self, "drop_threshold", drop_threshold)
        object.__setattr__(self, "persistence", persistence)

    def summary(self) -> dict[str, float | int | bool | None]:
        """Return a flat scalar summary for logging or metric export.

        Returns
        -------
        dict[str, float | int | bool | None]
            Window/baseline counts, the baseline fit, the entropy-index and
            robust-z extremes, the maximum relative drop, and the alarm verdict.
        """
        return {
            "n_windows": int(self.entropy_index.shape[0]),
            "n_baseline_windows": self.n_baseline_windows,
            "baseline_median": self.baseline_median,
            "baseline_scale": self.baseline_scale,
            "min_entropy_index": float(self.entropy_index.min())
            if self.entropy_index.size
            else 0.0,
            "min_robust_z": float(self.robust_z.min()) if self.robust_z.size else 0.0,
            "max_relative_drop": float(self.relative_drop.max())
            if self.relative_drop.size
            else 0.0,
            "warning_triggered": self.warning_triggered,
            "warning_window": self.warning_window,
            "warning_sample": self.warning_sample,
        }


def _validate_signals(signals: object) -> FloatArray:
    """Return the signals as a validated 2-D finite array, else raise."""
    try:
        raw = np.asarray(signals)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals must be a real float array") from exc
    object_values = tuple(raw.flat) if raw.dtype.kind == "O" else ()
    if raw.dtype.kind == "b" or any(
        isinstance(value, (bool, np.bool_)) for value in object_values
    ):
        raise ValueError("signals must not contain boolean values")
    if raw.dtype.kind == "c" or any(
        isinstance(value, (complex, np.complexfloating)) for value in object_values
    ):
        raise ValueError("signals must contain real-valued samples")
    if raw.dtype.kind not in "iufO" or any(
        isinstance(value, (str, bytes, np.str_, np.bytes_))
        or not isinstance(value, Real)
        for value in object_values
    ):
        raise ValueError("signals must be a real float array")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals must be a real float array") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"signals shape {raw.shape} must be one- or two-dimensional")
    if array.shape[0] == 0:
        raise ValueError("signals must contain at least one node")
    if array.shape[1] == 0:
        raise ValueError("signals must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("signals must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_result_array(value: object, name: str, *, ndim: int) -> FloatArray:
    """Return a copied non-empty finite real result array."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if raw.dtype.kind == "O":
        if any(
            isinstance(item, (bool, np.bool_, complex, str, bytes, np.str_, np.bytes_))
            or not isinstance(item, Real)
            for item in raw.flat
        ):
            raise ValueError(f"{name} must be a finite real array")
    elif raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real array")
    result = np.asarray(raw, dtype=np.float64).copy()
    if result.ndim != ndim or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite {ndim}-D array")
    return np.ascontiguousarray(result)


def _validate_window_starts(value: object) -> IntArray:
    """Return a copied non-empty signed-64 integer window vector."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("window_starts must be an integer vector") from exc
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError("window_starts must be an integer vector")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("window_starts must fit signed 64-bit integers")
    return np.ascontiguousarray(raw, dtype=np.int64).copy()


def _validate_optional_index(value: object, name: str) -> int | None:
    """Return an optional non-negative integer index."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer or None")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return result


def _validate_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _validate_unit_fraction(value: object, name: str) -> float:
    """Return ``value`` as a fraction in [0, 1], else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in (0, 1), got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must lie in the open interval (0, 1), got {result}")
    return result


def _validate_non_negative_real(value: object, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {result}")
    return result


def _validate_unit_real(value: object, name: str) -> float:
    """Return ``value`` as a finite real in [0, 1], else raise."""
    result = _validate_non_negative_real(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {result}")
    return result


def explosive_sync_warning(
    signals: FloatArray,
    *,
    dimension: int = DEFAULT_DIMENSION,
    delay: int = DEFAULT_DELAY,
    window: int = 128,
    step: int = 16,
    baseline_fraction: float = 0.25,
    min_baseline_windows: int = 3,
    z_threshold: float = 3.0,
    drop_threshold: float = 0.1,
    persistence: int = 2,
) -> ExplosiveSyncWarning:
    """Sweep a multi-node signal for an explosive-synchronisation warning.

    Parameters
    ----------
    signals : FloatArray
        Per-node scalar observables (e.g. phase velocity), shape ``(N, T)``;
        a one-dimensional array is treated as a single node.
    dimension, delay : int
        Ordinal-pattern embedding dimension ``D`` in ``[2, 7]`` and positive
        delay ``τ`` passed to ``transition_entropy``.
    window : int
        Analysis window length in samples; must admit at least two ordinal
        transitions, i.e. ``window ≥ (D − 1)·τ + 3``.
    step : int
        Hop between consecutive window starts in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit the baseline, in ``(0, 1)``.
    min_baseline_windows : int
        Lower bound on the number of baseline windows.
    z_threshold : float
        Robust z-score magnitude below which a window breaches the gate.
    drop_threshold : float
        Minimum fractional drop below the baseline median to breach the gate.
    persistence : int
        Number of consecutive breaching windows required to raise the alarm.

    Returns
    -------
    ExplosiveSyncWarning
        The per-window entropy field, baseline fit, and the alarm decision.

    Raises
    ------
    ValueError
        If the inputs are malformed or the window does not fit the series.
    """
    array = _validate_signals(signals)
    dimension, delay = _validate_ordinal_params(dimension, delay)
    window = _validate_positive_int(window, "window")
    step = _validate_positive_int(step, "step")
    min_baseline_windows = _validate_positive_int(
        min_baseline_windows, "min_baseline_windows"
    )
    baseline_fraction = _validate_unit_fraction(baseline_fraction, "baseline_fraction")
    z_threshold = _validate_non_negative_real(z_threshold, "z_threshold")
    drop_threshold = _validate_non_negative_real(drop_threshold, "drop_threshold")
    persistence = _validate_positive_int(persistence, "persistence")

    n_nodes, n_samples = int(array.shape[0]), int(array.shape[1])
    min_window = (dimension - 1) * delay + 3
    if window < min_window:
        raise ValueError(
            f"window {window} must be at least (D - 1)·τ + 3 = {min_window} "
            "to admit two ordinal transitions"
        )
    if window > n_samples:
        raise ValueError(f"window {window} exceeds the series length {n_samples}")

    starts = list(range(0, n_samples - window + 1, step))
    n_windows = len(starts)
    window_starts = np.asarray(starts, dtype=np.int64)
    per_node = np.empty((n_windows, n_nodes), dtype=np.float64)
    for w, start in enumerate(starts):
        segment = array[:, start : start + window]
        for node in range(n_nodes):
            per_node[w, node] = transition_entropy(segment[node], dimension, delay)
    entropy_index = per_node.mean(axis=1)

    n_baseline = max(
        min_baseline_windows,
        int(np.ceil(baseline_fraction * n_windows)),
    )
    n_baseline = min(n_baseline, n_windows)
    baseline = entropy_index[:n_baseline]
    baseline_median = float(np.median(baseline)) if baseline.size else 0.0
    mad = float(np.median(np.abs(baseline - baseline_median))) if baseline.size else 0.0
    baseline_scale = _MAD_TO_STD * mad
    guarded_scale = max(baseline_scale, _SCALE_FLOOR)

    robust_z = (entropy_index - baseline_median) / guarded_scale
    if baseline_median > 0.0:
        relative_drop = (baseline_median - entropy_index) / baseline_median
    else:
        relative_drop = np.zeros_like(entropy_index)

    breaches = (
        (np.arange(n_windows) >= n_baseline)
        & (robust_z <= -z_threshold)
        & (relative_drop >= drop_threshold)
    )
    warning_window = _first_sustained_breach(breaches, persistence)
    warning_triggered = warning_window is not None
    warning_sample = (
        int(window_starts[warning_window]) if warning_window is not None else None
    )

    return ExplosiveSyncWarning(
        window_starts=window_starts,
        entropy_index=np.ascontiguousarray(entropy_index, dtype=np.float64),
        per_node_entropy=np.ascontiguousarray(per_node, dtype=np.float64),
        robust_z=np.ascontiguousarray(robust_z, dtype=np.float64),
        relative_drop=np.ascontiguousarray(relative_drop, dtype=np.float64),
        baseline_median=baseline_median,
        baseline_scale=baseline_scale,
        n_baseline_windows=n_baseline,
        warning_triggered=warning_triggered,
        warning_window=warning_window,
        warning_sample=warning_sample,
        dimension=dimension,
        delay=delay,
        window=window,
        step=step,
        z_threshold=z_threshold,
        drop_threshold=drop_threshold,
        persistence=persistence,
    )


def _first_sustained_breach(
    breaches: NDArray[np.bool_], persistence: int
) -> int | None:
    """Return the start index of the first ``persistence``-long breach run."""
    run = 0
    for index in range(int(breaches.shape[0])):
        if breaches[index]:
            run += 1
            if run >= persistence:
                return index - persistence + 1
        else:
            run = 0
    return None
