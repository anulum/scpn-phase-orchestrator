# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Critical-slowing-down early-warning monitor

"""Critical-slowing-down early warning from rising variance and autocorrelation.

The established generic early-warning framework for an approaching critical
transition is *critical slowing down*: as a system nears a bifurcation its
recovery from perturbations lengthens, which shows up as a rising variance and a
rising lag-one autocorrelation of the observable ahead of the transition. This
module implements that classical indicator as a passive monitor, so it can serve
as the literature baseline against the ordinal-transition-entropy detector in
``monitor/explosive_sync.py``.

``critical_slowing_down_warning`` slides a window across a multi-node signal
array, computes each window's mean-detrended variance and lag-one autocorrelation
per node, aggregates them across nodes, and raises a fail-early alarm when either
indicator rises a robust (median / MAD) margin above its leading baseline (a
rising variance or a lengthening autocorrelation is each a valid slowing-down
warning; requiring both understates the classical method). The
alarm logic — robust z-score against a leading baseline, a relative-change gate,
and a persistence run — mirrors ``explosive_sync_warning`` exactly (sign
reversed, since slowing-down is a *rise* and entropy regularisation is a *drop*),
so a lead-time comparison between the two is a same-alarm, different-indicator
test rather than an artefact of differing detector machinery. The monitor is
passive: it reads observables and emits a warning record; it never actuates.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — early-warning signals for critical
  transitions.
* Dakos, Carpenter, Brock, Ellison, Guttal, Ives, Kéfi, Livina, Seekell, van
  Nes & Scheffer 2012, *PLoS ONE* 7, e41010 — methods for detecting early
  warning signals of critical transitions in time series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Consistent-estimator scale factor making the median absolute deviation an
# unbiased estimate of the standard deviation for a normal distribution.
_MAD_TO_STD = 1.4826
_SCALE_FLOOR = 1.0e-12

__all__ = [
    "CriticalSlowingDownWarning",
    "critical_slowing_down_warning",
    "critical_slowing_down_multiscale_warning",
    "surrogate_score_threshold",
]


@dataclass(frozen=True)
class CriticalSlowingDownWarning:
    """Result of a critical-slowing-down early-warning sweep.

    Attributes
    ----------
    window_starts : IntArray
        First sample index of each analysis window, shape ``(W,)``.
    variance_index : FloatArray
        Mean per-node window variance per window, shape ``(W,)``.
    autocorrelation_index : FloatArray
        Mean per-node lag-one autocorrelation per window, shape ``(W,)``.
    combined_z : FloatArray
        Per-window rising indicator: the larger of the variance and
        autocorrelation robust z-scores, shape ``(W,)``. Large positive means
        at least one indicator rose — the sensitive critical-slowing-down
        signature (either a rising variance or a lengthening autocorrelation is
        a valid early warning; requiring both agrees less often and understates
        the classical method).
    robust_z_variance, robust_z_autocorrelation : FloatArray
        Median / MAD robust z-scores of each indicator against its baseline,
        shape ``(W,)``.
    relative_rise : FloatArray
        Larger of the two indicators' fractional rise above baseline, shape
        ``(W,)``.
    baseline_variance, baseline_autocorrelation : float
        Median of each indicator over the leading baseline windows.
    baseline_scale_variance, baseline_scale_autocorrelation : float
        Robust scale (``1.4826 × MAD``) of each baseline.
    n_baseline_windows : int
        Number of leading windows used to fit the baseline.
    warning_triggered : bool
        Whether a sustained rise crossed both the z and relative gates.
    warning_window : int | None
        Index of the first window of the triggering run, or ``None``.
    warning_sample : int | None
        Sample index ``window_starts[warning_window]``, or ``None``.
    window, step : int
        Echoed analysis parameters.
    z_threshold, rise_threshold : float
        Echoed alarm gates.
    persistence : int
        Echoed number of consecutive breaching windows required to alarm.
    """

    window_starts: IntArray = field(repr=False)
    variance_index: FloatArray = field(repr=False)
    autocorrelation_index: FloatArray = field(repr=False)
    combined_z: FloatArray = field(repr=False)
    robust_z_variance: FloatArray = field(repr=False)
    robust_z_autocorrelation: FloatArray = field(repr=False)
    relative_rise: FloatArray = field(repr=False)
    baseline_variance: float
    baseline_autocorrelation: float
    baseline_scale_variance: float
    baseline_scale_autocorrelation: float
    n_baseline_windows: int
    warning_triggered: bool
    warning_window: int | None
    warning_sample: int | None
    window: int
    step: int
    z_threshold: float
    rise_threshold: float
    persistence: int

    def summary(self) -> dict[str, float | int | bool | None]:
        """Return a flat scalar summary for logging or metric export.

        Returns
        -------
        dict[str, float | int | bool | None]
            Window/baseline counts, the peak rising z-score, the maximum
            relative rise, and the alarm verdict.
        """
        return {
            "n_windows": int(self.combined_z.shape[0]),
            "n_baseline_windows": self.n_baseline_windows,
            "baseline_variance": self.baseline_variance,
            "baseline_autocorrelation": self.baseline_autocorrelation,
            "max_combined_z": float(self.combined_z.max())
            if self.combined_z.size
            else 0.0,
            "max_relative_rise": float(self.relative_rise.max())
            if self.relative_rise.size
            else 0.0,
            "warning_triggered": self.warning_triggered,
            "warning_window": self.warning_window,
            "warning_sample": self.warning_sample,
        }


def critical_slowing_down_warning(
    signals: FloatArray,
    *,
    window: int = 128,
    step: int = 16,
    baseline_fraction: float = 0.25,
    min_baseline_windows: int = 3,
    z_threshold: float = 3.0,
    rise_threshold: float = 0.1,
    persistence: int = 2,
) -> CriticalSlowingDownWarning:
    """Sweep a multi-node signal for a critical-slowing-down warning.

    Parameters
    ----------
    signals : FloatArray
        Per-node scalar observables, shape ``(N, T)``; a one-dimensional array
        is treated as a single node.
    window : int
        Analysis window length in samples; must be at least three to admit a
        lag-one autocorrelation estimate.
    step : int
        Hop between consecutive window starts in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit the baseline, in ``(0, 1)``.
    min_baseline_windows : int
        Lower bound on the number of baseline windows.
    z_threshold : float
        Robust z-score magnitude above which a window breaches the rise gate.
    rise_threshold : float
        Minimum fractional rise above the baseline median to breach the gate.
    persistence : int
        Number of consecutive breaching windows required to raise the alarm.

    Returns
    -------
    CriticalSlowingDownWarning
        The per-window variance and autocorrelation fields, baseline fit, and
        the alarm decision.

    Raises
    ------
    ValueError
        If the inputs are malformed or the window does not fit the series.
    """
    array = _validate_signals(signals)
    window = _validate_positive_int(window, "window")
    step = _validate_positive_int(step, "step")
    min_baseline_windows = _validate_positive_int(
        min_baseline_windows, "min_baseline_windows"
    )
    baseline_fraction = _validate_unit_fraction(baseline_fraction, "baseline_fraction")
    z_threshold = _validate_non_negative_real(z_threshold, "z_threshold")
    rise_threshold = _validate_non_negative_real(rise_threshold, "rise_threshold")
    persistence = _validate_positive_int(persistence, "persistence")

    n_nodes, n_samples = int(array.shape[0]), int(array.shape[1])
    if window < 3:
        raise ValueError(f"window {window} must be at least 3 for autocorrelation")
    if window > n_samples:
        raise ValueError(f"window {window} exceeds the series length {n_samples}")

    starts = list(range(0, n_samples - window + 1, step))
    n_windows = len(starts)
    window_starts = np.asarray(starts, dtype=np.int64)
    variance_per_node = np.empty((n_windows, n_nodes), dtype=np.float64)
    autocorr_per_node = np.empty((n_windows, n_nodes), dtype=np.float64)
    for w, start in enumerate(starts):
        segment = array[:, start : start + window]
        for node in range(n_nodes):
            variance_per_node[w, node], autocorr_per_node[w, node] = _window_indicators(
                segment[node]
            )
    variance_index = variance_per_node.mean(axis=1)
    autocorrelation_index = autocorr_per_node.mean(axis=1)

    n_baseline = min(
        n_windows,
        max(min_baseline_windows, int(np.ceil(baseline_fraction * n_windows))),
    )
    z_var, base_var, scale_var = _robust_rise(variance_index, n_baseline)
    z_ac, base_ac, scale_ac = _robust_rise(autocorrelation_index, n_baseline)
    combined_z = np.maximum(z_var, z_ac)
    rise_var = _relative_rise(variance_index, base_var)
    rise_ac = _relative_rise(autocorrelation_index, base_ac)
    relative_rise = np.maximum(rise_var, rise_ac)

    breaches = (
        (np.arange(n_windows) >= n_baseline)
        & (combined_z >= z_threshold)
        & (relative_rise >= rise_threshold)
    )
    warning_window = _first_sustained_breach(breaches, persistence)
    warning_triggered = warning_window is not None
    warning_sample = (
        int(window_starts[warning_window]) if warning_window is not None else None
    )

    return CriticalSlowingDownWarning(
        window_starts=window_starts,
        variance_index=np.ascontiguousarray(variance_index, dtype=np.float64),
        autocorrelation_index=np.ascontiguousarray(
            autocorrelation_index, dtype=np.float64
        ),
        combined_z=np.ascontiguousarray(combined_z, dtype=np.float64),
        robust_z_variance=np.ascontiguousarray(z_var, dtype=np.float64),
        robust_z_autocorrelation=np.ascontiguousarray(z_ac, dtype=np.float64),
        relative_rise=np.ascontiguousarray(relative_rise, dtype=np.float64),
        baseline_variance=base_var,
        baseline_autocorrelation=base_ac,
        baseline_scale_variance=scale_var,
        baseline_scale_autocorrelation=scale_ac,
        n_baseline_windows=n_baseline,
        warning_triggered=warning_triggered,
        warning_window=warning_window,
        warning_sample=warning_sample,
        window=window,
        step=step,
        z_threshold=z_threshold,
        rise_threshold=rise_threshold,
        persistence=persistence,
    )


def critical_slowing_down_multiscale_warning(
    signals: FloatArray,
    *,
    windows: Sequence[int] | None = None,
    step: int = 16,
    baseline_fraction: float = 0.25,
    min_baseline_windows: int = 3,
    z_threshold: float = 3.0,
    rise_threshold: float = 0.1,
    persistence: int = 2,
    aggregation: str = "max",
) -> CriticalSlowingDownWarning:
    """Multi-scale critical-slowing-down warning.

    Variance and lag-one autocorrelation are computed at every window start for
    each of the supplied window lengths. The per-scale indices are then
    aggregated across scales on the shared window grid before the robust-rise
    alarm rule is applied. This lets the detector respond to precursors that
    emerge at horizons shorter or longer than a single fixed window.

    Parameters
    ----------
    signals : FloatArray
        Per-node scalar observables, shape ``(N, T)``; a one-dimensional array
        is treated as a single node.
    windows : sequence of int or None
        Window lengths to combine. Defaults to ``(64, 128, 256)``.
    step : int
        Hop between consecutive window starts in samples; shared by all scales.
    baseline_fraction : float
        Leading fraction of windows used to fit the baseline.
    min_baseline_windows : int
        Lower bound on the number of baseline windows.
    z_threshold : float
        Robust z-score gate applied to the aggregated combined score.
    rise_threshold : float
        Minimum fractional rise above the baseline median.
    persistence : int
        Number of consecutive breaching windows required to raise the alarm.
    aggregation : str
        How to aggregate scales: ``"max"`` (recommended) takes the strongest
        scale per window; ``"mean"`` averages scales.

    Returns
    -------
    CriticalSlowingDownWarning
        The aggregated per-window fields, baseline fit, and alarm decision.

    Raises
    ------
    ValueError
        If ``windows`` contains duplicate lengths, if ``aggregation`` is neither
        ``"max"`` nor ``"mean"``, if any window is smaller than three samples,
        or if the largest window exceeds the series length. Parameter validation
        also raises ``ValueError`` for non-positive integers or fractions
        outside the unit interval.
    """
    array = _validate_signals(signals)
    if windows is None:
        windows = (64, 128, 256)
    windows_tuple = tuple(_validate_positive_int(w, "windows") for w in windows)
    if len(set(windows_tuple)) != len(windows_tuple):
        raise ValueError("windows must be unique")
    step = _validate_positive_int(step, "step")
    min_baseline_windows = _validate_positive_int(
        min_baseline_windows, "min_baseline_windows"
    )
    baseline_fraction = _validate_unit_fraction(baseline_fraction, "baseline_fraction")
    z_threshold = _validate_non_negative_real(z_threshold, "z_threshold")
    rise_threshold = _validate_non_negative_real(rise_threshold, "rise_threshold")
    persistence = _validate_positive_int(persistence, "persistence")
    if aggregation not in {"max", "mean"}:
        raise ValueError(f"aggregation must be 'max' or 'mean', got {aggregation!r}")

    n_nodes, n_samples = int(array.shape[0]), int(array.shape[1])
    max_window = max(windows_tuple)
    if max_window < 3:
        raise ValueError("all windows must be at least 3 for autocorrelation")
    if max_window > n_samples:
        raise ValueError(
            f"largest window {max_window} exceeds the series length {n_samples}"
        )

    starts = list(range(0, n_samples - max_window + 1, step))
    n_windows = len(starts)
    window_starts = np.asarray(starts, dtype=np.int64)

    n_scales = len(windows_tuple)
    variance_scales = np.empty((n_scales, n_windows, n_nodes), dtype=np.float64)
    autocorr_scales = np.empty((n_scales, n_windows, n_nodes), dtype=np.float64)
    for scale_idx, window_len in enumerate(windows_tuple):
        for w_idx, start in enumerate(starts):
            segment = array[:, start : start + window_len]
            for node in range(n_nodes):
                (
                    variance_scales[scale_idx, w_idx, node],
                    autocorr_scales[scale_idx, w_idx, node],
                ) = _window_indicators(segment[node])

    n_baseline = min(
        n_windows,
        max(min_baseline_windows, int(np.ceil(baseline_fraction * n_windows))),
    )

    # Normalise each scale with its own robust baseline, then aggregate the
    # oriented scores. This prevents large-window raw variance from dominating
    # small-window autocorrelation signals.
    combined_z_scales = np.empty((n_scales, n_windows), dtype=np.float64)
    relative_rise_scales = np.empty((n_scales, n_windows), dtype=np.float64)
    for scale_idx in range(n_scales):
        variance_index = variance_scales[scale_idx].mean(axis=1)
        autocorrelation_index = autocorr_scales[scale_idx].mean(axis=1)
        z_var, base_var, _ = _robust_rise(variance_index, n_baseline)
        z_ac, base_ac, _ = _robust_rise(autocorrelation_index, n_baseline)
        combined_z_scales[scale_idx] = np.maximum(z_var, z_ac)
        rise_var = _relative_rise(variance_index, base_var)
        rise_ac = _relative_rise(autocorrelation_index, base_ac)
        relative_rise_scales[scale_idx] = np.maximum(rise_var, rise_ac)

    if aggregation == "max":
        combined_z = combined_z_scales.max(axis=0)
        relative_rise = relative_rise_scales.max(axis=0)
    else:  # aggregation == "mean"
        combined_z = combined_z_scales.mean(axis=0)
        relative_rise = relative_rise_scales.mean(axis=0)

    # Report the aggregated indices for introspection.
    variance_index = variance_scales.mean(axis=0).mean(axis=1)
    autocorrelation_index = autocorr_scales.mean(axis=0).mean(axis=1)
    z_var, base_var, scale_var = _robust_rise(variance_index, n_baseline)
    z_ac, base_ac, scale_ac = _robust_rise(autocorrelation_index, n_baseline)

    breaches = (
        (np.arange(n_windows) >= n_baseline)
        & (combined_z >= z_threshold)
        & (relative_rise >= rise_threshold)
    )
    warning_window = _first_sustained_breach(breaches, persistence)
    warning_triggered = warning_window is not None
    warning_sample = (
        int(window_starts[warning_window]) if warning_window is not None else None
    )

    return CriticalSlowingDownWarning(
        window_starts=window_starts,
        variance_index=np.ascontiguousarray(variance_index, dtype=np.float64),
        autocorrelation_index=np.ascontiguousarray(
            autocorrelation_index, dtype=np.float64
        ),
        combined_z=np.ascontiguousarray(combined_z, dtype=np.float64),
        robust_z_variance=np.ascontiguousarray(z_var, dtype=np.float64),
        robust_z_autocorrelation=np.ascontiguousarray(z_ac, dtype=np.float64),
        relative_rise=np.ascontiguousarray(relative_rise, dtype=np.float64),
        baseline_variance=base_var,
        baseline_autocorrelation=base_ac,
        baseline_scale_variance=scale_var,
        baseline_scale_autocorrelation=scale_ac,
        n_baseline_windows=n_baseline,
        warning_triggered=warning_triggered,
        warning_window=warning_window,
        warning_sample=warning_sample,
        window=max_window,
        step=step,
        z_threshold=z_threshold,
        rise_threshold=rise_threshold,
        persistence=persistence,
    )


def surrogate_score_threshold(
    signals: FloatArray,
    *,
    n_surrogates: int = 200,
    percentile: float = 95.0,
    block_length: int | None = None,
    window: int = 128,
    step: int = 16,
    baseline_fraction: float = 0.25,
    min_baseline_windows: int = 3,
    persistence: int = 2,
    rng: int | np.random.Generator | None = None,
) -> float:
    """Return a false-alarm threshold from a block-bootstrap null distribution.

    The order-parameter series is resampled by circular block bootstrap. For each
    surrogate, the critical-slowing-down combined score is computed with a zero
    z-threshold and the maximum post-baseline score is recorded. The returned
    threshold is the requested percentile of those maxima and can be passed as
    ``z_threshold`` to :func:`critical_slowing_down_warning` to control the
    empirical false-alarm probability.

    Parameters
    ----------
    signals : FloatArray
        Per-node scalar observables, shape ``(N, T)`` or one-dimensional.
    n_surrogates : int
        Number of bootstrap surrogates to draw.
    percentile : float
        Percentile of the surrogate max-score distribution to return as the
        threshold; e.g. ``95.0`` targets roughly a 5% false-alarm rate.
    block_length : int or None
        Bootstrap block length in samples; defaults to ``max(1, T // 20)``.
    window, step : int
        Analysis window length and hop passed to the underlying warning sweep.
    baseline_fraction : float
        Baseline fraction passed to the underlying warning sweep.
    min_baseline_windows : int
        Minimum baseline windows passed to the underlying warning sweep.
    persistence : int
        Persistence passed to the underlying warning sweep.
    rng : int or np.random.Generator or None
        Seed or generator for reproducible bootstrapping.

    Returns
    -------
    float
        A positive threshold on the combined score.

    Raises
    ------
    ValueError
        If ``percentile`` exceeds 100, or if any count or length parameter is
        not a positive integer.
    """
    array = _validate_signals(signals)
    n_surrogates = _validate_positive_int(n_surrogates, "n_surrogates")
    percentile = _validate_non_negative_real(percentile, "percentile")
    if percentile > 100.0:
        raise ValueError(f"percentile must be <= 100, got {percentile}")
    window = _validate_positive_int(window, "window")
    step = _validate_positive_int(step, "step")
    persistence = _validate_positive_int(persistence, "persistence")
    rng = np.random.default_rng(rng)

    n_samples = int(array.shape[1])
    if block_length is None:
        block_length = max(1, n_samples // 20)
    else:
        block_length = _validate_positive_int(block_length, "block_length")

    max_scores = np.empty(n_surrogates, dtype=np.float64)
    for surrogate_idx in range(n_surrogates):
        surrogate = _block_bootstrap(array, block_length, rng)
        warning = critical_slowing_down_warning(
            surrogate,
            window=window,
            step=step,
            baseline_fraction=baseline_fraction,
            min_baseline_windows=min_baseline_windows,
            z_threshold=0.0,
            rise_threshold=0.0,
            persistence=persistence,
        )
        if warning.combined_z.size == 0:
            max_scores[surrogate_idx] = 0.0
            continue
        post = warning.combined_z[warning.n_baseline_windows :]
        max_scores[surrogate_idx] = float(post.max()) if post.size else 0.0
    return float(np.percentile(max_scores, percentile))


def _block_bootstrap(
    array: FloatArray, block_length: int, rng: np.random.Generator
) -> FloatArray:
    """Return a circular block-bootstrap resample of ``array`` along time."""
    n_samples = int(array.shape[1])
    n_blocks = int(np.ceil(n_samples / block_length))
    starts = rng.integers(0, n_samples, size=n_blocks)
    blocks: list[FloatArray] = []
    for start in starts:
        idx = (np.arange(start, start + block_length) % n_samples).astype(np.int64)
        blocks.append(array[:, idx])
    surrogate = np.concatenate(blocks, axis=1)[:, :n_samples]
    return np.ascontiguousarray(surrogate, dtype=np.float64)


def _window_indicators(segment: FloatArray) -> tuple[float, float]:
    """Return the mean-detrended variance and lag-one autocorrelation."""
    centred = segment - float(np.mean(segment))
    variance = float(np.mean(centred * centred))
    if variance <= _SCALE_FLOOR:
        return variance, 0.0
    lagged = float(np.mean(centred[:-1] * centred[1:]))
    return variance, lagged / variance


def _robust_rise(index: FloatArray, n_baseline: int) -> tuple[FloatArray, float, float]:
    """Return the robust z-score of ``index`` against its leading baseline.

    ``n_baseline`` is always at least one (the caller floors it), so the leading
    baseline slice is never empty.
    """
    baseline = index[:n_baseline]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    scale = _MAD_TO_STD * mad
    guarded = max(scale, _SCALE_FLOOR)
    return (index - median) / guarded, median, scale


def _relative_rise(index: FloatArray, baseline_median: float) -> FloatArray:
    """Return the fractional rise of ``index`` above the baseline median."""
    if baseline_median > 0.0:
        return (index - baseline_median) / baseline_median
    return np.zeros_like(index)


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


def _validate_signals(signals: object) -> FloatArray:
    """Return the signals as a validated 2-D finite array, else raise."""
    raw = np.asarray(signals)
    if raw.dtype == np.bool_:
        raise ValueError("signals must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("signals must contain real-valued samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals must be a real float array") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"signals shape {raw.shape} must be one- or two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("signals must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _validate_unit_fraction(value: object, name: str) -> float:
    """Return ``value`` as a fraction in the open interval (0, 1), else raise."""
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
