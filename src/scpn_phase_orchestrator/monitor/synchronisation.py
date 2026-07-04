# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Rising-synchronisation early-warning monitor

"""Rising-synchronisation early warning from the Kuramoto order parameter.

A synchronisation transition — the abrupt collective phase-locking behind a
seizure onset or a grid coherence collapse — is preceded by the population's
phase coherence climbing toward the locked state. The Kuramoto order parameter
``R(t) = |⟨e^{iθ}⟩|`` measures that coherence directly, so a sustained rise in
its windowed level is a first-moment early-warning signal complementary to the
second-moment critical-slowing-down indicators (``monitor/critical_slowing_down.py``,
which read a variance/autocorrelation *rise*) and to the ordinal-transition-
entropy detector (``monitor/explosive_sync.py``, which reads a regularisation
*drop*). On a real scalp-EEG seizure the order parameter is the signal that
carries the leading precursor, which is why it is a first-class member of the
early-warning detector suite.

``synchronisation_warning`` computes the instantaneous order parameter across the
per-node phases, averages it within each sliding window, and raises a fail-early
alarm when the windowed coherence rises a robust (median / MAD) margin above its
leading baseline. The alarm logic — robust z-score against a leading baseline, a
relative-change gate, and a persistence run — is the suite's shared contract, so
this detector is directly comparable with the others at a matched false-alarm
rate. The monitor is passive: it reads phases and emits a warning record; it
never actuates.

References
----------
* Kuramoto 1984, *Chemical Oscillations, Waves, and Turbulence* — the order
  parameter of coupled phase oscillators.
* Scheffer et al. 2009, *Nature* 461, 53 — early-warning signals for critical
  transitions (the framework this contributes a synchrony indicator to).
"""

from __future__ import annotations

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
    "SynchronisationWarning",
    "synchronisation_warning",
]


@dataclass(frozen=True)
class SynchronisationWarning:
    """Result of a rising-synchronisation early-warning sweep.

    Attributes
    ----------
    window_starts : IntArray
        First sample index of each analysis window, shape ``(W,)``.
    synchrony_index : FloatArray
        Mean Kuramoto order parameter within each window, shape ``(W,)``; the
        headline coherence level in ``[0, 1]``.
    robust_z : FloatArray
        Median / MAD robust z-score of ``synchrony_index`` against the baseline,
        shape ``(W,)``. Strongly positive means a sharp coherence rise.
    relative_rise : FloatArray
        Fractional rise of ``synchrony_index`` above the baseline median, shape
        ``(W,)``.
    baseline_median : float
        Median synchrony index over the leading baseline windows.
    baseline_scale : float
        Robust scale (``1.4826 × MAD``) of the baseline windows.
    n_baseline_windows : int
        Number of leading windows used to fit the baseline.
    warning_triggered : bool
        Whether a sustained rise crossed both the z and relative-rise gates.
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
    synchrony_index: FloatArray = field(repr=False)
    robust_z: FloatArray = field(repr=False)
    relative_rise: FloatArray = field(repr=False)
    baseline_median: float
    baseline_scale: float
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
            Window/baseline counts, the baseline coherence, the peak rising
            z-score, the maximum relative rise, and the alarm verdict.
        """
        return {
            "n_windows": int(self.synchrony_index.shape[0]),
            "n_baseline_windows": self.n_baseline_windows,
            "baseline_median": self.baseline_median,
            "max_synchrony_index": float(self.synchrony_index.max())
            if self.synchrony_index.size
            else 0.0,
            "max_robust_z": float(self.robust_z.max()) if self.robust_z.size else 0.0,
            "max_relative_rise": float(self.relative_rise.max())
            if self.relative_rise.size
            else 0.0,
            "warning_triggered": self.warning_triggered,
            "warning_window": self.warning_window,
            "warning_sample": self.warning_sample,
        }


def synchronisation_warning(
    phases: FloatArray,
    *,
    window: int = 128,
    step: int = 16,
    baseline_fraction: float = 0.25,
    min_baseline_windows: int = 3,
    z_threshold: float = 3.0,
    rise_threshold: float = 0.1,
    persistence: int = 2,
) -> SynchronisationWarning:
    """Sweep per-node phases for a rising-synchronisation warning.

    Parameters
    ----------
    phases : FloatArray
        Per-node instantaneous phases in radians, shape ``(N, T)`` with at least
        two nodes (synchrony is undefined for a single oscillator).
    window : int
        Analysis window length in samples; must be at least one.
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
    SynchronisationWarning
        The per-window coherence field, baseline fit, and the alarm decision.

    Raises
    ------
    ValueError
        If the inputs are malformed or the window does not fit the series.
    """
    array = _validate_phases(phases)
    window = _validate_positive_int(window, "window")
    step = _validate_positive_int(step, "step")
    min_baseline_windows = _validate_positive_int(
        min_baseline_windows, "min_baseline_windows"
    )
    baseline_fraction = _validate_unit_fraction(baseline_fraction, "baseline_fraction")
    z_threshold = _validate_non_negative_real(z_threshold, "z_threshold")
    rise_threshold = _validate_non_negative_real(rise_threshold, "rise_threshold")
    persistence = _validate_positive_int(persistence, "persistence")

    n_samples = int(array.shape[1])
    if window > n_samples:
        raise ValueError(f"window {window} exceeds the series length {n_samples}")

    coherence = np.abs(np.mean(np.exp(1j * array), axis=0))
    starts = list(range(0, n_samples - window + 1, step))
    window_starts = np.asarray(starts, dtype=np.int64)
    synchrony_index = np.asarray(
        [float(np.mean(coherence[start : start + window])) for start in starts],
        dtype=np.float64,
    )

    n_windows = synchrony_index.shape[0]
    n_baseline = min(
        n_windows,
        max(min_baseline_windows, int(np.ceil(baseline_fraction * n_windows))),
    )
    baseline = synchrony_index[:n_baseline]
    baseline_median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - baseline_median)))
    baseline_scale = _MAD_TO_STD * mad
    guarded_scale = max(baseline_scale, _SCALE_FLOOR)

    robust_z = (synchrony_index - baseline_median) / guarded_scale
    if baseline_median > _SCALE_FLOOR:
        relative_rise = (synchrony_index - baseline_median) / baseline_median
    else:
        # A near-zero baseline coherence (e.g. an anti-phase population) makes a
        # fractional rise undefined, so the relative gate is left unarmed rather
        # than dividing by an epsilon.
        relative_rise = np.zeros_like(synchrony_index)

    breaches = (
        (np.arange(n_windows) >= n_baseline)
        & (robust_z >= z_threshold)
        & (relative_rise >= rise_threshold)
    )
    warning_window = _first_sustained_breach(breaches, persistence)
    warning_triggered = warning_window is not None
    warning_sample = (
        int(window_starts[warning_window]) if warning_window is not None else None
    )

    return SynchronisationWarning(
        window_starts=window_starts,
        synchrony_index=np.ascontiguousarray(synchrony_index, dtype=np.float64),
        robust_z=np.ascontiguousarray(robust_z, dtype=np.float64),
        relative_rise=np.ascontiguousarray(relative_rise, dtype=np.float64),
        baseline_median=baseline_median,
        baseline_scale=baseline_scale,
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


def _validate_phases(phases: object) -> FloatArray:
    """Return the phases as a validated 2-D finite array with ≥ 2 nodes."""
    raw = np.asarray(phases)
    if raw.dtype == np.bool_:
        raise ValueError("phases must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("phases must be real-valued radians")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a real float array") from exc
    if array.ndim != 2:
        raise ValueError(f"phases shape {raw.shape} must be two-dimensional (N, T)")
    if array.shape[0] < 2:
        raise ValueError("phases must have at least two nodes for synchrony")
    if not np.all(np.isfinite(array)):
        raise ValueError("phases must contain only finite values")
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
