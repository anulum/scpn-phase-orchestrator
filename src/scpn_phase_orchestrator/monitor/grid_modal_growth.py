# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — power-grid modal-growth early-warning detector

"""The power-grid modal-growth early-warning detector — the operational core.

This is the product-side detector that the offline head-to-head
(``bench.grid_modal_head_to_head``) *certifies*: on the real PSML 23-bus corpus, at a
matched false alarm, its growth-rate ``σ`` leads growing-instability transitions far
more than any generic early-warning member, which are all at chance. The offline
benchmark adds the matched-false-alarm calibration and label-permutation significance
around this core; this module is the detector itself — the passive quantity a live
monitor (``monitor.grid_modal_stream``) scores on each window.

When a disturbance leaves an electromechanical mode under-damped, the amplitude of the
cross-bus voltage deviation grows *exponentially* — the real part ``σ`` of the dominant
mode's eigenvalue is positive (negative damping), the canonical wide-area-monitoring
early-warning quantity (Kundur 1994). The detector estimates it directly:

* :func:`cross_bus_deviation` — the per-sample mean absolute deviation of the bus
  voltages from their cross-bus mean, an amplitude envelope of the collective mode;
* :func:`per_bus_deviation` — the same deviation *kept per bus*, so a growth rate can be
  measured on each bus and the most unstable one taken;
* :func:`envelope_growth_rate` — the exponential growth rate ``σ`` of a deviation
  envelope, the slope of its log against time; ``σ > 0`` grows (unstable), ``σ < 0``
  damps. A ``recency_top`` weighting lets later samples count for more, because a real
  instability *accelerates* toward onset;
* :func:`growth_rate_and_fit` — the same ``σ`` *and* the exponential-fit quality ``R²``,
  which is high for a smooth exponential and low for a step-like transient;
* :func:`fit_gated_growth_rate` — ``σ`` gated by that fit quality: the growth rate is
  clamped to non-positive when ``R²`` falls below a gate, so a damped fault's step-like
  transient (a poor exponential fit) cannot pass as an instability — the one principled
  lever that lifts the *streaming* operating point above the plain rate;
* :func:`modal_growth_score` — one segment's ``σ`` under the ``"focal"`` (most unstable
  bus) or ``"mean"`` (whole network) aggregation, the certified default being
  ``"focal"`` with a recency weighting (:data:`DEFAULT_AGGREGATION`,
  :data:`DEFAULT_RECENCY_TOP`); an optional ``r2_gate`` applies the fit-quality gate.

The detector is passive: it reads bus voltages and returns a growth rate; it never
actuates and never recalibrates — the operating point is fixed by the certification.

References
----------
* Zheng et al. 2021 — the PSML power-system dataset (23-bus millisecond-level PMU
  measurements) with disturbance-type annotations.
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability: a
  mode's eigenvalue real part is its growth rate, the sign of instability.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: A deviation floor so a perfectly flat window cannot take a logarithm of zero.
_DEVIATION_FLOOR = 1.0e-12

#: The validated default aggregation — the most unstable bus, un-diluted.
DEFAULT_AGGREGATION = "focal"

#: The validated default recency weighting: later samples (nearer the disturbance, where
#: an instability has accelerated) count up to this many times as much as the earliest.
DEFAULT_RECENCY_TOP = 3.0

__all__ = [
    "DEFAULT_AGGREGATION",
    "DEFAULT_RECENCY_TOP",
    "cross_bus_deviation",
    "envelope_growth_rate",
    "fit_gated_growth_rate",
    "growth_rate_and_fit",
    "modal_growth_score",
    "per_bus_deviation",
]


def cross_bus_deviation(voltages: FloatArray) -> FloatArray:
    """Return the per-sample cross-bus voltage-deviation envelope.

    At each time sample, the mean absolute deviation of the bus voltages from their
    cross-bus mean — an amplitude envelope of the collective oscillation that grows as a
    mode goes unstable.

    Parameters
    ----------
    voltages : FloatArray
        Per-bus voltage magnitudes, shape ``(buses, samples)`` with at least one bus.

    Returns
    -------
    FloatArray
        The deviation envelope, shape ``(samples,)``.

    Raises
    ------
    ValueError
        If ``voltages`` is not a two-dimensional buses-by-samples array with a bus and a
        sample.
    """
    values = _validate_voltage_matrix(voltages)
    centred = values - values.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(np.abs(centred).mean(axis=0))


def per_bus_deviation(voltages: FloatArray) -> FloatArray:
    """Return the per-bus voltage-deviation envelopes, one per bus.

    The absolute deviation of each bus voltage from the cross-bus mean, *without*
    averaging over the buses — the raw material for the ``"focal"`` aggregation, which
    measures the growth rate on every bus and keeps the most unstable one.

    Parameters
    ----------
    voltages : FloatArray
        Per-bus voltage magnitudes, shape ``(buses, samples)`` with at least one bus.

    Returns
    -------
    FloatArray
        The per-bus deviation envelopes, shape ``(buses, samples)``.

    Raises
    ------
    ValueError
        If ``voltages`` is not a two-dimensional buses-by-samples array with a bus and a
        sample.
    """
    values = _validate_voltage_matrix(voltages)
    centred = values - values.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(np.abs(centred))


def _recency_weighted_slope(
    times: FloatArray, logs: FloatArray, recency_top: float
) -> float:
    """Return the weighted-least-squares slope of ``logs`` on ``times``.

    Linear weights rise from one at the first sample to ``recency_top`` at the last, so
    the growth close to the disturbance dominates the fit. With ``recency_top == 1`` the
    weights are uniform and this reduces to ordinary least squares; the caller reserves
    that case for :func:`numpy.polyfit` and only calls here for a genuine weighting.
    """
    weights = np.linspace(1.0, recency_top, times.shape[0])
    total = weights.sum()
    mean_t = float((weights * times).sum() / total)
    mean_y = float((weights * logs).sum() / total)
    denom = float((weights * (times - mean_t) ** 2).sum())
    return float((weights * (times - mean_t) * (logs - mean_y)).sum() / denom)


def _validate_real_array(value: object, name: str) -> FloatArray:
    """Return a copied finite, non-coercive real array."""
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
    try:
        result = np.asarray(raw, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _validate_voltage_matrix(value: object) -> FloatArray:
    """Return a non-empty buses-by-samples voltage matrix."""
    values = _validate_real_array(value, "voltages")
    if values.ndim != 2:
        raise ValueError("voltages must be two-dimensional (buses × samples)")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("voltages must have at least one bus and one sample")
    return values


def _validate_deviation(value: object) -> FloatArray:
    """Return a finite non-negative deviation vector with at least two samples."""
    values = _validate_real_array(value, "deviation")
    if values.ndim != 1:
        raise ValueError("deviation must be one-dimensional")
    if values.shape[0] < 2:
        raise ValueError("deviation must have at least two samples")
    if np.any(values < 0.0):
        raise ValueError("deviation must contain only non-negative values")
    return values


def _validate_finite_real(value: object, name: str) -> float:
    """Return a finite, non-boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _validate_growth_inputs(
    deviation: object,
    rate: object,
    recency_top: object,
) -> tuple[FloatArray, float, float]:
    """Return validated envelope-fit inputs."""
    values = _validate_deviation(deviation)
    sample_rate = _validate_finite_real(rate, "rate")
    if sample_rate <= 0.0:
        raise ValueError("rate must be a positive finite number")
    recency = _validate_finite_real(recency_top, "recency_top")
    if recency < 1.0:
        raise ValueError("recency_top must be a finite number at least one")
    with np.errstate(over="ignore"):
        times = np.arange(values.shape[0], dtype=np.float64) / sample_rate
    if not np.all(np.isfinite(times)):
        raise ValueError("rate is too small to define a finite time axis")
    if recency > 1.0 and recency > np.finfo(np.float64).max / values.shape[0]:
        raise ValueError("recency_top is too large to define finite weights")
    return values, sample_rate, recency


def _growth_rate_from_validated(
    values: FloatArray,
    rate: float,
    recency_top: float,
) -> float:
    """Return the log-envelope slope for validated inputs."""
    times = np.arange(values.shape[0], dtype=np.float64) / rate
    logs = np.log(np.maximum(values, _DEVIATION_FLOOR))
    if recency_top == 1.0:
        return float(np.polyfit(times, logs, 1)[0])
    return _recency_weighted_slope(times, logs, recency_top)


def _validate_r2_gate(value: object) -> float:
    """Return a finite fit-quality gate in [0, 1]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("r2_gate must be a finite number in [0, 1]")
    gate = float(value)
    if not np.isfinite(gate) or not 0.0 <= gate <= 1.0:
        raise ValueError("r2_gate must be a finite number in [0, 1]")
    return gate


def envelope_growth_rate(
    deviation: FloatArray, *, rate: float, recency_top: float = 1.0
) -> float:
    """Return the exponential growth rate of a deviation envelope.

    Fits ``log(envelope)`` linearly against time and returns the slope ``σ``: the real
    part of the dominant mode's eigenvalue. ``σ > 0`` is a growing (unstable) mode,
    ``σ < 0`` a damped one. The envelope is floored away from zero before the logarithm.
    With ``recency_top > 1`` the later samples are up-weighted (see
    :func:`_recency_weighted_slope`), as a real instability accelerates toward onset.

    Parameters
    ----------
    deviation : FloatArray
        The deviation envelope over the segment, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive, so ``σ`` is per second.
    recency_top : float
        Weight of the last sample relative to the first, ``>= 1``. ``1.0`` (the default)
        is an unweighted ordinary-least-squares fit.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds; ``0.0`` if the fit is undefined.

    Raises
    ------
    ValueError
        If ``deviation`` is not a one-dimensional array of at least two samples,
        ``rate`` is not positive, or ``recency_top`` is not a finite number ``>= 1``.
    """
    values, sample_rate, recency = _validate_growth_inputs(
        deviation,
        rate,
        recency_top,
    )
    slope = _growth_rate_from_validated(values, sample_rate, recency)
    if not np.isfinite(slope):
        raise ValueError("deviation does not define a finite growth rate")
    return slope


def growth_rate_and_fit(
    deviation: FloatArray, *, rate: float, recency_top: float = 1.0
) -> tuple[float, float]:
    """Return the growth rate ``σ`` and the quality ``R²`` of its exponential fit.

    ``σ`` is exactly :func:`envelope_growth_rate` (the log-envelope slope under the same
    recency weighting), so the gate composes with the certified detector without
    changing its rate. ``R²`` is the (recency-weighted) coefficient of determination of
    that straight-line log fit: near one for a smooth exponential — a genuine growing
    mode — and low for a step-like transient such as a damped fault, whose log envelope
    is a poor line. Comparing ``R²`` to a gate is what separates the two online.

    Parameters
    ----------
    deviation : FloatArray
        The deviation envelope over the segment, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive.
    recency_top : float
        Weight of the last sample relative to the first, ``>= 1``; ``1.0`` unweighted.

    Returns
    -------
    tuple of float
        ``(σ, R²)``: the growth rate in inverse seconds and the fit quality. ``R²`` is
        ``0.0`` when the log envelope is flat (no variance to explain).

    Raises
    ------
    ValueError
        If ``deviation`` is not a one-dimensional array of at least two samples,
        ``rate`` is not positive, or ``recency_top`` is not a finite number ``>= 1``.
    """
    values, sample_rate, recency = _validate_growth_inputs(
        deviation,
        rate,
        recency_top,
    )
    slope = _growth_rate_from_validated(values, sample_rate, recency)
    times = np.arange(values.shape[0], dtype=np.float64) / sample_rate
    logs = np.log(np.maximum(values, _DEVIATION_FLOOR))
    weights = np.linspace(1.0, recency, values.shape[0])
    total = weights.sum()
    mean_t = float((weights * times).sum() / total)
    mean_y = float((weights * logs).sum() / total)
    predicted = mean_y + slope * (times - mean_t)
    ss_res = float((weights * (logs - predicted) ** 2).sum())
    ss_tot = float((weights * (logs - mean_y) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, r2


def fit_gated_growth_rate(
    deviation: FloatArray,
    *,
    rate: float,
    recency_top: float = 1.0,
    r2_gate: float = 0.0,
) -> float:
    """Return the growth rate ``σ`` gated by its exponential-fit quality ``R²``.

    With ``r2_gate`` at ``0`` the gate is off and this is exactly
    :func:`envelope_growth_rate`. With ``r2_gate > 0`` the growth rate is kept only when
    its fit quality reaches the gate (``R² >= r2_gate``); otherwise it is clamped to
    non-positive (``min(σ, 0)``), so a step-like transient — a damped fault, which fits
    an exponential poorly — cannot be read as a growing instability. This is the
    streaming detector's principled false-alarm lever.

    Parameters
    ----------
    deviation : FloatArray
        The deviation envelope over the segment, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive.
    recency_top : float
        Weight of the last sample relative to the first, ``>= 1``.
    r2_gate : float
        Fit-quality gate in ``[0, 1]``. ``0.0`` (the default) disables the gate and
        preserves the plain growth rate exactly.

    Returns
    -------
    float
        The gated growth rate ``σ`` in inverse seconds.

    Raises
    ------
    ValueError
        If ``r2_gate`` is not a finite number in ``[0, 1]``, or the arguments forwarded
        to :func:`envelope_growth_rate` are invalid.
    """
    gate = _validate_r2_gate(r2_gate)
    if gate <= 0.0:
        return envelope_growth_rate(deviation, rate=rate, recency_top=recency_top)
    sigma, r2 = growth_rate_and_fit(deviation, rate=rate, recency_top=recency_top)
    return sigma if r2 >= gate else min(sigma, 0.0)


def modal_growth_score(
    segment: FloatArray,
    *,
    rate: float,
    aggregation: str = DEFAULT_AGGREGATION,
    recency_top: float = DEFAULT_RECENCY_TOP,
    r2_gate: float = 0.0,
) -> float:
    """Return one segment's modal growth rate ``σ`` under the chosen aggregation.

    ``"mean"`` scores the growth rate of the whole-network :func:`cross_bus_deviation`
    envelope; ``"focal"`` scores the maximum growth rate over the per-bus envelopes
    (:func:`per_bus_deviation`) — the most unstable bus, un-diluted. An ``r2_gate``
    above zero applies the fit-quality gate (:func:`fit_gated_growth_rate`) to every
    envelope before aggregating; at ``r2_gate == 0`` (the certified offline default) the
    score is the plain growth rate exactly.

    Parameters
    ----------
    segment : FloatArray
        The segment's per-bus voltages, shape ``(buses, samples)``.
    rate : float
        Sampling rate in Hz.
    aggregation : str
        ``"mean"`` (whole network) or ``"focal"`` (most unstable bus).
    recency_top : float
        Recency weighting passed to :func:`envelope_growth_rate`.
    r2_gate : float
        Fit-quality gate in ``[0, 1]`` passed to :func:`fit_gated_growth_rate`; ``0.0``
        disables it.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds.

    Raises
    ------
    ValueError
        If ``aggregation`` is neither ``"mean"`` nor ``"focal"``, or ``r2_gate`` is
        not a finite number in ``[0, 1]``.
    """
    gate = _validate_r2_gate(r2_gate)
    if not isinstance(aggregation, str):
        raise ValueError("aggregation must be 'mean' or 'focal'")
    if aggregation == "mean":
        return fit_gated_growth_rate(
            cross_bus_deviation(segment),
            rate=rate,
            recency_top=recency_top,
            r2_gate=gate,
        )
    if aggregation == "focal":
        return max(
            fit_gated_growth_rate(
                envelope, rate=rate, recency_top=recency_top, r2_gate=gate
            )
            for envelope in per_bus_deviation(segment)
        )
    raise ValueError(f"aggregation must be 'mean' or 'focal', got {aggregation!r}")
