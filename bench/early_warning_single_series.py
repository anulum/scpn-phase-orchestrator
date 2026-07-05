# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — single-series critical-slowing-down lead-time harness

"""Single-series critical-slowing-down matched-false-alarm lead-time harness.

The multi-node harness (:mod:`bench.early_warning_domain`) proves the early-warning
design on domains that carry a *population* of coupled oscillators — scalp-EEG
channels, ECG leads, grid buses — where rising synchronisation and ordinal-transition
entropy are defined. Two canonical early-warning domains are not like that: a
palaeoclimate proxy record (Dakos et al. 2008) and a collapsing ecological population
(Dai et al. 2012) are **single scalar time-series** approaching a bifurcation. There
is no second node to synchronise with, so the synchronisation and entropy members do
not apply — but *critical slowing down*, the rising variance and lag-one
autocorrelation of the one observable, is exactly the indicator those studies use.

This module is the single-detector counterpart of the multi-node harness: it segments
a scalar series, calibrates the critical-slowing-down detector to a matched
false-alarm rate on a no-transition null, measures the honest lead of its alarm
against an annotated onset, and seals the alarm — or the silence — into the same
:class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`
record the multi-node harness seals. It deliberately reuses the multi-node harness's
:class:`~bench.early_warning_domain.DetectorTrajectory`, its continuous
matched-false-alarm :func:`~bench.early_warning_domain.calibrate_threshold`, and its
:class:`~bench.early_warning_domain.Calibration` container, so the one detector is
calibrated and scored by the identical machinery — a single-series result is directly
comparable to a multi-node one, not an artefact of a different harness.

The two paths stay in lock-step exactly as the multi-node harness's do:
:func:`critical_slowing_down_trajectory` runs the detector once at a zero gate for a
threshold-free score trajectory that calibration searches, and
:func:`evaluate_single_series` re-runs the detector at the calibrated threshold and
seals *that* alarm decision — the detector's per-window ``combined_z`` and
``relative_rise`` are independent of the gates, so the alarm found at calibration and
the alarm sealed at evaluation are the same one.

References
----------
* Dakos, Scheffer, van Nes, Brovkin, Petoukhov & Held 2008, *PNAS* 105, 14308 —
  slowing down as an early warning for abrupt climate change.
* Dai, Vorselen, Korolev & Gore 2012, *Science* 336, 1175 — generic indicators for
  loss of resilience before a tipping point leading to population collapse.
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals for
  critical transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bench.early_warning_domain import (
    DEFAULT_BASELINE_FRACTION,
    DEFAULT_PERSISTENCE,
    DEFAULT_RELATIVE_GATE,
    DEFAULT_STEP,
    DEFAULT_TARGET_FALSE_ALARM,
    DEFAULT_WINDOW,
    Calibration,
    DetectorTrajectory,
    calibrate_threshold,
    false_alarm_rate,
)
from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EarlyWarningEvidence,
    seal_critical_slowing_down_alarm,
)
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import CRITICAL_SLOWING_DOWN

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

FloatArray = NDArray[np.float64]

#: The one detector this harness runs — the label under which its threshold, achieved
#: false-alarm rate, and sealed evidence are keyed, matching the multi-node harness.
DETECTOR = CRITICAL_SLOWING_DOWN

__all__ = [
    "DEFAULT_BASELINE_FRACTION",
    "DEFAULT_PERSISTENCE",
    "DEFAULT_RELATIVE_GATE",
    "DEFAULT_STEP",
    "DEFAULT_TARGET_FALSE_ALARM",
    "DEFAULT_WINDOW",
    "DETECTOR",
    "SingleSeriesObservable",
    "SingleSeriesResult",
    "calibrate_single_series",
    "critical_slowing_down_trajectory",
    "evaluate_single_series",
    "null_series_trials",
    "single_series_verdict",
    "slice_series",
]


# --------------------------------------------------------------------------- #
# The scalar observable and its segmentation                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SingleSeriesObservable:
    """One scalar early-warning observable sampled at a fixed rate.

    The neutral input to the single-series harness: a palaeoclimate proxy, an
    ecological density, or any one quantity whose rising variance and lag-one
    autocorrelation the critical-slowing-down detector reads. It plays the role
    :class:`~scpn_phase_orchestrator.monitor.early_warning_suite.SuiteObservables`
    plays for the multi-node harness, but carries a single series because there is
    no population to synchronise.

    Attributes
    ----------
    series : FloatArray
        The scalar observable, shape ``(T,)`` with at least three finite samples
        (the shortest window that admits a lag-one autocorrelation).
    sampling_rate_hz : float
        Sampling rate of ``series`` in hertz; converts a sample lead into seconds.
        For an irregularly-timed palaeo record this is the resampled cadence.
    """

    series: FloatArray
    sampling_rate_hz: float

    def __post_init__(self) -> None:
        """Validate and normalise the series to a contiguous 1-D float array."""
        array = np.asarray(self.series)
        if np.iscomplexobj(array):
            raise ValueError("series must be real-valued")
        try:
            values = array.astype(np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("series must be a real float array") from exc
        if values.ndim != 1:
            raise ValueError(f"series must be one-dimensional, got shape {array.shape}")
        if values.shape[0] < 3:
            raise ValueError("series must have at least three samples")
        if not np.all(np.isfinite(values)):
            raise ValueError("series must contain only finite values")
        rate = float(self.sampling_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("sampling_rate_hz must be a positive finite number")
        object.__setattr__(self, "series", np.ascontiguousarray(values))
        object.__setattr__(self, "sampling_rate_hz", rate)

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the series."""
        return int(self.series.shape[0])


def slice_series(
    observable: SingleSeriesObservable, *, start: int, stop: int
) -> SingleSeriesObservable:
    """Return the observable restricted to the half-open sample range.

    Slicing an already-loaded series keeps the analysis on a chosen interval — the
    fixed pre-onset segment of a transition, or a null trial cut from a
    no-transition stretch — without reloading the record.

    Parameters
    ----------
    observable : SingleSeriesObservable
        The series to slice.
    start, stop : int
        Half-open sample range ``[start, stop)``; ``0 <= start < stop <= n``.

    Returns
    -------
    SingleSeriesObservable
        The restricted series at the same sampling rate.

    Raises
    ------
    ValueError
        If the range is malformed or exceeds the series length.
    """
    start_int = _non_negative_int(start, "start")
    stop_int = _non_negative_int(stop, "stop")
    n_samples = observable.n_samples
    if start_int >= stop_int:
        raise ValueError(f"start {start_int} must be below stop {stop_int}")
    if stop_int > n_samples:
        raise ValueError(f"stop {stop_int} exceeds the series length {n_samples}")
    return SingleSeriesObservable(
        series=observable.series[start_int:stop_int],
        sampling_rate_hz=observable.sampling_rate_hz,
    )


def null_series_trials(
    baseline_observables: Sequence[SingleSeriesObservable],
    *,
    segment_samples: int,
) -> list[SingleSeriesObservable]:
    """Cut each no-transition series into non-overlapping null trials.

    Every trial is the same length as a transition's pre-onset analysis segment, so
    the false-alarm rate is estimated over many comparable trials rather than a few
    whole records — the same matched-false-alarm discipline as the multi-node
    harness's :func:`~bench.early_warning_domain.null_trials`.

    Parameters
    ----------
    baseline_observables : sequence of SingleSeriesObservable
        Transition-free series to segment (a stationary stretch of the same
        observable, well before any tipping point).
    segment_samples : int
        Trial length in samples; must be a positive integer.

    Returns
    -------
    list[SingleSeriesObservable]
        The non-overlapping trials, in record then time order.

    Raises
    ------
    ValueError
        If ``segment_samples`` is not a positive integer.
    """
    length = _positive_int(segment_samples, "segment_samples")
    trials: list[SingleSeriesObservable] = []
    for observable in baseline_observables:
        n_samples = observable.n_samples
        for start in range(0, n_samples - length + 1, length):
            trials.append(slice_series(observable, start=start, stop=start + length))
    return trials


# --------------------------------------------------------------------------- #
# Detector trajectory and matched-false-alarm calibration                      #
# --------------------------------------------------------------------------- #


def critical_slowing_down_trajectory(
    observable: SingleSeriesObservable,
    *,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    persistence: int = DEFAULT_PERSISTENCE,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> DetectorTrajectory:
    """Run the detector once at a zero gate and return its oriented trajectory.

    The critical-slowing-down detector is run with a zero z-threshold and zero
    rise-threshold, so its per-window ``combined_z`` (the larger of the variance and
    autocorrelation robust z-scores) is a threshold-free oriented score that
    calibration and lead measurement then apply a threshold to. Returned as a
    :class:`~bench.early_warning_domain.DetectorTrajectory` so the multi-node
    harness's matched-false-alarm machinery scores it unchanged.

    Parameters
    ----------
    observable : SingleSeriesObservable
        The scalar series to sweep.
    window, step : int
        Analysis window length and hop in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit the baseline.
    persistence : int
        Consecutive breaching windows required to alarm; carried for symmetry and
        applied at scoring time.
    relative_gate : float
        Fractional-change gate carried on the trajectory, applied alongside the
        calibrated threshold exactly as the detector applies its own rise gate.

    Returns
    -------
    DetectorTrajectory
        The detector's oriented score, relative rise, window grid, and baseline
        length, keyed under :data:`DETECTOR`.
    """
    warning = critical_slowing_down_warning(
        observable.series[np.newaxis, :],
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=persistence,
    )
    return DetectorTrajectory(
        name=DETECTOR,
        score=np.asarray(warning.combined_z, dtype=np.float64),
        relative=np.asarray(warning.relative_rise, dtype=np.float64),
        relative_gate=relative_gate,
        window_starts=np.asarray(warning.window_starts, dtype=np.int64),
        n_baseline=warning.n_baseline_windows,
    )


def calibrate_single_series(
    null_observables: Sequence[SingleSeriesObservable],
    *,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    persistence: int = DEFAULT_PERSISTENCE,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> Calibration:
    """Calibrate the detector to a matched false alarm on the no-transition null.

    Each null trial's critical-slowing-down trajectory is scored, and the continuous
    :func:`~bench.early_warning_domain.calibrate_threshold` sets the tightest
    threshold holding the trial false-alarm rate at or below ``target_fa`` — the same
    ceiling-free matched-false-alarm calibration the multi-node harness uses. The
    achieved rate is reported alongside so a null that cannot be held at target
    (a degenerate corpus) is visible rather than hidden.

    Parameters
    ----------
    null_observables : sequence of SingleSeriesObservable
        Transition-free trials forming the false-alarm null.
    target_fa : float
        Target false-alarm rate the detector is held at or below.
    persistence, window, step, baseline_fraction, relative_gate :
        Analysis parameters forwarded to :func:`critical_slowing_down_trajectory`.

    Returns
    -------
    Calibration
        The matched-false-alarm threshold and achieved rate, keyed under
        :data:`DETECTOR`.

    Raises
    ------
    ValueError
        If the null ensemble is empty.
    """
    if not null_observables:
        raise ValueError("null_observables must not be empty")
    trajectories = [
        critical_slowing_down_trajectory(
            observable,
            window=window,
            step=step,
            baseline_fraction=baseline_fraction,
            persistence=persistence,
            relative_gate=relative_gate,
        )
        for observable in null_observables
    ]
    threshold = calibrate_threshold(
        trajectories, target_fa=target_fa, persistence=persistence
    )
    achieved = false_alarm_rate(trajectories, threshold, persistence=persistence)
    return Calibration(
        thresholds={DETECTOR: threshold},
        achieved_false_alarm={DETECTOR: achieved},
    )


# --------------------------------------------------------------------------- #
# Per-transition sealing (re-runs the detector at its calibrated threshold)     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SingleSeriesResult:
    """The sealed evidence and matched-false-alarm lead for one transition.

    Attributes
    ----------
    record_id : str
        The corpus record label, e.g. ``glaciation_III`` or ``yeast_dilution_1103``.
    onset_sample : int
        Annotated transition onset in analysis samples.
    evidence : EarlyWarningEvidence
        The sealed critical-slowing-down record — a sealed silence when the detector
        did not fire, an alarm with its honest lead when it did.
    """

    record_id: str
    onset_sample: int
    evidence: EarlyWarningEvidence

    def lead_seconds(self) -> float | None:
        """Return the detector's honest lead in seconds, or ``None`` if not early.

        Returns
        -------
        float | None
            The lead in seconds when the alarm was strictly before the onset, else
            ``None`` (a sealed silence or a late alarm).
        """
        return self.evidence.lead_seconds if self.evidence.lead_is_early else None

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the sealed record.

        Returns
        -------
        dict[str, object]
            The record label, the onset sample, and the sealed detector evidence's
            :meth:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence.to_audit_record`.
        """
        return {
            "record_id": self.record_id,
            "onset_sample": self.onset_sample,
            "detector": self.evidence.to_audit_record(),
        }


def evaluate_single_series(
    observable: SingleSeriesObservable,
    *,
    record_id: str,
    onset_sample: int,
    signal_source: str,
    captured_at: str,
    threshold: float,
    observable_description: str,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    persistence: int = DEFAULT_PERSISTENCE,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> SingleSeriesResult:
    """Run the detector at its calibrated threshold and seal the alarm.

    The critical-slowing-down detector is re-run with the matched-false-alarm
    ``threshold`` as its z-gate and ``relative_gate`` as its rise gate, so the sealed
    :class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`
    records the alarm decision at the calibrated operating point and an honest lead
    against ``onset_sample``. Because the detector's per-window fields do not depend
    on the gates, this alarm is the one calibration measured.

    Parameters
    ----------
    observable : SingleSeriesObservable
        The transition series' pre-onset segment.
    record_id : str
        Corpus record label, carried into the result.
    onset_sample : int
        Annotated transition onset in analysis samples.
    signal_source, captured_at : str
        Provenance forwarded into the sealed record.
    threshold : float
        The matched-false-alarm z-threshold from :func:`calibrate_single_series`.
    observable_description : str
        The domain observable description, sealed verbatim so the record names
        exactly what was screened.
    window, step, baseline_fraction, persistence, relative_gate :
        Analysis parameters; ``window``/``step``/``baseline_fraction``/``persistence``
        must match calibration, and ``relative_gate`` is the detector's rise gate.

    Returns
    -------
    SingleSeriesResult
        The sealed record and its provenance.
    """
    warning = critical_slowing_down_warning(
        observable.series[np.newaxis, :],
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=threshold,
        rise_threshold=relative_gate,
        persistence=persistence,
    )
    evidence = seal_critical_slowing_down_alarm(
        warning,
        observable=observable_description,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=observable.sampling_rate_hz,
        transition_onset_sample=onset_sample,
    )
    return SingleSeriesResult(
        record_id=record_id, onset_sample=onset_sample, evidence=evidence
    )


# --------------------------------------------------------------------------- #
# Verdict                                                                      #
# --------------------------------------------------------------------------- #


def single_series_verdict(
    leads: Mapping[str, float | None],
    n_transitions: int,
    *,
    noun: str = "transitions",
    singular: str = "transition",
) -> str:
    """Return the honest matched-false-alarm verdict across the transitions.

    With one detector there is no fusion to claim an advantage for, so the verdict
    states the detection count first and reads the lead as evidence, not a
    prediction: a lead on one or two records is not a robust precursor, and the
    auditable sealed evidence — including the sealed silences — is the deliverable
    regardless of how many records the detector leads.

    Parameters
    ----------
    leads : Mapping[str, float | None]
        Per-record lead in seconds, ``None`` where the detector did not lead. The
        keys are the evaluated record labels.
    n_transitions : int
        Number of evaluated transitions the count is out of.
    noun : str
        Plural noun for the transition kind, e.g. ``collapses`` or ``glaciations``.
    singular : str
        Singular of ``noun``.

    Returns
    -------
    str
        A verdict sentence leading with the critical-slowing-down detection count.
    """
    led_values = [lead for lead in leads.values() if lead is not None]
    led = len(led_values)
    if led == 0:
        return (
            "NO EARLY WARNING: at a matched false-alarm rate critical slowing down "
            f"leads none of the {n_transitions} evaluated {noun}. Detection is a "
            "commodity here; the auditable sealed evidence, not a lead, is the "
            "deliverable."
        )
    median_lead = float(np.median(led_values))
    subject = "one " + singular if led == 1 else f"{led} {noun}"
    return (
        "SINGLE-INDICATOR DETECTION: at a matched false-alarm rate critical slowing "
        f"down leads {led}/{n_transitions} {noun} (median lead {median_lead:.0f} s). "
        f"A lead on {subject} is evidence, not a robust precursor; consistent with "
        "detection as a commodity, the auditable sealed evidence — including the "
        "sealed silences — is the deliverable."
    )


# --------------------------------------------------------------------------- #
# Validators                                                                   #
# --------------------------------------------------------------------------- #


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _non_negative_int(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {result}")
    return result
