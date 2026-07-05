# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — domain-neutral matched-false-alarm lead-time harness

"""Domain-neutral matched-false-alarm lead-time harness for the warning suite.

The early-warning suite (:mod:`scpn_phase_orchestrator.monitor.early_warning_suite`)
reads a neutral ``SuiteObservables`` bundle, so the machinery that *validates* it
on a labelled corpus is neutral too:
segment a recording, calibrate each detector to a matched false-alarm rate on a
no-transition null, measure the honest lead of each alarm against an annotated
onset, and seal every alarm — or silence — into an
:class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`.
None of that is domain-specific. This module is that harness; a per-domain
capstone (scalp EEG, cardiac ECG, grid PMU) supplies an adapter that produces the
bundle and the provenance strings, then reuses everything here.

The design keeps two paths in lock-step so calibration and the final seal agree:

* :func:`detector_trajectories` runs the suite once at a zero gate, yielding each
  detector's threshold-free oriented score trajectory (the slow ordinal-entropy
  sweep runs once per recording, not once per grid point). :func:`calibrate_detectors`
  then searches a threshold grid for the smallest gate holding the trial
  false-alarm rate at or below the target.
* :func:`evaluate_seizure` re-runs the suite through
  :func:`~scpn_phase_orchestrator.monitor.early_warning_suite.run_early_warning_suite`
  at those calibrated thresholds and seals each member and the fusion, so the
  sealed alarm decision is the one at the matched operating point.

The alarm rule — past the baseline, oriented score at or above the threshold, and
the fractional-change gate held, sustained for a persistence run — is the suite's
shared contract, applied identically by calibration and lead measurement
(:func:`_alarm_sample`). **The gain from fusion is reported as improved
matched-false-alarm lead, never as a raw detection rate**; :func:`domain_verdict`
enforces that reading.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals for
  critical transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EarlyWarningEvidence,
    seal_critical_slowing_down_alarm,
    seal_ensemble_alarm,
    seal_synchronisation_alarm,
    seal_transition_entropy_alarm,
)
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    CRITICAL_SLOWING_DOWN,
    ENSEMBLE_WEIGHTED,
    SUITE_DETECTORS,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    SuiteObservables,
    run_early_warning_suite,
)
from scpn_phase_orchestrator.monitor.ensemble_warning import (
    WEIGHTED_RULE,
    ensemble_warning,
    member_from_critical_slowing_down,
    member_from_synchronisation,
    member_from_transition_entropy,
)
from scpn_phase_orchestrator.monitor.explosive_sync import explosive_sync_warning
from scpn_phase_orchestrator.monitor.synchronisation import synchronisation_warning

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Detector labels in report order — the three suite members then the fusion.
DETECTORS = SUITE_DETECTORS

#: Default analysis window length in samples.
DEFAULT_WINDOW = 128
#: Default hop between consecutive windows in samples.
DEFAULT_STEP = 16
#: Default leading fraction of windows used to fit each detector's baseline.
DEFAULT_BASELINE_FRACTION = 0.25
#: Default consecutive breaching windows required to raise an alarm.
DEFAULT_PERSISTENCE = 2
#: Default minimum fractional change gate shared by the member detectors.
DEFAULT_RELATIVE_GATE = 0.05
#: Default target false-alarm rate the detectors are calibrated to on the null.
DEFAULT_TARGET_FALSE_ALARM = 0.10
#: Default threshold grid (0.25 … 10.0) the calibration searches.
DEFAULT_THRESHOLD_GRID = tuple(round(0.25 * k, 2) for k in range(1, 41))

__all__ = [
    "DEFAULT_BASELINE_FRACTION",
    "DEFAULT_PERSISTENCE",
    "DEFAULT_RELATIVE_GATE",
    "DEFAULT_STEP",
    "DEFAULT_TARGET_FALSE_ALARM",
    "DEFAULT_THRESHOLD_GRID",
    "DEFAULT_WINDOW",
    "DETECTORS",
    "DetectorTrajectory",
    "SeizureLeadResult",
    "calibrate_detectors",
    "calibrate_threshold",
    "detector_trajectories",
    "domain_verdict",
    "evaluate_seizure",
    "false_alarm_rate",
    "null_trials",
    "seizure_lead_samples",
    "slice_observables",
]


# --------------------------------------------------------------------------- #
# Segmentation (neutral over the observable bundle)                            #
# --------------------------------------------------------------------------- #


def slice_observables(
    observables: SuiteObservables, *, start: int, stop: int
) -> SuiteObservables:
    """Return the observables restricted to the half-open sample range.

    Slicing an already-derived observable keeps the analysis on a chosen interval
    — the fixed pre-onset segment of a transition, or a null trial cut from a
    no-transition recording — without re-running the domain pipeline.

    Parameters
    ----------
    observables : SuiteObservables
        The field to slice.
    start, stop : int
        Half-open sample range ``[start, stop)``; ``0 <= start < stop <= n``.

    Returns
    -------
    SuiteObservables
        The restricted field at the same sampling rate.

    Raises
    ------
    ValueError
        If the range is malformed or exceeds the field length.
    """
    start_int = _non_negative_int(start, "start")
    stop_int = _non_negative_int(stop, "stop")
    n_samples = observables.n_samples
    if start_int >= stop_int:
        raise ValueError(f"start {start_int} must be below stop {stop_int}")
    if stop_int > n_samples:
        raise ValueError(f"stop {stop_int} exceeds the field length {n_samples}")
    return SuiteObservables(
        phases=np.ascontiguousarray(
            observables.phases[:, start_int:stop_int], dtype=np.float64
        ),
        phase_field=np.ascontiguousarray(
            observables.phase_field[:, start_int:stop_int], dtype=np.float64
        ),
        order_parameter=np.ascontiguousarray(
            observables.order_parameter[start_int:stop_int], dtype=np.float64
        ),
        sampling_rate_hz=observables.sampling_rate_hz,
    )


def null_trials(
    interictal_observables: Sequence[SuiteObservables],
    *,
    segment_samples: int,
) -> list[SuiteObservables]:
    """Cut each no-transition recording into non-overlapping null trials.

    Every trial has the same length as a transition's pre-onset analysis segment,
    so the false-alarm rate is estimated over many comparable trials rather than a
    handful of whole recordings — a finer, fairer matched-false-alarm calibration.

    Parameters
    ----------
    interictal_observables : sequence of SuiteObservables
        Transition-free recordings to segment.
    segment_samples : int
        Trial length in samples; must be a positive integer.

    Returns
    -------
    list[SuiteObservables]
        The non-overlapping trials, in recording then time order.

    Raises
    ------
    ValueError
        If ``segment_samples`` is not a positive integer.
    """
    length = _positive_int(segment_samples, "segment_samples")
    trials: list[SuiteObservables] = []
    for observables in interictal_observables:
        n_samples = observables.n_samples
        for start in range(0, n_samples - length + 1, length):
            trials.append(
                slice_observables(observables, start=start, stop=start + length)
            )
    return trials


# --------------------------------------------------------------------------- #
# Detector trajectories and the shared matched-false-alarm machinery           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectorTrajectory:
    """A detector's oriented per-window score and gate on one recording.

    ``score`` is signed so larger always means more anomalous (the entropy drop
    is negated), and ``relative`` with ``relative_gate`` is the fractional-change
    guard applied alongside the calibrated threshold — the same gate the detector
    applies internally, so calibration and the final seal agree.
    """

    name: str
    score: FloatArray = field(repr=False)
    relative: FloatArray = field(repr=False)
    relative_gate: float
    window_starts: IntArray = field(repr=False)
    n_baseline: int


def detector_trajectories(
    observables: SuiteObservables,
    *,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    persistence: int = DEFAULT_PERSISTENCE,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> dict[str, DetectorTrajectory]:
    """Run the suite once at a zero gate and return each detector's trajectory.

    The three members are run with a zero z-threshold so their oriented z-score
    trajectories are threshold-free; the fused score is their weighted mean.
    Calibration and lead measurement then apply a threshold to these
    trajectories, so the slow ordinal-entropy sweep runs once per recording
    rather than once per grid point.

    Parameters
    ----------
    observables : SuiteObservables
        The neutral observable field.
    window, step : int
        Analysis window length and hop in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit each baseline.
    persistence : int
        Echoed for symmetry; the alarm run length is applied at scoring time.
    relative_gate : float
        Fractional-change gate carried on the member trajectories.

    Returns
    -------
    dict[str, DetectorTrajectory]
        A trajectory per label in :data:`DETECTORS`, all on one window grid.
    """
    csd = critical_slowing_down_warning(
        observables.order_parameter[np.newaxis, :],
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=persistence,
    )
    sync = synchronisation_warning(
        observables.phases,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=persistence,
    )
    entropy = explosive_sync_warning(
        observables.phase_field,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=0.0,
        drop_threshold=0.0,
        persistence=persistence,
    )
    members = [
        member_from_critical_slowing_down(csd),
        member_from_synchronisation(sync),
        member_from_transition_entropy(entropy),
    ]
    fused = ensemble_warning(
        members,
        rule=WEIGHTED_RULE,
        fused_threshold=0.0,
        persistence=persistence,
    )
    ones = np.ones(int(fused.fused_score.shape[0]), dtype=np.float64)
    return {
        CRITICAL_SLOWING_DOWN: DetectorTrajectory(
            name=CRITICAL_SLOWING_DOWN,
            score=np.asarray(csd.combined_z, dtype=np.float64),
            relative=np.asarray(csd.relative_rise, dtype=np.float64),
            relative_gate=relative_gate,
            window_starts=np.asarray(csd.window_starts, dtype=np.int64),
            n_baseline=csd.n_baseline_windows,
        ),
        SYNCHRONISATION: DetectorTrajectory(
            name=SYNCHRONISATION,
            score=np.asarray(sync.robust_z, dtype=np.float64),
            relative=np.asarray(sync.relative_rise, dtype=np.float64),
            relative_gate=relative_gate,
            window_starts=np.asarray(sync.window_starts, dtype=np.int64),
            n_baseline=sync.n_baseline_windows,
        ),
        TRANSITION_ENTROPY: DetectorTrajectory(
            name=TRANSITION_ENTROPY,
            score=-np.asarray(entropy.robust_z, dtype=np.float64),
            relative=np.asarray(entropy.relative_drop, dtype=np.float64),
            relative_gate=relative_gate,
            window_starts=np.asarray(entropy.window_starts, dtype=np.int64),
            n_baseline=entropy.n_baseline_windows,
        ),
        ENSEMBLE_WEIGHTED: DetectorTrajectory(
            name=ENSEMBLE_WEIGHTED,
            score=np.asarray(fused.fused_score, dtype=np.float64),
            relative=ones,
            relative_gate=0.0,
            window_starts=np.asarray(fused.window_starts, dtype=np.int64),
            n_baseline=fused.n_baseline_windows,
        ),
    }


def _alarm_sample(
    trajectory: DetectorTrajectory, *, threshold: float, persistence: int
) -> int | None:
    """Return the sample of the first sustained breach, or ``None``.

    A window breaches when it is past the baseline, its oriented score meets the
    threshold, and its relative change meets the trajectory's gate; ``persistence``
    consecutive breaches raise the alarm. This is the one rule shared by
    calibration and lead measurement, and it reproduces each detector's own gate.
    """
    n_windows = int(trajectory.score.shape[0])
    past_baseline = np.arange(n_windows) >= trajectory.n_baseline
    breaches = (
        past_baseline
        & (trajectory.score >= threshold)
        & (trajectory.relative >= trajectory.relative_gate)
    )
    run = 0
    for index in range(n_windows):
        if breaches[index]:
            run += 1
            if run >= persistence:
                start = index - persistence + 1
                return int(trajectory.window_starts[start])
        else:
            run = 0
    return None


def false_alarm_rate(
    null_trajectories: Sequence[DetectorTrajectory],
    threshold: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> float:
    """Return the fraction of no-transition null trajectories that alarm.

    Parameters
    ----------
    null_trajectories : sequence of DetectorTrajectory
        One detector's trajectory on each null trial.
    threshold : float
        Oriented-score gate applied to the trajectories.
    persistence : int
        Consecutive breaching windows required to alarm.

    Returns
    -------
    float
        The fraction of null trials that raise an alarm.

    Raises
    ------
    ValueError
        If ``null_trajectories`` is empty.
    """
    if not null_trajectories:
        raise ValueError("null_trajectories must not be empty")
    alarms = sum(
        _alarm_sample(trajectory, threshold=threshold, persistence=persistence)
        is not None
        for trajectory in null_trajectories
    )
    return alarms / len(null_trajectories)


def calibrate_threshold(
    null_trajectories: Sequence[DetectorTrajectory],
    *,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    persistence: int = DEFAULT_PERSISTENCE,
    grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> float:
    """Return the smallest grid threshold whose null false-alarm rate ≤ target.

    Falls back to the largest grid threshold when even that exceeds the target,
    so the null ensemble can never leave the detector uncalibrated.

    Parameters
    ----------
    null_trajectories : sequence of DetectorTrajectory
        One detector's trajectory on each null trial.
    target_fa : float
        Target false-alarm rate the detector is held at or below.
    persistence : int
        Consecutive breaching windows required to alarm.
    grid : sequence of float
        Ascending threshold grid searched for the matched rate.

    Returns
    -------
    float
        The matched-false-alarm threshold.

    Raises
    ------
    ValueError
        If ``null_trajectories`` is empty.
    """
    if not null_trajectories:
        raise ValueError("null_trajectories must not be empty")
    for threshold in grid:
        if (
            false_alarm_rate(null_trajectories, threshold, persistence=persistence)
            <= target_fa
        ):
            return float(threshold)
    return float(grid[-1])


def calibrate_detectors(
    null_observables: Sequence[SuiteObservables],
    *,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    persistence: int = DEFAULT_PERSISTENCE,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> dict[str, float]:
    """Calibrate every detector to a matched false alarm on the no-transition null.

    Parameters
    ----------
    null_observables : sequence of SuiteObservables
        Transition-free recordings forming the false-alarm null.
    target_fa : float
        Target false-alarm rate each detector is held at or below.
    persistence, window, step, baseline_fraction, relative_gate :
        Suite analysis parameters, forwarded to :func:`detector_trajectories`.

    Returns
    -------
    dict[str, float]
        The matched-false-alarm threshold for each label in :data:`DETECTORS`.

    Raises
    ------
    ValueError
        If the null ensemble is empty.
    """
    if not null_observables:
        raise ValueError("null_observables must not be empty")
    per_detector: dict[str, list[DetectorTrajectory]] = {name: [] for name in DETECTORS}
    for observables in null_observables:
        trajectories = detector_trajectories(
            observables,
            window=window,
            step=step,
            baseline_fraction=baseline_fraction,
            persistence=persistence,
            relative_gate=relative_gate,
        )
        for name, trajectory in trajectories.items():
            per_detector[name].append(trajectory)
    return {
        name: calibrate_threshold(
            trajectories, target_fa=target_fa, persistence=persistence
        )
        for name, trajectories in per_detector.items()
    }


def seizure_lead_samples(
    trajectory: DetectorTrajectory,
    *,
    onset_sample: int,
    threshold: float,
    persistence: int = DEFAULT_PERSISTENCE,
) -> int | None:
    """Return the leading alarm's lead in samples, or ``None`` if not leading.

    A detection is an alarm at or before the annotated onset; its lead is
    ``onset_sample − alarm_sample``. An alarm after the onset (or no alarm) is
    not a lead and returns ``None``.
    """
    alarm = _alarm_sample(trajectory, threshold=threshold, persistence=persistence)
    if alarm is None or alarm > onset_sample:
        return None
    return onset_sample - alarm


# --------------------------------------------------------------------------- #
# Per-transition sealing (re-runs the suite at its calibrated thresholds)       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeizureLeadResult:
    """The sealed evidence and matched-false-alarm leads for one transition.

    Attributes
    ----------
    record_id : str
        The corpus record label, e.g. ``chb01_03``.
    onset_sample : int
        Annotated transition onset in analysis samples.
    evidences : dict[str, EarlyWarningEvidence]
        A sealed record per label in :data:`DETECTORS` — including a sealed
        silence when a detector did not fire.
    """

    record_id: str
    onset_sample: int
    evidences: dict[str, EarlyWarningEvidence]

    def lead_seconds(self) -> dict[str, float | None]:
        """Return each detector's honest lead in seconds (``None`` if it was late).

        Returns
        -------
        dict[str, float | None]
            One ``label -> lead_seconds`` entry per detector, ``None`` when the
            alarm was not early (a sealed silence or a late alarm).
        """
        return {
            name: (evidence.lead_seconds if evidence.lead_is_early else None)
            for name, evidence in self.evidences.items()
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the sealed per-detector evidence.

        Returns
        -------
        dict[str, object]
            The record label, the onset sample, and each detector's sealed
            :meth:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence.to_audit_record`.
        """
        return {
            "record_id": self.record_id,
            "onset_sample": self.onset_sample,
            "detectors": {
                name: evidence.to_audit_record()
                for name, evidence in self.evidences.items()
            },
        }


def evaluate_seizure(
    observables: SuiteObservables,
    *,
    record_id: str,
    onset_sample: int,
    signal_source: str,
    captured_at: str,
    thresholds: Mapping[str, float],
    observable_descriptions: Mapping[str, str],
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    persistence: int = DEFAULT_PERSISTENCE,
    relative_gate: float = DEFAULT_RELATIVE_GATE,
) -> SeizureLeadResult:
    """Run the suite at the calibrated thresholds and seal each detector's alarm.

    The suite is re-run through
    :func:`~scpn_phase_orchestrator.monitor.early_warning_suite.run_early_warning_suite`
    with the matched-false-alarm thresholds as the gates, so the sealed
    :class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`
    records the alarm decision at the calibrated operating point and an honest
    lead against ``onset_sample``.

    Parameters
    ----------
    observables : SuiteObservables
        The transition recording's neutral observables.
    record_id : str
        Corpus record label, carried into the result.
    onset_sample : int
        Annotated transition onset in analysis samples.
    signal_source, captured_at : str
        Provenance forwarded into every sealed record.
    thresholds : Mapping[str, float]
        The matched-false-alarm threshold per label in :data:`DETECTORS`.
    observable_descriptions : Mapping[str, str]
        The domain observable description per label in :data:`DETECTORS`, sealed
        verbatim so each record names exactly what was screened.
    window, step, baseline_fraction, persistence, relative_gate :
        Suite analysis parameters.

    Returns
    -------
    SeizureLeadResult
        The four sealed records and the recording provenance.

    Raises
    ------
    KeyError
        If ``thresholds`` or ``observable_descriptions`` is missing a label.
    """
    warnings = run_early_warning_suite(
        observables,
        thresholds=thresholds,
        relative_gate=relative_gate,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        persistence=persistence,
    )
    fs = observables.sampling_rate_hz
    evidences = {
        CRITICAL_SLOWING_DOWN: seal_critical_slowing_down_alarm(
            warnings.critical_slowing_down,
            observable=observable_descriptions[CRITICAL_SLOWING_DOWN],
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        SYNCHRONISATION: seal_synchronisation_alarm(
            warnings.synchronisation,
            observable=observable_descriptions[SYNCHRONISATION],
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        TRANSITION_ENTROPY: seal_transition_entropy_alarm(
            warnings.transition_entropy,
            observable=observable_descriptions[TRANSITION_ENTROPY],
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        ENSEMBLE_WEIGHTED: seal_ensemble_alarm(
            warnings.ensemble,
            observable=observable_descriptions[ENSEMBLE_WEIGHTED],
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            window=window,
            step=step,
            transition_onset_sample=onset_sample,
        ),
    }
    return SeizureLeadResult(
        record_id=record_id, onset_sample=onset_sample, evidences=evidences
    )


# --------------------------------------------------------------------------- #
# Verdict                                                                      #
# --------------------------------------------------------------------------- #


def domain_verdict(
    leads_by_detector: Mapping[str, list[float]],
    n_transitions: int,
    *,
    noun: str = "transitions",
    singular: str = "transition",
) -> str:
    """Return the honest matched-false-alarm verdict across the transitions.

    The headline is the detection count first, then the lead: a lead measured on
    one transition is not a robust advantage however long it is, so a fusion
    advantage is named only when the fusion *leads more transitions* than every
    single member. When detection is sparse — or no detector leads any transition
    — the verdict says so plainly, since the auditable moat holds regardless of
    which detector leads or whether any does.

    Parameters
    ----------
    leads_by_detector : Mapping[str, list[float]]
        Per-label list of the leads (seconds) each detector achieved.
    n_transitions : int
        Number of evaluated transitions the counts are out of.
    noun : str
        Plural noun for the transition kind, e.g. ``seizures`` for the scalp-EEG
        capstone; the verdict reads naturally in each domain.
    singular : str
        Singular of ``noun``, e.g. ``seizure``.

    Returns
    -------
    str
        A verdict sentence leading with the detection count per detector.
    """
    led = {name: len(leads_by_detector.get(name, [])) for name in DETECTORS}
    median = {
        name: (float(np.median(leads_by_detector[name])) if led[name] else None)
        for name in DETECTORS
    }
    members = {name: led[name] for name in DETECTORS if name != ENSEMBLE_WEIGHTED}
    detail = "; ".join(
        f"{name} {led[name]}/{n_transitions}"
        + (f" (median lead {median[name]:.0f} s)" if median[name] is not None else "")
        for name in DETECTORS
    )
    if all(count == 0 for count in led.values()):
        return (
            "NO EARLY WARNING: at a matched false-alarm rate no detector leads any "
            f"of the {n_transitions} evaluated {noun}. Detection is a commodity "
            "here; the auditable sealed evidence, not a lead, is the deliverable."
        )
    fusion_led = led[ENSEMBLE_WEIGHTED]
    if members and fusion_led > max(members.values()):
        return (
            f"FUSION DETECTS MORE: at a matched false-alarm rate the weighted "
            f"fusion leads {fusion_led}/{n_transitions} {noun}, more than any single "
            f"member ({detail}). The gain is more leading detections at fixed false "
            "alarm, not a spent false-alarm budget."
        )
    return (
        "SPARSE DETECTION, NO ROBUST ADVANTAGE: at a matched false-alarm rate "
        f"detection is sparse and no detector leads more {noun} than the fusion "
        f"({detail}). A longer lead on one {singular} is not a robust advantage; "
        "consistent with detection as a commodity, the auditable sealed evidence "
        "is the deliverable."
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
