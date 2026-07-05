# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real scalp-EEG early-warning lead-time capstone

"""Real scalp-EEG early-warning capstone: matched-false-alarm seizure lead time.

A fair synthetic head-to-head (``bench/early_warning_leadtime.py``) and the real
EEG check behind it established that *detection* is a commodity: no single
early-warning indicator beats the others by a decisive margin, so "our detector
warns earlier" is not a claim this platform can honestly make. What it *can*
claim — and what this capstone demonstrates on real data — is the auditable
envelope: the same three-member detector suite and its fusion, run on a public
scalp-EEG seizure corpus, calibrated to a matched false-alarm rate on genuine
interictal recordings, with every alarm (or silence) sealed into a
content-addressed, claim-bounded :class:`EarlyWarningEvidence` record carrying an
honest lead time.

Corpus
------
The CHB-MIT Scalp EEG Database (Shoeb 2009; Goldberger et al. 2000, PhysioNet):
23-channel bipolar scalp EEG sampled at 256 Hz, with clinician-annotated seizure
onset times. The raw EDF recordings are **citation-only**: they are downloaded
from PhysioNet and are **never redistributed here**. This module reads them, but
its tests exercise every signal-processing and evaluation path on **synthetic
arrays** so the coverage of the logic never depends on the protected corpus.

Observable pipeline
-------------------
Each recording is turned into one coherent decimated analytic-phase field, from
which all three suite members read complementary moments:

1. Zero-phase Butterworth band-pass, 4–30 Hz — the θ–β range that carries the
   pre-ictal synchronisation build-up, away from mains and drift.
2. Per-channel Hilbert analytic phase ``φ(t)``.
3. Decimation 256 → 32 Hz applied to the *continuous* components ``sin φ`` and
   ``cos φ`` (a wrapped phase must never be low-pass filtered — its ±π jumps are
   not band-limited), with the phase reconstructed by ``atan2``. All three
   observables are then derived from that single decimated field, so the members
   read one consistent representation rather than three independently filtered
   ones:

   * critical slowing down reads the cross-channel order parameter
     ``R(t) = |⟨e^{iφ}⟩|`` (second-moment variance / autocorrelation rise);
   * rising synchronisation reads the per-channel phases (first-moment coherence
     rise, its own internal ``R``);
   * ordinal-transition entropy reads the per-channel projection ``sin φ`` (a
     regularisation drop).

Window 128, step 16 at 32 Hz — a 4-second window hopped every 0.5 s.

Fixed pre-onset segment — a clean baseline
------------------------------------------
Each seizure is evaluated on a fixed segment ending at onset: a leading
``BASELINE_SECONDS`` baseline followed by a ``HORIZON_SECONDS`` detection window
(:data:`SEGMENT_SECONDS` total). The baseline is a fixed pre-onset interval, not
a fraction of the whole recording, so an early-onset seizure cannot contaminate
its own baseline with ictal samples; the onset sits at the segment end, so every
window is pre-ictal and any alarm is a genuine lead. A seizure whose onset is
earlier than :data:`SEGMENT_SECONDS` cannot form a clean baseline and is
**excluded and reported as such**, never counted as a silent null.

Matched false alarm — the honest comparison
--------------------------------------------
A lead time is only meaningful at a fixed false-alarm rate, and one ictal file
cannot self-calibrate its own. Each interictal recording is cut into many
non-overlapping null trials of the *same* length and baseline/horizon structure
as a seizure's pre-onset segment (:func:`null_trials`), giving a fine
false-alarm estimate rather than a handful of whole-recording trials. Each
detector's threshold is the smallest that holds the trial false-alarm rate at or
below :data:`TARGET_FALSE_ALARM`; only then is its lead — ``onset − alarm`` in
samples, converted to seconds — measured on the seizures. **The gain from fusion
is reported as improved matched-false-alarm lead, never as a raw detection
rate**: an OR of the members trivially raises the rate by spending the
false-alarm budget. If the fusion does not beat the best single member at matched
false alarm, this capstone says so — a valid result, and the auditable moat holds
regardless.

Every evaluated seizure yields four sealed :class:`EarlyWarningEvidence` records
(three members + fusion), including a sealed *silence* when a detector does not
fire — the honest half of the moat. Only these derived, sealed artefacts are
committed, mirroring ``examples/real_data/iso_ne_case1/``; the raw EDF never is.

References
----------
* A. H. Shoeb, *Application of Machine Learning to Epileptic Seizure Onset
  Detection and Treatment*, PhD thesis, MIT, 2009 — the CHB-MIT corpus.
* A. L. Goldberger et al. 2000, *Circulation* 101(23):e215 — PhysioNet /
  PhysioBank.
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, decimate, hilbert, sosfiltfilt

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

#: Native sampling rate of the CHB-MIT scalp-EEG recordings, in hertz.
SAMPLING_RATE_HZ = 256.0
#: Decimation factor applied to the analytic-phase components (256 → 32 Hz).
DECIMATION = 8
#: Analysis sampling rate after decimation, in hertz.
DECIMATED_RATE_HZ = SAMPLING_RATE_HZ / DECIMATION
#: Band-pass passband in hertz — the θ–β range carrying pre-ictal synchronisation.
BAND_HZ = (4.0, 30.0)
#: Butterworth band-pass order.
FILTER_ORDER = 4

#: Analysis window length in samples (4 s at 32 Hz).
WINDOW = 128
#: Hop between consecutive windows in samples (0.5 s at 32 Hz).
STEP = 16
#: Leading fraction of windows used to fit each detector's baseline.
BASELINE_FRACTION = 0.25
#: Consecutive breaching windows required to raise an alarm.
PERSISTENCE = 2
#: Minimum fractional change gate shared by the member detectors.
RELATIVE_GATE = 0.05
#: Target false-alarm rate the detectors are calibrated to on the interictal null.
TARGET_FALSE_ALARM = 0.10
#: Threshold grid (0.25 … 10.0) the calibration searches for the matched rate.
THRESHOLD_GRID = tuple(round(0.25 * k, 2) for k in range(1, 41))

#: Leading baseline length, in seconds, of an analysis segment. Fixed (not a
#: fraction of the whole recording) so an early-onset seizure cannot contaminate
#: the baseline with ictal samples.
BASELINE_SECONDS = 300.0
#: Detection horizon, in seconds, before onset — the pre-onset window the suite
#: is allowed to warn within, and the maximum measurable lead.
HORIZON_SECONDS = 600.0
#: Full analysis-segment length (baseline + horizon), in seconds.
SEGMENT_SECONDS = BASELINE_SECONDS + HORIZON_SECONDS
#: Analysis-segment length in decimated samples.
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * DECIMATED_RATE_HZ)
#: Baseline fraction of an analysis segment (its leading ``BASELINE_SECONDS``).
SEGMENT_BASELINE_FRACTION = BASELINE_SECONDS / SEGMENT_SECONDS

#: Detector labels in report order — the three suite members then the fusion.
CRITICAL_SLOWING_DOWN = "critical_slowing_down"
SYNCHRONISATION = "synchronisation"
TRANSITION_ENTROPY = "transition_entropy"
ENSEMBLE_WEIGHTED = "ensemble_weighted"
DETECTORS = (
    CRITICAL_SLOWING_DOWN,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    ENSEMBLE_WEIGHTED,
)

_OBSERVABLE_CSD = (
    "cross-channel Kuramoto order parameter R(t) of scalp-EEG analytic phase "
    "(4-30 Hz, decimated to 32 Hz)"
)
_OBSERVABLE_SYNC = "per-channel scalp-EEG analytic phase (4-30 Hz, decimated to 32 Hz)"
_OBSERVABLE_ENTROPY = (
    "per-channel phase projection sin(phase) of scalp EEG (4-30 Hz, decimated to 32 Hz)"
)
_OBSERVABLE_ENSEMBLE = (
    "fused early-warning suite over scalp-EEG analytic phase "
    "(4-30 Hz, decimated to 32 Hz)"
)

#: Annotated seizure onset times (seconds into each CHB-MIT chb01 record).
SEIZURE_ONSETS_S: dict[str, int] = {
    "chb01_03": 2996,
    "chb01_04": 1467,
    "chb01_15": 1732,
    "chb01_16": 1015,
    "chb01_18": 1720,
    "chb01_21": 327,
    "chb01_26": 1862,
}
#: A few seizure-free chb01 records used as the interictal false-alarm null.
INTERICTAL_RECORDS: tuple[str, ...] = (
    "chb01_01",
    "chb01_02",
    "chb01_05",
    "chb01_06",
    "chb01_07",
)

__all__ = [
    "BAND_HZ",
    "DECIMATED_RATE_HZ",
    "DECIMATION",
    "DETECTORS",
    "FILTER_ORDER",
    "PERSISTENCE",
    "RELATIVE_GATE",
    "SAMPLING_RATE_HZ",
    "STEP",
    "TARGET_FALSE_ALARM",
    "THRESHOLD_GRID",
    "WINDOW",
    "DetectorTrajectory",
    "EEGObservables",
    "SeizureLeadResult",
    "analytic_phase",
    "bandpass",
    "calibrate_detectors",
    "calibrate_threshold",
    "decimate_analytic_phase",
    "detector_trajectories",
    "eeg_observables",
    "evaluate_seizure",
    "false_alarm_rate",
    "load_edf_channels",
    "null_trials",
    "seizure_lead_samples",
    "slice_observables",
]


# --------------------------------------------------------------------------- #
# Observable pipeline (pure — fully exercised on synthetic arrays)             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EEGObservables:
    """The decimated analytic-phase observables the detector suite reads.

    Attributes
    ----------
    phases : FloatArray
        Per-channel reconstructed analytic phase in radians, shape ``(N, T)`` at
        :attr:`sampling_rate_hz`; the rising-synchronisation input.
    phase_field : FloatArray
        Per-channel projection ``sin(phase)``, shape ``(N, T)``; the
        ordinal-transition-entropy input.
    order_parameter : FloatArray
        Cross-channel Kuramoto order parameter ``R(t) = |⟨e^{iφ}⟩|``, shape
        ``(T,)``; the critical-slowing-down input.
    sampling_rate_hz : float
        Analysis sampling rate after decimation, in hertz.
    """

    phases: FloatArray = field(repr=False)
    phase_field: FloatArray = field(repr=False)
    order_parameter: FloatArray = field(repr=False)
    sampling_rate_hz: float

    @property
    def n_channels(self) -> int:
        """Number of channels in the observable field."""
        return int(self.phases.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples per channel after decimation."""
        return int(self.phases.shape[1])


def bandpass(
    signals: FloatArray,
    *,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    band_hz: tuple[float, float] = BAND_HZ,
    order: int = FILTER_ORDER,
) -> FloatArray:
    """Zero-phase Butterworth band-pass along the last axis.

    Parameters
    ----------
    signals : FloatArray
        Per-channel samples, shape ``(N, T)``; a one-dimensional array is treated
        as a single channel.
    sampling_rate_hz : float
        Sampling rate of ``signals`` in hertz.
    band_hz : tuple[float, float]
        ``(low, high)`` passband edges in hertz, ``0 < low < high < Nyquist``.
    order : int
        Butterworth order (per edge).

    Returns
    -------
    FloatArray
        The band-passed signal, same shape as the validated input.

    Raises
    ------
    ValueError
        If the signal is malformed or the band is not a valid passband.
    """
    array = _validate_signals(signals, "signals")
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    order_int = _positive_int(order, "order")
    low, high = _validate_band(band_hz, fs)
    nyquist = 0.5 * fs
    sos = butter(order_int, [low / nyquist, high / nyquist], btype="band", output="sos")
    filtered = sosfiltfilt(sos, array, axis=-1)
    return np.ascontiguousarray(filtered, dtype=np.float64)


def analytic_phase(signals: FloatArray) -> FloatArray:
    """Return the per-channel Hilbert analytic phase in radians, shape ``(N, T)``.

    Parameters
    ----------
    signals : FloatArray
        Real band-passed samples, shape ``(N, T)`` (a one-dimensional array is a
        single channel).

    Returns
    -------
    FloatArray
        The instantaneous phase ``angle(hilbert(signals))``.

    Raises
    ------
    ValueError
        If the signal is malformed.
    """
    array = _validate_signals(signals, "signals")
    analytic = hilbert(array, axis=-1)
    return np.ascontiguousarray(np.angle(analytic), dtype=np.float64)


def decimate_analytic_phase(
    phases: FloatArray, *, factor: int = DECIMATION
) -> FloatArray:
    """Anti-aliased decimation of a phase field via its ``sin``/``cos`` parts.

    A wrapped phase must never be low-pass filtered directly — its ±π
    discontinuities are not band-limited — so the *continuous* components
    ``sin φ`` and ``cos φ`` are decimated (zero-phase FIR anti-alias) and the
    phase reconstructed with ``atan2``.

    Parameters
    ----------
    phases : FloatArray
        Per-channel phase in radians, shape ``(N, T)``.
    factor : int
        Integer decimation factor; ``1`` returns the input unchanged.

    Returns
    -------
    FloatArray
        The reconstructed phase at the decimated rate, shape ``(N, T // factor)``.

    Raises
    ------
    ValueError
        If the phase field is malformed or the factor is not a positive integer.
    """
    array = _validate_signals(phases, "phases")
    q = _positive_int(factor, "factor")
    if q == 1:
        return np.ascontiguousarray(array, dtype=np.float64)
    sin_d = decimate(np.sin(array), q, ftype="fir", zero_phase=True, axis=-1)
    cos_d = decimate(np.cos(array), q, ftype="fir", zero_phase=True, axis=-1)
    reconstructed = np.arctan2(sin_d, cos_d)
    return np.ascontiguousarray(reconstructed, dtype=np.float64)


def eeg_observables(
    raw: FloatArray,
    *,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    decimation: int = DECIMATION,
    band_hz: tuple[float, float] = BAND_HZ,
    filter_order: int = FILTER_ORDER,
) -> EEGObservables:
    """Turn a raw multichannel recording into the suite's decimated observables.

    Parameters
    ----------
    raw : FloatArray
        Raw per-channel EEG, shape ``(N, T)`` with at least two channels (the
        order parameter and synchrony are undefined for a single channel).
    sampling_rate_hz : float
        Native sampling rate of ``raw`` in hertz.
    decimation : int
        Decimation factor applied to the analytic-phase components.
    band_hz : tuple[float, float]
        Band-pass passband in hertz.
    filter_order : int
        Butterworth order.

    Returns
    -------
    EEGObservables
        The reconstructed phases, the ``sin(phase)`` field, and the cross-channel
        order parameter, all at ``sampling_rate_hz / decimation``.

    Raises
    ------
    ValueError
        If the recording has fewer than two channels or is otherwise malformed.
    """
    array = _validate_signals(raw, "raw")
    if array.shape[0] < 2:
        raise ValueError("raw must have at least two channels for synchrony")
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    q = _positive_int(decimation, "decimation")
    filtered = bandpass(array, sampling_rate_hz=fs, band_hz=band_hz, order=filter_order)
    phases = decimate_analytic_phase(analytic_phase(filtered), factor=q)
    phase_field = np.sin(phases)
    order_parameter = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return EEGObservables(
        phases=np.ascontiguousarray(phases, dtype=np.float64),
        phase_field=np.ascontiguousarray(phase_field, dtype=np.float64),
        order_parameter=np.ascontiguousarray(order_parameter, dtype=np.float64),
        sampling_rate_hz=fs / q,
    )


def slice_observables(
    observables: EEGObservables, *, start: int, stop: int
) -> EEGObservables:
    """Return the observables restricted to the sample half-open range.

    Slicing an already-decimated observable keeps the analysis on a chosen
    interval — the fixed pre-onset segment of a seizure, or a null trial cut from
    an interictal recording — without re-running the pipeline.

    Parameters
    ----------
    observables : EEGObservables
        The field to slice.
    start, stop : int
        Half-open sample range ``[start, stop)``; ``0 <= start < stop <= n``.

    Returns
    -------
    EEGObservables
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
    return EEGObservables(
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
    interictal_observables: Sequence[EEGObservables],
    *,
    segment_samples: int = SEGMENT_SAMPLES,
) -> list[EEGObservables]:
    """Cut each interictal recording into non-overlapping null trials.

    Every trial has the same length as a seizure's pre-onset analysis segment, so
    the false-alarm rate is estimated over many comparable trials rather than a
    handful of whole recordings — a finer, fairer matched-false-alarm calibration.

    Parameters
    ----------
    interictal_observables : sequence of EEGObservables
        Seizure-free recordings to segment.
    segment_samples : int
        Trial length in decimated samples; must be a positive integer.

    Returns
    -------
    list[EEGObservables]
        The non-overlapping trials, in recording then time order.

    Raises
    ------
    ValueError
        If ``segment_samples`` is not a positive integer.
    """
    length = _positive_int(segment_samples, "segment_samples")
    trials: list[EEGObservables] = []
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
    observables: EEGObservables,
    *,
    window: int = WINDOW,
    step: int = STEP,
    baseline_fraction: float = BASELINE_FRACTION,
    persistence: int = PERSISTENCE,
) -> dict[str, DetectorTrajectory]:
    """Run the suite once at a zero gate and return each detector's trajectory.

    The three members are run with a zero z-threshold so their oriented z-score
    trajectories are threshold-free; the fused score is their weighted mean.
    Calibration and lead measurement then apply a threshold to these
    trajectories, so the slow ordinal-entropy sweep runs once per recording
    rather than once per grid point.

    Parameters
    ----------
    observables : EEGObservables
        The decimated observable field.
    window, step : int
        Analysis window length and hop in samples.
    baseline_fraction : float
        Leading fraction of windows used to fit each baseline.
    persistence : int
        Echoed for symmetry; the alarm run length is applied at scoring time.

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
            relative_gate=RELATIVE_GATE,
            window_starts=np.asarray(csd.window_starts, dtype=np.int64),
            n_baseline=csd.n_baseline_windows,
        ),
        SYNCHRONISATION: DetectorTrajectory(
            name=SYNCHRONISATION,
            score=np.asarray(sync.robust_z, dtype=np.float64),
            relative=np.asarray(sync.relative_rise, dtype=np.float64),
            relative_gate=RELATIVE_GATE,
            window_starts=np.asarray(sync.window_starts, dtype=np.int64),
            n_baseline=sync.n_baseline_windows,
        ),
        TRANSITION_ENTROPY: DetectorTrajectory(
            name=TRANSITION_ENTROPY,
            score=-np.asarray(entropy.robust_z, dtype=np.float64),
            relative=np.asarray(entropy.relative_drop, dtype=np.float64),
            relative_gate=RELATIVE_GATE,
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
    persistence: int = PERSISTENCE,
) -> float:
    """Return the fraction of interictal null trajectories that alarm."""
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
    target_fa: float = TARGET_FALSE_ALARM,
    persistence: int = PERSISTENCE,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> float:
    """Return the smallest grid threshold whose null false-alarm rate ≤ target.

    Falls back to the largest grid threshold when even that exceeds the target,
    so the null ensemble can never leave the detector uncalibrated.
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
    null_observables: Sequence[EEGObservables],
    *,
    target_fa: float = TARGET_FALSE_ALARM,
    persistence: int = PERSISTENCE,
    window: int = WINDOW,
    step: int = STEP,
    baseline_fraction: float = BASELINE_FRACTION,
) -> dict[str, float]:
    """Calibrate every detector to a matched false alarm on the interictal null.

    Parameters
    ----------
    null_observables : sequence of EEGObservables
        Interictal (seizure-free) recordings forming the false-alarm null.
    target_fa : float
        Target false-alarm rate each detector is held at or below.
    persistence, window, step, baseline_fraction :
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
    persistence: int = PERSISTENCE,
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
# Per-seizure sealing (re-runs each detector at its calibrated threshold)       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeizureLeadResult:
    """The sealed evidence and matched-false-alarm leads for one seizure.

    Attributes
    ----------
    record_id : str
        The CHB-MIT record label, e.g. ``chb01_03``.
    onset_sample : int
        Annotated seizure onset in decimated samples.
    evidences : dict[str, EarlyWarningEvidence]
        A sealed record per label in :data:`DETECTORS` — including a sealed
        silence when a detector did not fire.
    """

    record_id: str
    onset_sample: int
    evidences: dict[str, EarlyWarningEvidence]

    def lead_seconds(self) -> dict[str, float | None]:
        """Return each detector's honest lead in seconds (``None`` if it was late)."""
        return {
            name: (evidence.lead_seconds if evidence.lead_is_early else None)
            for name, evidence in self.evidences.items()
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the sealed per-detector evidence."""
        return {
            "record_id": self.record_id,
            "onset_sample": self.onset_sample,
            "detectors": {
                name: evidence.to_audit_record()
                for name, evidence in self.evidences.items()
            },
        }


def evaluate_seizure(
    observables: EEGObservables,
    *,
    record_id: str,
    onset_sample: int,
    signal_source: str,
    captured_at: str,
    thresholds: Mapping[str, float],
    window: int = WINDOW,
    step: int = STEP,
    baseline_fraction: float = BASELINE_FRACTION,
    persistence: int = PERSISTENCE,
) -> SeizureLeadResult:
    """Run the suite at the calibrated thresholds and seal each detector's alarm.

    Each detector is re-run with its matched-false-alarm threshold as the gate,
    so the sealed :class:`EarlyWarningEvidence` records the alarm decision at the
    calibrated operating point and an honest lead against ``onset_sample``.

    Parameters
    ----------
    observables : EEGObservables
        The seizure recording's decimated observables.
    record_id : str
        CHB-MIT record label, carried into the result.
    onset_sample : int
        Annotated seizure onset in decimated samples.
    signal_source, captured_at : str
        Provenance forwarded into every sealed record.
    thresholds : Mapping[str, float]
        The matched-false-alarm threshold per label in :data:`DETECTORS`.
    window, step, baseline_fraction, persistence :
        Suite analysis parameters.

    Returns
    -------
    SeizureLeadResult
        The four sealed records and the recording provenance.

    Raises
    ------
    KeyError
        If ``thresholds`` is missing a detector label.
    """
    csd = critical_slowing_down_warning(
        observables.order_parameter[np.newaxis, :],
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[CRITICAL_SLOWING_DOWN],
        rise_threshold=RELATIVE_GATE,
        persistence=persistence,
    )
    sync = synchronisation_warning(
        observables.phases,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[SYNCHRONISATION],
        rise_threshold=RELATIVE_GATE,
        persistence=persistence,
    )
    entropy = explosive_sync_warning(
        observables.phase_field,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
        z_threshold=thresholds[TRANSITION_ENTROPY],
        drop_threshold=RELATIVE_GATE,
        persistence=persistence,
    )
    fused = ensemble_warning(
        [
            member_from_critical_slowing_down(csd),
            member_from_synchronisation(sync),
            member_from_transition_entropy(entropy),
        ],
        rule=WEIGHTED_RULE,
        fused_threshold=thresholds[ENSEMBLE_WEIGHTED],
        persistence=persistence,
    )
    fs = observables.sampling_rate_hz
    evidences = {
        CRITICAL_SLOWING_DOWN: seal_critical_slowing_down_alarm(
            csd,
            observable=_OBSERVABLE_CSD,
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        SYNCHRONISATION: seal_synchronisation_alarm(
            sync,
            observable=_OBSERVABLE_SYNC,
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        TRANSITION_ENTROPY: seal_transition_entropy_alarm(
            entropy,
            observable=_OBSERVABLE_ENTROPY,
            signal_source=signal_source,
            captured_at=captured_at,
            sampling_rate_hz=fs,
            transition_onset_sample=onset_sample,
        ),
        ENSEMBLE_WEIGHTED: seal_ensemble_alarm(
            fused,
            observable=_OBSERVABLE_ENSEMBLE,
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
# EDF ingestion (thin — touches the citation-only corpus, tested on synthetic  #
# EDF fixtures, never on the redistributed raw recordings)                      #
# --------------------------------------------------------------------------- #


def load_edf_channels(
    path: str | Path, *, expected_rate_hz: float = SAMPLING_RATE_HZ
) -> FloatArray:
    """Read the same-rate signal channels of an EDF recording into an array.

    Only channels sampled at ``expected_rate_hz`` are kept (CHB-MIT files
    occasionally carry an off-rate annotation or dummy channel), so the returned
    array is a clean ``(N, T)`` block ready for :func:`eeg_observables`.

    Parameters
    ----------
    path : str or Path
        Path to the EDF recording.
    expected_rate_hz : float
        Sampling rate a channel must have to be included.

    Returns
    -------
    FloatArray
        The selected channels, shape ``(N, T)``.

    Raises
    ------
    ValueError
        If no channel matches ``expected_rate_hz``.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        selected = [
            index
            for index in range(reader.signals_in_file)
            if abs(float(reader.getSampleFrequency(index)) - expected_rate_hz) < 1.0e-6
        ]
        if not selected:
            raise ValueError(
                f"no channel in {path} is sampled at {expected_rate_hz} Hz"
            )
        channels = [
            np.asarray(reader.readSignal(index), dtype=np.float64) for index in selected
        ]
    finally:
        reader.close()
    return np.ascontiguousarray(np.vstack(channels), dtype=np.float64)


def edf_start_datetime(path: str | Path) -> str:
    """Return the EDF recording's start datetime as an ISO-8601 string.

    CHB-MIT anonymises the physical capture instant, so this is provenance from
    the file rather than a wall-clock reading, and is carried into the seals
    verbatim for reproducibility.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        start = reader.getStartdatetime()
    finally:
        reader.close()
    return start.isoformat()


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                              #
# --------------------------------------------------------------------------- #


def _verdict(leads_by_detector: Mapping[str, list[float]]) -> str:
    """Return the honest matched-false-alarm verdict across the seizures.

    Names a fusion advantage only if the weighted fusion's median leading lead
    strictly exceeds every single member's at the matched false-alarm rate; says
    so plainly otherwise, since the auditable moat holds either way.
    """
    medians = {
        name: (float(np.median(leads)) if leads else None)
        for name, leads in leads_by_detector.items()
    }
    fusion = medians.get(ENSEMBLE_WEIGHTED)
    members = {
        name: value
        for name, value in medians.items()
        if name != ENSEMBLE_WEIGHTED and value is not None
    }
    if fusion is None:
        return (
            "NO FUSION LEAD: the weighted fusion produced no leading detection at "
            "the matched false-alarm rate; report the members' leads on their own."
        )
    if members and all(fusion > value for value in members.values()):
        best = max(members, key=members.__getitem__)
        return (
            "FUSION LEADS: at a matched false-alarm rate the weighted fusion's "
            f"median leading lead ({fusion:.1f} s) exceeds every single member's "
            f"(best single: {best} at {members[best]:.1f} s). The advantage is a "
            "lead-time improvement at fixed false alarm, not a spent false-alarm "
            "budget."
        )
    return (
        "NO FUSION ADVANTAGE: at a matched false-alarm rate the weighted fusion "
        f"(median leading lead {fusion:.1f} s) does not beat the best single "
        "member; report the honest per-detector leads. The auditable, sealed "
        "evidence is the deliverable regardless of which detector leads."
    )


def main(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    interictal_records: Sequence[str] = INTERICTAL_RECORDS,
    seizures: Mapping[str, int] = SEIZURE_ONSETS_S,
    segment_samples: int = SEGMENT_SAMPLES,
    baseline_fraction: float = SEGMENT_BASELINE_FRACTION,
) -> None:
    """Run the capstone over CHB-MIT chb01 and write the sealed derived artefacts.

    Reads the interictal null and the annotated seizures from ``data_dir`` (raw
    EDF, citation-only), calibrates every detector to a matched false-alarm rate,
    seals each detector's alarm per seizure, and writes the sealed records plus an
    aggregate results JSON to ``output_dir``. Only the derived, sealed artefacts
    are written; the raw EDF is never copied.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding the ``<record>.edf`` recordings.
    output_dir : str or Path
        Directory the sealed derived artefacts are written to.
    interictal_records : sequence of str
        Seizure-free record stems forming the false-alarm null.
    seizures : Mapping[str, int]
        Record stem to annotated onset (seconds) for each seizure to evaluate.
    segment_samples : int
        Pre-onset analysis-segment length in decimated samples; also the null
        trial length. Defaults to :data:`SEGMENT_SAMPLES`.
    baseline_fraction : float
        Leading baseline fraction of each segment. Defaults to
        :data:`SEGMENT_BASELINE_FRACTION`.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment_seconds = segment_samples / DECIMATED_RATE_HZ

    # Finer matched-false-alarm null: cut each interictal hour into many trials of
    # the same length and structure as a seizure's pre-onset analysis segment.
    null_observables = [
        eeg_observables(load_edf_channels(data / f"{record}.edf"))
        for record in interictal_records
    ]
    trials = null_trials(null_observables, segment_samples=segment_samples)
    thresholds = calibrate_detectors(trials, baseline_fraction=baseline_fraction)

    leads_by_detector: dict[str, list[float]] = {name: [] for name in DETECTORS}
    seizure_records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for record_id, onset_s in seizures.items():
        path = data / f"{record_id}.edf"
        observables = eeg_observables(load_edf_channels(path))
        onset_sample = int(round(onset_s * observables.sampling_rate_hz))
        if onset_sample < segment_samples:
            # Too early for a clean pre-onset baseline; excluded, not counted null.
            excluded.append({"record_id": record_id, "onset_s": onset_s})
            continue
        # Fixed pre-onset segment with a guaranteed-clean leading baseline; the
        # onset sits at the segment end, so any alarm is a genuine lead.
        segment = slice_observables(
            observables, start=onset_sample - segment_samples, stop=onset_sample
        )
        result = evaluate_seizure(
            segment,
            record_id=record_id,
            onset_sample=segment_samples,
            signal_source=(
                f"CHB-MIT scalp EEG {record_id} (Shoeb 2009) / seizure onset "
                f"{onset_s} s / {segment_seconds:g} s pre-onset segment"
            ),
            captured_at=edf_start_datetime(path),
            thresholds=thresholds,
            baseline_fraction=baseline_fraction,
        )
        (out / f"{record_id}_early_warning_evidence.json").write_text(
            json.dumps(result.to_audit_record(), indent=2) + "\n", encoding="utf-8"
        )
        for name, lead in result.lead_seconds().items():
            if lead is not None:
                leads_by_detector[name].append(lead)
        seizure_records.append(
            {
                "record_id": record_id,
                "onset_s": onset_s,
                "lead_seconds": result.lead_seconds(),
            }
        )

    payload = {
        "benchmark": "early_warning_leadtime_eeg",
        "corpus": "CHB-MIT Scalp EEG Database, subject chb01 (Shoeb 2009)",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "decimated_rate_hz": DECIMATED_RATE_HZ,
        "band_hz": list(BAND_HZ),
        "window": WINDOW,
        "step": STEP,
        "baseline_seconds": baseline_fraction * segment_seconds,
        "horizon_seconds": (1.0 - baseline_fraction) * segment_seconds,
        "segment_seconds": segment_seconds,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "interictal_null_records": list(interictal_records),
        "n_null_trials": len(trials),
        "matched_false_alarm_thresholds": thresholds,
        "seizures": seizure_records,
        "excluded_seizures": excluded,
        "verdict": _verdict(leads_by_detector),
    }
    (out / "early_warning_leadtime_eeg_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(f"{len(trials)} null trials; {len(excluded)} seizures excluded (early onset)")
    print(f"results written to {out}")


# --------------------------------------------------------------------------- #
# Validators                                                                   #
# --------------------------------------------------------------------------- #


def _validate_signals(signals: object, name: str) -> FloatArray:
    """Return ``signals`` as a validated 2-D finite float array, else raise."""
    raw = np.asarray(signals)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} shape {raw.shape} must be one- or two-dimensional")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_band(band_hz: object, sampling_rate_hz: float) -> tuple[float, float]:
    """Return ``band_hz`` as a valid ``(low, high)`` passband, else raise."""
    if (
        isinstance(band_hz, (str, bytes))
        or not isinstance(band_hz, (tuple, list))
        or len(band_hz) != 2
    ):
        raise ValueError("band_hz must be a (low, high) pair in hertz")
    low = _positive_real(band_hz[0], "band_hz low edge")
    high = _positive_real(band_hz[1], "band_hz high edge")
    if low >= high:
        raise ValueError(f"band_hz low {low} must be below high {high}")
    if high >= 0.5 * sampling_rate_hz:
        raise ValueError(
            f"band_hz high {high} must be below the Nyquist {0.5 * sampling_rate_hz}"
        )
    return low, high


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    from numbers import Integral

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    from numbers import Real

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {result}")
    return result


def _non_negative_int(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    from numbers import Integral

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {result}")
    return result


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory holding the CHB-MIT chb01 EDFs")
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir)
